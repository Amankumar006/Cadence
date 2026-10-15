"""CP-SAT schedule solver and an independent verifier.

The schedule used to be produced by asking an LLM for a list of blocks with
invented start times, or by a rule-based builder that emitted hardcoded times
(15:00 for CIE prep, 14:00 for tasks, 18:00 for DSA, 21:30 for movie night)
with no overlap check at all. Two CIEs inside three days therefore produced two
180-minute blocks both starting at 15:00.

Constraint satisfaction is a job for a solver, not a language model — LLMs are
measurably unreliable at it (see DCP-Bench, arXiv:2506.06052). So the split is:

    extract  -> what needs placing, and with what constraints   (LLM, easy)
    solve    -> a non-overlapping, in-window placement          (CP-SAT)
    verify   -> an independent check of the solution            (pure Python)

The verifier is deliberately separate from the solver: it re-derives every hard
constraint from the problem rather than trusting the solver's own bookkeeping,
so a solver bug cannot mark its own homework.
"""

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("app.agents.scheduler")

try:  # pragma: no cover - exercised implicitly by every test that solves
    from ortools.sat.python import cp_model

    _HAS_ORTOOLS = True
except Exception:  # pragma: no cover
    cp_model = None
    _HAS_ORTOOLS = False

# Block types, kept in sync with schemas.BLOCK_TYPES.
BLOCK_TYPES = ("work", "study", "personal", "health", "rest", "ent")


def _to_minutes(hhmm: str) -> int:
    """Parse ``HH:MM`` into minutes since midnight. Returns 0 on garbage."""
    try:
        parts = str(hhmm).strip().split(":")
        h, m = int(parts[0]), int(parts[1])
        if not (0 <= h < 24 and 0 <= m < 60):
            return 0
        return h * 60 + m
    except Exception:
        return 0


def _to_hhmm(minutes: int) -> str:
    minutes = max(0, min(23 * 60 + 59, int(minutes)))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


# ── Problem description ─────────────────────────────────────────────────────


@dataclass
class FixedBlock:
    """A block that must appear at exactly its stated time.

    Classes and non-negotiable personal routines are fixed: moving them would
    make the schedule wrong, not merely suboptimal.
    """

    time: str
    title: str
    type: str = "personal"
    duration_min: int = 60
    notes: str = ""

    @property
    def start(self) -> int:
        return _to_minutes(self.time)

    @property
    def end(self) -> int:
        return self.start + int(self.duration_min)


@dataclass
class FlexibleItem:
    """Something that needs a slot, with the constraints it must respect."""

    title: str
    duration_min: int
    type: str = "study"
    notes: str = ""
    weight: float = 1.0
    # Optional placement window, in minutes since midnight.
    earliest: Optional[int] = None
    latest: Optional[int] = None
    # Prefer placement inside this window (soft constraint).
    preferred_start: Optional[int] = None
    preferred_end: Optional[int] = None
    # Soft "as early as possible" pressure; higher = more urgent.
    urgency: float = 0.0
    # Hard: must not be scheduled at all when False.
    required: bool = False


@dataclass
class SchedulingProblem:
    """Everything the solver needs. Deliberately plain data."""

    date: str
    day_start: str = "07:00"
    day_end: str = "23:00"
    fixed: List[FixedBlock] = field(default_factory=list)
    flexible: List[FlexibleItem] = field(default_factory=list)
    buffer_min: int = 0
    is_weekend: bool = False
    is_holiday: bool = False

    @property
    def day_start_min(self) -> int:
        return _to_minutes(self.day_start)

    @property
    def day_end_min(self) -> int:
        return _to_minutes(self.day_end)


# ── Solving ─────────────────────────────────────────────────────────────────


@dataclass
class SolveResult:
    blocks: List[Dict[str, Any]]
    status: str
    dropped: List[str] = field(default_factory=list)
    solver_time_s: float = 0.0

    @property
    def ok(self) -> bool:
        # StatusName() returns uppercase ("OPTIMAL", "FEASIBLE"), so compare
        # case-insensitively rather than trusting a particular casing.
        return self.status.lower() in ("optimal", "feasible")


def solve(problem: SchedulingProblem, time_limit_s: float = 5.0) -> SolveResult:
    """Place every flexible item into the day without overlaps.

    Fixed blocks are pinned. Flexible items are optional: if the day is
    over-committed, the lowest-weight items are dropped rather than the problem
    becoming infeasible — a schedule missing one nice-to-have beats no schedule.
    """
    if not _HAS_ORTOOLS:
        return SolveResult(
            blocks=_fallback_blocks(problem),
            status="no_solver",
            dropped=[i.title for i in problem.flexible],
        )

    model = cp_model.CpModel()
    day_start = problem.day_start_min
    day_end = problem.day_end_min

    intervals: List[Any] = []
    item_vars: List[Tuple[FlexibleItem, Any, Any]] = []
    buffer_min = max(0, int(problem.buffer_min))

    # Fixed blocks pin their own interval and are always present. They are
    # added at their *real* size: two back-to-back classes in a real timetable
    # have no gap between them, and inflating them here would make an ordinary
    # week infeasible.
    #
    # A fixed block is deliberately allowed to run past day_end. Sleep starts
    # at the sleep target and lasts eight hours, so it necessarily crosses
    # midnight; clamping it to the day window collapsed it to zero length and
    # let flexible blocks be scheduled on top of it. AddNoOverlap imposes no
    # window requirement, so there is nothing to gain by clamping.
    for idx, block in enumerate(problem.fixed):
        start = block.start
        end = block.end
        if end <= start:
            logger.warning(
                "Fixed block %r has non-positive duration (%s, %d min); skipping",
                block.title,
                block.time,
                block.duration_min,
            )
            continue
        suffix = model.NewIntVar(start, start, f"fixed_s_{idx}")
        efix = model.NewIntVar(end, end, f"fixed_e_{idx}")
        intervals.append(model.NewIntervalVar(suffix, end - start, efix, f"fixed_{idx}"))

    # Flexible items get a start within their window and an optional presence.
    for idx, item in enumerate(problem.flexible):
        duration = max(1, int(item.duration_min))
        earliest = max(day_start, item.earliest if item.earliest is not None else day_start)
        latest_limit = min(day_end, item.latest if item.latest is not None else day_end)
        latest_start = latest_limit - duration
        if latest_start < earliest:
            # Cannot fit at all; leave it out of the model entirely.
            logger.info(
                "Item %r (%d min) cannot fit in [%s, %s]; dropping",
                item.title,
                duration,
                _to_hhmm(earliest),
                _to_hhmm(latest_limit),
            )
            continue

        start = model.NewIntVar(earliest, latest_start, f"start_{idx}")
        end = model.NewIntVar(earliest + duration, latest_limit, f"end_{idx}")
        present = model.NewBoolVar(f"present_{idx}")
        model.Add(start + duration == end)

        # The interval fed to AddNoOverlap is inflated by the buffer on BOTH
        # sides. AddNoOverlap alone only prevents true overlap, so without this
        # the buffer would be silently ignored. The inflation must be symmetric:
        # inflating only the right side leaves no gap in front of a block that
        # follows a fixed one, which the verifier then correctly rejects.
        #
        # Because the interval is optional, a dropped item is excluded from the
        # constraint entirely — which is what keeps an over-committed day
        # solvable rather than infeasible.
        iv = model.NewOptionalIntervalVar(
            start - buffer_min,
            duration + 2 * buffer_min,
            end + buffer_min,
            present,
            f"iv_{idx}",
        )
        intervals.append(iv)
        item_vars.append((item, start, present))

    if intervals:
        model.AddNoOverlap(intervals)

    # Objective. The scaling matters: weight must dominate, and urgency and the
    # peak-focus preference must only ever break ties. An earlier version used
    # weight*1000 with a -10/minute urgency penalty, which let the penalty
    # exceed the reward for placing a high-weight item at all — the solver then
    # correctly dropped the most urgent item, which is the opposite of intended.
    # So: one unit of weight is worth 100000, and the largest possible urgency
    # penalty (URGENCY_COEFF * 1440) stays below that.
    WEIGHT_SCALE = 100_000
    URGENCY_COEFF = 50
    PREFERENCE_BONUS = 1_000

    objective = []
    for item, start, present in item_vars:
        objective.append(int(round(item.weight * WEIGHT_SCALE)) * present)

        # Urgency: reward earlier starts. OR-Tools cannot multiply two
        # variables, so the "start if placed, else 0" product is expressed as a
        # bounded variable with an enforced equality.
        if item.urgency:
            lateness = model.NewIntVar(0, day_end, f"late_{len(objective)}")
            model.Add(lateness == start).OnlyEnforceIf(present)
            model.Add(lateness == 0).OnlyEnforceIf(present.Not())
            objective.append(-int(round(item.urgency * URGENCY_COEFF)) * lateness)

        if item.preferred_start is not None and item.preferred_end is not None:
            # Soft preference for the peak-focus window: reward landing inside.
            inside = model.NewBoolVar(f"pref_{len(objective)}")
            model.Add(start >= item.preferred_start).OnlyEnforceIf(inside)
            model.Add(
                start + int(item.duration_min) <= item.preferred_end
            ).OnlyEnforceIf(inside)
            objective.append(PREFERENCE_BONUS * inside)

    if objective:
        model.Maximize(sum(objective))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    # Determinism matters: the same problem must give the same schedule, or the
    # user sees their day reshuffle between identical requests.
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 20261015

    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    placed: List[Dict[str, Any]] = []
    dropped: List[str] = []

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for block in problem.fixed:
            placed.append(_block_dict(block.start, block))
        for item, start, present in item_vars:
            if solver.Value(present):
                placed.append(
                    _block_dict(solver.Value(start), item)
                )
            else:
                dropped.append(item.title)
        # Items dropped before modelling never reached item_vars.
        modelled = {id(i) for i, _, _ in item_vars}
        dropped.extend(i.title for i in problem.flexible if id(i) not in modelled)
        placed.sort(key=lambda b: _to_minutes(b["time"]))
    else:
        logger.warning("CP-SAT returned %s; falling back", status_name)
        return SolveResult(
            blocks=_fallback_blocks(problem),
            status=status_name,
            dropped=[i.title for i in problem.flexible],
        )

    return SolveResult(
        blocks=placed,
        status=status_name,
        dropped=dropped,
        solver_time_s=solver.WallTime(),
    )


def _block_dict(start_min: int, source) -> Dict[str, Any]:
    return {
        "time": _to_hhmm(start_min),
        "title": source.title,
        "type": source.type if source.type in BLOCK_TYPES else "work",
        "duration_min": int(source.duration_min),
        "notes": getattr(source, "notes", "") or "",
    }


def _fallback_blocks(problem: SchedulingProblem) -> List[Dict[str, Any]]:
    """Best-effort ordering used only when the solver is unavailable."""
    blocks = [_block_dict(b.start, b) for b in problem.fixed]
    for item in problem.flexible:
        blocks.append(_block_dict(problem.day_start_min, item))
    blocks.sort(key=lambda b: _to_minutes(b["time"]))
    return blocks


# ── Verification ────────────────────────────────────────────────────────────


@dataclass
class Violation:
    code: str
    detail: str


def verify_candidate(
    blocks: Sequence[Dict[str, Any]], problem: SchedulingProblem
) -> List[Violation]:
    """Check a candidate schedule against every hard constraint.

    Independent of the solver on purpose: this re-derives the constraints from
    the problem rather than reading back solver state.
    """
    violations: List[Violation] = []
    if not blocks:
        return [Violation("empty", "the schedule has no blocks")]

    spans: List[Tuple[int, int, Dict[str, Any]]] = []
    for block in blocks:
        start = _to_minutes(block.get("time", ""))
        duration = block.get("duration_min", 0)
        try:
            duration = int(duration)
        except (TypeError, ValueError):
            violations.append(
                Violation("bad_duration", f"{block.get('title')!r} has duration {duration!r}")
            )
            continue
        if duration <= 0:
            violations.append(
                Violation("bad_duration", f"{block.get('title')!r} has duration {duration}")
            )
            continue
        if block.get("type") not in BLOCK_TYPES:
            violations.append(
                Violation("bad_type", f"{block.get('title')!r} has type {block.get('type')!r}")
            )
        spans.append((start, start + duration, block))

    day_start, day_end = problem.day_start_min, problem.day_end_min

    # 1. Everything inside the day window.
    for start, end, block in spans:
        title = str(block.get("title", ""))
        # Sleep is the block that closes the day, so it legitimately runs past
        # the wake time into the next morning. Checking it against day_end
        # would make every correct schedule fail.
        closes_day = title.lower() == "sleep"
        if start < day_start:
            violations.append(
                Violation(
                    "before_wake",
                    f"{block['title']!r} starts {_to_hhmm(start)}, before wake {_to_hhmm(day_start)}",
                )
            )
        if end > day_end and not closes_day:
            violations.append(
                Violation(
                    "past_sleep",
                    f"{block['title']!r} ends {_to_hhmm(end)}, past sleep {_to_hhmm(day_end)}",
                )
            )

    # 2. No overlaps, with the configured buffer.
    #
    # The buffer only applies where a flexible block is involved. Two fixed
    # blocks (back-to-back classes in a real timetable) are legitimately
    # adjacent, and demanding a gap between them would reject every correct
    # weekday. This mirrors the solver, which only inflates flexible intervals.
    fixed_titles = {f.title for f in problem.fixed}
    buffer_min = max(0, int(problem.buffer_min))
    ordered = sorted(spans, key=lambda s: s[0])
    for (s1, e1, b1), (s2, e2, b2) in zip(ordered, ordered[1:]):
        both_fixed = (
            str(b1.get("title", "")) in fixed_titles
            and str(b2.get("title", "")) in fixed_titles
        )
        required_gap = 0 if both_fixed else buffer_min
        if s2 < e1 + required_gap:
            violations.append(
                Violation(
                    "overlap",
                    f"{b1['title']!r} ({_to_hhmm(s1)}-{_to_hhmm(e1)}) overlaps "
                    f"{b2['title']!r} ({_to_hhmm(s2)}-{_to_hhmm(e2)})",
                )
            )

    # 3. Every fixed block present, at its exact time.
    by_title: Dict[str, Dict[str, Any]] = {}
    for block in blocks:
        by_title.setdefault(str(block.get("title", "")), block)
    for fixed in problem.fixed:
        got = by_title.get(fixed.title)
        if got is None:
            violations.append(
                Violation("missing_fixed", f"fixed block {fixed.title!r} is absent")
            )
            continue
        if _to_minutes(got.get("time", "")) != fixed.start:
            violations.append(
                Violation(
                    "moved_fixed",
                    f"fixed block {fixed.title!r} moved to {got.get('time')!r} "
                    f"(expected {fixed.time})",
                )
            )

    # 4. Sleep must be the last block of the day.
    sleep = [b for b in blocks if str(b.get("title", "")).lower() == "sleep"]
    if sleep:
        last = ordered[-1][2] if ordered else None
        if last is not None and str(last.get("title", "")).lower() != "sleep":
            violations.append(
                Violation("sleep_not_last", f"the last block is {last.get('title')!r}, not Sleep")
            )

    # 5. Duplicate titles at the same time are almost always a bug.
    seen: Dict[Tuple[str, str], int] = {}
    for block in blocks:
        key = (str(block.get("title", "")), str(block.get("time", "")))
        seen[key] = seen.get(key, 0) + 1
    for (title, time), count in seen.items():
        if count > 1:
            violations.append(
                Violation("duplicate", f"{title!r} appears {count}x at {time}")
            )

    return violations


def is_valid(blocks: Sequence[Dict[str, Any]], problem: SchedulingProblem) -> bool:
    return not verify_candidate(blocks, problem)


# ── Problem extraction ──────────────────────────────────────────────────────


def problem_from_context(
    date: str,
    weekday_name: str,
    is_weekend: bool,
    classes: Sequence[Dict[str, Any]],
    personal_fixed_blocks: Sequence[Dict[str, Any]],
    academic_events: Sequence[Dict[str, Any]],
    academic_calendar: Dict[str, Any],
    user_profile: Dict[str, Any],
    pending_tasks: Optional[Sequence[Dict[str, Any]]] = None,
    recent_notes: Optional[Sequence[Dict[str, Any]]] = None,
    movie_info: Optional[Dict[str, Any]] = None,
    buffer_min: int = 15,
) -> SchedulingProblem:
    """Turn hydrated context into a SchedulingProblem.

    This is the *extraction* half: deciding what needs placing and with what
    constraints. It replaces the old rule-based builder, which both decided and
    placed. Keeping the decision here and the placement in CP-SAT is what makes
    the result verifiable.
    """
    pending_tasks = pending_tasks or []
    recent_notes = recent_notes or []

    wake = user_profile.get("wake_time") or "07:00"
    sleep = user_profile.get("sleep_target") or "23:00"
    if weekday_name.lower() == "saturday":
        sleep = "23:30"

    # A holiday relaxes study expectations but keeps the routine.
    is_holiday = False
    for holiday in academic_calendar.get("holidays") or []:
        if holiday.get("date") == date:
            is_holiday = True
            break

    # Collect the fixed blocks first; the day window has to contain all of
    # them, so it is derived after rather than before.
    pending_fixed: List[FixedBlock] = []

    day_short = weekday_name[:3]
    for block in personal_fixed_blocks:
        # No `days` means every day. save_user_info defaults it to ["daily"], so
        # this never bites when the block comes through the parser; anything the
        # API or the client writes may omit it, and dropping a routine the user
        # explicitly saved -- silently -- is the wrong default.
        days = [str(d).lower() for d in block.get("days") or ["daily"]]
        applies = "daily" in days or any(d.startswith(day_short.lower()) for d in days)
        if not applies:
            continue
        non_negotiable = bool(block.get("non_negotiable"))
        pending_fixed.append(
            FixedBlock(
                time=block.get("time", "06:00"),
                title=block.get("title", "Routine Block"),
                type=block.get("type", "personal"),
                duration_min=int(block.get("duration_min", 30) or 30),
                notes=f"Personal {'fixed' if non_negotiable else 'routine'} block.",
            )
        )

    if not is_holiday and not is_weekend:
        for c in classes:
            pending_fixed.append(
                FixedBlock(
                    time=c.get("time", "09:00"),
                    title=c.get("subject", "Class"),
                    type="personal",
                    duration_min=int(c.get("duration_min", 60) or 60),
                    notes=f"Class lecture. Room: {c.get('room', 'N/A')}",
                )
            )

    # A pre-wake routine (meditation at 06:00 with a 07:00 wake time) is
    # legitimate, so the day starts at whichever comes first.
    day_start = wake
    for block in pending_fixed:
        day_start = min(day_start, block.time, key=_to_minutes)

    problem = SchedulingProblem(
        date=date,
        day_start=day_start,
        day_end=sleep,
        buffer_min=buffer_min,
        is_weekend=is_weekend,
        is_holiday=is_holiday,
    )
    problem.fixed.extend(pending_fixed)

    # Sleep is fixed at the target so nothing can be scheduled over it.
    problem.fixed.append(
        FixedBlock(
            time=sleep,
            title="Sleep",
            type="rest",
            duration_min=480,
            notes="Target sleep time",
        )
    )

    # ── Flexible: study load from upcoming academic events ──
    peak = str(user_profile.get("peak_focus_time") or "").lower()
    preferred = _peak_window(peak)

    try:
        dt_today = datetime.date.fromisoformat(date)
    except Exception:
        dt_today = datetime.date.today()

    if not is_holiday and weekday_name.lower() != "sunday":
        for event in academic_events:
            due_str = event.get("due_date")
            subject = event.get("subject") or "Study"
            if not due_str:
                continue
            try:
                days_remaining = (datetime.date.fromisoformat(due_str) - dt_today).days
            except ValueError:
                continue
            if days_remaining < 0:
                continue

            # Urgency scales with proximity; the nearer the exam, the more
            # pressure to place it early and the longer the block.
            if days_remaining <= 3:
                duration, urgency, critical = 180, 1.0, True
            elif days_remaining <= 7:
                duration, urgency, critical = 120, 0.6, False
            else:
                duration, urgency, critical = 90, 0.3, False

            problem.flexible.append(
                FlexibleItem(
                    title=f"{subject} CIE prep{' (CRITICAL)' if critical else ''}",
                    duration_min=duration,
                    type="study",
                    notes=f"{'CIE prep (CRITICAL)' if critical else 'CIE prep'}. "
                    f"Exam on {due_str}.",
                    weight=3.0 if critical else 2.0,
                    urgency=urgency,
                    preferred_start=preferred[0],
                    preferred_end=preferred[1],
                )
            )

    # ── Flexible: tasks due exactly today ──
    for task in pending_tasks:
        if task.get("due_date") != date or task.get("completed"):
            continue
        priority = str(task.get("priority", "medium")).lower()
        problem.flexible.append(
            FlexibleItem(
                title=f"Task: {task.get('title')}",
                duration_min=60,
                type="work",
                notes=f"Complete pending task: {task.get('title')}. "
                f"Priority: {task.get('priority')}.",
                weight={"high": 3.0, "medium": 2.0, "low": 1.0}.get(priority, 2.0),
                urgency=0.4,
            )
        )

    # ── Flexible: study blocks implied by notes ──
    # Both a holiday and a Sunday relax study expectations, notes included.
    study_relaxed = is_holiday or weekday_name.lower() == "sunday"
    for note in ([] if study_relaxed else recent_notes):
        body = str(note.get("body", "")).lower()
        tags = [str(t).lower() for t in note.get("tags", [])]
        if "dsa" in body or "dsa" in tags:
            problem.flexible.append(
                FlexibleItem(
                    title="DSA Study Block",
                    duration_min=90,
                    type="study",
                    notes=f"Study block for DSA. Note context: {note.get('ai_summary')}",
                    weight=2.0,
                    urgency=0.3,
                    preferred_start=preferred[0],
                    preferred_end=preferred[1],
                )
            )

    # ── Flexible: Saturday movie mandate ──
    if weekday_name.lower() == "saturday":
        title = (movie_info or {}).get("title") or "Terminator 2: Judgment Day"
        problem.flexible.append(
            FlexibleItem(
                title=f"Movie Night: {title}",
                duration_min=120,
                type="ent",
                notes=f"⭐ {(movie_info or {}).get('rating', '8.6')}/10 - "
                f"{(movie_info or {}).get('overview', '')}",
                weight=1.0,
                earliest=_to_minutes("20:00"),
                latest=_to_minutes("23:30"),
            )
        )

    # ── Flexible: gym on the user's gym days ──
    gym_days = [str(d).lower() for d in user_profile.get("gym_days", [])]
    gym_time = user_profile.get("gym_time")
    if gym_days and (any(d.startswith(day_short.lower()) for d in gym_days) or "daily" in gym_days):
        problem.flexible.append(
            FlexibleItem(
                title="Gym",
                duration_min=60,
                type="health",
                notes="Scheduled workout.",
                weight=1.5,
                earliest=_to_minutes(gym_time) - 60 if gym_time else None,
                latest=_to_minutes(gym_time) + 120 if gym_time else None,
            )
        )

    return problem


def _peak_window(peak: str) -> Tuple[Optional[int], Optional[int]]:
    """Map a peak-focus label to a preferred placement window."""
    table = {
        "morning": ("08:00", "12:00"),
        "afternoon": ("13:00", "17:00"),
        "evening": ("17:00", "21:00"),
        "night": ("20:00", "23:00"),
    }
    window = table.get(peak)
    if not window:
        return None, None
    return _to_minutes(window[0]), _to_minutes(window[1])
