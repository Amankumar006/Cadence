"""Tests for the CP-SAT scheduler and its verifier.

The verifier is the important half: it is the only thing standing between a
solver bug and a user being told their day has a 15:00 block that starts at
15:00 twice. Every test here checks a *constraint*, not an implementation.
"""


from app.agents.scheduler import (
    FixedBlock,
    FlexibleItem,
    SchedulingProblem,
    SolveResult,
    _peak_window,
    _to_hhmm,
    _to_minutes,
    is_valid,
    problem_from_context,
    solve,
    verify_candidate,
)


def _violation_codes(blocks, problem):
    return {v.code for v in verify_candidate(blocks, problem)}


def _block(time, title, duration=60, type="study", notes=""):
    return {
        "time": time,
        "title": title,
        "type": type,
        "duration_min": duration,
        "notes": notes,
    }


# ── Time helpers ────────────────────────────────────────────────────────────


def test_minutes_round_trip():
    for hhmm in ("00:00", "07:15", "12:00", "23:59"):
        assert _to_hhmm(_to_minutes(hhmm)) == hhmm


def test_minutes_rejects_garbage_without_raising():
    assert _to_minutes("not a time") == 0
    assert _to_minutes("") == 0
    assert _to_minutes(None) == 0
    assert _to_minutes("99:99") == 0
    assert _to_minutes("12:75") == 0


def test_peak_window_covers_known_labels():
    for label in ("morning", "afternoon", "evening", "night"):
        start, end = _peak_window(label)
        assert start is not None and end is not None and start < end


def test_peak_window_is_none_for_unknown():
    assert _peak_window("whenever") == (None, None)
    assert _peak_window("") == (None, None)


# ── The core property: no overlaps ──────────────────────────────────────────


def test_two_critical_cies_do_not_both_land_at_1500():
    """The exact bug this module exists to fix.

    The old rule-based builder emitted a hardcoded 15:00 block per critical
    CIE, so two exams within three days produced two 180-minute blocks starting
    at the same minute.
    """
    problem = SchedulingProblem(
        date="2026-10-15",
        day_start="07:00",
        day_end="23:00",
        fixed=[FixedBlock("09:00", "DSA Class", duration_min=60)],
        flexible=[
            FlexibleItem("DSA CIE prep", duration_min=180, weight=3.0),
            FlexibleItem("DBMS CIE prep", duration_min=180, weight=3.0),
            FlexibleItem("CN CIE prep", duration_min=180, weight=3.0),
        ],
    )
    result = solve(problem)

    assert result.ok, result.status
    assert not _violation_codes(result.blocks, problem), _violation_codes(result.blocks, problem)
    assert len(result.blocks) == 4  # 3 study blocks + 1 class

    starts = [b["time"] for b in result.blocks if "CIE prep" in b["title"]]
    assert len(set(starts)) == 3, f"study blocks collided at {starts}"


def test_solver_respects_the_buffer_between_blocks():
    problem = SchedulingProblem(
        date="2026-10-15",
        day_start="07:00",
        day_end="23:00",
        buffer_min=15,
        flexible=[
            FlexibleItem("A", duration_min=60),
            FlexibleItem("B", duration_min=60),
            FlexibleItem("C", duration_min=60),
        ],
    )
    result = solve(problem)
    assert result.ok
    assert not _violation_codes(result.blocks, problem)

    ordered = sorted(result.blocks, key=lambda b: _to_minutes(b["time"]))
    for first, second in zip(ordered, ordered[1:]):
        gap = _to_minutes(second["time"]) - (
            _to_minutes(first["time"]) + first["duration_min"]
        )
        assert gap >= 15, f"{first['title']} -> {second['title']} gap is {gap}"


def test_fixed_blocks_never_move():
    problem = SchedulingProblem(
        date="2026-10-15",
        day_start="07:00",
        day_end="23:00",
        fixed=[
            FixedBlock("09:00", "DSA", duration_min=60),
            FixedBlock("11:00", "DBMS", duration_min=60),
        ],
        flexible=[FlexibleItem("Gym", duration_min=60)],
    )
    result = solve(problem)

    dsa = next(b for b in result.blocks if b["title"] == "DSA")
    dbms = next(b for b in result.blocks if b["title"] == "DBMS")
    assert dsa["time"] == "09:00"
    assert dbms["time"] == "11:00"
    assert not _violation_codes(result.blocks, problem)


def test_over_committed_day_drops_the_lowest_weight_items():
    """A schedule missing one nice-to-have beats no schedule at all."""
    problem = SchedulingProblem(
        date="2026-10-15",
        day_start="07:00",
        day_end="10:00",  # 3 hours
        flexible=[
            FlexibleItem("Critical", duration_min=120, weight=10.0),
            FlexibleItem("Nice", duration_min=120, weight=1.0),
        ],
    )
    result = solve(problem)

    assert result.ok
    assert "Critical" in [b["title"] for b in result.blocks]
    assert "Nice" in result.dropped
    assert not _violation_codes(result.blocks, problem)


def test_an_item_that_cannot_fit_is_dropped_not_crashed():
    problem = SchedulingProblem(
        date="2026-10-15",
        day_start="07:00",
        day_end="09:00",
        flexible=[
            FlexibleItem("Too big", duration_min=600),
            FlexibleItem("Fits", duration_min=60),
        ],
    )
    result = solve(problem)
    assert result.ok
    assert "Too big" in result.dropped
    assert "Fits" in [b["title"] for b in result.blocks]


def test_an_item_with_a_narrow_window_is_respected():
    problem = SchedulingProblem(
        date="2026-10-15",
        day_start="07:00",
        day_end="23:00",
        flexible=[
            FlexibleItem(
                "Evening only", duration_min=60, earliest=_to_minutes("20:00")
            )
        ],
    )
    result = solve(problem)
    block = result.blocks[0]
    assert _to_minutes(block["time"]) >= _to_minutes("20:00")


def test_solving_is_deterministic():
    """The same problem must produce the same schedule, or the user's day
    reshuffles between two identical requests."""
    problem = SchedulingProblem(
        date="2026-10-15",
        day_start="07:00",
        day_end="23:00",
        fixed=[FixedBlock("09:00", "Class", duration_min=60)],
        flexible=[
            FlexibleItem("A", duration_min=90, weight=2.0),
            FlexibleItem("B", duration_min=60, weight=1.0),
            FlexibleItem("C", duration_min=120, weight=3.0),
        ],
    )
    first = solve(problem).blocks
    for _ in range(3):
        assert solve(problem).blocks == first


def test_urgent_items_are_placed_earlier():
    problem = SchedulingProblem(
        date="2026-10-15",
        day_start="07:00",
        day_end="23:00",
        flexible=[
            FlexibleItem("Urgent", duration_min=60, urgency=1.0, weight=2.0),
            FlexibleItem("Relaxed", duration_min=60, urgency=0.0, weight=2.0),
        ],
    )
    result = solve(problem)
    urgent = next(b for b in result.blocks if b["title"] == "Urgent")
    relaxed = next(b for b in result.blocks if b["title"] == "Relaxed")
    assert _to_minutes(urgent["time"]) <= _to_minutes(relaxed["time"])


# ── The verifier ────────────────────────────────────────────────────────────


def test_verifier_accepts_a_good_schedule():
    problem = SchedulingProblem(
        date="2026-10-15",
        day_start="07:00",
        day_end="23:00",
        fixed=[FixedBlock("09:00", "Class", duration_min=60)],
    )
    blocks = [
        _block("07:30", "Gym", 60, "health"),
        _block("09:00", "Class", 60, "personal"),
        _block("11:00", "Study", 90),
        _block("23:00", "Sleep", 480, "rest"),
    ]
    assert is_valid(blocks, problem)
    assert verify_candidate(blocks, problem) == []


def test_verifier_catches_the_old_hardcoded_time_bug():
    """Two 180-minute blocks at 15:00 must be rejected."""
    problem = SchedulingProblem(date="2026-10-15", day_start="07:00", day_end="23:00")
    blocks = [
        _block("15:00", "DSA CIE prep", 180),
        _block("15:00", "DBMS CIE prep", 180),
        _block("23:00", "Sleep", 480, "rest"),
    ]
    codes = _violation_codes(blocks, problem)
    assert "overlap" in codes
    assert not is_valid(blocks, problem)


def test_verifier_catches_a_block_before_wake():
    problem = SchedulingProblem(date="2026-10-15", day_start="07:00", day_end="23:00")
    blocks = [_block("05:00", "Early bird", 60), _block("23:00", "Sleep", 480, "rest")]
    assert "before_wake" in _violation_codes(blocks, problem)


def test_verifier_catches_a_block_past_sleep():
    problem = SchedulingProblem(date="2026-10-15", day_start="07:00", day_end="23:00")
    blocks = [_block("22:00", "Late", 240), _block("23:00", "Sleep", 480, "rest")]
    assert "past_sleep" in _violation_codes(blocks, problem)


def test_verifier_catches_a_moved_fixed_block():
    problem = SchedulingProblem(
        date="2026-10-15",
        day_start="07:00",
        day_end="23:00",
        fixed=[FixedBlock("09:00", "DSA Class", duration_min=60)],
    )
    blocks = [
        _block("14:00", "DSA Class", 60, "personal"),  # moved!
        _block("23:00", "Sleep", 480, "rest"),
    ]
    codes = _violation_codes(blocks, problem)
    assert "moved_fixed" in codes
    assert "missing_fixed" not in codes, "the block is present, just displaced"


def test_verifier_catches_a_missing_fixed_block():
    problem = SchedulingProblem(
        date="2026-10-15",
        day_start="07:00",
        day_end="23:00",
        fixed=[FixedBlock("09:00", "Mandatory", duration_min=60)],
    )
    blocks = [_block("10:00", "Other", 60), _block("23:00", "Sleep", 480, "rest")]
    assert "missing_fixed" in _violation_codes(blocks, problem)


def test_verifier_catches_sleep_not_last():
    problem = SchedulingProblem(date="2026-10-15", day_start="07:00", day_end="23:00")
    blocks = [
        _block("07:00", "Gym", 60, "health"),
        _block("23:00", "Sleep", 480, "rest"),
        _block("23:30", "Afterparty", 30),
    ]
    codes = _violation_codes(blocks, problem)
    assert "sleep_not_last" in codes
    # It also runs past the sleep target, which is a separate violation.
    assert "past_sleep" in codes


def test_verifier_catches_an_empty_schedule():
    problem = SchedulingProblem(date="2026-10-15")
    codes = _violation_codes([], problem)
    assert "empty" in codes


def test_verifier_catches_a_zero_duration_block():
    problem = SchedulingProblem(date="2026-10-15")
    blocks = [_block("09:00", "Nothing", 0), _block("23:00", "Sleep", 480, "rest")]
    assert "bad_duration" in _violation_codes(blocks, problem)


def test_verifier_catches_an_unknown_block_type():
    problem = SchedulingProblem(date="2026-10-15")
    blocks = [_block("09:00", "Weird", 60, "banana"), _block("23:00", "Sleep", 480, "rest")]
    assert "bad_type" in _violation_codes(blocks, problem)


def test_verifier_catches_a_duplicate_title_at_the_same_time():
    problem = SchedulingProblem(date="2026-10-15")
    blocks = [
        _block("09:00", "Gym", 60),
        _block("09:00", "Gym", 60),
        _block("23:00", "Sleep", 480, "rest"),
    ]
    assert "duplicate" in _violation_codes(blocks, problem)


def test_verifier_does_not_crash_on_garbage_input():
    problem = SchedulingProblem(date="2026-10-15")
    blocks = [
        {"time": "nonsense", "title": "X", "duration_min": "abc"},
        {"title": "no time"},
        {},
    ]
    # Must not raise; it just reports what it found.
    verify_candidate(blocks, problem)


def test_solver_output_always_passes_the_verifier():
    """The round-trip property: whatever the solver emits must verify clean.

    Run across a spread of shapes, including deliberately over-committed ones.
    """
    shapes = [
        SchedulingProblem(
            date="2026-10-15",
            day_start="07:00",
            day_end="23:00",
            fixed=[FixedBlock("23:00", "Sleep", type="rest", duration_min=480)],
        ),
        SchedulingProblem(
            date="2026-10-15",
            day_start="06:00",
            day_end="23:30",
            fixed=[FixedBlock("09:00", "Class", 60), FixedBlock("14:00", "Lab", 120)],
            flexible=[FlexibleItem("Study", 90), FlexibleItem("Gym", 60)],
        ),
        SchedulingProblem(
            date="2026-10-17",
            day_start="08:00",
            day_end="23:30",
            buffer_min=30,
            flexible=[
                FlexibleItem("A", 180, weight=3.0),
                FlexibleItem("B", 180, weight=3.0),
                FlexibleItem("C", 120, weight=2.0),
                FlexibleItem("D", 60, weight=1.0),
            ],
        ),
        # Deliberately over-committed: most items must be dropped, but at least
        # one fits, so the result is a real (if thin) schedule.
        SchedulingProblem(
            date="2026-10-15",
            day_start="07:00",
            day_end="08:00",
            fixed=[FixedBlock("07:00", "Sleep", type="rest", duration_min=60)],
            flexible=[FlexibleItem(f"T{i}", 90) for i in range(6)],
        ),
    ]
    for problem in shapes:
        result = solve(problem)
        assert result.ok, f"status {result.status} for {problem}"
        codes = _violation_codes(result.blocks, problem)
        assert not codes, f"{problem} produced {codes}"


# ── Extraction ──────────────────────────────────────────────────────────────


PROFILE = {
    "wake_time": "07:00",
    "sleep_target": "23:00",
    "peak_focus_time": "morning",
    "gym_days": ["Mon", "Wed", "Fri"],
    "gym_time": "18:00",
}

CLASSES = [
    {"time": "09:00", "subject": "Data Structures", "room": "CS-201", "duration_min": 60},
    {"time": "11:00", "subject": "DBMS", "room": "CS-202", "duration_min": 60},
]


def _extract(**overrides):
    kwargs = dict(
        date="2026-10-15",  # a Thursday
        weekday_name="Thursday",
        is_weekend=False,
        classes=CLASSES,
        personal_fixed_blocks=[
            {"title": "Meditation", "time": "06:00", "duration_min": 20,
             "days": ["daily"], "non_negotiable": True}
        ],
        academic_events=[
            {"subject": "DSA", "due_date": "2026-10-17", "type": "cie"},
            {"subject": "DBMS", "due_date": "2026-10-25", "type": "cie"},
        ],
        academic_calendar={"holidays": []},
        user_profile=PROFILE,
        pending_tasks=[{"title": "Buy lab manual", "due_date": "2026-10-15", "priority": "high"}],
        recent_notes=[{"body": "DSA unit 4 weightage", "tags": ["dsa"], "ai_summary": "graphs"}],
    )
    kwargs.update(overrides)
    return problem_from_context(**kwargs)


def test_extraction_pins_classes_as_fixed():
    problem = _extract()
    titles = {b.title for b in problem.fixed}
    assert "Data Structures" in titles
    assert "DBMS" in titles


def test_extraction_adds_sleep_as_the_last_fixed_block():
    problem = _extract()
    assert problem.fixed[-1].title == "Sleep"
    assert problem.fixed[-1].time == "23:00"


def test_extraction_skips_classes_on_a_weekend():
    problem = _extract(weekday_name="Saturday", is_weekend=True)
    titles = {b.title for b in problem.fixed}
    assert "Data Structures" not in titles
    # Sleep is still there.
    assert "Sleep" in titles


def test_extraction_skips_study_blocks_on_a_holiday():
    problem = _extract(academic_calendar={"holidays": [{"date": "2026-10-15", "name": "Deepavali"}]})
    study = [f for f in problem.flexible if f.type == "study"]
    assert not study, "a holiday should relax study expectations"


def test_extraction_skips_study_blocks_on_sunday():
    problem = _extract(weekday_name="Sunday", is_weekend=True)
    study = [f for f in problem.flexible if f.type == "study"]
    assert not study


def test_extraction_marks_a_critical_cie():
    problem = _extract()
    dsa = next(f for f in problem.flexible if "DSA" in f.title)
    assert "CRITICAL" in dsa.title
    assert dsa.duration_min == 180


def test_extraction_gives_a_far_exam_a_shorter_block():
    problem = _extract()
    dbms = next(f for f in problem.flexible if "DBMS" in f.title)
    assert "CRITICAL" not in dbms.title
    assert dbms.duration_min < 180


def test_extraction_includes_tasks_due_today():
    problem = _extract()
    task = next(f for f in problem.flexible if f.title.startswith("Task:"))
    assert "lab manual" in task.title
    assert task.duration_min == 60


def test_extraction_ignores_tasks_due_on_another_day():
    problem = _extract(
        pending_tasks=[{"title": "Later", "due_date": "2026-11-01", "priority": "high"}]
    )
    assert not [f for f in problem.flexible if f.title.startswith("Task:")]


def test_extraction_adds_a_movie_night_on_saturday():
    problem = _extract(
        weekday_name="Saturday",
        is_weekend=True,
        movie_info={"title": "Inception", "rating": "8.8", "overview": "Heists."},
    )
    movie = next(f for f in problem.flexible if f.title.startswith("Movie Night:"))
    assert "Inception" in movie.title
    assert movie.type == "ent"


def test_extraction_omits_movie_night_on_a_weekday():
    problem = _extract()
    assert not [f for f in problem.flexible if f.title.startswith("Movie Night:")]


def test_extraction_adds_gym_only_on_gym_days():
    thursday = _extract()  # Thu is not in [Mon, Wed, Fri]
    assert not [f for f in thursday.flexible if f.title == "Gym"]

    monday = _extract(date="2026-10-19", weekday_name="Monday")
    assert [f for f in monday.flexible if f.title == "Gym"]


def test_extracted_problem_solves_and_verifies():
    """The whole point: extract -> solve -> verify, end to end."""
    problem = _extract()
    result = solve(problem)
    assert result.ok, result.status
    assert not _violation_codes(result.blocks, problem), _violation_codes(result.blocks, problem)

    titles = [b["title"] for b in result.blocks]
    assert "Data Structures" in titles
    assert "Sleep" in titles
    assert titles[-1] == "Sleep"
    assert any("CIE prep" in t for t in titles)


def test_extracted_problem_on_a_busy_day_still_verifies():
    """Six subjects with exams inside a week: the old builder could not do this."""
    events = [
        {"subject": f"Sub{i}", "due_date": "2026-10-1%d" % (i + 1), "type": "cie"}
        for i in range(6)
    ]
    problem = _extract(academic_events=events)
    result = solve(problem)
    assert result.ok
    codes = _violation_codes(result.blocks, problem)
    assert not codes, codes


def test_a_problem_where_nothing_fits_yields_an_empty_schedule():
    """An impossible day drops every item, and the verifier rejects the result.

    That is the intended failure mode: the caller sees "empty" rather than a
    fabricated schedule, and can fall back or tell the user the day is full.
    """
    problem = SchedulingProblem(
        date="2026-10-15",
        day_start="07:00",
        day_end="07:30",
        flexible=[FlexibleItem("Huge", 600)],
    )
    result = solve(problem)
    assert result.ok
    assert result.blocks == []
    assert "Huge" in result.dropped
    assert "empty" in _violation_codes(result.blocks, problem)


def test_sleep_crossing_midnight_is_not_collapsed():
    """Regression: Sleep starts at the sleep target and lasts 8 hours, so it
    necessarily runs past day_end. Clamping it to the window collapsed it to
    zero length and let flexible blocks be scheduled on top of it."""
    problem = SchedulingProblem(
        date="2026-10-15",
        day_start="07:00",
        day_end="23:00",
        fixed=[FixedBlock("23:00", "Sleep", type="rest", duration_min=480)],
        flexible=[FlexibleItem("Late study", duration_min=60)],
    )
    result = solve(problem)

    assert result.ok
    sleep = next(b for b in result.blocks if b["title"] == "Sleep")
    assert sleep["duration_min"] == 480, "Sleep must keep its full duration"

    # The flexible block cannot sit on top of Sleep.
    codes = _violation_codes(result.blocks, problem)
    assert not codes, codes


def test_back_to_back_classes_are_not_flagged_as_overlapping():
    """A real timetable has classes ending exactly when the next begins.

    The buffer exists to separate a *flexible* block from its neighbours, not
    to invent gaps between fixed lectures.
    """
    problem = SchedulingProblem(
        date="2026-10-15",
        day_start="07:00",
        day_end="23:00",
        buffer_min=15,
        fixed=[
            FixedBlock("09:00", "DSA", duration_min=60),
            FixedBlock("10:00", "DBMS", duration_min=60),
            FixedBlock("11:00", "CN", duration_min=60),
        ],
    )
    blocks = [
        _block("09:00", "DSA", 60, "personal"),
        _block("10:00", "DBMS", 60, "personal"),
        _block("11:00", "CN", 60, "personal"),
        _block("23:00", "Sleep", 480, "rest"),
    ]
    assert is_valid(blocks, problem)


def test_a_flexible_block_still_needs_the_buffer_around_a_class():
    problem = SchedulingProblem(
        date="2026-10-15",
        day_start="07:00",
        day_end="23:00",
        buffer_min=15,
        fixed=[FixedBlock("10:00", "DBMS", duration_min=60)],
    )
    # Starts the instant the class ends: no buffer.
    assert "overlap" in _violation_codes(
        [
            _block("10:00", "DBMS", 60, "personal"),
            _block("11:00", "Study", 60),
            _block("23:00", "Sleep", 480, "rest"),
        ],
        problem,
    )
    # Fifteen minutes clear: accepted.
    assert is_valid(
        [
            _block("10:00", "DBMS", 60, "personal"),
            _block("11:15", "Study", 60),
            _block("23:00", "Sleep", 480, "rest"),
        ],
        problem,
    )


def test_solver_and_verifier_agree_on_a_full_weekday():
    """The round-trip property on a realistic extracted problem."""
    problem = problem_from_context(
        date="2026-10-15",
        weekday_name="Thursday",
        is_weekend=False,
        classes=[
            {"time": "09:00", "subject": "Data Structures", "room": "CS-201", "duration_min": 60},
            {"time": "10:00", "subject": "DBMS", "room": "CS-202", "duration_min": 60},
            {"time": "11:00", "subject": "Computer Networks", "room": "CS-203", "duration_min": 60},
        ],
        personal_fixed_blocks=[
            {"title": "Meditation", "time": "06:00", "duration_min": 20,
             "days": ["daily"], "non_negotiable": True},
            {"title": "Gym", "time": "18:00", "duration_min": 60,
             "days": ["Mon", "Wed", "Fri"], "non_negotiable": False},
        ],
        academic_events=[
            {"subject": "DSA", "due_date": "2026-10-17", "type": "cie"},
            {"subject": "DBMS", "due_date": "2026-10-20", "type": "cie"},
        ],
        academic_calendar={"holidays": []},
        user_profile={
            "wake_time": "07:00",
            "sleep_target": "23:00",
            "peak_focus_time": "morning",
            "gym_days": [],
            "gym_time": "18:00",
        },
        pending_tasks=[{"title": "Buy lab manual", "due_date": "2026-10-15", "priority": "high"}],
        recent_notes=[{"body": "DSA graphs weightage", "tags": ["dsa"], "ai_summary": "graphs"}],
    )

    result = solve(problem)
    assert result.ok, result.status
    codes = _violation_codes(result.blocks, problem)
    assert not codes, codes

    titles = [b["title"] for b in result.blocks]
    assert titles[-1] == "Sleep"
    assert "Data Structures" in titles
    assert any("CIE prep" in t for t in titles)


def test_solve_result_without_ortools_still_reports_status():
    """Guards the ImportError path: it must degrade, not raise."""
    result = SolveResult(blocks=[], status="no_solver", dropped=["x"])
    assert not result.ok
    assert result.dropped == ["x"]


# ── Personal blocks: the `days` default ──────────────────────────────────────


def test_a_personal_block_with_no_days_applies_every_day():
    """A routine saved without a `days` list must not vanish.

    save_user_info defaults `days` to ["daily"], so this never bites when the
    block comes through the parser. Anything written by the API or the client
    omits it, and the reader then dropped the block with no signal at all.
    """
    problem = _extract(personal_fixed_blocks=[
        {"title": "Gym", "time": "18:00", "duration_min": 60}
    ])
    assert "Gym" in {b.title for b in problem.fixed}


def test_a_personal_block_with_days_is_still_filtered():
    """The default is 'every day', not 'always ignore the filter'."""
    problem = _extract(personal_fixed_blocks=[
        {"title": "Gym", "time": "18:00", "duration_min": 60, "days": ["Sat", "Sun"]}
    ])
    assert "Gym" not in {b.title for b in problem.fixed}


def test_a_monday_block_is_dropped_on_a_thursday():
    problem = _extract(personal_fixed_blocks=[
        {"title": "Morning run", "time": "06:30", "duration_min": 30, "days": ["Mon"]}
    ])
    assert "Morning run" not in {b.title for b in problem.fixed}


def test_the_gym_rule_still_requires_configured_gym_days():
    """The flexible gym item is separate from a saved personal block."""
    problem = _extract(user_profile={**PROFILE, "gym_days": []})
    assert not any(i.title == "Gym" for i in problem.flexible)
