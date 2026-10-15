"""Extract structured academic/schedule information from free text.

Two paths, same contract (``{"type": ..., "data": ...}``):

1. **LLM path** — Gemini with a JSON-schema-constrained ``ParsedInfo`` output,
   so a malformed response is impossible. This is the production path.
2. **Heuristic path** — pure Python, used for ``test_*`` users, demo mode, and
   as the fallback when the LLM is unavailable. It must therefore be *correct*,
   not just plausible: see :func:`parse_due_date`, which previously hardcoded
   ``2026-06-25`` and silently stored the wrong date for every event.
"""

import datetime
import logging
import re
from typing import Optional

from app.agents.llm import astructured, bypass_llm
from app.agents.schemas import ParsedInfo, SUBJECT_ALIASES

logger = logging.getLogger("app.agents.parser")

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thurs": 3,
    "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}

MIN_YEAR, MAX_YEAR = 2000, 2100


def _safe_date(year: int, month: int, day: int) -> Optional[datetime.date]:
    try:
        return datetime.date(year, month, day)
    except ValueError:
        return None


def parse_due_date(message: str, today: Optional[datetime.date] = None) -> str:
    """Extract a due date from free text and return it as ``YYYY-MM-DD``.

    Understands ISO dates, ``DD/MM/YYYY``, ``20 Oct`` / ``Oct 20`` /
    ``20th October 2026``, ``tomorrow`` / ``today`` / ``day after tomorrow``,
    ``next <weekday>``, and ``in N days|weeks``. A year-less date that lands in
    the past rolls forward to next year. Returns ``""`` when nothing parses.
    """
    if not message:
        return ""
    today = today or datetime.date.today()
    msg = message.lower()

    # 1. Explicit ISO date: 2026-10-20 / 2026/10/20
    m = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", msg)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if MIN_YEAR <= year <= MAX_YEAR:
            d = _safe_date(year, month, day)
            if d:
                return d.isoformat()

    # 2. Numeric with a year: 20/10/2026 or 20-10-2026
    m = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", msg)
    if m:
        a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if MIN_YEAR <= year <= MAX_YEAR:
            # Day-first (Indian/UK convention) unless only month-first parses.
            d = _safe_date(year, b, a)
            if d is None and a <= 12:
                d = _safe_date(year, a, b)
            if d:
                return d.isoformat()

    # 3. Relative keywords
    if re.search(r"\bday after tomorrow\b", msg):
        return (today + datetime.timedelta(days=2)).isoformat()
    if re.search(r"\btomorrow\b|\btmrw\b", msg):
        return (today + datetime.timedelta(days=1)).isoformat()
    if re.search(r"\btoday\b|\btonight\b", msg):
        return today.isoformat()
    if re.search(r"\byesterday\b", msg):
        return (today - datetime.timedelta(days=1)).isoformat()

    # 4. in N days / weeks / months
    m = re.search(r"\bin\s+(\d{1,3})\s+(day|days|week|weeks|month|months)\b", msg)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("day"):
            delta = datetime.timedelta(days=n)
        elif unit.startswith("week"):
            delta = datetime.timedelta(weeks=n)
        else:
            delta = datetime.timedelta(days=30 * n)
        return (today + delta).isoformat()

    # 5. Weekday names — next occurrence strictly after today
    m = re.search(
        r"\b(next\s+|this\s+|coming\s+)?"
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thurs|fri|sat|sun)\b",
        msg,
    )
    if m:
        target = WEEKDAYS[m.group(2)]
        days_ahead = target - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return (today + datetime.timedelta(days=days_ahead)).isoformat()

    # 6. "20 Oct", "20th October", "Oct 20", "October 20th", optional year
    month_pattern = "|".join(sorted(MONTHS, key=len, reverse=True))
    patterns = [
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_pattern})\b(?:\s+(\d{{4}}))?",
        rf"\b({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:\s+(\d{{4}}))?",
    ]
    for pattern in patterns:
        m = re.search(pattern, msg)
        if not m:
            continue
        groups = [g for g in m.groups() if g is not None]
        if groups[0] in MONTHS:
            month = MONTHS[groups[0]]
            day = int(groups[1])
            year = int(groups[2]) if len(groups) > 2 else today.year
        else:
            day = int(groups[0])
            month = MONTHS[groups[1]]
            year = int(groups[2]) if len(groups) > 2 else today.year
        if not (MIN_YEAR <= year <= MAX_YEAR):
            continue
        d = _safe_date(year, month, day)
        if d is None:
            continue
        if d < today:
            # A year-less date in the past means next year.
            d = _safe_date(year + 1, month, day) or d
        return d.isoformat()

    return ""


def parse_time_of_day(message: str) -> str:
    """Extract a ``HH:MM`` 24h time from free text, or ``""``."""
    if not message:
        return ""
    msg = message.lower()

    # 24h, optionally with am/pm: 18:00 / 6:30 pm
    m = re.search(r"\b([01]?\d|2[0-3])[:.](\d{2})\s*(am|pm)?\b", msg)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        ampm = m.group(3)
        if ampm == "pm" and hour < 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"

    # 12h without minutes: 6am / 6 pm
    m = re.search(r"\b(1[0-2]|[1-9])\s*(am|pm)\b", msg)
    if m:
        hour = int(m.group(1))
        if m.group(2) == "pm" and hour < 12:
            hour += 12
        elif m.group(2) == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:00"

    return ""


def detect_subject(message: str) -> str:
    """Find a canonical subject name mentioned in free text."""
    if not message:
        return ""
    msg = message.lower()
    # Longest aliases first so "data structures" wins over shorter matches.
    for alias in sorted(SUBJECT_ALIASES, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", msg):
            return SUBJECT_ALIASES[alias]
    return ""


def detect_event_type(message: str) -> str:
    """Classify an academic event from free text."""
    msg = message.lower()
    if re.search(r"\bcie[\s-]*\d\b|\bcie\b|\binternal\b|\bia[\s-]*\d\b", msg):
        return "cie"
    if re.search(r"\bsee\b|\bexternal\b|\bfinal exam\b|\bsemester end\b", msg):
        return "see"
    if re.search(r"\blab record\b|\brecord\b|\bobservation\b|\bmanual\b", msg):
        return "lab_record"
    if re.search(r"\bviva\b|\boral\b|\bpresentation\b", msg):
        return "viva"
    if re.search(r"\bassignment\b|\bhomework\b|\bhw\b|\bsubmit\b", msg):
        return "assignment"
    if re.search(r"\bproject\b|\bcapstone\b", msg):
        return "project"
    if re.search(r"\bexam\b|\btest\b|\bmidterm\b|\bmid term\b", msg):
        return "exam"
    return ""


# A message only counts as an *event* if it also carries a temporal cue.
# "DSA Unit 4 has 40% CIE weightage" is study information, not an exam.
_MONTH_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))
_TEMPORAL_CUE = re.compile(
    r"\b(due|deadline|starts?|is on|will be|tomorrow|today|tonight|yesterday|"
    r"next week|coming|upcoming|soon|schedule[ds]?|postponed|rescheduled)\b"
    rf"|\b\d{{1,2}}(st|nd|rd|th)?\s+({_MONTH_ALT})\b"
    rf"|\b({_MONTH_ALT})\s+\d{{1,2}}(st|nd|rd|th)?\b"
    r"|\bon\s+\d{1,2}[-/]\d{1,2}"
)


def _is_scheduled_event(msg: str) -> bool:
    """True when the message describes an event at a point in time."""
    if detect_event_type(msg) and _TEMPORAL_CUE.search(msg):
        return True
    return bool(detect_event_type(msg) and parse_due_date(msg))


def detect_days(message: str) -> list:
    """Extract recurrence days for a personal block."""
    msg = message.lower()
    if re.search(r"\bdaily\b|\bevery day\b|\beveryday\b|\beach day\b", msg):
        return ["daily"]
    if re.search(r"\bweekdays?\b", msg):
        return ["Mon", "Tue", "Wed", "Thu", "Fri"]
    if re.search(r"\bweekends?\b", msg):
        return ["Sat", "Sun"]

    abbr = {"mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu",
            "fri": "Fri", "sat": "Sat", "sun": "Sun"}
    days = []
    for token, short in abbr.items():
        if re.search(rf"(?<![a-z]){token}(?![a-z])", msg):
            days.append(short)
    return days or ["daily"]


def _looks_like_task(msg_lower: str) -> bool:
    """True when the message is an action the user must *do*.

    Checked with the verb at the *start* of the message, so "buy lab manual
    due tomorrow" is a task while "lab record submission due tomorrow" is an
    academic event.
    """
    return bool(
        re.match(
            r"^(buy|submit|pay|call|email|book|renew|pick up|bring|remind me to|"
            r"remind me|i need to|i have to|i must|please|todo|task)\b",
            msg_lower,
        )
    )


def _parse_heuristically(message: str) -> dict:
    """Deterministic extraction used offline and as the LLM fallback."""
    msg = (message or "").strip()
    msg_lower = msg.lower()
    if not msg_lower:
        return {"type": "general", "data": {}, "message": msg}

    # ── College info ──
    if re.search(
        r"\b(rvce|bmsce|msrit|ms ramaiah|vtu|bti|college|university|campus)\b", msg_lower
    ):
        branch = ""
        if re.search(r"\b(ise|information science)\b", msg_lower):
            branch = "ISE"
        elif re.search(r"\b(ece|electronics)\b", msg_lower):
            branch = "ECE"
        elif re.search(r"\b(cse|computer science)\b", msg_lower):
            branch = "CSE"
        elif re.search(r"\b(aiml|ai/ml|ai and ml)\b", msg_lower):
            branch = "AIML"
        elif re.search(r"\b(mech|mechanical)\b", msg_lower):
            branch = "ME"
        elif re.search(r"\b(civil)\b", msg_lower):
            branch = "CV"

        semester = 0
        m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:sem|semester)\b", msg_lower)
        if m:
            semester = int(m.group(1))

        scheme = ""
        m = re.search(r"\b(20\d{2})\s*(?:scheme|syllabus|curriculum)\b", msg_lower)
        if m:
            scheme = m.group(1)

        college = ""
        for key, name in (
            ("bti", "BTI"), ("bmsce", "BMSCE"), ("msrit", "MSRIT"),
            ("ms ramaiah", "MSRIT"), ("rvce", "RVCE"), ("vtu", "VTU"),
        ):
            if key in msg_lower:
                college = name
                break

        data: dict = {}
        if college:
            data["college"] = college
        if branch:
            data["branch"] = branch
        if semester:
            data["semester"] = semester
        if scheme:
            data["scheme"] = scheme
        if data:
            return {"type": "college_info", "data": data, "message": msg}

    # ── Tasks / reminders ── (checked before academic events: "buy lab manual
    # due tomorrow" is an errand, not a lab record exam)
    if _looks_like_task(msg_lower):
        title = msg
        for prefix in ("remind me to", "remind me", "i need to", "i have to", "i must", "please"):
            if title.lower().startswith(prefix):
                title = title[len(prefix):].strip()
        title = title[:1].upper() + title[1:] if title else "Task"
        priority = (
            "high" if re.search(r"\b(urgent|critical|asap|important)\b", msg_lower) else "medium"
        )
        return {
            "type": "task",
            "data": {
                "title": title,
                "type": "reminder",
                "due_date": parse_due_date(msg),
                "priority": priority,
                "subject": detect_subject(msg_lower),
                "completed": False,
            },
            "message": msg,
        }

    # ── Personal fixed blocks ──
    if re.search(
        r"\b(meditat\w*|workout|gym|exercise|run|jog|pray\w*|journal|reading|routine|sleep|wake)\b",
        msg_lower,
    ):
        title = "Routine"
        for keyword, name in (
            ("meditat", "Meditation"), ("workout", "Workout"), ("gym", "Gym"),
            ("exercise", "Exercise"), ("jog", "Jogging"), ("run", "Running"),
            ("pray", "Prayer"), ("journal", "Journaling"), ("reading", "Reading"),
            ("sleep", "Sleep"), ("wake", "Wake up"),
        ):
            if keyword in msg_lower:
                title = name
                break
        duration = 30
        m = re.search(r"\b(\d{1,3})\s*(minutes?|mins?|hours?|hrs?|h)\b", msg_lower)
        if m:
            value = int(m.group(1))
            duration = value * 60 if m.group(2).startswith(("h", "hr")) else value
        block_type = (
            "health"
            if re.search(r"gym|workout|exercise|jog|run|meditat|sleep", msg_lower)
            else "personal"
        )
        return {
            "type": "personal_block",
            "data": {
                "title": title,
                "time": parse_time_of_day(msg) or "06:00",
                "days": detect_days(msg),
                "duration_min": duration,
                "type": block_type,
                "non_negotiable": True,
            },
            "message": msg,
        }

    # ── Academic events ──
    event_type = detect_event_type(msg_lower) if _is_scheduled_event(msg_lower) else ""
    if event_type:
        subject = detect_subject(msg_lower)
        title_bits = [event_type.replace("_", " ").upper()]
        if subject:
            title_bits.append(subject)
        return {
            "type": "academic_event",
            "data": {
                "title": " ".join(title_bits),
                "type": event_type,
                "subject": subject,
                "due_date": parse_due_date(msg),
                "syllabus_units": [],
            },
            "message": msg,
        }

    # ── Notes ──
    if (
        re.search(r"\b(note|weightage|marks|remember|syllabus says|unit \d|chapter \d|tip)\b", msg_lower)
        or len(msg.split()) >= 8
    ):
        return {"type": "note", "data": {"title": "", "body": msg, "tags": []}, "message": msg}

    return {"type": "general", "data": {}, "message": msg}


async def parse_user_info(message: str, profile: dict, user_id: str = None) -> dict:
    """Extract structured info from a message.

    Returns ``{"type": ..., "data": ..., "message": ...}``. ``type`` is one of
    the values in ``app.agents.schemas.INFO_TYPES``; ``data`` matches the shape
    the Firestore writers expect. Never raises.
    """
    logger.info("Parsing user info from message: '%s'", (message or "")[:120])

    if bypass_llm(user_id) or (profile and str(profile.get("user_id", "")).startswith("test_")):
        logger.info("Bypassing LLM parser for user=%s", user_id)
        return _parse_heuristically(message)

    current_date = datetime.date.today().isoformat()

    last_topic = (profile or {}).get("last_discovery_topic")
    context_str = ""
    if last_topic:
        context_str = (
            f"\nContext: The user is replying to a discovery question about their "
            f"'{last_topic.replace('_', ' ')}'."
        )
        if last_topic == "college":
            context_str += (
                "\nCRITICAL: If the user provides a college name, abbreviation, or association "
                "(e.g. 'Bti', 'BTI associated with vtu', 'RVCE'), you MUST set type to "
                "'college_info' and put the college name in the 'college' field. Do not "
                "classify it as 'general' or 'note'."
            )

    prompt = f"""Extract academic/schedule information from this message. {context_str}
Current Date: {current_date}

Set "type" to exactly one of:
- "academic_event" - an exam, CIE, SEE, viva, lab record, assignment or project, ideally with a date
- "personal_block" - a recurring routine (meditation, gym, prayer, sleep)
- "college_info" - college name, branch, semester or scheme
- "syllabus_update" - progress on syllabus units
- "timetable" - a weekly class timetable (fill the Monday..Sunday fields)
- "task" - a one-off to-do or reminder ("buy lab manual", "submit assignment")
- "note" - study notes, weightage, tips, or anything informational worth saving
- "profile_update" - preferences about the user (wake time, sleep target, gym days, movie genres)
- "general" - nothing worth saving

Rules:
- Resolve relative dates ("tomorrow", "next Monday", "20 Oct", "in 3 days") to an absolute
  YYYY-MM-DD date using Current Date. If there is genuinely no date, leave due_date empty.
- Put the canonical subject name in "subject" (DSA, DBMS, Computer Networks, DAA, ...).
- For personal_block, extract the start time as 24h HH:MM and the recurrence days
  ("daily", or a list like ["Mon", "Wed"]).
- For timetable, fill only the weekdays that have classes; leave the rest empty.
- For note, put the full note text in "body".
- Never invent a date that is not stated or implied.

Message: {message}"""

    parsed = await astructured(prompt, ParsedInfo, user_id=user_id)
    if parsed is None or not parsed.has_data:
        logger.info("LLM extraction produced nothing usable; using heuristic fallback")
        return _parse_heuristically(message)

    data = parsed.to_data()
    # A note needs a body; the LLM sometimes leaves it out.
    if parsed.type == "note" and not data.get("body"):
        data["body"] = message

    logger.info("Extracted type=%s data_keys=%s", parsed.type, sorted(data))
    return {"type": parsed.type, "data": data, "message": message}
