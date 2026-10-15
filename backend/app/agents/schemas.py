"""Pydantic schemas for every structured LLM output in the backend.

WHY THESE SCHEMAS LOOK THE WAY THEY DO
--------------------------------------
Gemini's ``response_schema`` (used by ``ChatGoogleGenerativeAI.with_structured_output(
method="json_mode")``) supports only a subset of JSON Schema. Two constructs are
known to be rejected or unreliable:

1. ``anyOf`` / ``oneOf`` - produced automatically by Pydantic for every
   ``Optional[...]`` field.
2. ``$ref`` / ``$defs`` - produced for nested models. LangChain resolves these
   via ``replace_defs_in_schema`` (verified in ``langchain_google_genai.
   _function_utils``), so *nested models are safe*, but unions are not.

Therefore every field here is a concrete type with an "empty" default
(``""``, ``0``, ``[]``) instead of ``Optional``. Absent values arrive as the
empty default, which the existing call sites already treat as falsy
(``if block.get("notes"):``), so behaviour is preserved.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── Controlled vocabularies ────────────────────────────────────────────────

INTENTS = (
    "schedule_build",
    "schedule_edit",
    "entertainment_rec",
    "location_trigger",
    "info_provided",
    "general",
)

SCOPES = ("day", "week")

BLOCK_TYPES = ("work", "study", "personal", "health", "rest", "ent")

# Types produced by parse_user_info. ``personal_schedule`` is accepted as an
# alias of ``personal_block`` because older prompts and the heuristic parser
# both emit it.
INFO_TYPES = (
    "timetable",
    "timetable_update",
    "academic_event",
    "personal_block",
    "personal_schedule",
    "syllabus_update",
    "college_info",
    "task",
    "note",
    "profile_update",
    "general",
)

EVENT_TYPES = ("cie", "see", "lab_record", "viva", "assignment", "project", "exam")

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

SUBJECT_ALIASES: Dict[str, str] = {
    "dsa": "DSA",
    "data structures": "DSA",
    "data structure": "DSA",
    "dbms": "DBMS",
    "database": "DBMS",
    "databases": "DBMS",
    "daa": "DAA",
    "design and analysis of algorithms": "DAA",
    "algorithms": "DAA",
    "cn": "Computer Networks",
    "computer networks": "Computer Networks",
    "networks": "Computer Networks",
    "networking": "Computer Networks",
    "os": "Operating Systems",
    "operating systems": "Operating Systems",
    "oop": "OOP",
    "oops": "OOP",
    "java": "Java",
    "python": "Python",
    "web": "Web Technologies",
    "web tech": "Web Technologies",
    "web technologies": "Web Technologies",
    "ai": "Artificial Intelligence",
    "artificial intelligence": "Artificial Intelligence",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "maths": "Mathematics",
    "math": "Mathematics",
    "mathematics": "Mathematics",
    "discrete maths": "Discrete Mathematics",
    "discrete mathematics": "Discrete Mathematics",
    "physics": "Physics",
    "chemistry": "Chemistry",
    "electronics": "Electronics",
    "ece": "Electronics",
    "mech": "Mechanical Engineering",
    "mechanical": "Mechanical Engineering",
    "stats": "Statistics",
    "statistics": "Statistics",
    "english": "English",
    "konkani": "Konkani",
    "kannada": "Kannada",
    "hindi": "Hindi",
}


def normalise_subject(raw: str) -> str:
    """Map a free-text subject onto a canonical subject name."""
    if not raw:
        return ""
    key = raw.strip().lower().rstrip(".")
    if key in SUBJECT_ALIASES:
        return SUBJECT_ALIASES[key]
    for alias, canonical in SUBJECT_ALIASES.items():
        if alias in key:
            return canonical
    return raw.strip()


def _clean(value: Any) -> str:
    """Coerce to a stripped string; None/absent become ''."""
    if value is None:
        return ""
    return str(value).strip()


# ── Intent classification ──────────────────────────────────────────────────


class IntentClassification(BaseModel):
    """Result of classifying a user message into intent + scope + date."""

    intent: str = "general"
    scope: str = "day"
    date: str = ""

    @field_validator("intent")
    @classmethod
    def _validate_intent(cls, v: str) -> str:
        v = _clean(v).lower().replace("-", "_").replace(" ", "_")
        return v if v in INTENTS else "general"

    @field_validator("scope")
    @classmethod
    def _validate_scope(cls, v: str) -> str:
        v = _clean(v).lower()
        return v if v in SCOPES else "day"


# ── Note summarisation ─────────────────────────────────────────────────────


class NoteSummary(BaseModel):
    """LLM-produced summary of a free-text note."""

    summary: str = ""
    subject: str = "General"
    tags: List[str] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def _summary_required(cls, v: str) -> str:
        return _clean(v)

    @field_validator("subject")
    @classmethod
    def _normalise_subject(cls, v: str) -> str:
        return normalise_subject(v) or "General"

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, v: List[str]) -> List[str]:
        out: List[str] = []
        for tag in v:
            tag = _clean(tag).lstrip("#")
            if tag and tag.lower() not in {t.lower() for t in out}:
                out.append(tag)
        return out[:6]


# ── Schedule generation ────────────────────────────────────────────────────


class ScheduleBlock(BaseModel):
    """One block of a proposed daily schedule."""

    time: str = Field(description="Start time in 24h HH:MM format, e.g. 09:00")
    title: str = Field(description="Short human-readable block title")
    type: str = Field(default="work", description="One of: work, study, personal, health, rest, ent")
    duration_min: int = Field(default=60, description="Duration in minutes")
    notes: str = Field(default="", description="Context or reminder text shown to the user")

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        v = _clean(v).lower()
        return v if v in BLOCK_TYPES else "work"

    @field_validator("time")
    @classmethod
    def _validate_time(cls, v: str) -> str:
        return _clean(v)


class ScheduleProposal(BaseModel):
    """Wrapper so the LLM returns an object rather than a bare array."""

    blocks: List[ScheduleBlock] = Field(default_factory=list)


# ── Entity extraction from free text ───────────────────────────────────────


class TimetableClass(BaseModel):
    time: str = Field(default="", description="Start time HH:MM")
    subject: str = Field(default="", description="Subject name")
    room: str = Field(default="", description="Room or venue")
    professor: str = Field(default="", description="Professor name, if visible")


class SyllabusUnit(BaseModel):
    number: int = Field(default=0, description="Unit number, 1-indexed")
    title: str = Field(default="", description="Unit title")
    topics: List[str] = Field(default_factory=list)
    status: str = Field(default="not_started", description="done | in_progress | not_started")
    completion_percent: int = Field(default=0, description="0-100")
    weightage: str = Field(default="", description="Optional, e.g. '20%'")


class SyllabusSubject(BaseModel):
    """One subject of a fetched syllabus."""

    subject_code: str = ""
    subject: str = ""
    credits: int = 0
    units: List[SyllabusUnit] = Field(default_factory=list)
    cie_marks: int = 0
    see_marks: int = 0
    see_duration: str = ""
    type: str = "theory"


class SyllabusExtraction(BaseModel):
    """Wrapper so the syllabus PDF parser returns an object, not a bare array."""

    subjects: List[SyllabusSubject] = Field(default_factory=list)


class ParsedInfo(BaseModel):
    """Structured information extracted from a user's chat message.

    Deliberately flat: one object with a ``type`` discriminator plus every
    field any type might need. ``to_data()`` projects this onto the exact
    dict shape the Firestore writers expect, so no downstream code changes.
    """

    model_config = ConfigDict(extra="allow")

    type: str = "general"

    # academic_event
    title: str = ""
    event_type: str = ""
    subject: str = ""
    due_date: str = ""
    syllabus_units: List[str] = Field(default_factory=list)

    # personal_block
    time: str = ""
    days: List[str] = Field(default_factory=list)
    duration_min: int = 0
    non_negotiable: bool = True
    block_type: str = ""

    # college_info
    college: str = ""
    branch: str = ""
    semester: int = 0
    scheme: str = ""

    # syllabus_update
    units: List[SyllabusUnit] = Field(default_factory=list)

    # task
    priority: str = ""

    # note
    body: str = ""
    tags: List[str] = Field(default_factory=list)

    # timetable (weekday -> classes)
    Monday: List[TimetableClass] = Field(default_factory=list)
    Tuesday: List[TimetableClass] = Field(default_factory=list)
    Wednesday: List[TimetableClass] = Field(default_factory=list)
    Thursday: List[TimetableClass] = Field(default_factory=list)
    Friday: List[TimetableClass] = Field(default_factory=list)
    Saturday: List[TimetableClass] = Field(default_factory=list)
    Sunday: List[TimetableClass] = Field(default_factory=list)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        v = _clean(v).lower().replace("-", "_").replace(" ", "_")
        if v == "personal_schedule":
            return "personal_block"
        if v not in INFO_TYPES:
            return "general"
        return v

    @field_validator("subject")
    @classmethod
    def _normalise_subject(cls, v: str) -> str:
        return normalise_subject(v)

    @property
    def has_data(self) -> bool:
        return self.type not in ("general", "unknown")

    def timetable_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Weekday -> list of class dicts, in the shape Firestore stores."""
        return {
            day: [c.model_dump() for c in getattr(self, day)]
            for day in WEEKDAYS
            if getattr(self, day)
        }

    def to_data(self) -> Dict[str, Any]:
        """Project onto the dict shape the persistence layer expects."""
        if self.type == "academic_event":
            return {
                "title": self.title or f"{self.event_type or 'Event'} {self.subject}".strip(),
                "type": self.event_type or "cie",
                "subject": self.subject,
                "due_date": self.due_date,
                "syllabus_units": self.syllabus_units,
            }
        if self.type == "personal_block":
            return {
                "title": self.title or "Routine",
                "time": self.time or "08:00",
                "days": self.days or ["daily"],
                "duration_min": self.duration_min or 30,
                "type": self.block_type or "personal",
                "non_negotiable": self.non_negotiable,
            }
        if self.type == "college_info":
            data: Dict[str, Any] = {}
            if self.college:
                data["college"] = self.college
            if self.branch:
                data["branch"] = self.branch
            if self.semester:
                data["semester"] = self.semester
            if self.scheme:
                data["scheme"] = self.scheme
            return data
        if self.type == "syllabus_update":
            return {
                "subject": self.subject,
                "units": [u.model_dump() for u in self.units],
            }
        if self.type == "task":
            return {
                "title": self.title or "Task",
                "type": self.event_type or "reminder",
                "due_date": self.due_date,
                "priority": self.priority or "medium",
                "subject": self.subject,
            }
        if self.type == "note":
            return {
                "title": self.title,
                "body": self.body,
                "tags": self.tags,
            }
        if self.type in ("timetable", "timetable_update"):
            return self.timetable_dict()
        if self.type == "profile_update":
            # Any extra keys the model captured that are not part of the
            # known schema are treated as profile fields.
            extras = dict(self.model_extra or {})
            extras.pop("type", None)
            return extras
        return {}


# ── College calendar scraping ──────────────────────────────────────────────


class CieDate(BaseModel):
    name: str = ""
    start_date: str = ""
    end_date: str = ""


class Holiday(BaseModel):
    date: str = ""
    name: str = ""


class AcademicCalendar(BaseModel):
    """Academic calendar scraped (or estimated) for a college."""

    semester_start: str = ""
    semester_end: str = ""
    cie_dates: List[CieDate] = Field(default_factory=list)
    see_start: str = ""
    see_end: str = ""
    holidays: List[Holiday] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.semester_start or self.semester_end or self.cie_dates)


class CollegeMatch(BaseModel):
    """Result of matching free-text college input to the college database."""

    college_key: str = ""

    @field_validator("college_key")
    @classmethod
    def _clean_key(cls, v: str) -> str:
        v = _clean(v).lower().replace("'", "").replace('"', "")
        if v in ("none", "null", "unknown", "n/a", "no match", ""):
            return ""
        return v


# ── Note -> syllabus linking ───────────────────────────────────────────────


class SyllabusLink(BaseModel):
    """Maps a note onto a syllabus subject/unit. Empty subject == no match."""

    subject: str = ""
    unit_number: int = 0
    unit_title: str = ""

    @property
    def matched(self) -> bool:
        return bool(self.subject) and self.unit_number > 0
