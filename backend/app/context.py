"""Typed, per-neice context loaders.

Before this module, three nodes each hand-rolled the same Firestore reads with
their own filtering rules: ``hydrate_context`` ran six parallel reads plus one
per subject with an upcoming exam, ``general_chat`` ran three of its own, and
``save_user_info_node`` re-read the profile. The filtering logic (events due
within 14 days, notes from the last 7, incomplete tasks) was duplicated and had
already drifted between them.

The rule now is: a node asks for the slice it needs, and only that slice.

Why not FastMCP?
----------------
The obvious alternative is to expose these as MCP tools. FastMCP's value is
sharing tools with *external* clients over a protocol boundary; this app has no
external clients, and every caller is in the same process. An MCP server here
would add a process hop, a serialization boundary and a failure mode in exchange
for nothing. The tool boundary that matters — one typed loader per concept, with
the filtering rules in exactly one place — is what this module provides. If an
external client ever needs these, they are already thin enough to wrap.
"""

import datetime
import logging
from typing import Any, Dict, List, Optional

from firebase_admin import firestore_async

logger = logging.getLogger("app.context")

# How far ahead an academic event is still worth hydrating.
EVENT_HORIZON_DAYS = 14
# How far back a note is still considered recent.
NOTE_FRESHNESS_DAYS = 7


def _db():
    return firestore_async.client()


def _parse_date(value: Any) -> Optional[datetime.date]:
    if not value:
        return None
    try:
        return datetime.date.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


# ── Individual slices ───────────────────────────────────────────────────────


async def load_timetable(user_id: str, date: str) -> Dict[str, Any]:
    """Classes for the weekday of ``date`` plus whether a timetable exists."""
    weekday = _weekday_of(date)
    try:
        doc = await _db().collection("timetables").document(user_id).get()
    except Exception as err:
        logger.warning("timetable read failed for %s: %s", user_id, err)
        return {"classes": [], "weekday": weekday, "timetable_exists": False}

    classes = (doc.to_dict() or {}).get(weekday, []) if doc.exists else []
    return {"classes": classes or [], "weekday": weekday, "timetable_exists": doc.exists}


async def load_academic_events(
    user_id: str, date: str, horizon_days: int = EVENT_HORIZON_DAYS
) -> List[Dict[str, Any]]:
    """Incomplete events due within ``horizon_days``, nearest first."""
    today = _parse_date(date) or datetime.date.today()
    try:
        snap = await _db().collection("academic_events").document(user_id).collection("items").get()
    except Exception as err:
        logger.warning("academic_events read failed for %s: %s", user_id, err)
        return []

    events: List[Dict[str, Any]] = []
    for doc in snap:
        data = doc.to_dict()
        data["id"] = doc.id
        if data.get("completed"):
            continue
        due = _parse_date(data.get("due_date"))
        if due is None:
            continue
        remaining = (due - today).days
        if 0 <= remaining <= horizon_days:
            data["days_remaining"] = remaining
            events.append(data)

    events.sort(key=lambda e: e.get("due_date", "9999-12-31"))
    return events


async def load_personal_blocks(user_id: str) -> List[Dict[str, Any]]:
    """The user's fixed personal routine blocks."""
    try:
        doc = await _db().collection("personal_schedule").document(user_id).get()
    except Exception as err:
        logger.warning("personal_schedule read failed for %s: %s", user_id, err)
        return []
    if not doc.exists:
        return []
    return (doc.to_dict() or {}).get("fixed_blocks", []) or []


async def load_academic_calendar(user_id: str) -> Dict[str, Any]:
    """CIE/SEE dates and holidays."""
    try:
        doc = await _db().collection("academic_calendar").document(user_id).get()
    except Exception as err:
        logger.warning("academic_calendar read failed for %s: %s", user_id, err)
        return {}
    return (doc.to_dict() or {}) if doc.exists else {}


async def load_pending_tasks(user_id: str) -> List[Dict[str, Any]]:
    """Incomplete tasks, nearest deadline first."""
    try:
        snap = await _db().collection("tasks").document(user_id).collection("items").get()
    except Exception as err:
        logger.warning("tasks read failed for %s: %s", user_id, err)
        return []

    tasks: List[Dict[str, Any]] = []
    for doc in snap:
        data = doc.to_dict()
        if data.get("completed"):
            continue
        data["id"] = doc.id
        tasks.append(data)
    tasks.sort(key=lambda t: t.get("due_date", "9999-12-31"))
    return tasks


async def load_recent_notes(
    user_id: str, date: str, freshness_days: int = NOTE_FRESHNESS_DAYS
) -> List[Dict[str, Any]]:
    """Notes created within ``freshness_days`` of ``date``.

    A note with no parseable ``created_at`` is kept: dropping it would silently
    lose the user's most recent input, which is the opposite of the intent.
    """
    today = _parse_date(date) or datetime.date.today()
    cutoff = today - datetime.timedelta(days=freshness_days)
    try:
        snap = await _db().collection("notes").document(user_id).collection("items").get()
    except Exception as err:
        logger.warning("notes read failed for %s: %s", user_id, err)
        return []

    notes: List[Dict[str, Any]] = []
    for doc in snap:
        data = doc.to_dict()
        data["id"] = doc.id
        created = _parse_datetime(data.get("created_at"))
        if created is not None and created < cutoff:
            continue
        data.setdefault("type", "text")
        notes.append(data)
    return notes


async def load_syllabus(user_id: str, subjects: List[str]) -> Dict[str, Any]:
    """Syllabus status for the named subjects, in one fan-out."""
    wanted = [s for s in dict.fromkeys(subjects) if s and str(s).lower() != "all"]
    if not wanted:
        return {}

    async def _one(subject: str):
        try:
            doc = (
                await _db()
                .collection("syllabus")
                .document(user_id)
                .collection("subjects")
                .document(subject)
                .get()
            )
            return subject, (doc.to_dict() if doc.exists else None)
        except Exception as err:
            logger.warning("syllabus read failed for %s/%s: %s", user_id, subject, err)
            return subject, None

    import asyncio

    results = await asyncio.gather(*(_one(s) for s in wanted))
    return {s: data for s, data in results if data}


# ── Composite loads ─────────────────────────────────────────────────────────


async def load_schedule_context(user_id: str, date: str) -> Dict[str, Any]:
    """Everything ``build_schedule`` needs, in one fan-out.

    This is the full context load and it is deliberately expensive: it runs
    once per schedule build, not once per message.
    """
    import asyncio

    timetable, events, blocks, calendar, tasks, notes = await asyncio.gather(
        load_timetable(user_id, date),
        load_academic_events(user_id, date),
        load_personal_blocks(user_id),
        load_academic_calendar(user_id),
        load_pending_tasks(user_id),
        load_recent_notes(user_id, date),
    )

    # Only fetch syllabi for subjects that actually have something due.
    subjects_due = {
        e.get("subject") for e in events if e.get("subject") and str(e.get("subject")).lower() != "all"
    }
    syllabus = await load_syllabus(user_id, sorted(subjects_due))

    return {
        "timetable_today": {"classes": timetable["classes"], "weekday": timetable["weekday"]},
        "academic_events": events,
        "personal_fixed_blocks": blocks,
        "academic_calendar": calendar,
        "pending_tasks": tasks,
        "task_queue": tasks,
        "recent_notes": notes,
        "syllabus_status": syllabus,
    }


async def load_chat_context(user_id: str, date: str) -> Dict[str, Any]:
    """The smaller slice ``general_chat`` needs.

    Chat answers questions about "what's on today", so it wants today's classes,
    the events and the task list. It does not need the full syllabus, the
    calendar or the notes — reading them cost three round trips per greeting.
    """
    import asyncio

    timetable, events, tasks = await asyncio.gather(
        load_timetable(user_id, date),
        load_academic_events(user_id, date),
        load_pending_tasks(user_id),
    )
    return {
        "timetable_today": {"classes": timetable["classes"], "weekday": timetable["weekday"]},
        "academic_events": events,
        "task_queue": tasks,
    }


# ── Helpers ─────────────────────────────────────────────────────────────────


def _weekday_of(date: str) -> str:
    parsed = _parse_date(date)
    return (parsed or datetime.date.today()).strftime("%A")


def _parse_datetime(value: Any) -> Optional[datetime.date]:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.datetime.fromisoformat(text).date()
    except (ValueError, TypeError):
        return None
