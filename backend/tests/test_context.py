"""Tests for app.context: the typed per-slice context loaders.

The point of these is the filtering rules. hydrate_context and general_chat
used to re-derive them independently, and the two versions had already drifted:
hydrate_context filtered events to a 14-day horizon while general_chat just
took whatever the timetable helper happened to return. One place, one rule.
"""

import asyncio
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import context as ctx  # noqa: E402
from tests.fakes import install_fake_firestore  # noqa: E402

import pytest  # noqa: E402

TODAY = datetime.date(2026, 10, 15)
TODAY_STR = TODAY.isoformat()
UID = "ctx-user-1"


@pytest.fixture
def seeded_db(monkeypatch):
    """A FakeFirestoreClient pre-loaded with one user's data."""
    
    data = {
        ("timetables", UID): {"Thursday": [{"subject": "DSA", "time": "09:00", "room": "A1"}]},
        ("academic_events", UID, "items", "e1"): {
            "title": "CIE DSA",
            "subject": "DSA",
            "due_date": TODAY_STR,
            "completed": False,
        },
        ("academic_events", UID, "items", "e2"): {
            "title": "SEE CN",
            "subject": "CN",
            "due_date": (TODAY + datetime.timedelta(days=30)).isoformat(),
            "completed": False,
        },
        ("academic_events", UID, "items", "e3"): {
            "title": "Old CIE",
            "subject": "OS",
            "due_date": (TODAY - datetime.timedelta(days=2)).isoformat(),
            "completed": False,
        },
        ("academic_events", UID, "items", "e4"): {
            "title": "Done already",
            "subject": "DBMS",
            "due_date": TODAY_STR,
            "completed": True,
        },
        ("personal_schedule", UID): {"fixed_blocks": [{"title": "Gym", "time": "18:00"}]},
        ("academic_calendar", UID): {"holidays": {"2026-10-18": "Diwali"}},
        ("tasks", UID, "items", "t1"): {
            "title": "Lab manual",
            "due_date": TODAY_STR,
            "priority": "high",
            "completed": False,
        },
        ("tasks", UID, "items", "t2"): {
            "title": "Old assignment",
            "due_date": (TODAY - datetime.timedelta(days=5)).isoformat(),
            "completed": False,
        },
        ("tasks", UID, "items", "t3"): {
            "title": "Submitted",
            "due_date": TODAY_STR,
            "completed": True,
        },
        ("notes", UID, "items", "n1"): {
            "title": "Fresh note",
            "created_at": (TODAY - datetime.timedelta(days=1)).isoformat(),
        },
        ("notes", UID, "items", "n2"): {
            "title": "Stale note",
            "created_at": (TODAY - datetime.timedelta(days=20)).isoformat(),
        },
        ("notes", UID, "items", "n3"): {"title": "Undated note"},
        ("syllabus", UID, "subjects", "DSA"): {"units_done": 3},
    }

    holder = install_fake_firestore(monkeypatch)
    holder.db._data.update(data)
    monkeypatch.setattr(ctx, "_db", holder.client)
    return holder.db


@pytest.mark.asyncio
async def test_timetable_returns_the_weekday_classes(seeded_db):
    result = await ctx.load_timetable(UID, TODAY_STR)
    assert result["weekday"] == "Thursday"
    assert [c["subject"] for c in result["classes"]] == ["DSA"]
    assert result["timetable_exists"] is True


@pytest.mark.asyncio
async def test_timetable_for_a_missing_user_is_empty_not_an_error(seeded_db):
    result = await ctx.load_timetable("nobody", TODAY_STR)
    assert result["classes"] == []
    assert result["timetable_exists"] is False


@pytest.mark.asyncio
async def test_events_are_windowed_to_the_horizon(seeded_db):
    events = await ctx.load_academic_events(UID, TODAY_STR)
    titles = [e["title"] for e in events]
    assert "CIE DSA" in titles
    # 30 days out, past, and completed are all out of the window.
    assert "SEE CN" not in titles
    assert "Old CIE" not in titles
    assert "Done already" not in titles
    assert events[0]["days_remaining"] == 0


@pytest.mark.asyncio
async def test_events_are_sorted_nearest_first(seeded_db):
    events = await ctx.load_academic_events(UID, TODAY_STR)
    dates = [e["due_date"] for e in events]
    assert dates == sorted(dates)


@pytest.mark.asyncio
async def test_personal_blocks_and_calendar(seeded_db):
    assert await ctx.load_personal_blocks(UID) == [{"title": "Gym", "time": "18:00"}]
    assert (await ctx.load_academic_calendar(UID))["holidays"]["2026-10-18"] == "Diwali"


@pytest.mark.asyncio
async def test_pending_tasks_drop_completed_and_sort_by_due(seeded_db):
    tasks = await ctx.load_pending_tasks(UID)
    assert [t["title"] for t in tasks] == ["Old assignment", "Lab manual"]
    assert all("id" in t for t in tasks)


@pytest.mark.asyncio
async def test_recent_notes_keep_undated_ones_and_drop_stale_ones(seeded_db):
    notes = await ctx.load_recent_notes(UID, TODAY_STR)
    titles = [n["title"] for n in notes]
    assert "Fresh note" in titles
    assert "Undated note" in titles, "a note with no timestamp must not be silently dropped"
    assert "Stale note" not in titles


@pytest.mark.asyncio
async def test_recent_notes_default_a_missing_type(seeded_db):
    notes = {n["title"]: n for n in await ctx.load_recent_notes(UID, TODAY_STR)}
    assert notes["Fresh note"]["type"] == "text"


@pytest.mark.asyncio
async def test_syllabus_only_loads_the_subjects_asked_for(seeded_db):
    result = await ctx.load_syllabus(UID, ["DSA", "CN"])
    assert result == {"DSA": {"units_done": 3}}, "an absent subject must not appear as None"


@pytest.mark.asyncio
async def test_syllabus_ignores_the_all_sentinel(seeded_db):
    assert await ctx.load_syllabus(UID, ["all", "ALL", ""]) == {}


@pytest.mark.asyncio
async def test_syllabus_deduplicates_subjects(seeded_db):
    result = await ctx.load_syllabus(UID, ["DSA", "DSA", "DSA"])
    assert result == {"DSA": {"units_done": 3}}


@pytest.mark.asyncio
async def test_schedule_context_returns_every_key_the_solver_reads(seeded_db):
    result = await ctx.load_schedule_context(UID, TODAY_STR)
    for key in (
        "timetable_today",
        "academic_events",
        "personal_fixed_blocks",
        "academic_calendar",
        "pending_tasks",
        "task_queue",
        "recent_notes",
        "syllabus_status",
    ):
        assert key in result, f"build_schedule reads {key} and the loader dropped it"
    # Syllabus is derived from the events, not from a separate read of everything.
    assert result["syllabus_status"] == {"DSA": {"units_done": 3}}


@pytest.mark.asyncio
async def test_chat_context_is_a_strict_subset_of_the_schedule_context(seeded_db):
    """This is the saving: chat must not read calendar, notes or syllabus."""
    chat = await ctx.load_chat_context(UID, TODAY_STR)
    full = await ctx.load_schedule_context(UID, TODAY_STR)
    assert set(chat) < set(full), set(chat) - set(full)
    for key in chat:
        assert chat[key] == full[key], f"{key} diverged between the two loaders"


@pytest.mark.asyncio
async def test_chat_context_makes_no_syllabus_reads(seeded_db, monkeypatch):
    called = []

    async def _spy(*a, **kw):
        called.append(a)
        return {}

    monkeypatch.setattr(ctx, "load_syllabus", _spy)
    await ctx.load_chat_context(UID, TODAY_STR)
    assert called == []


@pytest.mark.asyncio
async def test_a_read_failure_degrades_to_empty_instead_of_raising(monkeypatch):
    """hydrate_context used to abort the whole turn on any single read error."""

    class _Boom:
        def collection(self, *a, **kw):
            raise RuntimeError("firestore is down")

    monkeypatch.setattr(ctx, "_db", lambda: _Boom())
    result = await ctx.load_schedule_context(UID, TODAY_STR)
    assert result["timetable_today"]["classes"] == []
    assert result["academic_events"] == []
    assert result["pending_tasks"] == []
    assert result["recent_notes"] == []


@pytest.mark.asyncio
async def test_a_read_failure_still_returns_the_expected_shape(monkeypatch):
    class _Boom:
        def collection(self, *a, **kw):
            raise RuntimeError("firestore is down")

    monkeypatch.setattr(ctx, "_db", lambda: _Boom())
    result = await ctx.load_chat_context(UID, TODAY_STR)
    assert result["timetable_today"]["weekday"] == "Thursday"
    assert result["task_queue"] == []


@pytest.mark.asyncio
async def test_the_loaders_run_concurrently(monkeypatch):
    """Six sequential round trips would cost 6x; they are gathered."""
    order = []

    def _tracker(name, delay, result):
        async def _inner(*a, **kw):
            order.append(f"start-{name}")
            await asyncio.sleep(delay)
            order.append(f"end-{name}")
            return result

        return _inner

    # Each loader must return its real shape, or load_schedule_context fails
    # before the ordering assertion ever runs.
    monkeypatch.setattr(
        ctx, "load_timetable", _tracker("t", 0.05, {"classes": [], "weekday": "Thursday"})
    )
    monkeypatch.setattr(ctx, "load_academic_events", _tracker("e", 0.05, []))
    monkeypatch.setattr(ctx, "load_personal_blocks", _tracker("p", 0.05, []))
    monkeypatch.setattr(ctx, "load_academic_calendar", _tracker("c", 0.05, {}))
    monkeypatch.setattr(ctx, "load_pending_tasks", _tracker("t2", 0.05, []))
    monkeypatch.setattr(ctx, "load_recent_notes", _tracker("n", 0.05, []))
    monkeypatch.setattr(ctx, "load_syllabus", _tracker("s", 0.0, {}))

    await ctx.load_schedule_context(UID, TODAY_STR)
    # If they were sequential, every start would be immediately followed by its
    # own end. Concurrent gathers interleave.
    assert order[1] != order[0].replace("start", "end")


@pytest.mark.asyncio
async def test_a_malformed_date_does_not_raise(seeded_db):
    result = await ctx.load_timetable(UID, "not-a-date")
    assert result["weekday"] == datetime.date.today().strftime("%A")


@pytest.mark.asyncio
async def test_a_malformed_event_date_is_skipped(monkeypatch):
    class _Snap:
        def __iter__(self):
            return iter([_Doc({"title": "bad", "due_date": "not-a-date", "completed": False})])

    class _Doc:
        def __init__(self, data):
            self._data = data
            self.id = "x"

        def to_dict(self):
            return dict(self._data)

    class _Coll:
        def document(self, *a, **kw):
            return self

        def collection(self, *a, **kw):
            return self

        async def get(self):
            return _Snap()

    monkeypatch.setattr(ctx, "_db", lambda: _Coll())
    assert await ctx.load_academic_events(UID, TODAY_STR) == []
