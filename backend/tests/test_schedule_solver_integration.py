"""Integration: build_schedule must produce a schedule CP-SAT can stand behind.

These tests drive the real node with a fake LLM, so they exercise the whole
extract -> solve -> verify -> annotate path rather than the solver in isolation.
The property under test is the one the old implementation got wrong: a busy day
must not contain two blocks at the same start time.
"""

import datetime

import pytest

from app.agents import nodes
from app.agents.scheduler import _to_minutes, verify_candidate

TODAY = datetime.date(2026, 10, 15)  # Thursday
TODAY_STR = TODAY.isoformat()

BUSY_STATE = {
    "user_id": "u1",
    "intent_date": TODAY_STR,
    "intent_scope": "day",
    "user_profile": {
        "role": "student",
        "wake_time": "07:00",
        "sleep_target": "23:00",
        "peak_focus_time": "morning",
        "study_block_pref": "90",
        "gym_days": ["Mon", "Wed", "Fri"],
        "gym_time": "18:00",
    },
    "timetable_today": {
        "weekday": "Thursday",
        "classes": [
            {"time": "09:00", "subject": "Data Structures", "room": "CS-201"},
            {"time": "10:00", "subject": "Database Management Systems", "room": "CS-202"},
            {"time": "11:00", "subject": "Computer Networks", "room": "CS-203"},
            {"time": "14:00", "subject": "Design and Analysis of Algorithms", "room": "CS-201"},
        ],
    },
    # Two CIEs inside three days: the exact case that collided before.
    "academic_events": [
        {"title": "CIE 1 DSA", "subject": "DSA", "due_date": "2026-10-17", "type": "cie"},
        {"title": "CIE 1 DBMS", "subject": "DBMS", "due_date": "2026-10-18", "type": "cie"},
        {"title": "CIE 1 CN", "subject": "CN", "due_date": "2026-10-17", "type": "cie"},
    ],
    "personal_fixed_blocks": [
        {"title": "Meditation", "time": "06:00", "duration_min": 20,
         "days": ["daily"], "non_negotiable": True},
    ],
    "syllabus_status": {},
    "academic_calendar": {"holidays": []},
    "pending_tasks": [
        {"title": "Buy lab manual", "due_date": TODAY_STR, "priority": "high"},
        {"title": "Submit CN assignment", "due_date": TODAY_STR, "priority": "medium"},
    ],
    "recent_notes": [
        {"title": "DSA weightage", "body": "Unit 4 graphs has 40% weightage",
         "tags": ["dsa"], "ai_summary": "Focus on graph traversal."},
    ],
}


@pytest.fixture
def solver_fake_llm(fake_llm):
    """A fake that declines to annotate.

    build_schedule calls the LLM once to write per-block notes. Returning None
    makes that pass a no-op, so these tests pin the *placement* the solver
    produced rather than whatever the model would have said about it.

    Without this fixture the node builds a real client with the test key and
    hangs on the network -- the same trap every other LLM-touching test avoids.
    """
    return fake_llm(lambda prompt, schema: None)


async def _run_build(state):
    """Run build_schedule and return just the proposed blocks."""
    result = await nodes.build_schedule(state)
    return result["proposed_schedule"] or []


def _starts(blocks):
    return [_to_minutes(b["time"]) for b in blocks]


async def test_a_busy_day_has_no_two_blocks_at_the_same_start(solver_fake_llm):
    """The regression this whole module exists for."""
    blocks = await _run_build(BUSY_STATE)

    assert blocks, "the schedule must not be empty"
    starts = _starts(blocks)
    assert len(starts) == len(set(starts)), (
        f"blocks collided at {[b['time'] for b in blocks]}"
    )


async def test_a_busy_day_schedule_passes_the_verifier(solver_fake_llm):
    """build_schedule must never hand back a schedule it cannot defend."""
    from app.agents.scheduler import problem_from_context

    blocks = await _run_build(BUSY_STATE)
    problem = problem_from_context(
        date=TODAY_STR,
        weekday_name="Thursday",
        is_weekend=False,
        classes=BUSY_STATE["timetable_today"]["classes"],
        personal_fixed_blocks=BUSY_STATE["personal_fixed_blocks"],
        academic_events=BUSY_STATE["academic_events"],
        academic_calendar=BUSY_STATE["academic_calendar"],
        user_profile=BUSY_STATE["user_profile"],
        pending_tasks=BUSY_STATE["pending_tasks"],
        recent_notes=BUSY_STATE["recent_notes"],
    )
    codes = {v.code for v in verify_candidate(blocks, problem)}
    assert not codes, f"build_schedule returned an invalid schedule: {codes}"


async def test_classes_keep_their_timetable_times(solver_fake_llm):
    blocks = await _run_build(BUSY_STATE)
    by_title = {b["title"]: b for b in blocks}
    assert by_title["Data Structures"]["time"] == "09:00"
    assert by_title["Database Management Systems"]["time"] == "10:00"
    assert by_title["Computer Networks"]["time"] == "11:00"


async def test_the_schedule_ends_with_sleep(solver_fake_llm):
    blocks = await _run_build(BUSY_STATE)
    assert blocks[-1]["title"] == "Sleep"
    assert blocks[-1]["time"] == "23:00"
    assert blocks[-1]["duration_min"] == 480


async def test_blocks_are_sorted_by_start_time(solver_fake_llm):
    blocks = await _run_build(BUSY_STATE)
    assert _starts(blocks) == sorted(_starts(blocks))


async def test_the_high_priority_task_survives_a_full_day(solver_fake_llm):
    """The day in BUSY_STATE is genuinely over-committed.

    The solver must keep the high-priority task and drop the medium one, rather
    than either dropping both or overlapping them. That is the whole point of
    handing placement to CP-SAT: the old builder emitted both at a hardcoded
    14:00.
    """
    blocks = await _run_build(BUSY_STATE)
    titles = [b["title"] for b in blocks]
    assert any("lab manual" in t for t in titles), "the high-priority task was dropped"


async def test_priority_decides_what_is_dropped_when_the_day_is_full(solver_fake_llm):
    """With only room for one task, the higher-priority one wins."""
    state = {
        "user_id": "u1",
        "intent_date": TODAY_STR,
        "user_profile": {"wake_time": "07:00", "sleep_target": "09:00"},
        "timetable_today": {"classes": [], "weekday": "Thursday"},
        "academic_events": [],
        "personal_fixed_blocks": [],
        "syllabus_status": {},
        "academic_calendar": {},
        "pending_tasks": [
            {"title": "urgent thing", "due_date": TODAY_STR, "priority": "high"},
            {"title": "nice to have", "due_date": TODAY_STR, "priority": "low"},
        ],
        "recent_notes": [],
    }
    titles = [b["title"] for b in await _run_build(state)]
    assert any("urgent thing" in t for t in titles)
    assert not any("nice to have" in t for t in titles), "a low-priority task beat a high one"


async def test_every_block_has_the_fields_the_ui_renders(solver_fake_llm):
    for block in await _run_build(BUSY_STATE):
        assert isinstance(block["time"], str) and block["time"]
        assert isinstance(block["title"], str) and block["title"]
        assert isinstance(block["type"], str)
        assert isinstance(block["duration_min"], int) and block["duration_min"] > 0
        assert isinstance(block["notes"], str)


async def test_a_weekend_day_omits_classes(solver_fake_llm):
    saturday = dict(BUSY_STATE)
    saturday["intent_date"] = "2026-10-17"  # a Saturday
    saturday["timetable_today"] = {"weekday": "Saturday", "classes": BUSY_STATE["timetable_today"]["classes"]}
    blocks = await _run_build(saturday)

    titles = [b["title"] for b in blocks]
    assert "Data Structures" not in titles
    # Saturday still gets the movie mandate.
    assert any(t.startswith("Movie Night:") for t in titles)


async def test_the_result_is_deterministic(solver_fake_llm):
    first = await _run_build(BUSY_STATE)
    for _ in range(3):
        assert await _run_build(BUSY_STATE) == first


async def test_an_empty_context_still_yields_a_sleep_block(solver_fake_llm):
    state = {
        "user_id": "u1",
        "intent_date": TODAY_STR,
        "user_profile": {},
        "timetable_today": {"classes": [], "weekday": "Thursday"},
        "academic_events": [],
        "personal_fixed_blocks": [],
        "syllabus_status": {},
        "academic_calendar": {},
        "pending_tasks": [],
        "recent_notes": [],
    }
    blocks = await _run_build(state)
    assert [b["title"] for b in blocks] == ["Sleep"]
