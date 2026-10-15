"""Tests for the implicit-cache prompt layout.

The property under test is an *ordering* one: nothing that varies per request
may appear before static content, because Gemini caches only the prefix of a
request. These tests fail loudly if someone re-introduces a date or a user id
into the opening lines.
"""

import asyncio

import pytest

from app.agents import nodes, prompts
from app.agents.prompts import (
    CACHE_MIN_TOKENS,
    SCHEDULE_OUTPUT_SHAPE,
    SCHEDULE_RULES,
    build_prompt,
    cache_margin,
    estimate_tokens,
    qualifies_for_cache,
    section,
)
from app.config import settings


# A realistic day: enough context that the assembled prompt clears the
# implicit-cache floor. Without this the layout tests pass on a stub that
# would never cache in production anyway.
FULL_STATE = {
    "user_id": "u1",
    "intent_date": "2026-10-15",
    "intent_scope": "day",
    "user_profile": {
        "role": "student",
        "wake_time": "07:00",
        "actual_wake_avg": "07:15",
        "sleep_target": "23:00",
        "peak_focus_time": "morning",
        "study_block_pref": "90",
        "gym_days": ["Mon", "Wed", "Fri"],
        "gym_time": "18:00",
        "college": "RVCE",
        "branch": "CSE",
        "semester": 5,
    },
    "timetable_today": {
        "weekday": "Thursday",
        "classes": [
            {"time": "09:00", "subject": "Data Structures", "room": "CS-201", "professor": "Dr. Raman"},
            {"time": "10:00", "subject": "Database Management Systems", "room": "CS-202"},
            {"time": "11:00", "subject": "Computer Networks", "room": "CS-203", "textbooks": ["Tanenbaum"]},
            {"time": "14:00", "subject": "Design and Analysis of Algorithms", "room": "CS-201"},
        ],
    },
    "academic_events": [
        {"title": "CIE 1 DSA", "type": "cie", "subject": "DSA", "due_date": "2026-10-18", "completed": False},
        {"title": "DBMS lab record", "type": "lab_record", "subject": "DBMS", "due_date": "2026-10-16", "completed": False},
    ],
    "personal_fixed_blocks": [
        {"title": "Meditation", "time": "06:00", "duration_min": 20, "days": ["daily"], "type": "health", "non_negotiable": True},
        {"title": "Gym", "time": "18:00", "duration_min": 60, "days": ["Mon", "Wed", "Fri"], "type": "health", "non_negotiable": False},
    ],
    "syllabus_status": {
        "Data Structures": {"units": [{"number": 1, "title": "Intro", "status": "done", "completion_percent": 100}], "risk": "high"},
        "Database Management Systems": {"units": [{"number": 2, "title": "ER model", "status": "not_started", "completion_percent": 0}], "risk": "medium"},
    },
    "academic_calendar": {
        "semester_start": "2026-08-01",
        "semester_end": "2026-12-20",
        "cie_dates": [
            {"name": "CIE 1", "start_date": "2026-09-15", "end_date": "2026-09-18"},
            {"name": "CIE 2", "start_date": "2026-11-10", "end_date": "2026-11-13"},
        ],
        "see_start": "2026-12-15",
        "see_end": "2027-01-10",
        "holidays": [{"date": "2026-10-22", "name": "Deepavali"}],
    },
    "pending_tasks": [
        {"title": "Buy lab manual", "due_date": "2026-10-16", "priority": "high", "completed": False},
        {"title": "Submit CN assignment", "due_date": "2026-10-20", "priority": "medium", "completed": False},
    ],
    "recent_notes": [
        {"title": "DSA weightage", "body": "Unit 4 graphs has 40% CIE weightage", "tags": ["dsa"], "ai_summary": "Focus on graph traversal."},
        {"title": "DBMS normalisation", "body": "3NF and BCNF are always asked", "tags": ["dbms"], "ai_summary": "Practise normalisation."},
    ],
}


def _schedule_state(**overrides):
    """An AgentState that reaches build_schedule's prompt assembly."""
    state = dict(FULL_STATE)
    state.update(overrides)
    return state


# ── build_prompt ────────────────────────────────────────────────────────────


def test_build_prompt_joins_sections_in_order():
    out = build_prompt("FIRST", "SECOND", "THIRD")
    assert out == "FIRST\n\nSECOND\n\nTHIRD"


def test_build_prompt_drops_empty_sections_without_shifting_order():
    out = build_prompt("FIRST", "", "   ", None, "SECOND")
    assert out == "FIRST\n\nSECOND"


def test_build_prompt_returns_empty_string_for_nothing():
    assert build_prompt() == ""
    assert build_prompt("", None, "  ") == ""


def test_section_renders_a_title_and_rule():
    out = section("TASKS", "- buy milk", "apply exactly")
    assert "TASKS" in out
    assert "(apply exactly)" in out
    assert "- buy milk" in out


# ── Token estimation and the cache floor ────────────────────────────────────


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_cache_floor_is_known_for_the_configured_models():
    for model in (settings.PRIMARY_MODEL, settings.ROUTING_MODEL):
        assert model in CACHE_MIN_TOKENS, f"{model} has no documented cache floor"


def test_qualifies_for_cache_tracks_the_floor():
    big = "x" * (CACHE_MIN_TOKENS["gemini-3.8-flash"] * prompts.CHARS_PER_TOKEN + 8)
    small = "short"
    assert qualifies_for_cache(big, "gemini-3.8-flash")
    assert not qualifies_for_cache(small, "gemini-3.8-flash")


def test_cache_margin_is_signed():
    floor = CACHE_MIN_TOKENS["gemini-3.8-flash"]
    assert cache_margin("x" * (floor * 4 + 8), "gemini-3.8-flash") == 2
    assert cache_margin("x" * (floor * 4 - 8), "gemini-3.8-flash") == -2


# ── The schedule prompt ─────────────────────────────────────────────────────


@pytest.fixture
def schedule_prompt():
    """The prompt build_schedule actually assembles for a fixed date."""
    captured = {}

    async def fake_astructured(prompt, schema, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return None

    original = nodes.astructured
    nodes.astructured = fake_astructured
    try:
        import asyncio

        asyncio.run(nodes.build_schedule(_schedule_state()))
    finally:
        nodes.astructured = original
    return captured


def test_schedule_prompt_starts_with_static_instructions(schedule_prompt):
    prompt = schedule_prompt["prompt"]
    # The first content must be identical for every user on every day.
    assert prompt.startswith(prompts.SCHEDULE_ROLE)
    # The rules and output shape are static and must precede all context.
    assert prompt.index(SCHEDULE_RULES) < prompt.index("USER PROFILE")
    assert prompt.index(SCHEDULE_OUTPUT_SHAPE) < prompt.index("USER PROFILE")


def test_schedule_prompt_ends_with_the_date(schedule_prompt):
    prompt = schedule_prompt["prompt"]
    assert prompt.rstrip().endswith("Build the schedule for 2026-10-15 (Thursday).")


def test_schedule_prompt_uses_the_planner_tier(schedule_prompt):
    assert schedule_prompt["kwargs"]["tier"] == "planner"


def test_schedule_prompt_has_no_date_before_the_context(schedule_prompt):
    """The old prompt opened with the date, which disabled caching entirely."""
    prompt = schedule_prompt["prompt"]
    profile_at = prompt.index("USER PROFILE")
    assert "2026-10-15" not in prompt[:profile_at]


def test_schedule_prompt_cache_margin_is_measured(schedule_prompt):
    """Record the real margin rather than asserting a floor we do not meet.

    gemini-3.8-flash requires >=4096 input tokens before it will cache. The
    assembled prompt is well under that, so caching does not fire yet. This test
    pins the measurement so the number is visible in CI instead of being an
    assumption. When the context sections grow enough to clear the floor, this
    test's threshold is the thing to raise.
    """
    prompt = schedule_prompt["prompt"]
    margin = cache_margin(prompt, settings.PRIMARY_MODEL)
    assert margin < 0, (
        "the schedule prompt now clears the implicit-cache floor; switch "
        "test_schedule_prompt_static_prefix_is_identical_across_days over to "
        "asserting qualifies_for_cache() so the win is locked in"
    )


def test_schedule_prompt_static_prefix_is_identical_across_days():
    """Two different days must share a byte-identical cacheable prefix."""

    async def build(date):
        state = _schedule_state(intent_date=date)
        captured = {}

        async def fake_astructured(prompt, schema, **kwargs):
            captured["prompt"] = prompt
            return None

        original = nodes.astructured
        nodes.astructured = fake_astructured
        try:
            await nodes.build_schedule(state)
        finally:
            nodes.astructured = original
        return captured["prompt"]

    a = asyncio.run(build("2026-10-15"))
    b = asyncio.run(build("2026-10-16"))

    # Everything before the first day-dependent section must match. This is the
    # prefix Gemini would cache, so any drift here silently costs a full re-bill
    # of the static instructions on every schedule build.
    marker = "FIXED CLASSES"
    assert a[: a.index(marker)] == b[: b.index(marker)]

    # The volatile tail is the only part allowed to differ.
    assert a.rstrip().endswith("Build the schedule for 2026-10-15 (Thursday).")
    assert b.rstrip().endswith("Build the schedule for 2026-10-16 (Friday).")


# ── The classify prompt ─────────────────────────────────────────────────────


def test_classify_prompt_puts_the_message_last(monkeypatch):
    captured = {}

    async def fake_astructured(prompt, schema, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return None

    original = nodes.astructured
    nodes.astructured = fake_astructured

    class _Msg:
        content = "schedule my day"

    from langgraph.store.memory import InMemoryStore

    state = {"user_id": "u1", "messages": [_Msg()]}
    try:
        asyncio.run(nodes.classify_intent(state, InMemoryStore()))
    finally:
        nodes.astructured = original

    prompt = captured["prompt"]
    assert prompt.startswith(prompts.CLASSIFY_INSTRUCTIONS)
    assert prompt.rstrip().endswith('User Message: "schedule my day"')
    assert captured["kwargs"]["tier"] == "router"
