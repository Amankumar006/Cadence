"""Tests for the LLM-driven half of schedule extraction.

The deterministic rules in problem_from_context handle classes, CIEs, tasks, gym
and the movie. extract_extra_items covers what the user just said, which no rule
can know. The tests below are mostly about the clamping: the model proposes, and
every proposal has to survive contact with a day that can only hold so much.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents import nodes  # noqa: E402
from app.agents.scheduler import problem_from_context  # noqa: E402

import pytest  # noqa: E402

TODAY_STR = "2026-10-15"  # a Thursday
UID = "u1"

PROFILE = {
    "role": "student",
    "wake_time": "07:00",
    "sleep_target": "23:00",
    "peak_focus_time": "morning",
    "gym_days": [],
}


def _extract(extra_items=None):
    return problem_from_context(
        date=TODAY_STR,
        weekday_name="Thursday",
        is_weekend=False,
        classes=[],
        personal_fixed_blocks=[],
        academic_events=[],
        academic_calendar={"holidays": []},
        user_profile=PROFILE,
        pending_tasks=[],
        recent_notes=[],
        extra_items=extra_items,
    )


def _item(**kw):
    base = {"title": "Call plumber", "duration_min": 45, "type": "work", "priority": "high"}
    base.update(kw)
    return base


# ── The extraction node ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_message_mentioning_nothing_yields_no_items(fake_llm):
    """A bare "schedule my day" must not cost an LLM call."""
    fake_llm(lambda p, s: None)
    out = await nodes.extract_extra_items(
        {"messages": [{"role": "user", "content": "schedule my day"}], "user_id": UID}
    )
    assert out == []


@pytest.mark.asyncio
async def test_a_short_message_is_skipped_without_an_llm_call(fake_llm):
    """Under 25 chars there is nothing to extract; skip the round trip."""
    calls = []

    def _resp(p, s):
        calls.append(p)
        return None

    fake_llm(_resp)
    await nodes.extract_extra_items(
        {"messages": [{"role": "user", "content": "plan thursday pls"}], "user_id": UID}
    )
    assert calls == []


@pytest.mark.asyncio
async def test_a_longer_message_triggers_the_call(fake_llm):
    calls = []

    def _resp(p, s):
        calls.append(p)
        return None

    fake_llm(_resp)
    await nodes.extract_extra_items(
        {
            "messages": [
                {"role": "user", "content": "schedule thursday, I have a dentist appointment at 4pm"}
            ],
            "user_id": UID,
        }
    )
    assert len(calls) == 1
    assert "dentist" in calls[0]


@pytest.mark.asyncio
async def test_a_real_extraction_reaches_the_solver(fake_llm):
    from app.agents.schemas import ExtraItems, ExtraSchedulingItem

    fake_llm(
        lambda p, s: ExtraItems(
            items=[ExtraSchedulingItem(title="Dentist", duration_min=60, type="personal")]
        )
    )
    out = await nodes.extract_extra_items(
        {
            "messages": [
                {"role": "user", "content": "schedule thursday, I have a dentist appointment"}
            ],
            "user_id": UID,
        }
    )
    assert out == [
        {
            "title": "Dentist",
            "duration_min": 60,
            "type": "personal",
            "priority": "medium",
            "preferred_window": "",
            "reason": "",
        }
    ]
    # And the solver will actually place it.
    problem = _extract(extra_items=out)
    assert any(i.title == "Dentist" for i in problem.flexible)


@pytest.mark.asyncio
async def test_a_failed_extraction_is_not_fatal(fake_llm):
    """A broken model response must degrade to the deterministic rules."""

    def _boom(p, s):
        raise RuntimeError("model unavailable")

    fake_llm(_boom)
    out = await nodes.extract_extra_items(
        {
            "messages": [{"role": "user", "content": "schedule thursday, I have a dentist appointment"}],
            "user_id": UID,
        }
    )
    assert out == []


@pytest.mark.asyncio
async def test_no_messages_at_all_is_safe(fake_llm):
    fake_llm(lambda p, s: None)
    assert await nodes.extract_extra_items({"user_id": UID}) == []
    assert await nodes.extract_extra_items({"messages": [], "user_id": UID}) == []


@pytest.mark.asyncio
async def test_demo_mode_bypasses_the_call(fake_llm, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "DEMO_MODE", True)
    calls = []
    fake_llm(lambda p, s: calls.append(p))
    out = await nodes.extract_extra_items(
        {
            "messages": [{"role": "user", "content": "schedule thursday, I have a dentist appointment"}],
            "user_id": UID,
        }
    )
    assert out == []
    assert calls == []


# ── Clamping: the model is not trusted with any field ───────────────────────


def test_an_absurd_duration_is_clamped_not_trusted():
    """A hallucinated 600-minute block must not consume the day."""
    problem = _extract(extra_items=[_item(duration_min=600)])
    item = [i for i in problem.flexible if i.title == "Call plumber"][0]
    assert item.duration_min == 240


def test_a_tiny_duration_is_clamped_up_to_something_solvable():
    problem = _extract(extra_items=[_item(duration_min=1)])
    item = [i for i in problem.flexible if i.title == "Call plumber"][0]
    assert item.duration_min == 15


def test_a_nonsense_duration_becomes_the_default():
    problem = _extract(extra_items=[_item(duration_min="banana")])
    item = [i for i in problem.flexible if i.title == "Call plumber"][0]
    assert item.duration_min == 60


def test_an_unknown_type_falls_back_to_work():
    problem = _extract(extra_items=[_item(type="teleportation")])
    item = [i for i in problem.flexible if i.title == "Call plumber"][0]
    assert item.type == "work"


def test_an_unknown_priority_falls_back_to_medium():
    problem = _extract(extra_items=[_item(priority="URGENT!!!")])
    item = [i for i in problem.flexible if i.title == "Call plumber"][0]
    assert item.weight == 2.0


def test_an_inflated_priority_cannot_outrank_a_critical_cie():
    """Weight is derived from priority only, so 'high' never beats a CIE."""
    problem = _extract(extra_items=[_item(priority="high")])
    item = [i for i in problem.flexible if i.title == "Call plumber"][0]
    assert item.weight == 3.0  # the cap
    # A critical CIE is weight 3.0 too but with urgency 1.0, so it wins ties.
    assert item.urgency < 1.0


def test_a_blank_title_is_dropped():
    problem = _extract(extra_items=[_item(title="   "), _item(title="", duration_min=30)])
    assert not [i for i in problem.flexible if "plumber" in i.title.lower()]


def test_a_long_title_is_truncated():
    problem = _extract(extra_items=[_item(title="x" * 200)])
    item = problem.flexible[-1]
    assert len(item.title) == 60


def test_a_malformed_preference_window_is_ignored_not_fatal():
    for bad in ["", "afternoon", "17:00-14:00", "25:00-30:00", "noon-ish", None]:
        problem = _extract(extra_items=[_item(preferred_window=bad)])
        item = [i for i in problem.flexible if i.title == "Call plumber"][0]
        assert item.earliest is None, bad
        assert item.latest is None, bad


def test_a_valid_preference_window_becomes_a_hint():
    problem = _extract(extra_items=[_item(preferred_window="14:00-17:00")])
    item = [i for i in problem.flexible if i.title == "Call plumber"][0]
    assert item.earliest == 14 * 60
    assert item.latest == 17 * 60


def test_a_preference_window_never_becomes_a_hard_constraint():
    """The whole point: the solver may ignore the window if the day is full."""
    problem = _extract(extra_items=[_item(preferred_window="14:00-17:00", duration_min=120)])
    item = [i for i in problem.flexible if i.title == "Call plumber"][0]
    assert item.type != "fixed"
    assert item in problem.flexible


def test_no_extra_items_is_fine():
    problem = _extract(extra_items=None)
    assert not [i for i in problem.flexible if "plumber" in i.title.lower()]


def test_an_empty_list_is_fine():
    """An empty list means "deterministic rules only" -- here that is just Sleep."""
    problem = _extract(extra_items=[])
    assert not problem.flexible
    assert [b.title for b in problem.fixed] == ["Sleep"]


# ── Ordering: ad-hoc work must not displace the user's real obligations ─────


def test_extra_items_are_lower_weight_than_critical_cie_prep():
    problem = problem_from_context(
        date=TODAY_STR,
        weekday_name="Thursday",
        is_weekend=False,
        classes=[],
        personal_fixed_blocks=[],
        academic_events=[{"subject": "DSA", "due_date": TODAY_STR, "type": "cie"}],
        academic_calendar={"holidays": []},
        user_profile=PROFILE,
        pending_tasks=[],
        recent_notes=[],
        extra_items=[_item(priority="high")],
    )
    cie = [i for i in problem.flexible if "CRITICAL" in i.title][0]
    extra = [i for i in problem.flexible if i.title == "Call plumber"][0]
    assert cie.weight >= extra.weight
    assert cie.urgency > extra.urgency


def test_a_full_day_drops_the_ad_hoc_item_before_the_cie():
    """Over-commit the day; the CIE must survive and the errand must not."""
    problem = problem_from_context(
        date=TODAY_STR,
        weekday_name="Thursday",
        is_weekend=False,
        classes=[{"subject": "DSA", "time": "09:00"}],
        personal_fixed_blocks=[],
        academic_events=[{"subject": "DSA", "due_date": TODAY_STR, "type": "cie"}],
        academic_calendar={"holidays": []},
        # A 07:00-17:00 day fits the 180-min CIE but not the 240-min errand
        # alongside it, so exactly one has to go.
        user_profile={**PROFILE, "wake_time": "07:00", "sleep_target": "17:00"},
        pending_tasks=[],
        recent_notes=[],
        extra_items=[_item(priority="high", duration_min=240)],
    )
    from app.agents.scheduler import solve

    result = solve(problem)
    assert result.ok
    titles = [b["title"] for b in result.blocks]
    assert any("CRITICAL" in t for t in titles), titles
    assert not any("plumber" in t.lower() for t in titles), titles


def test_an_extra_item_is_actually_placed_on_a_normal_day():
    """The point of the feature: it must land, not just be parsed."""
    from app.agents.scheduler import solve

    problem = _extract(extra_items=[_item(duration_min=45)])
    result = solve(problem)
    assert result.ok
    assert any("plumber" in b["title"].lower() for b in result.blocks)


def test_an_extra_item_never_overlaps_anything():
    from app.agents.scheduler import solve, verify_candidate

    problem = problem_from_context(
        date=TODAY_STR,
        weekday_name="Thursday",
        is_weekend=False,
        classes=[{"subject": "DSA", "time": "09:00"}, {"subject": "DBMS", "time": "10:00"}],
        personal_fixed_blocks=[],
        academic_events=[],
        academic_calendar={"holidays": []},
        user_profile=PROFILE,
        pending_tasks=[{"title": "Lab manual", "due_date": TODAY_STR, "priority": "high"}],
        recent_notes=[],
        extra_items=[_item(duration_min=60), _item(title="Team sync", duration_min=45)],
    )
    result = solve(problem)
    assert result.ok
    assert verify_candidate(result.blocks, problem) == []


def test_at_most_a_handful_of_extra_items_are_kept():
    """A chatty model must not flood the solver."""
    from app.agents.scheduler import _EXTRA_MAX_ITEMS

    many = [_item(title=f"Errand {i}") for i in range(20)]
    problem = _extract(extra_items=many)
    extras = [i for i in problem.flexible if i.title.startswith("Errand")]
    assert len(extras) <= _EXTRA_MAX_ITEMS
