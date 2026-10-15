"""Tests for the deterministic (rule-based) schedule generator.

This is the offline path used for ``test_*`` users, demo mode, and whenever the
LLM is unavailable, so its behaviour is pinned here.
"""

import datetime

from app.agents.nodes import generate_rule_based_schedule

TODAY = datetime.date(2026, 10, 15)  # Thursday
TODAY_STR = TODAY.isoformat()


def _build(date=TODAY_STR, weekday="Thursday", is_weekend=False, **kwargs):
    return generate_rule_based_schedule(
        user_id="u1",
        date=date,
        weekday_name=weekday,
        is_weekend=is_weekend,
        classes=kwargs.get("classes", []),
        personal_fixed_blocks=kwargs.get("personal_fixed_blocks", []),
        academic_events=kwargs.get("academic_events", []),
        academic_calendar=kwargs.get("academic_calendar", {}),
        user_profile=kwargs.get("user_profile", {}),
        pending_tasks=kwargs.get("pending_tasks", []),
        recent_notes=kwargs.get("recent_notes", []),
    )


def _titles(blocks):
    return [b["title"] for b in blocks]


def test_always_ends_with_a_sleep_block():
    blocks = _build()
    assert blocks[-1]["title"] == "Sleep"
    assert blocks[-1]["time"] == "23:00"
    assert blocks[-1]["duration_min"] == 480


def test_blocks_are_sorted_by_start_time():
    blocks = _build(
        classes=[{"time": "09:00", "subject": "DSA", "room": "CS-201"}],
        academic_events=[{"subject": "DBMS", "due_date": TODAY_STR, "type": "cie"}],
        pending_tasks=[{"title": "Buy manual", "due_date": TODAY_STR, "priority": "high"}],
    )
    times = [b["time"] for b in blocks]
    assert times == sorted(times)


def test_personal_fixed_blocks_are_included_when_recurring():
    blocks = _build(
        personal_fixed_blocks=[
            {"title": "Meditation", "time": "06:00", "duration_min": 20, "days": ["daily"]}
        ]
    )
    assert "Meditation" in _titles(blocks)
    meditation = next(b for b in blocks if b["title"] == "Meditation")
    assert meditation["time"] == "06:00"
    assert meditation["duration_min"] == 20


def test_personal_fixed_blocks_respect_weekday_restrictions():
    blocks = _build(
        personal_fixed_blocks=[
            {"title": "Gym", "time": "07:00", "duration_min": 60, "days": ["Mon", "Wed"]}
        ]
    )
    assert "Gym" not in _titles(blocks)  # today is a Thursday


def test_classes_are_added_on_weekdays():
    blocks = _build(classes=[{"time": "09:00", "subject": "DSA", "room": "CS-201"}])
    dsa = next((b for b in blocks if b["title"] == "DSA"), None)
    assert dsa is not None
    assert dsa["time"] == "09:00"
    assert "CS-201" in dsa["notes"]


def test_classes_are_skipped_on_the_weekend():
    blocks = _build(
        date="2026-10-17",
        weekday="Saturday",
        is_weekend=True,
        classes=[{"time": "09:00", "subject": "DSA", "room": "CS-201"}],
    )
    assert "DSA" not in _titles(blocks)


def test_cie_within_three_days_gets_a_critical_three_hour_block():
    due = (TODAY + datetime.timedelta(days=2)).isoformat()
    blocks = _build(academic_events=[{"subject": "DSA", "due_date": due, "type": "cie"}])
    critical = [b for b in blocks if "CRITICAL" in b["title"]]
    assert len(critical) == 1
    assert critical[0]["duration_min"] == 180
    assert critical[0]["time"] == "15:00"


def test_cie_within_a_week_gets_a_two_hour_block():
    due = (TODAY + datetime.timedelta(days=5)).isoformat()
    blocks = _build(academic_events=[{"subject": "DSA", "due_date": due, "type": "cie"}])
    prep = [b for b in blocks if b["title"] == "DSA study prep"]
    assert len(prep) == 1
    assert prep[0]["duration_min"] == 120


def test_cie_beyond_a_week_gets_no_study_block():
    due = (TODAY + datetime.timedelta(days=20)).isoformat()
    blocks = _build(academic_events=[{"subject": "DSA", "due_date": due, "type": "cie"}])
    assert not [b for b in blocks if "DSA" in b["title"]]


def test_past_cie_gets_no_study_block():
    due = (TODAY - datetime.timedelta(days=2)).isoformat()
    blocks = _build(academic_events=[{"subject": "DSA", "due_date": due, "type": "cie"}])
    assert not [b for b in blocks if "DSA" in b["title"]]


def test_holiday_suppresses_classes_and_study_blocks():
    blocks = _build(
        classes=[{"time": "09:00", "subject": "DSA", "room": "CS-201"}],
        academic_events=[{"subject": "DSA", "due_date": TODAY_STR, "type": "cie"}],
        academic_calendar={"holidays": [{"date": TODAY_STR, "name": "Deepavali"}]},
    )
    assert "DSA" not in _titles(blocks)
    assert not [b for b in blocks if "CRITICAL" in b["title"]]


def test_saturday_gets_the_movie_night_block():
    blocks = _build(date="2026-10-17", weekday="Saturday", is_weekend=True)
    movie = [b for b in blocks if b["title"].startswith("Movie Night:")]
    assert len(movie) == 1
    assert movie[0]["time"] == "21:30"
    assert movie[0]["type"] == "ent"


def test_saturday_sleep_is_thirty_minutes_later():
    blocks = _build(date="2026-10-17", weekday="Saturday", is_weekend=True)
    assert blocks[-1]["time"] == "23:30"


def test_sleep_target_comes_from_the_profile():
    blocks = _build(user_profile={"sleep_target": "22:30"})
    assert blocks[-1]["time"] == "22:30"


def test_pending_tasks_due_today_are_included():
    blocks = _build(
        pending_tasks=[
            {"title": "Buy lab manual", "due_date": TODAY_STR, "priority": "high"},
            {"title": "Future task", "due_date": "2027-01-01", "priority": "low"},
            {"title": "Done task", "due_date": TODAY_STR, "priority": "low", "completed": True},
        ]
    )
    task_titles = [t for t in _titles(blocks) if t.startswith("Task:")]
    assert task_titles == ["Task: Buy lab manual"]


def test_dsa_notes_produce_a_study_block():
    blocks = _build(
        recent_notes=[{"body": "unit 2 is heavy", "tags": ["dsa"], "ai_summary": "focus on graphs"}]
    )
    dsa = [b for b in blocks if b["title"] == "DSA Study Block"]
    assert len(dsa) == 1
    assert dsa[0]["duration_min"] == 90


def test_unrelated_notes_do_not_produce_study_blocks():
    blocks = _build(recent_notes=[{"body": "history chapter 3", "tags": ["history"]}])
    assert not [b for b in blocks if "Study Block" in b["title"]]


def test_schedule_is_never_empty():
    assert _build()
    assert _build(date="2026-10-18", weekday="Sunday", is_weekend=True)


def test_empty_inputs_do_not_raise():
    blocks = generate_rule_based_schedule("u1", TODAY_STR, "Thursday", False, [], [], [], {}, {})
    assert [b["title"] for b in blocks] == ["Sleep"]
