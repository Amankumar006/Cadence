"""Tests for the onboarding / knowledge-gap discovery priority order."""

from app.agents.discovery import identify_missing_slots

TIMETABLE = {"Monday": [{"time": "09:00", "subject": "DSA"}]}
FULL_PROFILE = {
    "college": "RVCE",
    "personal_blocks": [{"title": "Meditation", "time": "06:00"}],
    "completed_discovery": [],
}


def test_no_data_at_all_asks_for_the_timetable_first():
    slot = identify_missing_slots({}, {}, [])
    assert slot["topic"] == "timetable"


def test_timetable_but_no_college_asks_for_college():
    slot = identify_missing_slots({}, TIMETABLE, [])
    assert slot["topic"] == "college"


def test_timetable_and_college_but_no_events_asks_for_events():
    slot = identify_missing_slots(FULL_PROFILE, TIMETABLE, [])
    assert slot["topic"] == "academic_events"


def test_only_personal_blocks_missing_asks_for_them():
    profile = dict(FULL_PROFILE)
    profile.pop("personal_blocks")
    slot = identify_missing_slots(profile, TIMETABLE, [{"title": "CIE DSA"}])
    assert slot["topic"] == "personal_blocks"


def test_everything_known_returns_none():
    slot = identify_missing_slots(FULL_PROFILE, TIMETABLE, [{"title": "CIE DSA"}])
    assert slot is None


def test_completed_topics_are_not_asked_again():
    slot = identify_missing_slots(
        {"college": "RVCE", "completed_discovery": ["timetable", "college"]},
        {},
        [{"title": "CIE DSA"}],
    )
    # Timetable and college are done, events are known -> personal blocks next.
    assert slot["topic"] == "personal_blocks"


def test_all_topics_completed_returns_none_even_with_no_data():
    slot = identify_missing_slots(
        {"completed_discovery": ["timetable", "college", "academic_events", "personal_blocks"]},
        {},
        [],
    )
    assert slot is None


def test_question_is_always_a_non_empty_string():
    slot = identify_missing_slots({}, {}, [])
    assert isinstance(slot["question"], str) and slot["question"].strip()
