"""Regression tests for the free-text parser.

The original implementation hardcoded ``2026-06-25`` as the due date for every
academic event and dropped anything it could not pattern-match, so these tests
pin the behaviour down with a frozen "today" (Thursday 2026-10-15).
"""

import datetime

import pytest

from app.agents.nodes import parse_date_from_message
from app.agents.parser import (
    _looks_like_task,
    _parse_heuristically,
    detect_days,
    detect_event_type,
    detect_subject,
    parse_due_date,
    parse_time_of_day,
)

TODAY = datetime.date(2026, 10, 15)  # Thursday


@pytest.mark.parametrize(
    "message,expected",
    [
        # The bug this module exists to prevent
        ("CIE DSA is on 2026-10-20", "2026-10-20"),
        ("my SEE starts 2026-11-05", "2026-11-05"),
        ("viva for CN on 3rd November", "2026-11-03"),
        # ISO
        ("assignment on 2026-12-01", "2026-12-01"),
        ("deadline 2025-01-09", "2025-01-09"),
        ("nothing on 2026-13-40", ""),  # invalid month/day
        # Numeric day-first
        ("CIE 2 is on 20/10/2026", "2026-10-20"),
        ("CIE 2 is on 20-10-2026", "2026-10-20"),
        # Relative
        ("assignment due tomorrow", "2026-10-16"),
        ("due today", "2026-10-15"),
        ("due day after tomorrow", "2026-10-17"),
        ("was due yesterday", "2026-10-14"),
        ("deadline in 3 days", "2026-10-18"),
        ("deadline in 2 weeks", "2026-10-29"),
        # Weekdays — always the next occurrence, strictly after today
        ("lab record due next monday", "2026-10-19"),
        ("due monday", "2026-10-19"),
        ("due thursday", "2026-10-22"),  # today is Thursday -> next week
        ("due sunday", "2026-10-18"),
        # Month names
        ("exam on Oct 20", "2026-10-20"),
        ("exam on 20 Oct", "2026-10-20"),
        ("exam on 20th October", "2026-10-20"),
        ("exam on October 20th", "2026-10-20"),
        ("exam on 20th October 2027", "2027-10-20"),
        ("exam on Jan 5", "2027-01-05"),  # past month rolls to next year
        # No date at all
        ("buy lab manual", ""),
        ("", ""),
    ],
)
def test_parse_due_date(message, expected):
    assert parse_due_date(message, today=TODAY) == expected


def test_parse_due_date_never_raises_on_garbage():
    for garbage in ["%%%", "2026", "13/13/2026", "in 9999 days", "\x00\x01"]:
        assert isinstance(parse_due_date(garbage, today=TODAY), str)


@pytest.mark.parametrize(
    "message,expected",
    [
        ("gym at 18:30", "18:30"),
        ("gym at 6:30 pm", "18:30"),
        ("gym at 6pm", "18:00"),
        ("gym at 6am", "06:00"),
        ("gym at 12am", "00:00"),
        ("gym at 12pm", "12:00"),
        ("gym at 23:59", "23:59"),
        ("no time here", ""),
    ],
)
def test_parse_time_of_day(message, expected):
    assert parse_time_of_day(message) == expected


@pytest.mark.parametrize(
    "message,expected",
    [
        ("CIE DSA tomorrow", "DSA"),
        ("dbms assignment", "DBMS"),
        ("viva for CN", "Computer Networks"),
        ("operating systems exam", "Operating Systems"),
        ("design and analysis of algorithms", "DAA"),
        ("buy lab manual", ""),
        ("", ""),
    ],
)
def test_detect_subject(message, expected):
    assert detect_subject(message) == expected


@pytest.mark.parametrize(
    "message,expected",
    [
        ("CIE 1 is coming", "cie"),
        ("my SEE starts soon", "see"),
        ("lab record due", "lab_record"),
        ("viva tomorrow", "viva"),
        ("assignment on monday", "assignment"),
        ("final project review", "project"),
        ("midterm exam", "exam"),
        ("just a message", ""),
    ],
)
def test_detect_event_type(message, expected):
    assert detect_event_type(message) == expected


@pytest.mark.parametrize(
    "message,expected",
    [
        ("meditation daily", ["daily"]),
        ("gym every day", ["daily"]),
        ("workout on weekdays", ["Mon", "Tue", "Wed", "Thu", "Fri"]),
        ("run on weekends", ["Sat", "Sun"]),
        ("gym on mon wed fri", ["Mon", "Wed", "Fri"]),
        ("something else", ["daily"]),
    ],
)
def test_detect_days(message, expected):
    assert detect_days(message) == expected


def test_parse_date_from_message_defaults_to_today():
    assert parse_date_from_message("nothing here") == datetime.date.today().isoformat()


# ── Heuristic classifier ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "message",
    [
        "buy lab manual",
        "submit CN assignment",
        "pay electricity bill",
        "remind me to call mom",
        "i need to renew my bus pass",
    ],
)
def test_looks_like_task(message):
    assert _looks_like_task(message)


@pytest.mark.parametrize(
    "message",
    ["lab record submission due tomorrow", "CIE DSA is on 2026-10-20", "hi there"],
)
def test_not_a_task(message):
    assert not _looks_like_task(message)


@pytest.mark.parametrize(
    "message,expected_type",
    [
        ("I study at RVCE", "college_info"),
        ("I'm in 5th sem CSE at bmsce", "college_info"),
        ("CIE DSA is on 2026-10-20", "academic_event"),
        ("my SEE starts 2026-11-05", "academic_event"),
        ("viva for CN on 3rd November", "academic_event"),
        ("buy lab manual due tomorrow", "task"),
        ("submit CN assignment by 20 Oct", "task"),
        ("I go to gym at 6am daily", "personal_block"),
        ("meditation at 6am daily", "personal_block"),
        ("unit 2 of DSA has heavy weightage in exams", "note"),
        ("DSA Unit 4 has 40% CIE weightage", "note"),
        ("CIE 1 is coming next week", "academic_event"),
        ("hey how are you", "general"),
        ("", "general"),
    ],
)
def test_heuristic_type(message, expected_type):
    assert _parse_heuristically(message)["type"] == expected_type


def test_heuristic_academic_event_carries_the_real_date():
    parsed = _parse_heuristically("CIE DSA is on 2026-10-20")
    assert parsed["data"]["due_date"] == "2026-10-20"
    assert parsed["data"]["subject"] == "DSA"
    assert parsed["data"]["type"] == "cie"


def test_heuristic_task_strips_the_lead_verb():
    parsed = _parse_heuristically("remind me to pay electricity bill in 3 days")
    assert parsed["data"]["title"] == "Pay electricity bill in 3 days"
    assert parsed["data"]["due_date"] == "2026-10-18"


def test_heuristic_personal_block_defaults_to_6am_and_daily():
    parsed = _parse_heuristically("I go to gym daily")
    assert parsed["data"]["time"] == "06:00"
    assert parsed["data"]["days"] == ["daily"]
    assert parsed["data"]["non_negotiable"] is True


def test_heuristic_never_invents_a_date():
    for message in ["CIE DSA tomorrow", "buy milk", "hi"]:
        parsed = _parse_heuristically(message)
        assert parsed["data"].get("due_date", "") in ("", "2026-10-16")
