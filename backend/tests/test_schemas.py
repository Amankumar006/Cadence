"""Tests for the structured-output schemas.

Every schema field is a concrete type with an empty default, so the JSON schema
sent to Gemini never contains ``anyOf`` (which ``response_schema`` rejects).
"""

import json

import pytest
from pydantic import ValidationError

from app.agents.schemas import (
    AcademicCalendar,
    CollegeMatch,
    IntentClassification,
    NoteSummary,
    ParsedInfo,
    ScheduleBlock,
    ScheduleProposal,
    SUBJECT_ALIASES,
    SyllabusExtraction,
    SyllabusLink,
    normalise_subject,
)


def _schema_of(model):
    return model.model_json_schema()


# ── The Gemini ``response_schema`` constraint ───────────────────────────────


@pytest.mark.parametrize(
    "model",
    [
        AcademicCalendar,
        CollegeMatch,
        IntentClassification,
        NoteSummary,
        ParsedInfo,
        ScheduleProposal,
        SyllabusExtraction,
        SyllabusLink,
    ],
)
def test_schemas_have_no_any_of(model):
    """Gemini's response_schema rejects anyOf, so schemas must avoid Optional."""
    assert "anyOf" not in json.dumps(_schema_of(model))


@pytest.mark.parametrize(
    "model",
    [
        AcademicCalendar,
        CollegeMatch,
        IntentClassification,
        NoteSummary,
        ParsedInfo,
        ScheduleProposal,
        SyllabusExtraction,
        SyllabusLink,
    ],
)
def test_schemas_accept_an_empty_object(model):
    """Every field defaults, so a minimal response still validates."""
    model.model_validate({})


def test_nested_models_are_defined():
    """ScheduleProposal nests a list of models; the schema must declare them."""
    schema = _schema_of(ScheduleProposal)
    assert "$defs" in schema or "ScheduleBlock" in json.dumps(schema)
    assert "blocks" in schema["properties"]


# ── ParsedInfo projections ──────────────────────────────────────────────────


def test_parsed_info_academic_event_projection():
    parsed = ParsedInfo(
        type="academic_event",
        title="CIE DSA",
        subject="DSA",
        due_date="2026-10-20",
    )
    data = parsed.to_data()
    assert data == {
        "title": "CIE DSA",
        "type": "cie",
        "subject": "DSA",
        "due_date": "2026-10-20",
        "syllabus_units": [],
    }
    assert parsed.has_data


def test_parsed_info_timetable_projection():
    parsed = ParsedInfo.model_validate(
        {
            "type": "timetable",
            "Monday": [{"time": "09:00", "subject": "DSA", "room": "CS-201"}],
            "Tuesday": [{"time": "10:00", "subject": "DBMS"}],
        }
    )
    data = parsed.timetable_dict()
    assert data == {
        "Monday": [
            {"time": "09:00", "subject": "DSA", "room": "CS-201", "professor": ""}
        ],
        "Tuesday": [
            {"time": "10:00", "subject": "DBMS", "room": "", "professor": ""}
        ],
    }


def test_parsed_info_timetable_omits_empty_days():
    assert ParsedInfo.model_validate({"type": "timetable"}).timetable_dict() == {}


def test_parsed_info_type_defaults_to_cie_when_unset():
    parsed = ParsedInfo(type="academic_event", title="Exam")
    assert parsed.to_data()["type"] == "cie"


def test_parsed_info_general_has_no_data():
    assert not ParsedInfo(type="general").has_data


def test_parsed_info_accepts_extra_keys():
    """Gemini sometimes adds fields; they must not fail validation."""
    ParsedInfo.model_validate({"type": "task", "surprise": 42})


def test_parsed_info_rejects_wrong_types():
    with pytest.raises(ValidationError):
        ParsedInfo(type="academic_event", due_date=["not", "a", "string"])


# ── Subject normalisation ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("dsa", "DSA"),
        ("Data Structures", "DSA"),
        ("dbms", "DBMS"),
        ("cn", "Computer Networks"),
        ("computer networks", "Computer Networks"),
        ("", ""),
        ("Something Unknown", "Something Unknown"),
    ],
)
def test_normalise_subject(raw, expected):
    assert normalise_subject(raw) == expected


def test_subject_aliases_values_are_canonical_names():
    # Canonical names must not themselves be alias keys, or the mapping drifts.
    for canonical in set(SUBJECT_ALIASES.values()):
        assert canonical not in SUBJECT_ALIASES


# ── Remaining schemas ───────────────────────────────────────────────────────


def test_college_match_strips_quotes_and_case():
    match = CollegeMatch.model_validate({"college_key": ' "RVCE" '})
    assert match.college_key == "rvce"


def test_college_match_normalises_none():
    assert CollegeMatch.model_validate({"college_key": "None"}).college_key == ""


def test_intent_classification_defaults():
    intent = IntentClassification.model_validate({})
    assert intent.intent == "general"
    assert intent.scope == "day"
    assert intent.date == ""


def test_note_summary_truncates():
    note = NoteSummary(summary="x" * 300, subject="DSA", tags=["a", "b"])
    assert len(note.summary) == 300
    assert note.tags == ["a", "b"]


def test_schedule_block_defaults():
    block = ScheduleBlock.model_validate({"time": "09:00", "title": "DSA"})
    assert block.duration_min == 60
    assert block.type == "work"
    assert block.notes == ""


def test_schedule_proposal_requires_blocks_list():
    proposal = ScheduleProposal.model_validate(
        {"blocks": [{"time": "09:00", "title": "DSA", "duration_min": 90}]}
    )
    assert len(proposal.blocks) == 1
    assert proposal.blocks[0].model_dump()["duration_min"] == 90


def test_syllabus_extraction_nested_units():
    extraction = SyllabusExtraction.model_validate(
        {
            "subjects": [
                {
                    "subject_code": "21CS51",
                    "subject": "DSA",
                    "credits": 4,
                    "units": [{"number": 1, "title": "Intro", "topics": ["a", "b"]}],
                }
            ]
        }
    )
    unit = extraction.subjects[0].units[0]
    assert unit.status == "not_started"
    assert unit.completion_percent == 0
    assert unit.topics == ["a", "b"]


def test_syllabus_link_matched_flag():
    assert not SyllabusLink.model_validate({}).matched
    assert SyllabusLink(subject="DSA", unit_number=1).matched


def test_academic_calendar_is_empty():
    calendar = AcademicCalendar.model_validate({})
    assert calendar.is_empty
    assert calendar.cie_dates == []
    assert calendar.holidays == []


def test_academic_calendar_not_empty_with_one_field():
    assert not AcademicCalendar(semester_start="2026-08-01").is_empty
