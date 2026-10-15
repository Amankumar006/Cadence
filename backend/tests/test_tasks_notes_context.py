"""The repo's original "tasks & notes context" scenario, as a real test.

Previously this was a manual script inside ``app/`` with a hardcoded macOS
``sys.path`` and a dependency on a live server, Postgres and Firestore. The
same scenario now runs against the real graph with in-memory fakes:

1. "Buy lab manual due tomorrow"  -> a task is persisted
2. "DSA Unit 4 has 40% CIE weightage" -> a note is persisted and summarised
3. "schedule my day for <tomorrow>" -> the proposed schedule contains both a
   task block for the lab manual and a DSA study block
"""

import datetime
import json


def _events(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            out.append(json.loads(line[6:]))
    return out


def _chat(client, message, thread_id):
    resp = client.post(
        "/api/chat",
        json={"user_id": "test_tasks_notes", "message": message, "thread_id": thread_id},
    )
    assert resp.status_code == 200
    return _events(resp.text)


def test_tasks_and_notes_shape_the_schedule(client):
    thread = "tasks_notes_thread"
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()

    _chat(client, "Buy lab manual due tomorrow", thread)
    _chat(client, "DSA Unit 4 has 40% CIE weightage", thread)

    events = _chat(client, f"schedule my day for {tomorrow}", thread)
    proposals = [e for e in events if e["type"] == "proposed_schedule"]
    assert len(proposals) == 1, f"expected one proposal, got {[e['type'] for e in events]}"

    schedule = proposals[0]["schedule"]
    assert schedule, "the proposed schedule must not be empty"

    titles = " ".join(
        f"{b.get('title', '')} {b.get('notes', '')}".lower() for b in schedule
    )
    assert "lab manual" in titles, "the pending task was not scheduled"
    assert "dsa" in titles, "the DSA note did not produce a study block"
    assert schedule[-1]["title"] == "Sleep"


def test_task_and_note_are_persisted_to_firestore(client, firestore):
    thread = "persist_thread"
    _chat(client, "Buy lab manual due tomorrow", thread)
    _chat(client, "DSA Unit 4 has 40% CIE weightage", thread)

    tasks = firestore.db.docs("tasks", "test_tasks_notes", "items")
    assert tasks, "no task was written to Firestore"
    assert any("lab manual" in (t.get("title", "").lower()) for t in tasks)

    notes = firestore.db.docs("notes", "test_tasks_notes", "items")
    assert notes, "no note was written to Firestore"
    assert any("dsa" in (n.get("body", "").lower()) for n in notes)
