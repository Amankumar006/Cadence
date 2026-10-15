"""End-to-end check of the chat contract the way the client consumes it.

Simulates the AI SDK's frame handling over a real /api/chat response: asserts
the protocol header, reassembles the text from text-delta frames, and pulls the
schedule out of the data-schedule part. This is the guard that would have caught
the body-shape mismatch (the SDK posts `messages[]`, the backend reads
`message`).
"""

import json


def _sse_payloads(text):
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            body = block[len("data: "):]
            out.append("[DONE]" if body == "[DONE]" else json.loads(body))
    return out


def _reassemble(payloads):
    """Mirror the SDK's reduction of frames into one assistant message."""
    text = ""
    data_parts = {}
    for p in payloads:
        if p == "[DONE]":
            continue
        if p["type"] == "text-delta":
            text += p["delta"]
        elif p["type"].startswith("data-"):
            data_parts[p["type"][len("data-"):]] = p["data"]
    return text, data_parts


def test_protocol_header_is_present(client):
    resp = client.post(
        "/api/chat",
        json={"user_id": "demo_anonymous", "message": "hello", "thread_id": "e2e-1"},
    )
    assert resp.headers["x-vercel-ai-ui-message-stream"] == "v1"


def test_text_frames_reassemble_into_a_message(client):
    resp = client.post(
        "/api/chat",
        json={"user_id": "demo_anonymous", "message": "hello there", "thread_id": "e2e-2"},
    )
    payloads = _sse_payloads(resp.text)

    assert payloads[0]["type"] == "start"
    assert payloads[-1] == "[DONE]"

    text, data_parts = _reassemble(payloads)
    assert text, "the client would render an empty bubble"
    # No schedule part for a plain chat turn.
    assert "schedule" not in data_parts


def test_schedule_part_carries_a_parsable_payload(client):
    resp = client.post(
        "/api/chat",
        json={"user_id": "demo_anonymous", "message": "schedule my day", "thread_id": "e2e-3"},
    )
    text, data_parts = _reassemble(_sse_payloads(resp.text))

    assert "schedule" in data_parts, "the client's proposal UI would never open"
    part = data_parts["schedule"]
    assert part["status"] == "proposed"
    assert part["schedule"], "the proposed schedule must not be empty"
    assert part["schedule"][-1]["title"] == "Sleep"

    # Every block must have the fields the mobile schema validates.
    for block in part["schedule"]:
        assert isinstance(block["time"], str) and block["time"]
        assert isinstance(block["title"], str) and block["title"]
        assert isinstance(block["type"], str)
        assert isinstance(block["duration_min"], int)
        assert isinstance(block["notes"], str)


def test_a_second_turn_does_not_replay_the_schedule(client):
    """The client keys its proposal UI off the data part, so a replay would
    re-open the edit modal on an unrelated message."""
    first = client.post(
        "/api/chat",
        json={"user_id": "demo_anonymous", "message": "schedule my day", "thread_id": "e2e-4"},
    )
    assert "schedule" in _reassemble(_sse_payloads(first.text))[1]

    second = client.post(
        "/api/chat",
        json={"user_id": "demo_anonymous", "message": "thanks!", "thread_id": "e2e-4"},
    )
    _, data_parts = _reassemble(_sse_payloads(second.text))
    assert "schedule" not in data_parts


def test_approval_commits_and_the_next_turn_stays_clean(client):
    chat = client.post(
        "/api/chat",
        json={"user_id": "demo_anonymous", "message": "schedule my day", "thread_id": "e2e-5"},
    )
    _, data_parts = _reassemble(_sse_payloads(chat.text))
    assert data_parts["schedule"]["status"] == "proposed"

    approved = client.post(
        "/api/schedule/approve",
        json={
            "thread_id": "e2e-5",
            "user_id": "demo_anonymous",
            "feedback": {"approved": True, "edits": []},
        },
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "success"
    assert approved.json()["approved_schedule"]
