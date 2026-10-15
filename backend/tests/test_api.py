"""API-level tests for the chat and upload routers.

Two things are proven here:

1. **Auth is enforced.** With Firebase configured, a request without a token is
   a 401, and a body ``user_id`` that disagrees with the token is a 403.
2. **The SSE contract holds.** The chat stream emits ``text`` chunks, a
   ``proposed_schedule`` payload when a schedule was built, and terminates with
   ``[DONE]``.
"""

import json


from app.main import app


def prod_auth(monkeypatch):
    """Simulate production: Firebase is configured, tokens are verified."""
    monkeypatch.setattr("app.api.auth.firebase_initialized", True)
    monkeypatch.setattr(
        "app.api.auth.auth.verify_id_token", lambda token: {"uid": "real_uid"}
    )


def _sse_events(text):
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[6:]))
    return events


# ── Health / smoke ──────────────────────────────────────────────────────────


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Auth enforcement ────────────────────────────────────────────────────────


def test_chat_requires_a_token_in_production(client, prod_auth):
    resp = client.post("/api/chat", json={"user_id": "real_uid", "message": "hi"})
    assert resp.status_code == 401


def test_chat_rejects_a_forged_user_id_in_production(client, prod_auth):
    resp = client.post(
        "/api/chat",
        json={"user_id": "attacker_uid", "message": "hi"},
        headers={"Authorization": "Bearer valid-token"},
    )
    assert resp.status_code == 403


def test_chat_accepts_a_matching_user_id_in_production(client, prod_auth):
    resp = client.post(
        "/api/chat",
        json={"user_id": "real_uid", "message": "hey", "thread_id": "prod-thread"},
        headers={"Authorization": "Bearer valid-token"},
    )
    assert resp.status_code == 200


def test_approve_requires_a_token_in_production(client, prod_auth):
    resp = client.post(
        "/api/schedule/approve", json={"thread_id": "t", "feedback": {"approved": True}}
    )
    assert resp.status_code == 401


def test_summarize_rejects_a_forged_user_id_in_production(client, prod_auth):
    resp = client.post(
        "/api/notes/summarize",
        json={"note_content": "unit 2 is heavy", "user_id": "attacker_uid"},
        headers={"Authorization": "Bearer valid-token"},
    )
    assert resp.status_code == 403


def test_upload_timetable_requires_a_token_in_production(client, prod_auth):
    resp = client.post(
        "/api/upload/timetable",
        data={"user_id": "real_uid"},
        files={"file": ("t.jpg", b"x", "image/jpeg")},
    )
    assert resp.status_code == 401


def test_confirm_syllabus_rejects_a_forged_user_id_in_production(client, prod_auth):
    resp = client.post(
        "/api/upload/confirm/syllabus",
        json={"user_id": "attacker_uid", "subject": "DSA", "syllabus": {}},
        headers={"Authorization": "Bearer valid-token"},
    )
    assert resp.status_code == 403


# ── Dev fallback (no Firebase configured) ───────────────────────────────────


def test_chat_works_in_dev_mode_without_a_token(client):
    resp = client.post(
        "/api/chat",
        json={"user_id": "demo_anonymous", "message": "hey", "thread_id": "dev-thread"},
    )
    assert resp.status_code == 200
    assert "data: [DONE]" in resp.text


def test_chat_uses_the_default_uid_when_none_supplied(client):

    resp = client.post(
        "/api/chat",
        json={"message": "hey", "thread_id": "dev-thread-2"},
    )
    assert resp.status_code == 200


def test_confirm_timetable_works_in_dev_mode(client):
    resp = client.post(
        "/api/upload/confirm/timetable",
        json={"user_id": "demo_anonymous", "timetable": {"Monday": []}},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_summarize_works_in_dev_mode(client):
    resp = client.post(
        "/api/notes/summarize",
        json={"note_content": "graphs have heavy weightage in DSA unit 2", "user_id": "demo_anonymous"},
    )
    assert resp.status_code == 200
    body = resp.json()
    # The LLM is faked to fail, so the deterministic fallback is returned.
    assert body["subject"] == "General"
    assert body["summary"]


# ── SSE contract ────────────────────────────────────────────────────────────


def test_chat_stream_emits_text_and_done(client):
    resp = client.post(
        "/api/chat",
        json={"user_id": "demo_anonymous", "message": "hey how are you", "thread_id": "sse-1"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(resp.text)
    assert any(e["type"] == "text" for e in events)
    assert resp.text.rstrip().endswith("data: [DONE]")


def test_schedule_build_stream_emits_the_proposal(client):
    resp = client.post(
        "/api/chat",
        json={"user_id": "demo_anonymous", "message": "schedule my day", "thread_id": "sse-2"},
    )
    assert resp.status_code == 200
    events = _sse_events(resp.text)
    proposals = [e for e in events if e["type"] == "proposed_schedule"]
    assert len(proposals) == 1
    assert proposals[0]["schedule"], "the proposed schedule must not be empty"
    assert proposals[0]["schedule"][-1]["title"] == "Sleep"


def test_no_approved_schedule_is_emitted_before_approval(client):
    resp = client.post(
        "/api/chat",
        json={"user_id": "demo_anonymous", "message": "schedule my day", "thread_id": "sse-3"},
    )
    events = _sse_events(resp.text)
    assert not [e for e in events if e["type"] == "approved_schedule"]


def test_second_message_does_not_re_emit_the_stale_proposal(client):
    """The previous turn's proposal must not be replayed on a new message."""
    first = client.post(
        "/api/chat",
        json={"user_id": "demo_anonymous", "message": "schedule my day", "thread_id": "sse-4"},
    )
    assert [e for e in _sse_events(first.text) if e["type"] == "proposed_schedule"]

    second = client.post(
        "/api/chat",
        json={"user_id": "demo_anonymous", "message": "hey there", "thread_id": "sse-4"},
    )
    assert not [e for e in _sse_events(second.text) if e["type"] == "proposed_schedule"]


async def test_threads_are_namespaced_per_user(client):
    """Two users must not share the constant client thread id."""
    for user in ("alice", "bob"):
        client.post(
            "/api/chat",
            json={"user_id": user, "message": "first message", "thread_id": "shared"},
        )

    graph = app.state.graph
    alice = await graph.aget_state({"configurable": {"thread_id": "alice:shared"}})
    bob = await graph.aget_state({"configurable": {"thread_id": "bob:shared"}})

    # Assistant replies may be appended, but only this user's message is present.
    assert [m.content for m in alice.values["messages"] if m.type == "human"] == ["first message"]
    assert [m.content for m in bob.values["messages"] if m.type == "human"] == ["first message"]
    assert alice.values["user_id"] == "alice"
    assert bob.values["user_id"] == "bob"

    # The un-namespaced thread the client actually sent must not exist.
    shared = await graph.aget_state({"configurable": {"thread_id": "shared"}})
    assert not shared.values


# ── Approve endpoint ────────────────────────────────────────────────────────


def test_approve_requires_a_thread_id(client):
    resp = client.post("/api/schedule/approve", json={"feedback": {"approved": True}})
    assert resp.json()["status"] == "error"


def test_approve_on_a_fresh_thread_does_not_crash(client):
    resp = client.post(
        "/api/schedule/approve",
        json={"thread_id": "never-used", "user_id": "demo_anonymous", "feedback": {"approved": True}},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] in ("success", "error")
    assert "approved_schedule" in resp.json()


def test_full_approve_flow_commits_a_schedule(client):
    chat = client.post(
        "/api/chat",
        json={"user_id": "demo_anonymous", "message": "schedule my day", "thread_id": "flow-1"},
    )
    assert [e for e in _sse_events(chat.text) if e["type"] == "proposed_schedule"]

    approved = client.post(
        "/api/schedule/approve",
        json={
            "thread_id": "flow-1",
            "user_id": "demo_anonymous",
            "feedback": {"approved": True, "edits": []},
        },
    )
    assert approved.json()["status"] == "success"
    assert approved.json()["approved_schedule"]
