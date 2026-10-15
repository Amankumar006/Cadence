"""Tests for the AI SDK UI-message-stream frame writer.

These assert the exact wire shapes the Vercel AI SDK client expects. A drift
here shows up on the phone as a chat screen that never renders, so the frame
names and ids are pinned rather than inferred.
"""

import json

from app.api.sse import PROTOCOL_HEADER, UIMessageStream


def _frames(payload: str):
    """Parse an SSE payload into a list (``[DONE]`` -> the string ``"[DONE]"``)."""
    out = []
    for block in payload.split("\n\n"):
        block = block.strip()
        if not block.startswith("data: "):
            continue
        body = block[len("data: "):]
        out.append("[DONE]" if body == "[DONE]" else json.loads(body))
    return out


def _by_type(payload: str, frame_type: str):
    return [f for f in _frames(payload) if isinstance(f, dict) and f["type"] == frame_type]


def _text_of(payload: str) -> str:
    return "".join(f["delta"] for f in _by_type(payload, "text-delta"))


# ── lifecycle ───────────────────────────────────────────────────────────────


def test_start_frame_is_first_and_carries_a_message_id():
    stream = UIMessageStream(message_id="msg-1")
    assert _frames(stream.start()) == [{"type": "start", "messageId": "msg-1"}]


def test_start_is_emitted_only_once():
    stream = UIMessageStream()
    payload = stream.start() + stream.text("a") + stream.text("b")
    assert len(_by_type(payload, "start")) == 1


def test_done_terminates_the_stream():
    stream = UIMessageStream()
    assert _frames(stream.done()) == ["[DONE]"]


def test_done_closes_an_open_text_part():
    stream = UIMessageStream()
    frames = _frames(stream.start() + stream.text("hi") + stream.done())
    ends = [f for f in frames if isinstance(f, dict) and f["type"] == "text-end"]
    assert ends
    assert frames[-2] == {"type": "text-end", "id": ends[0]["id"]}
    assert frames[-1] == "[DONE]"


def test_a_message_id_is_generated_when_not_supplied():
    stream = UIMessageStream()
    starts = _by_type(stream.start(), "start")
    assert starts[0]["messageId"]
    assert starts[0]["messageId"] != ""


# ── text ────────────────────────────────────────────────────────────────────


def test_text_produces_a_shared_id_trio():
    stream = UIMessageStream()
    payload = stream.start() + stream.text("a") + stream.text("b") + stream.done()
    deltas = _by_type(payload, "text-delta")
    starts = _by_type(payload, "text-start")
    ends = _by_type(payload, "text-end")
    assert len(starts) == 1 and len(ends) == 1
    assert starts[0]["id"] == ends[0]["id"] == deltas[0]["id"] == deltas[1]["id"]
    assert [d["delta"] for d in deltas] == ["a", "b"]
    assert _text_of(payload) == "ab"


def test_empty_delta_emits_nothing():
    stream = UIMessageStream()
    assert stream.text("") == ""
    assert stream.start() != ""


def test_error_frame_uses_error_text():
    stream = UIMessageStream()
    payload = stream.start() + stream.error("boom")
    assert _by_type(payload, "error") == [{"type": "error", "errorText": "boom"}]


# ── data parts ──────────────────────────────────────────────────────────────


def test_data_part_uses_the_data_prefix():
    stream = UIMessageStream()
    payload = stream.start() + stream.data("schedule", {"status": "proposed"})
    parts = _by_type(payload, "data-schedule")
    assert parts == [{"type": "data-schedule", "data": {"status": "proposed"}}]


def test_data_part_closes_the_open_text_before_emitting():
    stream = UIMessageStream()
    payload = stream.start() + stream.text("hi") + stream.data("schedule", {"a": 1}) + stream.text(" more")
    frames = _frames(payload)
    last_text_end = max(i for i, f in enumerate(frames) if isinstance(f, dict) and f["type"] == "text-end")
    first_data = min(i for i, f in enumerate(frames) if isinstance(f, dict) and f["type"] == "data-schedule")
    assert first_data > last_text_end, "text must be closed before a data part"


def test_reopening_text_after_a_data_part_gets_a_fresh_id():
    """A new text part must not reuse the id of the closed one, or the client
    would append the continuation to the earlier part."""
    stream = UIMessageStream()
    payload = stream.start() + stream.text("a") + stream.data("x", {}) + stream.text("b")
    starts = _by_type(payload, "text-start")
    assert len(starts) == 2
    assert starts[0]["id"] != starts[1]["id"]
    assert _text_of(payload) == "ab"


def test_data_parts_do_not_leak_into_the_text():
    stream = UIMessageStream()
    payload = stream.start() + stream.text("hi") + stream.data("schedule", {"secret": 1})
    assert _text_of(payload) == "hi"


# ── headers ─────────────────────────────────────────────────────────────────


def test_headers_declare_the_protocol_version():
    headers = UIMessageStream.headers()
    assert headers[PROTOCOL_HEADER] == "v1"
    assert headers["Cache-Control"] == "no-cache"
    # Proxies must not buffer, or the client sees nothing until the run ends.
    assert headers["X-Accel-Buffering"] == "no"
