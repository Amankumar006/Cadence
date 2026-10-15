"""AI SDK v1 UI-message-stream frame writer.

The mobile client uses the Vercel AI SDK's ``useChat``, which speaks the
UI-message-stream protocol (see https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol).
This module turns the graph's raw events into those frames so the client needs
no bespoke SSE parsing.

Frame shapes used here:

    data: {"type":"start","messageId":"..."}
    data: {"type":"text-start","id":"..."}
    data: {"type":"text-delta","id":"...","delta":"..."}
    data: {"type":"text-end","id":"..."}
    data: {"type":"data-<name>","data":{...}}
    data: [DONE]

A custom backend must send the ``x-vercel-ai-ui-message-stream: v1`` response
header or the SDK will not recognise the stream.
"""

import json
import uuid

PROTOCOL_HEADER = "x-vercel-ai-ui-message-stream"
PROTOCOL_VERSION = "v1"


def _frame(obj: dict) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


class UIMessageStream:
    """Builds the SSE payload for one assistant message.

    Text is accumulated as a single text part (``text-start`` / ``text-delta`` /
    ``text-end`` sharing one ``id``). Custom data parts are emitted between text
    parts: emitting a data part closes any open text part, and the next text
    delta re-opens a new one, so the ordering the client sees is always valid.
    """

    def __init__(self, message_id: str | None = None):
        self.message_id = message_id or f"msg-{uuid.uuid4().hex}"
        self._text_id = f"txt-{uuid.uuid4().hex}"
        self._text_open = False
        self._started = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> str:
        if self._started:
            return ""
        self._started = True
        return _frame({"type": "start", "messageId": self.message_id})

    def done(self) -> str:
        return self.close_text() + "data: [DONE]\n\n"

    # ── text ─────────────────────────────────────────────────────────────────

    def _open_text(self) -> str:
        if self._text_open:
            return ""
        self._text_open = True
        # A fresh id per part: re-using one would make the client append the
        # continuation onto the already-closed part.
        self._text_id = f"txt-{uuid.uuid4().hex}"
        return _frame({"type": "text-start", "id": self._text_id})

    def text(self, delta: str) -> str:
        """Append ``delta`` to the assistant's text, opening a part if needed."""
        if not delta:
            return ""
        return self.start() + self._open_text() + _frame(
            {"type": "text-delta", "id": self._text_id, "delta": delta}
        )

    def close_text(self) -> str:
        if not self._text_open:
            return ""
        self._text_open = False
        return _frame({"type": "text-end", "id": self._text_id})

    # ── custom data parts ────────────────────────────────────────────────────

    def data(self, name: str, payload: dict) -> str:
        """Emit a ``data-<name>`` part the client reads via ``dataPartSchemas``."""
        return self.close_text() + self.start() + _frame(
            {"type": f"data-{name}", "data": payload}
        )

    # ── errors ───────────────────────────────────────────────────────────────

    def error(self, message: str) -> str:
        return self.close_text() + self.start() + _frame(
            {"type": "error", "errorText": message}
        )

    @staticmethod
    def headers() -> dict:
        return {
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            PROTOCOL_HEADER: PROTOCOL_VERSION,
        }
