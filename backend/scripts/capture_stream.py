"""Capture a real /api/chat SSE response to a file for the mobile verifier.

Runs the same in-memory harness the test suite uses (no Postgres, no network,
no credentials), so the sample is the exact byte stream the backend emits.

    python scripts/capture_stream.py ../mobile/.stream-samples/schedule.sse
"""

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "tests"))

os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("DEMO_MODE", "false")
os.environ.setdefault("LLM_MAX_ATTEMPTS", "1")
os.environ.setdefault("DEV_FALLBACK_USER_ID", "mock_user_123")

import firebase_admin  # noqa: E402

if not firebase_admin._apps:
    firebase_admin.initialize_app()

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from app.agents.graph import create_graph  # noqa: E402
from app.main import app  # noqa: E402
from tests.fakes import FakeFirestore, install_fake_firestore, install_fake_llm  # noqa: E402


class _Patch:
    """Minimal monkeypatch stand-in: the fakes only ever call setattr."""

    def setattr(self, target, name, value=..., raising=True):
        # The fakes pass either a module object or a "pkg.mod.attr" path.
        if isinstance(target, str):
            import importlib

            try:
                target = importlib.import_module(target)
            except ModuleNotFoundError:
                module_name, _, attr = target.rpartition(".")
                target = importlib.import_module(module_name)
                name = attr
        setattr(target, name, value)


def _responder(prompt, schema):
    """A proposal shaped like the real one, for whichever schema is asked."""
    name = getattr(schema, "__name__", "")
    if name == "ScheduleProposal":
        return {
            "blocks": [
                {"time": "07:00", "title": "Wake up and freshen up", "type": "personal",
                 "duration_min": 60, "notes": ""},
                {"time": "09:00", "title": "Data Structures", "type": "study",
                 "duration_min": 90, "notes": "Graphs and trees - CIE 1 on the 18th"},
                {"time": "11:00", "title": "Database Management Systems", "type": "study",
                 "duration_min": 90, "notes": "Normalisation"},
                {"time": "13:00", "title": "Lunch", "type": "rest",
                 "duration_min": 60, "notes": ""},
                {"time": "14:00", "title": "Computer Networks", "type": "study",
                 "duration_min": 90, "notes": "Transport layer"},
                {"time": "18:00", "title": "Gym", "type": "health",
                 "duration_min": 60, "notes": "Leg day"},
                {"time": "23:00", "title": "Sleep", "type": "rest",
                 "duration_min": 480, "notes": "Target sleep time"},
            ]
        }
    if name == "IntentClassification":
        return {"intent": "schedule_build", "scope": "day", "date": ""}
    return ""


def main() -> int:
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "../mobile/.stream-samples/schedule.sse")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    patch = _Patch()
    store = install_fake_firestore(patch)
    install_fake_llm(patch, _responder)

    app.state.store = InMemoryStore()
    app.state.graph = create_graph(store=app.state.store, checkpointer=MemorySaver())

    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={
            "user_id": "demo_anonymous",
            "message": "schedule my day",
            "thread_id": "capture-1",
        },
    )
    if resp.status_code != 200:
        print(f"backend returned HTTP {resp.status_code}: {resp.text[:400]}")
        return 1

    out_path.write_text(resp.text, encoding="utf-8")
    print(f"wrote {len(resp.text)} bytes to {out_path}")
    print(f"protocol header: {resp.headers.get('x-vercel-ai-ui-message-stream')!r}")
    print(f"firestore holder: {type(store).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
