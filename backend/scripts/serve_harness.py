"""Run the FastAPI app with the test harness, for client-side verification.

Lifespan needs a real Postgres, so the state is built by hand and uvicorn is
pointed at the already-imported app object.

    python scripts/serve_harness.py 8000
"""

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

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

from app.agents.graph import create_graph  # noqa: E402
from app.main import app  # noqa: E402
from tests.fakes import install_fake_firestore, install_fake_llm  # noqa: E402


class _Patch:
    def setattr(self, target, name, value=..., raising=True):
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
    name = getattr(schema, "__name__", "")
    if name == "ScheduleProposal":
        return {
            "blocks": [
                {"time": "09:00", "title": "Data Structures", "type": "study",
                 "duration_min": 90, "notes": "Graphs and trees"},
                {"time": "11:00", "title": "Database Management Systems", "type": "study",
                 "duration_min": 90, "notes": "Normalisation"},
                {"time": "18:00", "title": "Gym", "type": "health",
                 "duration_min": 60, "notes": ""},
                {"time": "23:00", "title": "Sleep", "type": "rest",
                 "duration_min": 480, "notes": "Target sleep time"},
            ]
        }
    if name == "IntentClassification":
        return {"intent": "schedule_build", "scope": "day", "date": ""}
    return ""


def build_state():
    patch = _Patch()
    install_fake_firestore(patch)
    install_fake_llm(patch, _responder)
    app.state.store = InMemoryStore()
    app.state.graph = create_graph(store=app.state.store, checkpointer=MemorySaver())


build_state()

if __name__ == "__main__":
    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
