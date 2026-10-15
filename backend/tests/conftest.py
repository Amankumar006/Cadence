"""Shared pytest fixtures.

Environment is configured *before* ``app.config`` is imported so the settings
singleton picks up deterministic values, and no real credentials are needed.
"""

import os
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("DEMO_MODE", "false")
os.environ.setdefault("LLM_MAX_ATTEMPTS", "1")
os.environ.setdefault("DEV_FALLBACK_USER_ID", "mock_user_123")

# firebase_admin's ``firestore_async.client()`` resolves the default app before
# doing anything, so an app must exist even though every call is faked.
import firebase_admin  # noqa: E402

if not firebase_admin._apps:
    firebase_admin.initialize_app()

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

from tests.fakes import install_fake_firestore, install_fake_llm  # noqa: E402


@pytest.fixture(autouse=True)
def _fake_firestore_always(monkeypatch):
    """Every test gets an in-memory Firestore, so no credentials are needed."""
    return install_fake_firestore(monkeypatch)


@pytest.fixture
def firestore(_fake_firestore_always):
    """The in-memory Firestore backing the patched ``client()``."""
    return _fake_firestore_always


@pytest.fixture
def fake_llm(monkeypatch):
    """Install the LLM fake; returns a factory taking a responder callable."""

    def _install(responder=None):
        return install_fake_llm(monkeypatch, responder)

    return _install


@pytest.fixture
def no_llm(monkeypatch):
    """Force every LLM call to fail, exercising deterministic fallbacks."""

    def _fail(prompt, schema):
        raise RuntimeError("LLM unavailable (test)")

    return install_fake_llm(monkeypatch, _fail)


@pytest.fixture
def demo_mode(monkeypatch):
    """Make ``bypass_llm`` return True for everyone."""
    monkeypatch.setattr("app.config.settings.DEMO_MODE", True)


# ── In-memory LangGraph store / checkpointer ────────────────────────────────


@pytest.fixture
def checkpointer():
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


@pytest.fixture
def store():
    from langgraph.store.memory import InMemoryStore

    return InMemoryStore()


@pytest.fixture
def graph(store, checkpointer, no_llm):
    """A compiled graph backed by in-memory store/checkpointer."""
    from app.agents.graph import create_graph

    return create_graph(store=store, checkpointer=checkpointer)


@pytest.fixture
def client(store, checkpointer, no_llm):
    """The real FastAPI app with in-memory graph/store (no Postgres lifespan)."""
    from app.agents.graph import create_graph

    app.state.store = store
    app.state.graph = create_graph(store=store, checkpointer=checkpointer)
    return TestClient(app)


@pytest.fixture
def prod_auth(monkeypatch):
    """Simulate production: Firebase is configured and tokens are verified."""
    monkeypatch.setattr("app.api.auth.firebase_initialized", True)
    monkeypatch.setattr(
        "app.api.auth.auth.verify_id_token", lambda token: {"uid": "real_uid"}
    )
