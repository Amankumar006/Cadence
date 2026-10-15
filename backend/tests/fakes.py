"""In-memory fakes for Firebase Firestore and the LLM chain.

The backend imports ``firebase_admin.firestore_async`` and calls
``firestore_async.client()`` at request time, so patching that one function is
enough to run the whole graph and API without credentials or a database.

Nothing here is a mock with assertions — it is a small, faithful store so the
real production code paths execute unchanged.
"""

import copy
import datetime
import re
from typing import Any, Dict, List, Optional

from firebase_admin import firestore_async


class _Sentinel:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"<sentinel {self.name}>"


class FakeSnapshot:
    """Mimics firebase_admin.firestore_async.DocumentSnapshot."""

    def __init__(self, ref: "FakeDocumentRef", data: Optional[dict]):
        self._ref = ref
        self._data = data
        self.reference = ref

    @property
    def id(self) -> str:
        return self._ref.id

    @property
    def exists(self) -> bool:
        return self._data is not None

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data) if self._data else {}


class FakeDocumentRef:
    def __init__(self, collection: "FakeCollection", doc_id: str):
        self._collection = collection
        self.id = doc_id

    # ── navigation ──
    def collection(self, name: str) -> "FakeCollection":
        path = (*self._collection.path, self.id, name)
        return FakeCollection(self._collection._db, path)

    @property
    def path(self) -> tuple:
        return (*self._collection.path, self.id)

    # ── reads ──
    async def get(self) -> FakeSnapshot:
        return FakeSnapshot(self, self._collection._db._read(self.path))

    # ── writes ──
    async def set(self, data: dict, merge: bool = False) -> None:
        data = self._resolve(data)
        if merge:
            existing = self._collection._db._read(self.path) or {}
            self._deep_merge(existing, data)
            self._collection._db._write(self.path, existing)
        else:
            self._collection._db._write(self.path, copy.deepcopy(data))

    async def update(self, data: dict) -> None:
        data = self._resolve(data)
        existing = self._collection._db._read(self.path)
        if existing is None:
            raise KeyError(f"No document to update at {'/'.join(self.path)}")
        existing.update(copy.deepcopy(data))
        self._collection._db._write(self.path, existing)

    async def delete(self) -> None:
        self._collection._db._write(self.path, None)

    def _resolve(self, data: dict) -> dict:
        """Replace Firestore sentinels with plain values."""
        resolved = {}
        for key, value in data.items():
            if value is firestore_async.SERVER_TIMESTAMP:
                value = datetime.datetime.now(datetime.timezone.utc).isoformat()
            resolved[key] = value
        return resolved

    @staticmethod
    def _deep_merge(base: dict, updates: dict) -> None:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                FakeDocumentRef._deep_merge(base[key], value)
            else:
                base[key] = value


class FakeCollection:
    def __init__(self, db: "FakeFirestore", path: tuple):
        self._db = db
        self.path = path

    def document(self, doc_id: Optional[str] = None) -> FakeDocumentRef:
        return FakeDocumentRef(self, doc_id or "auto")

    async def add(self, data: dict) -> FakeDocumentRef:
        doc_id = self._db._next_id(self.path)
        ref = self.document(doc_id)
        await ref.set(data)
        return ref

    async def get(self) -> List[FakeSnapshot]:
        prefix = self.path
        docs = self._db._children(prefix)
        return [FakeSnapshot(self.document(doc_id), data) for doc_id, data in docs]

    async def stream(self):
        for snapshot in await self.get():
            yield snapshot


class FakeFirestore:
    """Minimal async Firestore keyed on the document path tuple."""

    def __init__(self):
        self._data: Dict[tuple, dict] = {}
        self._counters: Dict[tuple, int] = {}

    def _next_id(self, path: tuple) -> str:
        self._counters[path] = self._counters.get(path, 0) + 1
        return f"doc_{self._counters[path]}"

    def _read(self, path: tuple) -> Optional[dict]:
        return self._data.get(path)

    def _write(self, path: tuple, data: Optional[dict]) -> None:
        if data is None:
            self._data.pop(path, None)
        else:
            self._data[path] = data

    def _children(self, path: tuple) -> List[tuple]:
        """Direct children documents of a collection path, sorted for stability."""
        out = []
        for key, value in self._data.items():
            if len(key) == len(path) + 1 and key[:-1] == path and value is not None:
                out.append((key[-1], value))
        return sorted(out, key=lambda item: item[0])

    def collection(self, name: str) -> FakeCollection:
        return FakeCollection(self, (name,))

    # ── test helpers ──
    def doc(self, *path: str) -> Optional[dict]:
        return copy.deepcopy(self._data.get(tuple(path)))

    def docs(self, *path: str) -> List[dict]:
        """All documents directly inside a collection, e.g. ``tasks/u1/items``."""
        return [copy.deepcopy(data) for _, data in self._children(tuple(path))]

    def clear(self) -> None:
        self._data.clear()


class FakeFirestoreHolder:
    """Holds the active fake and exposes ``client()``."""

    def __init__(self):
        self.db = FakeFirestore()

    def client(self) -> FakeFirestore:
        return self.db


def install_fake_firestore(monkeypatch) -> FakeFirestoreHolder:
    """Patch ``firestore_async.client`` for the duration of a test."""
    holder = FakeFirestoreHolder()
    monkeypatch.setattr(firestore_async, "client", holder.client)
    return holder


# ── LLM fakes ───────────────────────────────────────────────────────────────


class FakeStructuredResponse:
    """Returned by the fake chain's ``with_structured_output``."""

    def __init__(self, schema, responder):
        self._schema = schema
        self._responder = responder

    async def ainvoke(self, prompt, config=None, **kwargs):
        value = self._responder(prompt, self._schema)
        if isinstance(value, BaseException):
            raise value
        if value is None:
            return None
        if isinstance(value, dict):
            return self._schema.model_validate(value)
        return value


class FakeChatModel:
    """Duck-typed stand-in for a LangChain chat model."""

    def __init__(self, responder=None):
        self._responder = responder
        self.calls: List[str] = []
        self.structured_calls: List[Any] = []

    def with_fallbacks(self, others):
        return self

    def with_structured_output(self, schema, method=None, **kwargs):
        self.structured_calls.append((schema, method))
        return FakeStructuredResponse(schema, self._responder)

    async def ainvoke(self, prompt, config=None, **kwargs):
        self.calls.append(prompt)
        if self._responder is None:
            return _FakeMessage("FALLBACK TEXT")
        value = self._responder(prompt, None)
        if isinstance(value, BaseException):
            raise value
        return _FakeMessage(value if isinstance(value, str) else str(value))


class _FakeMessage:
    def __init__(self, content):
        self.content = content


def install_fake_llm(monkeypatch, responder=None) -> FakeChatModel:
    """Patch the LLM chain so no network call is made.

    ``responder(prompt, schema)`` returns a dict (validated against ``schema``)
    or a string (for free-text calls).
    """
    fake = FakeChatModel(responder)

    def fake_chain():
        return [fake]

    monkeypatch.setattr("app.agents.llm.get_llm_chain", fake_chain)
    monkeypatch.setattr("app.agents.llm._structured_cache", {})
    return fake


def normalise(value: str) -> str:
    """Collapse whitespace — handy for prompt assertions."""
    return re.sub(r"\s+", " ", value or "").strip()
