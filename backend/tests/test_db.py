"""Tests for app.db: the Neon pooled/direct connection split.

The important behaviour is which DSN gets used for what. Neon hands back two
connection strings for the same database and they are not interchangeable --
running DDL over the pooler fails in ways that never name pooling. These tests
pin the routing so that mistake cannot come back.

Live-database tests are skipped unless DATABASE_URL is set and reachable, so a
clone with no database still passes the suite.
"""

import os
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db  # noqa: E402
from app.config import settings  # noqa: E402

POOLED = "postgresql://user:pw@ep-cool-pooler.us-east-2.aws.neon.tech/appdb"
DIRECT = "postgresql://user:pw@ep-cool.us-east-2.aws.neon.tech/appdb"


@pytest.fixture
def neon_env(monkeypatch):
    """Simulate what `neon env pull` writes into the environment."""
    monkeypatch.setenv("DATABASE_URL", POOLED)
    monkeypatch.setenv("DATABASE_URL_UNPOOLED", DIRECT)


@pytest.fixture
def local_env(monkeypatch):
    """A plain local Postgres: one DSN, no pooler."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://postgres:pw@localhost:5432/appdb")
    monkeypatch.delenv("DATABASE_URL_UNPOOLED", raising=False)


# ── DSN routing ─────────────────────────────────────────────────────────────


def test_neon_pooled_dsn_is_preferred_for_app_traffic(neon_env):
    assert settings.pooled_dsn() == POOLED


def test_neon_direct_dsn_is_preferred_for_ddl(neon_env):
    assert settings.direct_dsn() == DIRECT


def test_without_neon_the_dsn_falls_back_to_db_uri(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL_UNPOOLED", raising=False)
    assert settings.pooled_dsn() == settings.DB_URI
    assert settings.direct_dsn() == settings.DB_URI


def test_local_postgres_uses_one_dsn_for_both(local_env):
    """A local Postgres has no pooler, so one DSN is correct for both."""
    assert settings.pooled_dsn() == settings.direct_dsn()


def test_neon_is_detected_by_the_pooler_host(neon_env):
    assert settings.using_neon is True


def test_local_postgres_is_not_detected_as_neon(local_env):
    assert settings.using_neon is False


# ── describe(): the credential-free summary /health returns ────────────────


def test_describe_reports_both_hosts(neon_env):
    info = db.describe()
    assert info["pooled_host"] == "ep-cool-pooler.us-east-2.aws.neon.tech"
    assert info["direct_host"] == "ep-cool.us-east-2.aws.neon.tech"
    assert info["pooled_is_pooler"] is True
    assert info["direct_is_pooler"] is False


def test_describe_never_leaks_the_password(neon_env):
    blob = repr(db.describe())
    assert "pw" not in blob, "describe() must not expose credentials"


def test_describe_flags_a_mismatched_database(neon_env, monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL_UNPOOLED",
        "postgresql://user:pw@ep-cool.us-east-2.aws.neon.tech/somewhere_else",
    )
    assert db.describe()["same_database"] is False


def test_describe_handles_a_unix_socket_dsn(monkeypatch):
    """A local Postgres hands back a socket URI with no hostname at all."""
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://postgres:@/postgres?host=/var/run/postgresql"
    )
    monkeypatch.delenv("DATABASE_URL_UNPOOLED", raising=False)
    info = db.describe()
    assert info["pooled_host"] == "socket:/var/run/postgresql"
    assert info["pooled_is_pooler"] is False


# ── Connection kwargs ───────────────────────────────────────────────────────


def test_connections_disable_client_side_prepared_statements():
    """PgBouncer runs in transaction mode and cannot keep them alive.

    Without this, Neon's pooler raises
    ``prepared statement "s0" already exists`` under concurrency.
    """
    assert db._conn_kwargs()["prepare_threshold"] == 0


def test_connections_use_autocommit_and_dict_rows():
    kwargs = db._conn_kwargs()
    assert kwargs["autocommit"] is True
    assert kwargs["row_factory"].__name__ == "dict_row"


# ── The warning that keeps DDL off the pooler ───────────────────────────────


def test_a_missing_unpooled_dsn_on_neon_logs_a_warning(neon_env, monkeypatch, caplog):
    import asyncio

    monkeypatch.delenv("DATABASE_URL_UNPOOLED", raising=False)

    # open_memory() will fail to connect, but the warning fires first.
    async def _run():
        try:
            async with db.open_memory():
                pass
        except Exception:
            pass

    with caplog.at_level("WARNING"):
        asyncio.run(_run())
    assert any("DATABASE_URL_UNPOOLED" in r.message for r in caplog.records)


def test_no_warning_when_the_unpooled_dsn_is_present(neon_env, caplog):
    import asyncio

    async def _run():
        try:
            async with db.open_memory():
                pass
        except Exception:
            pass

    with caplog.at_level("WARNING"):
        asyncio.run(_run())
    assert not any("DATABASE_URL_UNPOOLED" in r.message for r in caplog.records)


# ── check_connection(): the failure path must be clean, not a hang ──────────


@pytest.mark.asyncio
async def test_check_connection_reports_failure_without_raising(monkeypatch):
    """An unreachable database must return (False, detail), never hang."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/nodb")
    ok, detail = await db.check_connection()
    assert ok is False
    assert detail  # a human-readable reason, not an empty string


@pytest.mark.asyncio
async def test_check_connection_names_the_error_type(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/nodb")
    ok, detail = await db.check_connection()
    assert ":" in detail, detail


# ── Live-database tests: skipped unless a reachable DATABASE_URL exists ─────


def _live_dsn():
    """A DSN we can actually reach: TCP host or a Unix-socket directory."""
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        return None
    parsed = urlparse(dsn)
    if parsed.hostname:
        return dsn
    # A local Postgres hands back ``...?host=/var/run/postgresql``.
    if parse_qs(parsed.query).get("host"):
        return dsn
    return None


@pytest.fixture
def live_db():
    dsn = _live_dsn()
    if dsn is None:
        pytest.skip("no reachable DATABASE_URL; live Postgres tests skipped")
    return dsn


@pytest.mark.asyncio
async def test_live_round_trip_through_both_memory_stores(live_db, monkeypatch):
    """The end-to-end proof: open the pools the app serves from and use them."""
    monkeypatch.setenv("DATABASE_URL", live_db)
    monkeypatch.setenv("DATABASE_URL_UNPOOLED", live_db)

    ok, _ = await db.check_connection()
    assert ok, "the live database must be reachable for this test"

    await db.run_migrations()

    async with db.open_memory() as (store, checkpointer):
        ns = ("test_db", "smoke")
        await store.aput(ns, "k", {"v": 1})
        got = await store.aget(ns, "k")
        assert got is not None and got.value == {"v": 1}

        from langgraph.checkpoint.base import Checkpoint

        cfg = {"configurable": {"thread_id": "test-db-thread", "checkpoint_ns": ""}}
        ckpt = Checkpoint(v=1, id="test-db-cp", ts="2026-10-15T00:00:00Z",
                          channel_values={}, channel_versions={}, versions_seen={})
        await checkpointer.aput(cfg, ckpt, {}, {})
        tup = await checkpointer.aget_tuple(cfg)
        assert tup is not None and tup.checkpoint["id"] == "test-db-cp"


@pytest.mark.asyncio
async def test_live_migrations_are_idempotent(live_db, monkeypatch):
    """Running setup() twice must not fail -- it runs on every boot."""
    monkeypatch.setenv("DATABASE_URL", live_db)
    monkeypatch.setenv("DATABASE_URL_UNPOOLED", live_db)
    await db.run_migrations()
    await db.run_migrations()
