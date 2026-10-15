"""Tests for app.db: the Neon pooled/direct connection split.

The important behaviour is which DSN gets used for what. Neon hands back two
connection strings for the same database and they are not interchangeable --
running DDL over the pooler fails in ways that never name pooling. These tests
pin the routing so that mistake cannot come back.

Live-database tests are skipped unless DATABASE_URL is set and reachable, so a
clone with no database still passes the suite.
"""

import os
import socket
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
    monkeypatch.setattr(settings, "DATABASE_URL", POOLED)
    monkeypatch.setattr(settings, "DATABASE_URL_UNPOOLED", DIRECT)


@pytest.fixture
def local_env(monkeypatch):
    """A plain local Postgres: one DSN, no pooler."""
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql://postgres:pw@localhost:5432/appdb")
    monkeypatch.setattr(settings, "DATABASE_URL_UNPOOLED", "")


# ── DSN routing ─────────────────────────────────────────────────────────────


def test_neon_pooled_dsn_is_preferred_for_app_traffic(neon_env):
    assert settings.pooled_dsn() == POOLED


def test_neon_direct_dsn_is_preferred_for_ddl(neon_env):
    assert settings.direct_dsn() == DIRECT


def test_without_neon_the_dsn_falls_back_to_db_uri(monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "")
    monkeypatch.setattr(settings, "DATABASE_URL_UNPOOLED", "")
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
    monkeypatch.setattr(
        settings,
        "DATABASE_URL_UNPOOLED",
        "postgresql://user:pw@ep-cool.us-east-2.aws.neon.tech/somewhere_else",
    )
    assert db.describe()["same_database"] is False


def test_describe_handles_a_unix_socket_dsn(monkeypatch):
    """A local Postgres hands back a socket URI with no hostname at all."""
    monkeypatch.setattr(
        settings, "DATABASE_URL", "postgresql://postgres:@/postgres?host=/var/run/postgresql"
    )
    monkeypatch.setattr(settings, "DATABASE_URL_UNPOOLED", "")
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

    monkeypatch.setattr(settings, "DATABASE_URL_UNPOOLED", "")

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
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql://nobody@127.0.0.1:1/nodb")
    ok, detail = await db.check_connection()
    assert ok is False
    assert detail  # a human-readable reason, not an empty string


@pytest.mark.asyncio
async def test_check_connection_names_the_error_type(monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql://nobody@127.0.0.1:1/nodb")
    ok, detail = await db.check_connection()
    assert ":" in detail, detail


# ── Live-database tests: skipped unless a reachable DATABASE_URL exists ─────


def _live_dsn():
    """A DSN the caller explicitly exported, or None.

    Deliberately reads os.environ rather than settings: a developer with a real
    Neon .env would otherwise have the whole suite pointed at their database, and
    the live tests would run (and fail) on a machine that never asked for them.
    Exporting DATABASE_URL in the shell is the opt-in.
    """
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
    monkeypatch.setattr(settings, "DATABASE_URL", live_db)
    monkeypatch.setattr(settings, "DATABASE_URL_UNPOOLED", live_db)

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
    monkeypatch.setattr(settings, "DATABASE_URL", live_db)
    monkeypatch.setattr(settings, "DATABASE_URL_UNPOOLED", live_db)
    await db.run_migrations()
    await db.run_migrations()


# ── IPv4 fallback ───────────────────────────────────────────────────────────


def test_with_hostaddr_keeps_the_host_for_sni():
    """`host` is the SNI and certificate name; only the dialled address changes.

    Neon routes on the TLS SNI hostname, so replacing `host` with an IP makes the
    proxy close the connection. `hostaddr` pins the address while leaving `host`
    alone.
    """
    dsn = "postgresql://u:p@ep-x-pooler.c-6.aws.neon.tech/db?sslmode=require"
    out = db._with_hostaddr(dsn, "52.14.39.200")
    assert "hostaddr=52.14.39.200" in out
    assert "ep-x-pooler.c-6.aws.neon.tech" in out, "the SNI hostname must survive"
    assert "sslmode=require" in out, "existing query params must survive"


def test_with_hostaddr_preserves_the_database_and_credentials():
    dsn = "postgresql://u:p@host/dbname?sslmode=require"
    out = db._with_hostaddr(dsn, "10.0.0.1")
    parsed = urlparse(out)
    assert parsed.username == "u"
    assert parsed.password == "p"
    assert parsed.path == "/dbname"


def test_with_hostaddr_overwrites_an_existing_hostaddr():
    dsn = "postgresql://u:p@host/db?hostaddr=1.2.3.4"
    out = db._with_hostaddr(dsn, "10.0.0.1")
    assert "hostaddr=10.0.0.1" in out
    assert "1.2.3.4" not in out


def test_ipv4_for_returns_an_a_record(monkeypatch):
    def fake_getaddrinfo(host, port, family, proto):
        assert family == socket.AF_INET, "must ask for A records only"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("52.14.39.200", 5432))]

    monkeypatch.setattr(db.socket, "getaddrinfo", fake_getaddrinfo)
    assert db._ipv4_for("ep-x-pooler.c-6.aws.neon.tech") == "52.14.39.200"


def test_ipv4_for_returns_none_when_there_is_no_a_record(monkeypatch):
    def boom(*a, **kw):
        raise OSError("no A record")

    monkeypatch.setattr(db.socket, "getaddrinfo", boom)
    assert db._ipv4_for("ipv6-only.example.com") is None


@pytest.mark.asyncio
async def test_check_connection_retries_ipv4_on_a_network_error(monkeypatch):
    """libpq prefers AAAA and never falls back, so a routing error is retried."""
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql://u:p@dual.example.com/db")
    monkeypatch.setattr(db, "_ipv4_for", lambda host: "52.14.39.200")

    tried = []

    class _Conn:
        async def execute(self, sql):
            class _C:
                async def fetchone(self):
                    return {"ok": 1}

            return _C()

        async def close(self):
            pass

    async def fake_connect(dsn, **kw):
        tried.append(dsn)
        if "hostaddr" not in dsn:
            raise OSError("Network is unreachable")
        return _Conn()

    monkeypatch.setattr(db.AsyncConnection, "connect", fake_connect)
    ok, detail = await db.check_connection()
    assert ok is True, detail
    assert len(tried) == 2, f"expected an IPv4 retry, got {tried}"
    assert "hostaddr" in tried[1]


@pytest.mark.asyncio
async def test_check_connection_does_not_retry_a_non_network_error(monkeypatch):
    """An auth failure is not a routing problem; retrying just doubles the wait."""
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql://u:p@dual.example.com/db")
    monkeypatch.setattr(db, "_ipv4_for", lambda host: "52.14.39.200")

    tried = []

    async def fake_connect(dsn, **kw):
        tried.append(dsn)
        raise PermissionError("password authentication failed")

    monkeypatch.setattr(db.AsyncConnection, "connect", fake_connect)
    ok, detail = await db.check_connection()
    assert ok is False
    assert len(tried) == 1, "an auth failure must not trigger an IPv4 retry"
    assert "authentication" in detail


@pytest.mark.asyncio
async def test_check_connection_skips_the_retry_when_there_is_no_a_record(monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql://u:p@ipv6-only.example.com/db")
    monkeypatch.setattr(db, "_ipv4_for", lambda host: None)

    tried = []

    async def fake_connect(dsn, **kw):
        tried.append(dsn)
        raise OSError("Network is unreachable")

    monkeypatch.setattr(db.AsyncConnection, "connect", fake_connect)
    ok, detail = await db.check_connection()
    assert ok is False
    assert len(tried) == 1


def test_network_errors_are_the_retryable_kind():
    assert "network is unreachable" in db._NETWORK_ERRORS
    assert "connection refused" in db._NETWORK_ERRORS
    assert "password authentication failed" not in db._NETWORK_ERRORS
