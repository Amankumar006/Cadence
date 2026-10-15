"""Postgres connection management for the LangGraph memory layer.

Neon hands back two connection strings for the same database and they are not
interchangeable. This module is the single place that decides which one to use,
so no caller has to remember:

    pooled (DATABASE_URL)          PgBouncer, transaction mode — app traffic
    direct (DATABASE_URL_UNPOOLED) DDL, migrations, session state

Why the split matters here specifically: ``store.setup()`` and
``checkpointer.setup()`` are DDL. Running them over Neon's pooled endpoint fails
in ways that never mention pooling — ``prepared statement "s0" already
exists``, a ``SET search_path`` that does not outlive its own transaction, or a
write landing in a read-only transaction the pooler inherited from an earlier
client. So migrations go direct, and only the live traffic goes pooled.

Driver notes, per the Neon agent skills:
  - The existing driver is preserved. This is a Python/FastAPI app served by
    uvicorn — a long-running process, not serverless — so the right driver is
    psycopg (v3) with a pool, not the HTTP-only serverless driver.
  - ``prepare_threshold=0`` is set on every connection. PgBouncer in transaction
    mode cannot keep a prepared statement alive between clients, so client-side
    prepared statements must be disabled. langgraph sets this itself; we set it
    explicitly on our own connections so the health check behaves the same way.
  - The checkpointer takes a single connection by default. It accepts an
    ``AsyncConnectionPool``, so we give it one rather than serialising every
    request through one socket.
"""

import logging
import socket
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import settings

logger = logging.getLogger("app.db")


def _conn_kwargs() -> Dict[str, Any]:
    """Connection kwargs shared by every pool we open.

    ``autocommit`` because LangGraph issues one statement at a time and manages
    its own transactions; ``prepare_threshold=0`` for PgBouncer (see module
    docstring); ``dict_row`` because every caller wants dicts, not tuples.
    """
    return {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
    }


def _pool(dsn: str, *, min_size: int, max_size: int) -> AsyncConnectionPool:
    return AsyncConnectionPool(
        dsn,
        min_size=min_size,
        max_size=max_size,
        kwargs=_conn_kwargs(),
        # Fail fast rather than hanging the whole lifespan when the database is
        # unreachable: uvicorn would sit at "waiting for application startup"
        # with no indication of why.
        timeout=10,
    )


@asynccontextmanager
async def open_memory() -> AsyncIterator[Tuple[AsyncPostgresStore, AsyncPostgresSaver]]:
    """Open the pooled store and checkpointer for the app's lifetime.

    Migrations run first, on the *direct* connection, then the live pools open
    on the *pooled* one. Both pools are closed on exit.
    """
    if settings.using_neon and not settings.DATABASE_URL_UNPOOLED:
        logger.warning(
            "Neon DSN detected but DATABASE_URL_UNPOOLED is unset; running DDL "
            "over the pooled endpoint, which can fail on Neon. Set it with "
            "`neon env pull`."
        )

    await run_migrations()

    logger.info(
        "Opening LangGraph memory pools (store min=%d max=%d)...",
        settings.DB_POOL_MIN_SIZE,
        settings.DB_POOL_MAX_SIZE,
    )

    store_pool = _pool(
        settings.pooled_dsn(),
        min_size=settings.DB_POOL_MIN_SIZE,
        max_size=settings.DB_POOL_MAX_SIZE,
    )
    saver_pool = _pool(
        settings.pooled_dsn(),
        min_size=1,
        # The checkpointer is hit once per graph step, so it needs less headroom
        # than the store but must never be zero.
        max_size=max(2, settings.DB_POOL_MAX_SIZE // 2),
    )

    async with store_pool, saver_pool:
        store = AsyncPostgresStore(conn=store_pool)
        checkpointer = AsyncPostgresSaver(conn=saver_pool)
        yield store, checkpointer


async def run_migrations() -> None:
    """Create the LangGraph tables if they do not exist, on the direct DSN.

    Idempotent — LangGraph's ``setup()`` uses ``CREATE TABLE IF NOT EXISTS`` —
    so this is safe to run on every boot.
    """
    dsn = settings.direct_dsn()
    logger.info("Running schema migrations on the %s connection...", _kind(dsn))

    # A one-off connection rather than a pool: this runs once at startup, and a
    # pool would sit idle for the rest of the process's life.
    conn = await AsyncConnection.connect(dsn, **_conn_kwargs())
    try:
        # NB: AsyncPostgresSaver is not itself an async context manager — only
        # its `from_conn_string` classmethod is, and that opens its own
        # connection. Constructing it directly and calling setup() lets us reuse
        # this one, and we close it in the finally below.
        await AsyncPostgresSaver(conn=conn).setup()
        async with AsyncPostgresStore.from_conn_string(dsn) as store:
            await store.setup()
    finally:
        await conn.close()


# Errors that mean "we never reached the server", as opposed to "the server
# rejected us". Only these are worth an IPv4 retry.
_NETWORK_ERRORS = ("network is unreachable", "no route to host", "connection refused")


def _ipv4_for(host: str) -> Optional[str]:
    """The first IPv4 address for ``host``, or None.

    A host can resolve to both A and AAAA records. libpq prefers the AAAA record
    and does not fall back when IPv6 has no route, so a machine without IPv6
    cannot reach a dual-stack host at all. Pinning an A record with ``hostaddr``
    fixes that without touching the DSN's ``host``, which stays the SNI name and
    the certificate name -- Neon routes on it.
    """
    try:
        infos = socket.getaddrinfo(host, 5432, socket.AF_INET, proto=socket.IPPROTO_TCP)
    except OSError:
        return None
    return infos[0][4][0] if infos else None


def _with_hostaddr(dsn: str, ipv4: str) -> str:
    """Add ``hostaddr`` to a DSN, leaving ``host`` (the SNI name) untouched."""
    parsed = urlparse(dsn)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["hostaddr"] = ipv4
    return urlunparse(parsed._replace(query=urlencode(query)))


async def check_connection() -> Tuple[bool, str]:
    """Verify the database answers, returning ``(ok, detail)``.

    Async, so it never blocks the event loop the way a sync ``psycopg.connect``
    inside an async endpoint does.

    If the first attempt fails with a routing error, it retries with an IPv4
    address pinned. That is not sandbox hackery: any host without an IPv6 route
    hits this, and libpq will not recover on its own. Reporting the *second*
    failure is also more honest -- otherwise an unreachable IPv6 address masks
    whatever is really wrong further down the stack.
    """
    dsn = settings.pooled_dsn()
    attempts = [dsn]

    host = urlparse(dsn).hostname
    ipv4 = _ipv4_for(host) if host else None
    if ipv4:
        attempts.append(_with_hostaddr(dsn, ipv4))

    last = ""
    for attempt in attempts:
        try:
            conn = await AsyncConnection.connect(
                attempt, connect_timeout=8, **_conn_kwargs()
            )
        except Exception as err:
            last = f"{type(err).__name__}: {err}"
            low = last.lower()
            if not any(e in low for e in _NETWORK_ERRORS):
                return False, last  # not a routing problem; retrying won't help
            continue
        try:
            cur = await conn.execute("SELECT 1 AS ok")
            row = await cur.fetchone()
            return True, f"SELECT 1 -> {row['ok'] if row else 'no row'}"
        except Exception as err:
            return False, f"{type(err).__name__}: {err}"
        finally:
            await conn.close()

    return False, last


def describe() -> Dict[str, Any]:
    """A safe, credential-free description of the current DB configuration.

    Used by ``/health`` and by the verification script. Never includes the
    password.
    """
    pooled, direct = settings.pooled_dsn(), settings.direct_dsn()
    return {
        "pooled_host": _host(pooled),
        "direct_host": _host(direct),
        "pooled_is_pooler": "-pooler." in pooled,
        "direct_is_pooler": "-pooler." in direct,
        "same_database": _dbname(pooled) == _dbname(direct),
        "database": _dbname(pooled),
        "pool_min": settings.DB_POOL_MIN_SIZE,
        "pool_max": settings.DB_POOL_MAX_SIZE,
    }


# ── helpers ──


def _kind(dsn: str) -> str:
    return "pooled" if "-pooler." in dsn else "direct"


def _host(dsn: str) -> str:
    """Hostname, or the Unix-socket directory for a socket DSN.

    A local Postgres (and pgserver) hands back ``...?host=/var/run/...``, which
    has no hostname at all. Reporting "?" there hid the real target, so the
    socket directory is reported instead.
    """
    try:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(dsn)
        if parsed.hostname:
            return parsed.hostname
        socket_dir = parse_qs(parsed.query).get("host")
        return f"socket:{socket_dir[0]}" if socket_dir else "?"
    except Exception:
        return "?"


def _dbname(dsn: str) -> str:
    try:
        from urllib.parse import urlparse

        path = urlparse(dsn).path or ""
        return path.lstrip("/") or "?"
    except Exception:
        return "?"
