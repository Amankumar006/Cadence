#!/usr/bin/env python3
"""Prove the Neon Postgres wiring works, end to end.

Run it with the connection string in the environment:

    neon env pull                 # writes .env with DATABASE_URL
    set -a && . ./.env && set +a  # or: export $(cat .env | xargs)
    npm run verify:db

Every check drives the same code the running app uses -- app.db.check_connection,
app.db.run_migrations, app.db.open_memory -- so a pass here means the app's
startup path works, not just that a socket opened.

Exits 0 only if every check passes.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def _failure_hint(detail: str) -> str:
    """Turn a driver error into the one thing to actually check.

    "Connection refused" and "connection reset by peer" mean completely
    different things, and the generic message sent people looking for a
    suspended endpoint when the real problem was a firewall.
    """
    low = detail.lower()
    if (
        "reset by peer" in low
        or "eof" in low
        or "ssl" in low
        or "tls" in low
        # libpq's wording when the TLS handshake is killed rather than refused.
        or "server closed the connection unexpectedly" in low
    ):
        return (
            "\n  TCP connected (an IPv4 address was tried after the default route"
            "\n  failed) but the TLS handshake was killed. That is a network"
            "\n  filter, not a credential: the port is reachable but encrypted"
            "\n  traffic to it is being reset. Check any proxy, VPN or egress"
            "\n  allowlist between here and Neon."
        )
    if "network is unreachable" in low:
        return (
            "\n  No route to the database host. If it resolves only to IPv6 and"
            "\n  this machine has no IPv6, libpq will not fall back to IPv4 --"
            "\n  pin an A record with hostaddr=<ipv4> in the DSN."
        )
    if "connection refused" in low or "timed out" in low or "timeout" in low:
        return (
            "\n  Nothing is listening. Check the DSN host and port, and that the"
            "\n  Neon endpoint is not suspended -- Neon scales to zero and the"
            "\n  first connect after an idle period can be slow or refused."
        )
    if "password authentication failed" in low or "authentication" in low:
        return (
            "\n  Authentication was rejected. The role or password is wrong, or"
            "\n  the role is not granted on this database. Re-run `neon env pull`."
        )
    if "does not exist" in low:
        return (
            "\n  The database or role named in the DSN does not exist. Re-run"
            "\n  `neon env pull` to get the current values."
        )
    return (
        "\n  The database is unreachable. Check the DSN and that the endpoint is"
        "\n  not suspended (Neon scales to zero)."
    )


async def main() -> int:
    from app.config import settings
    from app.db import check_connection, describe, open_memory, run_migrations

    pooled = os.environ.get("DATABASE_URL") or settings.DB_URI
    unpooled = settings.DATABASE_URL_UNPOOLED

    print("\n=== Configuration ===")
    info = describe()
    for key, value in info.items():
        print(f"  {key} = {value}")

    print("\n=== Checks ===")

    # 1. A connection string is present at all.
    if not check("DATABASE_URL is set", bool(pooled)):
        print("\n  Set it with `neon env pull`, then re-run.")
        return 1

    # 2. Neon hands back two DSNs. Without the unpooled one, DDL runs over the
    #    pooler, which fails on Neon in ways that never name pooling.
    if info["pooled_is_pooler"]:
        check("DATABASE_URL_UNPOOLED is set (needed for DDL on Neon)", bool(unpooled),
              "absent — migrations will run over the pooler" if not unpooled else "present")
    else:
        check("Local Postgres detected (no pooler)", True, "single DSN is correct here")

    # 3. Both DSNs must point at the same database, or migrations land somewhere
    #    the app never reads.
    if unpooled:
        check("Pooled and direct DSNs target the same database", info["same_database"],
              f"{info['database']}")

    # 4. The database answers.
    ok, detail = await check_connection()
    check("Database answers SELECT 1", ok, detail)
    if not ok:
        print(_failure_hint(detail))
        return 1

    # 5. Migrations are idempotent and must succeed on a clean database.
    try:
        await run_migrations()
        check("Schema migrations ran (idempotent)", True)
    except Exception as err:
        check("Schema migrations ran (idempotent)", False, f"{type(err).__name__}: {err}")
        return 1

    # 6. The pools the app actually serves from open, and both stores round-trip.
    try:
        async with open_memory() as (store, checkpointer):
            check("Store pool opened", store is not None)
            check("Checkpointer pool opened", checkpointer is not None)

            ns = ("verify", "smoke")
            await store.aput(ns, "ping", {"at": "verify_db"})
            got = await store.aget(ns, "ping")
            check("Store round-trip (aput → aget)",
                  got is not None and got.value.get("at") == "verify_db")

            from langgraph.checkpoint.base import Checkpoint

            cfg = {"configurable": {"thread_id": "verify-db", "checkpoint_ns": ""}}
            ckpt = Checkpoint(v=1, id="verify-db-cp", ts="2026-10-15T00:00:00Z",
                              channel_values={}, channel_versions={}, versions_seen={})
            await checkpointer.aput(cfg, ckpt, {}, {})
            tup = await checkpointer.aget_tuple(cfg)
            check("Checkpointer round-trip (aput → aget_tuple)",
                  tup is not None and tup.checkpoint["id"] == "verify-db-cp")
    except Exception as err:
        check("Memory pools opened and round-tripped", False, f"{type(err).__name__}: {err}")
        return 1

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("\n=== Summary ===")
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("  Failed: " + ", ".join(failed))
        return 1
    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
