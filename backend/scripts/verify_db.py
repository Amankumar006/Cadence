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


async def main() -> int:
    from app.config import settings
    from app.db import check_connection, describe, open_memory, run_migrations

    pooled = os.environ.get("DATABASE_URL") or settings.DB_URI
    unpooled = os.environ.get("DATABASE_URL_UNPOOLED")

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
        print("\n  The database is unreachable. Check the DSN and that the")
        print("  endpoint is not suspended (Neon scales to zero).")
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
