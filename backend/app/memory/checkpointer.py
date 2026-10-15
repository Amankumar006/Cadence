"""
Checkpointer Module - AsyncPostgresSaver Documentation & Usage Pattern

In LangGraph v1, checkpointers provide session memory (thread-scoped continuity).
This file documents the production pattern for using `AsyncPostgresSaver` in the FastAPI backend.

Key Design Rules:
1. NEVER use `InMemorySaver` in production - session history will be lost on container restarts.
2. AsyncPostgresSaver maintains a connection pool and must be initialized within FastAPI's
   `lifespan()` context manager. Do not create/teardown checkpointers per request.
3. Always run `await checkpointer.setup()` during application startup. This creates the
   required tables ('checkpoints', 'writes', etc.) automatically.
4. Pass `checkpointer` to `compile()` when compiling the graph.

Usage Example:
```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from app.agents.graph import create_graph
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize session checkpointer and long-term store connection pools
    async with (
        AsyncPostgresStore.from_conn_string(settings.DB_URI) as store,
        AsyncPostgresSaver.from_conn_string(settings.DB_URI) as checkpointer,
    ):
        # Setup tables automatically
        await store.setup()
        await checkpointer.setup()
        
        # Compile graph with both layers
        app.state.graph = create_graph(store=store, checkpointer=checkpointer)
        yield
```

Accessing checkpoint state in routes:
```python
# Fetch current state of a thread:
state = await graph.aget_state(config)

# Resume execution of an interrupted thread:
async for event in graph.astream(None, config, stream_mode="messages"):
    # ... stream events
```
"""

import logging

logger = logging.getLogger("app.memory.checkpointer")

# Self-documentation module only. Code behaves as documentation structure.
def get_checkpointer_info():
    return {
        "class": "AsyncPostgresSaver",
        "package": "langgraph-checkpoint-postgres",
        "purpose": "Persist message history and state variables within a specific thread_id (Session Memory).",
        "concurrency_safe": True
    }
