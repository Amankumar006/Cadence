import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.db import check_connection, describe, open_memory
from app.agents.graph import create_graph
from app.api.chat import router as chat_router
from app.api.upload import router as upload_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Export the LangSmith env vars BEFORE the first LLM call. LangChain reads
    # them lazily per client, so this must happen after the settings are
    # imported but before any chain is built or invoked.
    from app.agents.llm import setup_tracing

    if setup_tracing():
        logger.info(
            "LangSmith tracing enabled for project '%s'", settings.LANGCHAIN_PROJECT
        )

    # Initialize LangGraph v1 checkpointer and store inside lifespan.
    # open_memory() runs the DDL migrations on Neon's *direct* connection first,
    # then opens the live pools on the *pooled* one — see app/db.py.
    logger.info("Initializing database connection pools...")
    async with open_memory() as (store, checkpointer):
        # Run setup to automatically create postgres tables if they do not exist
        logger.info("Running store and checkpointer setups...")
        await store.setup()
        await checkpointer.setup()
        
        # Compile the state graph and attach to app.state
        logger.info("Compiling StateGraph...")
        app.state.graph = create_graph(store=store, checkpointer=checkpointer)
        app.state.store = store
        
        # Start the background notification scheduler
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from app.tasks.push_sender import send_pending_notifications
        from app.tasks.nightly_learning import run_nightly_learning
        
        logger.info("Starting AsyncIOScheduler for push notifications and nightly learning...")
        scheduler = AsyncIOScheduler()
        scheduler.add_job(send_pending_notifications, "interval", minutes=1)
        scheduler.add_job(run_nightly_learning, "cron", hour=23, minute=55)
        scheduler.start()
        app.state.scheduler = scheduler
        
        try:
            yield  # App runs here
        finally:
            logger.info("Stopping APScheduler...")
            try:
                scheduler.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down APScheduler: {e}")

app = FastAPI(
    title="Personal AI Agent Backend",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware for mobile development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root smoke test endpoint
@app.get("/")
async def root():
    return {"status": "ok", "message": "Personal AI Agent Backend Scaffolding is running!"}

@app.get("/health")
async def health(request: Request):
    import datetime
    
    scheduler = getattr(request.app.state, "scheduler", None)
    scheduler_status = "running" if (scheduler and scheduler.running) else "stopped"
    graph_status = "ready" if hasattr(request.app.state, "graph") else "not_ready"
    
    # Async, and on the pooled DSN. The previous version called the
    # synchronous psycopg.connect() inside an async endpoint, which blocks the
    # event loop for every other in-flight request for as long as the connect
    # takes.
    ok, detail = await check_connection()
    db_status = "connected" if ok else f"disconnected: {detail}"

    return {
        "status": "ok" if ok else "error",
        "scheduler": scheduler_status,
        "graph": graph_status,
        "database": db_status,
        "db": describe(),
        "timestamp": datetime.datetime.now().isoformat(),
    }


# Include chat router
app.include_router(chat_router, prefix="/api")
app.include_router(upload_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", settings.PORT))
    uvicorn.run("app.main:app", host=settings.HOST, port=port, reload=True)
