import logging
from fastapi import APIRouter, Request, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from app.api.sse import UIMessageStream
from langgraph.types import Command
from app.api.auth import get_current_user, resolve_user_id

logger = logging.getLogger("app.api.chat")
router = APIRouter()


def _scope_thread(user_id: str, thread_id: str) -> str:
    """Namespace a thread per user.

    The client sends a constant thread id ("default-chat-thread"). Without
    namespacing, two different users would share one LangGraph thread and its
    whole state (schedule, tasks, profile).
    """
    if not thread_id:
        thread_id = "default_thread"
    return f"{user_id}:{thread_id}"


@router.post("/chat")
async def chat_endpoint(
    request: Request,
    body: dict,
    background_tasks: BackgroundTasks,
    verified_uid: str | None = Depends(get_current_user),
):
    graph = request.app.state.graph
    user_id = resolve_user_id(verified_uid, body.get("user_id"))
    thread_id = _scope_thread(user_id, body.get("thread_id", "default_thread"))
    message = body.get("message")
    human_feedback = body.get("human_feedback")

    config = {"configurable": {"thread_id": thread_id}, "background_tasks": background_tasks}

    # Touch the thread so a brand-new one is registered before we resume it.
    await graph.aget_state(config)

    if human_feedback is not None:
        logger.info("Resuming graph '%s' with human feedback: %s", thread_id, human_feedback)
        # Update state with human feedback and resume graph from interrupt
        await graph.aupdate_state(config, {"human_feedback": human_feedback}, as_node="propose_schedule")
        inputs = None  # Passing None resumes the graph
    else:
        logger.info("Starting new graph run for thread '%s' with message: %s", thread_id, message)
        # NOTE: 'asked_discovery_this_session' is deliberately NOT reset here.
        # It is per-session state; resetting it on every message made the
        # onboarding question repeat on every turn. 'human_feedback' *is*
        # reset on purpose — a stale approval must never apply to a new run.
        inputs = {
            "messages": [{"role": "user", "content": message}] if message else [],
            "user_id": user_id,
            "user_profile": {},
            "task_queue": [],
            "pending_tasks": [],
            "recent_notes": [],
            "needs_clarification": False,
            "proposed_schedule": None,
            "approved_schedule": None,
            "human_feedback": None,
            "academic_calendar": None
        }

    async def event_stream():
        stream = UIMessageStream()
        # Track what *this* run produced so we never re-emit a stale schedule
        # left over from a previous turn on the same thread.
        produced = {"propose": False, "commit": False}
        try:
            yield stream.start()

            # Stream updates from LangGraph to capture messages returned by custom nodes
            async for chunk in graph.astream(inputs, config, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    if not node_output or not isinstance(node_output, dict):
                        continue
                    if node_name == "build_schedule" and node_output.get("proposed_schedule"):
                        produced["propose"] = True
                    if node_name == "commit_schedule" and node_output.get("approved_schedule"):
                        produced["commit"] = True
                    if "messages" in node_output:
                        for msg in node_output["messages"]:
                            content = msg.content if hasattr(msg, "content") else (msg.get("content", "") if isinstance(msg, dict) else str(msg))
                            if content:
                                yield stream.text(content)

            # After the run, attach the schedule this run produced as a typed
            # data part. The client declares the schema in `dataPartSchemas`,
            # so it receives a parsed object rather than a JSON string.
            final_state = await graph.aget_state(config)
            values = final_state.values or {}

            if produced["propose"] and values.get("proposed_schedule"):
                yield stream.data("schedule", {
                    "status": "proposed",
                    "schedule": values["proposed_schedule"],
                })

            if produced["commit"] and values.get("approved_schedule"):
                yield stream.data("schedule", {
                    "status": "approved",
                    "schedule": values["approved_schedule"],
                })

        except Exception as e:
            logger.error("Error in event_stream: %s", e, exc_info=True)
            yield stream.error(str(e))

        yield stream.done()

    return StreamingResponse(event_stream(), headers=UIMessageStream.headers())


@router.post("/schedule/approve")
async def approve_schedule(
    request: Request,
    body: dict,
    verified_uid: str | None = Depends(get_current_user),
):
    graph = request.app.state.graph
    thread_id = body.get("thread_id")
    feedback = body.get("feedback")  # {"approved": true, "edits": []} or {"approved": false, "edits": [...]}

    if not thread_id:
        return {"status": "error", "message": "thread_id is required"}

    # The client sends a constant thread id, so the live thread is namespaced
    # by user. Try the body uid first, then the uid recorded in the thread's
    # own state, and reject the request if either disagrees with the token.
    user_id = resolve_user_id(verified_uid, body.get("user_id"))
    if not verified_uid and not body.get("user_id"):
        snapshot = await graph.aget_state(
            {"configurable": {"thread_id": _scope_thread(user_id, thread_id)}}
        )
        state_user_id = (snapshot.values or {}).get("user_id")
        if state_user_id and state_user_id != user_id:
            return {"status": "error", "message": "user_id does not match the authenticated user"}

    config = {"configurable": {"thread_id": _scope_thread(user_id, thread_id)}}

    logger.info("Approve endpoint called for thread '%s' with feedback: %s", thread_id, feedback)

    try:
        # Update the state first so that the router/nodes can read it
        await graph.aupdate_state(config, {"human_feedback": feedback}, as_node="propose_schedule")

        # Resume the graph from the interrupt using Command(resume=feedback)
        await graph.ainvoke(Command(resume=feedback), config)

        # Get final state to check the committed schedule
        final_state = await graph.aget_state(config)
        values = final_state.values or {}

        return {
            "status": "success",
            "proposed_schedule": values.get("proposed_schedule"),
            "approved_schedule": values.get("approved_schedule")
        }
    except Exception as e:
        logger.error("Error in approve_schedule: %s", e, exc_info=True)
        return {"status": "error", "message": str(e)}


# Note summarizer route
from pydantic import BaseModel

class SummarizeNoteRequest(BaseModel):
    note_content: str
    user_id: str

@router.post("/notes/summarize")
async def summarize_note_endpoint(
    req: SummarizeNoteRequest,
    verified_uid: str | None = Depends(get_current_user),
):
    from app.agents.nodes import summarize_note_content
    user_id = resolve_user_id(verified_uid, req.user_id)
    logger.info("Summarizing note for user %s via API", user_id)
    try:
        res = await summarize_note_content(req.note_content, user_id=user_id)
        try:
            from app.api.upload import link_note_to_syllabus
            linked = await link_note_to_syllabus(
                user_id=user_id,
                note_subject=res.get("subject", "General"),
                note_tags=res.get("tags", []),
                note_content=req.note_content
            )
            if linked:
                res["linked_syllabus"] = linked
        except Exception as ex:
            logger.error("Syllabus linker failed in chat summarize: %s", ex)
        return res
    except Exception as e:
        logger.error("Error in summarize_note_endpoint: %s", e, exc_info=True)
        return {
            "summary": req.note_content[:60] + "..." if len(req.note_content) > 60 else req.note_content,
            "subject": "General",
            "tags": ["general"]
        }
