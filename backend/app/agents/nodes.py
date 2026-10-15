import logging
import asyncio
import datetime
import json
import re
from typing import Any, Dict
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from app.agents.llm import atext, astructured, bypass_llm
from app.agents.schemas import (
    IntentClassification,
    NoteSummary,
    ScheduleProposal,
)
from app.agents.state import AgentState
from app.memory.store import save_user_profile, sync_and_load_user_profile
from app.tools.memory import store_preference_func
from app.tools.timetable import fetch_timetable_func
from app.tools.calendar import write_calendar_func
from app.tools.notifications import schedule_push_func, send_immediate_push
from app.tools.tmdb import search_movies_func
from app.agents.discovery import identify_missing_slots
from app.agents.parser import parse_due_date, parse_user_info
from firebase_admin import firestore_async

logger = logging.getLogger("app.agents.nodes")

async def summarize_note_content(body: str, user_id: str = None) -> dict:
    """Uses the LLM to summarize note content and extract subject + tags.

    Schema-constrained: the model cannot return malformed JSON, and the
    fallback is deterministic. Never raises.
    """
    logger.info("Summarizing note content: %s...", (body or "")[:60])
    prompt = f"""Extract a 1-sentence summary, the relevant subject, and 2-4 tags from this note.
Note content: {body}"""
    fallback = NoteSummary(
        summary=(body[:60] + "...") if len(body) > 60 else body,
        subject="General",
        tags=["general"],
    )
    result = await astructured(prompt, NoteSummary, user_id=user_id, fallback=fallback)
    logger.info("Note summarized: subject=%s tags=%s", result.subject, result.tags)
    return {"summary": result.summary, "subject": result.subject, "tags": result.tags}


def parse_date_from_message(message: str) -> str:
    """Resolve a date mentioned in a message, defaulting to today.

    Delegates to :func:`app.agents.parser.parse_due_date` so there is exactly
    one date parser in the codebase.
    """
    parsed = parse_due_date(message)
    return parsed or datetime.date.today().isoformat()


def _classify_heuristically(message: str) -> tuple:
    """Regex intent classification used for test users and LLM failure."""
    last_msg_lower = (message or "").lower().strip()

    # A question about existing data is a retrieval query, not new info.
    is_query_for_info = any(
        q in last_msg_lower
        for q in [
            "what", "when", "show", "give", "list", "check", "tell me",
            "my monday", "my tuesday", "my wednesday", "my thursday",
            "my friday", "my saturday", "my sunday", "my class", "my timetable",
            "get my",
        ]
    )

    if "arrived at" in last_msg_lower and any(
        loc in last_msg_lower for loc in ["gym", "college", "library", "home"]
    ):
        return "location_trigger", "day"
    if any(
        kw in last_msg_lower
        for kw in ["movie", "recommend", "watch", "film", "suggest", "binge", "entertainment"]
    ):
        return "entertainment_rec", "day"
    if re.search(r"\b(edit|change|move|remove|add)\b", last_msg_lower):
        return "schedule_edit", "day"
    if not is_query_for_info and (
        any(
            kw in last_msg_lower
            for kw in [
                "cie", "exam", "assignment", "viva", "meditate", "workout", "unit",
                "timetable", "time table", "schedule info", "study", "rvce", "bmsce",
                "msrit", "bti", "vtu", "semester", "sem", "scheme", "buy", "task",
                "todo", "reminder", "weightage", "note", "lab manual", "syllabus",
                "class at", "routine", "deadline", "due",
            ]
        )
        or last_msg_lower in ["no", "none", "nothing", "not yet", "no i don't", "no, thanks", "no thanks"]
    ):
        return "info_provided", "day"
    if "schedule" in last_msg_lower or "today" in last_msg_lower:
        return "schedule_build", "day"
    return "general", "day"

async def classify_intent(state: AgentState, store: BaseStore) -> Dict[str, Any]:
    """Classifies the user message into intent, scope, and date."""
    logger.info("Node: classify_intent")
    
    user_id = state.get("user_id", "default_user")
    messages = state.get("messages", [])
    last_message = messages[-1].content if messages else ""
    
    if not last_message:
        return {
            "intent": "general",
            "intent_scope": "day",
            "intent_date": datetime.date.today().isoformat(),
            "needs_clarification": False
        }
        
    user_profile = await sync_and_load_user_profile(store, user_id)
    last_topic = user_profile.get("last_discovery_topic") if user_profile else None
    context_str = ""
    if last_topic:
        context_str = f"\nContext: The assistant recently asked the user a question about their '{last_topic.replace('_', ' ')}'."
        if last_topic == "college":
            context_str += "\nCRITICAL: If the user provides a college name, abbreviation, or association (e.g. 'BTI', 'Bti', 'RVCE', 'BTI associated with vtu'), you MUST classify the 'intent' as 'info_provided'."
            
    classify_prompt = f"""You are an intent classifier for a personal AI academic scheduler. {context_str}
Classify the user's latest message into one of these intents:
- "schedule_build": User wants to build/generate/plan a schedule for a day or week (e.g., "schedule my day", "plan today", "what's my schedule").
- "schedule_edit": User wants to edit/change a proposed or existing schedule (e.g., "move gym to 7pm", "remove the study block").
- "entertainment_rec": User wants movie, show, or entertainment recommendations.
- "location_trigger": Automated geofence arrival messages (e.g., "arrived at gym", "arrived at college").
- "info_provided": User is sharing/entering/providing NEW information to the system about their college name, course details, timetable, syllabus, academic events (CIE/SEE/viva/record), or personal routines (e.g., "I study at RVCE", "CIE-1 is on June 20", "I have a lab record due tomorrow", "I go to gym at 6pm daily", "I finished unit 2", "no", "none", "nothing this week", "Bti", "BTI associated with vtu"). IMPORTANT: This is only for inputting/saving new data.
- "general": General chit-chat, greetings, or questions, including requests to view, retrieve, check, or show their existing classes, timetable, calendar, or tasks (e.g., "what are my classes on Monday", "show my timetable", "give my Monday college timetable", "hi", "how are you", "who are you").

User Message: "{last_message}"

Respond with the intent, the scope ("day" or "week"), and the date the message refers to.
Resolve relative dates ("today", "tomorrow", "next Monday", "20 Oct") to an absolute
YYYY-MM-DD date using today's date: {datetime.date.today().isoformat()}.
If no date is mentioned or implied, leave the date empty."""

    intent = "general"
    intent_scope = "day"
    intent_date = datetime.date.today().isoformat()

    if bypass_llm(user_id):
        logger.info("Bypassing LLM classification for user: %s", user_id)
        intent, intent_scope = _classify_heuristically(last_message)
        intent_date = parse_date_from_message(last_message)
    else:
        parsed = await astructured(
            classify_prompt, IntentClassification, user_id=user_id
        )
        if parsed is None:
            logger.warning("Classification failed; using heuristic fallback")
            intent, intent_scope = _classify_heuristically(last_message)
            intent_date = parse_date_from_message(last_message)
        else:
            intent = parsed.intent
            intent_scope = parsed.scope
            intent_date = parsed.date or parse_date_from_message(last_message)

    logger.info("Classified intent: %s, scope: %s, date: %s", intent, intent_scope, intent_date)
    return {
        "intent": intent,
        "intent_scope": intent_scope,
        "intent_date": intent_date,
        "needs_clarification": False
    }

async def save_parsed_info(user_id: str, parsed: dict, store: BaseStore) -> str:
    """Helper to save parsed information to both Firestore and LangGraph Store."""
    info_type = parsed.get("type")
    info_data = parsed.get("data")
    if not info_data or info_type == "unknown":
        return "No data extracted."
        
    db = firestore_async.client()
    user_profile = await sync_and_load_user_profile(store, user_id)
    
    saved_any_info = False
    thing = "information"
    
    if info_type == "academic_event":
        if "confidence" not in info_data:
            info_data["confidence"] = "manual"
        await db.collection("academic_events").document(user_id).collection("items").add(info_data)
        completed = user_profile.get("completed_discovery", [])
        if "academic_events" not in completed:
            completed.append("academic_events")
            user_profile["completed_discovery"] = completed
            await save_user_profile(store, user_id, user_profile)
            await db.collection("users").document(user_id).set({"completed_discovery": completed}, merge=True)
        thing = info_data.get("title") or info_data.get("subject", "academic event")
        saved_any_info = True
        
    elif info_type in ["timetable", "timetable_update"]:
        await db.collection("timetables").document(user_id).set(info_data, merge=True)
        user_profile["timetable"] = info_data
        await save_user_profile(store, user_id, user_profile)
        
        completed = user_profile.get("completed_discovery", [])
        if "timetable" not in completed:
            completed.append("timetable")
            user_profile["completed_discovery"] = completed
            await save_user_profile(store, user_id, user_profile)
            await db.collection("users").document(user_id).set({"completed_discovery": completed}, merge=True)
        thing = "timetable"
        saved_any_info = True
        
    elif info_type in ["personal_block", "personal_schedule"]:
        doc_ref = db.collection("personal_schedule").document(user_id)
        doc = await doc_ref.get()
        existing_blocks = (doc.to_dict() or {}).get("fixed_blocks", []) if doc.exists else []
        
        new_blocks = info_data if isinstance(info_data, list) else [info_data]
        for nb in new_blocks:
            if nb and isinstance(nb, dict):
                nb_title = nb.get("title", "Routine")
                nb_time = nb.get("time", "08:00")
                nb_days = nb.get("days", ["daily"])
                nb_duration = nb.get("duration_min", 30)
                nb_type = nb.get("type", "personal")
                nb_non_neg = nb.get("non_negotiable")
                if nb_non_neg is None:
                    nb_non_neg = True
                if any(x in nb_title.lower() for x in ["meditat", "sleep"]):
                    nb_non_neg = True
                
                block_to_save = {
                    "title": nb_title,
                    "time": nb_time,
                    "days": nb_days,
                    "duration_min": nb_duration,
                    "type": nb_type,
                    "non_negotiable": nb_non_neg
                }
                
                if not any(eb.get("title") == block_to_save["title"] and eb.get("time") == block_to_save["time"] for eb in existing_blocks):
                    existing_blocks.append(block_to_save)
                    
        await doc_ref.set({"fixed_blocks": existing_blocks}, merge=True)
        user_profile["personal_blocks"] = existing_blocks
        await save_user_profile(store, user_id, user_profile)
        
        completed = user_profile.get("completed_discovery", [])
        if "personal_blocks" not in completed:
            completed.append("personal_blocks")
            user_profile["completed_discovery"] = completed
            await save_user_profile(store, user_id, user_profile)
            await db.collection("users").document(user_id).set({"completed_discovery": completed}, merge=True)
        thing = new_blocks[0].get("title", "routine") if new_blocks else "routine"
        saved_any_info = True
        
    elif info_type == "syllabus_update":
        subject = info_data.get("subject")
        if subject:
            doc_ref = db.collection("syllabus").document(user_id).collection("subjects").document(subject)
            doc = await doc_ref.get()
            if doc.exists:
                existing_data = doc.to_dict() or {}
                existing_units = existing_data.get("units", [])
                new_units = info_data.get("units", [])
                for nu in new_units:
                    unit_num = nu.get("number")
                    found = False
                    for i, eu in enumerate(existing_units):
                        if eu.get("number") == unit_num:
                            existing_units[i] = {**eu, **nu}
                            found = True
                            break
                    if not found:
                        existing_units.append(nu)
                info_data["units"] = existing_units
            await doc_ref.set(info_data, merge=True)
            
            completed = user_profile.get("completed_discovery", [])
            if "syllabus_status" not in completed:
                completed.append("syllabus_status")
                user_profile["completed_discovery"] = completed
                await save_user_profile(store, user_id, user_profile)
                await db.collection("users").document(user_id).set({"completed_discovery": completed}, merge=True)
        thing = f"{subject} syllabus"
        saved_any_info = True
        
    elif info_type == "college_info":
        college_name = info_data.get("college")
        branch = info_data.get("branch")
        semester = info_data.get("semester")
        scheme = info_data.get("scheme")
        
        updates = {}
        if college_name:
            user_profile["college"] = college_name
            updates["college"] = college_name
        if branch:
            user_profile["branch"] = branch
            updates["branch"] = branch
        if semester:
            user_profile["semester"] = semester
            updates["semester"] = semester
        if scheme:
            user_profile["scheme"] = scheme
            updates["scheme"] = scheme
            
        if updates:
            await save_user_profile(store, user_id, user_profile)
            await db.collection("users").document(user_id).set(updates, merge=True)
            
        import asyncio
        if college_name:
            from app.tasks.college_scraper import run_college_calendar_scraper_task
            asyncio.create_task(run_college_calendar_scraper_task(user_id, college_name))
        if branch and semester and scheme:
            from app.tasks.college_scraper import fetch_syllabus_for_student_task
            asyncio.create_task(fetch_syllabus_for_student_task(user_id, branch, scheme, semester))
            
        completed = user_profile.get("completed_discovery", [])
        if "college" not in completed:
            completed.append("college")
            user_profile["completed_discovery"] = completed
            await save_user_profile(store, user_id, user_profile)
            await db.collection("users").document(user_id).set({"completed_discovery": completed}, merge=True)
        thing = college_name or "college"
        saved_any_info = True
        
    elif info_type == "task":
        task_ref = db.collection("tasks").document(user_id).collection("items")
        await task_ref.add({
            "title": info_data.get("title", "Task"),
            "type": info_data.get("type", "reminder"),
            "due_date": info_data.get("due_date", datetime.date.today().isoformat()),
            "priority": info_data.get("priority", "medium"),
            "completed": False,
            "subject": info_data.get("subject"),
            "created_at": datetime.datetime.utcnow().isoformat()
        })
        completed = user_profile.get("completed_discovery", [])
        if "academic_events" not in completed:
            completed.append("academic_events")
            user_profile["completed_discovery"] = completed
            await save_user_profile(store, user_id, user_profile)
            await db.collection("users").document(user_id).set({"completed_discovery": completed}, merge=True)
        thing = info_data.get("title", "task")
        saved_any_info = True
        
    elif info_type == "note":
        note_body = info_data.get("body") or parsed.get("message", "")
        summary_data = await summarize_note_content(note_body, user_id=user_id)
        note_ref = db.collection("notes").document(user_id).collection("items")
        await note_ref.add({
            "title": info_data.get("title") or summary_data.get("subject") or "Note",
            "body": note_body,
            "ai_summary": summary_data.get("summary", ""),
            "subject": summary_data.get("subject", "General"),
            "tags": info_data.get("tags") or summary_data.get("tags") or ["general"],
            "created_at": datetime.datetime.utcnow().isoformat()
        })
        thing = info_data.get("title", "note")
        saved_any_info = True
        
    elif info_type == "profile_update":
        await db.collection("users").document(user_id).set(info_data, merge=True)
        if "gym_days" in info_data or "gym_time" in info_data:
            existing_completed = user_profile.get("completed_discovery", [])
            if "gym_schedule" not in existing_completed:
                existing_completed.append("gym_schedule")
                user_profile["completed_discovery"] = existing_completed
        for k, v in info_data.items():
            if k == "completed_discovery":
                existing_completed = user_profile.get("completed_discovery", [])
                for item in v:
                    if item not in existing_completed:
                        existing_completed.append(item)
                user_profile["completed_discovery"] = existing_completed
            else:
                user_profile[k] = v
        await save_user_profile(store, user_id, user_profile)
        thing = "profile"
        saved_any_info = True
        
    else:
        return "Unknown data type."
        
    if saved_any_info:
        user_profile["last_discovery_topic"] = None
        await save_user_profile(store, user_id, user_profile)
        await db.collection("users").document(user_id).set({"last_discovery_topic": None}, merge=True)
        
    return f"Saved {thing}."

async def hydrate_context(state: AgentState, store: BaseStore) -> Dict[str, Any]:
    """Hydrates the agent state with user profile, timetable, and academic context."""
    logger.info("Node: hydrate_context")
    
    user_id = state.get("user_id", "default_user")
    date = state.get("intent_date") or datetime.date.today().isoformat()
    
    # 1. Load user profile and sync/merge with Firestore users doc
    user_profile = await sync_and_load_user_profile(store, user_id)
    
    messages = state.get("messages", [])
    last_user_msg = (
        messages[-1].content
        if hasattr(messages[-1], "content")
        else (messages[-1].get("content", "") if isinstance(messages[-1], dict) else "")
        if messages
        else ""
    )

    # If the last message provided information, parse it
    if last_user_msg:
        try:
            parsed = await parse_user_info(last_user_msg, user_profile, user_id)
            if parsed.get("type") != "unknown":
                logger.info(f"Parsing inline info: {parsed}")
                await save_parsed_info(user_id, parsed, store)
                user_profile = await sync_and_load_user_profile(store, user_id)  # Reload updated profile
        except Exception as parse_err:
            logger.warning(f"Failed parsing inline user message: {parse_err}")

    # 2. Fetch timetable, academic events, personal schedule, calendar, and notes in parallel
    db = firestore_async.client()
    
    timetable_task = fetch_timetable_func(user_id=user_id, date=date)
    tasks_task = db.collection("tasks").document(user_id).collection("items").get()
    events_task = db.collection("academic_events").document(user_id).collection("items").get()
    personal_task = db.collection("personal_schedule").document(user_id).get()
    calendar_task = db.collection("academic_calendar").document(user_id).get()
    notes_task = db.collection("notes").document(user_id).collection("items").get()
    
    timetable_res, tasks_snap, events_snap, personal_doc, calendar_doc, notes_snap = await asyncio.gather(
        timetable_task,
        tasks_task,
        events_task,
        personal_task,
        calendar_task,
        notes_task
    )
    
    # Process tasks
    tasks = []
    for doc in tasks_snap:
        tdata = doc.to_dict()
        if not tdata.get("completed", False):
            tdata["id"] = doc.id
            tasks.append(tdata)
            
    classes = timetable_res.get("classes") or []
    weekday = timetable_res.get("weekday", "")
    timetable_today = {"classes": classes, "weekday": weekday}
    
    # Filter academic events: due within 14 days and completed=false, sorted by due_date
    academic_events = []
    dt_today = datetime.date.fromisoformat(date)
    for doc in events_snap:
        event_data = doc.to_dict()
        event_data["id"] = doc.id
        if not event_data.get("completed", False):
            due_date_str = event_data.get("due_date")
            if due_date_str:
                try:
                    dt_due = datetime.date.fromisoformat(due_date_str)
                    days_remaining = (dt_due - dt_today).days
                    if 0 <= days_remaining <= 14:
                        event_data["days_remaining"] = days_remaining
                        academic_events.append(event_data)
                except ValueError:
                    pass
    academic_events.sort(key=lambda x: x.get("due_date", "9999-12-31"))
    
    # Personal fixed blocks
    personal_fixed_blocks = (personal_doc.to_dict() or {}).get("fixed_blocks", []) if personal_doc.exists else []
    
    # Academic calendar
    academic_calendar = calendar_doc.to_dict() or {} if calendar_doc.exists else {}

    # Fetch syllabus status in parallel for subjects with upcoming events
    syllabus_status = {}
    subjects_due_soon = set()
    for event in academic_events:
        sub = event.get("subject")
        if sub and sub.lower() != "all":
            subjects_due_soon.add(sub)
                
    async def fetch_subject_syllabus(subj):
        try:
            doc = await db.collection("syllabus").document(user_id).collection("subjects").document(subj).get()
            if doc.exists:
                return subj, doc.to_dict()
        except Exception as e:
            logger.error(f"Error fetching syllabus for {subj}: {e}")
        return subj, None

    if subjects_due_soon:
        syllabus_results = await asyncio.gather(*(fetch_subject_syllabus(s) for s in subjects_due_soon))
        for subj, syl_data in syllabus_results:
            if syl_data:
                syllabus_status[subj] = syl_data

    college_info = state.get("college_info")
    if college_info is None:
        college_info = user_profile.get("college_info") or {}
        
    calendar_status = state.get("calendar_status")
    if calendar_status is None:
        calendar_status = user_profile.get("calendar_status") or "unknown"
        
    # Process recent notes (last 7 days)
    recent_notes = []
    dt_today = datetime.date.fromisoformat(date)
    dt_7_days_ago = dt_today - datetime.timedelta(days=7)
    for doc in notes_snap:
        ndata = doc.to_dict()
        ndata["id"] = doc.id
        created_at_str = ndata.get("created_at")
        is_recent = True
        if created_at_str:
            try:
                dt_created = datetime.datetime.fromisoformat(created_at_str.replace("Z", "+00:00")).date()
                if dt_created < dt_7_days_ago:
                    is_recent = False
            except Exception:
                pass
        if is_recent:
            if "type" not in ndata:
                ndata["type"] = "text"
            recent_notes.append(ndata)

    # Bypassed guided discovery loop in hydrate_context (handled in check_clarification_node)
    return {
        "user_profile": user_profile,
        "timetable_today": timetable_today,
        "task_queue": tasks,
        "pending_tasks": tasks,
        "recent_notes": recent_notes,
        "academic_events": academic_events,
        "personal_fixed_blocks": personal_fixed_blocks,
        "syllabus_status": syllabus_status,
        "needs_clarification": False,
        "clarification_question": None,
        "college_info": college_info,
        "calendar_status": calendar_status,
        "academic_calendar": academic_calendar
    }

async def ask_clarification(state: AgentState, store: BaseStore) -> Dict[str, Any]:
    """Asks the user for clarification when critical information is missing."""
    logger.info("Node: ask_clarification")
    question = state.get("clarification_question") or "Could you clarify what time you'd like to schedule this?"
    return {
        "messages": [{"role": "assistant", "content": question}],
        "needs_clarification": False  # Reset flag once asked
    }

def generate_rule_based_schedule(
    user_id: str,
    date: str,
    weekday_name: str,
    is_weekend: bool,
    classes: list,
    personal_fixed_blocks: list,
    academic_events: list,
    academic_calendar: dict,
    user_profile: dict,
    pending_tasks: list = None,
    recent_notes: list = None
) -> list:
    """Programmatically generates a schedule according to Phase 5 system rules."""
    logger.info(f"Generating rule-based fallback schedule for user: {user_id}, date: {date} ({weekday_name})")
    blocks = []
    
    # Check if today is a VTU holiday
    is_holiday = False
    holidays = academic_calendar.get("holidays") or []
    for h in holidays:
        h_date = h.get("date")
        if h_date == date:
            is_holiday = True
            logger.info(f"Today is a VTU holiday: {h.get('name')}. Excluding study expectations.")
            break
            
    # 1. Personal Fixed blocks (e.g. Meditation at 6am daily)
    day_short = weekday_name[:3]
    for block in personal_fixed_blocks:
        title = block.get("title", "Routine Block")
        time = block.get("time", "06:00")
        duration = block.get("duration_min", 30)
        days = block.get("days", [])
        
        is_today = "daily" in (d.lower() for d in days) or any(d.lower().startswith(day_short.lower()) for d in days)
        if is_today:
            blocks.append({
                "time": time,
                "title": title,
                "type": block.get("type", "personal"),
                "duration_min": duration,
                "notes": f"Personal fixed block: {title}."
            })
            
    # 2. Add classes from timetable if not weekend and not holiday
    if not is_holiday and not is_weekend:
        for c in classes:
            blocks.append({
                "time": c.get("time", "09:00"),
                "title": c.get("subject", "Class"),
                "type": "personal",
                "duration_min": c.get("duration_min", 60),
                "notes": f"Class lecture. Room: {c.get('room', 'N/A')}"
            })
            
    # 3. Add academic study blocks if not Sunday and not holiday
    if not is_holiday and weekday_name.lower() != "sunday":
        try:
            dt_today = datetime.date.fromisoformat(date)
        except Exception:
            dt_today = datetime.date.today()
            
        for event in academic_events:
            due_date_str = event.get("due_date")
            subject = event.get("subject", "Study")
            if due_date_str:
                try:
                    dt_due = datetime.date.fromisoformat(due_date_str)
                    days_remaining = (dt_due - dt_today).days
                    # Rules:
                    # CIE within 3 days -> 3hr study block (180 min) containing "CRITICAL"
                    # CIE within 7 days -> 2hr study block (120 min)
                    if 0 <= days_remaining <= 3:
                        blocks.append({
                            "time": "15:00",
                            "title": f"{subject} CIE prep (CRITICAL)",
                            "type": "study",
                            "duration_min": 180,
                            "notes": f"CIE prep (CRITICAL). Exam on {due_date_str}."
                        })
                    elif 4 <= days_remaining <= 7:
                        blocks.append({
                            "time": "16:00",
                            "title": f"{subject} study prep",
                            "type": "study",
                            "duration_min": 120,
                            "notes": f"CIE prep. Exam on {due_date_str}."
                        })
                except Exception as ex:
                    logger.error(f"Error parsing event due date in schedule builder: {ex}")
                    
    # 4. Add Saturday Movie Night Mandate
    if weekday_name.lower() == "saturday":
        movie_title = "Terminator 2: Judgment Day"
        movie_rating = "8.6"
        movie_overview = "A mechanical killer is sent to protect young John Connor."
        blocks.append({
            "time": "21:30",
            "title": f"Movie Night: {movie_title}",
            "type": "ent",
            "duration_min": 120,
            "notes": f"⭐ {movie_rating}/10 - {movie_overview}"
        })
    # 6. Add pending tasks due on this date
    if pending_tasks:
        for t in pending_tasks:
            due_date = t.get("due_date")
            if due_date == date and not t.get("completed"):
                blocks.append({
                    "time": "14:00",
                    "title": f"Task: {t.get('title')}",
                    "type": "work",
                    "duration_min": 60,
                    "notes": f"Complete pending task: {t.get('title')}. Priority: {t.get('priority')}."
                })
                
    # 7. Add study blocks based on note weightage / details
    if recent_notes:
        for n in recent_notes:
            body_lower = n.get("body", "").lower()
            tags_lower = [tag.lower() for tag in n.get("tags", [])]
            if "dsa" in body_lower or "dsa" in tags_lower:
                blocks.append({
                    "time": "18:00",
                    "title": "DSA Study Block",
                    "type": "study",
                    "duration_min": 90,
                    "notes": f"Study block for DSA. Note context: {n.get('ai_summary')}"
                })
        
    # Sort blocks by start time to keep it ordered
    blocks.sort(key=lambda x: x.get("time", "00:00"))
    
    # 5. Add sleep block
    sleep_target = user_profile.get("sleep_target", "23:00")
    if weekday_name.lower() == "saturday":
        sleep_time = "23:30"
    else:
        sleep_time = sleep_target
        
    blocks.append({
        "time": sleep_time,
        "title": "Sleep",
        "type": "rest",
        "duration_min": 480,
        "notes": "Target sleep time"
    })
    
    return blocks

async def build_schedule(state: AgentState) -> Dict[str, Any]:
    """Generates the proposed schedule based on timetable, profile, and tasks."""
    logger.info("Node: build_schedule")
    
    user_id = state.get("user_id", "default_user")
    date = state.get("intent_date") or "2026-06-08"
    
    # Retrieve fetched timetable, profile, events, personal blocks, and academic calendar
    timetable_today = state.get("timetable_today") or {}
    classes = timetable_today.get("classes") or []
    user_profile = state.get("user_profile") or {}
    academic_events = state.get("academic_events") or []
    personal_fixed_blocks = state.get("personal_fixed_blocks") or []
    syllabus_status = state.get("syllabus_status") or {}
    academic_calendar = state.get("academic_calendar") or {}
    pending_tasks = state.get("pending_tasks") or []
    recent_notes = state.get("recent_notes") or []
    
    # 1. Parse date and determine weekday / weekend mode
    try:
        dt = datetime.date.fromisoformat(date)
    except Exception:
        dt = datetime.date(2026, 6, 8)
    weekday_name = dt.strftime("%A")
    is_weekend = dt.weekday() >= 5
    current_mode = "weekend" if is_weekend else "weekday"
    
    # 2. Format inputs for the prompt
    timetable_classes = ""
    for c in classes:
        time = c.get("time", "09:00")
        subject = c.get("subject", "Class")
        room = c.get("room", "")
        textbooks = c.get("textbooks", [])
        notes = f"Room {room}" if room else ""
        if textbooks:
            notes += f" (Read: {', '.join(textbooks)})" if notes else f"Read: {', '.join(textbooks)}"
        timetable_classes += f"- {time}: {subject} ({notes})\n"
    if not timetable_classes:
        timetable_classes = "No classes scheduled today."
        
    role = user_profile.get("role", "student")
    wake_time = user_profile.get("wake_time", "07:00")
    actual_wake_avg = user_profile.get("actual_wake_avg", "07:15")
    sleep_target = user_profile.get("sleep_target", "23:00")
    peak_focus_time = user_profile.get("peak_focus_time", "morning")
    study_block_pref = user_profile.get("study_block_pref", "90")
    gym_days_list = user_profile.get("gym_days", [])
    gym_days = ", ".join(gym_days_list) if isinstance(gym_days_list, list) else str(gym_days_list)
    gym_time = user_profile.get("gym_time", "18:00")
    
    # Format personal fixed blocks (non-negotiable vs flexible)
    non_negotiable_str = ""
    flexible_str = ""
    day_short = weekday_name[:3]
    for block in personal_fixed_blocks:
        title = block.get("title", "Routine Block")
        time = block.get("time", "00:00")
        duration = block.get("duration_min", 30)
        days = block.get("days", [])
        non_negotiable = block.get("non_negotiable", False)
        
        is_today = "daily" in (d.lower() for d in days) or any(d.lower().startswith(day_short.lower()) for d in days)
        if is_today:
            block_info = f"- {time} ({duration} min): {title} (Type: {block.get('type', 'personal')})\n"
            if non_negotiable:
                non_negotiable_str += block_info
            else:
                flexible_str += block_info
                
    if not non_negotiable_str:
        non_negotiable_str = "None scheduled for today."
    if not flexible_str:
        flexible_str = "None scheduled for today."
        
    # Format academic events and syllabus risk assessment
    academic_events_str = ""
    risk_assessment_str = ""
    dt_today = datetime.date.fromisoformat(date)
    sorted_events = sorted(academic_events, key=lambda x: x.get("due_date", "9999-12-31"))
    
    for event in sorted_events:
        title = event.get("title", "Event")
        subject = event.get("subject", "")
        due_date_str = event.get("due_date")
        completed = event.get("completed", False)
        if completed:
            continue
            
        days_remaining = None
        if due_date_str:
            try:
                dt_due = datetime.date.fromisoformat(due_date_str)
                days_remaining = (dt_due - dt_today).days
            except ValueError:
                pass
                
        if days_remaining is not None and days_remaining >= 0:
            confidence = event.get("confidence", "manual")
            academic_events_str += f"- {title} (Subject: {subject}, Due: {due_date_str}, {days_remaining} days remaining, Marks: {event.get('marks', 'N/A')}, Confidence: {confidence})\n"
            
            # Syllabus Risk Assessment for events due soon (within 14 days)
            if days_remaining <= 14:
                syl = syllabus_status.get(subject) or {}
                units = syl.get("units", [])
                
                units_pending = []
                for u in units:
                    status = u.get("status", "not_started")
                    if status != "done":
                        units_pending.append(f"Unit {u.get('number')}: {u.get('title', 'Untitled')} ({status})")
                        
                completion_pct = 0
                if units:
                    completion_pct = int(sum(u.get("completion_percent", 0) for u in units) / len(units))
                    
                risk_assessment_str += f"- Subject: {subject}\n"
                risk_assessment_str += f"  Event: {title} in {days_remaining} days\n"
                risk_assessment_str += f"  Syllabus Progress: {completion_pct}% complete\n"
                risk_assessment_str += f"  Units Pending: {', '.join(units_pending) if units_pending else 'None'}\n"

    if not academic_events_str:
        academic_events_str = "No upcoming academic events."
    if not risk_assessment_str:
        risk_assessment_str = "No upcoming events require syllabus risk assessment."
        
    # Format academic calendar dates
    cie_dates = academic_calendar.get("cie_dates") or []
    see_start = academic_calendar.get("see_start") or "N/A"
    
    # Find next upcoming holiday
    holidays = academic_calendar.get("holidays") or []
    next_holiday = "N/A"
    upcoming_holidays = []
    for h in holidays:
        h_date = h.get("date")
        if h_date:
            try:
                dt_h = datetime.date.fromisoformat(h_date)
                if dt_h >= dt_today:
                    upcoming_holidays.append((dt_h, h.get("name", "Holiday")))
            except ValueError:
                pass
    if upcoming_holidays:
        upcoming_holidays.sort(key=lambda x: x[0])
        next_holiday = f"{upcoming_holidays[0][1]} on {upcoming_holidays[0][0].isoformat()}"
    
    # ── Skip Probabilities & Reschedule Patterns ──
    skip_probabilities = user_profile.get("skip_probabilities", {}) or {}
    reschedule_patterns = user_profile.get("reschedule_patterns", {}) or {}
    patterns = {**skip_probabilities, **reschedule_patterns}
    
    warnings_list = []
    for pattern_key, prob in patterns.items():
        if "_" in pattern_key:
            parts = pattern_key.split("_")
            b_type = parts[0]
            d_name = parts[1]
            if d_name == weekday_name.lower():
                if prob > 0.6:
                    warnings_list.append(
                        f"User has a high skip probability ({prob:.2f}) for '{b_type}' on {weekday_name}s. "
                        f"If you schedule '{b_type}' today, you MUST append '(you often skip this)' to the block's notes. "
                        f"Alternatively, you may choose to remove/omit '{b_type}' from today's schedule entirely."
                    )
                    
    patterns_str = "\n".join([f"- {k}: {v}" for k, v in patterns.items()]) if patterns else "No learned patterns yet."
    if warnings_list:
        patterns_str += "\n\n⚠️ HIGH SKIP WARNINGS FOR TODAY:\n" + "\n".join([f"- {w}" for w in warnings_list])
 
    # ── Saturday Movie Night TMDB Recommendation ──
    movie_info_str = ""
    if weekday_name.lower() == "saturday":
        try:
            logger.info("Today is Saturday. Fetching TMDB recommendation for schedule block...")
            movie_genres = user_profile.get("movie_genres", ["action", "sci-fi", "thriller"])
            watched_ids = user_profile.get("watched_movie_ids", [])
            max_runtime = user_profile.get("movie_max_runtime", 150)
            
            movies = await search_movies_func(
                genres=movie_genres,
                max_runtime=max_runtime,
                exclude_ids=watched_ids,
                min_rating=7.0
            )
            if movies:
                top_movie = movies[0]
                movie_title = top_movie.get("title", "Curated Movie")
                movie_rating = top_movie.get("rating", "7.0")
                movie_overview = top_movie.get("overview", "A special movie pick for you.")
                movie_info_str = f"\n🎥 SATURDAY MOVIE NIGHT MANDATE:\nYou MUST schedule a movie block at exactly 21:30 titled 'Movie Night: {movie_title}' (type: ent, duration: 120 min) with notes: '⭐ {movie_rating}/10 - {movie_overview[:120]}...'"
                logger.info(f"Top Saturday movie found: {movie_title}")
            else:
                movie_info_str = "\n🎥 SATURDAY MOVIE NIGHT MANDATE:\nYou MUST schedule a movie block at exactly 21:30 titled 'Movie Night' (type: ent, duration: 120 min) with notes: 'Popcorn time! Curated by your AI.'"
        except Exception as e:
            logger.error(f"Error fetching Saturday movie recommendation: {e}")
            movie_info_str = "\n🎥 SATURDAY MOVIE NIGHT MANDATE:\nYou MUST schedule a movie block at exactly 21:30 titled 'Movie Night' (type: ent, duration: 120 min) with notes: 'Popcorn time! Curated by your AI.'"
 
    pending_tasks_str = ""
    for t in sorted(pending_tasks, key=lambda x: x.get("due_date", "9999-12-31")):
        pending_tasks_str += f"- {t.get('title')} (Type: {t.get('type')}, Priority: {t.get('priority')}, Due: {t.get('due_date')})\n"
    if not pending_tasks_str:
        pending_tasks_str = "No pending tasks."
        
    recent_notes_str = ""
    for n in recent_notes:
        note_type = n.get('type', 'text')
        recent_notes_str += f"- [{note_type}] {n.get('title')} (Subject: {n.get('subject')}, Tags: {', '.join(n.get('tags', []))}). AI Summary: {n.get('ai_summary')}\n"
    if not recent_notes_str:
        recent_notes_str = "No recent notes."
        
    feedback = state.get("human_feedback") or {}
    if feedback and not feedback.get("approved"):
        edits = feedback.get("edits", [])
        feedback_str = "Apply these edits:\n"
        for e in edits:
            action = e.get("action")
            block = e.get("block")
            if action == "move":
                feedback_str += f"- Move block '{block}' to new time {e.get('new_time')}\n"
            elif action == "remove":
                feedback_str += f"- Remove block '{block}'\n"
            elif action == "add":
                block_details = e.get("block")
                feedback_str += f"- Add block: {block_details}\n"
    else:
        feedback_str = "No previous feedback."
        
    # Format the SCHEDULE_PROMPT
    prompt = f"""You are a personal AI scheduler. Build a realistic, personalized schedule for {date} ({weekday_name}).
 
═══════════════════════════════════════════
FIXED CLASSES — DO NOT MOVE OR REMOVE
═══════════════════════════════════════════
{timetable_classes}
{movie_info_str}
 
═══════════════════════════════════════════
PERSONAL FIXED BLOCKS — NON-NEGOTIABLE
(treat exactly like college classes, never move)
═══════════════════════════════════════════
{non_negotiable_str}
 
═══════════════════════════════════════════
PERSONAL FLEXIBLE BLOCKS — schedule if space allows
═══════════════════════════════════════════
{flexible_str}
 
═══════════════════════════════════════════
PENDING TASKS DUE SOON
═══════════════════════════════════════════
{pending_tasks_str}
 
═══════════════════════════════════════════
RECENT NOTES CONTEXT
═══════════════════════════════════════════
{recent_notes_str}
 
═══════════════════════════════════════════
ACADEMIC EVENTS DUE SOON
═══════════════════════════════════════════
{academic_events_str}
 
═══════════════════════════════════════════
ACADEMIC CALENDAR
═══════════════════════════════════════════
CIE dates: {json.dumps(cie_dates)}
SEE starts: {see_start}
Next holiday: {next_holiday}
 
═══════════════════════════════════════════
SYLLABUS RISK ASSESSMENT
═══════════════════════════════════════════
{risk_assessment_str}
 
═══════════════════════════════════════════
USER PROFILE
═══════════════════════════════════════════
Role: {role}
Wake time target: {wake_time} (actual avg: {actual_wake_avg})
Sleep target: {sleep_target}
Peak focus time: {peak_focus_time}
Study block length: {study_block_pref}
Gym days: {gym_days} at {gym_time}
Mode: {current_mode}
 
═══════════════════════════════════════════
BEHAVIORAL PATTERNS (learned)
═══════════════════════════════════════════
{patterns_str}
Example: gym_monday: 0.8 means they skip gym 80% of Mondays — don't schedule gym today if it's Monday.
 
═══════════════════════════════════════════
USER FEEDBACK ON PREVIOUS PROPOSAL
═══════════════════════════════════════════
{feedback_str}
(Apply these edits exactly. If empty, ignore.)
 
═══════════════════════════════════════════
RULES
═══════════════════════════════════════════
1. Never schedule two blocks at the same time.
2. Never schedule over non-negotiable personal blocks. They must remain exactly at their fixed time.
3. Add 15-min buffer before every Fixed Class or Non-Negotiable Personal Block for preparation/commute.
4. Prioritize tasks by deadline proximity, not just priority label.
5. Never schedule deep study within 30 min of a meal.
6. If today is a gym day AND skip_probability < 0.6, include gym. If skip_probability > 0.6, either omit it or append '(you often skip this)' to the block's notes.
7. In weekend mode, on Saturdays you must include the Saturday movie night block at exactly 21:30.
8. Always end with a sleep block at sleep_target.
9. Apply these Scheduling Rules for Academic Events (THESE MANDATORILY OVERRIDE GENERAL PREFERENCES SUCH AS STUDY BLOCK LENGTH):
   - CIE within 7 days → You MUST add a 2hr (120 min) study block daily for that subject.
   - CIE within 3 days → You MUST add a 3hr (180 min) study block daily for that subject (do NOT split it, and do NOT use the user's standard 90 min preference). The block's notes or title MUST contain the word "CRITICAL" and the phrase "CIE prep (CRITICAL)".
   - Lab record due tomorrow → You MUST add a 1.5hr (90 min) lab writing block this evening (post 17:00).
   - Assignment due tomorrow → You MUST add a 1.5hr (90 min) study block today to complete it, and mark the notes or title as "Urgent".
   - SEE within 14 days → You MUST restructure the entire day's schedule to focus heavily on revision.
 
Return ONLY a JSON object of this exact shape:
{{
  "blocks": [
    {{
      "time": "HH:MM",
      "title": "Block title",
      "type": "work|study|personal|health|rest|ent",
      "duration_min": 60,
      "notes": "Context or reminder text shown to user"
    }}
  ]
}}
"""

    logger.info("Generated schedule prompt for date: %s", date)

    proposed_schedule = None

    # 1. Bypass LLM for test users to avoid API rate limit depletion during automated runs
    if bypass_llm(user_id):
        logger.info("Bypassing LLM call for test user: %s", user_id)
        proposed_schedule = generate_rule_based_schedule(
            user_id=user_id,
            date=date,
            weekday_name=weekday_name,
            is_weekend=is_weekend,
            classes=classes,
            personal_fixed_blocks=personal_fixed_blocks,
            academic_events=academic_events,
            academic_calendar=academic_calendar,
            user_profile=user_profile,
            pending_tasks=pending_tasks,
            recent_notes=recent_notes
        )
    else:
        # Schema-constrained call: the model cannot emit malformed JSON, and
        # every block is validated (time/title/duration/type) before use.
        proposal = await astructured(prompt, ScheduleProposal, user_id=user_id)

        if proposal is None or not proposal.blocks:
            logger.error(
                "All LLM models failed or returned no blocks. Building rule-based fallback."
            )
            proposed_schedule = generate_rule_based_schedule(
                user_id=user_id,
                date=date,
                weekday_name=weekday_name,
                is_weekend=is_weekend,
                classes=classes,
                personal_fixed_blocks=personal_fixed_blocks,
                academic_events=academic_events,
                academic_calendar=academic_calendar,
                user_profile=user_profile,
                pending_tasks=pending_tasks,
                recent_notes=recent_notes
            )
        else:
            proposed_schedule = [block.model_dump() for block in proposal.blocks]
            logger.info(
                "Successfully parsed proposed schedule with %d blocks.", len(proposed_schedule)
            )

    return {
        "proposed_schedule": proposed_schedule,
        "needs_clarification": False
    }

async def propose_schedule(state: AgentState) -> Dict[str, Any]:
    """Node where the schedule is proposed. Graph interrupts BEFORE this node."""
    logger.info("Node: propose_schedule")
    
    schedule_blocks = state.get("proposed_schedule") or []
    msg = "Here is your proposed schedule for today:\n\n"
    for b in schedule_blocks:
        msg += f"⏰ **{b.get('time')}** - **{b.get('title')}** ({b.get('type')}, {b.get('duration_min')} min)\n"
        if b.get("notes"):
            msg += f"   *Notes: {b.get('notes')}*\n"
    msg += "\nWould you like to approve this schedule, or do you have any changes?"
    
    return {
        "messages": [{"role": "assistant", "content": msg}]
    }

async def commit_schedule(state: AgentState, store: BaseStore) -> Dict[str, Any]:
    """Commits the approved schedule to the user's calendars/database."""
    logger.info("Node: commit_schedule")
    
    approved = state.get("proposed_schedule") or []
    user_id = state.get("user_id", "default_user")
    date = state.get("intent_date") or "2026-06-08"
    
    # Write to Firestore using the write_calendar tool
    logger.info(f"Writing {len(approved)} approved schedule blocks to Firestore.")
    await write_calendar_func(user_id=user_id, date=date, blocks=approved)
    
    # Call schedule_push to register push notifications in Firestore
    await schedule_push_func(user_id=user_id, date=date, blocks=approved)
    
    # Store behavioral signals if any feedback edits were applied
    feedback = state.get("human_feedback") or {}
    if feedback and not feedback.get("approved"):
        edits = feedback.get("edits", [])
        if edits:
            await store_preference_func(user_id=user_id, key="last_schedule_edits", value=edits, store=store)
            
    return {
        "approved_schedule": approved,
        "messages": [{"role": "assistant", "content": "Awesome! Your schedule has been saved and reminders are set."}]
    }

async def general_chat(state: AgentState, store: BaseStore) -> Dict[str, Any]:
    """Handles general chit-chat and queries not related to scheduling."""
    logger.info("Node: general_chat")
    
    user_id = state.get("user_id", "default_user")
    date = state.get("intent_date") or datetime.date.today().isoformat()
    
    # Load user profile, timetable, and pending tasks in parallel
    user_profile = {}
    classes = []
    weekday = "today"
    academic_events = []
    tasks = []
    
    try:
        user_profile = await sync_and_load_user_profile(store, user_id)
        db = firestore_async.client()
        
        schedule_info, task_queue = await asyncio.gather(
            fetch_timetable_func(user_id=user_id, date=date),
            db.collection("tasks").document(user_id).collection("items").get()
        )
        
        classes = schedule_info.get("classes") or []
        weekday = schedule_info.get("weekday", "today")
        academic_events = schedule_info.get("academic_events") or []
        
        for doc in task_queue:
            tdata = doc.to_dict()
            if not tdata.get("completed", False):
                tdata["id"] = doc.id
                tasks.append(tdata)
    except Exception as gather_err:
        logger.warning(f"general_chat failed to gather database context: {gather_err}")

    # Format the context for the LLM
    profile_summary = json.dumps(user_profile, indent=2) if user_profile else "No profile configured."
    classes_summary = "\n".join(
        f"- {c.get('subject')} at {c.get('time')} (Room {c.get('room', 'N/A')})" for c in classes
    ) if classes else "No classes scheduled."
    events_summary = "\n".join(
        f"- {e.get('title')} ({e.get('type')}) on {e.get('date') or e.get('due_date')}" for e in academic_events
    ) if academic_events else "No academic events."
    tasks_summary = "\n".join(
        f"- {t.get('title')} (Priority: {t.get('priority', 'medium')})" for t in tasks
    ) if tasks else "No pending tasks."

    messages = state.get("messages", [])
    history_parts = []
    for m in messages:
        if hasattr(m, "content"):
            role = "User" if getattr(m, "type", "human") == "human" else "Assistant"
            history_parts.append(f"{role}: {m.content}")
        elif isinstance(m, dict):
            role = "User" if m.get("role") == "user" else "Assistant"
            history_parts.append(f"{role}: {m.get('content', '')}")
    
    conversation = "\n".join(history_parts) or "User: Hello"
    
    chat_prompt = f"""You are a helpful personal AI assistant for students and working professionals.
You help with scheduling, studying, and productivity. Be concise, friendly, and helpful.

User Context:
- User ID: {user_id}
- Current Date/Query Date: {date} ({weekday})
- User Profile: {profile_summary}
- College Timetable for {weekday}:
{classes_summary}
- Academic Events:
{events_summary}
- Pending Tasks:
{tasks_summary}

Conversation so far:
{conversation}

Respond to the latest user message. If the user is asking about their classes, timetable, exams, or tasks, use the provided context to answer accurately. If they haven't uploaded a timetable/syllabus but are asking about it, guide them on how to upload a photo/document in the Academics tab."""
    
    # Free-text response: no schema, but still bounded retries + fallback.
    fallback_text = "I'm here to help you plan your day and stay productive! Try saying 'schedule my day' to get started."
    response_text = await atext(chat_prompt, user_id=user_id, fallback=fallback_text)
    if response_text == fallback_text:
        logger.warning("general_chat LLM call failed or was bypassed; using fallback response.")
        
    return {
        "messages": [{"role": "assistant", "content": response_text}]
    }

async def entertainment_node(state: AgentState, store: BaseStore) -> Dict[str, Any]:
    """Handles entertainment / movie recommendation requests.
    
    Triggers when:
      - intent == "entertainment_rec" (user asked explicitly), OR
      - It's Saturday and after 17:00 (auto-suggest)
    
    Loads user's movie_genres and watched_movie_ids from the LangGraph Store,
    calls the TMDB search_movies tool, and returns 3 formatted suggestions.
    """
    logger.info("Node: entertainment_node")
    
    user_id = state.get("user_id", "default_user")
    
    # ── Load user preferences from Store ──
    user_profile = await sync_and_load_user_profile(store, user_id)
    
    # Default genres if user hasn't set preferences
    movie_genres = user_profile.get("movie_genres", ["action", "sci-fi", "thriller"])
    watched_ids = user_profile.get("watched_movie_ids", [])
    max_runtime = user_profile.get("movie_max_runtime", 150)
    
    logger.info(f"Entertainment prefs: genres={movie_genres}, watched={len(watched_ids)}, max_runtime={max_runtime}")
    
    # ── Call TMDB ──
    movies = await search_movies_func(
        genres=movie_genres,
        max_runtime=max_runtime,
        exclude_ids=watched_ids,
        min_rating=7.0,
    )
    
    # ── Pick top 3 and format message ──
    top_picks = movies[:3]
    
    if not top_picks:
        return {
            "messages": [{"role": "assistant", "content": "I couldn't find any movie recommendations right now. Try again later or update your genre preferences!"}]
        }
    
    lines = ["🎬 **Here are 3 movie picks for tonight:**\n"]
    for i, m in enumerate(top_picks, 1):
        lines.append(f"**{i}. {m['title']}** ({m['release_year']})  ⭐ {m['rating']}/10")
        lines.append(f"   {m['overview']}")
        if m.get("poster_path"):
            lines.append(f"   🖼️ Poster: {m['poster_path']}")
        lines.append("")  # blank line between movies
    
    lines.append("Want me to add a movie night block to your schedule? Just say the word! 🍿")
    
    response_text = "\n".join(lines)
    
    return {
        "messages": [{"role": "assistant", "content": response_text}]
    }


async def location_trigger_node(state: AgentState, store: BaseStore) -> Dict[str, Any]:
    """Handles automatic context triggers when a user enters a geofenced area.
    
    Triggers immediate push notifications and returns custom chat responses:
    - Gym: Motivational ping and chat greeting.
    - College: Suggests switching schedule to study focus blocks.
    - Home: Shows evening routine summary and target sleep time.
    """
    logger.info("Node: location_trigger_node")
    user_id = state.get("user_id", "default_user")
    messages = state.get("messages", [])
    last_message = messages[-1].content if messages else ""
    last_msg_lower = last_message.lower()

    # Load user profile to check gym days, sleep preferences, etc.
    user_profile = await sync_and_load_user_profile(store, user_id)
    if not user_profile:
        user_profile = {}

    gym_days = user_profile.get("gym_days", ["Monday", "Wednesday", "Friday"])
    
    content = ""
    if "gym" in last_msg_lower:
        today_day_name = datetime.date.today().strftime("%A")
        is_gym_day = today_day_name in gym_days
        
        title = "Gym Time! 🏋️"
        body = "Let's crush this workout! Keep pushing hard."
        if not is_gym_day:
            body = "Extra credit workout! Let's get active!"
            
        await send_immediate_push(user_id, title, body)
        
        if is_gym_day:
            content = "Welcome to the Gym! Today is one of your scheduled gym days. Let's make it count! I've sent a motivational ping to your device."
        else:
            content = "Welcome to the Gym! I noticed you arrived for an extra workout today. Great dedication! I've sent a motivational ping to your device."
            
    elif "college" in last_msg_lower or "library" in last_msg_lower:
        await send_immediate_push(user_id, "Study Focus Mode 🎓", "Time to focus! Shall we start your study blocks?")
        content = "Welcome to College! I've detected your arrival at the campus/library. Would you like me to switch your schedule to study focus blocks now?"
        
    elif "home" in last_msg_lower:
        await send_immediate_push(user_id, "Welcome Home 🏡", "Time to unwind and prepare for your evening routine.")
        content = "Welcome back home! Excellent work today. Here is a summary of your evening routine:\n\n• 🍽️ Dinner: 19:30\n• 📚 Wind-down: 21:00\n• 💤 Target Sleep: 22:30\n\nHave a relaxing evening!"
        
    else:
        content = "I've detected a change in your location context. Let me know if you want to update your schedule!"

    return {
        "messages": [{"role": "assistant", "content": content}]
    }

async def save_user_info_node(state: AgentState, store: BaseStore, config: RunnableConfig = None) -> Dict[str, Any]:
    """Node that parses and saves incoming user schedule and academic details."""
    logger.info("Node: save_user_info_node")
    
    user_id = state.get("user_id", "default_user")
    messages = state.get("messages", [])
    last_message = messages[-1].content if messages else ""
    
    user_profile = await sync_and_load_user_profile(store, user_id)
    if not user_profile:
        user_profile = {}
    if "completed_discovery" not in user_profile:
        user_profile["completed_discovery"] = []
        
    last_topic = user_profile.get("last_discovery_topic")
    last_msg_lower = last_message.lower().strip()
    
    # Check if this is a negation response to a discovery question
    is_negation = last_msg_lower in ["no", "none", "nothing", "not yet", "no i don't", "no, thanks", "no thanks"]
    
    db = firestore_async.client()
    
    if is_negation and last_topic:
        # User is skipping/declining the topic asked in the last discovery question
        completed = user_profile.get("completed_discovery", [])
        if last_topic not in completed:
            completed.append(last_topic)
            user_profile["completed_discovery"] = completed
            await save_user_profile(store, user_id, user_profile)
            await db.collection("users").document(user_id).set({"completed_discovery": completed}, merge=True)
            
        # Clear last asked topic
        user_profile["last_discovery_topic"] = None
        await save_user_profile(store, user_id, user_profile)
        await db.collection("users").document(user_id).set({"last_discovery_topic": None}, merge=True)
        
        confirm_msg = f"Got it — I won't ask you about your {last_topic.replace('_', ' ')} again."
        return {
            "messages": [{"role": "assistant", "content": confirm_msg}],
            "asked_discovery_this_session": False
        }

    parsed = await parse_user_info(last_message, user_profile, user_id)
    info_type = parsed.get("type", "general")
    info_data = parsed.get("data") or {}
    
    thing = "information"
    
    if info_type == "task":
        # Save to tasks/{uid}/items
        task_ref = db.collection("tasks").document(user_id).collection("items")
        await task_ref.add({
            "title": info_data.get("title", "Task"),
            "type": info_data.get("type", "reminder"),
            "due_date": info_data.get("due_date", datetime.date.today().isoformat()),
            "priority": info_data.get("priority", "medium"),
            "completed": False,
            "subject": info_data.get("subject"),
            "created_at": datetime.datetime.utcnow().isoformat()
        })
        
        completed = user_profile.get("completed_discovery", [])
        if "academic_events" not in completed:
            completed.append("academic_events")
            user_profile["completed_discovery"] = completed
            await save_user_profile(store, user_id, user_profile)
            await db.collection("users").document(user_id).set({"completed_discovery": completed}, merge=True)
            
        thing = info_data.get("title", "task")

    elif info_type == "note":
        # Save to notes/{uid}/items
        note_body = info_data.get("body") or last_message
        summary_data = await summarize_note_content(note_body, user_id=user_id)
        
        note_ref = db.collection("notes").document(user_id).collection("items")
        await note_ref.add({
            "title": info_data.get("title") or summary_data.get("subject") or "Note",
            "body": note_body,
            "ai_summary": summary_data.get("summary", ""),
            "subject": summary_data.get("subject", "General"),
            "tags": info_data.get("tags") or summary_data.get("tags") or ["general"],
            "created_at": datetime.datetime.utcnow().isoformat()
        })
        
        thing = info_data.get("title", "note")

    elif info_type == "academic_event":
        # Save to academic_events/{uid}/items
        event_ref = db.collection("academic_events").document(user_id).collection("items")
        await event_ref.add(info_data)
        
        # Mark topic completed
        completed = user_profile.get("completed_discovery", [])
        if "academic_events" not in completed:
            completed.append("academic_events")
            user_profile["completed_discovery"] = completed
            await save_user_profile(store, user_id, user_profile)
            await db.collection("users").document(user_id).set({"completed_discovery": completed}, merge=True)
            
        thing = info_data.get("title") or info_data.get("subject", "academic event")
        
    elif info_type == "personal_block":
        # Save to personal_schedule/{uid}
        doc_ref = db.collection("personal_schedule").document(user_id)
        doc = await doc_ref.get()
        existing_blocks = (doc.to_dict() or {}).get("fixed_blocks", []) if doc.exists else []
        
        new_blocks = info_data if isinstance(info_data, list) else [info_data]
        for nb in new_blocks:
            if nb and isinstance(nb, dict):
                nb_title = nb.get("title", "Routine")
                nb_time = nb.get("time", "08:00")
                nb_days = nb.get("days", ["daily"])
                nb_duration = nb.get("duration_min", 30)
                nb_type = nb.get("type", "personal")
                nb_non_neg = nb.get("non_negotiable")
                if nb_non_neg is None:
                    nb_non_neg = True
                if any(x in nb_title.lower() for x in ["meditat", "sleep"]):
                    nb_non_neg = True
                
                block_to_save = {
                    "title": nb_title,
                    "time": nb_time,
                    "days": nb_days,
                    "duration_min": nb_duration,
                    "type": nb_type,
                    "non_negotiable": nb_non_neg
                }
                
                if not any(eb.get("title") == block_to_save["title"] and eb.get("time") == block_to_save["time"] for eb in existing_blocks):
                    existing_blocks.append(block_to_save)
                    
        await doc_ref.set({"fixed_blocks": existing_blocks}, merge=True)
        
        # Save personal blocks in store profile as well
        user_profile["personal_blocks"] = existing_blocks
        await save_user_profile(store, user_id, user_profile)
        
        # Mark topic completed
        completed = user_profile.get("completed_discovery", [])
        if "personal_blocks" not in completed:
            completed.append("personal_blocks")
            user_profile["completed_discovery"] = completed
            await save_user_profile(store, user_id, user_profile)
            await db.collection("users").document(user_id).set({"completed_discovery": completed}, merge=True)
            
        thing = new_blocks[0].get("title", "routine") if new_blocks else "routine"
        
    elif info_type == "college_info":
        college_name = info_data.get("college")
        branch = info_data.get("branch")
        semester = info_data.get("semester")
        scheme = info_data.get("scheme")
        
        updates = {}
        if college_name:
            user_profile["college"] = college_name
            updates["college"] = college_name
        if branch:
            user_profile["branch"] = branch
            updates["branch"] = branch
        if semester:
            user_profile["semester"] = semester
            updates["semester"] = semester
        if scheme:
            user_profile["scheme"] = scheme
            updates["scheme"] = scheme
            
        if updates:
            await save_user_profile(store, user_id, user_profile)
            await db.collection("users").document(user_id).set(updates, merge=True)
            
        background_tasks = config.get("background_tasks") if config else None
        
        if college_name:
            # Trigger college_scraper
            from app.tasks.college_scraper import run_college_calendar_scraper_task
            if background_tasks:
                background_tasks.add_task(run_college_calendar_scraper_task, user_id, college_name)
            else:
                asyncio.create_task(run_college_calendar_scraper_task(user_id, college_name))
                
        if branch and semester and scheme:
            # Trigger syllabus auto-fetch task
            from app.tasks.college_scraper import fetch_syllabus_for_student_task
            if background_tasks:
                background_tasks.add_task(fetch_syllabus_for_student_task, user_id, branch, scheme, semester)
            else:
                asyncio.create_task(fetch_syllabus_for_student_task(user_id, branch, scheme, semester))
                
        # Mark topic completed
        completed = user_profile.get("completed_discovery", [])
        if "college" not in completed:
            completed.append("college")
            user_profile["completed_discovery"] = completed
            await save_user_profile(store, user_id, user_profile)
            await db.collection("users").document(user_id).set({"completed_discovery": completed}, merge=True)
            
        thing = college_name or "college"
        
    elif info_type == "syllabus_update":
        subject = info_data.get("subject")
        if subject:
            doc_ref = db.collection("syllabus").document(user_id).collection("subjects").document(subject)
            doc = await doc_ref.get()
            if doc.exists:
                existing_data = doc.to_dict() or {}
                existing_units = existing_data.get("units", [])
                new_units = info_data.get("units", [])
                for nu in new_units:
                    unit_num = nu.get("number")
                    found = False
                    for i, eu in enumerate(existing_units):
                        if eu.get("number") == unit_num:
                            existing_units[i] = {**eu, **nu}
                            found = True
                            break
                    if not found:
                        existing_units.append(nu)
                info_data["units"] = existing_units
            await doc_ref.set(info_data, merge=True)
            
        thing = f"{subject} syllabus"
        
    elif info_type == "timetable":
        await db.collection("timetables").document(user_id).set(info_data, merge=True)
        user_profile["timetable"] = info_data
        await save_user_profile(store, user_id, user_profile)
        
        # Mark topic completed
        completed = user_profile.get("completed_discovery", [])
        if "timetable" not in completed:
            completed.append("timetable")
            user_profile["completed_discovery"] = completed
            await save_user_profile(store, user_id, user_profile)
            await db.collection("users").document(user_id).set({"completed_discovery": completed}, merge=True)
            
        thing = "timetable"
        
    else:
        confirm_msg = "Got it — I've noted that down."
        user_profile["last_discovery_topic"] = None
        await save_user_profile(store, user_id, user_profile)
        await db.collection("users").document(user_id).set({"last_discovery_topic": None}, merge=True)
        return {
            "messages": [{"role": "assistant", "content": confirm_msg}],
            "asked_discovery_this_session": False
        }
        
    user_profile["last_discovery_topic"] = None
    await save_user_profile(store, user_id, user_profile)
    await db.collection("users").document(user_id).set({"last_discovery_topic": None}, merge=True)
    
    confirm_msg = f"Got it — saved your {thing}. I'll factor this into your schedule from now on."
    return {
        "messages": [{"role": "assistant", "content": confirm_msg}],
        "asked_discovery_this_session": False
    }

async def check_clarification_node(state: AgentState, store: BaseStore) -> Dict[str, Any]:
    """Node that checks for gaps using identify_missing_slots and appends one discovery question if needed."""
    logger.info("Node: check_clarification_node")
    
    # Check if we have already asked this session
    if state.get("asked_discovery_this_session", False):
        logger.info("Discovery question already asked this session.")
        return {}
        
    user_id = state.get("user_id", "default_user")
    user_profile = await sync_and_load_user_profile(store, user_id)
    if not user_profile:
        user_profile = {}
        
    db = firestore_async.client()
    
    # Check if timetable is populated in user profile or Firestore
    timetable_param = {}
    if user_profile.get("timetable"):
        timetable_param = user_profile["timetable"]
    else:
        try:
            timetable_doc = await db.collection("timetables").document(user_id).get()
            if timetable_doc.exists:
                timetable_param = timetable_doc.to_dict() or {}
        except Exception:
            pass
            
    # Fetch academic events
    academic_events = state.get("academic_events")
    if not academic_events:
        try:
            events_snapshot = await db.collection("academic_events").document(user_id).collection("items").get()
            academic_events = [doc.to_dict() for doc in events_snapshot]
        except Exception:
            academic_events = []
            
    # Call identify_missing_slots
    gap = identify_missing_slots(
        profile=user_profile,
        timetable=timetable_param,
        academic_events=academic_events
    )
    
    if not gap:
        logger.info("No gaps found.")
        return {}
        
    topic = gap["topic"]
    question = gap["question"]
    logger.info(f"Gap identified: topic={topic}, question='{question}'")
    
    # Save the last asked topic so we can handle a "no" answer correctly
    user_profile["last_discovery_topic"] = topic
    await save_user_profile(store, user_id, user_profile)
    await db.collection("users").document(user_id).set({"last_discovery_topic": topic}, merge=True)
    
    from langchain_core.messages import AIMessage
    return {
        "messages": [AIMessage(content=f"By the way, {question}")],
        "asked_discovery_this_session": True
    }


