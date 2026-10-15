"""Prompt assembly rules for Gemini implicit caching.

Gemini caches the *prefix* of a request automatically — no setup, no cache
objects — but only under two conditions:

1. The prefix must exceed a per-model token floor. For ``gemini-3.8-flash``
   that floor is **4,096 tokens** (it was 1,024 on 2.5 Flash; the newer models
   are higher). Cached tokens then bill at roughly 10% of the input rate.
2. The prefix must repeat within a few minutes (the cache TTL is ~3-5 min).

Both conditions are decided entirely by prompt *layout*, which is what this
module is for. The invariant:

    static instructions -> stable context -> day-specific context -> volatile

Putting anything that changes per-request at the top — a date, a user id, a
timestamp — invalidates the whole prefix and silently disables caching. The
``build_schedule`` prompt used to open with the date, so it never cached once.

Use :func:`build_prompt` so the ordering is enforced in one place, and
:func:`estimate_tokens` to check a prompt clears the floor before you rely on
caching at all.
"""

from typing import Optional

# ~4 characters per token is the usual English heuristic. It is deliberately
# rough: this is only used to log "would this prompt even qualify for caching",
# never for billing.
CHARS_PER_TOKEN = 4

# Minimum prompt tokens for implicit caching, per model family. Keep in sync
# with https://ai.google.dev/gemini-api/docs/caching
CACHE_MIN_TOKENS = {
    "gemini-3.8-flash": 4096,
    "gemini-3.5-flash-lite": 4096,
    "gemini-3-flash": 4096,
    # 2.5 series had a much lower floor; kept for reference / rollback.
    "gemini-2.5-flash": 1024,
    "gemini-2.5-pro": 4096,
}
DEFAULT_CACHE_MIN_TOKENS = 4096


def estimate_tokens(text: str) -> int:
    """Rough token count, good enough to reason about the cache floor."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def build_prompt(*sections: str, separator: str = "\n\n") -> str:
    """Join prompt sections in cache-friendly order.

    Pass them static-first. Empty sections are dropped so a missing context
    block does not leave a hole that shifts the prefix for later sections.
    """
    parts = [s.strip("\n") for s in sections if s and s.strip()]
    return separator.join(parts)


def qualifies_for_cache(prompt: str, model: str) -> bool:
    """True when ``prompt`` is long enough to be eligible for implicit caching."""
    floor = CACHE_MIN_TOKENS.get(model, DEFAULT_CACHE_MIN_TOKENS)
    return estimate_tokens(prompt) >= floor


def cache_margin(prompt: str, model: str) -> int:
    """Tokens above (positive) or below (negative) the model's cache floor."""
    floor = CACHE_MIN_TOKENS.get(model, DEFAULT_CACHE_MIN_TOKENS)
    return estimate_tokens(prompt) - floor


# ── Static instruction blocks ───────────────────────────────────────────────
# These must never interpolate per-request values. Anything that varies belongs
# in a later section. Keep them module-level constants so the identity of the
# prefix is obvious at a glance.

SCHEDULE_RULES = """RULES
1. Never schedule two blocks at the same time.
2. Never schedule over non-negotiable personal blocks. They must remain exactly at their fixed time.
3. Add 15-min buffer before every Fixed Class or Non-Negotiable Personal Block for preparation/commute.
4. Prioritize tasks by deadline proximity, not just priority label.
5. Never schedule deep study within 30 min of a meal.
6. If today is a gym day AND skip_probability < 0.6, include gym. If skip_probability > 0.6, either omit it or append '(you often skip this)' to the block's notes.
7. In weekend mode, on Saturdays you must include the Saturday movie night block at exactly 21:30.
8. Always end with a sleep block at sleep_target.
9. Apply these Scheduling Rules for Academic Events (THESE MANDATORILY OVERRIDE GENERAL PREFERENCES SUCH AS STUDY BLOCK LENGTH):
   - CIE within 7 days: add a 2hr (120 min) study block daily for that subject.
   - CIE within 3 days: add a 3hr (180 min) study block daily for that subject (do NOT split it, and do NOT use the user's standard 90 min preference). The block's notes or title MUST contain the word "CRITICAL" and the phrase "CIE prep (CRITICAL)".
   - Lab record due tomorrow: add a 1.5hr (90 min) lab writing block this evening (post 17:00).
   - Assignment due tomorrow: add a 1.5hr (90 min) study block today to complete it, and mark the notes or title as "Urgent".
   - SEE within 14 days: restructure the entire day's schedule to focus heavily on revision."""

SCHEDULE_OUTPUT_SHAPE = """Return an object of this exact shape:
{
  "blocks": [
    {
      "time": "HH:MM",
      "title": "Block title",
      "type": "work|study|personal|health|rest|ent",
      "duration_min": 60,
      "notes": "Context or reminder text shown to user"
    }
  ]
}"""

SCHEDULE_ROLE = "You are a personal AI scheduler. Build a realistic, personalized schedule for the date given at the end of this prompt."

EXTRA_ITEMS_PROMPT = """Read the user's message and list any concrete activities they mention
that are not already covered by their timetable, their classes, their pending tasks,
or their saved routines.

Include things like: a meeting, an appointment, an errand, a call to make, a project
session, a visit, a deadline they are working on, a class or event they attend.

Do NOT include: things they are merely thinking about, hypotheticals, or vague plans
with no activity. Do NOT include anything already in their timetable or task list.

For each item give a realistic duration in minutes (15-240), a type
(work|study|personal|health|rest|ent), a priority, and optionally a soft preference
for when as two 24-hour times like "14:00-17:00". Do not choose an exact start time --
the scheduler decides that.

If the message mentions nothing concrete, return an empty list.

User message: {message}"""


CLASSIFY_INSTRUCTIONS = """You are an intent classifier for a personal AI academic scheduler.
Classify the user's latest message into exactly one intent:
- "schedule_build": the user wants to build/generate/plan a schedule for a day or week (e.g. "schedule my day", "plan today", "what's my schedule").
- "schedule_edit": the user wants to edit/change a proposed or existing schedule (e.g. "move gym to 7pm", "remove the study block").
- "entertainment_rec": the user wants movie, show, or entertainment recommendations.
- "location_trigger": automated geofence arrival messages (e.g. "arrived at gym", "arrived at college").
- "info_provided": the user is sharing/entering NEW information to the system - college name, course details, timetable, syllabus, academic events (CIE/SEE/viva/record), or personal routines (e.g. "I study at RVCE", "CIE-1 is on June 20", "I have a lab record due tomorrow", "I go to gym at 6pm daily", "I finished unit 2", "no", "none", "nothing this week", "Bti", "BTI associated with vtu"). This is only for inputting/saving new data.
- "general": general chit-chat, greetings, or questions, including requests to view, retrieve, check, or show existing classes, timetable, calendar, or tasks (e.g. "what are my classes on Monday", "show my timetable", "give my Monday college timetable", "hi", "how are you", "who are you").

Also report the scope ("day" or "week") and the date the message refers to.
Resolve relative dates to an absolute YYYY-MM-DD date using the date given at the end
of this prompt. If no date is mentioned or implied, leave the date empty."""

CHAT_INSTRUCTIONS = """Respond to the latest user message.
If the user is asking about their classes, timetable, exams, or tasks, use the provided
context to answer accurately. If they have not uploaded a timetable/syllabus but are
asking about it, guide them on how to upload a photo/document in the Academics tab."""


def section(title: str, body: str, rule: Optional[str] = None) -> str:
    """Render one titled context block.

    ``rule`` is the short instruction that belongs with the data, kept adjacent
    so the model reads the constraint next to the facts it constrains.
    """
    header = "═" * 47
    parts = [f"{header}\n{title}\n{header}"]
    if rule:
        parts.append(f"({rule})")
    parts.append(body.strip())
    return "\n".join(parts)


def log_prompt_shape(name: str, prompt: str, model: str) -> None:
    """Log whether a prompt clears the implicit-cache floor.

    Called from the LLM layer rather than the call sites so the numbers land
    next to the token counts in the same log line.
    """
    from app.agents.llm import logger  # local import avoids a cycle

    margin = cache_margin(prompt, model)
    if margin >= 0:
        logger.info(
            "prompt[%s] ~%d tokens (%d above the %s cache floor)",
            name, estimate_tokens(prompt), margin, model,
        )
    else:
        logger.info(
            "prompt[%s] ~%d tokens (%d BELOW the %s cache floor - will not cache)",
            name, estimate_tokens(prompt), -margin, model,
        )
