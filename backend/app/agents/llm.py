"""LLM clients and the single structured-call entry point.

Every LLM call in the backend goes through :func:`astructured` (JSON outputs)
or :func:`atext` (free-text outputs). Both are:

* **Schema-constrained** - ``with_structured_output(method="json_mode")`` makes
  Gemini emit JSON that must match the Pydantic schema, so the regex-stripping
  / ``json.loads`` failure class is gone.
* **Tiered** - callers pick a ``tier``. ``"router"`` calls go to the cheap,
  fast model (Flash-Lite) because they only choose one of a few labels;
  ``"planner"`` calls go to the workhorse (3.8 Flash) because their output
  quality drives the schedule the user actually sees.
* **Bounded** - a small number of attempts with backoff, then a caller-supplied
  deterministic fallback. Never raises.
* **Bypassable** - ``test_*`` user ids (and ``DEMO_MODE``) skip the network
  entirely so automated runs never burn quota.

Clients are built **lazily**: a missing API key must not prevent the app from
importing. Without a key the chain is empty and every call degrades to its
fallback, which is exactly the behaviour a fresh clone should have.

Implicit caching note
--------------------
Gemini caches the *prefix* of a request automatically (free, no setup), but
only above a per-model token floor and only for a few minutes. The rules that
follow from that are enforced by :func:`build_prompt` in
``app/agents/prompts.py``: static instructions first, stable context next, and
the volatile user message dead last.
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Type, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger("app.agents.llm")

T = TypeVar("T", bound=BaseModel)

# ── Tiers ───────────────────────────────────────────────────────────────────

ROUTER = "router"
PLANNER = "planner"

TIER_MODEL = {
    ROUTER: settings.ROUTING_MODEL,
    PLANNER: settings.PRIMARY_MODEL,
}

_tier_chains: Dict[str, Optional[List[BaseChatModel]]] = {}


def _build_gemini(model: str) -> Optional[BaseChatModel]:
    if not settings.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not set; model %s is unavailable.", model)
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        api_key=settings.GEMINI_API_KEY,
        max_retries=settings.LLM_MAX_RETRIES,
    )


def _build_groq() -> Optional[BaseChatModel]:
    if not settings.GROQ_API_KEY:
        logger.info("GROQ_API_KEY is not set; the secondary model is unavailable.")
        return None
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=settings.SECONDARY_MODEL,
        api_key=settings.GROQ_API_KEY,
        max_retries=settings.LLM_MAX_RETRIES,
    )


def get_tier_chain(tier: str = PLANNER) -> List[BaseChatModel]:
    """Primary model for ``tier`` -> Groq fallback.

    The fallback is shared between tiers: if Gemini is down entirely, a slower
    deterministic answer still beats a hard failure.
    """
    if tier not in _tier_chains:
        chain: List[BaseChatModel] = []
        for builder in (lambda: _build_gemini(TIER_MODEL[tier]), _build_groq):
            try:
                model = builder()
            except Exception as exc:  # noqa: BLE001 - a broken client must not break import
                logger.error("Failed to construct LLM client for tier %s: %s", tier, exc)
                model = None
            if model is not None:
                chain.append(model)
        _tier_chains[tier] = chain
        logger.info(
            "LLM chain ready for tier '%s': %s",
            tier,
            " -> ".join(type(m).__name__ for m in chain) or "EMPTY (fallbacks only)",
        )
    return _tier_chains[tier]


# Backwards-compatible name: the planner chain (used to be "the" chain).
def get_llm_chain() -> List[BaseChatModel]:
    return get_tier_chain(PLANNER)


def _runnable(chain: List[BaseChatModel]):
    """Attach fallbacks when there is more than one model."""
    if len(chain) > 1:
        return chain[0].with_fallbacks(chain[1:])
    return chain[0]


# Backwards-compatible module attribute: the first available model (or None).
def __getattr__(name: str):
    if name == "llm":
        chain = get_llm_chain()
        return _runnable(chain) if chain else None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ── Observability ───────────────────────────────────────────────────────────

_tracing_ready = False


def setup_tracing() -> bool:
    """Enable LangSmith tracing when a key is configured.

    Must run before the first LLM call. Returns whether tracing is active.
    Without a key we still log token/cache-hit counts locally, so implicit
    caching stays measurable either way.
    """
    global _tracing_ready
    if _tracing_ready:
        return bool(os.environ.get("LANGCHAIN_TRACING_V2"))

    if not settings.LANGCHAIN_API_KEY:
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        os.environ.pop("LANGCHAIN_API_KEY", None)
        logger.info(
            "LangSmith tracing is off (no LANGCHAIN_API_KEY). "
            "Token/cache-hit counts are logged locally instead."
        )
        _tracing_ready = True
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true" if settings.LANGCHAIN_TRACING_V2 else "false"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    logger.info(
        "LangSmith tracing %s (project=%s).",
        "enabled" if settings.LANGCHAIN_TRACING_V2 else "configured but disabled",
        settings.LANGCHAIN_PROJECT,
    )
    _tracing_ready = True
    return settings.LANGCHAIN_TRACING_V2


def _log_usage(label: str, tier: str, elapsed_ms: float, response) -> None:
    """Log latency plus token/cache-hit counts from the provider response.

    Cache hits are the only way to confirm implicit caching is actually firing,
    and they are invisible unless you surface them.
    """
    usage = getattr(response, "usage_metadata", None) or {}
    details = usage.get("input_token_details") or {}
    prompt_tokens = usage.get("input_tokens")
    cache_read = details.get("cache_read")
    if prompt_tokens is None:
        logger.info("%s tier=%s %.0fms (no usage metadata)", label, tier, elapsed_ms)
        return

    cached = cache_read or 0
    pct = (cached / prompt_tokens * 100) if prompt_tokens else 0.0
    logger.info(
        "%s tier=%s %.0fms prompt_tokens=%d cached=%d (%.0f%%) output=%d",
        label,
        tier,
        elapsed_ms,
        prompt_tokens,
        cached,
        pct,
        usage.get("output_tokens") or 0,
    )


# ── Structured / text entry points ──────────────────────────────────────────

# Cache of (schema, tier) -> structured runnable, so we don't rebuild per call.
_structured_cache: dict = {}


def _structured(schema: Type[T], tier: str = PLANNER):
    key = (schema, tier)
    cached = _structured_cache.get(key)
    if cached is None:
        chain = get_tier_chain(tier)
        if not chain:
            raise RuntimeError(
                "No LLM client is configured (set GEMINI_API_KEY); "
                f"cannot run structured call for {schema.__name__} on tier '{tier}'"
            )
        cached = _runnable(chain).with_structured_output(schema, method="json_mode")
        _structured_cache[key] = cached
    return cached


def bypass_llm(user_id: Optional[str] = None) -> bool:
    """True when LLM calls should be skipped (test users / demo mode)."""
    if settings.DEMO_MODE:
        return True
    return bool(user_id) and user_id.startswith("test_")


async def astructured(
    prompt: str,
    schema: Type[T],
    *,
    user_id: Optional[str] = None,
    fallback: Optional[T] = None,
    attempts: Optional[int] = None,
    tier: str = PLANNER,
    label: Optional[str] = None,
) -> Optional[T]:
    """Call the LLM and return a validated ``schema`` instance.

    Returns ``fallback`` (possibly ``None``) instead of raising, so callers
    always get a usable value. ``fallback`` is returned immediately for test
    users and demo mode without touching the network.

    ``tier`` selects the model: :data:`ROUTER` for cheap label-picking calls,
    :data:`PLANNER` (default) for calls whose output quality matters.
    """
    if bypass_llm(user_id):
        logger.info("LLM bypassed for user=%s (test/demo mode)", user_id)
        return fallback

    setup_tracing()
    attempts = attempts or settings.LLM_MAX_ATTEMPTS
    name = label or schema.__name__
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            result = await _structured(schema, tier).ainvoke(prompt)
            _log_usage(f"astructured[{name}]", tier, (time.perf_counter() - started) * 1000, result)
            if result is None:
                raise ValueError("LLM returned no content")
            if not isinstance(result, schema):
                # Defensive: some providers hand back a dict.
                result = schema.model_validate(result)
            return result
        except Exception as exc:  # noqa: BLE001 - we degrade, never crash
            last_error = exc
            logger.warning(
                "Structured call for %s failed (tier=%s, attempt %d/%d): %s",
                name, tier, attempt, attempts, exc,
            )
            if attempt < attempts:
                await asyncio.sleep(0.5 * attempt)

    logger.error(
        "Structured call for %s exhausted %d attempts; using fallback. Last error: %s",
        name, attempts, last_error,
    )
    return fallback


async def atext(
    prompt: str,
    *,
    user_id: Optional[str] = None,
    fallback: str = "",
    attempts: Optional[int] = None,
    tier: str = PLANNER,
    label: Optional[str] = None,
) -> str:
    """Free-text LLM call with the same bounded-retry contract."""
    if bypass_llm(user_id):
        logger.info("LLM bypassed (text) for user=%s (test/demo mode)", user_id)
        return fallback

    setup_tracing()
    attempts = attempts or settings.LLM_MAX_ATTEMPTS
    name = label or "text"
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            chain = get_tier_chain(tier)
            if not chain:
                raise RuntimeError("No LLM client is configured")
            response = await _runnable(chain).ainvoke(prompt)
            _log_usage(f"atext[{name}]", tier, (time.perf_counter() - started) * 1000, response)
            text = response.content
            if isinstance(text, list):
                # Gemini can return a list of content parts.
                text = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in text
                ).strip()
            if isinstance(text, str) and text.strip():
                return text.strip()
            raise ValueError("LLM returned empty content")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Text call for %s failed (tier=%s, attempt %d/%d): %s",
                name, tier, attempt, attempts, exc,
            )
            if attempt < attempts:
                await asyncio.sleep(0.5 * attempt)

    logger.error(
        "Text call for %s exhausted %d attempts; using fallback. Last error: %s",
        name, attempts, last_error,
    )
    return fallback
