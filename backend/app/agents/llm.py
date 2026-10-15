"""LLM clients and the single structured-call entry point.

Every LLM call in the backend goes through :func:`astructured` (JSON outputs)
or :func:`atext` (free-text outputs). Both are:

* **Schema-constrained** - ``with_structured_output(method="json_mode")`` makes
  Gemini emit JSON that must match the Pydantic schema, so the regex-stripping
  / ``json.loads`` failure class is gone.
* **Bounded** - a small number of attempts with backoff, then a caller-supplied
  deterministic fallback. Never raises.
* **Bypassable** - ``test_*`` user ids (and ``DEMO_MODE``) skip the network
  entirely so automated runs never burn quota.

Clients are built **lazily**: a missing API key must not prevent the app from
importing. Without a key the chain is empty and every call degrades to its
fallback, which is exactly the behaviour a fresh clone should have.
"""

import asyncio
import logging
from typing import List, Optional, Type, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger("app.agents.llm")

T = TypeVar("T", bound=BaseModel)

_llm_chain: Optional[List[BaseChatModel]] = None


def _build_primary() -> Optional[BaseChatModel]:
    if not settings.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not set; the primary model is unavailable.")
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.PRIMARY_MODEL,
        api_key=settings.GEMINI_API_KEY,
        max_retries=settings.LLM_MAX_RETRIES,
    )


def _build_secondary() -> Optional[BaseChatModel]:
    if not settings.GROQ_API_KEY:
        logger.info("GROQ_API_KEY is not set; the secondary model is unavailable.")
        return None
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=settings.SECONDARY_MODEL,
        api_key=settings.GROQ_API_KEY,
        max_retries=settings.LLM_MAX_RETRIES,
    )


def get_llm_chain() -> List[BaseChatModel]:
    """Primary → secondary. Deliberately short: a free-tier third model that
    rate-limits under load is a worse failure mode than a deterministic
    fallback."""
    global _llm_chain
    if _llm_chain is None:
        chain: List[BaseChatModel] = []
        for builder in (_build_primary, _build_secondary):
            try:
                model = builder()
            except Exception as exc:  # noqa: BLE001 - a broken client must not break import
                logger.error("Failed to construct LLM client %s: %s", builder.__name__, exc)
                model = None
            if model is not None:
                chain.append(model)
        _llm_chain = chain
        logger.info(
            "LLM chain ready: %s",
            " -> ".join(type(m).__name__ for m in chain) or "EMPTY (fallbacks only)",
        )
    return _llm_chain


def _chain_runnable(primary: BaseChatModel, chain: List[BaseChatModel]):
    """Attach fallbacks when there is more than one model."""
    if len(chain) > 1:
        return primary.with_fallbacks(chain[1:])
    return primary


# Backwards-compatible module attribute: the first available model (or None).
def __getattr__(name: str):
    if name == "llm":
        chain = get_llm_chain()
        return _chain_runnable(chain[0], chain) if chain else None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Cache of schema -> structured runnable, so we don't rebuild per request.
_structured_cache: dict = {}


def _structured(schema: Type[T]):
    """Build (and cache) a json_mode structured runnable for ``schema``."""
    cached = _structured_cache.get(schema)
    if cached is None:
        chain = get_llm_chain()
        if not chain:
            raise RuntimeError(
                "No LLM client is configured (set GEMINI_API_KEY); "
                f"cannot run structured call for {schema.__name__}"
            )
        cached = _chain_runnable(chain[0], chain).with_structured_output(
            schema, method="json_mode"
        )
        _structured_cache[schema] = cached
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
) -> Optional[T]:
    """Call the LLM and return a validated ``schema`` instance.

    Returns ``fallback`` (possibly ``None``) instead of raising, so callers
    always get a usable value. ``fallback`` is returned immediately for test
    users and demo mode without touching the network.
    """
    if bypass_llm(user_id):
        logger.info("LLM bypassed for user=%s (test/demo mode)", user_id)
        return fallback

    attempts = attempts or settings.LLM_MAX_ATTEMPTS
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            result = await _structured(schema).ainvoke(prompt)
            if result is None:
                raise ValueError("LLM returned no content")
            if not isinstance(result, schema):
                # Defensive: some providers hand back a dict.
                result = schema.model_validate(result)
            return result
        except Exception as exc:  # noqa: BLE001 - we degrade, never crash
            last_error = exc
            logger.warning(
                "Structured call for %s failed (attempt %d/%d): %s",
                schema.__name__, attempt, attempts, exc,
            )
            if attempt < attempts:
                await asyncio.sleep(0.5 * attempt)

    logger.error(
        "Structured call for %s exhausted %d attempts; using fallback. Last error: %s",
        schema.__name__, attempts, last_error,
    )
    return fallback


async def atext(
    prompt: str,
    *,
    user_id: Optional[str] = None,
    fallback: str = "",
    attempts: Optional[int] = None,
) -> str:
    """Free-text LLM call with the same bounded-retry contract."""
    if bypass_llm(user_id):
        logger.info("LLM bypassed (text) for user=%s (test/demo mode)", user_id)
        return fallback

    attempts = attempts or settings.LLM_MAX_ATTEMPTS
    last_error: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            chain = get_llm_chain()
            if not chain:
                raise RuntimeError("No LLM client is configured")
            response = await _chain_runnable(chain[0], chain).ainvoke(prompt)
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
            logger.warning("Text call failed (attempt %d/%d): %s", attempt, attempts, exc)
            if attempt < attempts:
                await asyncio.sleep(0.5 * attempt)

    logger.error(
        "Text call exhausted %d attempts; using fallback. Last error: %s", attempts, last_error
    )
    return fallback
