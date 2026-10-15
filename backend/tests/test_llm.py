"""Tests for the LLM entry points: schema binding, bypass, and fallbacks."""

import pytest
from pydantic import BaseModel

from app.agents import llm as llm_module
from app.agents.llm import astructured, atext, bypass_llm, get_llm_chain
from app.agents.schemas import NoteSummary, ParsedInfo


class _Sample(BaseModel):
    answer: str = ""


async def test_astructured_uses_json_mode(fake_llm):
    fake = fake_llm(lambda prompt, schema: {"answer": "42"})
    result = await astructured("what is the answer", _Sample)

    assert result.answer == "42"
    # The schema must be bound with json_mode so Gemini constrains its output.
    assert fake.structured_calls == [(_Sample, "json_mode")]


async def test_astructured_validates_a_dict_response(fake_llm):
    fake_llm(lambda prompt, schema: {"summary": "s", "subject": "DSA", "tags": ["x"]})
    result = await astructured("summarise", NoteSummary)
    assert result.subject == "DSA"
    assert isinstance(result, NoteSummary)


async def test_astructured_returns_fallback_when_all_attempts_fail(no_llm):
    fallback = ParsedInfo(type="general")
    result = await astructured("hello", ParsedInfo, fallback=fallback)
    assert result is fallback


async def test_astructured_returns_none_without_a_fallback(no_llm):
    assert await astructured("hello", ParsedInfo) is None


async def test_astructured_never_raises_on_bad_model_output(fake_llm):
    def bad(prompt, schema):
        return {"totally": "wrong", "shape": object()}

    fake_llm(bad)
    result = await astructured("x", ParsedInfo, fallback=ParsedInfo(type="general"))
    assert result.type == "general"


async def test_astructured_retries_before_giving_up(fake_llm, monkeypatch):
    attempts = {"n": 0}

    def flaky(prompt, schema):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient")
        return {"answer": "ok"}

    monkeypatch.setattr(
        llm_module, "settings", llm_module.settings.model_copy(update={"LLM_MAX_ATTEMPTS": 2})
    )
    fake_llm(flaky)

    result = await astructured("x", _Sample, attempts=2)
    assert result.answer == "ok"
    assert attempts["n"] == 2


async def test_atext_returns_the_stripped_text(fake_llm):
    fake = fake_llm(lambda prompt, schema: "  hello there  ")
    assert await atext("hi") == "hello there"
    assert fake.calls == ["hi"]


async def test_atext_returns_fallback_when_empty(fake_llm):
    fake_llm(lambda prompt, schema: "   ")
    assert await atext("hi", fallback="FALLBACK") == "FALLBACK"


async def test_atext_returns_fallback_on_error(no_llm):
    assert await atext("hi", fallback="FALLBACK") == "FALLBACK"


# ── Bypass rules ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("user_id", ["test_abc", "test_", "test_12345"])
async def test_bypass_for_test_users(user_id):
    assert bypass_llm(user_id)


@pytest.mark.parametrize("user_id", ["real_user", "mock_user_123", "", None])
async def test_no_bypass_for_real_users(user_id):
    assert not bypass_llm(user_id)


async def test_demo_mode_bypasses_everyone(demo_mode):
    assert bypass_llm("a_real_user")


async def test_bypass_skips_the_network_entirely(fake_llm, demo_mode):
    fake = fake_llm(lambda prompt, schema: {"answer": "should not be called"})
    result = await astructured("x", _Sample, fallback=_Sample(answer="fallback"))
    assert result.answer == "fallback"
    assert await atext("x", fallback="FB") == "FB"
    assert fake.calls == []
    assert fake.structured_calls == []


# ── Lazy client construction ────────────────────────────────────────────────


def test_chain_is_empty_without_keys(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "GEMINI_API_KEY", "")
    monkeypatch.setattr(llm_module.settings, "GROQ_API_KEY", "")
    monkeypatch.setattr(llm_module, "_llm_chain", None)
    assert get_llm_chain() == []


def test_structured_raises_clearly_without_a_client(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "GEMINI_API_KEY", "")
    monkeypatch.setattr(llm_module, "_llm_chain", [])
    with pytest.raises(RuntimeError, match="No LLM client is configured"):
        llm_module._structured(ParsedInfo)


def test_module_imports_without_any_api_key():
    """A fresh clone must be importable with no keys set at all."""
    import subprocess
    import sys

    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": "."}
    code = (
        "import os;"
        "os.environ.pop('GEMINI_API_KEY', None);"
        "os.environ.pop('GROQ_API_KEY', None);"
        "import app.agents.parser, app.agents.nodes, app.api.chat, app.api.upload;"
        "print('OK')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "OK" in proc.stdout
