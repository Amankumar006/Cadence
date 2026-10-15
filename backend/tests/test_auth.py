"""Tests for the authentication dependency.

The security property under test: **a request body can never override the uid
carried by a verified token.** In local-dev fallback (no Firebase credentials)
the body uid is accepted so the Expo mock-auth demo keeps working.
"""

import pytest
from fastapi import HTTPException

from app.api.auth import get_current_user, resolve_user_id


# ── resolve_user_id ─────────────────────────────────────────────────────────


def test_token_uid_wins_over_the_body():
    assert resolve_user_id("real_uid", "real_uid") == "real_uid"


def test_body_cannot_escalate_to_another_users_uid():
    """The core property: a forged body uid is refused, not honoured."""
    with pytest.raises(HTTPException) as exc:
        resolve_user_id("real_uid", "attacker_uid")
    assert exc.value.status_code == 403


def test_token_uid_used_when_body_is_empty():
    assert resolve_user_id("real_uid", None) == "real_uid"
    assert resolve_user_id("real_uid", "") == "real_uid"


def test_mismatched_body_is_rejected():
    with pytest.raises(HTTPException) as exc:
        resolve_user_id("real_uid", "someone_else")
    assert exc.value.status_code == 403


def test_dev_fallback_uses_the_body_uid():
    assert resolve_user_id(None, "demo_anonymous") == "demo_anonymous"


def test_dev_fallback_uses_the_default_uid():
    from app.config import settings

    assert resolve_user_id(None, None) == settings.DEV_FALLBACK_USER_ID


# ── get_current_user ────────────────────────────────────────────────────────


class _Creds:
    def __init__(self, token):
        self.credentials = token


async def test_dev_mode_returns_none_without_raising(monkeypatch):
    monkeypatch.setattr("app.api.auth.firebase_initialized", False)
    assert await get_current_user(None) is None
    assert await get_current_user(_Creds("mock-dev-token-jwt")) is None


async def test_missing_token_is_rejected_in_production(monkeypatch):
    monkeypatch.setattr("app.api.auth.firebase_initialized", True)
    with pytest.raises(HTTPException) as exc:
        await get_current_user(None)
    assert exc.value.status_code == 401
    assert "Missing bearer token" in exc.value.detail


async def test_invalid_token_is_rejected_in_production(monkeypatch):
    monkeypatch.setattr("app.api.auth.firebase_initialized", True)
    monkeypatch.setattr("app.api.auth.auth.verify_id_token", lambda token: (_ for _ in ()).throw(ValueError("bad token")))
    with pytest.raises(HTTPException) as exc:
        await get_current_user(_Creds("garbage"))
    assert exc.value.status_code == 401
    assert "Invalid token" in exc.value.detail


async def test_valid_token_returns_the_uid(monkeypatch):
    monkeypatch.setattr("app.api.auth.firebase_initialized", True)
    monkeypatch.setattr("app.api.auth.auth.verify_id_token", lambda token: {"uid": "verified_uid"})
    assert await get_current_user(_Creds("good-token")) == "verified_uid"
