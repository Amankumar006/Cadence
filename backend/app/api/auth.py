"""Firebase authentication for the backend.

Two modes:

**Production** (Firebase credentials present) — every request must carry a
valid Firebase ID token. The uid from the token is authoritative: the
``user_id`` in a request body is only used to *check* that the caller is not
trying to act on someone else's data (403 otherwise).

**Local development** (no credentials) — Firebase cannot verify anything, so
the dependency returns ``None`` and the body-supplied ``user_id`` is accepted
with a warning. This keeps the Expo Go / mock-auth demo flow working while
making it impossible to accidentally ship an unauthenticated endpoint.
"""

import os
import logging
from typing import Optional

from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import firebase_admin
from firebase_admin import auth, credentials, initialize_app

from app.config import settings

logger = logging.getLogger("app.api.auth")

# auto_error=False so that a missing token in dev mode is not an immediate 403;
# the production check below rejects it explicitly.
security = HTTPBearer(auto_error=False)

firebase_initialized = False

# Try service account file first
if os.path.exists(settings.FIREBASE_SERVICE_ACCOUNT_PATH):
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(settings.FIREBASE_SERVICE_ACCOUNT_PATH)
            initialize_app(cred)
        firebase_initialized = True
        logger.info("Firebase Admin SDK initialized successfully from service account file.")
    except Exception as e:
        logger.error("Failed to initialize Firebase Admin SDK from file: %s", e)

# The Firebase project this repo deploys to. `.firebaserc` is the source of
# truth; FIREBASE_PROJECT_ID overrides it. The previous fallback hardcoded
# "campus-connect-enanq", which is a *different* project from the one in
# .firebaserc (agent-ad07f) -- so a deploy without FIREBASE_PROJECT_ID set
# verified tokens against the wrong project and every request 401'd.
def _default_project_id() -> str:
    if settings.FIREBASE_PROJECT_ID:
        return settings.FIREBASE_PROJECT_ID
    try:
        import json
        from pathlib import Path

        # backend/app/api/auth.py -> repo root
        firebaserc = Path(__file__).resolve().parents[3] / ".firebaserc"
        if firebaserc.exists():
            data = json.loads(firebaserc.read_text())
            project = (data.get("projects") or {}).get("default")
            if project:
                logger.info("Using Firebase project '%s' from .firebaserc", project)
                return project
    except (OSError, ValueError) as err:  # pragma: no cover - defensive
        logger.warning("Could not read .firebaserc for the project id: %s", err)
    logger.warning(
        "No Firebase project id configured. Set FIREBASE_PROJECT_ID; token "
        "verification will fail without it."
    )
    return ""


# If not initialized, try env variables
if not firebase_initialized and settings.FIREBASE_CLIENT_EMAIL and settings.FIREBASE_PRIVATE_KEY:
    try:
        if not firebase_admin._apps:
            formatted_key = settings.FIREBASE_PRIVATE_KEY.replace("\\n", "\n")
            cred_dict = {
                "type": "service_account",
                "project_id": _default_project_id(),
                "private_key": formatted_key,
                "client_email": settings.FIREBASE_CLIENT_EMAIL,
                "token_uri": "https://oauth2.googleapis.com/token",
            }
            cred = credentials.Certificate(cred_dict)
            initialize_app(cred)
        firebase_initialized = True
        logger.info("Firebase Admin SDK initialized successfully from environment variables.")
    except Exception as e:
        logger.error("Failed to initialize Firebase Admin SDK from env variables: %s", e)

if not firebase_initialized:
    logger.warning(
        "Firebase credentials not found or failed to initialize. Running in LOCAL DEV auth "
        "fallback mode: request bodies may supply user_id and tokens are not verified. "
        "Set FIREBASE_SERVICE_ACCOUNT_PATH (or FIREBASE_CLIENT_EMAIL + FIREBASE_PRIVATE_KEY) "
        "before deploying."
    )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Optional[str]:
    """Verify the Firebase ID token and return the uid.

    Returns ``None`` in local-dev fallback mode (Firebase not configured).
    Raises 401 when Firebase *is* configured but the token is missing/invalid.
    """
    if not firebase_initialized:
        if credentials and credentials.credentials:
            logger.info(
                "Dev auth fallback: accepting unverified token '%s...'",
                credentials.credentials[:10],
            )
        return None

    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    try:
        decoded = auth.verify_id_token(credentials.credentials)
        return decoded["uid"]
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


def resolve_user_id(
    verified_uid: Optional[str], provided_user_id: Optional[str]
) -> str:
    """Decide which uid a request should act on.

    Production: the token uid wins, and a mismatching body uid is a 403.
    Dev: the body uid is accepted (demo/mock auth), else the dev fallback uid.
    """
    if verified_uid:
        if provided_user_id and provided_user_id != verified_uid:
            logger.warning(
                "Rejected request: body user_id=%s does not match token uid=%s",
                provided_user_id,
                verified_uid,
            )
            raise HTTPException(
                status_code=403,
                detail="user_id does not match the authenticated user",
            )
        return verified_uid

    if provided_user_id:
        logger.warning(
            "DEV AUTH: accepting body user_id=%s because Firebase is not configured",
            provided_user_id,
        )
        return provided_user_id

    logger.warning(
        "DEV AUTH: no user_id supplied and Firebase is not configured; using %s",
        settings.DEV_FALLBACK_USER_ID,
    )
    return settings.DEV_FALLBACK_USER_ID


# Convenience dependency for routes that only need the resolved uid.
async def require_user(
    verified_uid: Optional[str] = Depends(get_current_user),
) -> Optional[str]:
    """Dependency that yields the verified uid (or None in dev mode)."""
    return verified_uid
