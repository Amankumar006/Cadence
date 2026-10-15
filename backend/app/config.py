import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Storage ──
    DB_URI: str = "postgresql://postgres:postgres@localhost:5432/agent_planner"

    # ── LLM ──
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    TMDB_API_KEY: str = ""

    # Model ids. Gemini 2.5 is only served to accounts that already used it,
    # so the default is the current GA workhorse. Override per environment.
    PRIMARY_MODEL: str = "gemini-3.8-flash"
    SECONDARY_MODEL: str = "llama-3.3-70b-versatile"
    # Bound on LLM attempts before the deterministic fallback is used.
    LLM_MAX_ATTEMPTS: int = 2
    LLM_MAX_RETRIES: int = 1
    # When true, every LLM call is skipped and deterministic paths are used.
    DEMO_MODE: bool = False

    # ── Observability ──
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "personal-ai-agent"

    # ── Firebase ──
    FIREBASE_SERVICE_ACCOUNT_PATH: str = "firebase-service-account.json"
    FIREBASE_PROJECT_ID: str = ""
    FIREBASE_CLIENT_EMAIL: str = ""
    FIREBASE_PRIVATE_KEY: str = ""
    # Uid returned by the auth dependency when Firebase is not configured
    # (local development only — see app/api/auth.py).
    DEV_FALLBACK_USER_ID: str = "mock_user_123"

    # Use pydantic-settings config to load from .env file relative to the root directory
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
