import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Storage ──
    # These are real fields, not os.environ lookups. `neon env pull` writes them
    # into .env, and pydantic-settings loads a .env into fields -- it does NOT
    # populate os.environ. Reading os.environ directly therefore missed every
    # value the CLI pulled, and the app silently fell back to the localhost
    # default below.
    DATABASE_URL: str = ""
    DATABASE_URL_UNPOOLED: str = ""

    # Neon hands back two URLs for the same database, and they are not
    # interchangeable:
    #
    #   DATABASE_URL          pooled — PgBouncer in transaction mode. Normal
    #                         application traffic. No session state, no DDL.
    #   DATABASE_URL_UNPOOLED direct — schema migrations, `setup()` DDL, dumps,
    #                         LISTEN/NOTIFY, anything relying on SET.
    #
    # Running DDL over the pooled endpoint fails in ways that never name
    # pooling: `prepared statement "s0" already exists`, a `SET search_path`
    # that does not survive its own transaction, or a write landing in a
    # read-only transaction the pooler inherited from an earlier client.
    #
    # `neon env pull` writes both. If only one is present we fall back to it,
    # which is correct for local development against a plain Postgres and
    # merely degraded (not broken) on Neon.
    DB_URI: str = "postgresql://postgres:postgres@localhost:5432/agent_planner"
    DB_URI_UNPOOLED: str = ""

    # Pool sizing for the LangGraph store. The checkpointer takes a single
    # connection by design, so only the store is pooled; see app/db.py.
    DB_POOL_MIN_SIZE: int = 1
    DB_POOL_MAX_SIZE: int = 10

    # ── LLM ──
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    TMDB_API_KEY: str = ""

    # Model ids, per tier. Gemini 2.5 is only served to accounts that already
    # used it, so these are the current GA line-up:
    #   gemini-3.8-flash      workhorse — schedule building, parsing, summaries
    #   gemini-3.5-flash-lite routing   — intent/college classification
    # Tiering is a latency *and* cost lever: the routing tier is ~2.5x cheaper
    # than the workhorse and answers in a fraction of the time, so every call
    # that only needs to pick one of a few labels goes there.
    PRIMARY_MODEL: str = "gemini-3.8-flash"
    ROUTING_MODEL: str = "gemini-3.5-flash-lite"
    SECONDARY_MODEL: str = "llama-3.3-70b-versatile"
    # Bound on LLM attempts before the deterministic fallback is used.
    LLM_MAX_ATTEMPTS: int = 2
    LLM_MAX_RETRIES: int = 1
    # When true, every LLM call is skipped and deterministic paths are used.
    DEMO_MODE: bool = False

    # ── Observability ──
    # LangSmith tracing. Enabled automatically when a key is present; without
    # one, calls are still logged locally with token/cache-hit counts so
    # latency and cache behaviour stay measurable.
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
    # `neon env pull` writes .env at the *repo root*, next to neon.ts, because
    # that is where the Neon project is linked. This module used to look only at
    # backend/.env, so a freshly pulled DATABASE_URL was silently ignored and the
    # app fell back to the localhost default. Both locations are read, root first
    # so it wins when both exist.
    model_config = SettingsConfigDict(
        env_file=(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                ".env",
            ),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def pooled_dsn(self) -> str:
        """Connection string for normal application traffic.

        Prefers Neon's pooled `DATABASE_URL`, falling back to `DB_URI` for a
        plain local Postgres.
        """
        return self.DATABASE_URL or self.DB_URI

    def direct_dsn(self) -> str:
        """Connection string for DDL and migrations.

        Prefers Neon's unpooled `DATABASE_URL_UNPOOLED`. If it is absent we use
        the pooled DSN — on Neon that is the wrong choice for DDL, so the
        caller is expected to warn; on a local Postgres it is simply right.
        """
        return self.DATABASE_URL_UNPOOLED or self.pooled_dsn()

    @property
    def using_neon(self) -> bool:
        """True when the DSNs point at Neon (they carry a `-pooler` host)."""
        return "-pooler." in self.pooled_dsn()


settings = Settings()
