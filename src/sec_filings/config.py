"""Centralised env-driven configuration.

Uses pydantic-settings so config errors fail loudly at startup rather than
producing silent fallbacks (see PROJECT.md — no silent fallbacks).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="")
    voyage_api_key: str = Field(default="")
    cohere_api_key: str = Field(default="")
    fmp_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    edgar_identity: str = Field(default="")

    # Paths.
    data_dir: Path = REPO_ROOT / "data"
    filings_dir: Path = REPO_ROOT / "data" / "filings"
    chroma_dir: Path = REPO_ROOT / "data" / "chroma"
    bm25_dir: Path = REPO_ROOT / "data" / "bm25"
    results_dir: Path = REPO_ROOT / "results"
    eval_sets_dir: Path = REPO_ROOT / "data" / "eval_sets"

    # Models & service endpoints — pinned so retrieval/agent behaviour is reproducible.
    embed_model: str = "voyage-4-large"
    embed_dim: int = 1024
    agent_model: str = "claude-sonnet-4-6"
    fmp_base_url: str = "https://financialmodelingprep.com/stable"
    chroma_collection: str = "sec_filings"

    # Langfuse tracing (optional). We read these explicitly here — not via the
    # Langfuse SDK's own env auto-detection — because pydantic-settings loads
    # .env into THIS object, not into os.environ, so get_client() with no args
    # would never see the keys. Note the host comes from LANGFUSE_BASE_URL (the
    # name Langfuse's onboarding hands you), which the SDK also accepts. If the
    # keys are blank, tracing is a silent no-op (an absent optional, not a hidden
    # error) and the agent runs unchanged — see sec_filings.observability.
    langfuse_public_key: str = Field(default="")
    langfuse_secret_key: str = Field(default="")
    langfuse_base_url: str = "https://us.cloud.langfuse.com"


settings = Settings()
