"""Application configuration loaded from environment variables.

All settings can be overridden with environment variables prefixed with ``DOCUMIND_``
(for example ``DOCUMIND_DATABASE_URL``). Secrets such as LLM API keys are read from
the environment only and are never logged.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCUMIND_",
        env_file=(PROJECT_DIR / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General -----------------------------------------------------------------
    app_name: str = "DocuMind AI"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    api_prefix: str = "/api/v1"
    # Comma-separated list in the environment, e.g. "http://localhost:5173,http://localhost:8080".
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["http://localhost:5173"])

    # --- Database ----------------------------------------------------------------
    # No password in the default: set DOCUMIND_DATABASE_URL (see .env.example).
    database_url: str = "postgresql+psycopg://documind@localhost:5432/documind"
    db_pool_size: int = 5
    db_statement_timeout_ms: int = 15000

    # --- Uploads -----------------------------------------------------------------
    upload_dir: Path = PROJECT_DIR / "data" / "uploads"
    max_upload_mb: int = 20
    max_pdf_pages: int = 500
    max_text_chars: int = 5_000_000

    # --- Ingestion ---------------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_chunk_chars: int = 40
    min_page_text_chars: int = 20
    embedding_batch_size: int = 32
    ingest_in_background: bool = True

    # --- Embeddings --------------------------------------------------------------
    embedding_provider: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # --- Retrieval ---------------------------------------------------------------
    retrieval_top_k: int = 6
    retrieval_candidates: int = 20
    retrieval_mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"
    rrf_k: int = 60
    reranker: Literal["none", "lexical"] = "lexical"
    dedup_jaccard_threshold: float = 0.85
    min_semantic_score: float = 0.25

    # --- LLM ---------------------------------------------------------------------
    llm_provider: Literal["mock", "openai", "anthropic"] = "mock"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: SecretStr | None = None
    llm_timeout_s: float = 60.0
    llm_max_tokens: int = 800
    llm_temperature: float = 0.1
    llm_cost_per_1k_input: float | None = None
    llm_cost_per_1k_output: float | None = None
    conversation_history_turns: int = 4

    # --- Limits ------------------------------------------------------------------
    max_question_chars: int = 2000
    max_page_size: int = 100

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            if value.strip().startswith("["):
                return json.loads(value)
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_chunk(cls, value: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 1000)
        if value < 0 or value >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
