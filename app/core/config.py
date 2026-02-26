from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    openai_api_key: str
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 20          # candidates fetched before rerank
    retrieval_top_n: int = 5           # chunks sent to LLM after rerank
    similarity_threshold: float = 0.30 # below this → "no relevant context"
    bm25_weight: float = 0.4           # RRF weighting for keyword leg
    dense_weight: float = 0.6          # RRF weighting for semantic leg

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 50
    semantic_chunking: bool = True
    min_chunk_chars: int = 30

    # ── Guardrails ────────────────────────────────────────────────────────────
    enable_pii_scrubbing: bool = True
    pii_guard_provider: str = "presidio"  # presidio|regex
    enable_prompt_injection_guard: bool = True
    injection_guard_provider: str = "rebuff"  # rebuff|local|regex
    rebuff_api_key: str = ""
    rebuff_block_threshold: float = 0.6
    local_injection_model: str = "protectai/deberta-v3-base-prompt-injection"
    local_injection_block_threshold: float = 0.6
    max_input_chars: int = 4096

    # ── Observability ─────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "chat-with-docs"

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()