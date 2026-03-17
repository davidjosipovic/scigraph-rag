"""
Central configuration for the KG-RAG system.

All settings are loaded from environment variables (or .env file)
using pydantic-settings for validation and type coercion.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root: two levels up from this file (backend/config.py → kg-rag/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- ORKG / Knowledge Graph ---
    orkg_sparql_endpoint: str = "https://orkg.org/triplestore"
    use_local_rdf: bool = False
    local_rdf_path: str = "data/orkg_dump.nt"

    # --- Ollama / LLM ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_timeout: int = 120

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    log_level: str = "INFO"
    # Comma-separated list of allowed CORS origins.
    # Override with CORS_ORIGINS env var for production, e.g.:
    #   CORS_ORIGINS=https://myapp.example.com
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8080", "http://localhost:5173"]

    # --- Retrieval ---
    max_results: int = 5
    sparql_timeout: int = 10
    max_context_papers: int = 8

    # --- Concurrency ---
    # Max concurrent /ask requests allowed to run the full pipeline.
    # Ollama is single-threaded; too many parallel LLM calls saturate it.
    # Extra requests wait (up to ollama_timeout seconds) before getting 503.
    max_concurrent_requests: int = 3


# Singleton settings instance — import this everywhere
settings = Settings()
