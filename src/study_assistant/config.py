"""Application settings.

Every model name, path and tuning number lives here. Override any field with an
environment variable prefixed ``STUDY_`` (e.g. ``STUDY_LLM_MODEL=qwen3:8b``) or in a
``.env`` file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STUDY_", env_file=".env", extra="ignore")

    ollama_host: str = "http://localhost:11434"
    llm_model: str = "gemma2:9b"
    llm_timeout_s: float = 180.0
    embedding_model: str = "intfloat/multilingual-e5-small"
    data_dir: Path = Path("data")  # chroma at data/chroma, sqlite at data/quizzes.db
    chunk_size: int = 1500  # characters
    chunk_overlap: int = 200
    top_k: int = 6
    min_score: float = 0.75  # tune with evals (Task 12)
    quiz_max_chunks: int = 10
    max_upload_mb: int = 50
    max_pages: int = 300
    api_url: str = "http://localhost:8000"  # used by the UI


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (cached; call ``get_settings.cache_clear()`` to reload)."""
    return Settings()
