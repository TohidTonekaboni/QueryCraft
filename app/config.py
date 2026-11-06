"""Settings read from environment variables (and .env when running outside Docker)."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # no-op for variables already set (e.g. by docker compose)


@dataclass(frozen=True)
class ModelSettings:
    backend: str  # "local" (llama-cpp-python in-process) | "remote" (llama-server)
    remote_url: str
    repo: str  # Hugging Face repo of the GGUF file
    model_path: Path
    n_gpu_layers: int  # -1 = offload all layers; 0 = CPU only
    n_ctx: int
    max_tokens: int


@dataclass(frozen=True)
class Settings:
    # database (read-only role: this app never needs write access)
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    max_rows: int

    sql: ModelSettings  # SQLCoder: text-to-SQL
    chat: ModelSettings  # Gemma: translation + relevance check


def _model(prefix: str, default_repo: str, default_file: str, default_url: str,
           default_layers: str, default_ctx: str, default_tokens: str,
           default_backend: str) -> ModelSettings:
    backend = os.getenv(f"{prefix}_BACKEND", default_backend).lower()
    if backend not in ("local", "remote"):
        raise ValueError(f"{prefix}_BACKEND must be 'local' or 'remote'")
    return ModelSettings(
        backend=backend,
        remote_url=os.getenv(f"{prefix}_REMOTE_URL", default_url).rstrip("/"),
        repo=os.getenv(f"{prefix}_MODEL_REPO", default_repo),
        model_path=Path(os.getenv("SQL_MODEL_DIR", "./models")) / os.getenv(f"{prefix}_MODEL_FILE", default_file),
        n_gpu_layers=int(os.getenv(f"{prefix}_N_GPU_LAYERS", default_layers)),
        n_ctx=int(os.getenv(f"{prefix}_N_CTX", default_ctx)),
        max_tokens=int(os.getenv(f"{prefix}_MAX_TOKENS", default_tokens)),
    )


@lru_cache
def get_settings() -> Settings:
    sql_backend = os.getenv("SQL_BACKEND", "remote")
    return Settings(
        db_host=os.getenv("POSTGRES_HOST", "localhost"),
        db_port=int(os.getenv("POSTGRES_PORT", "5432")),
        db_name=os.environ["POSTGRES_DB"],
        db_user=os.environ["READONLY_USER"],
        db_password=os.environ["READONLY_PASSWORD"],
        max_rows=int(os.getenv("SQL_MAX_ROWS", "100")),
        sql=_model("SQL", "MaziyarPanahi/sqlcoder-7b-2-GGUF", "sqlcoder-7b-2.Q4_K_M.gguf",
                   "http://localhost:8080", "-1", "4096", "300", sql_backend),
        chat=_model("GEMMA", "ggml-org/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf",
                    "http://localhost:8081", "0", "2048", "128", sql_backend),
    )
