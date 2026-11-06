# QueryCraft

Ask a question about the database in **Persian or English**; the API writes the PostgreSQL query, runs it, and returns the SQL and the rows.

```
POST /query   {"question": "گران‌ترین محصول فروخته شده چیست؟"}
→ {"success": true, "sql_query": "SELECT ...", "final_answer": [{"name": "...", "price": 1482.33}]}
```

## Architecture

A LangGraph workflow (`app/workflow/`) owns the routing, shared state and the correction loop:

```
input → language → relevance ──no──→ respond
                       │yes
                       ▼
            ┌───→ generate → validate ──invalid──┐
            │                   │valid           │
            │                   ▼                ▼
            │                execute ──error──→ correct ──max attempts──→ respond
            │                   │ok               │
            └───────────────────┼─────────────────┘
                                ▼
                             respond
```

| Node | What it does |
|---|---|
| language | Detects Persian by script, normalises it (Arabic letters, digits, half-spaces), translates to English with **Gemma** |
| relevance | **Gemma** answers YES/NO: can this be answered by a read-only SELECT on our schema? |
| generate | **SQLCoder-7B-2** writes the query; on retries the failing SQL and its error are in the prompt |
| validate | `sqlglot`: one statement, SELECT only; no DDL/DML, `SELECT INTO`, `FOR UPDATE`, data-modifying CTEs, `pg_*`/`lo_*`/`dblink`/`set_config` functions |
| execute | `asyncpg` as a **read-only role**, read-only transaction, statement timeout, row limit |
| correct | Retry (max `MAX_SQL_ATTEMPTS`, default 3), then `success: false` with the last error |

Database: three related tables (`ingestion/schema.sql`): `categories` ← `products` ← `order_items`; seeded with 1000 products and 5000 sales. The `ingestion` service creates the schema, seeds it and creates the read-only role `querycraft_ro`; it is idempotent.

Models: both are quantized GGUF files run by llama.cpp. Everything runs locally. Gemma defaults to **CPU** and SQLCoder to the **GPU**, so two models never share GPU memory.

| Role | Model | Default file |
|---|---|---|
| SQL generation | SQLCoder-7B-2 | `MaziyarPanahi/sqlcoder-7b-2-GGUF` · `sqlcoder-7b-2.Q4_K_M.gguf` |
| Translation, relevance | Gemma 3 4B instruct | `ggml-org/gemma-3-4b-it-GGUF` · `gemma-3-4b-it-Q4_K_M.gguf` |

## Setup

1. Edit `.env` (change the passwords).
2. Download the weights into `./models` (about 4 GB + 2.5 GB):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r app/requirements.txt
   python -m app.scripts.download_model          # or: ... sql | chat
   ```
3. Start the models, then the stack. Pick the option for your hardware.

### macOS (Apple Silicon, MPS): models on the host, app in Docker

Docker containers on macOS cannot use MPS, so `llama-server` runs on the host and the container calls it (`SQL_BACKEND=remote`, `GEMMA_BACKEND=remote`, the defaults).

```bash
brew install llama.cpp
app/scripts/run_llama_server.sh sql     # terminal 1: SQLCoder on :8080, all layers on MPS
app/scripts/run_llama_server.sh gemma   # terminal 2: Gemma on :8081, CPU
docker compose up -d --build            # terminal 3
```

### NVIDIA GPU (CUDA): everything in Docker

Needs the NVIDIA Container Toolkit. The image builds `llama-cpp-python` with CUDA and loads the models in-process.

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build
```

### CPU only

Set `SQL_N_GPU_LAYERS=0` in `.env`, then use either option above.

### Without Docker for the app

`pip install -r app/requirements-local.txt` (Metal is built in on macOS; use `CMAKE_ARGS="-DGGML_CUDA=on"` for CUDA), set `SQL_BACKEND=local` and `GEMMA_BACKEND=local`, run `docker compose up -d postgres ingestion`, then `uvicorn app.main:app`. A one-off question: `python -m app.cli "most expensive product"`.

## Configuration (`.env`)

| Variable | Meaning | Default |
|---|---|---|
| `POSTGRES_*` | Database credentials (owner role, used by ingestion) | |
| `READONLY_USER/PASSWORD` | Role the API uses to run generated SQL | |
| `READONLY_STATEMENT_TIMEOUT` | Query time limit for that role | `5s` |
| `SQL_MAX_ROWS` | Max rows returned | `100` |
| `MAX_SQL_ATTEMPTS` | SQL generation attempts before giving up | `3` |
| `SQL_BACKEND` / `GEMMA_BACKEND` | `remote` (llama-server) or `local` (llama-cpp-python in-process) | `remote` |
| `SQL_REMOTE_URL` / `GEMMA_REMOTE_URL` | llama-server URLs | `…:8080` / `…:8081` |
| `SQL_N_GPU_LAYERS` / `GEMMA_N_GPU_LAYERS` | Layers offloaded to GPU (`-1` all, `0` CPU) | `-1` / `0` |
| `SQL_N_CTX` / `GEMMA_N_CTX` | Context size | `4096` / `2048` |
| `SQL_MODEL_DIR`, `*_MODEL_REPO`, `*_MODEL_FILE` | Where the weights are and which to download | `./models` |

## Tests

```bash
pip install -r app/requirements-dev.txt
docker compose up -d postgres ingestion    # tests use the real, seeded database
python -m pytest
```

The language models are replaced by deterministic fakes, so no weights are needed; the validator, LangGraph workflow, FastAPI app and PostgreSQL are real. The suite covers Persian vs. English, a three-table JOIN, GROUP BY, irrelevant and destructive questions, a forced SQL error with the correction loop, and the retry cap. Tests skip themselves if PostgreSQL is not reachable.

## Limits and notes

- Only the SQL model may be external per the assignment; both models here are local.
- Blocking inference runs in a worker thread (`local`) or over HTTP (`remote`), so the event loop is never blocked.
- `app/sql/prompt.py` holds a copy of the schema; keep it in sync with `ingestion/schema.sql`.
- Model outage or an unreachable database returns HTTP 503 with `success: false`.
