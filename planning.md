# QueryCraft: Next Steps

Based on the build order in [Project-Description.md](Project-Description.md).

## Status

| Step | Status |
|---|---|
| 1. Schema, seed data, Docker Compose with Postgres | Done |
| 2. LangGraph skeleton with stub nodes | Done |
| 3. English SQL generation, validation, execution | Implemented; real model not yet run |
| 4. Persian detection and translation (Gemma) | Implemented; real model not yet run |
| 5. Relevance check and retry loop | Implemented; real model not yet run |
| 6. FastAPI endpoint, tests, README | Done (38 tests pass with fake models) |

Remaining: an end-to-end run with the real SQLCoder and Gemma weights (see "Final test checklist" at the bottom).

## Step 1: finish (done)

- [x] Seed `categories` and `order_items` (5000 sales over the last year).
- [x] Read-only DB role `querycraft_ro` with a statement timeout; verified it can SELECT but not write.
- [x] Three tables: `categories`, `products`, `order_items` (no separate `orders` table).
- [x] Schema in standalone `ingestion/schema.sql`.

## Step 2: LangGraph skeleton (done, in `app/workflow/`)

- [x] Shared state in `state.py`: question, language, translated question, relevance, SQL, validity, rows, error, attempts, max attempts, response.
- [x] Async nodes in `nodes.py`: input, language, relevance, generate, validate, execute, correct, respond.
- [x] Routing in `graph.py`:
  - irrelevant questions exit early to respond;
  - validation and execution errors go to correct, then back to generate with the error and previous SQL in state;
  - the loop ends at `max_attempts` (`MAX_SQL_ATTEMPTS`, default 3) with `success: false` and the last error.
- [x] `build_graph(overrides)` swaps nodes by name, so tests and later steps can inject fakes.
- [x] Checked with ad-hoc runs: English, Persian, irrelevant, retry then success, retries exhausted, destructive SQL. There are no committed tests yet (Step 6).
- [x] `app/requirements.txt` pins `langgraph` and `langchain-core`.

Still stubs, replaced in later steps:

| Node | Stub behaviour | Replaced in |
|---|---|---|
| language | Persian detected by Unicode range; no translation | Step 4 |
| relevance | accepts any non-empty question | Step 5 |
| generate | returns a fixed query | Step 3 |
| validate | checks the SQL starts with `SELECT` | Step 3 |
| execute | returns no rows, no DB access | Step 3 |
| correct | no-op; the error is already in state | Step 5 |

## Step 3: SQL generation, validation, execution (English only)

Code is in `app/sql/`, `app/db.py`, `app/config.py`, `app/scripts/`.

- [x] SQL model: SQLCoder-7B-2 GGUF Q4_K_M (`MaziyarPanahi/sqlcoder-7b-2-GGUF`), two backends chosen by `SQL_BACKEND`:
  - `local`: `llama-cpp-python` in-process (Metal/MPS on macOS, CUDA on NVIDIA builds, CPU with `SQL_N_GPU_LAYERS=0`);
  - `remote`: OpenAI-compatible `llama-server` on the host. Use this with Docker on macOS, because containers cannot use MPS.
- [x] `python -m app.scripts.download_model` downloads the weights into `./models` (gitignored, mounted at `/models`).
- [x] `SQL_N_GPU_LAYERS`, `SQL_N_CTX`, `SQL_MAX_TOKENS`, `SQL_MAX_ROWS` are env vars; the local model loads lazily.
- [x] Schema-aware SQLCoder prompt (`app/sql/prompt.py`) that includes the previous SQL and error on retries.
- [x] `sqlglot` validator (`app/sql/validator.py`): one statement, SELECT only, rejects DDL/DML, `SELECT INTO`, `FOR UPDATE`, data-modifying CTEs, and `pg_*` / `lo_*` / `dblink` / `set_config` functions.
- [x] `asyncpg` execution as `querycraft_ro` in a read-only transaction, with the role's statement timeout and a cursor row limit (`app/db.py`).
- [x] Blocking inference runs in `asyncio.to_thread` behind a lock.
- [x] `app` service in `docker-compose.yml` (profile `app`) and `docker-compose.cuda.yml` override.
- [x] `python -m app.cli "question"` runs the full workflow.

Verified: validator cases; full graph in the container and on the host against a fake model server and real Postgres, including a failed query corrected on the second attempt.

Not verified yet:
- [ ] Real SQLCoder output (download ~4 GB, then run `app/scripts/run_llama_server.sh` or use `SQL_BACKEND=local`).
- [ ] `SQL_BACKEND=local` with `llama-cpp-python` on MPS.
- [ ] The CUDA image and compose override (no NVIDIA GPU available here).
- [ ] Prompt quality on JOIN and GROUP BY questions.

How to run it for real (macOS, model on MPS, app in Docker):
1. `python -m app.scripts.download_model`
2. `brew install llama.cpp`, then `app/scripts/run_llama_server.sh`
3. `docker compose --profile app run --rm app python -m app.cli "most expensive product sold"`

Keep in mind:
- `app/sql/prompt.py` holds a hand-copied schema that must stay in sync with `ingestion/schema.sql`.
- The `app` compose service is behind the `app` profile until Step 6 adds the API.
- The CUDA image uses Ubuntu 22.04, so it runs Python 3.10 rather than 3.12.

## Step 4: Persian support (done, in `app/language.py`, `app/workflow/nodes.py`)

- [x] Decision: translation by **Gemma 3 4B instruct** (GGUF Q4_K_M, `ggml-org/gemma-3-4b-it-GGUF`) through llama.cpp, as requested. It runs on CPU by default (`GEMMA_N_GPU_LAYERS=0`), so it never shares GPU memory with SQLCoder.
- [x] Detection by script (any Persian/Arabic letter, so mixed text counts as Persian).
- [x] Preprocessing: Arabic `ك ي` to `ک ی`, Persian/Arabic digits to ASCII, diacritics and tatweel removed, half-spaces to spaces, `؟` to `?`.
- [x] Translation prompt with a schema glossary (محصول=product, دسته‌بندی=category, ...). Falls back to the normalised Persian text if Gemma returns nothing.
- [x] English questions skip translation. SQLCoder only ever sees English.
- [x] Backends (`GEMMA_BACKEND=local|remote`), download script (`python -m app.scripts.download_model chat`) and `run_llama_server.sh gemma` (port 8081).
- [x] The choices are documented in the README.

## Step 5: relevance check and retry loop (done)

- [x] Relevance classifier: Gemma answers YES/NO against the schema. It says NO for general knowledge, chit-chat, weather, and any request to insert, change or delete data. Irrelevant questions exit before SQL generation.
- [x] On failure, the failing SQL and the Postgres/validation error go into the correction prompt.
- [x] Retries are capped (`MAX_SQL_ATTEMPTS`, default 3), then `success: false` with the last error.

## Step 6: API, tests, README (done)

- [x] `POST /query` in `app/main.py` (async FastAPI) takes `{"question": "..."}` and returns `success`, `sql_query`, `final_answer` (raw rows). Empty question gives 422. A model or DB outage gives 503 with `success: false`.
- [x] `app` service in `docker-compose.yml` (port 8000). It starts after `ingestion` finishes, which creates the read-only role. `docker-compose.cuda.yml` is the NVIDIA override.
- [x] Dependencies pinned in `app/requirements*.txt`.
- [x] Tests (`tests/`, `python -m pytest`, 38 pass):
  - [x] Persian and English versions of the same question give the same SQL and rows
  - [x] A JOIN across all three tables
  - [x] An aggregation with GROUP BY
  - [x] An irrelevant question (the SQL model is never called)
  - [x] A destructive prompt, rejected by relevance, and destructive SQL blocked by the validator (products count unchanged)
  - [x] A forced SQL error: the correction prompt contains the failing SQL and the Postgres error
  - [x] Retry cap, model outage (503), empty input, and unit tests for language, relevance, prompts and the validator
- [x] README: architecture, setup per hardware, env vars, tests.

## Final test checklist (real models)

1. `python -m app.scripts.download_model`
2. macOS: `app/scripts/run_llama_server.sh sql` and `... gemma` in two terminals, then `docker compose up -d --build`. (CUDA: use `docker-compose.cuda.yml`.)
3. Try:
   - `curl -X POST localhost:8000/query -H 'content-type: application/json' -d '{"question":"What is the most expensive product sold?"}'`
   - The same question in Persian.
   - "What's the weather?" (should be rejected).
   - "Delete all products" (should be rejected).
4. Check: does Gemma's Persian translation use the schema terms? Is SQLCoder's SQL right for JOIN and GROUP BY questions? Does `/v1/chat/completions` on llama-server apply Gemma's chat template correctly?
5. Not verified here: the CUDA image and compose override (no NVIDIA GPU), `SQL_BACKEND=local` with `llama-cpp-python` on MPS, real model output quality.

## Open questions

- Which models does the original architecture document expect for detection, translation and relevance? We chose Gemma for translation and relevance.
- `final_answer` stays as raw rows, following the spec's sample output.
