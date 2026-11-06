# QueryCraft: Next Steps

Based on the build order in [Project-Description.md](Project-Description.md).

## Status

| Step | Status |
|---|---|
| 1. Schema, seed data, Docker Compose with Postgres | Done |
| 2. LangGraph skeleton with stub nodes | Done |
| 3. English SQL generation, validation, execution | Not started |
| 4. Persian detection and translation | Not started |
| 5. Relevance check and retry loop | Not started |
| 6. FastAPI endpoint, tests, README | Not started |

Done so far: `postgres:16` with a volume, `.env` secrets, an `ingestion` service, three tables seeded with 1000 products and 5000 order items, a read-only role, and the LangGraph skeleton with stub nodes.

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

- [ ] Choose the SQL model. SQLCoder-7B in GGUF Q4/Q5 via `llama-cpp-python`, or an external inference service.
- [ ] Write the model download/prep script. Weights can't be shipped in the repo.
- [ ] Make `n_gpu_layers` and the CPU/GPU choice environment variables. Load the model lazily.
- [ ] Build a schema-aware prompt.
- [ ] Validate with `sqlglot`: a single `SELECT` only, no DDL/DML.
- [ ] Execute with `asyncpg` or SQLAlchemy async, using the read-only role, a statement timeout and a row limit.
- [ ] Run blocking inference in a thread or process pool so the event loop isn't stalled.

## Step 4: Persian support

- [ ] Decide the local models for language detection and Persian-to-English translation (the hardest part). Options: a small quantized Qwen or Gemma, or NLLB / Opus-MT.
- [ ] Document the choices. The spec refers to an architecture document we don't have.
- [ ] Don't keep several large models on the GPU at once. Load them sequentially.

## Step 5: relevance check and retry loop

- [ ] Add a local relevance classifier. It must reject questions like "What's the weather?".
- [ ] On execution failure, pass the failing SQL and the Postgres error into the correction prompt. (The loop and state are wired in Step 2; the generator must use them.)
- [x] Cap retries at 2–3, then return `success: false` with a useful message. (Done in Step 2.)

## Step 6: API, tests, README

- [ ] Add async FastAPI `POST /query` taking `{"question": "..."}` and returning `success`, `sql_query` and `final_answer` (raw rows).
- [ ] Add the API as a Docker Compose service.
- [ ] Pin the dependencies.
- [ ] Write tests for:
  - [ ] Persian and English versions of the same question
  - [ ] A JOIN across all three tables
  - [ ] An aggregation with GROUP BY
  - [ ] An irrelevant question
  - [ ] A malicious or destructive prompt
  - [ ] A forced SQL error, to prove the correction loop works
- [ ] Write the README: setup, model download, env vars, CPU/GPU config.

## Open questions

- Which models are expected for detection, translation and relevance?
- Should `final_answer` stay as raw rows? The spec's sample output suggests yes.
