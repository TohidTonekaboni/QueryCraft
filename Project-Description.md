# QueryCraft: Project Overview and Review

A summary and review of the Technical Project Assignment: a multilingual natural-language-to-SQL generation API.

## What the project is

You build an API where someone asks a question about a database in **Persian or English**, and the system writes the PostgreSQL query, runs it, and returns both the SQL and the results. The SQL query is the main deliverable, with the execution result as supporting output.

For example, "گران‌ترین محصول فروخته شده چیست؟" ("What is the most expensive product sold?") should come back as a valid `SELECT` plus the matching rows.

## The workflow (the core of the assignment)

The pipeline is a stateful graph built with **LangGraph**, using LangChain for the model wrappers:

1. **Input processing:** receive the question.
2. **Language processing:** detect Persian vs. English, preprocess, and translate Persian to English when needed.
3. **Relevance check:** decide whether the question can be answered from your schema. "What's the weather?" should be rejected here.
4. **SQL generation:** use SQLCoder-7B or an equivalent model to produce the query.
5. **SQL validation:** check the query before it touches the database.
6. **Execution:** run it on PostgreSQL.
7. **Error handling and self-correction:** if execution fails, feed the error back to the generator and retry, with a capped number of attempts.
8. **Response:** return `success`, `sql_query`, and `final_answer`.

LangGraph is meant to own the conditional routing (for example, irrelevant questions exit early), the shared state, and the correction loop.

## Technical constraints

- **Stack:** FastAPI (async), PostgreSQL, Docker and Docker Compose.
- **Database:** exactly three related tables with proper primary and foreign keys, plus enough sample data to exercise JOINs, aggregations, filters, and sorting.
- **Models:** everything except the SQL-generation model must run **locally**. The SQL model may be local or an external inference service.
- **Efficiency:** use quantization (e.g., GGUF), don't load several large models onto the GPU at once, and make CPU/GPU inference configurable.
- **API:** a single endpoint, `POST /query`, taking `{"question": "..."}`.

## Deliverables

Source code, schema and seed data, the LangGraph workflow, the FastAPI app, Docker config, a model download/prep script, a README, and tests. The tests must cover Persian and English questions, JOINs, invalid or irrelevant queries, and the error-correction loop.

## Review: points to watch

### Ambiguities to decide early

- "The models specified in the original project architecture" refers to a document you don't have. Only SQLCoder-7B is named, so ask what else is expected (language detection, translation, relevance classifier) or make and document your own choices.
- "Everything local except SQL generation" means the translator and relevance checker must be local. That makes Persian-to-English translation the hardest component to get right on modest hardware. Options include a small local LLM such as a quantized Qwen or Gemma, or a dedicated MT model such as NLLB or Opus-MT.
- The sample output shows `final_answer` as raw rows, not a natural-language answer. Follow that format unless told otherwise.

### Things that will matter to evaluators

- **SQL safety:** the validation step is where you can stand out. Allow only a single `SELECT` statement, block DDL/DML (`DROP`, `DELETE`, `UPDATE`), and ideally run queries with a **read-only DB role**, a statement timeout, and a row limit. Parsing with `sqlglot` or `sqlparse` is better than regex.
- **Retry design:** cap retries at 2–3, pass the failing SQL and the Postgres error into the correction prompt, and return a clean `success: false` with a useful message when retries run out.
- **Async correctness:** use `asyncpg` or SQLAlchemy async. Local model inference is blocking, so run it in a thread or process pool so it doesn't stall the event loop.
- **Memory management:** load models lazily or sequentially, and use `llama-cpp-python` with GGUF Q4/Q5 quantization. Make `n_gpu_layers` and device choice environment variables.
- **Reproducibility:** pin dependencies and have a script that downloads the model weights, since you can't ship them in the repo.

### Suggested test matrix

Persian and English versions of the same question, a JOIN across all three tables, an aggregation with GROUP BY, an irrelevant question, a malicious or destructive prompt, and a forced SQL error to prove the correction loop works.

## Suggested build order

1. Schema and seed data, plus Docker Compose with Postgres.
2. The LangGraph skeleton using stub nodes.
3. SQL generation, validation, and execution for English only.
4. Language detection and translation for Persian.
5. The relevance check and the retry loop.
6. The FastAPI endpoint, tests, and README.
