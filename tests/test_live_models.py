"""Live tests: the real Gemma and SQLCoder (through Ollama), the real validator, workflow, API and PostgreSQL.

Opt-in, because they are slow and need the models:

    python -m app.scripts.setup_ollama && docker compose up -d postgres ingestion
    python -m pytest -m live

Answers are compared with reference SQL run on the same database, never with the model's SQL text, so
harmless differences in how the model writes a query do not fail a test. They skip themselves when
Ollama or a model is missing. LIVE_OLLAMA_URL overrides the default http://localhost:11434.
"""

import os

import httpx
import pytest
import pytest_asyncio

from app import config
from app.db import run_select
from app.main import app
from app.sql import llm

pytestmark = [pytest.mark.live, pytest.mark.asyncio(loop_scope="session")]

OLLAMA_URL = os.getenv("LIVE_OLLAMA_URL", "http://localhost:11434")


@pytest_asyncio.fixture(loop_scope="session")
async def live_client(db_ready):
    settings = config.get_settings()
    wanted = {settings.sql.remote_model, settings.chat.remote_model}
    try:
        tags = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()["models"]
    except Exception as exc:
        pytest.skip(f"Ollama not reachable at {OLLAMA_URL} ({exc!r})")
    have = {m["name"] for m in tags} | {m["name"].removesuffix(":latest") for m in tags}
    if missing := wanted - have:
        pytest.skip(f"Ollama models missing: {sorted(missing)}; run: python -m app.scripts.setup_ollama")

    # Point both models at Ollama on the host, and drop any fakes injected by other tests.
    saved = {k: os.environ.get(k) for k in ("SQL_REMOTE_URL", "GEMMA_REMOTE_URL", "SQL_BACKEND", "GEMMA_BACKEND")}
    os.environ.update(SQL_REMOTE_URL=OLLAMA_URL, GEMMA_REMOTE_URL=OLLAMA_URL, SQL_BACKEND="remote", GEMMA_BACKEND="remote")
    config.get_settings.cache_clear()
    llm._models.clear()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=300) as c:
            yield c
    finally:
        for key, value in saved.items():
            os.environ.pop(key, None) if value is None else os.environ.__setitem__(key, value)
        config.get_settings.cache_clear()
        llm._models.clear()


async def ask(client, question: str) -> dict:
    resp = await client.post("/query", json={"question": question})
    assert resp.status_code == 200, resp.text
    return resp.json()


def values(rows: list[dict]) -> list[tuple]:
    """Rows as tuples of rounded values, ignoring column names and aliases."""
    return [tuple(round(v, 2) if isinstance(v, float) else v for v in row.values()) for row in rows]


async def test_count_question_in_english_and_persian(live_client):
    expected = values(await run_select("SELECT COUNT(*) FROM products WHERE stock = 0"))
    for question in ("How many products have zero stock?", "چند محصول موجودی صفر دارند؟"):
        body = await ask(live_client, question)
        assert body["success"], body
        assert values(body["final_answer"]) == expected, (question, body["sql_query"])


async def test_three_table_join_with_aggregation(live_client):
    expected = values(await run_select(
        "SELECT c.name, SUM(oi.quantity * oi.unit_price) FROM order_items oi "
        "JOIN products p ON p.id = oi.product_id JOIN categories c ON c.id = p.category_id "
        "GROUP BY c.name ORDER BY 2 DESC LIMIT 3"))
    body = await ask(live_client, "What are the top 3 categories by total revenue?")
    assert body["success"], body
    assert [v[:2] for v in values(body["final_answer"])] == expected, body["sql_query"]


async def test_most_expensive_product_sold(live_client):
    expected = await run_select(
        "SELECT p.name FROM products p WHERE EXISTS (SELECT 1 FROM order_items o WHERE o.product_id = p.id) "
        "ORDER BY p.price DESC LIMIT 1")
    for question in ("What is the most expensive product sold?", "گران‌ترین محصول فروخته شده چیست؟"):
        body = await ask(live_client, question)
        assert body["success"], body
        assert body["final_answer"][0]["name"] == expected[0]["name"], (question, body["sql_query"])


@pytest.mark.parametrize("question", ["What's the weather in Tehran today?", "Delete all products from the database"])
async def test_irrelevant_and_destructive_requests_are_refused(live_client, question):
    body = await ask(live_client, question)
    assert body["success"] is False and body["sql_query"] is None
    assert "cannot be answered" in body["final_answer"]
