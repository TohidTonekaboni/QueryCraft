"""End-to-end tests of POST /query (fake language models, real validator/graph/API/PostgreSQL)."""

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")

EN_QUESTION = "What is the most expensive product sold?"
FA_QUESTION = "گران‌ترین محصول فروخته شده چیست؟"


async def ask(client, question: str):
    resp = await client.post("/query", json={"question": question})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"success", "sql_query", "final_answer"}
    return body


async def test_english_question(client):
    body = await ask(client, EN_QUESTION)
    assert body["success"] is True
    assert body["sql_query"].lower().startswith("select")
    assert len(body["final_answer"]) == 1
    assert {"name", "price"} <= set(body["final_answer"][0])


async def test_persian_and_english_give_same_answer(client, fake_sql):
    en = await ask(client, EN_QUESTION)
    fa = await ask(client, FA_QUESTION)
    assert fa["success"] is True
    assert fa["sql_query"] == en["sql_query"]
    assert fa["final_answer"] == en["final_answer"]
    # SQLCoder only ever sees the English translation
    assert all("محصول" not in p for p in fake_sql.prompts)


async def test_join_across_all_three_tables(client):
    body = await ask(client, EN_QUESTION)
    sql = body["sql_query"].lower()
    assert body["success"] is True
    for table in ("categories", "products", "order_items"):
        assert table in sql


async def test_group_by_aggregation(client):
    body = await ask(client, "Total revenue per category")
    assert body["success"] is True
    assert "group by" in body["sql_query"].lower()
    revenues = [r["revenue"] for r in body["final_answer"]]
    assert len(revenues) == 5  # one row per seeded category
    assert revenues == sorted(revenues, reverse=True)


async def test_irrelevant_question_is_rejected_before_sql_generation(client, fake_sql):
    body = await ask(client, "What's the weather today?")
    assert body["success"] is False
    assert body["sql_query"] is None
    assert fake_sql.prompts == []  # the SQL model was never called


async def test_destructive_request_is_rejected_by_relevance(client, fake_sql):
    body = await ask(client, "Delete all products")
    assert body["success"] is False
    assert fake_sql.prompts == []


async def test_destructive_sql_is_blocked_by_validator(client, fake_chat, fake_sql):
    """Even if relevance and the model both misbehave, nothing destructive reaches Postgres."""
    fake_chat.relevance_override = True
    fake_sql.mode = "destructive"
    body = await ask(client, "Delete all products")
    assert body["success"] is False
    assert "Only SELECT" in body["final_answer"]
    assert len(fake_sql.prompts) == 3  # retried up to the cap, never executed
    fake_sql.mode = "normal"
    count = await ask(client, "How many products are there?")
    assert count["final_answer"] == [{"n": 1000}]


async def test_sql_error_triggers_correction_loop(client, fake_sql):
    fake_sql.mode = "bad_then_good"
    body = await ask(client, EN_QUESTION)
    assert body["success"] is True
    assert len(fake_sql.prompts) == 2
    retry_prompt = fake_sql.prompts[1]
    assert "prodcts" in retry_prompt  # the failing SQL...
    assert 'relation "prodcts" does not exist' in retry_prompt  # ...and the Postgres error


async def test_retries_are_capped(client, fake_sql):
    fake_sql.mode = "always_bad"
    body = await ask(client, EN_QUESTION)
    assert body["success"] is False
    assert len(fake_sql.prompts) == 3
    assert "3 attempt(s)" in body["final_answer"]


async def test_model_outage_returns_503(client, fake_chat):
    async def boom(_):
        raise RuntimeError("model server down")
    fake_chat.chat = boom
    resp = await client.post("/query", json={"question": EN_QUESTION})
    assert resp.status_code == 503
    assert resp.json()["success"] is False


async def test_empty_question_is_a_validation_error(client):
    resp = await client.post("/query", json={"question": ""})
    assert resp.status_code == 422
