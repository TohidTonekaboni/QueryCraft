"""Unit tests that need no database or models."""

import pytest

from app.language import clean_translation, detect_language, normalize_persian
from app.relevance import parse_relevance
from app.sql.prompt import build_prompt, extract_sql
from app.sql.validator import validate_sql


@pytest.mark.parametrize("text,lang", [
    ("What is the cheapest product?", "en"),
    ("ارزان‌ترین محصول کدام است؟", "fa"),
    ("قیمت iPhone چقدر است", "fa"),
    ("", "en"),
])
def test_detect_language(text, lang):
    assert detect_language(text) == lang


def test_normalize_persian():
    assert normalize_persian("كتاب ۱۲۳ و ٤٥٦ ؟") == "کتاب 123 و 456 ?"
    assert normalize_persian("می‌خواهم   سلام") == "می خواهم سلام"


def test_clean_translation():
    assert clean_translation('\n English: "What is the price?"\nExtra line') == "What is the price?"
    assert clean_translation("   ") == ""


@pytest.mark.parametrize("out,expected", [("YES", True), ("Yes.", True), (" no ", False), ("", False), ("Maybe", False)])
def test_parse_relevance(out, expected):
    assert parse_relevance(out) is expected


def test_extract_sql_strips_markers():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1;"
    assert extract_sql("SELECT 1; [/SQL] trailing") == "SELECT 1;"


def test_retry_prompt_contains_error_and_previous_sql():
    prompt = build_prompt("q", previous_sql="SELECT x", error="column x does not exist")
    assert "SELECT x" in prompt and "column x does not exist" in prompt
    assert "previous attempt" not in build_prompt("q")


@pytest.mark.parametrize("sql", [
    "SELECT * FROM products",
    "WITH t AS (SELECT 1 a) SELECT * FROM t",
    "SELECT 1 UNION SELECT 2",
])
def test_validator_accepts_selects(sql):
    assert validate_sql(sql) is None


@pytest.mark.parametrize("sql", [
    "DROP TABLE products", "DELETE FROM products", "UPDATE products SET price = 0",
    "SELECT 1; DROP TABLE products", "SELECT * INTO x FROM products", "SELECT pg_sleep(10)",
    "SELECT pg_read_file('/etc/passwd')", "SELECT * FROM products FOR UPDATE",
    "WITH d AS (DELETE FROM products RETURNING *) SELECT * FROM d", "SELEC FROM WHERE", "",
])
def test_validator_rejects(sql):
    assert validate_sql(sql) is not None


def test_prompt_schema_is_derived_from_schema_sql():
    from app.sql.prompt import SCHEMA

    for table in ("categories", "products", "order_items"):
        assert f"CREATE TABLE {table} (" in SCHEMA
    assert "IF NOT EXISTS" not in SCHEMA and "CREATE INDEX" not in SCHEMA
    assert "revenue = quantity * unit_price" in SCHEMA  # hint comments survive


def test_prompt_contains_revenue_hint():
    from app.sql.prompt import build_prompt

    prompt = build_prompt("total revenue per category")
    assert "SUM(order_items.quantity * order_items.unit_price)" in prompt
    assert prompt.index("### Hints") < prompt.index("### Answer")


async def test_remote_model_uses_langchain_openai_wrappers():
    import json
    from pathlib import Path

    import httpx

    from app.config import ModelSettings
    from app.sql.llm import RemoteModel

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        if request.url.path.endswith("/chat/completions"):
            choice = {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "YES"}}
            return httpx.Response(200, json={"id": "1", "object": "chat.completion", "created": 0,
                                             "model": "m", "choices": [choice], "usage": usage})
        choice = {"index": 0, "finish_reason": "stop", "text": "SELECT 1;"}
        return httpx.Response(200, json={"id": "1", "object": "text_completion", "created": 0,
                                         "model": "m", "choices": [choice], "usage": usage})

    cfg = ModelSettings("remote", "http://llm.test", "local", "repo", Path("x"), 0, 2048, 128)
    model = RemoteModel(cfg)
    model._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    assert await model.complete("q", stop=["[/SQL]"]) == "SELECT 1;"
    assert await model.chat("hi") == "YES"
    (p1, b1), (p2, b2) = seen
    assert b1["prompt"] == "q"  # a plain string: Ollama rejects a list
    assert p1 == "/v1/completions" and b1["stop"] == ["[/SQL]"] and b1["max_tokens"] == 128
    assert p2 == "/v1/chat/completions" and b2["max_tokens"] == 128 and b2["temperature"] == 0.0


async def test_remote_model_failure_is_a_runtime_error():
    from pathlib import Path

    import httpx

    from app.config import ModelSettings
    from app.sql.llm import RemoteModel

    def handler(request):
        raise httpx.ConnectError("down")

    model = RemoteModel(ModelSettings("remote", "http://llm.test", "local", "repo", Path("x"), 0, 2048, 128))
    model._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="llm.test"):
        await model.chat("hi")
