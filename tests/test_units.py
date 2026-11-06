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
