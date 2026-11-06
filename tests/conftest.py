"""Test fixtures.

The two language models are replaced by deterministic fakes, so the suite needs no model
weights. The SQL validator, the LangGraph workflow, FastAPI and PostgreSQL (the seeded
`docker compose up -d postgres ingestion` database, read-only role) are all real.
"""

import re

import asyncpg
import httpx
import pytest
import pytest_asyncio

from app.config import get_settings
from app.db import close_pool
from app.language import normalize_persian
from app.main import app
from app.sql.llm import set_models

# Persian question -> its English translation (what Gemma would produce).
# Keys are normalised, because the app sends normalised Persian to the model.
TRANSLATIONS = {
    normalize_persian("گران‌ترین محصول فروخته شده چیست؟"): "What is the most expensive product sold?",
}

BAD_SQL = "SELECT name FROM prodcts"  # misspelled table -> Postgres error
JOIN_SQL = (
    "SELECT p.name, p.price FROM order_items oi JOIN products p ON p.id = oi.product_id "
    "JOIN categories c ON c.id = p.category_id ORDER BY p.price DESC LIMIT 1;"
)
GROUP_SQL = (
    "SELECT c.name AS category, SUM(oi.quantity * oi.unit_price) AS revenue "
    "FROM order_items oi JOIN products p ON p.id = oi.product_id "
    "JOIN categories c ON c.id = p.category_id GROUP BY c.name ORDER BY revenue DESC;"
)

# English question -> SQL the fake SQLCoder returns.
SQL_BY_QUESTION = {
    "what is the most expensive product sold?": JOIN_SQL,
    "total revenue per category": GROUP_SQL,
    "how many products are there?": "SELECT COUNT(*) AS n FROM products;",
    "delete all products": "DELETE FROM products;",  # only reachable if relevance is bypassed
}


class FakeChat:
    """Stands in for Gemma: translation and relevance."""

    def __init__(self):
        self.relevance_override: bool | None = None

    async def chat(self, user_message: str) -> str:
        if user_message.startswith("Translate"):
            persian = user_message.rsplit("Persian:", 1)[1].split("\nEnglish:")[0].strip()
            return f"English: {TRANSLATIONS[persian]}"
        question = user_message.rsplit("Question:", 1)[1].split("\nAnswer:")[0].strip().lower()
        if self.relevance_override is not None:
            return "YES" if self.relevance_override else "NO"
        blocked = ("weather", "delete", "drop", "insert", "update", "joke")
        return "NO" if any(w in question for w in blocked) else "YES."

    async def complete(self, prompt, *, stop=None):  # pragma: no cover
        raise NotImplementedError


class FakeSQL:
    """Stands in for SQLCoder; records every prompt it receives."""

    def __init__(self):
        self.prompts: list[str] = []
        self.mode = "normal"  # "normal" | "bad_then_good" | "always_bad" | "destructive"

    async def complete(self, prompt: str, *, stop=None) -> str:
        self.prompts.append(prompt)
        question = re.search(r"\[QUESTION\](.*?)\[/QUESTION\]", prompt).group(1).strip().lower()
        retry = "A previous attempt failed" in prompt
        if self.mode == "always_bad":
            return BAD_SQL
        if self.mode == "bad_then_good":
            return SQL_BY_QUESTION[question] if retry else BAD_SQL
        if self.mode == "destructive":
            return "DROP TABLE products;"
        return SQL_BY_QUESTION[question]

    async def chat(self, user_message):  # pragma: no cover
        raise NotImplementedError


@pytest.fixture
def fake_chat():
    chat = FakeChat()
    return chat


@pytest.fixture
def fake_sql():
    return FakeSQL()


@pytest_asyncio.fixture(loop_scope="session")
async def db_ready():
    s = get_settings()
    try:
        conn = await asyncpg.connect(host=s.db_host, port=s.db_port, database=s.db_name,
                                     user=s.db_user, password=s.db_password, timeout=3)
        await conn.close()
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable ({exc!r}); run: docker compose up -d postgres ingestion")


@pytest_asyncio.fixture(loop_scope="session")
async def client(db_ready, fake_chat, fake_sql):
    set_models(sql=fake_sql, chat=fake_chat)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture(loop_scope="session", scope="session", autouse=True)
async def _close_pool_at_end():
    yield
    await close_pool()
