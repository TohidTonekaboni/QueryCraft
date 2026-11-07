"""Async query execution as the read-only role."""

import asyncio
import datetime as dt
import decimal
import logging
import uuid
from typing import Any

import asyncpg

from app.config import get_settings

logger = logging.getLogger("querycraft.db")

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()


async def get_pool() -> asyncpg.Pool:
    global _pool
    async with _pool_lock:
        if _pool is None:
            s = get_settings()
            _pool = await asyncpg.create_pool(
                host=s.db_host, port=s.db_port, database=s.db_name,
                user=s.db_user, password=s.db_password,
                min_size=1, max_size=5,
            )
            logger.info("db_pool_created", extra={"host": s.db_host, "port": s.db_port, "database": s.db_name,
                                                  "user": s.db_user, "max_size": 5})
        return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("db_pool_closed")


def _jsonable(value: Any) -> Any:
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, (uuid.UUID, dt.timedelta)):
        return str(value)
    return value


async def run_select(sql: str) -> list[dict[str, Any]]:
    """Run a validated SELECT in a read-only transaction, returning at most max_rows rows.

    The role's statement_timeout bounds run time; a cursor bounds the rows fetched,
    so the query text itself is never rewritten.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction(readonly=True):
            cursor = await conn.cursor(sql)
            records = await cursor.fetch(get_settings().max_rows)
    return [{k: _jsonable(v) for k, v in r.items()} for r in records]
