"""Async query execution as the read-only role."""

import asyncio
import datetime as dt
import decimal
import logging
import time
import uuid
from typing import Any

import asyncpg

from app import metrics
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


metrics.bind_pool_gauges(lambda: _pool)


async def ping() -> bool:
    """Health probe: can the read-only role run a trivial query? Updates querycraft_db_up."""
    try:
        pool = await get_pool()
        async with pool.acquire(timeout=5) as conn:
            await conn.fetchval("SELECT 1", timeout=5)
    except Exception as exc:
        logger.warning("db_probe_failed", extra={"error_type": type(exc).__name__})
        metrics.DB_UP.set(0)
        return False
    metrics.DB_UP.set(1)
    return True


async def run_select(sql: str) -> list[dict[str, Any]]:
    """Run a validated SELECT in a read-only transaction, returning at most max_rows rows.

    The role's statement_timeout bounds run time; a cursor bounds the rows fetched,
    so the query text itself is never rewritten.
    """
    max_rows = get_settings().max_rows
    start = time.perf_counter()
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction(readonly=True):
                cursor = await conn.cursor(sql)
                records = await cursor.fetch(max_rows + 1)  # one extra row tells us whether we truncated
    except asyncpg.PostgresError as exc:
        metrics.DB_QUERY_ERRORS.labels(type(exc).__name__).inc()
        if isinstance(exc, asyncpg.QueryCanceledError):
            metrics.DB_STATEMENT_TIMEOUTS.inc()
        raise
    finally:
        metrics.DB_QUERY_DURATION.observe(time.perf_counter() - start)
    if len(records) > max_rows:
        metrics.RESULT_TRUNCATED.inc()
        logger.info("result_truncated", extra={"max_rows": max_rows})
        records = records[:max_rows]
    return [{k: _jsonable(v) for k, v in r.items()} for r in records]
