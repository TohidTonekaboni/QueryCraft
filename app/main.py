"""FastAPI app: POST /query  {"question": "..."}  ->  {"success", "sql_query", "final_answer"}"""

import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db import close_pool
from app.logging_config import request_id_var, setup_logging
from app.workflow import build_graph, run_workflow

setup_logging()
logger = logging.getLogger("querycraft.api")
graph = build_graph()

# Only accept a caller-supplied request id that is short and harmless (it ends up in every log line).
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    logger.info("service_started", extra={
        "sql_backend": settings.sql.backend, "sql_model": settings.sql.remote_model
        if settings.sql.backend == "remote" else settings.sql.model_path.name,
        "chat_backend": settings.chat.backend, "chat_model": settings.chat.remote_model
        if settings.chat.backend == "remote" else settings.chat.model_path.name,
        "db_host": settings.db_host, "db_name": settings.db_name, "max_rows": settings.max_rows,
    })
    yield
    await close_pool()
    logger.info("service_stopped")


app = FastAPI(title="QueryCraft", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Give every request an id (header X-Request-ID is reused if valid), log it and echo the id back."""
    supplied = request.headers.get("x-request-id", "")
    request_id = supplied if _SAFE_REQUEST_ID.match(supplied) else uuid.uuid4().hex
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    fields = {"method": request.method, "path": request.url.path,
              "client": request.client.host if request.client else None}
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed", extra={**fields, "duration_ms": _ms(start)})
        raise
    else:
        response.headers["X-Request-ID"] = request_id
        logger.info("request", extra={**fields, "status": response.status_code, "duration_ms": _ms(start)})
        return response
    finally:
        request_id_var.reset(token)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class QueryResponse(BaseModel):
    success: bool
    sql_query: str | None
    final_answer: Any  # raw rows on success, a message otherwise


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        return await run_workflow(request.question, graph)
    except Exception:  # model server down, DB unreachable, ...: log details, return a clean error
        logger.exception("workflow_failed")
        return JSONResponse(
            status_code=503,
            content={"success": False, "sql_query": None,
                     "final_answer": "Service temporarily unavailable (a model or the database is unreachable)."},
        )
