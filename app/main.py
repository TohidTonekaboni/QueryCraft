"""FastAPI app: POST /query  {"question": "..."}  ->  {"success", "sql_query", "final_answer"}"""

import asyncio
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import metrics
from app.config import get_settings
from app.db import close_pool
from app.logging_config import request_id_var, setup_logging
from app.probes import probe_loop
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
    metrics.set_build_info(
        settings.sql.backend,
        settings.sql.remote_model if settings.sql.backend == "remote" else settings.sql.model_path.name,
        settings.chat.remote_model if settings.chat.backend == "remote" else settings.chat.model_path.name)
    metrics_server = metrics.start_metrics_server()
    probes = asyncio.create_task(probe_loop())
    yield
    probes.cancel()
    if metrics_server:
        metrics_server.shutdown()
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
    status = 500  # what is counted if the request raises
    metrics.HTTP_IN_PROGRESS.inc()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_failed", extra={**fields, "duration_ms": _ms(start)})
        raise
    else:
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        logger.info("request", extra={**fields, "status": status, "duration_ms": _ms(start)})
        return response
    finally:
        # Route template, not the raw URL, so the label set stays small (unknown URLs share one label).
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        metrics.HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
        metrics.HTTP_DURATION.labels(request.method, path).observe(time.perf_counter() - start)
        metrics.HTTP_IN_PROGRESS.dec()
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
