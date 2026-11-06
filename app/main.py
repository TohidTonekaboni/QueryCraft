"""FastAPI app: POST /query  {"question": "..."}  ->  {"success", "sql_query", "final_answer"}"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.db import close_pool
from app.workflow import build_graph, run_workflow

logger = logging.getLogger("querycraft")
graph = build_graph()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await close_pool()


app = FastAPI(title="QueryCraft", lifespan=lifespan)


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
        logger.exception("workflow failed")
        return JSONResponse(
            status_code=503,
            content={"success": False, "sql_query": None,
                     "final_answer": "Service temporarily unavailable (a model or the database is unreachable)."},
        )
