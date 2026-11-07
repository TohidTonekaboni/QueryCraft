"""LangGraph workflow: routing, shared state and the self-correction loop.

    input -> language -> relevance --irrelevant--> respond
                             |
                          relevant
                             v
                 +------> generate -> validate --invalid--+
                 |                       |                |
                 |                     valid              |
                 |                       v                |
                 |                    execute --error-----+
                 |                       |                |
                 |                       ok               v
                 |                       v             correct --max attempts--> respond
                 |                    respond             |
                 +----------------------------------------+
"""

import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.logging_config import error_kind, text_fields
from app.workflow import nodes
from app.workflow.state import QueryState

logger = logging.getLogger("querycraft.workflow")

NodeFn = Callable[[QueryState], Awaitable[QueryState]]

DEFAULT_MAX_ATTEMPTS = int(os.getenv("MAX_SQL_ATTEMPTS", "3"))


def route_after_relevance(state: QueryState) -> Literal["generate", "respond"]:
    return "generate" if state["is_relevant"] else "respond"


def route_after_validate(state: QueryState) -> Literal["execute", "correct"]:
    return "execute" if state["is_valid"] else "correct"


def route_after_execute(state: QueryState) -> Literal["respond", "correct"]:
    return "correct" if state.get("error") else "respond"


def route_after_correct(state: QueryState) -> Literal["generate", "respond"]:
    # Out of retries: respond() reports success=False with state['error'].
    return "respond" if state["attempts"] >= state["max_attempts"] else "generate"


def _node_fields(update: QueryState) -> dict:
    """What a node decided, as log fields. User text and SQL appear only when LOG_SENSITIVE is on."""
    fields: dict = {}
    for key in ("language", "is_relevant", "is_valid", "attempts"):
        if key in update:
            fields[key] = update[key]
    if "rows" in update:
        fields["row_count"] = len(update["rows"])
    if "translated_question" in update:
        fields.update(text_fields("translated_question", update["translated_question"]))
    if update.get("sql_query"):
        fields.update(text_fields("sql", update["sql_query"], limit=1000))
    if update.get("error"):
        fields["error_kind"] = error_kind(update["error"])
        fields.update(text_fields("error", update["error"], limit=300))
    return fields


def _instrument(name: str, fn: NodeFn) -> NodeFn:
    """Log how long each node took and what it decided, and log failures with the traceback."""

    async def wrapper(state: QueryState) -> QueryState:
        start = time.perf_counter()
        try:
            update = await fn(state)
        except Exception:
            logger.exception("node_failed", extra={"node": name, "duration_ms": _ms(start)})
            raise
        logger.info("node_completed", extra={"node": name, "duration_ms": _ms(start), **_node_fields(update)})
        return update

    return wrapper


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def build_graph(overrides: dict[str, NodeFn] | None = None) -> CompiledStateGraph:
    """Build the workflow. `overrides` swaps node implementations by name
    (e.g. {"execute": fake_execute}); used by tests and later steps."""
    impl: dict[str, NodeFn] = {
        "input": nodes.input_node,
        "language": nodes.language_node,
        "relevance": nodes.relevance_node,
        "generate": nodes.generate_node,
        "validate": nodes.validate_node,
        "execute": nodes.execute_node,
        "correct": nodes.correct_node,
        "respond": nodes.respond_node,
        **(overrides or {}),
    }

    graph = StateGraph(QueryState)
    for name, fn in impl.items():
        graph.add_node(name, _instrument(name, fn))

    graph.add_edge(START, "input")
    graph.add_edge("input", "language")
    graph.add_edge("language", "relevance")
    graph.add_conditional_edges("relevance", route_after_relevance)
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges("validate", route_after_validate)
    graph.add_conditional_edges("execute", route_after_execute)
    graph.add_conditional_edges("correct", route_after_correct)
    graph.add_edge("respond", END)
    return graph.compile()


async def run_workflow(question: str, graph: CompiledStateGraph | None = None) -> dict:
    """Run one question through the workflow and return the API response dict."""
    graph = graph or build_graph()
    start = time.perf_counter()
    logger.info("query_received", extra=text_fields("question", question))
    final = await graph.ainvoke({"question": question, "max_attempts": DEFAULT_MAX_ATTEMPTS})
    response = final["response"]
    logger.info("query_completed", extra={
        "success": response["success"], "language": final.get("language"),
        "relevant": final.get("is_relevant"), "attempts": final.get("attempts"),
        "row_count": len(final["rows"]) if response["success"] else None,
        "failure": None if response["success"] else (
            "irrelevant" if final.get("is_relevant") is False else error_kind(final.get("error"))),
        "duration_ms": _ms(start)})
    return response
