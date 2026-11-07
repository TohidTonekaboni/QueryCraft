"""Workflow nodes.

All nodes are async and return a partial state update. the graph
wiring in graph.py does not depend on their internals.
"""

import asyncpg

from app import metrics
from app.db import run_select
from app.language import build_translation_prompt, clean_translation, detect_language, normalize_persian
from app.relevance import build_relevance_prompt, parse_relevance
from app.sql.llm import get_chat_model, get_sql_model
from app.sql.prompt import STOP, build_prompt, extract_sql
from app.sql.validator import check_sql
from app.workflow.state import QueryState


async def input_node(state: QueryState) -> QueryState:
    """Receive and normalise the question; initialise counters."""
    question = (state.get("question") or "").strip()
    return {
        "question": question,
        "attempts": 0,
        "error": None,
        "max_attempts": state.get("max_attempts", 3),
    }


async def language_node(state: QueryState) -> QueryState:
    """Detect Persian vs. English; for Persian, normalise and translate to English with Gemma."""
    question = state["question"]
    language = detect_language(question)
    if language == "en":
        return {"language": language, "translated_question": question}
    persian = normalize_persian(question)
    translated = clean_translation(await get_chat_model().chat(build_translation_prompt(persian)))
    # If Gemma returns nothing usable, fall back to the normalised Persian text.
    return {"language": language, "translated_question": translated or persian}


async def relevance_node(state: QueryState) -> QueryState:
    """Ask Gemma whether the question can be answered from the schema."""
    question = state["translated_question"]
    if not question:
        metrics.RELEVANCE_REJECTIONS.inc()
        return {"is_relevant": False}
    answer = await get_chat_model().chat(build_relevance_prompt(question))
    relevant = parse_relevance(answer)
    if not relevant:
        metrics.RELEVANCE_REJECTIONS.inc()
    return {"is_relevant": relevant}


async def generate_node(state: QueryState) -> QueryState:
    """Generate a PostgreSQL query with SQLCoder. On retries the previous SQL and
    its validation/execution error are included in the prompt."""
    retry = state.get("attempts", 0) > 0
    prompt = build_prompt(
        state["translated_question"],
        previous_sql=state.get("sql_query") if retry else None,
        error=state.get("error") if retry else None,
    )
    raw = await get_sql_model().complete(prompt, stop=STOP)
    return {"sql_query": extract_sql(raw), "attempts": state.get("attempts", 0) + 1}


async def validate_node(state: QueryState) -> QueryState:
    """Check the query before it touches the database (single read-only SELECT)."""
    rejection = check_sql(state["sql_query"])
    if rejection:
        metrics.VALIDATION_REJECTIONS.labels(rejection[0]).inc()
    return {"is_valid": rejection is None, "error": rejection[1] if rejection else None}


async def execute_node(state: QueryState) -> QueryState:
    """Run the query as the read-only role; errors go back to the correction loop."""
    try:
        return {"rows": await run_select(state["sql_query"]), "error": None}
    except (asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def correct_node(state: QueryState) -> QueryState:
    """Prepare a retry after a validation/execution failure.

    The error and failing SQL stay in state, so generate_node can include them
    in its correction prompt. The retry cap is enforced by the router.
    """
    return {}


async def respond_node(state: QueryState) -> QueryState:
    """Build the API response: success, sql_query, final_answer."""
    if not state.get("is_relevant", True):
        response = {
            "success": False,
            "sql_query": None,
            "final_answer": (
                "This question cannot be answered from the database. Ask a read-only question "
                "about products, categories or sales."
            ),
        }
    elif state.get("error"):
        response = {
            "success": False,
            "sql_query": state.get("sql_query"),
            "final_answer": (
                f"Could not produce a working query after {state.get('attempts', 0)} "
                f"attempt(s). Last error: {state['error']}"
            ),
        }
    else:
        response = {
            "success": True,
            "sql_query": state["sql_query"],
            "final_answer": state.get("rows", []),
        }
    return {"response": response}
