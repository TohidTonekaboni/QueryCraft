"""Workflow nodes.

All nodes are async and return a partial state update. the graph
wiring in graph.py does not depend on their internals.
"""

import re

from app.workflow.state import QueryState

_PERSIAN_CHARS = re.compile(r"[؀-ۿ]")


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
    """Detect Persian vs. English, preprocess, translate Persian -> English.

    TODO(step 4): replace with a local language detector and translation model.
    """
    question = state["question"]
    language = "fa" if _PERSIAN_CHARS.search(question) else "en"
    # STUB: no translation yet, the text is passed through unchanged.
    return {"language": language, "translated_question": question}


async def relevance_node(state: QueryState) -> QueryState:
    """Decide whether the question can be answered from the schema.

    TODO(step 5): replace with a local relevance classifier.
    """
    # STUB: every non-empty question is considered relevant.
    return {"is_relevant": bool(state["translated_question"])}


async def generate_node(state: QueryState) -> QueryState:
    """Generate a PostgreSQL query. On retries, state['error'] and the previous
    state['sql_query'] are available for the correction prompt.

    TODO(step 3): replace with SQLCoder-7B (GGUF) via llama-cpp-python.
    """
    # STUB: fixed query.
    return {
        "sql_query": "SELECT name, price FROM products ORDER BY price DESC LIMIT 1",
        "attempts": state.get("attempts", 0) + 1,
    }


async def validate_node(state: QueryState) -> QueryState:
    """Check the query before it touches the database.

    TODO(step 3): sqlglot validation (single SELECT only, no DDL/DML).
    """
    # STUB: accepts anything that starts with SELECT.
    if state["sql_query"].lstrip().lower().startswith("select"):
        return {"is_valid": True, "error": None}
    return {"is_valid": False, "error": "Only SELECT statements are allowed."}


async def execute_node(state: QueryState) -> QueryState:
    """Run the query on PostgreSQL.

    TODO(step 3): execute with asyncpg using the read-only role, statement
    timeout and row limit. On failure return {"error": str(exc)}.
    """
    # STUB: no DB access, returns no rows.
    return {"rows": [], "error": None}


async def correct_node(state: QueryState) -> QueryState:
    """Prepare a retry after a validation/execution failure.

    The error and failing SQL stay in state, so generate_node can include them
    in its correction prompt. The retry cap is enforced by the router.

    TODO(step 5): add error-specific hints to the correction prompt if useful.
    """
    return {}


async def respond_node(state: QueryState) -> QueryState:
    """Build the API response: success, sql_query, final_answer."""
    if not state.get("is_relevant", True):
        response = {
            "success": False,
            "sql_query": None,
            "final_answer": "The question is not related to the database.",
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
