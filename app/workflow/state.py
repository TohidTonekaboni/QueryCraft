"""Shared state passed between LangGraph nodes."""

from typing import Any, Literal, TypedDict


class QueryState(TypedDict, total=False):
    # input
    question: str

    # language processing
    language: Literal["fa", "en"]
    translated_question: str  # English text used for the rest of the pipeline

    # relevance check
    is_relevant: bool

    # SQL generation / validation / execution
    sql_query: str
    is_valid: bool
    rows: list[dict[str, Any]]

    # error handling / self-correction
    error: str | None  # latest validation or execution error, fed back to the generator
    attempts: int  # number of SQL generations so far
    max_attempts: int

    # output
    response: dict[str, Any]  # {"success", "sql_query", "final_answer"}
