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

import os
from collections.abc import Awaitable, Callable
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.workflow import nodes
from app.workflow.state import QueryState

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
        graph.add_node(name, fn)

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
    final = await graph.ainvoke({"question": question, "max_attempts": DEFAULT_MAX_ATTEMPTS})
    return final["response"]
