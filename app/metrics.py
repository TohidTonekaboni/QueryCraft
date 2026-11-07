"""Prometheus metrics, served on a separate internal port (the API itself stays a single POST /query).

Environment variables:
    METRICS_PORT            port of the metrics server; 0 disables it            (default 9100)
    METRICS_ADDR            interface to bind                                    (default 127.0.0.1;
                            the Docker compose file sets 0.0.0.0 and publishes the port on localhost only)
    METRICS_PROBE_INTERVAL  seconds between model/database health probes         (default 30)
    APP_VERSION             shown in querycraft_build_info                       (default dev)

Label values are always from small fixed sets (node names, rule names, error classes, route templates);
never put a question, SQL text or request id in a label. Run a single uvicorn worker, or switch
prometheus_client to multiprocess mode, otherwise each worker would expose its own counters.
"""

import logging
import os

from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger("querycraft.metrics")

NS = "querycraft"

# Seconds. Model calls take seconds, so the top buckets are higher than the client default.
_SLOW = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120)
_FAST = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)

# --- API (RED) ---
HTTP_REQUESTS = Counter(f"{NS}_http_requests", "HTTP requests handled.", ["method", "path", "status"])
HTTP_DURATION = Histogram(f"{NS}_http_request_duration_seconds", "HTTP request latency.", ["method", "path"],
                          buckets=_SLOW)
HTTP_IN_PROGRESS = Gauge(f"{NS}_http_requests_in_progress", "HTTP requests currently being served.")

# --- query outcomes ---
QUERIES = Counter(f"{NS}_queries", "Questions processed, by outcome (success, irrelevant, failed, error) "
                  "and language (fa, en, unknown).", ["outcome", "language"])
QUERY_FAILURES = Counter(f"{NS}_query_failures", "Failed questions, by reason: the last validation or execution "
                         "error after retries ran out, or model_error / db_error / internal when the workflow raised.",
                         ["reason"])
QUERY_ATTEMPTS = Histogram(f"{NS}_query_attempts", "SQL generation attempts per question that reached SQL generation.",
                           buckets=(1, 2, 3, 4, 5))
QUERY_DURATION = Histogram(f"{NS}_query_duration_seconds", "End-to-end workflow time.", ["outcome"], buckets=_SLOW)
RESULT_ROWS = Histogram(f"{NS}_result_rows", "Rows returned by successful queries.",
                        buckets=(0, 1, 2, 5, 10, 25, 50, 100, 250, 500))
RESULT_TRUNCATED = Counter(f"{NS}_result_truncated", "Queries whose result was cut at SQL_MAX_ROWS.")

# --- workflow nodes ---
NODE_DURATION = Histogram(f"{NS}_node_duration_seconds", "Time spent in each workflow node.", ["node"], buckets=_SLOW)
NODE_ERRORS = Counter(f"{NS}_node_errors", "Workflow nodes that raised.", ["node"])
CORRECTION_LOOPS = Counter(f"{NS}_correction_loops", "Retries sent back to SQL generation, by what failed "
                           "(validation or execution).", ["trigger"])

# --- models ---
MODEL_CALLS = Counter(f"{NS}_model_calls", "Model calls.", ["model", "kind", "status"])
MODEL_DURATION = Histogram(f"{NS}_model_call_duration_seconds", "Model call latency.", ["model", "kind"],
                           buckets=_SLOW)
MODEL_PROMPT_CHARS = Histogram(f"{NS}_model_prompt_chars", "Prompt size sent to a model, in characters.", ["model"],
                               buckets=(250, 500, 1000, 2000, 4000, 8000, 16000))
MODEL_UP = Gauge(f"{NS}_model_up", "1 if the model server was reachable on the last probe or call.", ["model"])

# --- SQL validation and safety ---
VALIDATION_REJECTIONS = Counter(f"{NS}_validation_rejections", "Generated SQL rejected by the validator, by rule.",
                                ["rule"])
RELEVANCE_REJECTIONS = Counter(f"{NS}_relevance_rejections", "Questions refused before SQL generation.")

# --- database ---
DB_QUERY_DURATION = Histogram(f"{NS}_db_query_duration_seconds", "Time to run a generated query.", buckets=_FAST)
DB_QUERY_ERRORS = Counter(f"{NS}_db_query_errors", "Generated queries that failed in PostgreSQL, by error class.",
                          ["error_class"])
DB_STATEMENT_TIMEOUTS = Counter(f"{NS}_db_statement_timeouts", "Queries cancelled by the read-only role's timeout.")
DB_POOL_SIZE = Gauge(f"{NS}_db_pool_size", "Connections currently open in the pool.")
DB_POOL_IDLE = Gauge(f"{NS}_db_pool_idle", "Idle connections in the pool.")
DB_UP = Gauge(f"{NS}_db_up", "1 if the last database probe (SELECT 1) succeeded.")

# --- build ---
BUILD_INFO = Gauge(f"{NS}_build_info", "Always 1; labels describe this deployment.",
                   ["version", "sql_model", "chat_model", "sql_backend"])


def failure_reason(exc: BaseException) -> str:
    """Fixed label for an exception that escaped the workflow."""
    import asyncpg

    from app.sql.llm import ModelServerError

    if isinstance(exc, ModelServerError):
        return "model_error"
    if isinstance(exc, (asyncpg.PostgresError, asyncpg.InterfaceError, OSError)):
        return "db_error"
    return "internal"


def bind_pool_gauges(get_pool) -> None:
    """Report pool usage at scrape time; `get_pool` returns the current pool or None."""
    DB_POOL_SIZE.set_function(lambda: get_pool().get_size() if get_pool() else 0)
    DB_POOL_IDLE.set_function(lambda: get_pool().get_idle_size() if get_pool() else 0)


def set_build_info(sql_backend: str, sql_model: str, chat_model: str) -> None:
    BUILD_INFO.labels(os.getenv("APP_VERSION", "dev"), sql_model, chat_model, sql_backend).set(1)


def start_metrics_server():
    """Start the metrics HTTP server in a background thread; returns it, or None when disabled."""
    port = int(os.getenv("METRICS_PORT", "9100"))
    if port == 0:
        logger.info("metrics_disabled")
        return None
    addr = os.getenv("METRICS_ADDR", "127.0.0.1")
    server, _ = start_http_server(port, addr)
    logger.info("metrics_server_started", extra={"addr": addr, "port": port})
    return server
