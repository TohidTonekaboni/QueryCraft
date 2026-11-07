"""Prometheus metrics: every metric moves when it should, with labels from fixed sets."""

import socket
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import asyncpg
import httpx
import pytest
from prometheus_client import REGISTRY

from app import db, metrics
from app.config import ModelSettings
from app.probes import probe_model
from app.sql.llm import ModelServerError, RemoteModel
from app.sql.validator import RULES, check_sql

EN_QUESTION = "What is the most expensive product sold?"
FA_QUESTION = "گران‌ترین محصول فروخته شده چیست؟"


def val(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


class Delta:
    """Reads a metric now and reports how much it changed since construction."""

    def __init__(self, name: str, **labels):
        self.name, self.labels, self.before = name, labels, val(name, **labels)

    @property
    def change(self) -> float:
        return val(self.name, **self.labels) - self.before


def cfg(backend="remote", model="my-model") -> ModelSettings:
    return ModelSettings(backend, "http://llm.test", model, "repo", Path("x"), 0, 2048, 128)


# --- per-question outcomes ---

async def test_successful_english_query(client):
    ok = Delta("querycraft_queries_total", outcome="success", language="en")
    http = Delta("querycraft_http_requests_total", method="POST", path="/query", status="200")
    attempts = Delta("querycraft_query_attempts_sum")
    rows = Delta("querycraft_result_rows_count")
    duration = Delta("querycraft_query_duration_seconds_count", outcome="success")
    nodes = {n: Delta("querycraft_node_duration_seconds_count", node=n)
             for n in ("input", "language", "relevance", "generate", "validate", "execute", "respond")}
    in_progress = val("querycraft_http_requests_in_progress")

    assert (await client.post("/query", json={"question": EN_QUESTION})).status_code == 200

    assert ok.change == 1 and http.change == 1 and attempts.change == 1 and rows.change == 1 and duration.change == 1
    assert all(d.change == 1 for d in nodes.values())
    assert val("querycraft_http_requests_in_progress") == in_progress  # back to where it was
    assert Delta("querycraft_http_request_duration_seconds_count", method="POST", path="/query").before > 0


async def test_persian_query_is_labelled_fa(client):
    fa = Delta("querycraft_queries_total", outcome="success", language="fa")
    await client.post("/query", json={"question": FA_QUESTION})
    assert fa.change == 1


async def test_irrelevant_question(client):
    irrelevant = Delta("querycraft_queries_total", outcome="irrelevant", language="en")
    refused = Delta("querycraft_relevance_rejections_total")
    attempts = Delta("querycraft_query_attempts_count")
    await client.post("/query", json={"question": "What's the weather today?"})
    assert irrelevant.change == 1 and refused.change == 1
    assert attempts.change == 0  # never reached SQL generation


async def test_validation_rejection_and_correction_loop(client, fake_sql):
    fake_sql.mode = "unknown_table_then_good"
    rejected = Delta("querycraft_validation_rejections_total", rule="forbidden_table")
    loop = Delta("querycraft_correction_loops_total", trigger="validation")
    attempts = Delta("querycraft_query_attempts_sum")
    await client.post("/query", json={"question": EN_QUESTION})
    assert rejected.change == 1 and loop.change == 1 and attempts.change == 2


async def test_execution_error_and_correction_loop(client, fake_sql):
    fake_sql.mode = "bad_then_good"
    error = Delta("querycraft_db_query_errors_total", error_class="UndefinedColumnError")
    loop = Delta("querycraft_correction_loops_total", trigger="execution")
    await client.post("/query", json={"question": EN_QUESTION})
    assert error.change == 1 and loop.change == 1


async def test_retries_exhausted_counts_a_failure_and_only_real_retries(client, fake_sql):
    fake_sql.mode = "always_bad"
    failed = Delta("querycraft_queries_total", outcome="failed", language="en")
    reason = Delta("querycraft_query_failures_total", reason="execution")
    loops = Delta("querycraft_correction_loops_total", trigger="execution")
    await client.post("/query", json={"question": EN_QUESTION})
    assert failed.change == 1 and reason.change == 1
    assert loops.change == 2  # attempts 1 and 2 retry; the third has nowhere to go


async def test_destructive_sql_failure_reason_is_validation(client, fake_sql):
    fake_sql.mode = "destructive"
    not_select = Delta("querycraft_validation_rejections_total", rule="not_select")
    reason = Delta("querycraft_query_failures_total", reason="validation")
    await client.post("/query", json={"question": EN_QUESTION})
    assert not_select.change == 3 and reason.change == 1


async def test_model_outage_is_counted_as_error(client, fake_chat):
    async def down(_):
        raise ModelServerError("model server down")

    fake_chat.chat = down
    errors = Delta("querycraft_queries_total", outcome="error", language="unknown")
    reason = Delta("querycraft_query_failures_total", reason="model_error")
    node = Delta("querycraft_node_errors_total", node="relevance")  # language node is skipped for English
    http = Delta("querycraft_http_requests_total", method="POST", path="/query", status="503")
    resp = await client.post("/query", json={"question": EN_QUESTION})
    assert resp.status_code == 503
    assert errors.change == 1 and reason.change == 1 and node.change == 1 and http.change == 1


async def test_unknown_url_uses_a_single_path_label(client):
    unmatched = Delta("querycraft_http_requests_total", method="GET", path="unmatched", status="404")
    await client.get("/no/such/page/123")
    await client.get("/another/one")
    assert unmatched.change == 2
    assert not any(s.labels.get("path") == "/no/such/page/123"
                   for m in REGISTRY.collect() for s in m.samples)


# --- database ---

def settings_with(max_rows: int):
    return lambda: SimpleNamespace(max_rows=max_rows)


async def test_result_truncation_is_detected_exactly(db_ready, monkeypatch):
    monkeypatch.setattr(db, "get_settings", settings_with(5))
    truncated = Delta("querycraft_result_truncated_total")
    assert len(await db.run_select("SELECT id FROM products")) == 5
    assert truncated.change == 1
    assert len(await db.run_select("SELECT id FROM products LIMIT 5")) == 5  # exactly the cap: not truncated
    assert truncated.change == 1


async def test_pool_gauges_and_ping(db_ready):
    assert await db.ping() is True
    assert val("querycraft_db_up") == 1
    assert val("querycraft_db_pool_size") >= 1
    assert val("querycraft_db_pool_idle") >= 0


async def test_statement_timeout_is_counted(monkeypatch):
    class Pool:
        @asynccontextmanager
        async def acquire(self):
            raise asyncpg.QueryCanceledError("canceling statement due to statement timeout")
            yield

    async def fake_pool():
        return Pool()

    monkeypatch.setattr(db, "get_pool", fake_pool)
    monkeypatch.setattr(db, "get_settings", settings_with(10))
    timeouts = Delta("querycraft_db_statement_timeouts_total")
    errors = Delta("querycraft_db_query_errors_total", error_class="QueryCanceledError")
    with pytest.raises(asyncpg.QueryCanceledError):
        await db.run_select("SELECT 1")
    assert timeouts.change == 1 and errors.change == 1


# --- validator rules ---

@pytest.mark.parametrize("sql,rule", [
    ("", "syntax"), ("SELEC FROM WHERE", "syntax"), ("SELECT 1; SELECT 2", "multi_statement"),
    ("DROP TABLE products", "not_select"), ("SELECT * FROM pg_user", "forbidden_table"),
    ("SELECT * FROM generate_series(1, 3)", "forbidden_table"),
    ("SELECT * FROM products FOR UPDATE", "forbidden_node"), ("SELECT pg_sleep(1)", "forbidden_function"),
])
def test_validator_reports_a_fixed_rule(sql, rule):
    assert check_sql(sql)[0] == rule
    assert rule in RULES


def test_validator_accepts_good_sql_without_a_rule():
    assert check_sql("SELECT name FROM products") is None


# --- models ---

def chat_transport(fail: bool):
    def handler(request: httpx.Request) -> httpx.Response:
        if fail:
            raise httpx.ConnectError("down")
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        choice = {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "YES"}}
        return httpx.Response(200, json={"id": "1", "object": "chat.completion", "created": 0,
                                         "model": "m", "choices": [choice], "usage": usage})
    return httpx.MockTransport(handler)


async def test_model_call_metrics():
    model = RemoteModel(cfg(model="metrics-model"))
    model._http = httpx.AsyncClient(transport=chat_transport(fail=False))
    calls = Delta("querycraft_model_calls_total", model="metrics-model", kind="chat", status="ok")
    prompt = Delta("querycraft_model_prompt_chars_sum", model="metrics-model")
    latency = Delta("querycraft_model_call_duration_seconds_count", model="metrics-model", kind="chat")
    await model.chat("hello there")
    assert calls.change == 1 and prompt.change == 11 and latency.change == 1
    assert val("querycraft_model_up", model="metrics-model") == 1


async def test_failed_model_call_marks_model_down():
    model = RemoteModel(cfg(model="down-model"))
    model._http = httpx.AsyncClient(transport=chat_transport(fail=True))
    errors = Delta("querycraft_model_calls_total", model="down-model", kind="chat", status="error")
    with pytest.raises(ModelServerError):
        await model.chat("hi")
    assert errors.change == 1 and val("querycraft_model_up", model="down-model") == 0


# --- probes ---

def models_transport(ids: list[str] | None):
    def handler(request: httpx.Request) -> httpx.Response:
        if ids is None:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json={"object": "list", "data": [{"id": i} for i in ids]})
    return httpx.MockTransport(handler)


async def test_model_probe_up_missing_and_unreachable():
    for ids, expected in ((["probe-model:latest"], 1), (["other"], 0), (None, 0)):
        async with httpx.AsyncClient(transport=models_transport(ids)) as client:
            assert await probe_model(cfg(model="probe-model"), client) is bool(expected)
        assert val("querycraft_model_up", model="probe-model") == expected


async def test_local_model_probe_checks_the_file(tmp_path):
    present = tmp_path / "m.gguf"
    present.write_bytes(b"x")
    local = ModelSettings("local", "", "", "repo", present, 0, 2048, 128)
    async with httpx.AsyncClient() as client:
        assert await probe_model(local, client) is True
        assert await probe_model(ModelSettings("local", "", "", "repo", tmp_path / "gone.gguf", 0, 2048, 128),
                                 client) is False


# --- exposure ---

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_metrics_server_serves_prometheus_text(monkeypatch):
    port = free_port()
    monkeypatch.setenv("METRICS_PORT", str(port))
    monkeypatch.setenv("METRICS_ADDR", "127.0.0.1")
    metrics.set_build_info("remote", "m1", "m2")
    server = metrics.start_metrics_server()
    try:
        text = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=5).text
    finally:
        server.shutdown()
    assert "querycraft_queries_total" in text and "querycraft_build_info" in text
    assert 'sql_model="m1"' in text and 'chat_model="m2"' in text


def test_metrics_server_can_be_disabled(monkeypatch):
    monkeypatch.setenv("METRICS_PORT", "0")
    assert metrics.start_metrics_server() is None


def test_no_metric_label_ever_holds_user_text():
    """Spot check: label values are short, fixed-looking tokens (no spaces, no SQL, no questions)."""
    for family in REGISTRY.collect():
        if not family.name.startswith("querycraft"):
            continue
        for sample in family.samples:
            for key, value in sample.labels.items():
                assert len(value) < 80 and " " not in value and "SELECT" not in value, (sample.name, key, value)
