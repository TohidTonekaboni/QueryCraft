"""Logging: formats, request ids across the whole request, privacy defaults, failure logging."""

import json
import logging
import re
from pathlib import Path

import httpx
import pytest

from app.config import ModelSettings
from app.logging_config import JsonFormatter, TextFormatter, error_kind, request_id_var, setup_logging, text_fields
from app.sql.llm import RemoteModel

EN_QUESTION = "What is the most expensive product sold?"


def make_record(msg="hello", level=logging.INFO, exc_info=None, **extra):
    record = logging.LogRecord("querycraft.test", level, __file__, 1, msg, (), exc_info)
    record.request_id = "req-1"
    record.__dict__.update(extra)
    return record


def app_records(caplog):
    return [r for r in caplog.records if r.name.startswith("querycraft")]


# --- formatters and helpers (no database needed) ---

def test_json_formatter_emits_one_object_with_extras():
    line = JsonFormatter().format(make_record("node_completed", node="relevance", duration_ms=12.5))
    data = json.loads(line)
    assert "\n" not in line
    assert data["message"] == "node_completed" and data["level"] == "INFO" and data["logger"] == "querycraft.test"
    assert data["request_id"] == "req-1" and data["node"] == "relevance" and data["duration_ms"] == 12.5
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+00:00$", data["timestamp"])


def test_json_formatter_drops_uvicorn_colour_codes():
    data = json.loads(JsonFormatter().format(make_record("Started", color_message="\x1b[32mStarted\x1b[0m")))
    assert "color_message" not in data


def test_json_formatter_includes_traceback():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        data = json.loads(JsonFormatter().format(make_record("failed", logging.ERROR, exc_info=sys.exc_info())))
    assert "ValueError: boom" in data["exception"]


def test_json_formatter_keeps_persian_readable_and_survives_odd_values():
    data = json.loads(JsonFormatter().format(make_record("x", text="گران‌ترین", path=Path("/a"))))
    assert data["text"] == "گران‌ترین" and data["path"] == "/a"


def test_text_formatter_is_one_readable_line():
    line = TextFormatter().format(make_record("request", status=200))
    assert "[req-1]" in line and "querycraft.test: request" in line and line.endswith("status=200")


def test_error_kind_never_returns_message_text():
    assert error_kind('UndefinedColumnError: column "ssn" does not exist') == "UndefinedColumnError"
    assert error_kind("Table pg_user is not allowed; use only: products") == "validation"
    assert error_kind(None) is None


def test_text_fields_respect_log_sensitive(monkeypatch):
    monkeypatch.delenv("LOG_SENSITIVE", raising=False)
    assert text_fields("question", "secret") == {"question_chars": 6}
    monkeypatch.setenv("LOG_SENSITIVE", "true")
    assert text_fields("question", "secret") == {"question": "secret", "question_chars": 6}


def test_setup_logging_rejects_bad_settings(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    with pytest.raises(ValueError, match="LOG_LEVEL"):
        setup_logging()
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("LOG_FORMAT", "xml")
    with pytest.raises(ValueError, match="LOG_FORMAT"):
        setup_logging()


def test_setup_logging_is_idempotent():
    setup_logging()
    setup_logging()
    assert len([h for h in logging.getLogger().handlers if getattr(h, "_querycraft_handler", False)]) == 1


# --- through the API ---

async def test_request_id_is_echoed_generated_and_sanitised(client, fake_sql):
    ok = await client.post("/query", json={"question": EN_QUESTION}, headers={"X-Request-ID": "abc-123"})
    assert ok.headers["x-request-id"] == "abc-123"

    generated = await client.post("/query", json={"question": EN_QUESTION})
    assert re.fullmatch(r"[0-9a-f]{32}", generated.headers["x-request-id"])

    hostile = await client.post("/query", json={"question": EN_QUESTION},
                                headers={"X-Request-ID": 'x"}\n{"level":"CRITICAL"'})
    assert re.fullmatch(r"[0-9a-f]{32}", hostile.headers["x-request-id"])  # replaced, not logged


async def test_one_request_id_follows_the_whole_request(client, caplog):
    caplog.set_level(logging.INFO, logger="querycraft")
    resp = await client.post("/query", json={"question": EN_QUESTION}, headers={"X-Request-ID": "trace-me"})
    assert resp.status_code == 200

    records = [r for r in app_records(caplog) if r.request_id == "trace-me"]
    events = [r.getMessage() for r in records]
    nodes = [r.node for r in records if r.getMessage() == "node_completed"]
    assert nodes == ["input", "language", "relevance", "generate", "validate", "execute", "respond"]
    assert "query_received" in events and "query_completed" in events and events[-1] == "request"

    done = next(r for r in records if r.getMessage() == "query_completed")
    assert done.success is True and done.attempts == 1 and done.language == "en" and done.row_count == 1
    request = next(r for r in records if r.getMessage() == "request")
    assert request.status == 200 and request.path == "/query" and request.duration_ms >= 0
    assert request_id_var.get() == "-"  # the context is reset after the request


async def test_correction_loop_is_visible_in_logs(client, fake_sql, caplog):
    caplog.set_level(logging.INFO, logger="querycraft")
    fake_sql.mode = "bad_then_good"
    await client.post("/query", json={"question": EN_QUESTION})
    execute = [r for r in app_records(caplog) if r.getMessage() == "node_completed" and r.node == "execute"]
    assert execute[0].error_kind == "UndefinedColumnError"
    assert [r.getMessage() for r in app_records(caplog)].count("node_completed") > 7
    done = next(r for r in app_records(caplog) if r.getMessage() == "query_completed")
    assert done.success is True and done.attempts == 2


async def test_irrelevant_question_is_logged_as_such(client, caplog):
    caplog.set_level(logging.INFO, logger="querycraft")
    await client.post("/query", json={"question": "What's the weather today?"})
    done = next(r for r in app_records(caplog) if r.getMessage() == "query_completed")
    assert done.success is False and done.failure == "irrelevant" and done.relevant is False


async def test_questions_and_sql_are_not_logged_by_default(client, caplog, monkeypatch):
    monkeypatch.delenv("LOG_SENSITIVE", raising=False)
    caplog.set_level(logging.DEBUG, logger="querycraft")
    await client.post("/query", json={"question": EN_QUESTION})
    dump = "\n".join(JsonFormatter().format(r) for r in caplog.records)
    assert "most expensive" not in dump and "SELECT" not in dump
    assert "question_chars" in dump and "sql_chars" in dump


async def test_questions_and_sql_are_logged_when_enabled(client, caplog, monkeypatch):
    monkeypatch.setenv("LOG_SENSITIVE", "true")
    caplog.set_level(logging.INFO, logger="querycraft")
    await client.post("/query", json={"question": EN_QUESTION})
    dump = "\n".join(JsonFormatter().format(r) for r in app_records(caplog))
    assert EN_QUESTION in dump and "SELECT" in dump


async def test_failures_are_logged_with_traceback_and_request_id(client, fake_chat, caplog):
    async def boom(_):
        raise RuntimeError("model server down")

    fake_chat.chat = boom
    caplog.set_level(logging.INFO, logger="querycraft")
    resp = await client.post("/query", json={"question": EN_QUESTION}, headers={"X-Request-ID": "fail-1"})
    assert resp.status_code == 503
    failed = [r for r in app_records(caplog) if r.getMessage() in ("node_failed", "workflow_failed")]
    assert {r.getMessage() for r in failed} == {"node_failed", "workflow_failed"}
    assert all(r.request_id == "fail-1" and r.exc_info for r in failed)


# --- model calls ---

async def test_model_calls_are_timed_but_prompts_are_never_logged(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        choice = {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "YES"}}
        return httpx.Response(200, json={"id": "1", "object": "chat.completion", "created": 0,
                                         "model": "m", "choices": [choice], "usage": usage})

    cfg = ModelSettings("remote", "http://llm.test", "my-model", "repo", Path("x"), 0, 2048, 128)
    model = RemoteModel(cfg)
    model._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    caplog.set_level(logging.INFO, logger="querycraft")
    await model.chat("a very private prompt")

    record = next(r for r in app_records(caplog) if r.getMessage() == "model_call")
    assert record.model == "my-model" and record.kind == "chat" and record.prompt_chars == 21
    assert "private" not in JsonFormatter().format(record)


async def test_model_failures_are_logged_as_warnings(caplog):
    def handler(request):
        raise httpx.ConnectError("down")

    model = RemoteModel(ModelSettings("remote", "http://llm.test", "my-model", "repo", Path("x"), 0, 2048, 128))
    model._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    caplog.set_level(logging.INFO, logger="querycraft")
    with pytest.raises(RuntimeError):
        await model.chat("hi")
    record = next(r for r in app_records(caplog) if r.getMessage() == "model_call_failed")
    assert record.levelno == logging.WARNING and record.model == "my-model"
