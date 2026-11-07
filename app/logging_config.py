"""Logging setup: structured JSON (default) or readable text, request IDs, privacy-safe helpers.

Environment variables:
    LOG_LEVEL      DEBUG | INFO | WARNING | ERROR            (default INFO)
    LOG_FORMAT     json | text                               (default json; use text for local development)
    LOG_SENSITIVE  true to include user questions and generated SQL in logs (default false)

Logs go to stderr (the container runtime collects them). Every record carries the `request_id` of the
HTTP request being served, so one request can be followed across the API, workflow and model calls.
Passwords and tokens are never logged. By default neither are questions or SQL text, which can contain
personal data typed by users: only their lengths are logged.
"""

import contextvars
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_HANDLER_MARK = "_querycraft_handler"
# `color_message` is added by uvicorn and holds ANSI colour codes, which do not belong in log files.
_STANDARD_ATTRS = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime", "request_id", "color_message"}
_ERROR_KIND = re.compile(r"^([A-Za-z]+(?:Error|Exception)):")


def sensitive_logging_enabled() -> bool:
    return os.getenv("LOG_SENSITIVE", "false").strip().lower() in ("1", "true", "yes", "on")


def text_fields(name: str, value: str | None, limit: int = 300) -> dict:
    """Fields describing user-controlled text: the text itself only when LOG_SENSITIVE is on."""
    value = value or ""
    if sensitive_logging_enabled():
        return {name: value[:limit], f"{name}_chars": len(value)}
    return {f"{name}_chars": len(value)}


def error_kind(error: str | None) -> str | None:
    """A non-sensitive label for an error message: the exception class, else 'validation'.

    Database error messages can quote fragments of the query, so the text is logged only with LOG_SENSITIVE."""
    if not error:
        return None
    match = _ERROR_KIND.match(error)
    return match.group(1) if match else "validation"


def _extras(record: logging.LogRecord) -> dict:
    return {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS and not k.startswith("_")}


class JsonFormatter(logging.Formatter):
    """One JSON object per line; structured fields passed with `extra={...}` become top-level keys."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        payload.update(_extras(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable single line: time level [request_id] logger: message key=value ..."""

    def __init__(self):
        super().__init__("%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)  # also renders exc_info
        extras = " ".join(f"{k}={v}" for k, v in _extras(record).items())
        if not extras:
            return line
        head, sep, tail = line.partition("\n")
        return f"{head} {extras}{sep}{tail}"


def _install_record_factory() -> None:
    """Stamp every record, from any library, with the current request id."""
    previous = logging.getLogRecordFactory()
    if getattr(previous, "_querycraft", False):
        return

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        record.request_id = request_id_var.get()
        return record

    factory._querycraft = True
    logging.setLogRecordFactory(factory)


def setup_logging() -> None:
    """Configure the root logger. Safe to call more than once."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    if level not in logging.getLevelNamesMapping():
        raise ValueError(f"LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL (got {level!r})")
    fmt = os.getenv("LOG_FORMAT", "json").lower()
    if fmt not in ("json", "text"):
        raise ValueError(f"LOG_FORMAT must be 'json' or 'text' (got {fmt!r})")

    _install_record_factory()
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _HANDLER_MARK, False)]:
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    setattr(handler, _HANDLER_MARK, True)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root.addHandler(handler)
    root.setLevel(level)

    # uvicorn: route its error log through our handler; the access log is replaced by the
    # request log written by the API middleware (it adds the request id and duration).
    for name in ("uvicorn", "uvicorn.error"):
        uv = logging.getLogger(name)
        uv.handlers.clear()
        uv.propagate = True
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False

    # Client libraries log every HTTP call at INFO; keep them quiet unless something is wrong.
    for noisy in ("httpx", "httpcore", "openai", "asyncio"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, logging.getLogger().level))
