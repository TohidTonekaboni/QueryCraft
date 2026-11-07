"""Model backends for SQLCoder (completion) and Gemma (chat), built on LangChain wrappers.

local  - llama.cpp in-process via langchain-community (`LlamaCpp` / `ChatLlamaCpp`). GPU offload via
         n_gpu_layers (Metal/MPS on macOS, CUDA on NVIDIA builds, 0 = CPU). Loaded lazily.
remote - OpenAI-compatible server (Ollama, or llama.cpp `llama-server`) via langchain-openai (`OpenAI` / `ChatOpenAI`).

The workflow only sees the small `TextModel` protocol, so tests can inject fakes.
Blocking local inference never runs on the event loop.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

import httpx
from langchain_core.messages import HumanMessage

from app.config import ModelSettings, get_settings

logger = logging.getLogger("querycraft.llm")


@contextmanager
def _log_call(cfg: ModelSettings, kind: str, prompt_chars: int) -> Iterator[None]:
    """Log each model call: which model, how long, and whether it failed. Prompts and outputs are never logged."""
    start = time.perf_counter()
    fields = {"backend": cfg.backend, "kind": kind, "prompt_chars": prompt_chars,
              "model": cfg.remote_model if cfg.backend == "remote" else cfg.model_path.name}
    try:
        yield
    except Exception as exc:
        logger.warning("model_call_failed", extra={
            **fields, "error_type": type(exc).__name__, "duration_ms": round((time.perf_counter() - start) * 1000, 1)})
        raise
    logger.info("model_call", extra={**fields, "duration_ms": round((time.perf_counter() - start) * 1000, 1)})


class TextModel(Protocol):
    async def complete(self, prompt: str, *, stop: list[str] | None = None) -> str: ...
    async def chat(self, user_message: str) -> str: ...


def _check_model_file(cfg: ModelSettings) -> None:
    if not cfg.model_path.exists():
        raise FileNotFoundError(f"{cfg.model_path} not found; run: python -m app.scripts.download_model (local mode)")


class LocalLlamaModel:
    """In-process llama.cpp through LangChain. The LangChain object for each call style is built
    on first use, so only the one a role needs (completion for SQLCoder, chat for Gemma) is loaded."""

    def __init__(self, cfg: ModelSettings):
        self._cfg = cfg
        self._llm = None
        self._chat = None
        self._lock = threading.Lock()  # llama.cpp contexts are not thread-safe

    def _kwargs(self) -> dict:
        _check_model_file(self._cfg)
        return dict(model_path=str(self._cfg.model_path), n_gpu_layers=self._cfg.n_gpu_layers,
                    n_ctx=self._cfg.n_ctx, max_tokens=self._cfg.max_tokens, temperature=0.0, verbose=False)

    def _complete(self, prompt: str, stop: list[str] | None) -> str:
        with self._lock:
            if self._llm is None:  # lazy: first request pays the load cost
                try:
                    from langchain_community.llms import LlamaCpp
                except ImportError as exc:
                    raise RuntimeError("backend=local needs langchain-community and llama-cpp-python "
                                       "(pip install -r app/requirements-local.txt)") from exc
                self._llm = LlamaCpp(**self._kwargs())
            return self._llm.invoke(prompt, stop=stop)

    def _chat_once(self, user_message: str) -> str:
        with self._lock:
            if self._chat is None:
                try:
                    from langchain_community.chat_models import ChatLlamaCpp
                except ImportError as exc:
                    raise RuntimeError("backend=local needs langchain-community and llama-cpp-python "
                                       "(pip install -r app/requirements-local.txt)") from exc
                self._chat = ChatLlamaCpp(**self._kwargs())
            return self._chat.invoke([HumanMessage(user_message)]).content

    async def complete(self, prompt: str, *, stop: list[str] | None = None) -> str:
        with _log_call(self._cfg, "complete", len(prompt)):
            return await asyncio.to_thread(self._complete, prompt, stop)

    async def chat(self, user_message: str) -> str:
        with _log_call(self._cfg, "chat", len(user_message)):
            return await asyncio.to_thread(self._chat_once, user_message)


class _StringPrompt:
    """langchain-openai always sends `prompt` as a list; Ollama's /v1/completions only accepts a
    string. OpenAI and llama-server accept both, so send a single prompt as a plain string."""

    def __init__(self, completions):
        self._completions = completions

    async def create(self, *, prompt, **kwargs):
        if isinstance(prompt, list) and len(prompt) == 1:
            prompt = prompt[0]
        return await self._completions.create(prompt=prompt, **kwargs)


class RemoteModel:
    """OpenAI-compatible server (Ollama or llama-server) through langchain-openai."""

    def __init__(self, cfg: ModelSettings):
        self._cfg = cfg
        self._llm = None
        self._chat = None
        # trust_env=False: inference servers are local/host-side; never route them through a system proxy.
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0), trust_env=False)

    def _kwargs(self) -> dict:
        return dict(base_url=f"{self._cfg.remote_url}/v1", api_key="not-needed", model=self._cfg.remote_model,
                    temperature=0.0, max_retries=0, http_async_client=self._http)

    def _completion_llm(self):
        if self._llm is None:
            from langchain_openai import OpenAI
            llm = OpenAI(max_tokens=self._cfg.max_tokens, **self._kwargs())
            llm.async_client = _StringPrompt(llm.async_client)
            self._llm = llm
        return self._llm

    def _chat_llm(self):
        if self._chat is None:
            from langchain_openai import ChatOpenAI
            # ChatOpenAI sends `max_completion_tokens`; also send the classic `max_tokens` that every
            # llama-server version understands.
            self._chat = ChatOpenAI(max_tokens=self._cfg.max_tokens,
                                    extra_body={"max_tokens": self._cfg.max_tokens}, **self._kwargs())
        return self._chat

    async def complete(self, prompt: str, *, stop: list[str] | None = None) -> str:
        with _log_call(self._cfg, "complete", len(prompt)):
            try:
                return await self._completion_llm().ainvoke(prompt, stop=stop)
            except Exception as exc:
                raise RuntimeError(f"Model server at {self._cfg.remote_url} failed: {exc!r}") from exc

    async def chat(self, user_message: str) -> str:
        with _log_call(self._cfg, "chat", len(user_message)):
            try:
                return (await self._chat_llm().ainvoke([HumanMessage(user_message)])).content
            except Exception as exc:
                raise RuntimeError(f"Model server at {self._cfg.remote_url} failed: {exc!r}") from exc


_models: dict[str, TextModel] = {}


def _get(name: str) -> TextModel:
    if name not in _models:
        cfg = getattr(get_settings(), name)
        _models[name] = LocalLlamaModel(cfg) if cfg.backend == "local" else RemoteModel(cfg)
    return _models[name]


def get_sql_model() -> TextModel:
    return _get("sql")


def get_chat_model() -> TextModel:
    return _get("chat")


def set_models(sql: TextModel | None = None, chat: TextModel | None = None) -> None:
    """Inject model implementations (tests)."""
    if sql is not None:
        _models["sql"] = sql
    if chat is not None:
        _models["chat"] = chat
