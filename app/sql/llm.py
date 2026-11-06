"""Model backends for SQLCoder (completion) and Gemma (chat).

local  - llama-cpp-python in-process. GPU offload via n_gpu_layers
         (Metal/MPS on macOS, CUDA on NVIDIA builds, 0 = CPU). Loaded lazily.
remote - OpenAI-compatible llama.cpp `llama-server`.

Blocking inference never runs on the event loop.
"""

import asyncio
import threading
from typing import Protocol

import httpx

from app.config import ModelSettings, get_settings


class TextModel(Protocol):
    async def complete(self, prompt: str, *, stop: list[str] | None = None) -> str: ...
    async def chat(self, user_message: str) -> str: ...


class LocalLlamaModel:
    def __init__(self, cfg: ModelSettings):
        self._cfg = cfg
        self._llm = None
        self._lock = threading.Lock()  # llama.cpp contexts are not thread-safe

    def _load(self):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("backend=local needs llama-cpp-python (pip install -r app/requirements-local.txt)") from exc
        if not self._cfg.model_path.exists():
            raise FileNotFoundError(f"{self._cfg.model_path} not found; run: python -m app.scripts.download_model")
        return Llama(model_path=str(self._cfg.model_path), n_gpu_layers=self._cfg.n_gpu_layers,
                     n_ctx=self._cfg.n_ctx, verbose=False)

    def _run(self, fn_name: str, *args, **kwargs):
        with self._lock:
            if self._llm is None:  # lazy: first request pays the load cost
                self._llm = self._load()
            return getattr(self._llm, fn_name)(*args, **kwargs)

    async def complete(self, prompt: str, *, stop: list[str] | None = None) -> str:
        out = await asyncio.to_thread(
            self._run, "__call__", prompt, max_tokens=self._cfg.max_tokens, temperature=0.0, stop=stop)
        return out["choices"][0]["text"]

    async def chat(self, user_message: str) -> str:
        out = await asyncio.to_thread(
            self._run, "create_chat_completion",
            messages=[{"role": "user", "content": user_message}],
            max_tokens=self._cfg.max_tokens, temperature=0.0)
        return out["choices"][0]["message"]["content"]


class RemoteModel:
    def __init__(self, cfg: ModelSettings):
        self._cfg = cfg
        # trust_env=False: inference servers are local/host-side; never route them through a system proxy.
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0), trust_env=False)

    async def _post(self, path: str, body: dict) -> dict:
        try:
            resp = await self._client.post(f"{self._cfg.remote_url}{path}", json=body)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Model server at {self._cfg.remote_url} failed: {exc!r}") from exc
        return resp.json()

    async def complete(self, prompt: str, *, stop: list[str] | None = None) -> str:
        data = await self._post("/v1/completions", {
            "prompt": prompt, "max_tokens": self._cfg.max_tokens, "temperature": 0.0, "stop": stop})
        return data["choices"][0]["text"]

    async def chat(self, user_message: str) -> str:
        data = await self._post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": user_message}],
            "max_tokens": self._cfg.max_tokens, "temperature": 0.0})
        return data["choices"][0]["message"]["content"]


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
