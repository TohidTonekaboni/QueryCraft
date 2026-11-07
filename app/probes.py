"""Background health probes that feed querycraft_model_up and querycraft_db_up.

Probing on a timer (not only on user requests) means an outage shows up in the metrics even when
nobody is asking questions."""

import asyncio
import logging
import os

import httpx

from app import metrics
from app.config import ModelSettings, get_settings
from app.db import ping

logger = logging.getLogger("querycraft.probes")


async def probe_model(cfg: ModelSettings, client: httpx.AsyncClient) -> bool:
    """Remote: the server answers /v1/models and lists our model. Local: the GGUF file exists."""
    if cfg.backend == "local":
        up, label = cfg.model_path.exists(), cfg.model_path.name
    else:
        label = cfg.remote_model
        try:
            resp = await client.get(f"{cfg.remote_url}/v1/models")
            resp.raise_for_status()
            ids = {m["id"] for m in resp.json().get("data", [])}
            up = label in ids or f"{label}:latest" in ids
        except Exception as exc:
            logger.warning("model_probe_failed", extra={"model": label, "error_type": type(exc).__name__})
            metrics.MODEL_UP.labels(label).set(0)
            return False
        if not up:
            logger.warning("model_probe_failed", extra={"model": label, "error_type": "ModelNotListed"})
    metrics.MODEL_UP.labels(label).set(1 if up else 0)
    return up


async def probe_once() -> None:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=5, trust_env=False) as client:  # never via a system proxy
        await asyncio.gather(probe_model(settings.sql, client), probe_model(settings.chat, client), ping())


async def probe_loop() -> None:
    interval = float(os.getenv("METRICS_PROBE_INTERVAL", "30"))
    while True:
        try:
            await probe_once()
        except Exception:  # a probe must never take the service down
            logger.exception("probe_failed")
        await asyncio.sleep(interval)
