"""Prepare the models in Ollama: pull the base models and create the ones the app calls.

    python -m app.scripts.setup_ollama [sql|chat ...]

Ollama's OpenAI-compatible API cannot set GPU layers per request, so the settings that matter are baked
into a derived model (`ollama create`): SQL_N_GPU_LAYERS / GEMMA_N_GPU_LAYERS become `num_gpu`
(0 = CPU only, -1 = offload everything) and *_N_CTX becomes `num_ctx`. The app then calls the derived
models named by SQL_REMOTE_MODEL / GEMMA_REMOTE_MODEL.

Re-run this script after changing any of those variables.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# name -> (env prefix, default base model, default derived model, default GPU layers, default context)
MODELS = {
    "sql": ("SQL", "pxlksr/defog_sqlcoder-7b-2:Q4_K_M", "querycraft-sqlcoder", "-1", "4096"),
    "chat": ("GEMMA", "gemma3:4b", "querycraft-gemma", "0", "2048"),
}


def modelfile(base: str, gpu_layers: int, ctx: int) -> str:
    lines = [f"FROM {base}", f"PARAMETER num_ctx {ctx}"]
    if gpu_layers >= 0:  # -1 = let Ollama offload everything it can
        lines.append(f"PARAMETER num_gpu {gpu_layers}")
    return "\n".join(lines) + "\n"


def ollama(*args: str) -> None:
    try:
        subprocess.run(["ollama", *args], check=True)
    except FileNotFoundError:
        sys.exit("ollama not found: install it from https://ollama.com and make sure it is on PATH")


def setup(name: str) -> str:
    prefix, default_base, default_name, default_layers, default_ctx = MODELS[name]
    base = os.getenv(f"{prefix}_OLLAMA_BASE", default_base)
    derived = os.getenv(f"{prefix}_REMOTE_MODEL", default_name)
    gpu_layers = int(os.getenv(f"{prefix}_N_GPU_LAYERS", default_layers))
    ctx = int(os.getenv(f"{prefix}_N_CTX", default_ctx))

    ollama("pull", base)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Modelfile"
        path.write_text(modelfile(base, gpu_layers, ctx))
        ollama("create", derived, "-f", str(path))
    return f"{derived} (from {base}, num_gpu={gpu_layers if gpu_layers >= 0 else 'auto'}, num_ctx={ctx})"


def main(which: list[str]) -> None:
    for name in which or list(MODELS):
        if name not in MODELS:
            raise SystemExit(f"unknown model '{name}'; choose from: {', '.join(MODELS)}")
        print(f"{name} model ready: {setup(name)}")


if __name__ == "__main__":
    main(sys.argv[1:])
