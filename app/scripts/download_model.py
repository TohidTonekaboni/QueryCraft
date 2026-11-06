"""Download the GGUF weights (SQLCoder + Gemma): python -m app.scripts.download_model [sql|chat ...]"""

import sys

from huggingface_hub import hf_hub_download

from app.config import get_settings


def main(which: list[str]) -> None:
    settings = get_settings()
    for name in which or ["sql", "chat"]:
        cfg = getattr(settings, name)
        cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(repo_id=cfg.repo, filename=cfg.model_path.name, local_dir=cfg.model_path.parent)
        print(f"{name} model ready: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
