"""Run one question through the workflow: python -m app.cli "What is the most expensive product?" """

import asyncio
import json
import sys

from app.db import close_pool
from app.logging_config import setup_logging
from app.workflow import run_workflow


async def main(question: str) -> None:
    try:
        print(json.dumps(await run_workflow(question), indent=2, ensure_ascii=False))
    finally:
        await close_pool()


if __name__ == "__main__":
    setup_logging()  # logs go to stderr, so stdout stays clean JSON
    if len(sys.argv) < 2:
        sys.exit('usage: python -m app.cli "your question"')
    asyncio.run(main(" ".join(sys.argv[1:])))
