"""Verify configured runtime dependencies without starting a long-running worker."""

from __future__ import annotations

import argparse
import asyncio

from self_evolution_agent.worker import Worker


async def verify(timeout_seconds: float) -> None:
    worker = Worker()
    try:
        await asyncio.wait_for(worker.initialize(), timeout=timeout_seconds)
        print("Worker initialization passed: SQLite, ChromaDB, checkpoint, and Bitable are ready.")
    finally:
        await worker.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()
    asyncio.run(verify(args.timeout))


if __name__ == "__main__":
    main()
