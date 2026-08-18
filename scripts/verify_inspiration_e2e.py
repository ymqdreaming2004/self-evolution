"""Run one explicitly labelled end-to-end inspiration verification message."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from self_evolution_agent.schemas import IncomingMessage
from self_evolution_agent.worker import Worker


async def verify() -> str:
    worker = Worker()
    try:
        await worker.initialize()
        assert worker.workflow is not None
        run_id = uuid4().hex[:8]
        message = IncomingMessage(
            event_id=f"e2e-inspiration-{run_id}",
            message_id=f"e2e-inspiration-{run_id}",
            chat_id="e2e-verification",
            open_id=worker.settings.feishu_allowed_open_id,
            text=f"灵感：E2E 验收记录 {datetime.now(UTC):%Y-%m-%d %H:%M:%S UTC}",
        )
        result = await worker.workflow.invoke_message(message, thread_id=message.message_id)
        return result.get("reply", "")
    finally:
        await worker.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="write one labelled verification record and send the Bot reply",
    )
    args = parser.parse_args()
    if not args.execute:
        parser.error("pass --execute to run the real external verification")
    print(asyncio.run(verify()))


if __name__ == "__main__":
    main()
