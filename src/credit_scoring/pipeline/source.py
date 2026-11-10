import asyncio
import csv
import logging
import random
import signal
import time
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

from faststream.rabbit import RabbitBroker

from credit_scoring.broker import QueueOverloadedError, declare_topology, publish_task
from credit_scoring.config import Settings, settings
from credit_scoring.logging_config import configure_logging
from credit_scoring.schemas import FEATURE_COLUMNS, CreditFeatures, TaskMessage

logger = logging.getLogger("credit_scoring.source")


def read_tasks(path: str, batch_id: str, limit: int) -> Iterator[TaskMessage]:
    """Читает файл лениво: в памяти источника всегда не больше одной задачи."""
    produced = 0
    while produced < limit:
        with open(path, newline="") as file:
            for row in csv.DictReader(file):
                if produced >= limit:
                    return
                yield TaskMessage(
                    task_id=uuid5(NAMESPACE_URL, f"{batch_id}:{produced}"),
                    features=CreditFeatures(**{k: row[k] for k in FEATURE_COLUMNS}),
                    created_at=datetime.now(UTC),
                )
                produced += 1


async def publish_with_backoff(
    publish: Callable[[], Awaitable[None]],
    initial_delay: float,
    max_delay: float,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> int:
    """Повторяет публикацию с экспоненциальной задержкой, пока брокер не примет."""
    rejections = 0
    delay = initial_delay
    while True:
        try:
            await publish()
            return rejections
        except QueueOverloadedError:
            rejections += 1
            await sleep(delay * random.uniform(0.5, 1.0))
            delay = min(delay * 2, max_delay)


async def run_source(cfg: Settings) -> None:
    broker = RabbitBroker(cfg.rabbit_url, logger=logger)
    await broker.start()
    await declare_topology(broker, cfg)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    published = rejected = 0
    report_at = time.monotonic() + 5
    window_published = 0
    for task in read_tasks(cfg.source_file, cfg.source_batch_id, cfg.source_max_tasks):
        if stop.is_set():
            break
        rejected += await publish_with_backoff(
            lambda t=task: publish_task(broker, t, cfg),
            cfg.source_backoff_initial_seconds,
            cfg.source_backoff_max_seconds,
        )
        published += 1
        window_published += 1
        if time.monotonic() >= report_at:
            logger.info(
                "source progress",
                extra={
                    "published": published,
                    "rejected": rejected,
                    "rate_per_s": window_published / 5,
                },
            )
            window_published = 0
            report_at = time.monotonic() + 5
    logger.info("source finished", extra={"published": published, "rejected": rejected})
    await broker.stop()


if __name__ == "__main__":
    configure_logging(settings.log_level)
    asyncio.run(run_source(settings))
