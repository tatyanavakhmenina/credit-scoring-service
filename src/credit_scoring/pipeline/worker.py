import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from faststream import AckPolicy
from faststream.asgi import AsgiFastStream, make_ping_asgi
from faststream.rabbit import Channel, RabbitBroker
from faststream.rabbit.annotations import RabbitMessage

from credit_scoring.broker import declare_topology, tasks_queue
from credit_scoring.config import settings
from credit_scoring.database import create_pool, mark_failed, save_result
from credit_scoring.logging_config import configure_logging
from credit_scoring.ml.model import ScoringModel, load_model
from credit_scoring.schemas import TaskMessage

logger = logging.getLogger("credit_scoring.worker")


@dataclass
class WorkerState:
    model: ScoringModel | None = None
    pool: object | None = None


state = WorkerState()
executor = ThreadPoolExecutor(
    max_workers=settings.worker_concurrency, thread_name_prefix="inference"
)
broker = RabbitBroker(
    settings.rabbit_url,
    graceful_timeout=settings.graceful_timeout_seconds,
    logger=logger,
)


def score(features: dict) -> float:
    """Выполняется в пуле потоков, поэтому не блокирует event loop."""
    time.sleep(settings.inference_delay_seconds)
    return state.model.predict_proba(features)


@broker.subscriber(
    tasks_queue(settings),
    channel=Channel(prefetch_count=settings.effective_prefetch or None),
    # ack только после записи результата, при ошибке — requeue до x-delivery-limit.
    ack_policy=AckPolicy.NACK_ON_ERROR,
)
async def handle_task(task: TaskMessage, message: RabbitMessage) -> None:
    loop = asyncio.get_running_loop()
    try:
        probability = await loop.run_in_executor(
            executor, score, task.features.model_dump()
        )
        await save_result(
            state.pool,
            task.task_id,
            probability,
            state.model.info.version,
            task.created_at,
        )
    except Exception as e:
        redeliveries = int(message.headers.get("x-delivery-count", 0))
        logger.warning(
            "task failed",
            extra={"task_id": str(task.task_id), "redeliveries": redeliveries},
        )
        if redeliveries >= settings.delivery_limit:
            # Последняя попытка: сообщение уйдёт в DLQ, в БД не должно остаться PENDING.
            await mark_failed(state.pool, task.task_id, str(e), task.created_at)
        raise


async def startup() -> None:
    configure_logging(settings.log_level)
    state.pool = await create_pool(
        settings.database_url, max_size=settings.worker_concurrency
    )
    state.model = await asyncio.to_thread(load_model, settings)


async def declare() -> None:
    await declare_topology(broker, settings)


async def shutdown() -> None:
    executor.shutdown(wait=True)
    if state.pool:
        await state.pool.close()


app = AsgiFastStream(
    broker,
    asgi_routes=[("/healthz", make_ping_asgi(broker, timeout=1.0))],
    on_startup=[startup],
    after_startup=[declare],
    after_shutdown=[shutdown],
)
