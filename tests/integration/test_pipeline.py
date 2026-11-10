import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import aio_pika
import pytest
from faststream.rabbit import RabbitBroker
from httpx import ASGITransport, AsyncClient
from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.rabbitmq import RabbitMqContainer

from credit_scoring.broker import (
    QueueOverloadedError,
    declare_topology,
    dlq,
    publish_task,
)
from credit_scoring.config import Settings
from credit_scoring.database import create_pool, save_result
from credit_scoring.main import app
from credit_scoring.pipeline.source import publish_with_backoff
from credit_scoring.schemas import CreditFeatures, TaskMessage
from tests.conftest import FEATURES

pytestmark = pytest.mark.integration
LIMIT = 5


@pytest.fixture(scope="module")
def rabbit_url():
    with RabbitMqContainer("rabbitmq:4.1-management-alpine") as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5672)
        yield f"amqp://guest:guest@{host}:{port}/"


@pytest.fixture(scope="module")
def postgres_url():
    with PostgresContainer("postgres:17-alpine", driver=None) as container:
        yield container.get_connection_url()


@pytest.fixture
async def rabbit(rabbit_url):
    cfg = Settings(
        rabbit_url=rabbit_url, queue_name=f"t.{uuid4().hex}", queue_max_length=LIMIT
    )
    broker = RabbitBroker(rabbit_url)
    await broker.start()
    await declare_topology(broker, cfg)
    yield SimpleNamespace(broker=broker, cfg=cfg)
    await broker.stop()


def _task() -> TaskMessage:
    return TaskMessage(
        task_id=uuid4(),
        features=CreditFeatures(**FEATURES),
        created_at=datetime.now(UTC),
    )


async def _fill(rabbit) -> None:
    """Quorum-очередь проверяет лимит до записи и пропускает ещё одно сообщение."""
    for _ in range(LIMIT + 1):
        await publish_task(rabbit.broker, _task(), rabbit.cfg)
    await asyncio.sleep(0.2)


async def test_full_queue_rejects_publish_and_keeps_limit(rabbit, rabbit_url):
    await _fill(rabbit)

    with pytest.raises(QueueOverloadedError):
        await publish_task(rabbit.broker, _task(), rabbit.cfg)

    connection = await aio_pika.connect(rabbit_url)
    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue(rabbit.cfg.queue_name, passive=True)
        assert queue.declaration_result.message_count == LIMIT + 1


async def test_source_waits_and_publishes_when_space_appears(rabbit, rabbit_url):
    await _fill(rabbit)
    connection = await aio_pika.connect(rabbit_url)

    async def free_slot():
        await asyncio.sleep(0.5)
        channel = await connection.channel()
        queue = await channel.declare_queue(rabbit.cfg.queue_name, passive=True)
        # Quorum-очередь снова принимает, когда длина опустится до 80% лимита.
        for _ in range(2):
            message = await queue.get(timeout=2)
            await message.ack()

    async with connection:
        consumer = asyncio.create_task(free_slot())
        task = _task()
        rejections = await publish_with_backoff(
            lambda: publish_task(rabbit.broker, task, rabbit.cfg), 0.05, 0.2
        )
        await consumer

    assert rejections >= 1


async def test_poison_message_goes_to_dlq_after_delivery_limit(rabbit, rabbit_url):
    await publish_task(rabbit.broker, _task(), rabbit.cfg)
    connection = await aio_pika.connect(rabbit_url)

    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue(rabbit.cfg.queue_name, passive=True)
        deliveries = 0
        for _ in range(rabbit.cfg.delivery_limit + 1):
            message = await queue.get(timeout=5)
            deliveries += 1
            await message.nack(requeue=True)
            await asyncio.sleep(0.2)

        dead = await channel.declare_queue(dlq(rabbit.cfg).name, passive=True)
        dead_message = await dead.get(timeout=5)
        await dead_message.ack()
        assert await queue.get(fail=False) is None

    assert deliveries == rabbit.cfg.delivery_limit + 1


async def test_save_result_is_idempotent_on_concurrent_redelivery(postgres_url):
    pool = await create_pool(postgres_url)
    task = _task()

    await asyncio.gather(
        *[
            save_result(pool, task.task_id, 0.5, str(i), task.created_at)
            for i in range(10)
        ]
    )

    rows = await pool.fetch("SELECT * FROM tasks WHERE task_id = $1", task.task_id)
    await pool.close()
    assert len(rows) == 1
    assert rows[0]["status"] == "DONE"


async def test_api_503_leaves_no_pending_task(rabbit, postgres_url, monkeypatch):
    from credit_scoring.api.v1 import tasks

    monkeypatch.setattr(tasks, "settings", rabbit.cfg)
    await _fill(rabbit)
    pool = await create_pool(postgres_url)
    await pool.execute("TRUNCATE tasks")
    app.state.pool = pool
    app.state.broker = rabbit.broker

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/tasks", json=FEATURES)

    count = await pool.fetchval("SELECT count(*) FROM tasks")
    await pool.close()
    app.state.pool = app.state.broker = None
    assert response.status_code == 503
    assert response.headers["Retry-After"] == str(rabbit.cfg.retry_after_seconds)
    assert count == 0
