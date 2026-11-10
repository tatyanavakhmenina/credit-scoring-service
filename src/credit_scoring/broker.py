import aiormq
from faststream.rabbit import (
    ExchangeType,
    QueueType,
    RabbitBroker,
    RabbitExchange,
    RabbitQueue,
)

from credit_scoring.config import Settings
from credit_scoring.schemas import TaskMessage


class QueueOverloadedError(Exception):
    """Брокер не подтвердил публикацию: очередь заполнена или не отвечает."""


def dlx(settings: Settings) -> RabbitExchange:
    return RabbitExchange(
        f"{settings.queue_name}.dlx", type=ExchangeType.DIRECT, durable=True
    )


def dlq(settings: Settings) -> RabbitQueue:
    return RabbitQueue(
        f"{settings.queue_name}.dlq",
        durable=True,
        routing_key=settings.queue_name,
    )


def tasks_queue(settings: Settings) -> RabbitQueue:
    """Quorum-очередь: x-delivery-limit считает повторные доставки в брокере."""
    arguments = {
        "x-delivery-limit": settings.delivery_limit,
        "x-dead-letter-exchange": dlx(settings).name,
        "x-dead-letter-routing-key": settings.queue_name,
    }
    if settings.queue_max_length > 0:
        arguments |= {
            "x-max-length": settings.queue_max_length,
            "x-overflow": "reject-publish",
            "x-dead-letter-strategy": "at-least-once",
        }
    return RabbitQueue(
        settings.queue_name,
        durable=True,
        queue_type=QueueType.QUORUM,
        arguments=arguments,
    )


async def declare_topology(broker: RabbitBroker, settings: Settings) -> None:
    exchange = await broker.declare_exchange(dlx(settings))
    dead_letters = await broker.declare_queue(dlq(settings))
    await dead_letters.bind(exchange, routing_key=settings.queue_name)
    await broker.declare_queue(tasks_queue(settings))


async def publish_task(
    broker: RabbitBroker, message: TaskMessage, settings: Settings
) -> None:
    """Публикует persistent-сообщение и ждёт publisher confirm от брокера."""
    try:
        await broker.publish(
            message.model_dump(mode="json"),
            queue=settings.queue_name,
            message_id=str(message.task_id),
            persist=True,
            timeout=settings.publish_timeout_seconds,
        )
    except (aiormq.exceptions.DeliveryError, TimeoutError) as e:
        raise QueueOverloadedError(str(e)) from e
