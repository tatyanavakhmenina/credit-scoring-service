import logging
from datetime import UTC, datetime
from uuid import UUID, uuid4

import aiormq
from fastapi import APIRouter, HTTPException, Request, Response, status

from credit_scoring.broker import QueueOverloadedError, publish_task
from credit_scoring.config import settings
from credit_scoring.database import get_task, insert_pending
from credit_scoring.schemas import (
    CreditFeatures,
    TaskAccepted,
    TaskMessage,
    TaskStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _overloaded(reason: str) -> HTTPException:
    logger.warning("task rejected", extra={"reason": reason})
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Service overloaded, retry later",
        headers={"Retry-After": str(settings.retry_after_seconds)},
    )


@router.post(
    "/tasks", status_code=status.HTTP_202_ACCEPTED, response_model=TaskAccepted
)
async def create_task(features: CreditFeatures, request: Request):
    """Сброс нагрузки: клиента нельзя замедлить, поэтому отказ возвращается сразу."""
    pool = request.app.state.pool
    broker = request.app.state.broker
    if pool is None or broker is None:
        raise _overloaded("dependencies unavailable")

    message = TaskMessage(
        task_id=uuid4(), features=features, created_at=datetime.now(UTC)
    )
    try:
        # Транзакция закрывается только после confirm: при отказе брокера PENDING
        # откатывается, а воркер не обгонит INSERT, потому что ждёт коммита по PK.
        async with pool.acquire(timeout=1.0) as conn, conn.transaction():
            await insert_pending(conn, message.task_id, message.created_at)
            await publish_task(broker, message, settings)
    except (QueueOverloadedError, TimeoutError, aiormq.exceptions.AMQPError) as e:
        raise _overloaded(type(e).__name__) from e
    return TaskAccepted(task_id=message.task_id, status="PENDING")


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def read_task(task_id: UUID, request: Request, response: Response):
    pool = request.app.state.pool
    if pool is None:
        raise _overloaded("database unavailable")
    task = await get_task(pool, task_id)
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Task not found")
    if task["status"] == "PENDING":
        response.headers["Retry-After"] = "1"
    return task
