import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

import aio_pika
import httpx

from credit_scoring.config import Settings
from credit_scoring.database import postgres_version

logger = logging.getLogger(__name__)


async def rabbitmq_version(url: str) -> str:
    connection = await aio_pika.connect(url, timeout=2.0)
    try:
        return connection.transport.connection.server_properties["version"]
    finally:
        await connection.close()


async def mlflow_version(url: str) -> str:
    async with httpx.AsyncClient(timeout=2.0) as client:
        response = await client.get(f"{url.rstrip('/')}/version")
        response.raise_for_status()
        return response.text.strip()


async def _probe(name: str, check: Callable[[], Awaitable[str]]) -> dict:
    start = time.perf_counter()
    try:
        result = {"status": "healthy", "version": await check()}
    except Exception:  # noqa: BLE001 - любая ошибка компонента это его статус
        # Текст ошибки может раскрыть адреса и учётки, поэтому он только в логе.
        logger.exception("component unhealthy", extra={"component": name})
        result = {"status": "unhealthy", "version": None}
    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return result


async def check_components(settings: Settings) -> dict[str, dict]:
    postgres, rabbitmq, mlflow = await asyncio.gather(
        _probe("postgres", lambda: postgres_version(settings.database_url)),
        _probe("rabbitmq", lambda: rabbitmq_version(settings.rabbit_url)),
        _probe("mlflow", lambda: mlflow_version(settings.mlflow_tracking_uri)),
    )
    return {"postgres": postgres, "rabbitmq": rabbitmq, "mlflow": mlflow}
