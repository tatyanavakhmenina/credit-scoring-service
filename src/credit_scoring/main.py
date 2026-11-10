import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.concurrency import run_in_threadpool
from faststream.rabbit import RabbitBroker

from credit_scoring import __version__
from credit_scoring.api.process import router as process_router
from credit_scoring.api.v1.endpoints import router as api_v1_router
from credit_scoring.api.v1.tasks import router as tasks_router
from credit_scoring.broker import declare_topology
from credit_scoring.config import settings
from credit_scoring.database import create_pool
from credit_scoring.logging_config import configure_logging
from credit_scoring.ml.model import load_model

logger = logging.getLogger(__name__)


async def _optional(name: str, factory):
    """Сервис поднимается и без зависимости: это видно в /api/v1/health и по 503."""
    try:
        return await factory()
    except Exception:  # noqa: BLE001 - зависимость может упасть с любой ошибкой
        logger.exception("dependency unavailable on startup", extra={"name": name})
        return None


async def _start_broker() -> RabbitBroker:
    broker = RabbitBroker(settings.rabbit_url, logger=logger)
    await broker.start()
    await declare_topology(broker, settings)
    return broker


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(settings.log_level)
    app.state.pool = await _optional(
        "postgres", lambda: create_pool(settings.database_url)
    )
    app.state.broker = await _optional("rabbitmq", _start_broker)
    app.state.model = await _optional(
        "model", lambda: run_in_threadpool(load_model, settings)
    )
    logger.info("app started", extra={"version": __version__})
    yield
    if app.state.broker:
        await app.state.broker.stop()
    if app.state.pool:
        await app.state.pool.close()


app = FastAPI(
    title="Credit Scoring MLOps Service",
    description="Asynchronous microservice for credit scoring model inference",
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", uuid4().hex)
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
        },
    )
    return response


@app.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz():
    """Инфраструктурная liveness-проба."""
    return {"status": "ok"}


app.include_router(process_router)
app.include_router(api_v1_router, prefix="/api/v1")
app.include_router(tasks_router, prefix="/api/v1")
