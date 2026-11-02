import time
from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter, status

from credit_scoring.database import check_database_connection

router = APIRouter()


try:
    APP_VERSION = version("credit-scoring-service")
except PackageNotFoundError:
    APP_VERSION = "0.1.0-dev"


@router.get("/version", status_code=status.HTTP_200_OK)
async def get_version():
    return {"version": APP_VERSION}


@router.get("/health", status_code=status.HTTP_200_OK)
async def end_to_end_health():
    start_time = time.perf_counter()
    is_db_ok, db_error = await check_database_connection()
    db_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    db_status = "healthy" if is_db_ok else f"unhealthy: {db_error}"
    overall_status = "healthy" if is_db_ok else "degraded"

    return {
        "status": overall_status,
        "components": {"database": {"status": db_status, "latency_ms": db_latency_ms}},
    }
