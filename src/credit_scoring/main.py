from fastapi import FastAPI, status

from credit_scoring.api.v1.endpoints import router as api_v1_router

app = FastAPI(
    title="Credit Scoring MLOps Service",
    description="Asynchronous microservice for credit scoring model inference",
    version="0.1.0",
)


@app.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz():
    """Инфраструктурная liveness-проба."""
    return {"status": "ok"}


app.include_router(api_v1_router, prefix="/api/v1")
