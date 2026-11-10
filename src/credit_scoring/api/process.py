from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from credit_scoring.schemas import CreditFeatures, ModelInfoResponse, ProcessResponse

router = APIRouter()


@router.post("/process", response_model=ProcessResponse)
async def process(features: CreditFeatures, request: Request):
    """Синхронный inference моделью, загруженной из MLflow один раз при старте."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded"
        )
    probability = await run_in_threadpool(model.predict_proba, features.model_dump())
    return ProcessResponse(
        probability_default=probability,
        decision=model.decide(probability),
        model=ModelInfoResponse(
            name=model.info.name,
            alias=model.info.alias,
            version=model.info.version,
            run_id=model.info.run_id,
        ),
    )
