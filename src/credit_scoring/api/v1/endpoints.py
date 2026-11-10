from fastapi import APIRouter, Request, Response, status

from credit_scoring import __version__
from credit_scoring.api.health import check_components
from credit_scoring.config import settings

router = APIRouter()


@router.get("/version", status_code=status.HTTP_200_OK)
async def get_version():
    return {"version": __version__, "git_sha": settings.git_sha}


@router.get("/health")
async def end_to_end_health(request: Request, response: Response):
    components = await check_components(settings)
    is_healthy = all(c["status"] == "healthy" for c in components.values())
    response.status_code = (
        status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    model = getattr(request.app.state, "model", None)
    return {
        "status": "healthy" if is_healthy else "degraded",
        "components": components,
        "model": model.info.__dict__ if model else None,
    }
