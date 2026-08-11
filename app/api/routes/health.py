"""Application health routes."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.schemas.recommendations import HealthResponse
from app.services.recommendation_backend import RecommendationBackend
from app.services.recommendation_service import (
    get_recommendation_service,
)

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health(
    service: Annotated[RecommendationBackend, Depends(get_recommendation_service)],
) -> HealthResponse:
    """Return application and live SVD artifact health."""
    return HealthResponse(
        health="check",
        model_status="loaded" if service.is_model_loaded else "missing",
    )
