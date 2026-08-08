"""Application health routes."""

from fastapi import APIRouter, Depends

from app.api.schemas.recommendations import HealthResponse
from app.services.recommendation_service import (
    RecommendationService,
    get_recommendation_service,
)

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health(
    service: RecommendationService = Depends(get_recommendation_service),
) -> HealthResponse:
    """Return application and recommendation artifact health."""
    return HealthResponse(
        health="check",
        model_status="loaded" if service.is_model_loaded else "missing",
    )
