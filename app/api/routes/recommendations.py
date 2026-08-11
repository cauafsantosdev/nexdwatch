"""Recommendation routes."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.api.schemas.recommendations import (
    RecommendationResponse,
)
from app.services.recommendation_backend import RecommendationBackend
from app.services.recommendation_service import (
    ModelUnavailableError,
    get_recommendation_service,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/users/{user_id}/recommendations",
    response_model=RecommendationResponse,
    response_model_exclude_none=True,
)
async def recommendations(
    user_id: Annotated[int, Path(gt=0)],
    service: Annotated[RecommendationBackend, Depends(get_recommendation_service)],
) -> RecommendationResponse:
    """Return recommendations from the live SVD mean-pooling service."""
    try:
        result = await service.recommend(user_id)
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not loaded.",
        ) from exc
    except Exception as exc:
        logger.exception("Recommendation generation failed for user_id=%d", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recommendation generation failed.",
        ) from exc

    return RecommendationResponse.model_validate(result)
