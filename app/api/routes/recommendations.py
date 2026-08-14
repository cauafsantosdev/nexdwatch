"""Recommendation routes."""

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Request, status

from app.api.mappers.recommendations import map_recommendation_feed
from app.api.schemas.recommendations import (
    RecommendationFeedResponse,
    RecommendationResponse,
)
from app.services.categorized_recommendation_service import (
    CategorizedRecommendationService,
    CategoryPolicyResourcesUnavailableError,
    RecommendationUserNotFoundError,
)
from app.services.recommendation_backend import RecommendationBackend
from app.services.recommendation_service import (
    ModelUnavailableError,
    get_recommendation_service,
)

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_categorized_recommendation_service(
    request: Request,
) -> CategorizedRecommendationService:
    """Resolve the worker-owned categorized service from application state."""
    service = getattr(request.app.state, "categorized_recommendation_service", None)
    if service is None or not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Categorized recommendations are temporarily unavailable.",
        )
    return service


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


@router.get(
    "/recommendations/{user_id}/feed",
    response_model=RecommendationFeedResponse,
    response_model_exclude_none=True,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "User not found."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Categorized recommendation resources unavailable."
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Unexpected categorized recommendation failure."
        },
    },
)
async def recommendation_feed(
    user_id: Annotated[int, Path(gt=0, description="NexdWatch user identifier.")],
    service: Annotated[
        CategorizedRecommendationService,
        Depends(get_categorized_recommendation_service),
    ],
) -> RecommendationFeedResponse:
    """Return the active categorized recommendation rows for one user."""
    started = time.perf_counter()
    try:
        result = await service.recommend(user_id)
        mapping_started = time.perf_counter()
        response = map_recommendation_feed(result)
        mapping_ms = (time.perf_counter() - mapping_started) * 1000
    except RecommendationUserNotFoundError as exc:
        logger.info("categorized_feed user_not_found user_id=%d", user_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        ) from exc
    except CategoryPolicyResourcesUnavailableError as exc:
        logger.warning("categorized_feed resources_unavailable user_id=%d", user_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Categorized recommendations are temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception("categorized_feed failure user_id=%d", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Categorized recommendation generation failed.",
        ) from exc

    category_keys = [category.key for category in response.categories]
    film_count = sum(len(category.items) for category in response.categories)
    logger.info(
        "categorized_feed success user_id=%d latency_ms=%.2f mapping_ms=%.2f "
        "categories=%d films=%d category_keys=%s outside_usual_active=%s",
        user_id,
        (time.perf_counter() - started) * 1000,
        mapping_ms,
        len(response.categories),
        film_count,
        category_keys,
        "outside_usual" in category_keys,
    )
    return response
