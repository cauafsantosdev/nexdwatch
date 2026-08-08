"""Profile synchronization and recommendation routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.api.schemas.recommendations import (
    RecommendationResponse,
    SyncUserResponse,
)
from app.services.profile_service import (
    EmptyProfileError,
    ProfileService,
    get_profile_service,
)
from app.services.recommendation_service import (
    ModelUnavailableError,
    RecommendationService,
    get_recommendation_service,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/users/{username}/sync-logs", response_model=SyncUserResponse)
async def sync_logs(
    username: str = Path(min_length=1, max_length=15),
    service: ProfileService = Depends(get_profile_service),
) -> SyncUserResponse:
    """Scrape and persist a public Letterboxd profile."""
    try:
        result = await service.sync_profile(username.strip())
    except EmptyProfileError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found or no logs available.",
        ) from exc
    except Exception as exc:
        logger.exception("Profile synchronization failed for username=%s", username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile synchronization failed.",
        ) from exc

    return SyncUserResponse(
        status="ok",
        user_id=result.user_id,
        logs_count=result.logs_count,
    )


@router.get(
    "/users/{user_id}/recommendations",
    response_model=RecommendationResponse,
    response_model_exclude_none=True,
)
async def recommendations(
    user_id: int = Path(gt=0),
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    """Return current SVD recommendations for a persisted user."""
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
