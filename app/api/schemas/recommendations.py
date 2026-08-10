"""Response schemas for health and recommendations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Application health and model availability."""

    health: Literal["check"]
    model_status: Literal["loaded", "missing"]


class RecommendationItem(BaseModel):
    """Serialized film recommendation."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    director: str | list[str]
    year: int | None
    match_score: float


class RecommendationResponse(BaseModel):
    """Recommendation response preserving the existing API contract."""

    model_config = ConfigDict(from_attributes=True)

    user_id: int
    strategy: str | None = None
    info: str | None = None
    recommendations: list[RecommendationItem]
