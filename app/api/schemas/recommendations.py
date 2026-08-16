"""Product response schemas for health and recommendations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Application health and model availability."""

    health: Literal["check"]
    model_status: Literal["loaded", "missing"]
    model_version: str | None = None


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


PublicReasonCode = Literal[
    "GLOBAL_RRF",
    "NON_HEAD_TASTE_MATCH",
    "BRAZILIAN_CINEMA_DISCOVERY",
    "ANCHOR_SIMILARITY",
    "DIRECTOR_AFFINITY",
    "GENRE_AFFINITY",
    "DECADE_AFFINITY",
    "WORLD_CINEMA_DISCOVERY",
    "LATENT_MATCH_METADATA_NOVELTY",
    "CLASSIC_CINEMA_DISCOVERY",
]
PublicEntityType = Literal["director", "genre", "decade", "country", "language"]


class RecommendationAnchorResponse(BaseModel):
    """Product-safe film evidence for an anchor-based recommendation."""

    film_id: int = Field(description="NexdWatch film identifier.")
    title: str = Field(description="Display title of the positively rated anchor.")


class RecommendationEntityResponse(BaseModel):
    """Named preference evidence without internal entity identifiers or scores."""

    type: PublicEntityType = Field(description="Preference family.")
    name: str = Field(description="Frontend-displayable entity name.")


class RecommendationReasonResponse(BaseModel):
    """Structured explanation input for frontend-owned copy."""

    code: PublicReasonCode = Field(description="Stable product explanation concept.")
    anchor: RecommendationAnchorResponse | None = None
    entity: RecommendationEntityResponse | None = None


class RecommendationFeedItemResponse(BaseModel):
    """One film in a categorized recommendation row."""

    film_id: int = Field(description="NexdWatch film identifier.")
    title: str
    year: int | None
    directors: list[str]
    tmdb_id: int | None = Field(description="TMDB identifier for poster resolution.")
    slug: str = Field(description="Canonical Letterboxd film slug.")
    reason: RecommendationReasonResponse


class RecommendationPreferenceContextResponse(BaseModel):
    """Rating evidence explaining an affinity-derived category selection."""

    average_rating: float = Field(description="Mean explicit rating for the preference.")
    rated_count: int = Field(description="Rated films supporting the preference.")


class RecommendationCategoryResponse(BaseModel):
    """One active, non-empty product recommendation row."""

    key: str = Field(description="Stable frontend category key.")
    title: str = Field(description="Frontend-displayable category title.")
    experimental: bool = Field(
        description="Whether this category is under explicit product evaluation."
    )
    preference_context: RecommendationPreferenceContextResponse | None = None
    items: list[RecommendationFeedItemResponse]


class RecommendationFeedResponse(BaseModel):
    """Evidence-driven categorized recommendation feed for one user."""

    user_id: int
    categories: list[RecommendationCategoryResponse]
