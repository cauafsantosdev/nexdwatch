"""Recommendation-service contract shared by API dependencies."""

from typing import Protocol

from app.domain.recommendations import RecommendationResult

NO_WATCHED_FILMS_INFO = (
    "No watched films found for this user. Cannot provide recommendations."
)
NO_USABLE_RATINGS_INFO = (
    "No rated films found in the recommendation model for this user. "
    "Cannot provide recommendations."
)


class ModelUnavailableError(Exception):
    """Raised when recommendation artifacts have not been loaded."""


class RecommendationBackend(Protocol):
    """Lifecycle and inference operations required by recommendation routes."""

    @property
    def is_model_loaded(self) -> bool:
        """Report whether a complete recommendation bundle is serving."""
        ...

    def load_artifacts(self) -> bool:
        """Load the active bundle, returning whether serving is ready."""
        ...

    def unload_artifacts(self) -> None:
        """Release all in-process artifact references."""
        ...

    async def recommend(self, user_id: int) -> RecommendationResult:
        """Produce the transport-neutral recommendation result for one user."""
        ...
