"""Current SVD recommendation artifact lifecycle and inference."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.domain.recommendations import Recommendation, RecommendationResult
from app.repositories.films import FilmRepository
from app.repositories.interactions import InteractionRepository

logger = logging.getLogger(__name__)

RECOMMENDATION_STRATEGY = "SVD_Mean_Pooling"
NO_WATCHED_FILMS_INFO = (
    "No watched films found for this user. Cannot provide recommendations."
)
NO_USABLE_RATINGS_INFO = (
    "No rated films found in the recommendation model for this user. "
    "Cannot provide recommendations."
)


class ModelUnavailableError(Exception):
    """Raised when recommendation artifacts have not been loaded."""


@dataclass(frozen=True, slots=True)
class _SVDArtifacts:
    item_vectors: NDArray[np.floating]
    film_index: tuple[int, ...]
    id_to_position: dict[int, int]


class RecommendationService:
    """Serve recommendations using the existing SVD mean-pooling baseline."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = SessionLocal,
        artifact_root: str | Path | None = None,
    ) -> None:
        settings = get_settings()
        self._session_factory = session_factory
        self._artifact_root = Path(artifact_root or settings.ARTIFACT_ROOT)
        self._artifacts: _SVDArtifacts | None = None

    @property
    def is_model_loaded(self) -> bool:
        """Return whether valid SVD artifacts are available in memory."""
        return self._artifacts is not None

    def load_artifacts(self) -> bool:
        """Load and validate the current NumPy and JSON artifact files.

        Returns:
            True when both artifacts were loaded and validated; otherwise False.
        """
        embeddings_path = self._artifact_root / "item_embeddings.npy"
        index_path = self._artifact_root / "film_index.json"

        try:
            item_vectors = np.load(embeddings_path, allow_pickle=False)
            with index_path.open(encoding="utf-8") as index_file:
                raw_film_ids = json.load(index_file)

            film_index = tuple(int(film_id) for film_id in raw_film_ids)
            if item_vectors.ndim != 2:
                raise ValueError("item embeddings must be a two-dimensional array")
            if item_vectors.shape[0] != len(film_index):
                raise ValueError("embedding rows and film index length differ")
            if len(set(film_index)) != len(film_index):
                raise ValueError("film index contains duplicate IDs")

            self._artifacts = _SVDArtifacts(
                item_vectors=item_vectors,
                film_index=film_index,
                id_to_position={
                    film_id: index for index, film_id in enumerate(film_index)
                },
            )
            logger.info("Loaded recommendation artifacts for %d films", len(film_index))
            return True
        except FileNotFoundError:
            logger.warning(
                "Recommendation artifacts were not found in %s", self._artifact_root
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            logger.exception("Recommendation artifacts are invalid")

        self._artifacts = None
        return False

    def unload_artifacts(self) -> None:
        """Release loaded recommendation artifacts."""
        self._artifacts = None
        logger.info("Recommendation artifacts unloaded")

    async def recommend(self, user_id: int) -> RecommendationResult:
        """Generate up to ten recommendations using current SVD semantics.

        Args:
            user_id: Database identifier for the target user.

        Returns:
            Ordered recommendation results.

        Raises:
            ModelUnavailableError: If artifacts are not currently loaded.
        """
        artifacts = self._artifacts
        if artifacts is None:
            raise ModelUnavailableError

        async with self._session_factory() as session:
            interaction_repository = InteractionRepository(session)
            watched_film_ids = await interaction_repository.get_watched_film_ids(
                user_id
            )
            if not watched_film_ids:
                return RecommendationResult(
                    user_id=user_id,
                    info=NO_WATCHED_FILMS_INFO,
                    recommendations=(),
                )

            rated_film_ids = await interaction_repository.get_rated_film_ids(user_id)
            rated_indexes = [
                artifacts.id_to_position[film_id]
                for film_id in rated_film_ids
                if film_id in artifacts.id_to_position
            ]
            if not rated_indexes:
                return RecommendationResult(
                    user_id=user_id,
                    info=NO_USABLE_RATINGS_INFO,
                    recommendations=(),
                )

            user_vector = np.mean(artifacts.item_vectors[rated_indexes], axis=0)
            scores = np.dot(artifacts.item_vectors, user_vector)

            watched_indexes = {
                artifacts.id_to_position[film_id]
                for film_id in watched_film_ids
                if film_id in artifacts.id_to_position
            }
            candidate_indexes = np.array(
                [
                    index
                    for index in range(len(artifacts.film_index))
                    if index not in watched_indexes
                ],
                dtype=int,
            )
            candidate_scores = scores[candidate_indexes]
            top_count = min(10, len(candidate_indexes))
            top_indices = candidate_indexes[
                np.argsort(candidate_scores)[-top_count:][::-1]
            ]
            top_ids = [artifacts.film_index[index] for index in top_indices]
            films = await FilmRepository(session).get_by_ids(top_ids)

        films_by_id = {film.id: film for film in films}
        recommendations = []
        for film_id in top_ids:
            film = films_by_id.get(film_id)
            if film is None:
                continue
            director: str | list[str] = film.directors[0].name if film.directors else []
            recommendations.append(
                Recommendation(
                    id=film.id,
                    title=film.title,
                    director=director,
                    year=film.year,
                    match_score=round(
                        float(scores[artifacts.id_to_position[film_id]]), 4
                    ),
                )
            )

        return RecommendationResult(
            user_id=user_id,
            strategy=RECOMMENDATION_STRATEGY,
            recommendations=tuple(recommendations),
        )


_recommendation_service = RecommendationService()


def get_recommendation_service() -> RecommendationService:
    """Return the process-wide recommendation service."""
    return _recommendation_service
