"""Current SVD recommendation artifact lifecycle and inference."""

import logging
from pathlib import Path

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.domain.recommendations import Recommendation, RecommendationResult
from app.ml.svd_artifacts import SVDArtifacts, load_svd_artifacts
from app.repositories.films import FilmRepository
from app.repositories.interactions import InteractionRepository
from app.services.recommendation_backend import (
    NO_USABLE_RATINGS_INFO,
    NO_WATCHED_FILMS_INFO,
    ModelUnavailableError,
    RecommendationBackend,
)

logger = logging.getLogger(__name__)

RECOMMENDATION_STRATEGY = "SVD_Mean_Pooling"


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
        self._retrieval_top_k = settings.RETRIEVAL_TOP_K
        self._artifacts: SVDArtifacts | None = None

    @property
    def is_model_loaded(self) -> bool:
        """Return whether valid SVD artifacts are available in memory."""
        return self._artifacts is not None

    def load_artifacts(self) -> bool:
        """Load and validate NumPy, JSON, and exact FAISS artifacts.

        Returns:
            True when all artifacts were loaded and validated; otherwise False.
        """
        try:
            self._artifacts = load_svd_artifacts(self._artifact_root)
            logger.info(
                "Loaded recommendation artifacts for %d films with dimension %d",
                len(self._artifacts.film_index),
                self._artifacts.item_vectors.shape[1],
            )
            return True
        except FileNotFoundError:
            logger.warning(
                "Recommendation artifacts were not found in %s", self._artifact_root
            )
        except Exception:
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
            query = np.ascontiguousarray(
                user_vector.reshape(1, -1),
                dtype=np.float32,
            )
            watched_set = set(watched_film_ids)
            indexed_watched_count = sum(
                film_id in artifacts.id_to_position for film_id in watched_film_ids
            )
            requested_k = min(
                int(artifacts.retrieval_index.ntotal),
                self._retrieval_top_k + indexed_watched_count,
            )
            faiss_scores, faiss_ids = artifacts.retrieval_index.search(
                query,
                requested_k,
            )

            candidates: list[tuple[int, float]] = []
            for raw_film_id, raw_score in zip(
                faiss_ids[0], faiss_scores[0], strict=True
            ):
                film_id = int(raw_film_id)
                if film_id < 0 or film_id in watched_set:
                    continue
                candidates.append((film_id, float(raw_score)))
                if len(candidates) == self._retrieval_top_k:
                    break

            candidate_ids = [film_id for film_id, _ in candidates]
            films = await FilmRepository(session).get_by_ids(candidate_ids)

        films_by_id = {film.id: film for film in films}
        recommendations = []
        for film_id, score in candidates:
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
                    match_score=round(score, 4),
                )
            )
            if len(recommendations) == 10:
                break

        return RecommendationResult(
            user_id=user_id,
            strategy=RECOMMENDATION_STRATEGY,
            recommendations=tuple(recommendations),
        )


def build_recommendation_service(
    *,
    session_factory: async_sessionmaker[AsyncSession] = SessionLocal,
    artifact_root: str | Path | None = None,
) -> RecommendationBackend:
    """Construct the live SVD mean-pooling recommendation service."""
    settings = get_settings()
    root = Path(artifact_root or settings.ARTIFACT_ROOT)
    service: RecommendationBackend = RecommendationService(session_factory, root)
    logger.info("Configured live SVD mean-pooling recommendation service")
    return service


_recommendation_service = build_recommendation_service()


def get_recommendation_service() -> RecommendationBackend:
    """Return the process-wide live SVD recommendation service."""
    return _recommendation_service
