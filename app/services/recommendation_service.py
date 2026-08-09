"""Current SVD recommendation artifact lifecycle and inference."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.domain.recommendations import Recommendation, RecommendationResult
from app.ml.faiss_index import prepare_faiss_inputs, validate_faiss_index
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
    retrieval_index: faiss.IndexIDMap2


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
        self._artifacts: _SVDArtifacts | None = None

    @property
    def is_model_loaded(self) -> bool:
        """Return whether valid SVD artifacts are available in memory."""
        return self._artifacts is not None

    def load_artifacts(self) -> bool:
        """Load and validate NumPy, JSON, and exact FAISS artifacts.

        Returns:
            True when all artifacts were loaded and validated; otherwise False.
        """
        embeddings_path = self._artifact_root / "item_embeddings.npy"
        index_path = self._artifact_root / "film_index.json"
        retrieval_path = self._artifact_root / "retrieval.faiss"

        try:
            item_vectors = np.load(embeddings_path, allow_pickle=False)
            with index_path.open(encoding="utf-8") as index_file:
                raw_film_ids = json.load(index_file)
            retrieval_index = faiss.read_index(str(retrieval_path))

            _, validated_ids = prepare_faiss_inputs(item_vectors, raw_film_ids)
            validate_faiss_index(
                retrieval_index,
                item_vectors.shape,
                validated_ids,
            )
            film_index = tuple(validated_ids.tolist())

            self._artifacts = _SVDArtifacts(
                item_vectors=item_vectors,
                film_index=film_index,
                id_to_position={
                    film_id: index for index, film_id in enumerate(film_index)
                },
                retrieval_index=retrieval_index,
            )
            logger.info(
                "Loaded recommendation artifacts for %d films with dimension %d",
                len(film_index),
                item_vectors.shape[1],
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


_recommendation_service = RecommendationService()


def get_recommendation_service() -> RecommendationService:
    """Return the process-wide recommendation service."""
    return _recommendation_service
