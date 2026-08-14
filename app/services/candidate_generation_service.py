"""Internal broad candidate generation for a future personalized ranker."""

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.domain.candidates import CandidateGenerationResult, RecommendationCandidate
from app.ml.candidate_policy import (
    FINAL_POPULARITY_DEPTH,
    FINAL_WEIGHTED_SVD_DEPTH,
)
from app.ml.candidate_retrieval import retrieve_exact_candidates
from app.ml.popularity import PopularityArtifact, read_popularity_artifact
from app.ml.ratings import rating_to_bucket
from app.ml.svd_artifacts import SVDArtifacts, load_svd_artifacts
from app.ml.svd_profiles import build_svd_profile
from app.repositories.interactions import InteractionRepository, RecommendationHistory
from app.services.category_request_profile import (
    CategoryRequestProfile,
    request_stage,
)

logger = logging.getLogger(__name__)

DEFAULT_POPULARITY_ARTIFACT = Path("candidates/popularity.json")


class CandidateArtifactsUnavailableError(Exception):
    """Raised when candidate-generation artifacts are not loaded."""


class CandidateGenerationService:
    """Generate weighted-SVD and controlled-popularity candidate inventories."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = SessionLocal,
        artifact_root: str | Path | None = None,
        *,
        popularity_path: str | Path | None = None,
        svd_depth: int = FINAL_WEIGHTED_SVD_DEPTH,
        popularity_depth: int = FINAL_POPULARITY_DEPTH,
    ) -> None:
        if svd_depth < 0 or popularity_depth < 0:
            raise ValueError("candidate source depths must be non-negative")
        settings = get_settings()
        self._session_factory = session_factory
        self._artifact_root = Path(artifact_root or settings.ARTIFACT_ROOT)
        self._popularity_path = Path(
            popularity_path or self._artifact_root / DEFAULT_POPULARITY_ARTIFACT
        )
        self._svd_depth = svd_depth
        self._popularity_depth = popularity_depth
        self._svd_artifacts: SVDArtifacts | None = None
        self._popularity_artifact: PopularityArtifact | None = None

    @property
    def is_loaded(self) -> bool:
        return self._svd_artifacts is not None and self._popularity_artifact is not None

    def load_artifacts(self) -> bool:
        """Load both candidate sources once and validate their catalog parity."""
        try:
            svd = load_svd_artifacts(self._artifact_root)
            popularity = read_popularity_artifact(self._popularity_path)
            svd_ids = set(svd.film_index)
            popularity_ids = {entry.film_id for entry in popularity.films}
            if not popularity_ids.issubset(svd_ids):
                raise ValueError("popularity catalog is not contained in SVD catalog")
            self._svd_artifacts = svd
            self._popularity_artifact = popularity
            return True
        except (FileNotFoundError, OSError, ValueError, TypeError):
            logger.exception(
                "Candidate-generation artifacts are unavailable or invalid"
            )
            self.unload_artifacts()
            return False

    def unload_artifacts(self) -> None:
        self._svd_artifacts = None
        self._popularity_artifact = None

    @property
    def svd_artifacts(self) -> SVDArtifacts | None:
        """Expose loaded artifacts to internal downstream policy services."""
        return self._svd_artifacts

    @property
    def popularity_artifact(self) -> PopularityArtifact | None:
        """Expose controlled popularity metadata without mutation."""
        return self._popularity_artifact

    async def generate(self, user_id: int) -> CandidateGenerationResult:
        """Generate a deterministic variable-size candidate inventory."""
        svd = self._svd_artifacts
        popularity = self._popularity_artifact
        if svd is None or popularity is None:
            raise CandidateArtifactsUnavailableError

        async with self._session_factory() as session:
            history = await InteractionRepository(session).get_recommendation_history(
                user_id
            )
        return self.generate_from_history(user_id, history)

    def generate_from_history(
        self,
        user_id: int,
        history: RecommendationHistory,
        *,
        profiler: CategoryRequestProfile | None = None,
    ) -> CandidateGenerationResult:
        """Generate candidates from a history already loaded by an orchestrator."""
        svd = self._svd_artifacts
        popularity = self._popularity_artifact
        if svd is None or popularity is None:
            raise CandidateArtifactsUnavailableError

        with request_stage(profiler, "candidate_profile_construction"):
            watched_ids = set(history.watched_film_ids)
            rated = history.rated_interactions
            item_rows: list[int] = []
            rating_buckets: list[int] = []
            for interaction in rated:
                row = svd.id_to_position.get(interaction.film_id)
                if row is None:
                    continue
                try:
                    bucket = rating_to_bucket(interaction.rating)
                except ValueError:
                    logger.warning(
                        "Ignoring invalid persisted rating user_id=%d film_id=%d",
                        user_id,
                        interaction.film_id,
                    )
                    continue
                item_rows.append(row)
                rating_buckets.append(bucket)
            query = build_svd_profile(
                svd.item_vectors,
                np.ascontiguousarray(item_rows, dtype=np.int64),
                np.ascontiguousarray(rating_buckets, dtype=np.int64),
                "svd_positive_weighted",
            )
        with request_stage(profiler, "svd_candidate_retrieval"):
            svd_candidates = (
                retrieve_exact_candidates(
                    svd.retrieval_index,
                    query,
                    excluded_film_ids=watched_ids,
                    depth=self._svd_depth,
                )
                if query is not None
                else ()
            )
        with request_stage(profiler, "popularity_candidate_merge"):
            popularity_candidates = []
            for entry in popularity.films:
                if entry.film_id in watched_ids:
                    continue
                popularity_candidates.append(entry)
                if len(popularity_candidates) == self._popularity_depth:
                    break

            ordered_ids: list[int] = []
            merged: dict[int, RecommendationCandidate] = {}
            for rank, (film_id, score) in enumerate(svd_candidates, start=1):
                ordered_ids.append(film_id)
                merged[film_id] = RecommendationCandidate(
                    film_id=film_id,
                    svd_score=score,
                    svd_rank=rank,
                    retrieved_by_svd=True,
                )
            for entry in popularity_candidates:
                candidate = merged.get(entry.film_id)
                if candidate is None:
                    ordered_ids.append(entry.film_id)
                    candidate = RecommendationCandidate(film_id=entry.film_id)
                merged[entry.film_id] = replace(
                    candidate,
                    popularity_score=entry.positive_interaction_count,
                    popularity_rank=entry.rank,
                    retrieved_by_popularity=True,
                )
            candidates = tuple(merged[film_id] for film_id in ordered_ids)
        if watched_ids.intersection(candidate.film_id for candidate in candidates):
            raise RuntimeError("watched-film exclusion invariant was violated")
        if profiler is not None:
            profiler.count("watched_count", len(watched_ids))
            profiler.count("rated_count", len(rated))
            profiler.count("indexed_rated_count", len(item_rows))
            profiler.count("svd_candidate_count", len(svd_candidates))
            profiler.count("candidate_count", len(candidates))
        return CandidateGenerationResult(
            user_id=user_id,
            candidates=candidates,
            nominal_budget=self._svd_depth + self._popularity_depth,
            svd_depth=self._svd_depth,
            popularity_depth=self._popularity_depth,
            svd_profile_available=query is not None,
        )
