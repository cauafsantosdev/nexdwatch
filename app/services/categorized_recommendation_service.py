"""Internal orchestration for categorized recommendations; not a public backend."""

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.domain.categorized_recommendations import (
    CategorizedRecommendation,
    CategorizedRecommendationResult,
    RecommendationCategory,
)
from app.policy.catalog import PolicyCatalog, load_policy_catalog
from app.policy.config import DEFAULT_POLICY_CONFIG, CategoryPolicyConfig
from app.policy.engine import CategorizedPolicyEngine
from app.policy.profile import build_user_category_profile
from app.policy.ranking import rank_candidates_by_rrf
from app.repositories.films import FilmRepository
from app.repositories.interactions import InteractionRepository
from app.services.candidate_generation_service import CandidateGenerationService


class CategoryPolicyResourcesUnavailableError(RuntimeError):
    """Raised when internal candidate or catalog resources are not loaded."""


class CategorizedRecommendationService:
    """Compose finalized retrieval, RRF, and policy without FastAPI integration."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = SessionLocal,
        artifact_root: str | Path | None = None,
        *,
        config: CategoryPolicyConfig = DEFAULT_POLICY_CONFIG,
    ) -> None:
        settings = get_settings()
        self._session_factory = session_factory
        self._candidate_service = CandidateGenerationService(
            session_factory, artifact_root or settings.ARTIFACT_ROOT
        )
        self._catalog: PolicyCatalog | None = None
        self._config = config

    @property
    def is_loaded(self) -> bool:
        return self._candidate_service.is_loaded and self._catalog is not None

    async def load_resources(self) -> bool:
        """Load immutable artifacts and one bounded policy catalog snapshot."""
        if not self._candidate_service.load_artifacts():
            return False
        svd = self._candidate_service.svd_artifacts
        if svd is None:
            return False
        async with self._session_factory() as session:
            self._catalog = await load_policy_catalog(session, svd.film_index)
        return True

    def unload_resources(self) -> None:
        self._candidate_service.unload_artifacts()
        self._catalog = None

    async def recommend(self, user_id: int) -> CategorizedRecommendationResult:
        """Build an internal categorized result from one user-history read."""
        catalog = self._catalog
        svd = self._candidate_service.svd_artifacts
        popularity = self._candidate_service.popularity_artifact
        if catalog is None or svd is None or popularity is None:
            raise CategoryPolicyResourcesUnavailableError
        async with self._session_factory() as session:
            history = await InteractionRepository(session).get_recommendation_history(
                user_id
            )
        candidates = self._candidate_service.generate_from_history(user_id, history)
        popularity_rank_by_film = {
            entry.film_id: entry.rank for entry in popularity.films
        }
        ranked = rank_candidates_by_rrf(
            candidates.candidates,
            popularity_rank_by_film,
            popularity.film_count,
            config=self._config,
        )
        profile = build_user_category_profile(
            user_id, history, catalog, config=self._config
        )
        policy = CategorizedPolicyEngine(
            catalog,
            svd.item_vectors,
            svd.id_to_position,
            config=self._config,
        ).categorize(ranked, profile)
        selected_ids = tuple(
            dict.fromkeys(
                film_id
                for category in policy.allocated_categories
                for film_id in category.film_ids
            )
        )
        async with self._session_factory() as session:
            films = await FilmRepository(session).get_by_ids(selected_ids)
        display_by_id = {film.id: film for film in films}
        ranked_by_id = {candidate.film_id: candidate for candidate in ranked}
        categories = []
        for allocated in policy.allocated_categories:
            items = []
            for film_id in allocated.film_ids:
                film = display_by_id.get(film_id)
                ranked_candidate = ranked_by_id[film_id]
                if film is None:
                    continue
                items.append(
                    CategorizedRecommendation(
                        film_id=film_id,
                        title=film.title,
                        year=film.year,
                        directors=tuple(value.name for value in film.directors),
                        reason=allocated.proposal.reasons[film_id],
                        rrf_rank=ranked_candidate.rrf_rank,
                        popularity_stratum=ranked_candidate.popularity_stratum,
                        source_membership=ranked_candidate.source_membership,
                    )
                )
            if len(items) < allocated.proposal.minimum_size:
                continue
            categories.append(
                RecommendationCategory(
                    key=allocated.proposal.key,
                    family=allocated.proposal.family,
                    role=allocated.proposal.role,
                    title=allocated.proposal.title_template.format(
                        **allocated.proposal.title_parameters
                    ),
                    items=tuple(items),
                    evidence_tier=allocated.proposal.evidence_tier,
                    evidence_support=allocated.proposal.evidence_support,
                )
            )
        return CategorizedRecommendationResult(
            user_id=user_id,
            categories=tuple(categories),
            diagnostics=policy.diagnostics,
        )
