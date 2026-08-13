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
from app.repositories.interactions import InteractionRepository
from app.services.candidate_generation_service import CandidateGenerationService
from app.services.category_request_profile import (
    CategoryRequestProfile,
    request_stage,
)


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
        self._policy_engine: CategorizedPolicyEngine | None = None
        self._popularity_rank_by_film: dict[int, int] | None = None
        self._config = config

    @property
    def is_loaded(self) -> bool:
        return (
            self._candidate_service.is_loaded
            and self._catalog is not None
            and self._policy_engine is not None
            and self._popularity_rank_by_film is not None
        )

    def load_candidate_artifacts(self) -> bool:
        """Load frozen candidate artifacts separately for resource measurement."""
        return self._candidate_service.load_artifacts()

    async def load_policy_catalog(self) -> bool:
        """Load and intern the bounded policy metadata snapshot."""
        svd = self._candidate_service.svd_artifacts
        if svd is None:
            return False
        async with self._session_factory() as session:
            self._catalog = await load_policy_catalog(session, svd.film_index)
        self._policy_engine = CategorizedPolicyEngine(
            self._catalog,
            svd.item_vectors,
            svd.id_to_position,
            config=self._config,
        )
        popularity = self._candidate_service.popularity_artifact
        if popularity is None:
            self._catalog = None
            self._policy_engine = None
            return False
        self._popularity_rank_by_film = {
            entry.film_id: entry.rank for entry in popularity.films
        }
        return True

    async def load_resources(self) -> bool:
        """Load immutable artifacts and one bounded policy catalog snapshot."""
        if not self.load_candidate_artifacts():
            return False
        return await self.load_policy_catalog()

    def unload_resources(self) -> None:
        self._candidate_service.unload_artifacts()
        self._catalog = None
        self._policy_engine = None
        self._popularity_rank_by_film = None

    async def recommend(
        self,
        user_id: int,
        *,
        profiler: CategoryRequestProfile | None = None,
    ) -> CategorizedRecommendationResult:
        """Build an internal categorized result from one user-history read."""
        with request_stage(profiler, "total_request"):
            return await self._recommend(user_id, profiler)

    async def _recommend(
        self, user_id: int, profiler: CategoryRequestProfile | None
    ) -> CategorizedRecommendationResult:
        catalog = self._catalog
        svd = self._candidate_service.svd_artifacts
        popularity = self._candidate_service.popularity_artifact
        policy_engine = self._policy_engine
        popularity_rank_by_film = self._popularity_rank_by_film
        if (
            catalog is None
            or svd is None
            or popularity is None
            or policy_engine is None
            or popularity_rank_by_film is None
        ):
            raise CategoryPolicyResourcesUnavailableError
        with request_stage(profiler, "history_database_read"):
            async with self._session_factory() as session:
                history = await InteractionRepository(
                    session
                ).get_recommendation_history(user_id)
        candidates = self._candidate_service.generate_from_history(
            user_id, history, profiler=profiler
        )
        with request_stage(profiler, "rrf_construction_ranking"):
            ranked = rank_candidates_by_rrf(
                candidates.candidates,
                popularity_rank_by_film,
                popularity.film_count,
                config=self._config,
            )
        with request_stage(profiler, "user_category_profile"):
            profile = build_user_category_profile(
                user_id,
                history,
                catalog,
                config=self._config,
                profiler=profiler,
            )
        policy = policy_engine.categorize(ranked, profile, profiler=profiler)
        selected_ids = tuple(
            dict.fromkeys(
                film_id
                for category in policy.allocated_categories
                for film_id in category.film_ids
            )
        )
        with request_stage(profiler, "display_film_database_fetch"):
            display_by_id = {
                film_id: catalog.films[film_id]
                for film_id in selected_ids
                if film_id in catalog.films
            }
        with request_stage(profiler, "domain_result_materialization"):
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
            result = CategorizedRecommendationResult(
                user_id=user_id,
                categories=tuple(categories),
                diagnostics=policy.diagnostics,
            )
        if profiler is not None:
            profiler.count("selected_film_count", len(selected_ids))
            profiler.count("materialized_category_count", len(categories))
        return result
