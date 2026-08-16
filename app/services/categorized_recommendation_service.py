"""Orchestrates the production categorized feed from immutable model resources."""

import logging
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
from app.policy.request_metrics import (
    CategoryRequestProfile,
    request_stage,
)
from app.repositories.interactions import InteractionRepository
from app.services.candidate_generation_service import CandidateGenerationService

logger = logging.getLogger(__name__)


class CategoryPolicyResourcesUnavailableError(RuntimeError):
    """Raised when internal candidate or catalog resources are not loaded."""


class RecommendationUserNotFoundError(LookupError):
    """Raised when a categorized feed is requested for an unknown user ID."""


class CategorizedRecommendationService:
    """Compose finalized retrieval, RRF, policy, and display materialization.

    One service is owned by each API lifespan. Candidate artifacts and the policy
    catalog remain immutable after loading; requests read only current user history.
    """

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
        """Return whether every artifact, catalog, and policy dependency is ready."""
        return (
            self._candidate_service.is_loaded
            and self._catalog is not None
            and self._policy_engine is not None
            and self._popularity_rank_by_film is not None
        )

    def load_candidate_artifacts(self) -> bool:
        """Load frozen candidate artifacts separately for resource measurement.

        Returns:
            bool: Whether compatible SVD, FAISS, and popularity resources loaded.
        """
        return self._candidate_service.load_artifacts()

    def configure_artifacts(
        self, artifact_root: str | Path, popularity_path: str | Path
    ) -> None:
        """Bind candidate resources to the same resolved startup bundle."""
        if self.is_loaded:
            raise RuntimeError("cannot reconfigure loaded categorized resources")
        self._candidate_service = CandidateGenerationService(
            self._session_factory,
            artifact_root,
            popularity_path=popularity_path,
        )

    async def load_policy_catalog(self) -> bool:
        """Load policy metadata for exactly the model's film identity universe.

        The catalog, policy engine, and popularity-rank lookup are published only
        after every dependency is available, preventing partially loaded serving
        state.

        Returns:
            bool: ``True`` when all downstream policy resources are ready together.
        """
        svd = self._candidate_service.svd_artifacts
        if svd is None:
            return False
        # Bound the database snapshot to model identities so policy eligibility can
        # never materialize films that retrieval cannot produce.
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
        """Load immutable candidate artifacts and their bounded policy snapshot.

        Returns:
            bool: ``True`` only when the complete categorized serving graph is ready.
        """
        if not self.load_candidate_artifacts():
            return False
        loaded = await self.load_policy_catalog()
        if loaded:
            logger.info("Categorized recommendation resources loaded")
        return loaded

    def unload_resources(self) -> None:
        """Release lifespan-owned immutable resources during graceful shutdown."""
        self._candidate_service.unload_artifacts()
        self._catalog = None
        self._policy_engine = None
        self._popularity_rank_by_film = None
        logger.info("Categorized recommendation resources unloaded")

    async def recommend(
        self,
        user_id: int,
        *,
        profiler: CategoryRequestProfile | None = None,
    ) -> CategorizedRecommendationResult:
        """Build one categorized feed while recording optional request metrics.

        Args:
            user_id: Existing persisted user to personalize.
            profiler: Optional request-local timing and operation-count collector.

        Returns:
            CategorizedRecommendationResult: Ordered categories and policy diagnostics.

        Raises:
            CategoryPolicyResourcesUnavailableError: If lifespan resources are not
                fully loaded.
            RecommendationUserNotFoundError: If ``user_id`` does not exist.
        """
        with request_stage(profiler, "total_request"):
            return await self._recommend(user_id, profiler)

    async def _recommend(
        self, user_id: int, profiler: CategoryRequestProfile | None
    ) -> CategorizedRecommendationResult:
        """Execute retrieval, fusion, policy allocation, and materialization.

        User history is read exactly once and passed through all stages. Candidate
        generation excludes watched films, RRF freezes source consensus, and policy
        allocation selects unique film IDs before display values are materialized
        from the lifespan-owned catalog.
        """
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
            # Use one history snapshot for profile evidence, watched exclusion, and
            # source generation; a missing user is distinct from empty history.
            async with self._session_factory() as session:
                history = await InteractionRepository(
                    session
                ).get_existing_user_recommendation_history(user_id)
        if history is None:
            raise RecommendationUserNotFoundError(user_id)
        # Candidate generation and RRF establish the single deterministic ordering
        # from which every category proposal is allowed to select.
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
        # Policy allocation may reuse proposal eligibility internally, but selected
        # display IDs are deduplicated before metadata materialization.
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
            # Reapply each allocated category's order and minimum after catalog
            # projection so incomplete metadata cannot leak undersized shelves.
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
                            tmdb_id=film.tmdb_id,
                            slug=film.slug,
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
                        preference_context=allocated.proposal.preference_context,
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


def build_categorized_recommendation_service() -> CategorizedRecommendationService:
    """Construct one categorized service for ownership by an application worker."""
    return CategorizedRecommendationService()
