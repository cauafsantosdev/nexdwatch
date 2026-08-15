"""Pure internal categorized recommendation policy engine."""

from numpy.typing import NDArray

from app.domain.categorized_recommendations import (
    CategoryPolicyResult,
    RankedCandidate,
    UserCategoryProfile,
)
from app.policy.allocation import allocate_categories
from app.policy.catalog import PolicyCatalog
from app.policy.config import DEFAULT_POLICY_CONFIG, CategoryPolicyConfig
from app.policy.proposals import build_category_proposals
from app.policy.request_metrics import (
    CategoryRequestProfile,
    request_stage,
)


class CategorizedPolicyEngine:
    """Build and allocate category proposals without transport or persistence.

    The engine is lifespan-scoped and reads immutable catalog/vector resources. Its
    optional request metrics are observational and never affect policy output.
    """

    def __init__(
        self,
        catalog: PolicyCatalog,
        item_vectors: NDArray,
        id_to_position: dict[int, int],
        *,
        config: CategoryPolicyConfig = DEFAULT_POLICY_CONFIG,
    ) -> None:
        self._catalog = catalog
        self._item_vectors = item_vectors
        self._id_to_position = id_to_position
        self._config = config

    def categorize(
        self,
        ranked_candidates: tuple[RankedCandidate, ...],
        profile: UserCategoryProfile,
        *,
        profiler: CategoryRequestProfile | None = None,
    ) -> CategoryPolicyResult:
        """Construct eligible proposals and allocate the frozen category portfolio.

        Args:
            ranked_candidates: RRF-ordered unwatched inventory shared by all shelves.
            profile: Request-scoped preference evidence and history-depth band.
            profiler: Optional observational metrics collector.

        Returns:
            CategoryPolicyResult: Ranked input, all viable proposals, selected
                categories, and deterministic policy diagnostics.
        """
        # Proposal construction evaluates semantic eligibility once; allocation then
        # applies portfolio-level overlap, diversity, and reuse constraints.
        proposals = build_category_proposals(
            ranked_candidates,
            profile,
            self._catalog,
            self._item_vectors,
            self._id_to_position,
            config=self._config,
            profiler=profiler,
        )
        with request_stage(profiler, "category_allocation"):
            allocated, allocation_diagnostics = allocate_categories(
                proposals.proposals,
                profile,
                self._catalog,
                config=self._config,
            )
        diagnostics = {
            **proposals.diagnostics,
            "allocation": allocation_diagnostics,
            "history_depth_band": profile.history_depth_band,
        }
        result = CategoryPolicyResult(
            user_id=profile.user_id,
            ranked_candidates=ranked_candidates,
            proposals=proposals.proposals,
            allocated_categories=allocated,
            diagnostics=diagnostics,
        )
        if profiler is not None:
            profiler.count("category_count", len(allocated))
        return result
