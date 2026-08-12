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


class CategorizedPolicyEngine:
    """Build and allocate category proposals without transport or persistence."""

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
    ) -> CategoryPolicyResult:
        proposals = build_category_proposals(
            ranked_candidates,
            profile,
            self._catalog,
            self._item_vectors,
            self._id_to_position,
            config=self._config,
        )
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
        return CategoryPolicyResult(
            user_id=profile.user_id,
            ranked_candidates=ranked_candidates,
            proposals=proposals.proposals,
            allocated_categories=allocated,
            diagnostics=diagnostics,
        )
