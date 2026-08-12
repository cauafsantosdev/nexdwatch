"""Context-only bounded policy-alternative diagnostics for category V1.1."""

import asyncio
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from statistics import fmean, median
from typing import Any

from app.core.config import Settings, get_settings
from app.ml.catalog import load_catalog_slug_mapping
from app.ml.historical_interactions import (
    build_interaction_splits,
    load_historical_interactions,
)
from app.policy.allocation import allocate_categories
from app.policy.config import (
    DEFAULT_POLICY_CONFIG,
    V1_POLICY_CONFIG,
    CategoryPolicyConfig,
)
from app.policy.profile import (
    build_user_category_profile,
    preference_evidence_tier,
    qualifying_preferences,
)
from app.policy.proposals import (
    _because_you_liked,
    _classic_cinema,
    _hidden_gems,
    _outside_usual,
    _top_picks,
    _world_cinema,
)
from app.policy.ranking import rank_candidates_by_rrf
from experiments.category_policy.evaluate import (
    _atomic_json,
    _context_history,
    _load_catalog,
)
from experiments.ranker.config import (
    NEGATIVE_RATING_THRESHOLD,
    POSITIVE_RATING_THRESHOLD,
)
from experiments.ranker.protocol import build_user_folds
from experiments.ranker.rrf_calibration import (
    _build_artifacts,
    _evaluation_examples,
    _full_candidates,
)

REFINEMENT_PROTOCOL = "category_policy_v1_1_context_diagnostics"


def refinement_variants() -> dict[str, dict[str, CategoryPolicyConfig]]:
    """Return the intentionally small, label-independent policy comparison grid."""
    v1 = V1_POLICY_CONFIG
    return {
        "anchor": {
            "v1_positive": v1,
            "top_50": replace(v1, anchor_neighborhood_limit=50),
            "top_100": replace(v1, anchor_neighborhood_limit=100),
            "top_200": replace(v1, anchor_neighborhood_limit=200),
            "top_5_percent": replace(v1, anchor_neighborhood_fraction=0.05),
            "top_10_percent": replace(v1, anchor_neighborhood_fraction=0.10),
        },
        "classic_cinema": {
            "v1_unconstrained": v1,
            "head_cap_12": replace(v1, classic_head_cap=12),
            "head_cap_10": replace(v1, classic_head_cap=10),
        },
        "world_cinema": {
            "v1_unconstrained": v1,
            "head_cap_12": replace(v1, world_head_cap=12),
            "head_cap_10": replace(v1, world_head_cap=10),
        },
        "outside_usual": {
            "v1_non_head": v1,
            "deeper_mid_tail": replace(
                v1,
                outside_svd_rank=1000,
                outside_rrf_rank=1500,
            ),
            "exclude_hidden_depth_600": replace(
                v1,
                outside_svd_rank=600,
                outside_rrf_rank=1100,
                outside_exclude_hidden_neighborhood=True,
                outside_require_primary_viability=True,
            ),
            "exclude_hidden_depth_750": replace(
                v1,
                outside_svd_rank=750,
                outside_rrf_rank=1250,
                outside_exclude_hidden_neighborhood=True,
                outside_require_primary_viability=True,
            ),
            "exclude_hidden_deeper": replace(
                v1,
                outside_svd_rank=1000,
                outside_rrf_rank=1500,
                outside_exclude_hidden_neighborhood=True,
                outside_require_primary_viability=True,
            ),
            "lower_head_bridge": replace(
                v1,
                outside_exclude_head=False,
                outside_lower_head_cap=4,
                outside_lower_head_svd_rank=100,
            ),
            "exclude_hidden_lower_head_bridge": replace(
                v1,
                outside_svd_rank=750,
                outside_rrf_rank=1250,
                outside_exclude_head=False,
                outside_exclude_hidden_neighborhood=True,
                outside_lower_head_cap=4,
                outside_lower_head_svd_rank=100,
                outside_require_primary_viability=True,
            ),
        },
    }


def run_refinement_analysis(
    *,
    csv_path: str | Path,
    output_path: str | Path,
    settings: Settings | None = None,
    seed: int = 42,
    fold: int = 0,
    sample_stride: int = 5,
) -> dict[str, Any]:
    """Compare category alternatives without consulting held-out labels."""
    if sample_stride <= 0:
        raise ValueError("sample stride must be positive")
    started = time.perf_counter()
    active = settings or get_settings()
    mapping = load_catalog_slug_mapping(active)
    data = load_historical_interactions(csv_path, mapping)
    catalog = asyncio.run(_load_catalog(data.film_ids))
    splits = build_interaction_splits(
        data,
        positive_rating_threshold=POSITIVE_RATING_THRESHOLD,
        negative_rating_threshold=NEGATIVE_RATING_THRESHOLD,
        seed=seed,
    )
    assignment = build_user_folds(splits, data.film_ids, seed=seed)
    split_by_user = {split.cohort_user_id: split for split in splits}
    training_users, _, test_users = assignment.partitions(fold)
    artifacts, _ = _build_artifacts(data, split_by_user, training_users, seed=seed)
    examples = _evaluation_examples(splits, data, test_users, "test", artifacts)
    id_to_position = {
        int(film_id): row for row, film_id in enumerate(artifacts.film_ids)
    }
    popularity_rank_by_film = {
        int(film_id): int(artifacts.popularity_global_ranks[row])
        for row, film_id in enumerate(artifacts.film_ids)
    }
    variants = refinement_variants()
    accumulators = {
        family: {
            name: _VariantAccumulator(family, len(data.film_ids)) for name in values
        }
        for family, values in variants.items()
    }
    director_audits = {
        "support_2_high_1": _DirectorAudit(),
        "support_3_high_1": _DirectorAudit(),
        "support_2_high_2": _DirectorAudit(),
    }
    sampled_users = 0
    for example_index, example in enumerate(examples):
        if example_index % sample_stride:
            continue
        history = _context_history(example, artifacts.film_ids)
        profile = build_user_category_profile(
            example.user_id, history, catalog, config=DEFAULT_POLICY_CONFIG
        )
        candidates = _full_candidates(example, artifacts)
        ranked = rank_candidates_by_rrf(
            candidates,
            popularity_rank_by_film,
            len(artifacts.film_ids),
            config=DEFAULT_POLICY_CONFIG,
        )
        _observe_director_audits(profile, director_audits)
        for family, values in variants.items():
            for name, config in values.items():
                proposals, diagnostics = _isolated_proposals(
                    family,
                    ranked,
                    profile,
                    catalog,
                    artifacts.item_vectors,
                    id_to_position,
                    config,
                )
                allocated, _ = allocate_categories(
                    proposals, profile, catalog, config=config
                )
                accumulators[family][name].observe(
                    proposals,
                    allocated,
                    ranked,
                    profile,
                    catalog,
                    diagnostics,
                )
        sampled_users += 1

    report = {
        "protocol": REFINEMENT_PROTOCOL,
        "selection_basis": (
            "context-only portfolio diagnostics; designated targets and labels unused"
        ),
        "seed": seed,
        "fold": fold,
        "sample_stride": sample_stride,
        "sampled_users": sampled_users,
        "variants": {
            family: {
                name: {
                    "config": asdict(config),
                    **accumulators[family][name].report(),
                }
                for name, config in values.items()
            }
            for family, values in variants.items()
        },
        "director_audit": {
            name: audit.report() for name, audit in director_audits.items()
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, destination)
    return report


def _isolated_proposals(
    family,
    ranked,
    profile,
    catalog,
    item_vectors,
    id_to_position,
    config,
):
    top = _top_picks(ranked, profile, catalog, config)
    if top is None:
        return (), {}
    if family == "anchor":
        proposal, diagnostics = _because_you_liked(
            ranked,
            profile,
            catalog,
            item_vectors,
            id_to_position,
            config,
        )
        return tuple(value for value in (top, proposal) if value is not None), {
            "anchor": diagnostics
        }
    if family == "classic_cinema":
        proposal = _classic_cinema(ranked, profile, catalog, config)
        return tuple(value for value in (top, proposal) if value is not None), {}
    if family == "world_cinema":
        proposal = _world_cinema(ranked, profile, catalog, config)
        return tuple(value for value in (top, proposal) if value is not None), {}
    hidden = _hidden_gems(ranked, profile, catalog, config)
    outside = _outside_usual(ranked, profile, catalog, config)
    return tuple(value for value in (top, hidden, outside) if value is not None), {}


class _VariantAccumulator:
    def __init__(self, family: str, catalog_size: int) -> None:
        self.family = family
        self.catalog_size = catalog_size
        self.users = 0
        self.activations = 0
        self.row_sizes: list[int] = []
        self.strata: Counter[str] = Counter()
        self.catalog_films: set[int] = set()
        self.anchor_neighbors: list[int] = []
        self.anchor_quality: list[float] = []
        self.anchor_overlap: list[float] = []
        self.hidden_raw_overlap: list[float] = []
        self.hidden_final_overlap: list[float] = []
        self.top_overlap: list[float] = []
        self.svd_ranks: list[int] = []
        self.world_supported = 0
        self.world_discovery = 0
        self.world_country_counts: list[int] = []
        self.world_max_country_share: list[float] = []

    def observe(
        self,
        proposals,
        allocated,
        ranked,
        profile,
        catalog,
        diagnostics,
    ) -> None:
        del profile
        self.users += 1
        key = "because_you_liked" if self.family == "anchor" else self.family
        proposal_by_key = {value.key: value for value in proposals}
        allocated_by_key = {value.proposal.key: value for value in allocated}
        proposal = proposal_by_key.get(key)
        category = allocated_by_key.get(key)
        ranked_by_id = {value.film_id: value for value in ranked}
        if self.family == "outside_usual":
            hidden = proposal_by_key.get("hidden_gems")
            if proposal is not None and hidden is not None:
                self.hidden_raw_overlap.append(
                    _jaccard(
                        set(proposal.ordered_candidate_ids[:20]),
                        set(hidden.ordered_candidate_ids[:20]),
                    )
                )
            allocated_hidden = allocated_by_key.get("hidden_gems")
            if category is not None and allocated_hidden is not None:
                self.hidden_final_overlap.append(
                    _jaccard(set(category.film_ids), set(allocated_hidden.film_ids))
                )
        if category is None:
            return
        self.activations += 1
        self.row_sizes.append(len(category.film_ids))
        self.catalog_films.update(category.film_ids)
        self.strata.update(
            ranked_by_id[film_id].popularity_stratum for film_id in category.film_ids
        )
        top_ids = {value.film_id for value in ranked[:20]}
        self.top_overlap.append(_jaccard(set(category.film_ids), top_ids))
        self.svd_ranks.extend(
            ranked_by_id[film_id].svd_rank
            for film_id in category.film_ids
            if ranked_by_id[film_id].svd_rank is not None
        )
        if self.family == "anchor":
            selected = diagnostics["anchor"].get("selected")
            if selected:
                self.anchor_neighbors.append(selected["usable_neighbor_count"])
                self.anchor_quality.append(selected["mean_top_20_similarity"])
                self.anchor_overlap.append(selected["top_picks_overlap"])
        if self.family == "world_cinema" and proposal is not None:
            for film_id in category.film_ids:
                if proposal.reasons[film_id].support_count is None:
                    self.world_discovery += 1
                else:
                    self.world_supported += 1
            countries = Counter(
                entity.id
                for film_id in category.film_ids
                for entity in catalog.films[film_id].countries
            )
            self.world_country_counts.append(len(countries))
            self.world_max_country_share.append(
                max(countries.values(), default=0) / len(category.film_ids)
            )

    def report(self) -> dict[str, Any]:
        total_items = sum(self.strata.values())
        world_total = self.world_supported + self.world_discovery
        return {
            "activation_rate": self.activations / self.users if self.users else 0.0,
            "row_size": _summary(self.row_sizes),
            "popularity_strata": {
                key: {
                    "count": self.strata[key],
                    "fraction": self.strata[key] / total_items if total_items else 0.0,
                }
                for key in ("HEAD", "MID", "TAIL")
            },
            "catalog_coverage": len(self.catalog_films) / self.catalog_size,
            "top_picks_overlap": _summary(self.top_overlap),
            "svd_rank": _summary(self.svd_ranks),
            "anchor_neighborhood_size": _summary(self.anchor_neighbors),
            "anchor_top_20_similarity": _summary(self.anchor_quality),
            "anchor_top_picks_overlap": _summary(self.anchor_overlap),
            "hidden_raw_overlap": _summary(self.hidden_raw_overlap),
            "hidden_final_overlap": _summary(self.hidden_final_overlap),
            "world_item_origin": {
                "preference_supported": self.world_supported,
                "discovery_derived": self.world_discovery,
                "preference_supported_fraction": (
                    self.world_supported / world_total if world_total else 0.0
                ),
            },
            "world_distinct_countries": _summary(self.world_country_counts),
            "world_max_country_share": _summary(self.world_max_country_share),
        }


class _DirectorAudit:
    def __init__(self) -> None:
        self.qualifying_counts: list[int] = []
        self.selected: list[Any] = []
        self.not_selected: list[Any] = []
        self.config: CategoryPolicyConfig | None = None

    def observe(self, profile, config: CategoryPolicyConfig) -> None:
        self.config = config
        values = qualifying_preferences(profile, "director", config=config)
        self.qualifying_counts.append(len(values))
        self.selected.extend(values[: config.director_pool_size])
        self.not_selected.extend(values[config.director_pool_size :])

    def report(self) -> dict[str, Any]:
        if self.config is None:
            return {}
        return {
            "qualifying_directors_per_user": _summary(self.qualifying_counts),
            "selected": _preference_distribution(self.selected, self.config),
            "qualifying_not_selected": _preference_distribution(
                self.not_selected, self.config
            ),
        }


def _observe_director_audits(profile, audits: dict[str, _DirectorAudit]) -> None:
    audits["support_2_high_1"].observe(
        profile,
        replace(
            DEFAULT_POLICY_CONFIG,
            director_minimum_support=2,
            director_minimum_high=1,
        ),
    )
    audits["support_3_high_1"].observe(
        profile,
        replace(
            DEFAULT_POLICY_CONFIG,
            director_minimum_support=3,
            director_minimum_high=1,
        ),
    )
    audits["support_2_high_2"].observe(
        profile,
        replace(
            DEFAULT_POLICY_CONFIG,
            director_minimum_support=2,
            director_minimum_high=2,
        ),
    )


def _preference_distribution(values, config) -> dict[str, Any]:
    tiers = Counter(preference_evidence_tier(value, config=config) for value in values)
    return {
        "records": len(values),
        "support_count": _summary([value.support_count for value in values]),
        "mean_rating": _summary([value.mean_rating for value in values]),
        "high_rating_count": _summary([value.high_rating_count for value in values]),
        "positive_fraction": _summary([value.positive_fraction for value in values]),
        "affinity": _summary([value.affinity for value in values]),
        "evidence_tiers": dict(sorted(tiers.items())),
    }


def _summary(values: list[int | float]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": min(numeric),
        "mean": fmean(numeric),
        "median": median(numeric),
        "max": max(numeric),
    }


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0
