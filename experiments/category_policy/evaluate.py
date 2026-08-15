"""Strict held-out and portfolio diagnostics for category policy V1.1."""

import asyncio
import json
import os
import resource
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any

import numpy as np

from app.core.config import Settings, get_settings
from app.core.database import SessionLocal
from app.ml.historical_interactions import (
    build_interaction_splits,
    load_historical_interactions,
)
from app.ml.svd_profiles import build_svd_profile
from app.policy.catalog import PolicyCatalog, PolicyFilm, load_policy_catalog
from app.policy.config import DEFAULT_POLICY_CONFIG, CategoryPolicyConfig
from app.policy.engine import CategorizedPolicyEngine
from app.policy.profile import (
    build_user_category_profile,
    preference_evidence_tier,
    qualifying_preferences,
)
from app.policy.proposals import CATEGORY_KEYS
from app.policy.ranking import rank_candidates_by_rrf
from app.repositories.interactions import RatedInteraction, RecommendationHistory
from experiments.catalog import load_catalog_slug_mapping
from experiments.ranker.config import (
    NEGATIVE_RATING_THRESHOLD,
    POSITIVE_RATING_THRESHOLD,
    RANKER_PROTOCOL,
    RANKER_SEEDS,
)
from experiments.ranker.metrics import target_rank_metrics
from experiments.ranker.protocol import build_user_folds
from experiments.ranker.rrf_calibration import (
    _build_artifacts,
    _evaluation_examples,
    _full_candidates,
)

CATEGORY_POLICY_PROTOCOL = f"{RANKER_PROTOCOL}_category_policy_v1_1"
SENSITIVITY_SAMPLE_STRIDE = 25


def run_category_policy_evaluation(
    *,
    csv_path: str | Path,
    output_path: str | Path,
    settings: Settings | None = None,
    seeds: tuple[int, ...] = RANKER_SEEDS,
    folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    config: CategoryPolicyConfig = DEFAULT_POLICY_CONFIG,
) -> dict[str, Any]:
    """Evaluate policy portfolios and leakage-safe semantic held-out subsets."""
    started = time.perf_counter()
    active = settings or get_settings()
    mapping = load_catalog_slug_mapping(active)
    data = load_historical_interactions(csv_path, mapping)
    catalog = asyncio.run(_load_catalog(data.film_ids))
    aggregate = _EvaluationAccumulator(len(data.film_ids), config)
    sensitivity = _SensitivityAccumulator()
    observation_index = 0
    fold_runtime: list[dict[str, Any]] = []

    for seed in seeds:
        splits = build_interaction_splits(
            data,
            positive_rating_threshold=POSITIVE_RATING_THRESHOLD,
            negative_rating_threshold=NEGATIVE_RATING_THRESHOLD,
            seed=seed,
        )
        assignment = build_user_folds(splits, data.film_ids, seed=seed)
        split_by_user = {split.cohort_user_id: split for split in splits}
        for fold in folds:
            fold_started = time.perf_counter()
            training_users, _, test_users = assignment.partitions(fold)
            artifacts, _ = _build_artifacts(
                data, split_by_user, training_users, seed=seed
            )
            examples = _evaluation_examples(splits, data, test_users, "test", artifacts)
            popularity_rank_by_film = {
                int(film_id): int(artifacts.popularity_global_ranks[row])
                for row, film_id in enumerate(artifacts.film_ids)
            }
            id_to_position = {
                int(film_id): row for row, film_id in enumerate(artifacts.film_ids)
            }
            engine = CategorizedPolicyEngine(
                catalog,
                artifacts.item_vectors,
                id_to_position,
                config=config,
            )
            policy_seconds = 0.0
            evaluated = 0
            for example in examples:
                if example.designated_target_id is None:
                    continue
                request_started = time.perf_counter()
                candidates = _full_candidates(example, artifacts)
                ranked = rank_candidates_by_rrf(
                    candidates,
                    popularity_rank_by_film,
                    len(artifacts.film_ids),
                    config=config,
                )
                history = _context_history(example, artifacts.film_ids)
                profile = build_user_category_profile(
                    example.user_id, history, catalog, config=config
                )
                result = engine.categorize(ranked, profile)
                policy_seconds += time.perf_counter() - request_started
                target_svd_score = _target_personalized_score(
                    example, artifacts.item_vectors, id_to_position
                )
                aggregate.observe(
                    result,
                    profile,
                    catalog,
                    target_film_id=example.designated_target_id,
                    target_stratum=example.target_stratum,
                    target_svd_score=target_svd_score,
                    item_vectors=artifacts.item_vectors,
                    id_to_position=id_to_position,
                    config=config,
                )
                if observation_index % SENSITIVITY_SAMPLE_STRIDE == 0:
                    sensitivity.observe(
                        ranked,
                        profile,
                        catalog,
                        artifacts.item_vectors,
                        id_to_position,
                        config,
                    )
                observation_index += 1
                evaluated += 1
            fold_runtime.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "evaluated_users": evaluated,
                    "policy_seconds": policy_seconds,
                    "total_seconds": time.perf_counter() - fold_started,
                }
            )

    report = {
        "protocol": CATEGORY_POLICY_PROTOCOL,
        "base_protocol": RANKER_PROTOCOL,
        "policy_config": asdict(config),
        "candidate_policy": {
            "weighted_svd_depth": 2000,
            "popularity_depth": 2000,
            "rrf_weights": [1, 1],
            "rrf_k": config.rrf_k,
            "watched_exclusion": True,
            "deduplication": "deterministic_no_refill",
        },
        **aggregate.report(),
        "sensitivity": sensitivity.report(),
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "folds": fold_runtime,
            "mean_policy_ms_per_user": (
                1000
                * sum(value["policy_seconds"] for value in fold_runtime)
                / aggregate.user_count
                if aggregate.user_count
                else 0.0
            ),
            "peak_process_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
        },
        "evaluation_limits": [
            "portfolio metrics are offline proxies, not user satisfaction",
            "no impressions, clicks, conversion, or retention data exist",
            "semantic recall denominators include only independently relevant targets",
            "outside-usual held-out recall is secondary to novelty diagnostics",
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, destination)
    return report


async def _load_catalog(film_ids: np.ndarray) -> PolicyCatalog:
    async with SessionLocal() as session:
        return await load_policy_catalog(
            session, tuple(int(film_id) for film_id in film_ids)
        )


def _context_history(example: Any, film_ids: np.ndarray) -> RecommendationHistory:
    """Build policy evidence exclusively from the pre-target user context."""
    context_ids = tuple(int(film_ids[row]) for row in example.context_item_rows)
    if example.designated_target_id in context_ids:
        raise RuntimeError("held-out target leaked into category policy evidence")
    return RecommendationHistory(
        context_ids,
        tuple(
            RatedInteraction(int(film_ids[row]), (int(bucket) + 1) / 2.0)
            for row, bucket in zip(
                example.context_item_rows,
                example.context_rating_buckets,
                strict=True,
            )
        ),
    )


def _target_personalized_score(
    example: Any,
    item_vectors: np.ndarray,
    id_to_position: dict[int, int],
) -> float | None:
    """Score a target from context only without injecting it into retrieval."""
    target_position = id_to_position.get(example.designated_target_id)
    if target_position is None:
        return None
    query = build_svd_profile(
        item_vectors,
        example.context_item_rows,
        example.context_rating_buckets,
        "svd_positive_weighted",
    )
    if query is None:
        return None
    score = float(item_vectors[target_position] @ query)
    return score if np.isfinite(score) else None


class _EvaluationAccumulator:
    def __init__(self, catalog_size: int, config: CategoryPolicyConfig) -> None:
        self.catalog_size = catalog_size
        self.config = config
        self.user_count = 0
        self.category_counts: list[int] = []
        self.unique_counts: list[int] = []
        self.duplicate_rates: list[float] = []
        self.category_rows: dict[str, list[tuple[int, ...]]] = defaultdict(list)
        self.category_users: Counter[str] = Counter()
        self.category_catalog: dict[str, set[int]] = defaultdict(set)
        self.strata: dict[str, Counter[str]] = defaultdict(Counter)
        self.sources: dict[str, Counter[str]] = defaultdict(Counter)
        self.concentrations: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.evidence_support: dict[str, list[int]] = defaultdict(list)
        self.pairwise_overlap: dict[str, list[float]] = defaultdict(list)
        self.category_order: dict[str, list[int]] = defaultdict(list)
        self.semantic: dict[str, dict[str, Any]] = {
            key: {"eligible": 0, "retrieved": 0, "recommended": 0, "ranks": []}
            for key in CATEGORY_KEYS
        }
        self.anchor: list[dict[str, Any]] = []
        self.director_pool: list[dict[str, Any]] = []
        self.director_selected: list[Any] = []
        self.director_not_selected: list[Any] = []
        self.outside_svd_ranks: list[int] = []
        self.outside_top_overlap: list[float] = []
        self.outside_familiar_match_rate: list[float] = []
        self.world_distinct_countries: list[int] = []
        self.world_max_country_share: list[float] = []
        self.world_preference_supported = 0
        self.world_discovery_derived = 0

    def observe(
        self,
        result,
        profile,
        catalog,
        *,
        target_film_id: int,
        target_stratum: str,
        target_svd_score: float | None,
        item_vectors,
        id_to_position,
        config,
    ) -> None:
        self.user_count += 1
        allocated_by_key = {
            value.proposal.key: value for value in result.allocated_categories
        }
        ranked_by_id = {value.film_id: value for value in result.ranked_candidates}
        all_ids = [
            film_id
            for category in result.allocated_categories
            for film_id in category.film_ids
        ]
        self.category_counts.append(len(result.allocated_categories))
        self.unique_counts.append(len(set(all_ids)))
        self.duplicate_rates.append(
            (len(all_ids) - len(set(all_ids))) / len(all_ids) if all_ids else 0.0
        )
        for position, category in enumerate(result.allocated_categories, start=1):
            key = category.proposal.key
            self.category_users[key] += 1
            self.category_order[key].append(position)
            self.category_rows[key].append(category.film_ids)
            self.category_catalog[key].update(category.film_ids)
            self.evidence_support[key].append(category.proposal.evidence_support)
            for film_id in category.film_ids:
                candidate = ranked_by_id[film_id]
                self.strata[key][candidate.popularity_stratum] += 1
                self.sources[key][candidate.source_membership] += 1
            self._observe_concentration(key, category.film_ids, catalog)
            if key == "outside_usual":
                self.outside_svd_ranks.extend(
                    ranked_by_id[film_id].svd_rank
                    for film_id in category.film_ids
                    if ranked_by_id[film_id].svd_rank is not None
                )
                top = {value.film_id for value in result.ranked_candidates[:20]}
                self.outside_top_overlap.append(_jaccard(set(category.film_ids), top))
                familiar = {
                    family: {
                        value.entity_id
                        for value in qualifying_preferences(
                            profile, family, config=config
                        )[: config.outside_familiar_entities_per_family]
                    }
                    for family in (
                        "director",
                        "genre",
                        "decade",
                        "country",
                        "language",
                    )
                }
                familiar_matches = sum(
                    any(
                        familiar[family].intersection(
                            value.id
                            for value in catalog.films[film_id].entities(family)
                        )
                        for family in familiar
                    )
                    for film_id in category.film_ids
                )
                self.outside_familiar_match_rate.append(
                    familiar_matches / len(category.film_ids)
                )
            if key == "world_cinema":
                for film_id in category.film_ids:
                    if category.proposal.reasons[film_id].support_count is None:
                        self.world_discovery_derived += 1
                    else:
                        self.world_preference_supported += 1
                country_counts = Counter(
                    entity.name
                    for film_id in category.film_ids
                    for entity in catalog.films[film_id].countries
                )
                self.world_distinct_countries.append(len(country_counts))
                self.world_max_country_share.append(
                    max(country_counts.values(), default=0) / len(category.film_ids)
                )
        for left, right in combinations(result.proposals, 2):
            pair = "|".join(sorted((left.key, right.key)))
            self.pairwise_overlap[pair].append(
                _jaccard(
                    set(left.ordered_candidate_ids[: left.maximum_size]),
                    set(right.ordered_candidate_ids[: right.maximum_size]),
                )
            )
        selected_anchor = result.diagnostics["anchor"].get("selected")
        if selected_anchor:
            self.anchor.append(selected_anchor)
        self.director_pool.append(result.diagnostics["director_pool"])
        qualifying_directors = qualifying_preferences(
            profile, "director", config=config
        )
        self.director_selected.extend(qualifying_directors[: config.director_pool_size])
        self.director_not_selected.extend(
            qualifying_directors[config.director_pool_size :]
        )

        candidate_ids = set(ranked_by_id)
        for key in CATEGORY_KEYS:
            if not _semantic_scope(
                key,
                target_film_id,
                target_stratum,
                target_svd_score,
                profile,
                result,
                catalog,
                item_vectors,
                id_to_position,
                config,
            ):
                continue
            values = self.semantic[key]
            values["eligible"] += 1
            if target_film_id in candidate_ids:
                values["retrieved"] += 1
            allocated = allocated_by_key.get(key)
            rank = (
                allocated.film_ids.index(target_film_id) + 1
                if allocated is not None and target_film_id in allocated.film_ids
                else None
            )
            values["ranks"].append(rank)
            if rank is not None:
                values["recommended"] += 1

    def _observe_concentration(
        self, key: str, film_ids: tuple[int, ...], catalog: PolicyCatalog
    ) -> None:
        if not film_ids:
            return
        for family in ("director", "genre", "country"):
            counts = Counter(
                entity.id
                for film_id in film_ids
                for entity in catalog.films[film_id].entities(family)
            )
            self.concentrations[key][family].append(
                max(counts.values(), default=0) / len(film_ids)
            )
        decades = Counter(catalog.films[film_id].decade for film_id in film_ids)
        self.concentrations[key]["decade"].append(
            max(decades.values(), default=0) / len(film_ids)
        )

    def report(self) -> dict[str, Any]:
        category_reports = {}
        for key in CATEGORY_KEYS:
            rows = self.category_rows[key]
            total_items = sum(len(value) for value in rows)
            category_reports[key] = {
                "activation_rate": self.category_users[key] / self.user_count,
                "activated_users": self.category_users[key],
                "row_size": _summary([len(value) for value in rows]),
                "mean_position": (
                    fmean(self.category_order[key])
                    if self.category_order[key]
                    else None
                ),
                "catalog_coverage": len(self.category_catalog[key]) / self.catalog_size,
                "unique_catalog_films": len(self.category_catalog[key]),
                "popularity_strata": _counter_share(self.strata[key], total_items),
                "source_membership": _counter_share(self.sources[key], total_items),
                "concentration": {
                    family: _summary(values)
                    for family, values in self.concentrations[key].items()
                },
                "evidence_support": _summary(self.evidence_support[key]),
            }
        semantic = {}
        for key, values in self.semantic.items():
            eligible = values["eligible"]
            metrics = target_rank_metrics(values["ranks"])
            semantic[key] = {
                "eligible_targets": eligible,
                "candidate_retrieved_targets": values["retrieved"],
                "candidate_recall": values["retrieved"] / eligible if eligible else 0.0,
                "category_recommended_targets": values["recommended"],
                "category_recall": values["recommended"] / eligible
                if eligible
                else 0.0,
                "zero_inclusive": metrics,
            }
        return {
            "evaluated_user_appearances": self.user_count,
            "portfolio": {
                "categories_per_user": _summary(self.category_counts),
                "unique_films_per_response": _summary(self.unique_counts),
                "duplicate_rate": _summary(self.duplicate_rates),
                "category_count_distribution": dict(Counter(self.category_counts)),
            },
            "categories": category_reports,
            "proposal_pairwise_jaccard": {
                key: _summary(values)
                for key, values in sorted(self.pairwise_overlap.items())
            },
            "semantic_held_out": semantic,
            "anchor_diagnostics": {
                "selected_user_appearances": len(self.anchor),
                "rating": _summary(
                    [float(value["anchor_rating"]) for value in self.anchor]
                ),
                "usable_neighbor_count": _summary(
                    [int(value["usable_neighbor_count"]) for value in self.anchor]
                ),
                "top_picks_overlap": _summary(
                    [float(value["top_picks_overlap"]) for value in self.anchor]
                ),
                "mean_top_20_similarity": _summary(
                    [float(value["mean_top_20_similarity"]) for value in self.anchor]
                ),
                "local_similarity_cutoff": _summary(
                    [float(value["local_similarity_cutoff"]) for value in self.anchor]
                ),
                "neighborhood_rules": dict(
                    sorted(
                        Counter(
                            str(value["neighborhood_rule"]) for value in self.anchor
                        ).items()
                    )
                ),
            },
            "director_pool_diagnostics": {
                "qualifying_count": _summary(
                    [int(value["qualifying_count"]) for value in self.director_pool]
                ),
                "selected_count": _summary(
                    [int(value["selected_count"]) for value in self.director_pool]
                ),
                "selected_evidence": _preference_distribution(
                    self.director_selected, self.config
                ),
                "qualifying_not_selected_evidence": _preference_distribution(
                    self.director_not_selected, self.config
                ),
            },
            "outside_usual_diagnostics": {
                "svd_rank": _summary(self.outside_svd_ranks),
                "top_picks_overlap": _summary(self.outside_top_overlap),
                "familiar_metadata_match_rate": _summary(
                    self.outside_familiar_match_rate
                ),
                "head_items": self.strata["outside_usual"]["HEAD"],
            },
            "world_cinema_geography": {
                "distinct_countries_per_row": _summary(self.world_distinct_countries),
                "maximum_country_share": _summary(self.world_max_country_share),
                "item_origin": {
                    "preference_supported": self.world_preference_supported,
                    "discovery_derived": self.world_discovery_derived,
                    "preference_supported_fraction": (
                        self.world_preference_supported
                        / (
                            self.world_preference_supported
                            + self.world_discovery_derived
                        )
                        if self.world_preference_supported
                        + self.world_discovery_derived
                        else 0.0
                    ),
                },
            },
        }


class _SensitivityAccumulator:
    def __init__(self) -> None:
        self.samples = 0
        self.category_counts: dict[str, list[int]] = defaultdict(list)
        self.activations: dict[str, Counter[str]] = defaultdict(Counter)

    def observe(
        self,
        ranked,
        profile,
        catalog,
        vectors,
        id_to_position,
        config,
    ) -> None:
        variants = _sensitivity_variants(config)
        self.samples += 1
        for name, variant in variants.items():
            result = CategorizedPolicyEngine(
                catalog, vectors, id_to_position, config=variant
            ).categorize(ranked, profile)
            self.category_counts[name].append(len(result.allocated_categories))
            self.activations[name].update(
                value.proposal.key for value in result.allocated_categories
            )

    def report(self) -> dict[str, Any]:
        return {
            "sample_stride": SENSITIVITY_SAMPLE_STRIDE,
            "sampled_user_appearances": self.samples,
            "variants": {
                name: {
                    "categories_per_user": _summary(values),
                    "activation_rate": {
                        key: self.activations[name][key] / self.samples
                        for key in CATEGORY_KEYS
                    },
                }
                for name, values in self.category_counts.items()
            },
        }


def _sensitivity_variants(
    config: CategoryPolicyConfig,
) -> dict[str, CategoryPolicyConfig]:
    minimum_fields = {
        "hidden_minimum",
        "brazilian_minimum",
        "anchor_minimum",
        "directors_minimum",
        "genre_minimum",
        "decade_minimum",
        "world_minimum",
        "outside_minimum",
        "classic_minimum",
    }
    values = asdict(config)
    lower = {
        key: max(1, value - 2) if key in minimum_fields else value
        for key, value in values.items()
    }
    higher = {
        key: value + 2 if key in minimum_fields else value
        for key, value in values.items()
    }
    return {
        "default": config,
        "hidden_tighter": replace(
            config, hidden_mid_svd_rank=400, hidden_tail_svd_rank=200
        ),
        "hidden_looser": replace(
            config, hidden_mid_svd_rank=600, hidden_tail_svd_rank=300
        ),
        "classic_1959": replace(config, classic_year_boundary=1959),
        "classic_1979": replace(config, classic_year_boundary=1979),
        "minimum_sizes_minus_2": CategoryPolicyConfig(**lower),
        "minimum_sizes_plus_2": CategoryPolicyConfig(**higher),
        "overlap_0_60": replace(config, category_overlap_threshold=0.60),
        "overlap_0_80": replace(config, category_overlap_threshold=0.80),
        "broad_support_4": replace(config, broad_minimum_support=4),
        "broad_support_6": replace(config, broad_minimum_support=6),
        "director_pool_10": replace(config, director_pool_size=10),
        "director_pool_20": replace(config, director_pool_size=20),
        "outside_unrestricted_head": replace(
            config,
            outside_exclude_head=False,
            outside_lower_head_cap=0,
        ),
    }


def _semantic_scope(
    key: str,
    target_id: int,
    target_stratum: str,
    target_svd_score: float | None,
    profile,
    result,
    catalog: PolicyCatalog,
    vectors,
    id_to_position,
    config,
) -> bool:
    film = catalog.film(target_id)
    if film is None:
        return False
    if key == "top_picks":
        return True
    if key == "hidden_gems":
        return (
            target_stratum in {"MID", "TAIL"}
            and profile.indexed_positive_count
            >= config.hidden_minimum_indexed_positives
            and target_svd_score is not None
            and target_svd_score > 0
        )
    if key == "brazilian_cinema":
        return (
            any(
                value.name.casefold() in config.brazilian_country_names
                for value in film.countries
            )
            and target_svd_score is not None
            and target_svd_score > 0
        )
    proposals = {value.key: value for value in result.proposals}
    if key == "because_you_liked":
        proposal = proposals.get(key)
        if proposal is None:
            return False
        anchor_id = proposal.policy_metadata["anchor_film_id"]
        similarity_cutoff = proposal.policy_metadata.get(
            "local_similarity_cutoff", config.anchor_similarity_threshold
        )
        local_rule = proposal.policy_metadata.get("neighborhood_rule")
        target_similarity = (
            float(
                vectors[id_to_position[anchor_id]] @ vectors[id_to_position[target_id]]
            )
            if anchor_id in id_to_position and target_id in id_to_position
            else None
        )
        return target_similarity is not None and (
            target_similarity >= similarity_cutoff
            if local_rule != "legacy_positive_similarity"
            else target_similarity > config.anchor_similarity_threshold
        )
    if key == "directors_you_love":
        qualified = qualifying_preferences(profile, "director", config=config)[
            : config.director_pool_size
        ]
        return _target_matches(film, qualified)
    if key == "favorite_genre":
        values = qualifying_preferences(profile, "genre", config=config)
        return bool(values) and _target_matches(film, values[:1])
    if key == "favorite_decade":
        values = qualifying_preferences(profile, "decade", config=config)
        return bool(values) and _target_matches(film, values[:1])
    if key == "world_cinema":
        return _is_world(film, config)
    if key == "outside_usual":
        familiar = {
            family: {
                value.entity_id
                for value in qualifying_preferences(profile, family, config=config)[
                    : config.outside_familiar_entities_per_family
                ]
            }
            for family in ("director", "genre", "decade", "country", "language")
        }
        return (
            sum(bool(values) for values in familiar.values())
            >= config.outside_minimum_familiar_families
            and target_svd_score is not None
            and target_svd_score > 0
            and (not config.outside_exclude_head or target_stratum != "HEAD")
            and not any(
                familiar[family].intersection(
                    value.id for value in film.entities(family)
                )
                for family in familiar
            )
        )
    if key == "classic_cinema":
        return (
            film.year is not None
            and film.year <= config.classic_year_boundary
            and target_svd_score is not None
            and target_svd_score > 0
        )
    return False


def _target_matches(film: PolicyFilm, preferences: tuple) -> bool:
    return any(
        entity.id == preference.entity_id
        for preference in preferences
        for entity in film.entities(preference.family)
    )


def _is_world(film: PolicyFilm, config: CategoryPolicyConfig) -> bool:
    return any(
        value.name.casefold() not in config.english_core_country_names
        and value.name.casefold() not in config.metadata_none_names
        for value in film.countries
    ) and any(
        value.name.casefold() not in config.english_language_names
        and value.name.casefold() not in config.metadata_none_names
        for value in film.languages
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


def _summary(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
            "std": None,
        }
    numeric = [float(value) for value in values]
    return {
        "count": len(values),
        "min": min(numeric),
        "mean": fmean(numeric),
        "median": median(numeric),
        "max": max(numeric),
        "std": pstdev(numeric) if len(numeric) > 1 else 0.0,
    }


def _counter_share(counter: Counter[str], total: int) -> dict[str, Any]:
    return {
        key: {"count": value, "fraction": value / total if total else 0.0}
        for key, value in sorted(counter.items())
    }


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _atomic_json(payload: object, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, destination)
        destination.chmod(0o644)
    finally:
        temporary.unlink(missing_ok=True)
