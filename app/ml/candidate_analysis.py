"""Offline candidate-generation analysis over leakage-free shared holdouts."""

import json
import logging
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Literal

import faiss
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from app.core.config import Settings, get_settings
from app.ml.candidate_policy import FINAL_CANDIDATE_NOMINAL_BUDGET
from app.ml.catalog import load_catalog_slug_mapping
from app.ml.evaluation import (
    build_evaluation_svd_training_matrix,
    popularity_order_rows,
    training_positive_counts,
)
from app.ml.faiss_index import create_faiss_index
from app.ml.historical_interactions import (
    PreparedInteractions,
    UserSplit,
    build_interaction_splits,
    load_historical_interactions,
)
from app.ml.svd_profiles import build_svd_profile

CANDIDATE_ANALYSIS_PROTOCOL = "exact_holdout_v2"
CANDIDATE_CUTOFFS = (
    10,
    50,
    100,
    250,
    500,
    750,
    1000,
    1500,
    2000,
    2500,
    3000,
    4000,
    5000,
)
CANDIDATE_BUDGETS = (500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000)
POSITIVE_RATING_THRESHOLD = 3.5
NEGATIVE_RATING_THRESHOLD = 2.5
BUDGET_MARGINAL_RECALL_THRESHOLD = 0.02
NEAR_WINNER_RECALL_DELTA = 0.005
CANDIDATE_SVD_PROFILE = "svd_positive_weighted"
PRAGMATIC_MAX_CANDIDATE_BUDGET = FINAL_CANDIDATE_NOMINAL_BUDGET
PopularityStratum = Literal["HEAD", "MID", "TAIL"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MetricView:
    """Retrieval metrics for one explicit evaluation denominator."""

    recall_at: Mapping[int, float]
    ndcg_at_10: float
    mrr_at_10: float
    evaluated_users: int


@dataclass(frozen=True, slots=True)
class StrategyMetrics:
    """Conditional and global metrics plus query coverage."""

    conditional: MetricView
    global_view: MetricView
    eligible_users: int
    total_users: int
    coverage_percentage: float


@dataclass(frozen=True, slots=True)
class HybridCandidateSet:
    """Deterministically ordered union with source membership and source ranks."""

    ordered_ids: tuple[int, ...]
    sources_by_film: Mapping[int, tuple[str, ...]]
    source_ranks_by_film: Mapping[int, Mapping[str, int]]


@dataclass(slots=True)
class StrategyEvaluation:
    """Internal per-user candidates and reportable diagnostics."""

    name: str
    candidates_by_user: dict[int, NDArray[np.int64]]
    ranks_by_user: dict[int, int | None]
    eligible_user_ids: set[int]
    metrics: StrategyMetrics
    catalog_coverage: dict[int, dict[str, float | int]]
    retrieved_popularity: dict[int, dict[str, float | None]]
    stratified_metrics: dict[str, dict[str, Any]]

    def report(self) -> dict[str, Any]:
        return {
            "metrics": strategy_metrics_report(self.metrics),
            "catalog_coverage": self.catalog_coverage,
            "retrieved_item_popularity_percentile": self.retrieved_popularity,
            "stratified_metrics": self.stratified_metrics,
        }


@dataclass(slots=True)
class SeedAnalysis:
    """Internal state retained for aggregation and source-union analysis."""

    seed: int
    splits: tuple[UserSplit, ...]
    film_ids: NDArray[np.int64]
    targets_by_user: dict[int, int]
    target_strata_by_user: dict[int, str]
    known_by_user: dict[int, set[int]]
    source_overlap_ranks_by_user: dict[int, tuple[tuple[int, int], ...]]
    strata_by_row: NDArray[np.str_]
    popularity_percentiles: NDArray[np.float64]
    strategies: dict[str, StrategyEvaluation]
    runtime_seconds: float


def metric_view_from_ranks(
    ranks: Sequence[int | None],
    *,
    catalog_size: int,
    cutoffs: Sequence[int] = CANDIDATE_CUTOFFS,
) -> MetricView:
    """Compute generic cutoff metrics while respecting small candidate catalogs."""
    if catalog_size <= 0:
        raise ValueError("catalog_size must be positive")
    if any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("metric cutoffs must be positive")
    recalls = {
        int(cutoff): (
            sum(
                rank is not None and rank <= min(int(cutoff), catalog_size)
                for rank in ranks
            )
            / len(ranks)
            if ranks
            else 0.0
        )
        for cutoff in cutoffs
    }
    return MetricView(
        recall_at=recalls,
        ndcg_at_10=(
            sum(
                1.0 / np.log2(rank + 1)
                for rank in ranks
                if rank is not None and rank <= min(10, catalog_size)
            )
            / len(ranks)
            if ranks
            else 0.0
        ),
        mrr_at_10=(
            sum(
                1.0 / rank
                for rank in ranks
                if rank is not None and rank <= min(10, catalog_size)
            )
            / len(ranks)
            if ranks
            else 0.0
        ),
        evaluated_users=len(ranks),
    )


def strategy_metrics_from_ranks(
    ranks_by_user: Mapping[int, int | None],
    eligible_user_ids: set[int],
    *,
    catalog_size: int,
    cutoffs: Sequence[int] = CANDIDATE_CUTOFFS,
) -> StrategyMetrics:
    """Report both eligible-only and all-user-as-denominator views."""
    total_user_ids = tuple(ranks_by_user)
    conditional_ranks = [
        ranks_by_user[user_id]
        for user_id in total_user_ids
        if user_id in eligible_user_ids
    ]
    global_ranks = [ranks_by_user[user_id] for user_id in total_user_ids]
    total = len(total_user_ids)
    eligible = len(eligible_user_ids)
    return StrategyMetrics(
        conditional=metric_view_from_ranks(
            conditional_ranks, catalog_size=catalog_size, cutoffs=cutoffs
        ),
        global_view=metric_view_from_ranks(
            global_ranks, catalog_size=catalog_size, cutoffs=cutoffs
        ),
        eligible_users=eligible,
        total_users=total,
        coverage_percentage=(100.0 * eligible / total if total else 0.0),
    )


def assign_popularity_strata(
    counts: NDArray[np.int64], film_ids: NDArray[np.int64]
) -> tuple[NDArray[np.str_], NDArray[np.float64]]:
    """Assign exact rank buckets using count-descending, film-ID-ascending order.

    Equal counts are deterministically ordered by film ID. HEAD is the first
    ceil(10%) rows, MID extends through ceil(50%), and the remainder is TAIL.
    """
    order = popularity_order_rows(counts, film_ids)
    item_count = len(film_ids)
    head_stop = int(np.ceil(item_count * 0.10))
    mid_stop = int(np.ceil(item_count * 0.50))
    strata = np.empty(item_count, dtype="<U4")
    strata[order[:head_stop]] = "HEAD"
    strata[order[head_stop:mid_stop]] = "MID"
    strata[order[mid_stop:]] = "TAIL"
    percentiles = np.empty(item_count, dtype=np.float64)
    if item_count == 1:
        percentiles[order] = 1.0
    else:
        percentiles[order] = np.linspace(1.0, 0.0, item_count)
    return strata, percentiles


def catalog_coverage(
    candidates_by_user: Mapping[int, Sequence[int]],
    *,
    catalog_size: int,
    cutoffs: Sequence[int] = CANDIDATE_CUTOFFS,
) -> dict[int, dict[str, float | int]]:
    """Count unique films retrieved across users at each depth."""
    result: dict[int, dict[str, float | int]] = {}
    for cutoff in cutoffs:
        unique = {
            int(film_id)
            for candidates in candidates_by_user.values()
            for film_id in candidates[:cutoff]
        }
        result[int(cutoff)] = {
            "unique_films": len(unique),
            "catalog_percentage": 100.0 * len(unique) / catalog_size,
        }
    return result


def mean_jaccard(
    first: Mapping[int, Sequence[int]],
    second: Mapping[int, Sequence[int]],
    user_ids: Sequence[int],
    *,
    cutoff: int,
) -> float:
    """Return mean per-user Jaccard similarity at one candidate depth."""
    values: list[float] = []
    for user_id in user_ids:
        left = set(first.get(user_id, ())[:cutoff])
        right = set(second.get(user_id, ())[:cutoff])
        union = left | right
        values.append(len(left & right) / len(union) if union else 1.0)
    return fmean(values) if values else 0.0


def attribute_unique_target_hits(
    targets_by_user: Mapping[int, int],
    source_candidates: Mapping[str, Mapping[int, Sequence[int]]],
    *,
    cutoff: int,
) -> dict[str, dict[str, float | int]]:
    """Attribute every target to its exact source-hit combination."""
    ordered_sources = tuple(source_candidates)
    if ordered_sources != ("popularity", "svd", "ncf"):
        raise ValueError("target attribution requires popularity, svd, and ncf")
    labels = {
        (True, False, False): "popularity_only",
        (False, True, False): "svd_only",
        (False, False, True): "ncf_only",
        (True, True, False): "popularity_and_svd",
        (True, False, True): "popularity_and_ncf",
        (False, True, True): "svd_and_ncf",
        (True, True, True): "all_three",
        (False, False, False): "none",
    }
    counts = {label: 0 for label in labels.values()}
    for user_id, target in targets_by_user.items():
        hit_tuple = tuple(
            target in source_candidates[source].get(user_id, ())[:cutoff]
            for source in ordered_sources
        )
        counts[labels[hit_tuple]] += 1
    total = len(targets_by_user)
    return {
        label: {
            "count": count,
            "percentage": 100.0 * count / total if total else 0.0,
        }
        for label, count in counts.items()
    }


def build_budgeted_hybrid(
    source_candidates: Mapping[str, Sequence[int]],
    allocations: Mapping[str, int],
    *,
    known_film_ids: set[int],
    max_budget: int = 500,
) -> HybridCandidateSet:
    """Take source depths, exclude known IDs, and form a deterministic hard union."""
    if max_budget <= 0 or any(depth < 0 for depth in allocations.values()):
        raise ValueError("hybrid depths and budget must be non-negative")
    ordered: list[int] = []
    admitted: set[int] = set()
    memberships: dict[int, list[str]] = {}
    source_ranks: dict[int, dict[str, int]] = {}
    for source, depth in allocations.items():
        candidates = source_candidates.get(source, ())[:depth]
        for rank, raw_film_id in enumerate(candidates, start=1):
            film_id = int(raw_film_id)
            if film_id in known_film_ids:
                continue
            source_ranks.setdefault(film_id, {})[source] = rank
            memberships.setdefault(film_id, []).append(source)
            if film_id not in admitted and len(ordered) < max_budget:
                ordered.append(film_id)
                admitted.add(film_id)
    allowed = set(ordered)
    return HybridCandidateSet(
        ordered_ids=tuple(ordered),
        sources_by_film={film_id: tuple(memberships[film_id]) for film_id in ordered},
        source_ranks_by_film={
            film_id: source_ranks[film_id]
            for film_id in source_ranks
            if film_id in allowed
        },
    )


def run_candidate_analysis(
    seeds: Sequence[int],
    *,
    csv_path: str | Path | None = None,
    settings: Settings | None = None,
    slug_to_film_id: Mapping[str, int] | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the complete non-destructive candidate-generation analysis."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("analysis seeds must be a non-empty unique sequence")
    active_settings = settings or get_settings()
    source = Path(csv_path or (active_settings.ARTIFACT_ROOT / "users_data.csv"))
    mapping = (
        dict(slug_to_film_id)
        if slug_to_film_id is not None
        else load_catalog_slug_mapping(active_settings)
    )
    started = time.perf_counter()
    data = load_historical_interactions(source, mapping)
    seed_analyses: list[SeedAnalysis] = []
    for seed in seeds:
        logger.info("Candidate analysis starting seed=%d", seed)
        seed_analysis = _analyze_seed(data, seed=int(seed), settings=active_settings)
        seed_analysis.splits = ()
        seed_analyses.append(seed_analysis)
        logger.info(
            "Candidate analysis finished seed=%d seconds=%.3f",
            seed,
            seed_analysis.runtime_seconds,
        )
    aggregates = _aggregate_strategies(seed_analyses)
    best_svd = CANDIDATE_SVD_PROFILE
    hybrid_report = _analyze_svd_popularity_hybrids(seed_analyses, best_svd)
    recommendation = _recommend_candidate_sources(
        best_svd,
        hybrid_report,
    )
    report = {
        "evaluation_protocol": CANDIDATE_ANALYSIS_PROTOCOL,
        "seeds": [int(seed) for seed in seeds],
        "cutoffs": list(CANDIDATE_CUTOFFS),
        "popularity_definition": "positive training interaction count (rating >= 3.5)",
        "popularity_strata_definition": (
            "count descending then film ID ascending; first ceil(10%) HEAD, "
            "through ceil(50%) MID, remainder TAIL"
        ),
        "svd_query_normalization": "none; all profiles retain their formula scale",
        "candidate_source_scope": (
            "positive-weighted SVD plus controlled popularity only; mean SVD and "
            "NCF are excluded from this final calibration"
        ),
        "per_seed": {
            str(seed.seed): {
                "runtime_seconds": seed.runtime_seconds,
                "target_count": len(seed.targets_by_user),
                "strategies": {
                    name: evaluation.report()
                    for name, evaluation in seed.strategies.items()
                },
            }
            for seed in seed_analyses
        },
        "aggregate_strategy_metrics": aggregates,
        "best_svd_profile": best_svd,
        "svd_popularity_hybrids": hybrid_report,
        "neural_retrieval_status": (
            "research-only; prior fixed-budget evidence did not justify inclusion"
        ),
        "recommended_candidate_strategy": recommendation,
        "runtime": {
            "per_seed_seconds": {
                str(seed.seed): seed.runtime_seconds for seed in seed_analyses
            },
            "three_seed_svd_popularity_seconds": sum(
                seed.runtime_seconds for seed in seed_analyses
            ),
            "total_seconds": time.perf_counter() - started,
        },
    }
    if report_path is not None:
        _atomic_json_write(report, Path(report_path))
    return report


def _analyze_seed(
    data: PreparedInteractions,
    *,
    seed: int,
    settings: Settings,
) -> SeedAnalysis:
    started = time.perf_counter()
    splits = build_interaction_splits(
        data,
        positive_rating_threshold=POSITIVE_RATING_THRESHOLD,
        negative_rating_threshold=NEGATIVE_RATING_THRESHOLD,
        seed=seed,
    )
    matrix = build_evaluation_svd_training_matrix(splits, len(data.film_ids))
    svd = TruncatedSVD(n_components=32, random_state=seed)
    svd.fit(matrix)
    item_vectors = np.ascontiguousarray(
        normalize(svd.components_.T, axis=1).astype(np.float32), dtype=np.float32
    )
    index = create_faiss_index(item_vectors, data.film_ids)
    counts = training_positive_counts(splits, len(data.film_ids))
    popularity_rows = popularity_order_rows(counts, data.film_ids)
    strata, percentiles = assign_popularity_strata(counts, data.film_ids)
    targets_by_user = {
        user.cohort_user_id: int(data.film_ids[user.test_target])
        for user in splits
        if user.test_target is not None and len(user.context_item_rows)
    }
    target_strata_by_user = {
        user.cohort_user_id: str(strata[user.test_target])
        for user in splits
        if user.test_target is not None and user.cohort_user_id in targets_by_user
    }
    known_by_user = {
        user.cohort_user_id: {int(data.film_ids[row]) for row in user.context_item_rows}
        for user in splits
        if user.cohort_user_id in targets_by_user
    }
    shell = SeedAnalysis(
        seed=seed,
        splits=splits,
        film_ids=data.film_ids,
        targets_by_user=targets_by_user,
        target_strata_by_user=target_strata_by_user,
        known_by_user=known_by_user,
        source_overlap_ranks_by_user={},
        strata_by_row=strata,
        popularity_percentiles=percentiles,
        strategies={},
        runtime_seconds=0.0,
    )
    popularity_candidates = {
        user_id: _top_popularity_candidates(
            popularity_rows,
            data.film_ids,
            known_by_user[user_id],
        )
        for user_id in targets_by_user
    }
    shell.strategies["popularity"] = _evaluate_candidates(
        "popularity",
        popularity_candidates,
        set(targets_by_user),
        shell,
    )
    candidates, eligible = _retrieve_svd_candidates(
        splits,
        data.film_ids,
        item_vectors,
        index,
        CANDIDATE_SVD_PROFILE,
    )
    shell.strategies[CANDIDATE_SVD_PROFILE] = _evaluate_candidates(
        CANDIDATE_SVD_PROFILE,
        candidates,
        eligible,
        shell,
    )
    shell.source_overlap_ranks_by_user = _source_overlap_ranks(shell)
    shell.known_by_user = {}
    shell.runtime_seconds = time.perf_counter() - started
    return shell


def _top_popularity_candidates(
    popularity_rows: NDArray[np.int64],
    film_ids: NDArray[np.int64],
    excluded: set[int],
) -> NDArray[np.int64]:
    candidates: list[int] = []
    for row in popularity_rows:
        film_id = int(film_ids[row])
        if film_id in excluded:
            continue
        candidates.append(film_id)
        if len(candidates) == min(max(CANDIDATE_CUTOFFS), len(film_ids)):
            break
    return np.ascontiguousarray(candidates, dtype=np.int64)


def _retrieve_svd_candidates(
    splits: tuple[UserSplit, ...],
    film_ids: NDArray[np.int64],
    item_vectors: NDArray[np.float32],
    index: faiss.IndexIDMap2,
    strategy: str,
) -> tuple[dict[int, NDArray[np.int64]], set[int]]:
    user_ids: list[int] = []
    queries: list[NDArray[np.float32]] = []
    exclusions: list[set[int]] = []
    for user in splits:
        if user.test_target is None or not len(user.context_item_rows):
            continue
        query = build_svd_profile(
            item_vectors,
            user.context_item_rows,
            user.context_rating_buckets,
            strategy,  # type: ignore[arg-type]
        )
        if query is None:
            continue
        user_ids.append(user.cohort_user_id)
        queries.append(query)
        exclusions.append({int(film_ids[row]) for row in user.context_item_rows})
    if not queries:
        return {}, set()
    requested_k = min(
        int(index.ntotal),
        max(CANDIDATE_CUTOFFS) + max(len(excluded) for excluded in exclusions),
    )
    _, labels = index.search(
        np.ascontiguousarray(np.stack(queries), dtype=np.float32), requested_k
    )
    candidates = {
        user_id: np.ascontiguousarray(
            [
                int(label)
                for label in row_labels
                if int(label) >= 0 and int(label) not in excluded
            ][: max(CANDIDATE_CUTOFFS)],
            dtype=np.int64,
        )
        for user_id, row_labels, excluded in zip(
            user_ids, labels, exclusions, strict=True
        )
    }
    return candidates, set(user_ids)


def _evaluate_candidates(
    name: str,
    candidates_by_user: dict[int, tuple[int, ...]],
    eligible_user_ids: set[int],
    seed: SeedAnalysis,
) -> StrategyEvaluation:
    ranks: dict[int, int | None] = {}
    for user_id, target in seed.targets_by_user.items():
        candidates = candidates_by_user.get(user_id)
        if candidates is None:
            ranks[user_id] = None
            continue
        matches = np.flatnonzero(candidates == target)
        ranks[user_id] = int(matches[0]) + 1 if len(matches) else None
    metrics = strategy_metrics_from_ranks(
        ranks, eligible_user_ids, catalog_size=len(seed.film_ids)
    )
    id_to_row = {
        int(film_id): row for row, film_id in enumerate(seed.film_ids.tolist())
    }
    retrieved_popularity: dict[int, dict[str, float | None]] = {}
    for cutoff in CANDIDATE_CUTOFFS:
        values = [
            float(seed.popularity_percentiles[id_to_row[int(film_id)]])
            for candidates in candidates_by_user.values()
            for film_id in candidates[:cutoff]
        ]
        retrieved_popularity[cutoff] = {
            "mean": fmean(values) if values else None,
            "median": float(np.median(values)) if values else None,
        }
    stratified: dict[str, dict[str, Any]] = {}
    target_row_by_user = {
        user.cohort_user_id: user.test_target
        for user in seed.splits
        if user.cohort_user_id in seed.targets_by_user
    }
    for stratum in ("HEAD", "MID", "TAIL"):
        user_ids = [
            user_id
            for user_id, row in target_row_by_user.items()
            if row is not None and seed.strata_by_row[row] == stratum
        ]
        view = metric_view_from_ranks(
            [ranks[user_id] for user_id in user_ids], catalog_size=len(seed.film_ids)
        )
        stratified[stratum] = metric_view_report(view)
        stratified[stratum]["target_count"] = len(user_ids)
    return StrategyEvaluation(
        name=name,
        candidates_by_user=candidates_by_user,
        ranks_by_user=ranks,
        eligible_user_ids=eligible_user_ids,
        metrics=metrics,
        catalog_coverage=catalog_coverage(
            candidates_by_user, catalog_size=len(seed.film_ids)
        ),
        retrieved_popularity=retrieved_popularity,
        stratified_metrics=stratified,
    )


def _aggregate_strategies(
    seeds: Sequence[SeedAnalysis],
) -> dict[str, dict[str, Any]]:
    strategy_names = ("popularity", CANDIDATE_SVD_PROFILE)
    result: dict[str, dict[str, Any]] = {}
    for name in strategy_names:
        evaluations = [seed.strategies[name] for seed in seeds]
        result[name] = {
            "recall_at": {
                cutoff: _mean_std(
                    [
                        evaluation.metrics.global_view.recall_at[cutoff]
                        for evaluation in evaluations
                    ]
                )
                for cutoff in CANDIDATE_CUTOFFS
            },
            "ndcg_at_10": _mean_std(
                [
                    evaluation.metrics.global_view.ndcg_at_10
                    for evaluation in evaluations
                ]
            ),
            "mrr_at_10": _mean_std(
                [evaluation.metrics.global_view.mrr_at_10 for evaluation in evaluations]
            ),
            "profile_coverage_percentage": _mean_std(
                [evaluation.metrics.coverage_percentage for evaluation in evaluations]
            ),
        }
    return result


def _analyze_svd_popularity_hybrids(
    seeds: Sequence[SeedAnalysis], best_svd: str
) -> dict[str, Any]:
    unlimited: dict[int, dict[str, float]] = {}
    for cutoff in CANDIDATE_BUDGETS:
        recalls = []
        for seed in seeds:
            popularity = seed.strategies["popularity"].candidates_by_user
            svd = seed.strategies[best_svd].candidates_by_user
            hits = sum(
                target
                in (set(popularity[user_id][:cutoff]) | set(svd[user_id][:cutoff]))
                for user_id, target in seed.targets_by_user.items()
            )
            recalls.append(hits / len(seed.targets_by_user))
        unlimited[cutoff] = _mean_std(recalls)

    allocation_sweep: dict[int, dict[str, Any]] = {}
    shortlist: dict[int, dict[str, Any]] = {}
    for budget in CANDIDATE_BUDGETS:
        configurations = candidate_allocation_grid(budget)
        evaluated = {
            label: _evaluate_hybrid_configuration(
                seeds,
                best_svd=best_svd,
                budget=budget,
                allocations=allocations,
                include_diagnostics=False,
            )
            for label, allocations in configurations.items()
        }
        selected_label = max(
            evaluated,
            key=lambda label: (
                evaluated[label]["recall"]["mean"],
                -evaluated[label]["recall"]["population_std"],
            ),
        )
        best_recall = evaluated[selected_label]["recall"]["mean"]
        near_winning_labels = [
            label
            for label, result in evaluated.items()
            if best_recall - result["recall"]["mean"] <= NEAR_WINNER_RECALL_DELTA
        ]
        for label in near_winning_labels:
            evaluated[label] = _evaluate_hybrid_configuration(
                seeds,
                best_svd=best_svd,
                budget=budget,
                allocations=configurations[label],
                include_diagnostics=True,
            )
        winner_location = ratio_grid_location(
            evaluated[selected_label]["allocations"], budget=budget
        )
        allocation_sweep[budget] = evaluated
        shortlist[budget] = {
            "selected_configuration": selected_label,
            "selected": evaluated[selected_label],
            "near_winning_configurations": near_winning_labels,
            "near_winner_absolute_recall_delta": NEAR_WINNER_RECALL_DELTA,
            "winner_grid_location": winner_location,
            "ratio_optimum_status": (
                "resolved within the tested bounded grid"
                if winner_location == "interior"
                else "unresolved because the best tested ratio is a boundary"
            ),
        }

    marginal = calculate_marginal_candidate_value(shortlist)
    selected_budget = choose_candidate_budget(shortlist, marginal)
    selected_step = next(
        (
            values
            for values in marginal.values()
            if int(values["from_budget"]) == selected_budget
        ),
        None,
    )
    measured_plateau_found = bool(
        selected_step
        and selected_step["absolute_recall_gain"] < BUDGET_MARGINAL_RECALL_THRESHOLD
    )
    return {
        "unlimited_union_recall": unlimited,
        "allocation_sweep": allocation_sweep,
        "shortlist_by_budget": shortlist,
        "marginal_candidate_value": marginal,
        "selected_budget": selected_budget,
        "selected_configuration": shortlist[selected_budget],
        "budget_decision_rule": (
            "identify the first sub-2-point marginal gain at or above 1000; if "
            "the tested range has no plateau, apply a pragmatic 4000-source "
            "production cap for ranking cost and an approximately 200-slot product"
        ),
        "measured_plateau_found": measured_plateau_found,
        "selection_classification": (
            "measured_diminishing_returns"
            if measured_plateau_found
            else "pragmatic_production_cap"
        ),
    }


def candidate_allocation_grid(budget: int) -> dict[str, dict[str, int]]:
    """Return the bounded deterministic SVD/popularity allocation grid."""
    if budget not in CANDIDATE_BUDGETS:
        raise ValueError(f"unsupported candidate budget: {budget}")

    result: dict[str, dict[str, int]] = {}
    for svd_percentage in (80, 70, 60, 50, 40, 30, 20):
        svd_depth = budget * svd_percentage // 100
        popularity_depth = budget - svd_depth
        label = f"{svd_depth}_weighted_{popularity_depth}_popularity"
        result[label] = {
            "svd": svd_depth,
            "popularity": popularity_depth,
        }
    return result


def ratio_grid_location(allocations: Mapping[str, int], *, budget: int) -> str:
    """Classify a two-source allocation as an interior or boundary grid point."""
    if budget <= 0 or sum(allocations.values()) != budget:
        raise ValueError("candidate allocations must sum to the positive budget")
    if set(allocations) != {"svd", "popularity"}:
        raise ValueError("ratio grid requires SVD and popularity allocations")
    svd_percentage = round(100 * allocations["svd"] / budget)
    return "boundary" if svd_percentage in (20, 80) else "interior"


def _evaluate_hybrid_configuration(
    seeds: Sequence[SeedAnalysis],
    *,
    best_svd: str,
    budget: int,
    allocations: Mapping[str, int],
    include_diagnostics: bool,
) -> dict[str, Any]:
    recalls: list[float] = []
    mean_sizes: list[float] = []
    catalog_percentages: list[float] = []
    catalog_unique_counts: list[float] = []
    popularity_means: list[float] = []
    popularity_medians: list[float] = []
    stratified: dict[str, list[float]] = {name: [] for name in ("HEAD", "MID", "TAIL")}
    per_seed_recall: dict[str, float] = {}
    per_seed_sizes: dict[str, float] = {}
    for seed in seeds:
        hits = 0
        sizes: list[int] = []
        stratum_hits = {name: 0 for name in stratified}
        stratum_totals = {name: 0 for name in stratified}
        catalog_ids: set[int] = set()
        occurrence_counts = np.zeros(len(seed.film_ids), dtype=np.int64)
        id_to_row = (
            {int(film_id): row for row, film_id in enumerate(seed.film_ids)}
            if include_diagnostics
            else {}
        )
        svd_evaluation = seed.strategies[best_svd]
        popularity_evaluation = seed.strategies["popularity"]
        for user_id in seed.targets_by_user:
            svd_candidates = svd_evaluation.candidates_by_user[user_id][
                : allocations["svd"]
            ]
            popularity_candidates = popularity_evaluation.candidates_by_user[user_id][
                : allocations["popularity"]
            ]
            overlap_count = sum(
                svd_rank <= allocations["svd"]
                and popularity_rank <= allocations["popularity"]
                for svd_rank, popularity_rank in seed.source_overlap_ranks_by_user[
                    user_id
                ]
            )
            sizes.append(
                len(svd_candidates) + len(popularity_candidates) - overlap_count
            )
            hit = (
                svd_evaluation.ranks_by_user[user_id] is not None
                and svd_evaluation.ranks_by_user[user_id] <= allocations["svd"]
            ) or (
                popularity_evaluation.ranks_by_user[user_id] is not None
                and popularity_evaluation.ranks_by_user[user_id]
                <= allocations["popularity"]
            )
            hits += hit
            stratum = seed.target_strata_by_user[user_id]
            stratum_totals[stratum] += 1
            stratum_hits[stratum] += hit
            if include_diagnostics:
                candidate_ids = {int(value) for value in svd_candidates}
                candidate_ids.update(int(value) for value in popularity_candidates)
                catalog_ids.update(candidate_ids)
                for film_id in candidate_ids:
                    occurrence_counts[id_to_row[film_id]] += 1
        recall = hits / len(seed.targets_by_user)
        mean_size = fmean(sizes)
        recalls.append(recall)
        mean_sizes.append(mean_size)
        per_seed_recall[str(seed.seed)] = recall
        per_seed_sizes[str(seed.seed)] = mean_size

        if include_diagnostics:
            catalog_unique_counts.append(float(len(catalog_ids)))
            catalog_percentages.append(100.0 * len(catalog_ids) / len(seed.film_ids))
            popularity_means.append(
                float(
                    np.average(
                        seed.popularity_percentiles,
                        weights=occurrence_counts,
                    )
                )
            )
            popularity_medians.append(
                _weighted_median(seed.popularity_percentiles, occurrence_counts)
            )
        for stratum, values in stratified.items():
            values.append(
                stratum_hits[stratum] / stratum_totals[stratum]
                if stratum_totals[stratum]
                else 0.0
            )
    report: dict[str, Any] = {
        "allocations": dict(allocations),
        "nominal_budget": budget,
        "recall": _mean_std(recalls),
        "per_seed_recall": per_seed_recall,
        "mean_deduplicated_candidates": _mean_std(mean_sizes),
        "per_seed_mean_deduplicated_candidates": per_seed_sizes,
        "stratified_recall": {
            name: _mean_std(values) for name, values in stratified.items()
        },
    }
    if include_diagnostics:
        report["catalog_coverage"] = {
            "scope": "unique films across all evaluated users collectively",
            "unique_films": _mean_std(catalog_unique_counts),
            "catalog_percentage": _mean_std(catalog_percentages),
        }
        report["retrieved_item_popularity_percentile"] = {
            "mean": _mean_std(popularity_means),
            "median": _mean_std(popularity_medians),
        }
    return report


def _source_overlap_ranks(
    seed: SeedAnalysis,
) -> dict[int, tuple[tuple[int, int], ...]]:
    """Precompute source-prefix overlaps for the bounded allocation sweep."""
    result: dict[int, tuple[tuple[int, int], ...]] = {}
    svd = seed.strategies[CANDIDATE_SVD_PROFILE].candidates_by_user
    popularity = seed.strategies["popularity"].candidates_by_user
    for user_id in seed.targets_by_user:
        svd_ranks = {
            int(film_id): rank for rank, film_id in enumerate(svd[user_id], start=1)
        }
        result[user_id] = tuple(
            (svd_ranks[int(film_id)], popularity_rank)
            for popularity_rank, film_id in enumerate(popularity[user_id], start=1)
            if int(film_id) in svd_ranks
        )
    return result


def _weighted_median(values: NDArray[np.float64], weights: NDArray[np.int64]) -> float:
    """Return the exact median of repeated values without materializing repeats."""
    positive = weights > 0
    ordered = np.argsort(values[positive])
    ordered_values = values[positive][ordered]
    ordered_weights = weights[positive][ordered]
    cumulative = np.cumsum(ordered_weights)
    count = int(cumulative[-1])
    lower_position = (count - 1) // 2 + 1
    upper_position = count // 2 + 1
    lower = ordered_values[np.searchsorted(cumulative, lower_position)]
    upper = ordered_values[np.searchsorted(cumulative, upper_position)]
    return float((lower + upper) / 2.0)


def calculate_marginal_candidate_value(
    shortlist: Mapping[int, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Measure incremental recall and deduplicated-candidate cost by budget step."""
    ordered_budgets = sorted(shortlist)
    result: dict[str, dict[str, float]] = {}
    for previous, current in pairwise(ordered_budgets):
        previous_selected = shortlist[previous]["selected"]
        current_selected = shortlist[current]["selected"]
        recall_gain = (
            current_selected["recall"]["mean"] - previous_selected["recall"]["mean"]
        )
        candidate_gain = (
            current_selected["mean_deduplicated_candidates"]["mean"]
            - previous_selected["mean_deduplicated_candidates"]["mean"]
        )
        result[f"{previous}_to_{current}"] = {
            "from_budget": previous,
            "to_budget": current,
            "absolute_recall_gain": recall_gain,
            "percentage_point_gain": recall_gain * 100.0,
            "additional_unique_candidates_per_user": candidate_gain,
            "additional_candidates_per_recall_percentage_point": (
                candidate_gain / (recall_gain * 100.0)
                if recall_gain > 0
                else float("inf")
            ),
        }
    return result


def choose_candidate_budget(
    shortlist: Mapping[int, Mapping[str, Any]],
    marginal: Mapping[str, Mapping[str, float]],
) -> int:
    """Identify a plateau, otherwise enforce the explicit pragmatic production cap."""
    budgets = sorted(shortlist)
    for budget, next_budget in pairwise(budgets):
        if budget < 1000:
            continue
        gain = marginal[f"{budget}_to_{next_budget}"]["absolute_recall_gain"]
        if gain < BUDGET_MARGINAL_RECALL_THRESHOLD:
            return budget
    eligible_caps = [
        budget for budget in budgets if budget <= PRAGMATIC_MAX_CANDIDATE_BUDGET
    ]
    return eligible_caps[-1]


def _recommend_candidate_sources(
    best_svd: str,
    hybrid_report: Mapping[str, Any],
) -> dict[str, Any]:
    selected_budget = int(hybrid_report["selected_budget"])
    selection = hybrid_report["selected_configuration"]
    selected = selection["selected"]
    selected_label = selection["selected_configuration"]
    recommendation: dict[str, Any] = {
        "classification": "svd_popularity_hybrid",
        "best_svd_profile": best_svd,
        "nominal_budget": selected_budget,
        "configuration": selected_label,
        "allocations": selected["allocations"],
        "expected_unique_candidates": selected["mean_deduplicated_candidates"],
        "measured_recall": selected["recall"],
        "ratio_grid_location": selection["winner_grid_location"],
        "selection_classification": hybrid_report["selection_classification"],
        "mean_svd_included": False,
        "ncf_included": False,
    }
    return recommendation


def metric_view_report(view: MetricView) -> dict[str, Any]:
    return {
        "recall_at": dict(view.recall_at),
        "ndcg_at_10": view.ndcg_at_10,
        "mrr_at_10": view.mrr_at_10,
        "evaluated_users": view.evaluated_users,
    }


def strategy_metrics_report(metrics: StrategyMetrics) -> dict[str, Any]:
    return {
        "conditional": metric_view_report(metrics.conditional),
        "global": metric_view_report(metrics.global_view),
        "eligible_users": metrics.eligible_users,
        "total_users": metrics.total_users,
        "coverage_percentage": metrics.coverage_percentage,
    }


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    return {"mean": fmean(values), "population_std": pstdev(values)}


def _atomic_json_write(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
