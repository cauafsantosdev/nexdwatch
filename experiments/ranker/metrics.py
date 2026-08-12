"""Ranking metrics with explicit candidate-conditional and global denominators."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from experiments.ranker.config import LABEL_GAIN
from experiments.ranker.dataset import PartitionDataset, QueryAudit


@dataclass(frozen=True, slots=True)
class RankingView:
    """Aggregated target and graded metrics for one explicit denominator."""

    users: int
    recall_at_10: float
    recall_at_20: float
    recall_at_50: float
    ndcg_at_10: float
    ndcg_at_20: float
    mrr_at_10: float
    graded_ndcg_at_10: float
    graded_ndcg_at_20: float


@dataclass(frozen=True, slots=True)
class UserRankingResult:
    """Per-user sufficient statistics retained for segments and diagnostics."""

    user_id: int
    target_rank: int | None
    graded_ndcg_at_10: float
    graded_ndcg_at_20: float


def target_rank(
    film_ids: NDArray[np.int64],
    scores: NDArray[np.floating],
    target_film_id: int,
) -> int | None:
    """Return the exact deterministic target rank for one candidate group."""
    if len(film_ids) != len(scores):
        raise ValueError("ranking film IDs and scores differ in length")
    target_positions = np.flatnonzero(film_ids == target_film_id)
    if not len(target_positions):
        return None
    safe_scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-np.inf)
    target_score = safe_scores[int(target_positions[0])]
    return int(
        1
        + np.count_nonzero(safe_scores > target_score)
        + np.count_nonzero((safe_scores == target_score) & (film_ids < target_film_id))
    )


def target_rank_metrics(
    ranks: list[int | None],
    *,
    denominator: int | None = None,
) -> dict[str, float | int]:
    """Aggregate canonical-target metrics from already ranked user groups."""
    users = len(ranks) if denominator is None else denominator
    if users < len(ranks):
        raise ValueError("ranking denominator cannot be smaller than rank count")
    if users <= 0:
        return {
            "users": 0,
            "recall_at_10": 0.0,
            "recall_at_20": 0.0,
            "recall_at_50": 0.0,
            "ndcg_at_10": 0.0,
            "ndcg_at_20": 0.0,
            "mrr_at_10": 0.0,
        }
    return {
        "users": users,
        "recall_at_10": sum(rank is not None and rank <= 10 for rank in ranks) / users,
        "recall_at_20": sum(rank is not None and rank <= 20 for rank in ranks) / users,
        "recall_at_50": sum(rank is not None and rank <= 50 for rank in ranks) / users,
        "ndcg_at_10": sum(
            1.0 / np.log2(rank + 1) for rank in ranks if rank is not None and rank <= 10
        )
        / users,
        "ndcg_at_20": sum(
            1.0 / np.log2(rank + 1) for rank in ranks if rank is not None and rank <= 20
        )
        / users,
        "mrr_at_10": sum(
            1.0 / rank for rank in ranks if rank is not None and rank <= 10
        )
        / users,
    }


def evaluate_ranking_scores(
    dataset: PartitionDataset,
    scores: NDArray[np.floating],
) -> dict[str, Any]:
    """Evaluate scores while keeping candidate failures visible globally."""
    if len(scores) != len(dataset.labels):
        raise ValueError("ranking score count differs from dataset rows")
    all_results = _per_user_results(dataset, scores)
    result_by_user = {result.user_id: result for result in all_results}
    retrieved_user_ids = {
        query.user_id
        for query in dataset.all_queries
        if query.designated_target_retrieved
    }
    conditional_results = [
        result for result in all_results if result.user_id in retrieved_user_ids
    ]
    conditional = _aggregate(conditional_results, len(conditional_results))
    global_results = [
        result_by_user.get(query.user_id, _zero_result(query.user_id))
        for query in dataset.all_queries
        if query.designated_target_id is not None
    ]
    global_view = _aggregate(global_results, dataset.eligible_user_count)
    return {
        "candidate_recall": (
            dataset.candidate_retrieved_user_count / dataset.eligible_user_count
            if dataset.eligible_user_count
            else 0.0
        ),
        "candidate_retrieved_users": dataset.candidate_retrieved_user_count,
        "eligible_users": dataset.eligible_user_count,
        "candidate_conditional": asdict(conditional),
        "global": asdict(global_view),
        "segments": _segment_reports(dataset.all_queries, result_by_user),
        "per_user": [asdict(result) for result in conditional_results],
        "per_user_global": [asdict(result) for result in global_results],
    }


def baseline_reports(dataset: PartitionDataset) -> dict[str, dict[str, Any]]:
    """Evaluate every frozen baseline and the candidate oracle on identical rows."""
    names = ("popularity", "positive_weighted_svd", "rrf", "svd_mean_pooling")
    reports = {
        name: evaluate_ranking_scores(dataset, dataset.baseline_scores[:, index])
        for index, name in enumerate(names)
    }
    reports["candidate_oracle"] = evaluate_ranking_scores(
        dataset, dataset.labels.astype(np.float32)
    )
    return reports


def _per_user_results(
    dataset: PartitionDataset, scores: NDArray[np.floating]
) -> list[UserRankingResult]:
    results: list[UserRankingResult] = []
    stop = 0
    for size, query in zip(dataset.group_sizes, dataset.queries, strict=True):
        start, stop = stop, stop + int(size)
        group_scores = np.nan_to_num(
            np.asarray(scores[start:stop], dtype=np.float64), nan=-np.inf
        )
        group_film_ids = dataset.film_ids[start:stop]
        order = np.lexsort((group_film_ids, -group_scores))
        rank = (
            target_rank(group_film_ids, group_scores, query.designated_target_id)
            if query.designated_target_id is not None
            else None
        )
        ordered_labels = dataset.labels[start:stop][order]
        results.append(
            UserRankingResult(
                user_id=query.user_id,
                target_rank=rank,
                graded_ndcg_at_10=_graded_ndcg(ordered_labels, 10),
                graded_ndcg_at_20=_graded_ndcg(ordered_labels, 20),
            )
        )
    return results


def _aggregate(results: list[UserRankingResult], denominator: int) -> RankingView:
    if denominator <= 0:
        return RankingView(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ranks = [result.target_rank for result in results]
    return RankingView(
        users=denominator,
        recall_at_10=sum(rank is not None and rank <= 10 for rank in ranks)
        / denominator,
        recall_at_20=sum(rank is not None and rank <= 20 for rank in ranks)
        / denominator,
        recall_at_50=sum(rank is not None and rank <= 50 for rank in ranks)
        / denominator,
        ndcg_at_10=sum(
            1.0 / np.log2(rank + 1) for rank in ranks if rank is not None and rank <= 10
        )
        / denominator,
        ndcg_at_20=sum(
            1.0 / np.log2(rank + 1) for rank in ranks if rank is not None and rank <= 20
        )
        / denominator,
        mrr_at_10=sum(1.0 / rank for rank in ranks if rank is not None and rank <= 10)
        / denominator,
        graded_ndcg_at_10=sum(result.graded_ndcg_at_10 for result in results)
        / denominator,
        graded_ndcg_at_20=sum(result.graded_ndcg_at_20 for result in results)
        / denominator,
    )


def _graded_ndcg(ordered_labels: NDArray[np.int8], cutoff: int) -> float:
    gains = np.asarray([LABEL_GAIN[int(label)] for label in ordered_labels])
    ideal = np.sort(gains)[::-1]
    discounts = 1.0 / np.log2(np.arange(2, min(cutoff, len(gains)) + 2))
    ideal_dcg = float(ideal[:cutoff] @ discounts)
    return float(gains[:cutoff] @ discounts / ideal_dcg) if ideal_dcg else 0.0


def _segment_reports(
    queries: tuple[QueryAudit, ...],
    result_by_user: dict[int, UserRankingResult],
) -> dict[str, dict[str, dict[str, Any]]]:
    dimensions = {
        "target_stratum": lambda query: query.target_stratum,
        "history_depth": lambda query: (
            "bottom_quartile"
            if query.history_depth_bucket == 0
            else "top_quartile"
            if query.history_depth_bucket == 3
            else "middle_50_percent"
        ),
        "target_source": lambda query: query.target_source,
    }
    reports: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension, classifier in dimensions.items():
        grouped: dict[str, list[QueryAudit]] = {}
        for query in queries:
            if query.designated_target_id is not None:
                grouped.setdefault(classifier(query), []).append(query)
        reports[dimension] = {}
        for name, members in sorted(grouped.items()):
            global_values = [
                result_by_user.get(query.user_id, _zero_result(query.user_id))
                for query in members
            ]
            conditional_values = [
                result_by_user[query.user_id]
                for query in members
                if query.designated_target_retrieved
            ]
            retrieved = len(conditional_values)
            eligible = len(members)
            reports[dimension][name] = {
                "eligible_targets": eligible,
                "candidate_retrieved_targets": retrieved,
                "candidate_recall": retrieved / eligible if eligible else 0.0,
                "candidate_conditional": asdict(
                    _aggregate(conditional_values, retrieved)
                ),
                "global": asdict(_aggregate(global_values, eligible)),
            }
    return reports


def _zero_result(user_id: int) -> UserRankingResult:
    return UserRankingResult(user_id, None, 0.0, 0.0)
