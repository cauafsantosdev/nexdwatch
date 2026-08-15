"""Focused correctness tests for offline candidate-generation analysis."""

import subprocess
import sys

import numpy as np
import pytest

from app.ml.historical_interactions import UserSplit
from experiments.evaluation import training_positive_counts
from experiments.retrieval.candidate_analysis import (
    CANDIDATE_BUDGETS,
    CANDIDATE_CUTOFFS,
    assign_popularity_strata,
    attribute_unique_target_hits,
    build_budgeted_hybrid,
    build_svd_profile,
    calculate_marginal_candidate_value,
    candidate_allocation_grid,
    catalog_coverage,
    choose_candidate_budget,
    mean_jaccard,
    metric_view_from_ranks,
    ratio_grid_location,
)


def _buckets(*ratings: float) -> np.ndarray:
    return np.asarray([round(rating * 2) - 1 for rating in ratings], dtype=np.int64)


def _split(
    *,
    context: list[int],
    training_positives: list[int],
    validation: int | None,
    test: int | None,
) -> UserSplit:
    held_out = [target for target in (validation, test) if target is not None]
    all_rows = np.asarray(context + held_out, dtype=np.int64)
    return UserSplit(
        cohort_user_id=1,
        all_item_rows=all_rows,
        all_rating_buckets=np.full(len(all_rows), 7, dtype=np.int64),
        context_item_rows=np.asarray(context, dtype=np.int64),
        context_rating_buckets=np.full(len(context), 7, dtype=np.int64),
        training_positive_rows=np.asarray(training_positives, dtype=np.int64),
        explicit_negative_rows=np.array([], dtype=np.int64),
        validation_target=validation,
        test_target=test,
    )


def test_all_svd_profile_formulas_are_hand_computable() -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    rows = np.asarray([0, 1, 2], dtype=np.int64)

    assert build_svd_profile(
        vectors, rows, _buckets(1, 3, 5), "svd_mean"
    ) == pytest.approx([2 / 3, 2 / 3])
    assert build_svd_profile(
        vectors, rows, _buckets(1, 3, 5), "svd_positive_mean"
    ) == pytest.approx([1.0, 1.0])
    # The 1-star vector contributes in the opposite direction; 3.0 is zero.
    assert build_svd_profile(
        vectors, rows, _buckets(1, 3, 5), "svd_rating_centered"
    ) == pytest.approx([0.0, 0.5])
    assert build_svd_profile(
        vectors, rows, _buckets(1, 3, 5), "svd_positive_weighted"
    ) == pytest.approx([1.0, 1.0])


def test_user_centered_profile_uses_absolute_weight_denominator() -> None:
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    rows = np.asarray([0, 1, 2], dtype=np.int64)
    ratings = np.asarray([2.0, 4.0, 5.0], dtype=np.float32)
    weights = ratings - ratings.mean()
    expected = (weights @ vectors) / np.abs(weights).sum()

    query = build_svd_profile(vectors, rows, _buckets(*ratings), "svd_user_centered")

    assert weights.sum() == pytest.approx(0.0, abs=1e-6)
    assert query == pytest.approx(expected)


@pytest.mark.parametrize(
    ("strategy", "ratings"),
    [
        ("svd_positive_mean", (1.0, 2.5, 3.0)),
        ("svd_rating_centered", (3.0, 3.0, 3.0)),
        ("svd_user_centered", (4.0, 4.0, 4.0)),
        ("svd_positive_weighted", (1.0, 2.0, 3.0)),
    ],
)
def test_invalid_or_empty_weighted_profiles_are_unavailable(
    strategy: str, ratings: tuple[float, ...]
) -> None:
    vectors = np.eye(3, dtype=np.float32)
    rows = np.arange(3, dtype=np.int64)

    assert build_svd_profile(vectors, rows, _buckets(*ratings), strategy) is None


def test_candidate_depth_metrics_and_small_catalog_cutoffs() -> None:
    ranks = [1, 50, 100, 250, 500, None]

    metrics = metric_view_from_ranks(ranks, catalog_size=1_000)
    small = metric_view_from_ranks([10, None], catalog_size=10)

    assert metrics.recall_at == {
        10: 1 / 6,
        50: 2 / 6,
        100: 3 / 6,
        250: 4 / 6,
        500: 5 / 6,
        750: 5 / 6,
        1000: 5 / 6,
        1500: 5 / 6,
        2000: 5 / 6,
        2500: 5 / 6,
        3000: 5 / 6,
        4000: 5 / 6,
        5000: 5 / 6,
    }
    assert metrics.ndcg_at_10 == pytest.approx(1 / 6)
    assert metrics.mrr_at_10 == pytest.approx(1 / 6)
    assert small.recall_at[500] == 0.5


def test_popularity_strata_use_training_counts_and_deterministic_ties() -> None:
    film_ids = np.arange(10, 20, dtype=np.int64)
    split = _split(context=[0, 1], training_positives=[0, 1], validation=2, test=3)
    counts = training_positive_counts((split,), item_count=10)

    first, percentiles = assign_popularity_strata(counts, film_ids)
    second, _ = assign_popularity_strata(counts, film_ids)

    assert counts[split.validation_target] == counts[split.test_target] == 0
    assert first.tolist() == second.tolist()
    assert first[0] == "HEAD"
    # Film ID deterministically orders the tied one-count and zero-count films.
    assert first[1] == "MID"
    assert first[2] == "MID"
    assert first[5] == "TAIL"
    assert percentiles[0] > percentiles[1] > percentiles[2]


def test_catalog_coverage_counts_unique_films_once() -> None:
    coverage = catalog_coverage(
        {1: (1, 2, 3), 2: (2, 3, 4)}, catalog_size=10, cutoffs=(3,)
    )

    assert coverage[3] == {"unique_films": 4, "catalog_percentage": 40.0}


def test_candidate_overlap_uses_mean_user_jaccard() -> None:
    score = mean_jaccard({1: (1, 2, 3)}, {1: (2, 3, 4)}, (1,), cutoff=3)

    assert score == 0.5


def test_unique_target_hit_attribution_is_exact() -> None:
    targets = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}
    candidates = {
        "popularity": {1: (10,), 4: (40,)},
        "svd": {2: (20,), 4: (40,)},
        "ncf": {3: (30,), 4: (40,)},
    }

    result = attribute_unique_target_hits(targets, candidates, cutoff=100)

    assert result["popularity_only"]["count"] == 1
    assert result["svd_only"]["count"] == 1
    assert result["ncf_only"]["count"] == 1
    assert result["all_three"]["count"] == 1
    assert result["none"]["count"] == 1


def test_budgeted_hybrid_deduplicates_excludes_and_preserves_sources() -> None:
    hybrid = build_budgeted_hybrid(
        {"svd": (1, 2, 3), "popularity": (2, 4, 5)},
        {"svd": 3, "popularity": 3},
        known_film_ids={1},
        max_budget=4,
    )

    assert hybrid.ordered_ids == (2, 3, 4, 5)
    assert hybrid.sources_by_film[2] == ("svd", "popularity")
    assert hybrid.source_ranks_by_film[2] == {"svd": 2, "popularity": 1}
    assert 1 not in hybrid.ordered_ids
    assert 5 in hybrid.ordered_ids


def test_standard_candidate_analysis_import_does_not_load_torch() -> None:
    script = (
        "import sys; import experiments.retrieval.candidate_analysis; "
        "assert 'torch' not in sys.modules"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_expanded_allocation_grid_is_bounded_and_exact() -> None:
    assert CANDIDATE_CUTOFFS[-9:] == CANDIDATE_BUDGETS
    assert CANDIDATE_BUDGETS == (
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
    grid = candidate_allocation_grid(1000)

    assert len(grid) == 7
    assert {sum(allocation.values()) for allocation in grid.values()} == {1000}
    assert all(set(allocation) == {"svd", "popularity"} for allocation in grid.values())
    assert grid["800_weighted_200_popularity"] == {
        "svd": 800,
        "popularity": 200,
    }
    assert grid["600_weighted_400_popularity"] == {
        "svd": 600,
        "popularity": 400,
    }
    assert grid["200_weighted_800_popularity"] == {
        "svd": 200,
        "popularity": 800,
    }


def test_ratio_grid_location_detects_boundary_optima() -> None:
    assert ratio_grid_location({"svd": 200, "popularity": 800}, budget=1000) == (
        "boundary"
    )
    assert ratio_grid_location({"svd": 500, "popularity": 500}, budget=1000) == (
        "interior"
    )


def test_marginal_value_and_budget_decision_use_deduplicated_sizes() -> None:
    shortlist = {
        500: {
            "selected": {
                "recall": {"mean": 0.50},
                "mean_deduplicated_candidates": {"mean": 460.0},
            }
        },
        1000: {
            "selected": {
                "recall": {"mean": 0.62},
                "mean_deduplicated_candidates": {"mean": 910.0},
            }
        },
        1500: {
            "selected": {
                "recall": {"mean": 0.635},
                "mean_deduplicated_candidates": {"mean": 1360.0},
            }
        },
        2000: {
            "selected": {
                "recall": {"mean": 0.64},
                "mean_deduplicated_candidates": {"mean": 1810.0},
            }
        },
    }

    marginal = calculate_marginal_candidate_value(shortlist)

    assert marginal["1000_to_1500"]["absolute_recall_gain"] == pytest.approx(0.015)
    assert marginal["1000_to_1500"]["additional_unique_candidates_per_user"] == 450
    assert choose_candidate_budget(shortlist, marginal) == 1000
