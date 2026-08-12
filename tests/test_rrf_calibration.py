import json

import numpy as np

from experiments.ranker.metrics import target_rank, target_rank_metrics
from experiments.ranker.rrf_calibration import (
    DEFAULT_RRF_CONFIGURATION,
    RRF_CONFIGURATIONS,
    RRFConfiguration,
    _aggregate_validation_grid,
    _rrf_scores,
    recommend_fixed_configuration,
    select_validation_configuration,
)


def _validation_report(
    *, global_ndcg: float = 0.1, global_recall: float = 0.2, conditional: float = 0.3
) -> dict:
    return {
        "global": {
            "ndcg_at_20": global_ndcg,
            "recall_at_20": global_recall,
        },
        "candidate_conditional": {"ndcg_at_20": conditional},
    }


def test_rrf_grid_is_the_bounded_declared_search() -> None:
    assert len(RRF_CONFIGURATIONS) == 28
    assert len(set(RRF_CONFIGURATIONS)) == 28
    assert {configuration.k for configuration in RRF_CONFIGURATIONS} == {
        20,
        60,
        100,
        200,
    }
    assert {
        (configuration.svd_weight, configuration.popularity_weight)
        for configuration in RRF_CONFIGURATIONS
    } == {
        (50, 50),
        (60, 40),
        (70, 30),
        (80, 20),
        (40, 60),
        (30, 70),
        (20, 80),
    }


def test_weighted_rrf_missing_sources_are_zero_and_film_id_breaks_ties() -> None:
    configuration = RRFConfiguration(60, 40, 20)
    film_ids = np.asarray([30, 10, 20], dtype=np.int64)
    scores = _rrf_scores(
        np.asarray([1.0, np.inf, 1.0]),
        np.asarray([np.inf, 1.0, np.inf]),
        configuration,
    )

    assert scores.tolist() == [60 / 21, 40 / 21, 60 / 21]
    assert target_rank(film_ids, scores, 20) == 1
    assert target_rank(film_ids, scores, 30) == 2
    assert target_rank(film_ids, scores, 999) is None


def test_rank_metrics_keep_missed_targets_in_the_global_denominator() -> None:
    report = target_rank_metrics([1, 20, None])

    assert report["users"] == 3
    assert report["recall_at_10"] == 1 / 3
    assert report["recall_at_20"] == 2 / 3
    assert report["recall_at_50"] == 2 / 3
    assert report["mrr_at_10"] == 1 / 3


def test_fold_selection_uses_declared_metric_order_then_default_nearness() -> None:
    reports = {
        configuration: _validation_report() for configuration in RRF_CONFIGURATIONS
    }
    assert select_validation_configuration(reports) == DEFAULT_RRF_CONFIGURATION

    recall_winner = RRFConfiguration(60, 40, 60)
    reports[recall_winner] = _validation_report(global_recall=0.21)
    assert select_validation_configuration(reports) == recall_winner

    ndcg_winner = RRFConfiguration(20, 80, 200)
    reports[ndcg_winner] = _validation_report(global_ndcg=0.101)
    assert select_validation_configuration(reports) == ndcg_winner


def test_fixed_recommendation_prefers_default_inside_validation_plateau() -> None:
    aggregate = {
        configuration.key: {"global": {"ndcg_at_20": {"mean": 0.1}}}
        for configuration in RRF_CONFIGURATIONS
    }
    best = RRFConfiguration(70, 30, 20)
    aggregate[best.key]["global"]["ndcg_at_20"]["mean"] = 0.1009

    recommended, plateau = recommend_fixed_configuration(aggregate)

    assert recommended == DEFAULT_RRF_CONFIGURATION
    assert DEFAULT_RRF_CONFIGURATION in plateau
    assert best in plateau


def test_aggregate_validation_grid_is_json_serializable() -> None:
    metrics = {
        "recall_at_10": 0.1,
        "recall_at_20": 0.2,
        "recall_at_50": 0.3,
        "ndcg_at_10": 0.04,
        "ndcg_at_20": 0.05,
        "mrr_at_10": 0.03,
    }
    fold_report = {
        "candidate_recall": 0.8,
        "candidate_conditional": metrics,
        "global": metrics,
    }
    by_configuration = {
        configuration: [fold_report] * 15 for configuration in RRF_CONFIGURATIONS
    }

    payload = _aggregate_validation_grid(by_configuration)

    assert json.loads(json.dumps(payload))
