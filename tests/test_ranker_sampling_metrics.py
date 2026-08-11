from types import SimpleNamespace

import numpy as np

from app.domain.candidates import CandidateGenerationResult, RecommendationCandidate
from experiments.ranker import dataset as dataset_module
from experiments.ranker.config import FEATURE_NAMES
from experiments.ranker.dataset import (
    PartitionDataset,
    QueryAudit,
    RankerUserExample,
    build_partition_dataset,
    load_partition_dataset,
    write_partition_dataset,
)
from experiments.ranker.metrics import baseline_reports, evaluate_ranking_scores
from experiments.ranker.sampling import (
    build_full_evaluation_group,
    sample_ranker_group,
)
from experiments.ranker.uncertainty import paired_user_clustered_comparisons


def _candidate(film_id: int) -> RecommendationCandidate:
    return RecommendationCandidate(
        film_id=film_id,
        svd_score=1.0 / film_id,
        svd_rank=film_id if film_id <= 400 else None,
        popularity_score=1000 - film_id if film_id <= 600 else None,
        popularity_rank=film_id if film_id <= 600 else None,
        retrieved_by_svd=film_id <= 400,
        retrieved_by_popularity=film_id <= 600,
    )


def test_sampling_is_deterministic_capped_and_never_injects_misses() -> None:
    candidates = tuple(_candidate(film_id) for film_id in range(1, 801))
    labels = {2: 3, 799: 1, 9999: 3}
    first = sample_ranker_group(candidates, labels, seed=42, user_id=7)
    second = sample_ranker_group(candidates, labels, seed=42, user_id=7)
    assert len(first.candidates) == 512
    assert [candidate.film_id for candidate in first.candidates] == [
        candidate.film_id for candidate in second.candidates
    ]
    assert first.candidates[0].film_id == 2
    assert first.candidates[1].film_id == 799
    assert 9999 not in {candidate.film_id for candidate in first.candidates}
    assert first.retrieved_positive_count == 2
    assert first.requested_positive_count == 3


def test_full_evaluation_retains_inventory_without_target_injection() -> None:
    candidates = tuple(_candidate(film_id) for film_id in range(1, 801))
    group = build_full_evaluation_group(
        candidates,
        {799: 3, 9999: 3},
        forbidden_positive_ids={7},
    )
    assert len(group.candidates) == 799
    assert {candidate.film_id for candidate in group.candidates} == set(
        range(1, 801)
    ) - {7}
    assert group.retrieved_positive_count == 1
    assert group.requested_positive_count == 2
    assert 9999 not in {candidate.film_id for candidate in group.candidates}
    assert group.sampling_strata.count("full_pool_candidate") == 798


def test_partition_uses_sampled_train_but_materializes_full_missed_eval(
    monkeypatch,
) -> None:
    candidates = tuple(_candidate(film_id) for film_id in range(1, 801))
    monkeypatch.setattr(
        dataset_module,
        "generate_fold_candidates",
        lambda *args: CandidateGenerationResult(
            user_id=7,
            candidates=candidates,
            nominal_budget=4000,
            svd_depth=2000,
            popularity_depth=2000,
            svd_profile_available=True,
        ),
    )
    monkeypatch.setattr(
        dataset_module,
        "build_user_feature_profile",
        lambda *args, **kwargs: SimpleNamespace(
            base_features={"history_depth_bucket": 1.0}
        ),
    )
    monkeypatch.setattr(
        dataset_module,
        "build_feature_matrix",
        lambda rows, *args, **kwargs: np.zeros(
            (len(rows), len(FEATURE_NAMES)), dtype=np.float32
        ),
    )
    monkeypatch.setattr(
        dataset_module,
        "build_personalized_feature_matrix",
        lambda rows, *args, **kwargs: np.zeros((len(rows), 60), dtype=np.float32),
    )
    monkeypatch.setattr(
        dataset_module,
        "_baseline_scores",
        lambda rows, *args: np.zeros((len(rows), 4), dtype=np.float32),
    )
    evaluation = RankerUserExample(
        user_id=7,
        context_item_rows=np.asarray([0], dtype=np.int64),
        context_rating_buckets=np.asarray([9], dtype=np.int64),
        positive_labels={9999: 3},
        forbidden_positive_ids={7},
        designated_target_id=9999,
        designated_target_label=3,
        target_stratum="TAIL",
    )
    full = build_partition_dataset(
        (evaluation,),
        object(),
        object(),
        partition="validation",
        seed=42,
        fold=0,
        history_depth_thresholds=(1.0, 2.0, 3.0),
    )
    assert len(full.labels) == 799
    assert len(full.group_sizes) == 1
    assert full.candidate_retrieved_user_count == 0
    assert full.queries[0].ranking_inventory == "full_candidate_inventory"
    assert full.queries[0].ranking_row_count == 799
    assert not np.any(full.labels)

    training = RankerUserExample(
        user_id=7,
        context_item_rows=np.asarray([0], dtype=np.int64),
        context_rating_buckets=np.asarray([9], dtype=np.int64),
        positive_labels={799: 3, 9999: 3},
        forbidden_positive_ids={7},
        designated_target_id=None,
        designated_target_label=0,
        target_stratum="NONE",
    )
    sampled = build_partition_dataset(
        (training,),
        object(),
        object(),
        partition="train",
        seed=42,
        fold=0,
        history_depth_thresholds=(1.0, 2.0, 3.0),
    )
    assert len(sampled.labels) == 512
    assert sampled.queries[0].ranking_inventory == "sampled_512"


def _metric_dataset() -> PartitionDataset:
    features = np.zeros((6, len(FEATURE_NAMES)), dtype=np.float32)
    return PartitionDataset(
        features=features,
        shuffled_personalized_features=np.zeros((6, 60), dtype=np.float32),
        labels=np.asarray([0, 3, 0, 3, 0, 0], dtype=np.int8),
        query_ids=np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int64),
        film_ids=np.asarray([10, 11, 20, 21, 30, 32], dtype=np.int64),
        sampling_strata=np.zeros(6, dtype=np.int8),
        baseline_scores=np.asarray(
            [
                [-1, 0.1, 0.1, 0.1],
                [-2, 0.9, 0.9, 0.9],
                [-1, 0.1, 0.1, 0.1],
                [-2, 0.9, 0.9, 0.9],
                [-1, 0.8, 0.8, 0.8],
                [-2, 0.2, 0.2, 0.2],
            ],
            dtype=np.float32,
        ),
        group_sizes=np.asarray([2, 2, 2], dtype=np.int32),
        queries=(
            QueryAudit(
                1,
                1,
                1,
                11,
                True,
                "both",
                "HEAD",
                0,
                10,
                2,
                "full_candidate_inventory",
            ),
            QueryAudit(
                2,
                1,
                1,
                21,
                True,
                "svd_only",
                "TAIL",
                3,
                10,
                2,
                "full_candidate_inventory",
            ),
            QueryAudit(
                3,
                1,
                0,
                31,
                False,
                "missed",
                "MID",
                2,
                10,
                2,
                "full_candidate_inventory",
            ),
        ),
        all_queries=(
            QueryAudit(
                1,
                1,
                1,
                11,
                True,
                "both",
                "HEAD",
                0,
                10,
                2,
                "full_candidate_inventory",
            ),
            QueryAudit(
                2,
                1,
                1,
                21,
                True,
                "svd_only",
                "TAIL",
                3,
                10,
                2,
                "full_candidate_inventory",
            ),
            QueryAudit(
                3,
                1,
                0,
                31,
                False,
                "missed",
                "MID",
                2,
                10,
                2,
                "full_candidate_inventory",
            ),
        ),
        eligible_user_count=3,
        candidate_retrieved_user_count=2,
        failed_training_group_count=0,
    )


def test_metrics_keep_candidate_misses_in_global_denominator() -> None:
    dataset = _metric_dataset()
    dataset.validate()
    report = evaluate_ranking_scores(
        dataset, np.asarray([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    )
    assert report["candidate_conditional"]["recall_at_10"] == 1.0
    assert report["global"]["recall_at_10"] == 2 / 3
    assert report["candidate_recall"] == 2 / 3
    assert len(report["per_user"]) == 2
    assert len(report["per_user_global"]) == 3
    assert report["segments"]["target_stratum"]["MID"]["global"]["ndcg_at_20"] == 0.0


def test_baselines_rank_on_identical_group_rows() -> None:
    reports = baseline_reports(_metric_dataset())
    assert (
        reports["positive_weighted_svd"]["candidate_conditional"]["recall_at_10"] == 1.0
    )
    assert reports["candidate_oracle"]["candidate_conditional"]["ndcg_at_10"] == 1.0


def test_numeric_partition_round_trip_preserves_group_audits(tmp_path) -> None:
    expected = _metric_dataset()
    write_partition_dataset(expected, tmp_path, seed=42, fold=0, partition="test")
    actual = load_partition_dataset(tmp_path, "test")
    np.testing.assert_array_equal(actual.features, expected.features)
    np.testing.assert_array_equal(actual.group_sizes, expected.group_sizes)
    assert actual.queries == expected.queries
    assert actual.all_queries == expected.all_queries


def test_paired_interval_clusters_repeated_seed_appearances_by_user() -> None:
    def ranking(user_id: int, rank: int | None) -> dict[str, int | None]:
        return {"user_id": user_id, "target_rank": rank}

    reports = [
        {
            "models": {
                "full": {"test": {"per_user_global": [ranking(1, 1), ranking(2, 1)]}}
            },
            "baselines": {
                "test": {
                    "rrf": {"per_user_global": [ranking(1, None), ranking(2, None)]},
                    "positive_weighted_svd": {
                        "per_user_global": [ranking(1, None), ranking(2, None)]
                    },
                }
            },
        },
        {
            "models": {"full": {"test": {"per_user_global": [ranking(1, None)]}}},
            "baselines": {
                "test": {
                    "rrf": {"per_user_global": [ranking(1, 1)]},
                    "positive_weighted_svd": {"per_user_global": [ranking(1, 1)]},
                }
            },
        },
    ]
    comparison = paired_user_clustered_comparisons(reports)["rrf"]
    assert comparison["unique_historical_users"] == 2
    assert comparison["repeated_seed_observations"] == 3
    assert comparison["user_clustered_ndcg_at_20_delta"] == 0.5
