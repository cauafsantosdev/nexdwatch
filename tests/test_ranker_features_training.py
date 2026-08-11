import numpy as np
import pytest

from experiments.ranker.config import AFFINITY_FEATURES, FEATURE_NAMES
from experiments.ranker.dataset import PartitionDataset, QueryAudit
from experiments.ranker.features import feature_group_columns

pytest.importorskip("lightgbm")

from experiments.ranker.training import (
    _full_pool_global_ndcg_at_20,
    train_ranker_ablations,
)


def _dataset(user_offset: int, groups: int) -> PartitionDataset:
    group_size = 8
    rows = groups * group_size
    rng = np.random.default_rng(42 + user_offset)
    features = rng.normal(size=(rows, len(FEATURE_NAMES))).astype(np.float32)
    labels = np.tile(np.asarray([3, 2, 1, 0, 0, 0, 0, 0], dtype=np.int8), groups)
    query_ids = np.repeat(np.arange(user_offset, user_offset + groups), group_size)
    queries = tuple(
        QueryAudit(
            int(user),
            3,
            3,
            int(user * 100 + 1),
            True,
            "both",
            "HEAD",
            1,
            20,
            group_size,
            "full_candidate_inventory",
        )
        for user in range(user_offset, user_offset + groups)
    )
    return PartitionDataset(
        features=features,
        shuffled_personalized_features=features[
            :, feature_group_columns()["personalized_affinity_only"]
        ].copy(),
        labels=labels,
        query_ids=query_ids.astype(np.int64),
        film_ids=np.arange(1, rows + 1, dtype=np.int64),
        sampling_strata=np.zeros(rows, dtype=np.int8),
        baseline_scores=np.zeros((rows, 4), dtype=np.float32),
        group_sizes=np.full(groups, group_size, dtype=np.int32),
        queries=queries,
        all_queries=queries,
        eligible_user_count=groups,
        candidate_retrieved_user_count=groups,
        failed_training_group_count=0,
    )


def test_feature_schema_is_ordered_and_has_no_raw_identity() -> None:
    assert len(FEATURE_NAMES) == 115
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert not {"user_id", "film_id", "tmdb_id"} & set(FEATURE_NAMES)
    assert len(AFFINITY_FEATURES) == 48
    assert feature_group_columns() == feature_group_columns()
    no_svd_names = {
        FEATURE_NAMES[index] for index in feature_group_columns()["full_without_svd"]
    }
    assert "svd_score" not in no_svd_names
    assert "svd_rank" not in no_svd_names
    assert "retrieved_by_svd" in no_svd_names
    assert "svd_profile_available" in no_svd_names


def test_lightgbm_training_is_seed_reproducible() -> None:
    train = _dataset(0, 16)
    validation = _dataset(100, 4)
    test = _dataset(200, 4)
    first = train_ranker_ablations(
        train, validation, test, seed=42, names=("source_only",)
    )[0]
    second = train_ranker_ablations(
        train, validation, test, seed=42, names=("source_only",)
    )[0]
    np.testing.assert_allclose(first.test_scores, second.test_scores)


def test_checkpoint_metric_counts_all_zero_candidate_miss_group_as_zero() -> None:
    metric = _full_pool_global_ndcg_at_20(
        np.asarray([10, 11, 20, 21, 30, 31], dtype=np.int64),
        np.asarray([True, True, False], dtype=np.bool_),
    )
    name, value, higher_is_better = metric(
        np.asarray([0, 3, 0, 0, 0, 0], dtype=np.float32),
        np.asarray([0.0, 1.0, 1.0, 0.0, 2.0, 0.0], dtype=np.float32),
        None,
        np.asarray([2, 2, 2], dtype=np.int32),
    )
    assert name == "full_pool_global_ndcg_at_20"
    assert value == 0.5
    assert higher_is_better is True
