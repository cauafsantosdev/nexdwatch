import numpy as np

from app.ml.historical_interactions import UserSplit
from experiments.ranker.protocol import (
    build_user_folds,
    relevance_label_from_bucket,
    select_ranker_training_holdouts,
)


def _split(user_id: int, offset: int = 0) -> UserSplit:
    rows = np.arange(offset, offset + 12, dtype=np.int64)
    buckets = np.asarray([5, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8], dtype=np.int64)
    return UserSplit(
        cohort_user_id=user_id,
        all_item_rows=rows,
        all_rating_buckets=buckets,
        context_item_rows=rows[2:],
        context_rating_buckets=buckets[2:],
        training_positive_rows=rows[2:],
        explicit_negative_rows=np.empty(0, dtype=np.int64),
        validation_target=int(rows[0]),
        test_target=int(rows[1]),
    )


def test_user_folds_are_deterministic_disjoint_and_cover_test_once() -> None:
    splits = tuple(_split(user_id, user_id * 12) for user_id in range(50))
    film_ids = np.arange(1, 601, dtype=np.int64)
    first = build_user_folds(splits, film_ids, seed=42)
    second = build_user_folds(splits, film_ids, seed=42)
    assert first.fold_by_user == second.fold_by_user
    test_memberships: list[int] = []
    for fold in range(5):
        train, validation, test = first.partitions(fold)
        assert not train & validation
        assert not train & test
        assert not validation & test
        assert train | validation | test == set(range(50))
        test_memberships.extend(test)
    assert sorted(test_memberships) == list(range(50))


def test_training_holdouts_are_deterministic_capped_and_removed() -> None:
    split = _split(7)
    first = select_ranker_training_holdouts(split, seed=42)
    second = select_ranker_training_holdouts(split, seed=42)
    np.testing.assert_array_equal(first.item_rows, second.item_rows)
    assert len(first.item_rows) == 8
    assert not set(first.item_rows) & set(first.context_item_rows)
    assert split.validation_target not in first.item_rows
    assert split.test_target not in first.item_rows
    assert len(first.context_item_rows) + len(first.item_rows) == len(
        split.context_item_rows
    )


def test_graded_label_mapping() -> None:
    assert [relevance_label_from_bucket(bucket) for bucket in (5, 6, 7, 8, 9)] == [
        0,
        1,
        2,
        3,
        3,
    ]
