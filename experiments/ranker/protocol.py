"""Strict out-of-user folds and ranker-only training holdouts."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.ml.historical_interactions import UserSplit
from app.ml.ratings import rating_to_bucket
from experiments.evaluation import training_positive_counts
from experiments.ranker.config import (
    NEGATIVE_RATING_THRESHOLD,
    POSITIVE_RATING_THRESHOLD,
    TRAINING_HOLDOUT_LIMIT,
    USER_FOLD_COUNT,
)
from experiments.retrieval.candidate_analysis import assign_popularity_strata


@dataclass(frozen=True, slots=True)
class UserFoldAssignment:
    """One deterministic partition of every historical user."""

    fold_by_user: dict[int, int]
    history_depth_thresholds: tuple[float, float, float]
    target_stratum_by_user: dict[int, str]

    def partitions(self, fold: int) -> tuple[set[int], set[int], set[int]]:
        """Return disjoint train, validation, and test users for one fold."""
        if fold < 0 or fold >= USER_FOLD_COUNT:
            raise ValueError("ranker fold is out of range")
        test = {
            user_id
            for user_id, assigned in self.fold_by_user.items()
            if assigned == fold
        }
        validation_fold = (fold + 1) % USER_FOLD_COUNT
        validation = {
            user_id
            for user_id, assigned in self.fold_by_user.items()
            if assigned == validation_fold
        }
        training = set(self.fold_by_user) - test - validation
        return training, validation, test


@dataclass(frozen=True, slots=True)
class RankerTrainingHoldouts:
    """Additional positive labels hidden only for a ranker-training user."""

    item_rows: NDArray[np.int64]
    rating_buckets: NDArray[np.int64]
    context_item_rows: NDArray[np.int64]
    context_rating_buckets: NDArray[np.int64]


def relevance_label_from_bucket(rating_bucket: int) -> int:
    """Map a half-star bucket to the frozen graded ranking label."""
    rating = (int(rating_bucket) + 1) / 2.0
    if rating < 3.5:
        return 0
    if rating == 3.5:
        return 1
    if rating == 4.0:
        return 2
    return 3


def build_user_folds(
    splits: tuple[UserSplit, ...],
    film_ids: NDArray[np.int64],
    *,
    seed: int,
) -> UserFoldAssignment:
    """Stratify deterministic user folds by depth and canonical target stratum."""
    if not splits:
        raise ValueError("ranker folds require users")
    counts = training_positive_counts(splits, len(film_ids))
    strata, _ = assign_popularity_strata(counts, film_ids)
    depths = np.asarray([len(split.context_item_rows) for split in splits], dtype=float)
    thresholds = tuple(float(value) for value in np.quantile(depths, (0.25, 0.5, 0.75)))
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    target_stratum_by_user: dict[int, str] = {}
    for split, depth in zip(splits, depths, strict=True):
        depth_bucket = int(np.searchsorted(thresholds, depth, side="right"))
        target_stratum = (
            str(strata[split.test_target]) if split.test_target is not None else "NONE"
        )
        target_stratum_by_user[split.cohort_user_id] = target_stratum
        grouped[(depth_bucket, target_stratum)].append(split.cohort_user_id)

    fold_by_user: dict[int, int] = {}
    for group_index, key in enumerate(sorted(grouped)):
        user_ids = np.asarray(sorted(grouped[key]), dtype=np.int64)
        rng = _rng(seed, group_index, len(user_ids))
        shuffled = user_ids[rng.permutation(len(user_ids))]
        offset = int(rng.integers(0, USER_FOLD_COUNT))
        for position, user_id in enumerate(shuffled):
            fold_by_user[int(user_id)] = (position + offset) % USER_FOLD_COUNT
    if len(fold_by_user) != len(splits):
        raise RuntimeError("ranker fold assignment lost users")
    return UserFoldAssignment(
        fold_by_user=fold_by_user,
        history_depth_thresholds=thresholds,
        target_stratum_by_user=target_stratum_by_user,
    )


def select_ranker_training_holdouts(
    split: UserSplit,
    *,
    seed: int,
    limit: int = TRAINING_HOLDOUT_LIMIT,
) -> RankerTrainingHoldouts:
    """Hide up to eight context positives with deterministic grade representation."""
    if limit < 0:
        raise ValueError("ranker holdout limit must be non-negative")
    positive_bucket = rating_to_bucket(POSITIVE_RATING_THRESHOLD)
    candidates_by_label: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row, bucket in zip(
        split.context_item_rows,
        split.context_rating_buckets,
        strict=True,
    ):
        if int(bucket) < positive_bucket:
            continue
        label = relevance_label_from_bucket(int(bucket))
        candidates_by_label[label].append((int(row), int(bucket)))
    for label, candidates in candidates_by_label.items():
        rng = _rng(seed, split.cohort_user_id, label)
        order = rng.permutation(len(candidates))
        candidates_by_label[label] = [candidates[index] for index in order]

    selected: list[tuple[int, int]] = []
    while len(selected) < limit and any(candidates_by_label.values()):
        for label in (1, 2, 3):
            if candidates_by_label[label] and len(selected) < limit:
                selected.append(candidates_by_label[label].pop())
    selected_rows = {row for row, _ in selected}
    if split.validation_target in selected_rows or split.test_target in selected_rows:
        raise RuntimeError("canonical target was selected as a ranker holdout")
    keep = np.asarray(
        [int(row) not in selected_rows for row in split.context_item_rows],
        dtype=np.bool_,
    )
    return RankerTrainingHoldouts(
        item_rows=np.ascontiguousarray([row for row, _ in selected], dtype=np.int64),
        rating_buckets=np.ascontiguousarray(
            [bucket for _, bucket in selected], dtype=np.int64
        ),
        context_item_rows=np.ascontiguousarray(split.context_item_rows[keep]),
        context_rating_buckets=np.ascontiguousarray(split.context_rating_buckets[keep]),
    )


def negative_rating_bucket() -> int:
    """Expose the frozen explicit-negative cutoff for feature builders."""
    return rating_to_bucket(NEGATIVE_RATING_THRESHOLD)


def _rng(*values: int) -> np.random.Generator:
    components = [int(value) & 0xFFFFFFFF for value in values]
    return np.random.default_rng(np.random.SeedSequence(components))
