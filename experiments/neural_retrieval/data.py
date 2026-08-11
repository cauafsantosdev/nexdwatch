"""Deterministic example construction for neural retrieval training."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.ml.historical_interactions import UserSplit


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One history-derived neural pairwise-ranking example."""

    cohort_user_id: int
    context_rows: NDArray[np.int64]
    context_ratings: NDArray[np.int64]
    positive_row: int
    negative_rows: NDArray[np.int64]


def iter_training_examples(
    splits: tuple[UserSplit, ...],
    *,
    item_count: int,
    epoch: int,
    seed: int,
    targets_per_user: int,
    max_context_items: int,
    negatives_per_positive: int,
) -> Iterator[TrainingExample]:
    """Yield bounded deterministic examples for one neural training epoch."""
    for user in splits:
        if not len(user.training_positive_rows):
            continue
        rng = _rng(seed, epoch, user.cohort_user_id)
        known_rows = {int(value) for value in user.all_item_rows}
        target_count = min(targets_per_user, len(user.training_positive_rows))
        target_indexes = rng.choice(
            len(user.training_positive_rows), size=target_count, replace=False
        )
        for target_index in target_indexes:
            target = int(user.training_positive_rows[target_index])
            context_mask = user.context_item_rows != target
            context_rows = user.context_item_rows[context_mask]
            context_ratings = user.context_rating_buckets[context_mask]
            if not len(context_rows):
                continue
            if len(context_rows) > max_context_items:
                selected = rng.choice(
                    len(context_rows), size=max_context_items, replace=False
                )
                context_rows = context_rows[selected]
                context_ratings = context_ratings[selected]
            negatives = _sample_negatives(
                rng,
                item_count=item_count,
                known_rows=user.all_item_rows,
                known_row_set=known_rows,
                explicit_rows=user.explicit_negative_rows,
                count=negatives_per_positive,
            )
            if len(negatives) != negatives_per_positive:
                continue
            yield TrainingExample(
                cohort_user_id=user.cohort_user_id,
                context_rows=np.ascontiguousarray(context_rows),
                context_ratings=np.ascontiguousarray(context_ratings),
                positive_row=target,
                negative_rows=negatives,
            )


def make_evaluation_example(
    user: UserSplit,
    *,
    target: int,
    item_count: int,
    seed: int,
    negative_count: int,
) -> TrainingExample | None:
    """Create a deterministic sampled-ranking validation example."""
    if not len(user.context_item_rows):
        return None
    negatives = _sample_negatives(
        _rng(seed, user.cohort_user_id, target),
        item_count=item_count,
        known_rows=user.all_item_rows,
        known_row_set=None,
        explicit_rows=user.explicit_negative_rows,
        count=negative_count,
    )
    if len(negatives) != negative_count:
        return None
    return TrainingExample(
        cohort_user_id=user.cohort_user_id,
        context_rows=user.context_item_rows,
        context_ratings=user.context_rating_buckets,
        positive_row=target,
        negative_rows=negatives,
    )


def _sample_negatives(
    rng: np.random.Generator,
    *,
    item_count: int,
    known_rows: NDArray[np.int64],
    known_row_set: set[int] | None,
    explicit_rows: NDArray[np.int64],
    count: int,
) -> NDArray[np.int64]:
    explicit_count = min(len(explicit_rows), count // 2)
    selected: list[int] = []
    if explicit_count:
        selected.extend(
            int(value)
            for value in rng.choice(
                explicit_rows,
                size=explicit_count,
                replace=len(explicit_rows) < explicit_count,
            )
        )

    unseen_needed = count - len(selected)
    known = known_row_set or {int(value) for value in known_rows}
    unseen: list[int] = []
    seen_unseen: set[int] = set()
    for _ in range(4):
        if len(unseen) >= unseen_needed:
            break
        draw_count = max(32, (unseen_needed - len(unseen)) * 4)
        for raw_value in rng.integers(0, item_count, size=draw_count):
            value = int(raw_value)
            if value not in known and value not in seen_unseen:
                unseen.append(value)
                seen_unseen.add(value)
                if len(unseen) == unseen_needed:
                    break
    if len(unseen) < unseen_needed:
        remaining = np.setdiff1d(
            np.arange(item_count, dtype=np.int64), known_rows, assume_unique=False
        )
        if len(remaining):
            extra = rng.choice(
                remaining,
                size=unseen_needed - len(unseen),
                replace=len(remaining) < unseen_needed - len(unseen),
            )
            unseen.extend(int(value) for value in extra)
    selected.extend(unseen[:unseen_needed])

    if len(selected) < count and selected:
        selected.extend(
            int(value)
            for value in rng.choice(
                np.asarray(selected, dtype=np.int64),
                size=count - len(selected),
                replace=True,
            )
        )
    return np.ascontiguousarray(selected[:count], dtype=np.int64)


def _rng(*values: int) -> np.random.Generator:
    components = [int(value) & 0xFFFFFFFF for value in values]
    return np.random.default_rng(np.random.SeedSequence(components))
