from types import SimpleNamespace

import numpy as np
import pytest

from experiments.category_policy.evaluate import (
    _context_history,
    _target_personalized_score,
)


def test_held_out_target_never_constructs_category_profile_evidence() -> None:
    example = SimpleNamespace(
        context_item_rows=np.asarray([0, 2], dtype=np.int64),
        context_rating_buckets=np.asarray([9, 6], dtype=np.int64),
        designated_target_id=20,
    )

    history = _context_history(example, np.asarray([10, 20, 30], dtype=np.int64))

    assert history.watched_film_ids == (10, 30)
    assert [value.film_id for value in history.rated_interactions] == [10, 30]
    assert [value.rating for value in history.rated_interactions] == [5.0, 3.5]


def test_held_out_evaluator_rejects_a_leaked_target() -> None:
    example = SimpleNamespace(
        context_item_rows=np.asarray([0, 1], dtype=np.int64),
        context_rating_buckets=np.asarray([9, 8], dtype=np.int64),
        designated_target_id=20,
    )

    with pytest.raises(RuntimeError, match="held-out target leaked"):
        _context_history(example, np.asarray([10, 20], dtype=np.int64))


def test_target_personalization_score_uses_context_only() -> None:
    example = SimpleNamespace(
        context_item_rows=np.asarray([0], dtype=np.int64),
        context_rating_buckets=np.asarray([9], dtype=np.int64),
        designated_target_id=20,
    )
    vectors = np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)

    score = _target_personalized_score(example, vectors, {10: 0, 20: 1})

    assert score == pytest.approx(0.5)
