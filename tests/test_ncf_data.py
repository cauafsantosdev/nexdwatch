"""Tests for controlled neural training data and sampling."""

from pathlib import Path

import pytest

from app.ml.historical_interactions import (
    build_interaction_splits,
    load_historical_interactions,
)
from experiments.neural_retrieval.data import iter_training_examples


def _write_csv(path: Path, rows: list[str]) -> Path:
    path.write_text(
        "user_id;username;slug;rating\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_semicolon_csv_resolves_in_batch_skips_unknown_and_collapses_duplicates(
    tmp_path,
) -> None:
    source = _write_csv(
        tmp_path / "users_data.csv",
        [
            "1;one;a;5.0",
            "1;one;a;5.0",
            "1;one;b;3.0",
            "2;two;missing;4.0",
            "2;two;c;1.0",
        ],
    )

    data = load_historical_interactions(source, {"a": 10, "b": 20, "c": 30})

    assert data.film_ids.tolist() == [10, 20, 30]
    assert data.summary.csv_rows == 5
    assert data.summary.resolved_rows == 4
    assert data.summary.unresolved_rows == 1
    assert data.summary.duplicate_rows == 1
    assert data.summary.unique_users == 2
    assert sum(len(user.item_rows) for user in data.users) == 3


def test_conflicting_duplicate_ratings_fail(tmp_path) -> None:
    source = _write_csv(
        tmp_path / "users_data.csv",
        ["1;one;a;5.0", "1;one;a;4.5"],
    )
    with pytest.raises(ValueError, match="conflicting ratings"):
        load_historical_interactions(source, {"a": 10})


@pytest.mark.parametrize("rating", ["", "2.25", "6", "bad"])
def test_invalid_or_missing_ratings_fail(tmp_path, rating: str) -> None:
    source = _write_csv(tmp_path / "users_data.csv", [f"1;one;a;{rating}"])
    with pytest.raises(ValueError):
        load_historical_interactions(source, {"a": 10})


def test_split_is_deterministic_and_held_out_targets_do_not_leak(tmp_path) -> None:
    source = _write_csv(
        tmp_path / "users_data.csv",
        [
            "1;one;a;5.0",
            "1;one;b;4.5",
            "1;one;c;4.0",
            "1;one;d;3.5",
            "1;one;e;3.0",
            "1;one;f;2.5",
            "1;one;g;1.0",
        ],
    )
    mapping = {slug: index for index, slug in enumerate("abcdefg", start=10)}
    data = load_historical_interactions(source, mapping)

    first = build_interaction_splits(
        data,
        positive_rating_threshold=3.5,
        negative_rating_threshold=2.5,
        seed=42,
    )[0]
    second = build_interaction_splits(
        data,
        positive_rating_threshold=3.5,
        negative_rating_threshold=2.5,
        seed=42,
    )[0]

    assert first.validation_target == second.validation_target
    assert first.test_target == second.test_target
    assert first.validation_target not in first.context_item_rows
    assert first.test_target not in first.context_item_rows
    assert first.validation_target not in first.training_positive_rows
    assert first.test_target not in first.training_positive_rows
    assert len(first.training_positive_rows) == 2
    assert len(first.explicit_negative_rows) == 2


def test_sampling_caps_targets_and_context_and_excludes_known_random_negatives(
    tmp_path,
) -> None:
    rows = [
        f"1;one;film-{index};{5.0 if index < 8 else 1.0}" for index in range(12)
    ] + [f"2;two;film-{index};4.0" for index in range(12, 20)]
    source = _write_csv(tmp_path / "users_data.csv", rows)
    mapping = {f"film-{index}": index + 1 for index in range(20)}
    data = load_historical_interactions(source, mapping)
    split = build_interaction_splits(
        data,
        positive_rating_threshold=3.5,
        negative_rating_threshold=2.5,
        seed=42,
    )

    examples = list(
        iter_training_examples(
            split,
            item_count=len(data.film_ids),
            epoch=1,
            seed=42,
            targets_per_user=3,
            max_context_items=4,
            negatives_per_positive=4,
        )
    )

    assert len(examples) == 6
    assert all(
        sum(example.cohort_user_id == user_id for example in examples) <= 3
        for user_id in (1, 2)
    )
    known = {int(value) for value in split[0].all_item_rows}
    explicit = {int(value) for value in split[0].explicit_negative_rows}
    for example in [item for item in examples if item.cohort_user_id == 1]:
        assert len(example.context_rows) <= 4
        assert example.positive_row not in example.context_rows
        assert {int(value) for value in example.negative_rows[2:]}.isdisjoint(known)
        assert {int(value) for value in example.negative_rows[:2]}.issubset(explicit)


def test_training_data_requires_existing_controlled_csv(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="controlled training dataset"):
        load_historical_interactions(tmp_path / "missing.csv", {"a": 1})


def test_each_film_id_maps_to_one_embedding_row(tmp_path) -> None:
    source = _write_csv(tmp_path / "users_data.csv", ["1;one;a;5.0"])
    with pytest.raises(ValueError, match="exactly one slug"):
        load_historical_interactions(source, {"a": 1, "alias": 1})
