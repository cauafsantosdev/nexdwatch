"""Tests for controlled-popularity artifact construction and validation."""

import json

import numpy as np
import pytest

from app.ml.historical_interactions import (
    HistoricalUser,
    PreparedInteractions,
    TrainingDataSummary,
)
from app.ml.popularity import (
    POPULARITY_ARTIFACT_SCHEMA,
    POPULARITY_RATING_THRESHOLD,
    POPULARITY_SOURCE,
    PopularityArtifact,
    PopularityEntry,
    build_popularity_artifact,
    read_popularity_artifact,
    write_popularity_artifact,
)


def _prepared() -> PreparedInteractions:
    return PreparedInteractions(
        film_ids=np.array([10, 20, 30, 40], dtype=np.int64),
        users=(
            HistoricalUser(
                cohort_user_id=1,
                item_rows=np.array([0, 1, 2], dtype=np.int64),
                rating_buckets=np.array([6, 5, 9], dtype=np.int64),
            ),
            HistoricalUser(
                cohort_user_id=2,
                item_rows=np.array([0, 1, 3], dtype=np.int64),
                rating_buckets=np.array([7, 8, 1], dtype=np.int64),
            ),
        ),
        summary=TrainingDataSummary(6, 6, 0, 0, 2, 4),
    )


def test_popularity_uses_positive_counts_and_film_id_tie_breaking(tmp_path) -> None:
    artifact = build_popularity_artifact(_prepared())

    assert [
        (entry.film_id, entry.positive_interaction_count, entry.rank)
        for entry in artifact.films
    ] == [
        (10, 2, 1),
        (20, 1, 2),
        (30, 1, 3),
        (40, 0, 4),
    ]
    destination = write_popularity_artifact(artifact, tmp_path / "popularity.json")
    assert read_popularity_artifact(destination) == artifact


@pytest.mark.parametrize(
    "corruption", ["json", "duplicate", "negative", "order", "coerced_id"]
)
def test_corrupt_popularity_artifacts_are_rejected(tmp_path, corruption: str) -> None:
    path = tmp_path / "popularity.json"
    if corruption == "json":
        path.write_text("not-json", encoding="utf-8")
    else:
        films = [
            PopularityEntry(10, 2, 1),
            PopularityEntry(20, 1, 2),
        ]
        if corruption == "duplicate":
            films[1] = PopularityEntry(10, 1, 2)
        elif corruption == "negative":
            films[1] = PopularityEntry(20, -1, 2)
        elif corruption == "order":
            films = list(reversed(films))
        payload = PopularityArtifact(
            POPULARITY_ARTIFACT_SCHEMA,
            POPULARITY_RATING_THRESHOLD,
            2,
            POPULARITY_SOURCE,
            tuple(films),
        ).to_dict()
        if corruption == "coerced_id":
            payload["films"][0]["film_id"] = "10"
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        read_popularity_artifact(path)
