"""Controlled historical-popularity artifact construction and validation."""

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.ml.historical_interactions import PreparedInteractions
from app.ml.ratings import rating_to_bucket

POPULARITY_ARTIFACT_SCHEMA = 1
POPULARITY_RATING_THRESHOLD = 3.5
POPULARITY_SOURCE = "resolved positive interactions from controlled data/users_data.csv"


@dataclass(frozen=True, slots=True)
class PopularityEntry:
    """One resolved catalog film's deterministic controlled-popularity rank."""

    film_id: int
    positive_interaction_count: int
    rank: int


@dataclass(frozen=True, slots=True)
class PopularityArtifact:
    """Validated popularity metadata and rank-ordered entries."""

    schema: int
    rating_threshold: float
    film_count: int
    source_description: str
    films: tuple[PopularityEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "rating_threshold": self.rating_threshold,
            "film_count": self.film_count,
            "source_description": self.source_description,
            "films": [asdict(entry) for entry in self.films],
        }


def build_popularity_artifact(data: PreparedInteractions) -> PopularityArtifact:
    """Rank resolved films by positive controlled-cohort interaction count."""
    counts = np.zeros(len(data.film_ids), dtype=np.int64)
    positive_bucket = rating_to_bucket(POPULARITY_RATING_THRESHOLD)
    for user in data.users:
        positive_rows = user.item_rows[user.rating_buckets >= positive_bucket]
        if len(positive_rows):
            np.add.at(counts, positive_rows, 1)
    ordered_rows = np.lexsort((data.film_ids, -counts))
    films = tuple(
        PopularityEntry(
            film_id=int(data.film_ids[row]),
            positive_interaction_count=int(counts[row]),
            rank=rank,
        )
        for rank, row in enumerate(ordered_rows, start=1)
    )
    return validate_popularity_artifact(
        PopularityArtifact(
            schema=POPULARITY_ARTIFACT_SCHEMA,
            rating_threshold=POPULARITY_RATING_THRESHOLD,
            film_count=len(films),
            source_description=POPULARITY_SOURCE,
            films=films,
        )
    )


def validate_popularity_artifact(
    artifact: PopularityArtifact,
) -> PopularityArtifact:
    """Reject incompatible, corrupt, or nondeterministically ordered artifacts."""
    if artifact.schema != POPULARITY_ARTIFACT_SCHEMA:
        raise ValueError("unsupported popularity artifact schema")
    if artifact.rating_threshold != POPULARITY_RATING_THRESHOLD:
        raise ValueError("unexpected popularity rating threshold")
    if artifact.source_description != POPULARITY_SOURCE:
        raise ValueError("unexpected popularity source description")
    if artifact.film_count <= 0 or artifact.film_count != len(artifact.films):
        raise ValueError("popularity artifact film count is invalid")
    film_ids = [entry.film_id for entry in artifact.films]
    if any(film_id <= 0 for film_id in film_ids) or len(set(film_ids)) != len(film_ids):
        raise ValueError("popularity artifact film IDs must be unique positive IDs")
    if [entry.rank for entry in artifact.films] != list(
        range(1, len(artifact.films) + 1)
    ):
        raise ValueError("popularity ranks must be contiguous and one-based")
    if any(entry.positive_interaction_count < 0 for entry in artifact.films):
        raise ValueError("popularity counts must be non-negative")
    expected = sorted(
        artifact.films,
        key=lambda entry: (-entry.positive_interaction_count, entry.film_id),
    )
    if list(artifact.films) != expected:
        raise ValueError("popularity entries are not deterministically ordered")
    return artifact


def popularity_artifact_from_dict(
    payload: Mapping[str, Any],
) -> PopularityArtifact:
    """Parse and validate a JSON-compatible popularity payload."""
    try:
        films_payload = payload["films"]
        if not isinstance(films_payload, list):
            raise TypeError
        artifact = PopularityArtifact(
            schema=_strict_json_integer(payload["schema"]),
            rating_threshold=float(payload["rating_threshold"]),
            film_count=_strict_json_integer(payload["film_count"]),
            source_description=str(payload["source_description"]),
            films=tuple(
                PopularityEntry(
                    film_id=_strict_json_integer(entry["film_id"]),
                    positive_interaction_count=_strict_json_integer(
                        entry["positive_interaction_count"]
                    ),
                    rank=_strict_json_integer(entry["rank"]),
                )
                for entry in films_payload
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid popularity artifact payload") from exc
    return validate_popularity_artifact(artifact)


def _strict_json_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("popularity integer fields must be JSON integers")
    return value


def read_popularity_artifact(path: str | Path) -> PopularityArtifact:
    """Load and validate a popularity artifact from disk."""
    try:
        with Path(path).open(encoding="utf-8") as artifact_file:
            payload = json.load(artifact_file)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("popularity artifact is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError("popularity artifact root must be an object")
    return popularity_artifact_from_dict(payload)


def write_popularity_artifact(
    artifact: PopularityArtifact,
    path: str | Path,
) -> Path:
    """Validate and atomically write the controlled-popularity artifact."""
    validated = validate_popularity_artifact(artifact)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as artifact_file:
            json.dump(validated.to_dict(), artifact_file, indent=2)
            artifact_file.write("\n")
            artifact_file.flush()
            os.fsync(artifact_file.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination
