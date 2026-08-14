"""One coherent PostgreSQL snapshot for production SVD and popularity artifacts."""

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sqlalchemy import create_engine

from app.core.config import Settings, get_settings
from app.domain.maintenance import TrainingStatistics
from app.ml.faiss_index import FaissIndexBuildResult, build_faiss_index
from app.ml.popularity import (
    POPULARITY_ARTIFACT_SCHEMA,
    POPULARITY_RATING_THRESHOLD,
    PRODUCTION_POPULARITY_SOURCE,
    PopularityArtifact,
    PopularityEntry,
    validate_popularity_artifact,
)

logger = logging.getLogger(__name__)
SVD_DIMENSION = 32
TRAINING_SEMANTICS = "postgres-rated-deduplicated-user-film-svd32-v1"


@dataclass(frozen=True, slots=True)
class PreparedProductionTrainingData:
    interactions: pd.DataFrame
    statistics: TrainingStatistics
    extraction_seconds: float


@dataclass(frozen=True, slots=True)
class SVDTrainingResult:
    index: FaissIndexBuildResult
    training_seconds: float
    artifact_write_seconds: float
    faiss_build_seconds: float


def _sync_database_url(settings: Settings) -> str:
    return (
        "postgresql+psycopg2://"
        f"{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@"
        f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )


def load_production_training_data(
    *, settings: Settings | None = None
) -> PreparedProductionTrainingData:
    """Read and deduplicate the exact production rated-interaction universe once."""
    effective_settings = settings or get_settings()
    engine = create_engine(_sync_database_url(effective_settings))
    started = time.perf_counter()
    try:
        frame = pd.read_sql(
            "SELECT user_id, film_id, rating FROM logs "
            "WHERE rating IS NOT NULL ORDER BY user_id, film_id, id",
            engine,
        )
    finally:
        engine.dispose()
    frame = frame.drop_duplicates(subset=["user_id", "film_id"], keep="first")
    if frame.empty:
        raise ValueError("production training data contains no rated interactions")
    for column in ("user_id", "film_id"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    frame["rating"] = pd.to_numeric(frame["rating"], errors="raise")
    if not np.isfinite(frame["rating"].to_numpy(dtype=np.float64)).all():
        raise ValueError("production training ratings must be finite")
    measured_at = datetime.now(UTC)
    film_ids = tuple(sorted(int(value) for value in frame["film_id"].unique()))
    statistics = TrainingStatistics(
        measured_at=measured_at,
        eligible_user_count=int(frame["user_id"].nunique()),
        rated_interaction_count=len(frame),
        rated_film_ids=film_ids,
    )
    duration = time.perf_counter() - started
    logger.info(
        "Production training snapshot rows=%d users=%d films=%d extraction_s=%.3f",
        len(frame),
        statistics.eligible_user_count,
        statistics.model_film_count,
        duration,
    )
    return PreparedProductionTrainingData(frame, statistics, duration)


def measure_production_training_data(
    *, settings: Settings | None = None
) -> TrainingStatistics:
    """Measure retraining inputs without materializing ratings or ORM entities."""
    effective_settings = settings or get_settings()
    engine = create_engine(_sync_database_url(effective_settings))
    query = """
        WITH rated AS (
            SELECT user_id, film_id
            FROM logs
            WHERE rating IS NOT NULL
            GROUP BY user_id, film_id
        )
        SELECT
            COUNT(DISTINCT user_id) AS eligible_user_count,
            COUNT(*) AS rated_interaction_count,
            COALESCE(
                array_agg(DISTINCT film_id ORDER BY film_id),
                ARRAY[]::integer[]
            ) AS rated_film_ids
        FROM rated
    """
    try:
        frame = pd.read_sql(query, engine)
    finally:
        engine.dispose()
    row = frame.iloc[0]
    raw_film_ids = row["rated_film_ids"]
    return TrainingStatistics(
        measured_at=datetime.now(UTC),
        eligible_user_count=int(row["eligible_user_count"]),
        rated_interaction_count=int(row["rated_interaction_count"]),
        rated_film_ids=tuple(int(value) for value in raw_film_ids),
    )


def train_prepared_svd(
    data: PreparedProductionTrainingData, output_dir: str | Path
) -> SVDTrainingResult:
    """Train the frozen 32-dimensional SVD and exact FAISS artifacts."""
    started = time.perf_counter()
    matrix = data.interactions.pivot(
        index="user_id", columns="film_id", values="rating"
    ).fillna(0)
    film_ids = [int(value) for value in matrix.columns]
    svd = TruncatedSVD(n_components=SVD_DIMENSION, random_state=42)
    svd.fit(matrix)
    item_vectors = normalize(svd.components_.T, axis=1)
    training_seconds = time.perf_counter() - started

    write_started = time.perf_counter()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "item_embeddings.npy", item_vectors)
    with (destination / "film_index.json").open("w", encoding="utf-8") as stream:
        json.dump(film_ids, stream)
    artifact_write_seconds = time.perf_counter() - write_started
    faiss_started = time.perf_counter()
    index_result = build_faiss_index(
        item_vectors, film_ids, destination / "retrieval.faiss"
    )
    return SVDTrainingResult(
        index=index_result,
        training_seconds=training_seconds,
        artifact_write_seconds=artifact_write_seconds,
        faiss_build_seconds=time.perf_counter() - faiss_started,
    )


def build_production_popularity_artifact(
    data: PreparedProductionTrainingData,
) -> PopularityArtifact:
    """Build frozen-formula popularity from the same production snapshot as SVD."""
    counts = (
        data.interactions.loc[
            data.interactions["rating"] >= POPULARITY_RATING_THRESHOLD
        ]
        .groupby("film_id")
        .size()
        .to_dict()
    )
    ordered = sorted(
        data.statistics.rated_film_ids,
        key=lambda film_id: (-int(counts.get(film_id, 0)), film_id),
    )
    return validate_popularity_artifact(
        PopularityArtifact(
            schema=POPULARITY_ARTIFACT_SCHEMA,
            rating_threshold=POPULARITY_RATING_THRESHOLD,
            film_count=len(ordered),
            source_description=PRODUCTION_POPULARITY_SOURCE,
            films=tuple(
                PopularityEntry(
                    film_id=film_id,
                    positive_interaction_count=int(counts.get(film_id, 0)),
                    rank=rank,
                )
                for rank, film_id in enumerate(ordered, start=1)
            ),
        )
    )
