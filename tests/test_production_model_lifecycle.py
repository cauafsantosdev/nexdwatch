"""Coherent production snapshot and failure-safe candidate build tests."""

import json
import shutil
from datetime import UTC, datetime

import pandas as pd
import pytest

from app.domain.maintenance import TrainedModelStatistics, TrainingStatistics
from app.ml import model_lifecycle, production_training
from app.ml.model_lifecycle import build_model_bundle, decide_retraining
from app.ml.model_registry import (
    model_root,
    read_current_version,
    resolve_serving_model,
    validate_model_bundle,
)
from app.ml.production_training import (
    PreparedProductionTrainingData,
    build_production_popularity_artifact,
)


def _prepared() -> PreparedProductionTrainingData:
    rows = [
        {"user_id": user_id, "film_id": film_id, "rating": ((film_id % 10) + 1) / 2}
        for user_id in range(1, 41)
        for film_id in range(1, 41)
        if (user_id + film_id) % 3 == 0
    ]
    frame = pd.DataFrame(rows)
    now = datetime(2030, 1, 1, tzinfo=UTC)
    return PreparedProductionTrainingData(
        interactions=frame,
        statistics=TrainingStatistics(
            measured_at=now,
            eligible_user_count=int(frame["user_id"].nunique()),
            rated_interaction_count=len(frame),
            rated_film_ids=tuple(
                sorted(int(value) for value in frame["film_id"].unique())
            ),
        ),
        extraction_seconds=0.01,
    )


def test_production_popularity_uses_current_snapshot_and_frozen_formula() -> None:
    data = _prepared()
    artifact = build_production_popularity_artifact(data)
    expected = (
        data.interactions.loc[data.interactions["rating"] >= 3.5]
        .groupby("film_id")
        .size()
        .to_dict()
    )
    assert artifact.film_count == data.statistics.model_film_count
    assert [entry.film_id for entry in artifact.films] == sorted(
        data.statistics.rated_film_ids,
        key=lambda film_id: (-expected.get(film_id, 0), film_id),
    )


def test_actual_candidate_bundle_builds_and_fully_validates(tmp_path) -> None:
    version = "20300101T010203Z-1234abcd"
    result = build_model_bundle(
        _prepared(), artifact_root=tmp_path, model_version=version
    )
    validated = validate_model_bundle(result.root)
    assert validated.manifest.model_version == version
    assert validated.manifest.rated_interaction_count == len(_prepared().interactions)
    assert validated.artifacts.item_vectors.shape[1] == 32
    assert read_current_version(tmp_path) is None


@pytest.mark.parametrize(
    "failure_point",
    ["svd", "faiss", "popularity", "popularity_write", "manifest", "validation"],
)
def test_candidate_build_failure_never_changes_current_pointer(
    tmp_path, monkeypatch, failure_point
) -> None:
    models = model_root(tmp_path)
    models.mkdir(parents=True)
    old_version = "20291201T010203Z-11111111"
    pointer = models / "current.json"
    pointer.write_text(json.dumps({"model_version": old_version}), encoding="utf-8")

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"injected {failure_point} failure")

    target = {
        "svd": "train_prepared_svd",
        "popularity": "build_production_popularity_artifact",
        "popularity_write": "write_popularity_artifact",
        "manifest": "write_manifest",
        "validation": "validate_model_bundle",
    }.get(failure_point)
    if failure_point == "faiss":
        monkeypatch.setattr(production_training, "build_faiss_index", fail)
    else:
        monkeypatch.setattr(model_lifecycle, target, fail)
    with pytest.raises(RuntimeError, match="injected"):
        build_model_bundle(
            _prepared(),
            artifact_root=tmp_path,
            model_version="20300101T010203Z-22222222",
        )
    assert read_current_version(tmp_path) == old_version


def test_failed_legacy_bootstrap_keeps_complete_flat_serving_untouched(
    tmp_path, monkeypatch
) -> None:
    data = _prepared()
    seed = build_model_bundle(
        data,
        artifact_root=tmp_path,
        model_version="20300101T010203Z-11111111",
    )
    for filename in ("item_embeddings.npy", "film_index.json", "retrieval.faiss"):
        shutil.copy2(seed.root / filename, tmp_path / filename)
    (tmp_path / "candidates").mkdir()
    shutil.copy2(seed.root / "popularity.json", tmp_path / "candidates/popularity.json")
    trained = TrainedModelStatistics(
        trained_at=datetime(2029, 1, 1, tzinfo=UTC),
        eligible_user_count=-1,
        rated_interaction_count=-1,
        model_film_count=data.statistics.model_film_count,
        model_film_ids=data.statistics.rated_film_ids,
    )
    settings = type(
        "Thresholds",
        (),
        {
            "NEW_ELIGIBLE_USERS_THRESHOLD": 100,
            "NEW_MODEL_FILMS_THRESHOLD": 250,
            "MAX_MODEL_AGE_DAYS": 180,
            "ARTIFACT_ROOT": tmp_path,
            "MODEL_RETENTION_PREVIOUS": 2,
        },
    )()
    decision = decide_retraining(data.statistics, trained, settings=settings)
    monkeypatch.setattr(model_lifecycle, "evaluate_retraining", lambda **_: decision)
    monkeypatch.setattr(
        model_lifecycle, "load_production_training_data", lambda **_: data
    )
    monkeypatch.setattr(
        model_lifecycle,
        "build_model_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("bootstrap failed")
        ),
    )

    with pytest.raises(RuntimeError, match="bootstrap failed"):
        model_lifecycle.retrain_and_promote(artifact_root=tmp_path, settings=settings)
    assert read_current_version(tmp_path) is None
    assert resolve_serving_model(tmp_path).model_version == "legacy-flat"
