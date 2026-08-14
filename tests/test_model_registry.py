"""Complete-bundle validation, atomic promotion, retention, and rollback."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from app.ml.faiss_index import build_faiss_index
from app.ml.model_registry import (
    MANIFEST_FILENAME,
    MODEL_ARTIFACT_FILENAMES,
    apply_model_retention,
    create_manifest,
    list_valid_model_bundles,
    model_root,
    promote_model_bundle,
    read_current_version,
    resolve_serving_model,
    rollback_model,
    select_rollback_target,
    sha256_file,
    validate_model_bundle,
    write_manifest,
)
from app.ml.popularity import (
    POPULARITY_ARTIFACT_SCHEMA,
    POPULARITY_RATING_THRESHOLD,
    PRODUCTION_POPULARITY_SOURCE,
    PopularityArtifact,
    PopularityEntry,
    write_popularity_artifact,
)
from app.services.candidate_generation_service import CandidateGenerationService
from app.services.recommendation_service import RecommendationService


def _version(day: int) -> str:
    return f"203001{day:02d}T010203Z-{day:08x}"


def _bundle(
    base: Path, day: int, *, ids: tuple[int, ...] = tuple(range(1, 41))
) -> Path:
    version = _version(day)
    root = model_root(base) / version
    root.mkdir(parents=True)
    rng = np.random.default_rng(day)
    vectors = rng.normal(size=(len(ids), 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(root / "item_embeddings.npy", vectors)
    (root / "film_index.json").write_text(json.dumps(ids), encoding="utf-8")
    build_faiss_index(vectors, ids, root / "retrieval.faiss")
    popularity = PopularityArtifact(
        schema=POPULARITY_ARTIFACT_SCHEMA,
        rating_threshold=POPULARITY_RATING_THRESHOLD,
        film_count=len(ids),
        source_description=PRODUCTION_POPULARITY_SOURCE,
        films=tuple(
            PopularityEntry(film_id, 0, rank)
            for rank, film_id in enumerate(sorted(ids), start=1)
        ),
    )
    write_popularity_artifact(popularity, root / "popularity.json")
    trained_at = datetime(2030, 1, day, 1, 2, 3, tzinfo=UTC)
    manifest = create_manifest(
        bundle_root=root,
        model_version=version,
        trained_at=trained_at,
        snapshot_measured_at=trained_at - timedelta(seconds=2),
        eligible_user_count=100,
        rated_interaction_count=1_000,
        film_count=len(ids),
    )
    write_manifest(manifest, root / MANIFEST_FILENAME)
    return root


def _refresh_checksum(root: Path, filename: str) -> None:
    path = root / MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifacts"][filename] = sha256_file(root / filename)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_complete_bundle_validates_and_serving_resolves_one_version(tmp_path) -> None:
    root = _bundle(tmp_path, 1)
    validated = validate_model_bundle(root)
    promotion = promote_model_bundle(tmp_path, root.name)
    location = resolve_serving_model(tmp_path)
    assert validated.manifest.model_version == root.name
    assert promotion.previous_version is None
    assert promotion.serving_reload_required
    assert location.root == root
    assert location.popularity_path.parent == root


@pytest.mark.parametrize("missing", MODEL_ARTIFACT_FILENAMES)
def test_missing_bundle_artifact_is_rejected(tmp_path, missing) -> None:
    root = _bundle(tmp_path, 1)
    (root / missing).unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_model_bundle(root)


def test_checksum_mismatch_and_corrupt_numpy_are_rejected(tmp_path) -> None:
    root = _bundle(tmp_path, 1)
    (root / "item_embeddings.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_model_bundle(root)
    _refresh_checksum(root, "item_embeddings.npy")
    with pytest.raises((OSError, ValueError)):
        validate_model_bundle(root)


def test_mapping_count_mismatch_is_rejected_after_checksum_validation(tmp_path) -> None:
    root = _bundle(tmp_path, 1)
    (root / "film_index.json").write_text(
        json.dumps(list(range(1, 40))), encoding="utf-8"
    )
    _refresh_checksum(root, "film_index.json")
    with pytest.raises(ValueError, match="rows and film ID count differ"):
        validate_model_bundle(root)


def test_faiss_id_order_must_exactly_match_mapping(tmp_path) -> None:
    root = _bundle(tmp_path, 1)
    vectors = np.load(root / "item_embeddings.npy", allow_pickle=False)
    build_faiss_index(vectors, tuple(range(40, 0, -1)), root / "retrieval.faiss")
    _refresh_checksum(root, "retrieval.faiss")
    with pytest.raises(ValueError, match="ordering"):
        validate_model_bundle(root)


def test_corrupt_popularity_and_invalid_manifest_are_rejected(tmp_path) -> None:
    root = _bundle(tmp_path, 1)
    payload = json.loads((root / "popularity.json").read_text(encoding="utf-8"))
    payload["films"][0]["rank"] = 2
    (root / "popularity.json").write_text(json.dumps(payload), encoding="utf-8")
    _refresh_checksum(root, "popularity.json")
    with pytest.raises(ValueError, match="ranks"):
        validate_model_bundle(root)

    other = _bundle(tmp_path, 2)
    manifest = json.loads((other / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    manifest["svd_dimension"] = 31
    (other / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dimension"):
        validate_model_bundle(other)


def test_promotion_failure_leaves_old_pointer_authoritative(
    tmp_path, monkeypatch
) -> None:
    first = _bundle(tmp_path, 1)
    second = _bundle(tmp_path, 2)
    promote_model_bundle(tmp_path, first.name)

    from app.ml import model_registry

    real_replace = model_registry.os.replace

    def fail_current(source, destination):
        if Path(destination).name == "current.json":
            raise OSError("injected promotion failure")
        return real_replace(source, destination)

    monkeypatch.setattr(model_registry.os, "replace", fail_current)
    with pytest.raises(OSError, match="injected"):
        promote_model_bundle(tmp_path, second.name)
    assert read_current_version(tmp_path) == first.name


def test_rollback_and_retention_preserve_current_and_two_candidates(tmp_path) -> None:
    roots = [_bundle(tmp_path, day) for day in range(1, 5)]
    promote_model_bundle(tmp_path, roots[-1].name)
    removed = apply_model_retention(tmp_path, keep_previous=2)
    assert roots[0].name in removed
    assert read_current_version(tmp_path) == roots[-1].name
    assert len(list_valid_model_bundles(tmp_path)) == 3
    result = rollback_model(tmp_path)
    assert result.model_version == roots[-2].name
    assert read_current_version(tmp_path) == roots[-2].name


def test_repeated_rollback_moves_strictly_older_and_ignores_newer(tmp_path) -> None:
    roots = [_bundle(tmp_path, day) for day in range(1, 5)]
    promote_model_bundle(tmp_path, roots[-1].name)

    assert rollback_model(tmp_path).model_version == roots[2].name
    assert select_rollback_target(tmp_path).model_version == roots[1].name
    assert rollback_model(tmp_path).model_version == roots[1].name
    assert rollback_model(tmp_path).model_version == roots[0].name
    pointer_before = read_current_version(tmp_path)
    with pytest.raises(LookupError, match="no previous valid"):
        rollback_model(tmp_path)
    assert read_current_version(tmp_path) == pointer_before


def test_rollback_skips_invalid_older_bundle(tmp_path) -> None:
    roots = [_bundle(tmp_path, day) for day in range(1, 4)]
    promote_model_bundle(tmp_path, roots[-1].name)
    (roots[1] / "item_embeddings.npy").write_bytes(b"invalid")

    assert select_rollback_target(tmp_path).model_version == roots[0].name
    result = rollback_model(tmp_path)
    assert result.model_version == roots[0].name


def test_incomplete_build_directory_is_never_listed_or_resolved(tmp_path) -> None:
    incomplete = model_root(tmp_path) / f".building-{_version(1)}"
    incomplete.mkdir(parents=True)
    (incomplete / "item_embeddings.npy").write_bytes(b"partial")
    assert list_valid_model_bundles(tmp_path) == []
    location = resolve_serving_model(tmp_path, validate=False)
    assert not location.versioned
    assert location.model_version == "legacy-flat"


def test_promote_reload_and_rollback_keep_both_serving_sources_coherent(
    tmp_path,
) -> None:
    first = _bundle(tmp_path, 1)
    second = _bundle(tmp_path, 2)
    promote_model_bundle(tmp_path, first.name)
    first_location = resolve_serving_model(tmp_path)
    legacy = RecommendationService(artifact_root=first_location.root)
    candidates = CandidateGenerationService(
        artifact_root=first_location.root,
        popularity_path=first_location.popularity_path,
    )
    assert legacy.load_artifacts()
    assert candidates.load_artifacts()
    first_vectors = legacy._artifacts.item_vectors.copy()

    promote_model_bundle(tmp_path, second.name)
    # Promotion cannot mutate already loaded API resources.
    assert np.array_equal(legacy._artifacts.item_vectors, first_vectors)
    legacy.unload_artifacts()
    candidates.unload_artifacts()

    second_location = resolve_serving_model(tmp_path)
    reloaded_legacy = RecommendationService(artifact_root=second_location.root)
    reloaded_candidates = CandidateGenerationService(
        artifact_root=second_location.root,
        popularity_path=second_location.popularity_path,
    )
    assert reloaded_legacy.load_artifacts()
    assert reloaded_candidates.load_artifacts()
    assert not np.array_equal(reloaded_legacy._artifacts.item_vectors, first_vectors)

    rolled_back = rollback_model(tmp_path)
    assert rolled_back.model_version == first.name
    assert resolve_serving_model(tmp_path).model_version == first.name
