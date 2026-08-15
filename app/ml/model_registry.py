"""Versioned recommendation bundles, validation, promotion, and rollback."""

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from app.domain.maintenance import PromotionResult, TrainedModelStatistics
from app.ml.faiss_index import get_faiss_ids
from app.ml.popularity import (
    POPULARITY_RATING_THRESHOLD,
    PRODUCTION_POPULARITY_SOURCE,
    read_popularity_artifact,
)
from app.ml.svd_artifacts import SVDArtifacts, load_svd_artifacts

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA = 1
SVD_DIMENSION = 32
TRAINING_SEMANTICS = "postgres-rated-deduplicated-user-film-svd32-v1"
MANIFEST_FILENAME = "manifest.json"
CURRENT_POINTER_FILENAME = "current.json"
MODEL_ARTIFACT_FILENAMES = (
    "item_embeddings.npy",
    "film_index.json",
    "retrieval.faiss",
    "popularity.json",
)
_VERSION_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}$")


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Immutable identity, training statistics, compatibility, and checksums."""

    schema: int
    model_version: str
    created_at: str
    trained_at: str
    svd_dimension: int
    film_count: int
    eligible_user_count: int
    rated_interaction_count: int
    popularity_threshold: float
    artifacts: dict[str, str]
    training_semantics: str
    python_version: str
    sklearn_version: str
    faiss_version: str
    snapshot_measured_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible manifest representation."""
        return asdict(self)

    @property
    def trained_statistics(self) -> TrainedModelStatistics:
        """Project manifest counters into retraining-decision state."""
        return TrainedModelStatistics(
            trained_at=_parse_utc(self.trained_at),
            eligible_user_count=self.eligible_user_count,
            rated_interaction_count=self.rated_interaction_count,
            model_film_count=self.film_count,
            model_film_ids=(),
        )


@dataclass(frozen=True, slots=True)
class ValidatedModelBundle:
    """One fully checked bundle and its loaded SVD/FAISS resources."""

    root: Path
    manifest: ModelManifest
    artifacts: SVDArtifacts


@dataclass(frozen=True, slots=True)
class ServingModelLocation:
    """Resolved flat or versioned location shared by both serving services."""

    root: Path
    model_version: str
    popularity_path: Path
    versioned: bool
    manifest: ModelManifest | None = None


@dataclass(frozen=True, slots=True)
class ModelPointer:
    """Authoritative selected version plus activation-recovery lineage."""

    model_version: str
    previous_version: str | None = None


def model_root(artifact_root: str | Path) -> Path:
    """Return the immutable versioned-bundle directory below an artifact root."""
    return Path(artifact_root) / "models"


def new_model_version(now: datetime | None = None) -> str:
    """Create a sortable timestamped version identifier with collision entropy."""
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _validate_version(value: object) -> str:
    if not isinstance(value, str) or not _VERSION_PATTERN.fullmatch(value):
        raise ValueError("invalid model version identifier")
    return value


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("manifest timestamps must include a timezone")
    return parsed.astimezone(UTC)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def sha256_file(path: Path) -> str:
    """Stream one artifact into its SHA-256 identity without loading it into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_manifest(
    *,
    bundle_root: Path,
    model_version: str,
    trained_at: datetime,
    snapshot_measured_at: datetime,
    eligible_user_count: int,
    rated_interaction_count: int,
    film_count: int,
) -> ModelManifest:
    """Create a compatibility and checksum manifest for completed payload files.

    Every serving artifact is hashed after it exists, and runtime/training semantics
    are frozen into the manifest so incompatible bundles fail before activation.

    Raises:
        ValueError: If ``model_version`` does not match the safe version format.
        OSError: If any payload cannot be streamed for its checksum.
    """
    _validate_version(model_version)
    checksums = {
        filename: sha256_file(bundle_root / filename)
        for filename in MODEL_ARTIFACT_FILENAMES
    }
    timestamp = trained_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return ModelManifest(
        schema=MANIFEST_SCHEMA,
        model_version=model_version,
        created_at=timestamp,
        trained_at=timestamp,
        svd_dimension=SVD_DIMENSION,
        film_count=film_count,
        eligible_user_count=eligible_user_count,
        rated_interaction_count=rated_interaction_count,
        popularity_threshold=POPULARITY_RATING_THRESHOLD,
        artifacts=checksums,
        training_semantics=TRAINING_SEMANTICS,
        python_version=platform.python_version(),
        sklearn_version=_package_version("scikit-learn"),
        faiss_version=_package_version("faiss-cpu"),
        snapshot_measured_at=snapshot_measured_at.astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z"),
    )


def write_manifest(manifest: ModelManifest, path: Path) -> None:
    """Create and durably flush a manifest without overwriting existing identity."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest.to_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_manifest(path: str | Path) -> ModelManifest:
    """Parse and validate an exact manifest schema before trusting bundle identity.

    Extra and missing fields are rejected together with incompatible constants,
    timestamps, and checksum shapes.

    Raises:
        ValueError: If JSON, field identity, or compatibility validation fails.
    """
    try:
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict) or set(payload) != {
            field.name for field in ModelManifest.__dataclass_fields__.values()
        }:
            raise ValueError("manifest fields are invalid")
        manifest = ModelManifest(**payload)
    except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
        raise ValueError("invalid model manifest") from exc
    _validate_manifest(manifest)
    return manifest


def _validate_manifest(manifest: ModelManifest) -> None:
    """Enforce frozen schema, model-shape, training, and checksum invariants."""
    if manifest.schema != MANIFEST_SCHEMA:
        raise ValueError("unsupported model manifest schema")
    _validate_version(manifest.model_version)
    _parse_utc(manifest.created_at)
    _parse_utc(manifest.trained_at)
    _parse_utc(manifest.snapshot_measured_at)
    if manifest.svd_dimension != SVD_DIMENSION:
        raise ValueError("unexpected SVD dimension")
    if manifest.film_count <= 0:
        raise ValueError("manifest film count must be positive")
    if manifest.eligible_user_count <= 0 or manifest.rated_interaction_count <= 0:
        raise ValueError("manifest training counts must be positive")
    if manifest.popularity_threshold != POPULARITY_RATING_THRESHOLD:
        raise ValueError("unexpected popularity threshold")
    if manifest.training_semantics != TRAINING_SEMANTICS:
        raise ValueError("unexpected production training semantics")
    if set(manifest.artifacts) != set(MODEL_ARTIFACT_FILENAMES):
        raise ValueError("manifest artifact set is incomplete")
    if any(
        not isinstance(checksum, str) or not re.fullmatch(r"[a-f0-9]{64}", checksum)
        for checksum in manifest.artifacts.values()
    ):
        raise ValueError("manifest artifact checksum is invalid")


def validate_model_bundle(path: str | Path) -> ValidatedModelBundle:
    """Fully cross-validate one immutable model bundle before it is selectable.

    Directory/manifest identity, payload SHA-256 checksums, SVD/FAISS dimensions and
    exact ID ordering, and production-popularity membership must all agree.

    Returns:
        ValidatedModelBundle: Manifest and already loaded SVD resources for the valid
            immutable directory.

    Raises:
        ValueError: If any identity, checksum, compatibility, or cross-artifact
            invariant fails.
        OSError: If a required payload cannot be read.
    """
    # Validate immutable payload bytes before deserializing model resources.
    root = Path(path)
    manifest = read_manifest(root / MANIFEST_FILENAME)
    if root.name != manifest.model_version:
        raise ValueError("bundle directory and manifest version differ")
    for filename, expected_checksum in manifest.artifacts.items():
        artifact_path = root / filename
        if not artifact_path.is_file():
            raise ValueError(f"bundle artifact is missing: {filename}")
        if sha256_file(artifact_path) != expected_checksum:
            raise ValueError(f"bundle artifact checksum mismatch: {filename}")
    # Cross-check semantic identity after individual artifact formats are valid.
    artifacts = load_svd_artifacts(root)
    if artifacts.item_vectors.shape[1] != manifest.svd_dimension:
        raise ValueError("manifest and vectors have different dimensions")
    if len(artifacts.film_index) != manifest.film_count:
        raise ValueError("manifest and film mapping have different counts")
    stored_ids = get_faiss_ids(artifacts.retrieval_index)
    expected_ids = np.asarray(artifacts.film_index, dtype=np.int64)
    if not np.array_equal(stored_ids, expected_ids):
        raise ValueError("FAISS ID ordering differs from film mapping")
    popularity = read_popularity_artifact(root / "popularity.json")
    if popularity.source_description != PRODUCTION_POPULARITY_SOURCE:
        raise ValueError("bundle popularity is not production-sourced")
    if popularity.film_count != manifest.film_count:
        raise ValueError("popularity and model film counts differ")
    if {entry.film_id for entry in popularity.films} != set(artifacts.film_index):
        raise ValueError("popularity and model film identities differ")
    return ValidatedModelBundle(root, manifest, artifacts)


def read_model_pointer(artifact_root: str | Path) -> ModelPointer | None:
    """Read the authoritative pointer, accepting the original one-field format.

    Returns:
        ModelPointer | None: Selected version and optional recovery lineage, or
            ``None`` when serving should use the legacy-flat layout.

    Raises:
        ValueError: If pointer JSON, fields, or version identities are invalid.
    """
    pointer = model_root(artifact_root) / CURRENT_POINTER_FILENAME
    if not pointer.exists():
        return None
    try:
        with pointer.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if (
            not isinstance(payload, dict)
            or not set(payload).issubset({"model_version", "previous_version"})
            or "model_version" not in payload
        ):
            raise ValueError
        model_version = _validate_version(payload["model_version"])
        previous = payload.get("previous_version")
        if previous is not None and previous != "legacy-flat":
            previous = _validate_version(previous)
        return ModelPointer(model_version, previous)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("current model pointer is invalid") from exc


def read_current_version(artifact_root: str | Path) -> str | None:
    """Return the selected version, or ``None`` for legacy-flat serving."""
    pointer = read_model_pointer(artifact_root)
    return pointer.model_version if pointer is not None else None


def _validate_legacy_serving_model(base: Path) -> None:
    """Require a complete flat SVD/FAISS layout and compatible legacy popularity."""
    artifacts = load_svd_artifacts(base)
    popularity = read_popularity_artifact(base / "candidates" / "popularity.json")
    if not {entry.film_id for entry in popularity.films}.issubset(
        set(artifacts.film_index)
    ):
        raise ValueError("legacy popularity catalog is incompatible with SVD")


def resolve_serving_model(
    artifact_root: str | Path, *, validate: bool = True
) -> ServingModelLocation:
    """Resolve exactly one versioned bundle or the complete legacy-flat layout.

    ``current.json`` is authoritative when present. Without it, serving remains
    backward compatible only if flat SVD/FAISS artifacts and ``candidates``
    popularity are mutually compatible.

    Returns:
        ServingModelLocation: Effective roots, version identity, and manifest when
            serving a versioned bundle.

    Raises:
        ValueError: If the pointer or selected artifacts fail validation.
    """
    base = Path(artifact_root)
    current_version = read_current_version(base)
    if current_version is None:
        if validate:
            _validate_legacy_serving_model(base)
        return ServingModelLocation(
            root=base,
            model_version="legacy-flat",
            popularity_path=base / "candidates" / "popularity.json",
            versioned=False,
        )
    bundle_root = model_root(base) / current_version
    bundle = validate_model_bundle(bundle_root) if validate else None
    return ServingModelLocation(
        root=bundle_root,
        model_version=current_version,
        popularity_path=bundle_root / "popularity.json",
        versioned=True,
        manifest=bundle.manifest
        if bundle
        else read_manifest(bundle_root / MANIFEST_FILENAME),
    )


def promote_model_bundle(
    artifact_root: str | Path, model_version: str
) -> PromotionResult:
    """Validate a bundle and atomically replace the authoritative current pointer.

    The prior selection is retained as activation-recovery lineage. A valid
    legacy-flat layout is recorded explicitly so startup failure can restore it.

    Returns:
        PromotionResult: New and prior version identities after pointer replacement.

    Raises:
        ValueError: If the version or candidate bundle is invalid.
        OSError: If durable pointer publication fails.
    """
    # Never create a pointer to payloads that have not passed full bundle validation.
    version_id = _validate_version(model_version)
    models = model_root(artifact_root)
    validate_model_bundle(models / version_id)
    previous = read_current_version(artifact_root)
    previous_pointer = previous
    if previous_pointer is None:
        try:
            _validate_legacy_serving_model(Path(artifact_root))
        except (OSError, ValueError, TypeError):
            pass
        else:
            previous_pointer = "legacy-flat"
    _write_current_pointer(models, version_id, previous_pointer)
    logger.info(
        "Model promoted version=%s previous=%s activation=pending",
        version_id,
        previous or "legacy-flat",
    )
    return PromotionResult(version_id, previous)


def _write_current_pointer(
    models: Path,
    model_version: str,
    previous_version: str | None,
) -> None:
    """Atomically and durably replace current.json with activation lineage."""
    version_id = _validate_version(model_version)
    if previous_version is not None and previous_version != "legacy-flat":
        _validate_version(previous_version)
    # Flush file contents before atomic replacement, then fsync the parent directory
    # so the renamed selection survives a host crash.
    models.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".current.", suffix=".tmp", dir=models
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "model_version": version_id,
                    "previous_version": previous_version,
                },
                stream,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, models / CURRENT_POINTER_FILENAME)
        directory_descriptor = os.open(models, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def list_valid_model_bundles(artifact_root: str | Path) -> list[ModelManifest]:
    """Return fully validated bundles ordered from newest to oldest training time.

    Invalid or partially built directories are ignored and logged; only safe version
    names are considered retention or rollback candidates.
    """
    models = model_root(artifact_root)
    if not models.exists():
        return []
    manifests: list[ModelManifest] = []
    for child in models.iterdir():
        if not child.is_dir() or not _VERSION_PATTERN.fullmatch(child.name):
            continue
        try:
            manifests.append(validate_model_bundle(child).manifest)
        except (OSError, ValueError, TypeError):
            logger.warning("Ignoring invalid model bundle version=%s", child.name)
    return sorted(manifests, key=lambda item: _parse_utc(item.trained_at), reverse=True)


def apply_model_retention(
    artifact_root: str | Path, *, keep_previous: int
) -> tuple[str, ...]:
    """Keep current plus the configured number of newest rollback bundles.

    Deletion targets are resolved beneath the exact models directory and revalidated
    against the safe version pattern before removal.

    Returns:
        tuple[str, ...]: Removed model versions in newest-to-oldest scan order.

    Raises:
        ValueError: If a resolved deletion target escapes the models directory.
    """
    current = read_current_version(artifact_root)
    manifests = list_valid_model_bundles(artifact_root)
    keep = set(
        ([current] if current is not None else [])
        + [
            manifest.model_version
            for manifest in manifests
            if manifest.model_version != current
        ][:keep_previous]
    )
    removed: list[str] = []
    models = model_root(artifact_root).resolve()
    for manifest in manifests:
        if manifest.model_version in keep or manifest.model_version == current:
            continue
        target = (models / manifest.model_version).resolve()
        if target.parent != models or not _VERSION_PATTERN.fullmatch(target.name):
            raise ValueError("refusing unsafe model retention target")
        shutil.rmtree(target)
        removed.append(manifest.model_version)
    return tuple(removed)


def select_rollback_target(artifact_root: str | Path) -> ModelManifest:
    """Return the newest valid bundle trained strictly before the current model.

    Raises:
        LookupError: If no current version or eligible older valid bundle exists.
    """
    current = read_current_version(artifact_root)
    if current is None:
        raise LookupError("no versioned current model is configured")
    current_manifest = read_manifest(
        model_root(artifact_root) / current / MANIFEST_FILENAME
    )
    current_trained_at = _parse_utc(current_manifest.trained_at)
    candidates = [
        manifest
        for manifest in list_valid_model_bundles(artifact_root)
        if _parse_utc(manifest.trained_at) < current_trained_at
    ]
    if not candidates:
        raise LookupError("no previous valid model bundle is available")
    return candidates[0]


def rollback_model(artifact_root: str | Path) -> PromotionResult:
    """Atomically select the newest valid bundle strictly older than current."""
    target = select_rollback_target(artifact_root)
    return promote_model_bundle(artifact_root, target.model_version)


def recover_previous_model(
    artifact_root: str | Path, failed_version: str
) -> ServingModelLocation:
    """Restore activation lineage once when a promoted bundle cannot start.

    Recovery proceeds only if the failed version is still current, preventing a
    stale process from overwriting a newer promotion. Legacy restoration removes the
    pointer by renaming it to an activation-failure record; versioned restoration
    validates the predecessor before atomically selecting it.

    Returns:
        ServingModelLocation: Fully validated location restored for startup.

    Raises:
        RuntimeError: If another process has already changed the current selection.
        LookupError: If no valid predecessor can be selected.
        ValueError: If recovery lineage or predecessor artifacts are invalid.
    """
    base = Path(artifact_root)
    pointer = read_model_pointer(base)
    if pointer is None or pointer.model_version != failed_version:
        raise RuntimeError("failed activation is no longer the current model")
    previous = pointer.previous_version
    if previous == "legacy-flat":
        _validate_legacy_serving_model(base)
        current_path = model_root(base) / CURRENT_POINTER_FILENAME
        failed_path = model_root(base) / f".activation-failed-{failed_version}.json"
        os.replace(current_path, failed_path)
        logger.critical(
            "Model activation failed version=%s; restored legacy-flat serving",
            failed_version,
        )
        return resolve_serving_model(base, validate=True)
    if previous is None:
        previous = select_rollback_target(base).model_version
    validate_model_bundle(model_root(base) / previous)
    _write_current_pointer(model_root(base), previous, None)
    logger.critical(
        "Model activation failed version=%s; restored previous=%s",
        failed_version,
        previous,
    )
    return resolve_serving_model(base, validate=True)


def bundle_disk_bytes(path: str | Path) -> int:
    """Measure regular payload bytes directly contained in one bundle."""
    return sum(item.stat().st_size for item in Path(path).iterdir() if item.is_file())


def log_validation_timing(path: str | Path) -> ValidatedModelBundle:
    """Validate one bundle and log its identity, duration, and disk footprint."""
    started = time.perf_counter()
    validated = validate_model_bundle(path)
    logger.info(
        "Model bundle validated version=%s validation_s=%.3f disk_bytes=%d",
        validated.manifest.model_version,
        time.perf_counter() - started,
        bundle_disk_bytes(path),
    )
    return validated
