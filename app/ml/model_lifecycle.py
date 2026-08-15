"""Conditional production retraining and immutable bundle construction."""

import logging
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings, get_settings
from app.domain.maintenance import (
    PromotionResult,
    RetrainingDecision,
    RetrainingDeltas,
    RetrainingReason,
    TrainedModelStatistics,
    TrainingStatistics,
)
from app.ml.model_registry import (
    MANIFEST_FILENAME,
    ModelManifest,
    apply_model_retention,
    create_manifest,
    model_root,
    new_model_version,
    promote_model_bundle,
    read_current_version,
    resolve_serving_model,
    validate_model_bundle,
    write_manifest,
)
from app.ml.popularity import write_popularity_artifact
from app.ml.production_training import (
    PreparedProductionTrainingData,
    build_production_popularity_artifact,
    load_production_training_data,
    measure_production_training_data,
    train_prepared_svd,
)
from app.ml.svd_artifacts import load_svd_artifacts

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BundleBuildResult:
    """Completed immutable bundle and operational stage timings."""

    root: Path
    manifest: ModelManifest
    extraction_seconds: float
    training_seconds: float
    artifact_write_seconds: float
    faiss_build_seconds: float
    popularity_seconds: float
    validation_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class RetrainingRunResult:
    """Retraining decision plus optional build and atomic promotion results."""

    decision: RetrainingDecision
    build: BundleBuildResult | None
    promotion: PromotionResult | None


def _trained_state(artifact_root: str | Path) -> TrainedModelStatistics | None:
    """Resolve comparable training state from a versioned or legacy-flat model.

    Versioned manifests provide exact cohort counters and model identities. A valid
    legacy-flat layout can provide only file age and film IDs, so unknown counters
    are marked ``-1`` to trigger the explicit legacy-bootstrap retraining reason.
    """
    location = resolve_serving_model(artifact_root, validate=False)
    if location.versioned:
        bundle = validate_model_bundle(location.root)
        manifest = bundle.manifest
        return TrainedModelStatistics(
            trained_at=datetime.fromisoformat(
                manifest.trained_at.replace("Z", "+00:00")
            ).astimezone(UTC),
            eligible_user_count=manifest.eligible_user_count,
            rated_interaction_count=manifest.rated_interaction_count,
            model_film_count=manifest.film_count,
            model_film_ids=bundle.artifacts.film_index,
        )
    try:
        artifacts = load_svd_artifacts(location.root)
    except (OSError, ValueError, TypeError):
        return None
    trained_at = datetime.fromtimestamp(
        (location.root / "item_embeddings.npy").stat().st_mtime, tz=UTC
    )
    return TrainedModelStatistics(
        trained_at=trained_at,
        eligible_user_count=-1,
        rated_interaction_count=-1,
        model_film_count=len(artifacts.film_index),
        model_film_ids=artifacts.film_index,
    )


def decide_retraining(
    current: TrainingStatistics,
    trained: TrainedModelStatistics | None,
    *,
    force: bool = False,
    now: datetime | None = None,
    settings: Settings | None = None,
) -> RetrainingDecision:
    """Apply centralized retraining thresholds to measured and trained universes.

    User growth, newly model-eligible film IDs, and model age are evaluated against
    frozen settings. Legacy-flat models deliberately bypass ordinary thresholds and
    bootstrap once because they do not carry exact historical counters.

    Returns:
        RetrainingDecision: Trigger decision, ordered reasons, comparable states, and
            computed deltas used by operations.
    """
    effective = settings or get_settings()
    reference_time = (now or datetime.now(UTC)).astimezone(UTC)
    if trained is None:
        user_delta = current.eligible_user_count
        interaction_delta = current.rated_interaction_count
        new_films = current.model_film_count
        age_days = 0.0
    else:
        user_delta = (
            0
            if trained.eligible_user_count < 0
            else max(0, current.eligible_user_count - trained.eligible_user_count)
        )
        interaction_delta = (
            0
            if trained.rated_interaction_count < 0
            else current.rated_interaction_count - trained.rated_interaction_count
        )
        new_films = len(set(current.rated_film_ids) - set(trained.model_film_ids))
        age_days = max(
            0.0, (reference_time - trained.trained_at).total_seconds() / 86400
        )
    # Legacy state cannot support trustworthy user/interaction deltas; force one
    # versioned build rather than interpreting unknown counters as real values.
    reasons: list[RetrainingReason] = []
    if force:
        reasons.append(RetrainingReason.FORCED)
    legacy_bootstrap = trained is not None and (
        trained.eligible_user_count < 0 or trained.rated_interaction_count < 0
    )
    if legacy_bootstrap:
        reasons.append(RetrainingReason.LEGACY_MODEL_BOOTSTRAP)
    if not legacy_bootstrap and user_delta >= effective.NEW_ELIGIBLE_USERS_THRESHOLD:
        reasons.append(RetrainingReason.NEW_USERS_THRESHOLD)
    if not legacy_bootstrap and new_films >= effective.NEW_MODEL_FILMS_THRESHOLD:
        reasons.append(RetrainingReason.NEW_FILMS_THRESHOLD)
    if (
        not legacy_bootstrap
        and trained is not None
        and age_days >= effective.MAX_MODEL_AGE_DAYS
    ):
        reasons.append(RetrainingReason.MODEL_AGE_THRESHOLD)
    if not reasons:
        reasons.append(RetrainingReason.NONE)
    return RetrainingDecision(
        should_retrain=reasons != [RetrainingReason.NONE],
        reasons=tuple(reasons),
        current_stats=current,
        trained_stats=trained,
        deltas=RetrainingDeltas(
            eligible_users=user_delta,
            rated_interactions=interaction_delta,
            new_model_films=new_films,
            model_age_days=age_days,
        ),
    )


def evaluate_retraining(
    *,
    artifact_root: str | Path | None = None,
    force: bool = False,
    settings: Settings | None = None,
) -> RetrainingDecision:
    """Measure current PostgreSQL state and apply the frozen retraining policy.

    Args:
        artifact_root: Legacy/versioned artifact root to compare with PostgreSQL.
        force: Add an unconditional operational retraining reason.
        settings: Optional thresholds and connection settings override.

    Returns:
        RetrainingDecision: Current-versus-trained deltas and trigger reasons.
    """
    effective = settings or get_settings()
    root = Path(artifact_root or effective.ARTIFACT_ROOT)
    current = measure_production_training_data(settings=effective)
    trained = _trained_state(root)
    decision = decide_retraining(current, trained, force=force, settings=effective)
    logger.info(
        "Retraining evaluated model=%s user_delta=%d film_delta=%d "
        "interaction_delta=%d age_days=%.1f should_retrain=%s reasons=%s",
        read_current_version(root) or "legacy-flat",
        decision.deltas.eligible_users,
        decision.deltas.new_model_films,
        decision.deltas.rated_interactions,
        decision.deltas.model_age_days,
        decision.should_retrain,
        [reason.value for reason in decision.reasons],
    )
    return decision


def build_model_bundle(
    data: PreparedProductionTrainingData,
    *,
    artifact_root: str | Path,
    model_version: str | None = None,
    now: datetime | None = None,
) -> BundleBuildResult:
    """Build and validate a new directory without touching the current pointer.

    Partial output is confined to ``.building-*``. Any failure removes only the
    candidate, leaving the authoritative pointer and running API unchanged.

    Args:
        data: One coherent PostgreSQL snapshot shared by all artifact builders.
        artifact_root: Parent artifact root containing immutable version directories.
        model_version: Optional prevalidated identity override for deterministic tests.
        now: Optional UTC-compatible training timestamp override.

    Returns:
        BundleBuildResult: Validated immutable bundle, manifest, and stage timings.

    Raises:
        FileExistsError: If temporary or final output already uses the version ID.
        Exception: Propagates training, artifact, manifest, rename, or validation
            failures after removing only this candidate bundle.
    """
    started = time.perf_counter()
    trained_at = (now or datetime.now(UTC)).astimezone(UTC)
    version_id = model_version or new_model_version(trained_at)
    models = model_root(artifact_root)
    models.mkdir(parents=True, exist_ok=True)
    temporary = models / f".building-{version_id}"
    final = models / version_id
    if temporary.exists() or final.exists():
        raise FileExistsError(f"model version already exists: {version_id}")
    # A candidate is never visible through current.json while any payload, manifest,
    # checksum, or cross-artifact validation remains incomplete.
    temporary.mkdir()
    # Build SVD/FAISS and popularity from the same prepared snapshot before creating
    # a checksum manifest that covers every serving payload.
    try:
        svd_result = train_prepared_svd(data, temporary)
        popularity_started = time.perf_counter()
        popularity = build_production_popularity_artifact(data)
        write_popularity_artifact(popularity, temporary / "popularity.json")
        popularity_seconds = time.perf_counter() - popularity_started
        manifest = create_manifest(
            bundle_root=temporary,
            model_version=version_id,
            trained_at=trained_at,
            snapshot_measured_at=data.statistics.measured_at,
            eligible_user_count=data.statistics.eligible_user_count,
            rated_interaction_count=data.statistics.rated_interaction_count,
            film_count=data.statistics.model_film_count,
        )
        write_manifest(manifest, temporary / MANIFEST_FILENAME)
        # Rename the complete candidate to its immutable version name, then perform a
        # full cross-artifact read before promotion can select it.
        temporary.rename(final)
        validation_started = time.perf_counter()
        validate_model_bundle(final)
        validation_seconds = time.perf_counter() - validation_started
    except Exception:
        # Candidate cleanup never reads or mutates current.json, preserving the old
        # serving version through every build/validation failure.
        if temporary.exists():
            shutil.rmtree(temporary)
        if final.exists():
            shutil.rmtree(final)
        raise
    total_seconds = time.perf_counter() - started
    logger.info(
        "Model bundle built version=%s rows=%d users=%d films=%d extraction_s=%.3f "
        "training_s=%.3f artifact_write_s=%.3f faiss_s=%.3f popularity_s=%.3f "
        "validation_s=%.3f total_s=%.3f",
        version_id,
        data.statistics.rated_interaction_count,
        data.statistics.eligible_user_count,
        data.statistics.model_film_count,
        data.extraction_seconds,
        svd_result.training_seconds,
        svd_result.artifact_write_seconds,
        svd_result.faiss_build_seconds,
        popularity_seconds,
        validation_seconds,
        total_seconds,
    )
    return BundleBuildResult(
        root=final,
        manifest=manifest,
        extraction_seconds=data.extraction_seconds,
        training_seconds=svd_result.training_seconds,
        artifact_write_seconds=svd_result.artifact_write_seconds,
        faiss_build_seconds=svd_result.faiss_build_seconds,
        popularity_seconds=popularity_seconds,
        validation_seconds=validation_seconds,
        total_seconds=total_seconds,
    )


def retrain_and_promote(
    *,
    artifact_root: str | Path | None = None,
    force: bool = False,
    settings: Settings | None = None,
) -> RetrainingRunResult:
    """Evaluate, build, atomically promote, and retain with old-current safety.

    A cheap aggregate evaluation avoids unnecessary extraction. Eligible runs then
    load one coherent snapshot and re-evaluate to close the measurement/build gap.
    Only a fully validated immutable bundle replaces ``current.json``; retention runs
    afterward and always preserves the selected model.

    Returns:
        RetrainingRunResult: Final decision and optional build/promotion details.

    Raises:
        Exception: Propagates extraction, training, validation, promotion, or
            retention failures without intentionally changing a valid current model
            before atomic promotion.
    """
    effective = settings or get_settings()
    root = Path(artifact_root or effective.ARTIFACT_ROOT)
    initial_decision = evaluate_retraining(
        artifact_root=root, force=force, settings=effective
    )
    if not initial_decision.should_retrain:
        return RetrainingRunResult(initial_decision, None, None)
    # Re-evaluate against the exact materialized training snapshot so a threshold
    # that changed between measurement and extraction cannot build needlessly.
    snapshot = load_production_training_data(settings=effective)
    decision = decide_retraining(
        snapshot.statistics,
        initial_decision.trained_stats,
        force=force,
        settings=effective,
    )
    if not decision.should_retrain:
        return RetrainingRunResult(decision, None, None)
    # Promotion occurs only after bundle-level checksums and cross-artifact identity
    # validation succeed; retention cannot remove the newly current version.
    build = build_model_bundle(snapshot, artifact_root=root)
    promotion = promote_model_bundle(root, build.manifest.model_version)
    apply_model_retention(root, keep_previous=effective.MODEL_RETENTION_PREVIOUS)
    return RetrainingRunResult(decision, build, promotion)
