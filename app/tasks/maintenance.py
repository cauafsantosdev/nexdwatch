"""Celery adapters for bounded, lock-protected backend maintenance."""

import logging
from dataclasses import asdict
from datetime import UTC, datetime

from celery import Task

from app.core.config import get_settings
from app.db.loaders.sync_queue import sync_film_queue
from app.infrastructure.maintenance_lock import MaintenanceLock
from app.services.catalog_maintenance import refresh_recent_catalog
from app.workers.async_bridge import worker_async_bridge
from app.workers.celery_app import MAINTENANCE_QUEUE, celery_app

logger = logging.getLogger(__name__)
settings = get_settings()


def _lock(key: str) -> MaintenanceLock:
    """Create a task-local Redis lock using the shared crash-recovery TTL."""
    return MaintenanceLock(
        settings.MAINTENANCE_REDIS_URL,
        key=key,
        ttl_seconds=settings.MAINTENANCE_LOCK_TTL_SECONDS,
    )


def _retry(task: Task, exc: Exception, countdown: int) -> None:
    """Request a bounded Celery retry or re-raise after the retry budget is spent."""
    retries = int(task.request.retries)
    if retries < int(task.max_retries or 0):
        raise task.retry(exc=exc, countdown=countdown) from exc
    raise exc


@celery_app.task(
    bind=True,
    name="app.tasks.maintenance.process_film_queue",
    queue=MAINTENANCE_QUEUE,
    max_retries=2,
    soft_time_limit=settings.MAINTENANCE_SOFT_TIME_LIMIT_SECONDS,
    time_limit=settings.MAINTENANCE_HARD_TIME_LIMIT_SECONDS,
)
def process_film_queue_task(task: Task) -> dict[str, object]:
    """Run one bounded ``FilmQueue`` batch under singleton ownership.

    Lock contention is a successful skip. Batch scraper/database failures cross the
    worker async bridge and retry up to twice; per-film terminal outcomes are already
    isolated by the queue loader.

    Returns:
        A JSON-safe status plus queue outcome counts for worker logs/inspection.
    """
    lock = _lock("film-queue")
    try:
        # Acquire before entering async services so only one worker selects a pending
        # batch; TTL ownership recovers automatically if this process dies.
        with lock.held() as acquired:
            if not acquired:
                logger.info("Film queue maintenance skipped: lock already held")
                return {"status": "skipped_locked"}
            try:
                result = worker_async_bridge.run(
                    sync_film_queue(batch_size=settings.FILM_QUEUE_BATCH_SIZE)
                )
            except Exception as exc:  # noqa: BLE001 - Celery retry boundary
                _retry(task, exc, 60)
            return {"status": "completed", **asdict(result)}
    finally:
        lock.close()


@celery_app.task(
    bind=True,
    name="app.tasks.maintenance.refresh_recent_catalog",
    queue=MAINTENANCE_QUEUE,
    max_retries=2,
    soft_time_limit=settings.MAINTENANCE_SOFT_TIME_LIMIT_SECONDS,
    time_limit=settings.MAINTENANCE_HARD_TIME_LIMIT_SECONDS,
)
def refresh_recent_catalog_task(task: Task) -> dict[str, object]:
    """Refresh recent-film aggregates with period-scoped singleton ownership.

    The year/month lock permits later schedule periods while deduplicating concurrent
    delivery of the same January or July run. Transient batch failure retries twice.
    """
    today = datetime.now(UTC).date()
    lock = _lock(f"catalog-refresh:{today.year}-{today.month:02d}")
    try:
        with lock.held() as acquired:
            if not acquired:
                return {"status": "skipped_locked"}
            try:
                result = worker_async_bridge.run(refresh_recent_catalog(today))
            except Exception as exc:  # noqa: BLE001 - Celery retry boundary
                _retry(task, exc, 300)
            return {"status": "completed", **asdict(result)}
    finally:
        lock.close()


@celery_app.task(
    name="app.tasks.maintenance.evaluate_retraining",
    queue=MAINTENANCE_QUEUE,
)
def evaluate_retraining_task() -> dict[str, object]:
    """Evaluate frozen thresholds and enqueue retraining when eligible.

    Evaluation is singleton and lightweight; it never trains inline. An eligible
    decision publishes one task to the isolated maintenance queue, where a second
    lock protects the expensive build/promotion lifecycle.
    """
    from app.ml.model_lifecycle import evaluate_retraining

    lock = _lock("retraining-evaluation")
    try:
        with lock.held() as acquired:
            if not acquired:
                return {"status": "skipped_locked"}
            # Keep Beat evaluation cheap and publish CPU-heavy training separately.
            decision = evaluate_retraining()
            if decision.should_retrain:
                retrain_and_promote_task.apply_async(queue=MAINTENANCE_QUEUE)
            return {
                "status": "retraining_queued" if decision.should_retrain else "current",
                "reasons": [reason.value for reason in decision.reasons],
            }
    finally:
        lock.close()


@celery_app.task(
    bind=True,
    name="app.tasks.maintenance.retrain_and_promote",
    queue=MAINTENANCE_QUEUE,
    max_retries=1,
    soft_time_limit=settings.MAINTENANCE_SOFT_TIME_LIMIT_SECONDS,
    time_limit=settings.MAINTENANCE_HARD_TIME_LIMIT_SECONDS,
)
def retrain_and_promote_task(task: Task, force: bool = False) -> dict[str, object]:
    """Build, validate, and promote one versioned model without stopping serving.

    Singleton ownership prevents overlapping PostgreSQL snapshots and model builds.
    Failures leave the old pointer authoritative and retry once; successful promotion
    reports activation as pending because API workers restart through pointer watch.
    """
    from app.ml.model_lifecycle import retrain_and_promote

    lock = _lock("retraining")
    try:
        with lock.held() as acquired:
            if not acquired:
                return {"status": "skipped_locked"}
            # The lifecycle itself isolates candidate artifacts and updates the
            # pointer only after validation; this boundary owns Celery retry policy.
            try:
                result = retrain_and_promote(force=force)
            except Exception as exc:
                logger.exception(
                    "Production retraining failed; current model unchanged"
                )
                _retry(task, exc, 300)
            if result.promotion is None:
                return {"status": "not_required"}
            return {
                "status": "promoted",
                "model_version": result.promotion.model_version,
                "api_activation": "pending",
            }
    finally:
        lock.close()
