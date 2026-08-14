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
    return MaintenanceLock(
        settings.MAINTENANCE_REDIS_URL,
        key=key,
        ttl_seconds=settings.MAINTENANCE_LOCK_TTL_SECONDS,
    )


def _retry(task: Task, exc: Exception, countdown: int) -> None:
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
    lock = _lock("film-queue")
    try:
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
    from app.ml.model_lifecycle import evaluate_retraining

    lock = _lock("retraining-evaluation")
    try:
        with lock.held() as acquired:
            if not acquired:
                return {"status": "skipped_locked"}
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
    from app.ml.model_lifecycle import retrain_and_promote

    lock = _lock("retraining")
    try:
        with lock.held() as acquired:
            if not acquired:
                return {"status": "skipped_locked"}
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
