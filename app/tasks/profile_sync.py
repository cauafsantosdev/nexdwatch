"""Celery adapter for the existing profile synchronization service."""

import logging
import random
from typing import Protocol

from billiard.exceptions import SoftTimeLimitExceeded
from celery import Task

from app.core.config import get_settings
from app.domain.task_state import TaskError, TaskMetadata, TaskResult, TaskStatus
from app.infrastructure.task_store import SyncRedisTaskStore
from app.scraper.user_scraping import TransientProfileScrapeError
from app.services.profile_service import EmptyProfileError, ProfileService
from app.workers.async_bridge import WorkerAsyncBridge, worker_async_bridge
from app.workers.celery_app import PROFILE_SYNC_QUEUE, celery_app

logger = logging.getLogger(__name__)
settings = get_settings()

_PUBLIC_FAILURE = TaskError(
    code="profile_sync_failed",
    message="Unable to synchronize Letterboxd profile.",
)


class WorkerTaskStore(Protocol):
    """Synchronous task-store operations needed by the worker."""

    def get_task(self, task_id: str) -> TaskMetadata | None: ...

    def save_task(self, task: TaskMetadata) -> None: ...

    def claim_active(self, username: str, task_id: str) -> bool: ...

    def release_active(self, username: str, task_id: str) -> bool: ...

    def set_fresh(self, username: str, task_id: str) -> None: ...

    def close(self) -> None: ...


def _new_worker_store() -> SyncRedisTaskStore:
    return SyncRedisTaskStore(
        settings.TASK_STATE_REDIS_URL,
        task_ttl=settings.TASK_RESULT_TTL_SECONDS,
        active_ttl=settings.PROFILE_SYNC_ACTIVE_TTL_SECONDS,
        freshness_ttl=settings.PROFILE_SYNC_FRESHNESS_SECONDS,
    )


def execute_profile_sync(
    task: Task,
    username: str,
    task_id: str,
    *,
    store: WorkerTaskStore,
    profile_service: ProfileService,
    bridge: WorkerAsyncBridge,
) -> None:
    """Execute one profile sync with public state and selective retries."""
    metadata = store.get_task(task_id)
    if metadata is None:
        logger.error("Profile sync metadata missing task_id=%s", task_id)
        return
    if metadata.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
        logger.info("Ignoring terminal redelivery task_id=%s", task_id)
        return
    if not store.claim_active(username, task_id):
        logger.warning(
            "Profile sync superseded task_id=%s username=%s", task_id, username
        )
        store.save_task(metadata.failed(_PUBLIC_FAILURE))
        return

    processing = metadata.processing()
    store.save_task(processing)
    try:
        result = bridge.run(profile_service.sync_profile(username))
    except TransientProfileScrapeError as exc:
        retries = int(task.request.retries)
        if retries < int(task.max_retries or 0):
            countdown = min(60.0, 5.0 * (2**retries) + random.uniform(0.0, 2.0))
            logger.warning(
                "Retrying transient profile sync task_id=%s attempt=%d",
                task_id,
                processing.attempt,
            )
            raise task.retry(exc=exc, countdown=countdown) from exc
        _fail_task(store, processing, exc)
        raise
    except (EmptyProfileError, SoftTimeLimitExceeded) as exc:
        _fail_task(store, processing, exc)
        raise
    except Exception as exc:
        _fail_task(store, processing, exc)
        raise

    completed = processing.completed(
        TaskResult(user_id=result.user_id, logs_count=result.logs_count)
    )
    store.set_fresh(username, task_id)
    store.save_task(completed)
    store.release_active(username, task_id)
    logger.info(
        "Profile sync completed task_id=%s username=%s logs=%d",
        task_id,
        username,
        result.logs_count,
    )


def _fail_task(
    store: WorkerTaskStore,
    metadata: TaskMetadata,
    exception: Exception,
) -> None:
    logger.exception(
        "Profile sync failed task_id=%s username=%s",
        metadata.task_id,
        metadata.username,
        exc_info=exception,
    )
    store.save_task(metadata.failed(_PUBLIC_FAILURE))
    store.release_active(metadata.username, metadata.task_id)


@celery_app.task(
    bind=True,
    name="app.tasks.profile_sync",
    queue=PROFILE_SYNC_QUEUE,
    ignore_result=True,
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=2,
    soft_time_limit=settings.PROFILE_SYNC_SOFT_TIME_LIMIT_SECONDS,
    time_limit=settings.PROFILE_SYNC_HARD_TIME_LIMIT_SECONDS,
)
def profile_sync_task(task: Task, username: str, task_id: str) -> None:
    """Celery entrypoint containing only primitive message arguments."""
    store = _new_worker_store()
    try:
        execute_profile_sync(
            task,
            username,
            task_id,
            store=store,
            profile_service=ProfileService(),
            bridge=worker_async_bridge,
        )
    finally:
        store.close()
