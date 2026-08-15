"""Create, deduplicate, and poll durable background tasks."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from app.core.config import get_settings
from app.domain.task_state import TaskError, TaskMetadata, TaskStatus
from app.infrastructure.task_store import AsyncRedisTaskStore

logger = logging.getLogger(__name__)


class TaskInfrastructureError(RuntimeError):
    """Raised when Redis or the Celery broker is unavailable."""


class AsyncTaskStore(Protocol):
    """Async task-store operations required by the task service."""

    async def get_task(self, task_id: str) -> TaskMetadata | None:
        """Read application-owned task metadata by identifier."""
        ...

    async def save_task(self, task: TaskMetadata) -> None:
        """Persist the latest task state."""
        ...

    async def create_queued_if_inactive(self, task: TaskMetadata) -> bool:
        """Atomically create a task only when its username has no active owner."""
        ...

    async def get_active_task_id(self, username: str) -> str | None:
        """Return the current active task identifier for a username."""
        ...

    async def release_active(self, username: str, task_id: str) -> bool:
        """Release active ownership only for the matching task identifier."""
        ...

    async def get_fresh_task_id(self, username: str) -> str | None:
        """Return a recently completed task identifier when one exists."""
        ...

    async def close(self) -> None:
        """Close task-store resources."""
        ...


@dataclass(frozen=True, slots=True)
class TaskSubmission:
    """Task state returned from creation or reuse."""

    task: TaskMetadata
    reused: bool


EnqueueProfileSync = Callable[[str, str], None]
RunSync = Callable[..., Awaitable[object]]


def enqueue_profile_sync(username: str, task_id: str) -> None:
    """Publish one JSON-safe profile-sync task to its dedicated queue."""
    from app.tasks.profile_sync import profile_sync_task
    from app.workers.celery_app import PROFILE_SYNC_QUEUE

    profile_sync_task.apply_async(
        args=[username, task_id],
        task_id=task_id,
        queue=PROFILE_SYNC_QUEUE,
    )


class TaskService:
    """Coordinate API-side task ownership, reuse, and broker publication.

    The lifespan-scoped service owns an asynchronous Redis store. Redis scripts make
    username ownership atomic, while Celery publication runs off the event loop. A
    broker failure records safe terminal metadata and releases only this task's lock.
    """

    def __init__(
        self,
        store: AsyncTaskStore,
        enqueue: EnqueueProfileSync = enqueue_profile_sync,
        run_sync: RunSync = asyncio.to_thread,
    ) -> None:
        self._store = store
        self._enqueue = enqueue
        self._run_sync = run_sync

    async def submit_profile_sync(
        self,
        username: str,
        *,
        force: bool = False,
    ) -> TaskSubmission:
        """Create, reuse, or enqueue one durable profile synchronization task.

        Active queued/processing work always wins. Unless ``force`` is set, a recent
        completed task is also reused. New task metadata and username ownership are
        created atomically before broker publication, with bounded race recovery for
        stale ownership discovered between reads.

        Args:
            username: Letterboxd username; surrounding whitespace is ignored.
            force: Bypass completed-task freshness, but never active ownership.

        Returns:
            TaskSubmission: Public task metadata plus whether existing work was reused.

        Raises:
            TaskInfrastructureError: If Redis is unavailable, ownership cannot be
                resolved, or Celery publication fails.
        """
        normalized_username = username.strip()
        try:
            # Reuse live ownership first so forced callers cannot start concurrent
            # scraping for the same username.
            active = await self._get_active(normalized_username)
            if active is not None:
                return TaskSubmission(task=active, reused=True)

            if not force:
                fresh = await self._get_fresh(normalized_username)
                if fresh is not None:
                    return TaskSubmission(task=fresh, reused=True)

            # Atomically create metadata and ownership; a losing concurrent request
            # receives the winner's task instead of publishing duplicate work.
            task, created = await self._create_with_race_recovery(normalized_username)
            if not created:
                return TaskSubmission(task=task, reused=True)
            try:
                # Celery's synchronous producer is moved off the FastAPI event loop.
                await self._run_sync(
                    self._enqueue,
                    normalized_username,
                    task.task_id,
                )
            except Exception as exc:
                logger.exception(
                    "Unable to enqueue profile sync task_id=%s username=%s",
                    task.task_id,
                    normalized_username,
                )
                # Keep API-visible state truthful and release ownership conditionally
                # when publication failed after Redis creation.
                failed = task.failed(
                    TaskError(
                        code="profile_sync_unavailable",
                        message="Profile synchronization is temporarily unavailable.",
                    )
                )
                try:
                    await self._store.save_task(failed)
                    await self._store.release_active(
                        normalized_username,
                        task.task_id,
                    )
                except Exception:
                    logger.exception(
                        "Unable to clean failed enqueue task_id=%s", task.task_id
                    )
                raise TaskInfrastructureError from exc

            return TaskSubmission(task=task, reused=False)
        except TaskInfrastructureError:
            raise
        except Exception as exc:
            logger.exception(
                "Task infrastructure failure username=%s", normalized_username
            )
            raise TaskInfrastructureError from exc

    async def get_task(self, task_id: str) -> TaskMetadata | None:
        """Return application-owned task state without Celery state inference."""
        try:
            return await self._store.get_task(task_id)
        except Exception as exc:
            logger.exception("Unable to read task state task_id=%s", task_id)
            raise TaskInfrastructureError from exc

    async def close(self) -> None:
        """Close task infrastructure resources."""
        await self._store.close()

    async def _get_active(self, username: str) -> TaskMetadata | None:
        """Return live owned work and clean a stale active pointer when detected."""
        active_task_id = await self._store.get_active_task_id(username)
        if active_task_id is None:
            return None
        task = await self._store.get_task(active_task_id)
        if task is not None and task.status in {
            TaskStatus.QUEUED,
            TaskStatus.PROCESSING,
        }:
            return task
        await self._store.release_active(username, active_task_id)
        return None

    async def _get_fresh(self, username: str) -> TaskMetadata | None:
        """Return only a still-retained, successfully completed freshness target."""
        fresh_task_id = await self._store.get_fresh_task_id(username)
        if fresh_task_id is None:
            return None
        task = await self._store.get_task(fresh_task_id)
        if task is not None and task.status == TaskStatus.COMPLETED:
            return task
        return None

    async def _create_with_race_recovery(
        self, username: str
    ) -> tuple[TaskMetadata, bool]:
        """Acquire username ownership with bounded concurrent-race recovery.

        Returns:
            tuple[TaskMetadata, bool]: The newly created task and ``True``, or the
                concurrent winner and ``False``.

        Raises:
            TaskInfrastructureError: If three attempts cannot resolve ownership.
        """
        for _ in range(3):
            task = TaskMetadata.queued(str(uuid4()), username)
            if await self._store.create_queued_if_inactive(task):
                return task, True

            active = await self._get_active(username)
            if active is not None:
                return active, False
        raise TaskInfrastructureError("Unable to acquire profile synchronization lock")


_settings = get_settings()
_task_store = AsyncRedisTaskStore(
    _settings.TASK_STATE_REDIS_URL,
    task_ttl=_settings.TASK_RESULT_TTL_SECONDS,
    active_ttl=_settings.PROFILE_SYNC_ACTIVE_TTL_SECONDS,
    freshness_ttl=_settings.PROFILE_SYNC_FRESHNESS_SECONDS,
)
_task_service = TaskService(_task_store)


def get_task_service() -> TaskService:
    """Return the process-wide background task service."""
    return _task_service
