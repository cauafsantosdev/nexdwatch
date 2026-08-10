"""Redis-backed storage for application-owned task metadata."""

import logging

import redis
import redis.asyncio as async_redis

from app.domain.task_state import TaskMetadata

logger = logging.getLogger(__name__)

_CREATE_TASK_SCRIPT = """
if redis.call('EXISTS', KEYS[1]) == 0 then
    redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
    redis.call('SET', KEYS[2], ARGV[3], 'EX', ARGV[4])
    return 1
end
return 0
"""

_COMPARE_AND_DELETE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""

_CLAIM_ACTIVE_SCRIPT = """
local owner = redis.call('GET', KEYS[1])
if not owner or owner == ARGV[1] then
    redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
    return 1
end
return 0
"""


def task_key(task_id: str) -> str:
    """Return the namespaced task metadata key."""
    return f"nexdwatch:task:{task_id}"


def active_key(username: str) -> str:
    """Return the namespaced active profile-sync key."""
    return f"nexdwatch:profile-sync:active:{username}"


def fresh_key(username: str) -> str:
    """Return the namespaced successful profile-sync key."""
    return f"nexdwatch:profile-sync:fresh:{username}"


class AsyncRedisTaskStore:
    """Non-blocking Redis task store used by FastAPI services."""

    def __init__(
        self,
        redis_url: str,
        *,
        task_ttl: int,
        active_ttl: int,
        freshness_ttl: int,
        client: async_redis.Redis | None = None,
    ) -> None:
        self._client = client or async_redis.from_url(
            redis_url,
            decode_responses=True,
        )
        self._task_ttl = task_ttl
        self._active_ttl = active_ttl
        self._freshness_ttl = freshness_ttl

    async def get_task(self, task_id: str) -> TaskMetadata | None:
        """Return task metadata, or None when missing/expired."""
        value = await self._client.get(task_key(task_id))
        return TaskMetadata.from_json(value) if value is not None else None

    async def save_task(self, task: TaskMetadata) -> None:
        """Store task metadata with bounded retention."""
        await self._client.set(
            task_key(task.task_id), task.to_json(), ex=self._task_ttl
        )

    async def create_queued_if_inactive(self, task: TaskMetadata) -> bool:
        """Atomically acquire the username lock and write queued metadata."""
        created = await self._client.eval(
            _CREATE_TASK_SCRIPT,
            2,
            active_key(task.username),
            task_key(task.task_id),
            task.task_id,
            self._active_ttl,
            task.to_json(),
            self._task_ttl,
        )
        return bool(created)

    async def get_active_task_id(self, username: str) -> str | None:
        """Return the active task ID for a username."""
        return await self._client.get(active_key(username))

    async def release_active(self, username: str, task_id: str) -> bool:
        """Delete an active lock only when owned by task_id."""
        deleted = await self._client.eval(
            _COMPARE_AND_DELETE_SCRIPT,
            1,
            active_key(username),
            task_id,
        )
        return bool(deleted)

    async def get_fresh_task_id(self, username: str) -> str | None:
        """Return the recently completed task ID for a username."""
        return await self._client.get(fresh_key(username))

    async def set_fresh(self, username: str, task_id: str) -> None:
        """Store a recent successful synchronization reference."""
        await self._client.set(
            fresh_key(username),
            task_id,
            ex=self._freshness_ttl,
        )

    async def close(self) -> None:
        """Close the lazy asynchronous Redis client."""
        await self._client.aclose()


class SyncRedisTaskStore:
    """Synchronous Redis task store used inside Celery workers."""

    def __init__(
        self,
        redis_url: str,
        *,
        task_ttl: int,
        active_ttl: int,
        freshness_ttl: int,
        client: redis.Redis | None = None,
    ) -> None:
        self._client = client or redis.from_url(redis_url, decode_responses=True)
        self._task_ttl = task_ttl
        self._active_ttl = active_ttl
        self._freshness_ttl = freshness_ttl

    def get_task(self, task_id: str) -> TaskMetadata | None:
        """Return task metadata, or None when missing/expired."""
        value = self._client.get(task_key(task_id))
        return TaskMetadata.from_json(value) if value is not None else None

    def save_task(self, task: TaskMetadata) -> None:
        """Store task metadata with bounded retention."""
        self._client.set(task_key(task.task_id), task.to_json(), ex=self._task_ttl)

    def release_active(self, username: str, task_id: str) -> bool:
        """Delete an active lock only when owned by task_id."""
        deleted = self._client.eval(
            _COMPARE_AND_DELETE_SCRIPT,
            1,
            active_key(username),
            task_id,
        )
        return bool(deleted)

    def claim_active(self, username: str, task_id: str) -> bool:
        """Acquire an expired lock or refresh it only for the same owner."""
        claimed = self._client.eval(
            _CLAIM_ACTIVE_SCRIPT,
            1,
            active_key(username),
            task_id,
            self._active_ttl,
        )
        return bool(claimed)

    def set_fresh(self, username: str, task_id: str) -> None:
        """Store a recent successful synchronization reference."""
        self._client.set(
            fresh_key(username),
            task_id,
            ex=self._freshness_ttl,
        )

    def close(self) -> None:
        """Close worker Redis connections."""
        self._client.close()
