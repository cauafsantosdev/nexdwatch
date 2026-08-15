"""Crash-recoverable Redis locks for singleton maintenance operations."""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol

import redis

_COMPARE_AND_DELETE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""


class RedisLockClient(Protocol):
    """Minimal synchronous Redis operations required by the lock adapter."""

    def set(self, name: str, value: str, *, nx: bool, ex: int) -> object:
        """Set a token only when absent and apply a crash-recovery expiry."""
        ...

    def eval(self, script: str, numkeys: int, *keys_and_args: object) -> object:
        """Evaluate the atomic compare-and-delete release script."""
        ...

    def close(self) -> None:
        """Close the client connection."""
        ...


class MaintenanceLock:
    """Token-owned singleton lock whose TTL makes worker crashes recoverable.

    Each instance owns a random token. Release uses compare-and-delete so an expired
    lock can never delete a successor's ownership after a slow worker resumes.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        key: str,
        ttl_seconds: int,
        client: RedisLockClient | None = None,
    ) -> None:
        """Create one token owner for a namespaced maintenance operation.

        The Redis client is process-local and must be closed after use. ``ttl_seconds``
        bounds stale ownership if a worker dies without reaching release.
        """
        self._client = client or redis.from_url(redis_url, decode_responses=True)
        self._key = f"nexdwatch:maintenance:lock:{key}"
        self._ttl_seconds = ttl_seconds
        self._token = uuid.uuid4().hex
        self._acquired = False

    def acquire(self) -> bool:
        """Attempt one non-blocking Redis acquisition with crash-recovery expiry.

        Returns:
            bool: ``True`` only when this instance created the lock and owns its token.
        """
        self._acquired = bool(
            self._client.set(
                self._key,
                self._token,
                nx=True,
                ex=self._ttl_seconds,
            )
        )
        return self._acquired

    def release(self) -> bool:
        """Release only when this instance still owns the Redis token.

        The compare-and-delete script prevents an expired slow worker from deleting a
        successor's lock after ownership has changed.

        Returns:
            bool: ``True`` only when Redis deleted this instance's current ownership.
        """
        if not self._acquired:
            return False
        released = bool(
            self._client.eval(_COMPARE_AND_DELETE, 1, self._key, self._token)
        )
        self._acquired = False
        return released

    def close(self) -> None:
        """Close the underlying Redis client after maintenance finishes."""
        self._client.close()

    @contextmanager
    def held(self) -> Iterator[bool]:
        """Yield acquisition state and token-safely release successful ownership.

        Release runs in ``finally`` for normal completion and exceptions; the TTL is
        still the recovery boundary for process death or lost Redis connectivity.
        """
        acquired = self.acquire()
        try:
            yield acquired
        finally:
            if acquired:
                self.release()
