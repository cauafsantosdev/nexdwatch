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
    def set(self, name: str, value: str, *, nx: bool, ex: int) -> object: ...

    def eval(self, script: str, numkeys: int, *keys_and_args: object) -> object: ...

    def close(self) -> None: ...


class MaintenanceLock:
    """Token-owned lock whose TTL makes worker crashes recoverable."""

    def __init__(
        self,
        redis_url: str,
        *,
        key: str,
        ttl_seconds: int,
        client: RedisLockClient | None = None,
    ) -> None:
        self._client = client or redis.from_url(redis_url, decode_responses=True)
        self._key = f"nexdwatch:maintenance:lock:{key}"
        self._ttl_seconds = ttl_seconds
        self._token = uuid.uuid4().hex
        self._acquired = False

    def acquire(self) -> bool:
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
        if not self._acquired:
            return False
        released = bool(
            self._client.eval(_COMPARE_AND_DELETE, 1, self._key, self._token)
        )
        self._acquired = False
        return released

    def close(self) -> None:
        self._client.close()

    @contextmanager
    def held(self) -> Iterator[bool]:
        acquired = self.acquire()
        try:
            yield acquired
        finally:
            if acquired:
                self.release()
