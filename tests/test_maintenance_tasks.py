"""Celery maintenance adapters reuse processors and honor singleton locks."""

import asyncio
from contextlib import contextmanager

from app.domain.maintenance import FilmQueueRunResult
from app.tasks import maintenance


class _Lock:
    def __init__(self, acquired: bool) -> None:
        self.acquired = acquired
        self.closed = False

    @contextmanager
    def held(self):
        yield self.acquired

    def close(self):
        self.closed = True


class _Bridge:
    def run(self, awaitable):
        return asyncio.run(awaitable)


def test_weekly_task_invokes_existing_bounded_film_queue_processor(monkeypatch) -> None:
    lock = _Lock(True)
    calls = []

    async def processor(*, batch_size):
        calls.append(batch_size)
        return FilmQueueRunResult(3, 3, 1, 1, 1, 0.1)

    monkeypatch.setattr(maintenance, "_lock", lambda _key: lock)
    monkeypatch.setattr(maintenance, "sync_film_queue", processor)
    monkeypatch.setattr(maintenance, "worker_async_bridge", _Bridge())
    result = maintenance.process_film_queue_task.run()
    assert calls == [maintenance.settings.FILM_QUEUE_BATCH_SIZE]
    assert result["status"] == "completed"
    assert result["processed_count"] == 3
    assert lock.closed


def test_duplicate_weekly_task_is_skipped_without_processor_call(monkeypatch) -> None:
    lock = _Lock(False)

    async def forbidden(*, batch_size):
        raise AssertionError(batch_size)

    monkeypatch.setattr(maintenance, "_lock", lambda _key: lock)
    monkeypatch.setattr(maintenance, "sync_film_queue", forbidden)
    result = maintenance.process_film_queue_task.run()
    assert result == {"status": "skipped_locked"}
    assert lock.closed
