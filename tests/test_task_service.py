"""Tests for task deduplication, freshness, and enqueue recovery."""

import asyncio
from collections.abc import Callable
from unittest.mock import Mock

import pytest

from app.domain.task_state import TaskError, TaskMetadata, TaskResult, TaskStatus
from app.services.task_service import (
    TaskInfrastructureError,
    TaskService,
    enqueue_profile_sync,
)


class _FakeStore:
    def __init__(self) -> None:
        self.tasks: dict[str, TaskMetadata] = {}
        self.active: dict[str, str] = {}
        self.fresh: dict[str, str] = {}
        self.release_calls: list[tuple[str, str]] = []
        self.closed = False

    async def get_task(self, task_id: str) -> TaskMetadata | None:
        return self.tasks.get(task_id)

    async def save_task(self, task: TaskMetadata) -> None:
        self.tasks[task.task_id] = task

    async def create_queued_if_inactive(self, task: TaskMetadata) -> bool:
        if task.username in self.active:
            return False
        self.active[task.username] = task.task_id
        self.tasks[task.task_id] = task
        return True

    async def get_active_task_id(self, username: str) -> str | None:
        return self.active.get(username)

    async def release_active(self, username: str, task_id: str) -> bool:
        self.release_calls.append((username, task_id))
        if self.active.get(username) != task_id:
            return False
        del self.active[username]
        return True

    async def get_fresh_task_id(self, username: str) -> str | None:
        return self.fresh.get(username)

    async def close(self) -> None:
        self.closed = True


def _recording_enqueue(calls: list[tuple[str, str]]) -> Callable[[str, str], None]:
    def enqueue(username: str, task_id: str) -> None:
        calls.append((username, task_id))

    return enqueue


async def _run_inline(function: Callable, *args: object) -> object:
    return function(*args)


def test_first_submission_enqueues_and_active_duplicate_reuses() -> None:
    store = _FakeStore()
    enqueue_calls: list[tuple[str, str]] = []
    service = TaskService(store, _recording_enqueue(enqueue_calls), _run_inline)

    first = asyncio.run(service.submit_profile_sync(" cinephile "))
    second = asyncio.run(service.submit_profile_sync("cinephile"))

    assert not first.reused
    assert first.task.status == TaskStatus.QUEUED
    assert second.reused
    assert second.task.task_id == first.task.task_id
    assert enqueue_calls == [("cinephile", first.task.task_id)]


def test_concurrent_submissions_create_only_one_task() -> None:
    store = _FakeStore()
    enqueue_calls: list[tuple[str, str]] = []
    service = TaskService(store, _recording_enqueue(enqueue_calls), _run_inline)

    async def submit_twice():
        return await asyncio.gather(
            service.submit_profile_sync("cinephile"),
            service.submit_profile_sync("cinephile"),
        )

    first, second = asyncio.run(submit_twice())

    assert first.task.task_id == second.task.task_id
    assert sorted([first.reused, second.reused]) == [False, True]
    assert len(enqueue_calls) == 1


def test_stale_active_lock_is_released_before_new_task() -> None:
    store = _FakeStore()
    store.active["cinephile"] = "expired-task"
    calls: list[tuple[str, str]] = []
    service = TaskService(store, _recording_enqueue(calls), _run_inline)

    submission = asyncio.run(service.submit_profile_sync("cinephile"))

    assert not submission.reused
    assert store.release_calls[0] == ("cinephile", "expired-task")
    assert store.active["cinephile"] == submission.task.task_id


def test_recent_completed_task_is_reused_unless_forced() -> None:
    store = _FakeStore()
    completed = (
        TaskMetadata.queued("completed-task", "cinephile")
        .processing()
        .completed(TaskResult(user_id=7, logs_count=100))
    )
    store.tasks[completed.task_id] = completed
    store.fresh["cinephile"] = completed.task_id
    calls: list[tuple[str, str]] = []
    service = TaskService(store, _recording_enqueue(calls), _run_inline)

    reused = asyncio.run(service.submit_profile_sync("cinephile"))
    forced = asyncio.run(service.submit_profile_sync("cinephile", force=True))

    assert reused.reused
    assert reused.task == completed
    assert not forced.reused
    assert forced.task.task_id != completed.task_id
    assert len(calls) == 1


def test_force_still_reuses_processing_task() -> None:
    store = _FakeStore()
    processing = TaskMetadata.queued("active-task", "cinephile").processing()
    store.tasks[processing.task_id] = processing
    store.active["cinephile"] = processing.task_id
    calls: list[tuple[str, str]] = []
    service = TaskService(store, _recording_enqueue(calls), _run_inline)

    submission = asyncio.run(service.submit_profile_sync("cinephile", force=True))

    assert submission.reused
    assert submission.task == processing
    assert calls == []


def test_failed_fresh_reference_is_not_reused() -> None:
    store = _FakeStore()
    failed = TaskMetadata.queued("failed-task", "cinephile").failed(
        TaskError(code="failed", message="safe")
    )
    store.tasks[failed.task_id] = failed
    store.fresh["cinephile"] = failed.task_id
    calls: list[tuple[str, str]] = []
    service = TaskService(store, _recording_enqueue(calls), _run_inline)

    submission = asyncio.run(service.submit_profile_sync("cinephile"))

    assert not submission.reused
    assert submission.task.task_id != failed.task_id
    assert len(calls) == 1


def test_enqueue_failure_marks_failed_releases_lock_and_returns_outage() -> None:
    store = _FakeStore()

    def fail_enqueue(_: str, __: str) -> None:
        raise ConnectionError("broker unavailable")

    service = TaskService(store, fail_enqueue, _run_inline)

    with pytest.raises(TaskInfrastructureError):
        asyncio.run(service.submit_profile_sync("cinephile"))

    task = next(iter(store.tasks.values()))
    assert task.status == TaskStatus.FAILED
    assert task.error is not None
    assert task.error.message == "Profile synchronization is temporarily unavailable."
    assert "cinephile" not in store.active


def test_unknown_task_and_close_are_delegated() -> None:
    store = _FakeStore()
    service = TaskService(store, lambda *_: None, _run_inline)

    assert asyncio.run(service.get_task("unknown")) is None
    asyncio.run(service.close())
    assert store.closed


def test_celery_enqueue_uses_only_primitive_arguments(monkeypatch) -> None:
    from app.tasks.profile_sync import profile_sync_task

    apply_async = Mock()
    monkeypatch.setattr(profile_sync_task, "apply_async", apply_async)

    enqueue_profile_sync("cinephile", "task-1")

    apply_async.assert_called_once_with(
        args=["cinephile", "task-1"],
        task_id="task-1",
        queue="profile_sync",
    )
