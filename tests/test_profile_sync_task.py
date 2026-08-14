"""Tests for the Celery profile-sync execution adapter."""

import asyncio
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from celery.exceptions import Retry

from app.core.config import get_settings
from app.domain.task_state import TaskMetadata, TaskResult, TaskStatus
from app.scraper.user_scraping import (
    ProfileScrapeError,
    TransientProfileScrapeError,
)
from app.services.profile_service import ProfileSyncResult
from app.tasks.profile_sync import execute_profile_sync
from app.workers import async_bridge as async_bridge_module
from app.workers.async_bridge import WorkerAsyncBridge
from app.workers.celery_app import PROFILE_SYNC_QUEUE, celery_app


class _WorkerStore:
    def __init__(self, task: TaskMetadata) -> None:
        self.task = task
        self.saved: list[TaskMetadata] = []
        self.fresh: list[tuple[str, str]] = []
        self.released: list[tuple[str, str]] = []
        self.claimed = True

    def get_task(self, task_id: str) -> TaskMetadata | None:
        return self.task if self.task.task_id == task_id else None

    def save_task(self, task: TaskMetadata) -> None:
        self.task = task
        self.saved.append(task)

    def release_active(self, username: str, task_id: str) -> bool:
        self.released.append((username, task_id))
        return True

    def claim_active(self, username: str, task_id: str) -> bool:
        return self.claimed

    def set_fresh(self, username: str, task_id: str) -> None:
        self.fresh.append((username, task_id))

    def close(self) -> None:
        return None


class _ProfileService:
    def __init__(self, result: ProfileSyncResult | Exception) -> None:
        self.result = result
        self.calls: list[str] = []

    async def sync_profile(self, username: str) -> ProfileSyncResult:
        self.calls.append(username)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _task_context(retries: int = 0, retry_effect: Exception | None = None):
    retry = Mock()
    if retry_effect is not None:
        retry.side_effect = retry_effect
    return SimpleNamespace(
        request=SimpleNamespace(retries=retries),
        max_retries=2,
        retry=retry,
    )


def test_worker_transitions_processing_to_completed_and_sets_freshness() -> None:
    metadata = TaskMetadata.queued("task-1", "cinephile")
    store = _WorkerStore(metadata)
    profile_service = _ProfileService(ProfileSyncResult(user_id=7, logs_count=689))
    bridge = WorkerAsyncBridge()

    execute_profile_sync(
        _task_context(),
        "cinephile",
        "task-1",
        store=store,
        profile_service=profile_service,
        bridge=bridge,
    )
    bridge.close()

    assert [state.status for state in store.saved] == [
        TaskStatus.PROCESSING,
        TaskStatus.COMPLETED,
    ]
    assert store.task.result is not None
    assert store.task.result.user_id == 7
    assert store.task.result.logs_count == 689
    assert store.fresh == [("cinephile", "task-1")]
    assert store.released == [("cinephile", "task-1")]


def test_non_retryable_failure_is_safe_terminal_and_releases_lock() -> None:
    store = _WorkerStore(TaskMetadata.queued("task-1", "cinephile"))
    service = _ProfileService(ProfileScrapeError("raw upstream detail"))
    bridge = WorkerAsyncBridge()

    with pytest.raises(ProfileScrapeError):
        execute_profile_sync(
            _task_context(),
            "cinephile",
            "task-1",
            store=store,
            profile_service=service,
            bridge=bridge,
        )
    bridge.close()

    assert store.task.status == TaskStatus.FAILED
    assert store.task.error is not None
    assert store.task.error.code == "profile_sync_failed"
    assert "raw upstream" not in store.task.error.message
    assert store.fresh == []
    assert store.released == [("cinephile", "task-1")]


def test_transient_failure_retries_without_terminal_cleanup() -> None:
    store = _WorkerStore(TaskMetadata.queued("task-1", "cinephile"))
    service = _ProfileService(TransientProfileScrapeError("temporary"))
    bridge = WorkerAsyncBridge()
    retry_signal = Retry("retrying")
    task = _task_context(retries=0, retry_effect=retry_signal)

    with pytest.raises(Retry):
        execute_profile_sync(
            task,
            "cinephile",
            "task-1",
            store=store,
            profile_service=service,
            bridge=bridge,
        )
    bridge.close()

    assert store.task.status == TaskStatus.PROCESSING
    assert store.task.attempt == 1
    assert store.fresh == []
    assert store.released == []
    task.retry.assert_called_once()
    assert 5 <= task.retry.call_args.kwargs["countdown"] <= 7


def test_transient_failure_after_retry_budget_becomes_terminal() -> None:
    store = _WorkerStore(TaskMetadata.queued("task-1", "cinephile"))
    service = _ProfileService(TransientProfileScrapeError("temporary"))
    bridge = WorkerAsyncBridge()
    task = _task_context(retries=2)

    with pytest.raises(TransientProfileScrapeError):
        execute_profile_sync(
            task,
            "cinephile",
            "task-1",
            store=store,
            profile_service=service,
            bridge=bridge,
        )
    bridge.close()

    assert store.task.status == TaskStatus.FAILED
    assert store.released == [("cinephile", "task-1")]
    task.retry.assert_not_called()


def test_terminal_redelivery_does_not_repeat_profile_sync() -> None:
    completed = (
        TaskMetadata.queued("task-1", "cinephile")
        .processing()
        .completed(TaskResult(user_id=7, logs_count=1))
    )
    store = _WorkerStore(completed)
    service = _ProfileService(ProfileSyncResult(user_id=7, logs_count=1))
    bridge = WorkerAsyncBridge()

    execute_profile_sync(
        _task_context(),
        "cinephile",
        "task-1",
        store=store,
        profile_service=service,
        bridge=bridge,
    )
    bridge.close()

    assert service.calls == []
    assert store.saved == []


def test_superseded_delivery_does_not_start_concurrent_scrape() -> None:
    store = _WorkerStore(TaskMetadata.queued("task-1", "cinephile"))
    store.claimed = False
    service = _ProfileService(ProfileSyncResult(user_id=7, logs_count=1))
    bridge = WorkerAsyncBridge()

    execute_profile_sync(
        _task_context(),
        "cinephile",
        "task-1",
        store=store,
        profile_service=service,
        bridge=bridge,
    )
    bridge.close()

    assert service.calls == []
    assert store.task.status == TaskStatus.FAILED
    assert store.released == []


def test_async_bridge_reuses_runner_across_calls(monkeypatch) -> None:
    disposed = 0

    async def dispose() -> None:
        nonlocal disposed
        disposed += 1

    monkeypatch.setattr(
        async_bridge_module,
        "engine",
        SimpleNamespace(dispose=dispose),
    )
    bridge = WorkerAsyncBridge()

    async def loop_identity() -> int:
        return id(asyncio.get_running_loop())

    first_loop = bridge.run(loop_identity())
    second_loop = bridge.run(loop_identity())
    bridge.close()

    assert first_loop == second_loop
    assert disposed == 1


def test_celery_task_uses_durable_json_configuration() -> None:
    from app.tasks.profile_sync import profile_sync_task

    assert celery_app.conf.accept_content == ["json"]
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.result_backend is None
    assert celery_app.conf.worker_prefetch_multiplier == 1
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True
    assert celery_app.conf.broker_connection_retry_on_startup is True
    assert celery_app.conf.broker_transport_options["visibility_timeout"] == (
        get_settings().CELERY_BROKER_VISIBILITY_TIMEOUT_SECONDS
    )
    assert profile_sync_task.queue == PROFILE_SYNC_QUEUE
    assert profile_sync_task.ignore_result is True
    assert profile_sync_task.max_retries == 2
    assert profile_sync_task.soft_time_limit == 300
    assert profile_sync_task.time_limit == 330


def test_profile_sync_worker_import_does_not_eagerly_load_neural_runtime() -> None:
    script = (
        "import sys; import app.tasks.profile_sync; "
        "blocked={'torch','experiments.neural_retrieval.training',"
        "'experiments.neural_retrieval.service'}; "
        "assert blocked.isdisjoint(sys.modules), blocked.intersection(sys.modules)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
