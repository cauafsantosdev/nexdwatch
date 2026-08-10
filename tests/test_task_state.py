"""Tests for Redis-backed application task metadata."""

import asyncio
from unittest.mock import AsyncMock, Mock

from app.domain.task_state import TaskError, TaskMetadata, TaskResult, TaskStatus
from app.infrastructure.task_store import (
    AsyncRedisTaskStore,
    SyncRedisTaskStore,
    active_key,
    fresh_key,
    task_key,
)


def test_task_metadata_serializes_all_public_states() -> None:
    queued = TaskMetadata.queued("task-1", "cinephile")
    processing = queued.processing()
    completed = processing.completed(TaskResult(user_id=7, logs_count=689))
    failed = processing.failed(TaskError(code="failed", message="Safe message"))

    assert TaskMetadata.from_json(queued.to_json()) == queued
    assert TaskMetadata.from_json(processing.to_json()) == processing
    assert TaskMetadata.from_json(completed.to_json()) == completed
    assert TaskMetadata.from_json(failed.to_json()) == failed
    assert processing.status == TaskStatus.PROCESSING
    assert processing.started_at is not None
    assert processing.attempt == 1
    assert completed.result == TaskResult(user_id=7, logs_count=689)
    assert failed.error == TaskError(code="failed", message="Safe message")


def test_async_store_applies_ttl_and_returns_missing_as_unknown() -> None:
    client = AsyncMock()
    client.get.return_value = None
    store = AsyncRedisTaskStore(
        "redis://unused",
        task_ttl=86_400,
        active_ttl=600,
        freshness_ttl=900,
        client=client,
    )
    task = TaskMetadata.queued("task-1", "cinephile")

    asyncio.run(store.save_task(task))
    missing = asyncio.run(store.get_task("missing"))
    asyncio.run(store.set_fresh("cinephile", "task-1"))

    client.set.assert_any_await(
        task_key("task-1"),
        task.to_json(),
        ex=86_400,
    )
    client.set.assert_any_await(
        fresh_key("cinephile"),
        "task-1",
        ex=900,
    )
    assert missing is None


def test_atomic_create_and_compare_delete_use_single_redis_scripts() -> None:
    client = AsyncMock()
    client.eval.side_effect = [1, 1, 0]
    store = AsyncRedisTaskStore(
        "redis://unused",
        task_ttl=86_400,
        active_ttl=600,
        freshness_ttl=900,
        client=client,
    )
    task = TaskMetadata.queued("task-1", "cinephile")

    assert asyncio.run(store.create_queued_if_inactive(task))
    assert asyncio.run(store.release_active("cinephile", "task-1"))
    assert not asyncio.run(store.release_active("cinephile", "wrong-task"))

    create_call = client.eval.await_args_list[0]
    assert create_call.args[1:4] == (
        2,
        active_key("cinephile"),
        task_key("task-1"),
    )
    assert create_call.args[4] == "task-1"
    assert create_call.args[5] == 600
    release_call = client.eval.await_args_list[1]
    assert release_call.args[1:] == (
        1,
        active_key("cinephile"),
        "task-1",
    )


def test_sync_store_releases_only_matching_lock_and_sets_freshness() -> None:
    client = Mock()
    client.eval.side_effect = [1, 0, 1]
    store = SyncRedisTaskStore(
        "redis://unused",
        task_ttl=86_400,
        active_ttl=600,
        freshness_ttl=900,
        client=client,
    )

    assert store.release_active("cinephile", "task-1")
    assert not store.release_active("cinephile", "task-2")
    assert store.claim_active("cinephile", "task-1")
    store.set_fresh("cinephile", "task-1")

    client.set.assert_called_once_with(
        fresh_key("cinephile"),
        "task-1",
        ex=900,
    )
