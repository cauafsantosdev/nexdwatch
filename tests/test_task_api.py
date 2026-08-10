"""Tests for profile-sync submission and task polling routes."""

import asyncio

import pytest
from fastapi import HTTPException, Response

from app.api.routes.tasks import get_task_state, submit_profile_sync
from app.domain.task_state import TaskError, TaskMetadata, TaskResult, TaskStatus
from app.services.task_service import (
    TaskInfrastructureError,
    TaskSubmission,
)


class _RouteService:
    def __init__(
        self,
        *,
        submission: TaskSubmission | None = None,
        task: TaskMetadata | None = None,
        error: Exception | None = None,
    ) -> None:
        self.submission = submission
        self.task = task
        self.error = error
        self.submit_calls: list[tuple[str, bool]] = []

    async def submit_profile_sync(
        self, username: str, *, force: bool = False
    ) -> TaskSubmission:
        self.submit_calls.append((username, force))
        if self.error:
            raise self.error
        if self.submission is None:
            raise AssertionError("submission not configured")
        return self.submission

    async def get_task(self, _: str) -> TaskMetadata | None:
        if self.error:
            raise self.error
        return self.task


def test_new_profile_sync_returns_202_without_waiting() -> None:
    task = TaskMetadata.queued("task-1", "cinephile")
    service = _RouteService(submission=TaskSubmission(task=task, reused=False))
    response = Response()

    payload = asyncio.run(
        submit_profile_sync(
            username=" cinephile ",
            response=response,
            service=service,
            force=False,
        )
    )

    assert response.status_code == 202
    assert payload.task_id == "task-1"
    assert payload.status == TaskStatus.QUEUED
    assert not payload.reused
    assert service.submit_calls == [("cinephile", False)]


def test_active_reuse_returns_same_task_with_202() -> None:
    task = TaskMetadata.queued("task-1", "cinephile").processing()
    service = _RouteService(submission=TaskSubmission(task=task, reused=True))
    response = Response()

    payload = asyncio.run(
        submit_profile_sync(
            username="cinephile",
            response=response,
            service=service,
            force=True,
        )
    )

    assert response.status_code == 202
    assert payload.task_id == "task-1"
    assert payload.reused
    assert payload.status == TaskStatus.PROCESSING


def test_fresh_completed_reuse_returns_200() -> None:
    task = (
        TaskMetadata.queued("task-1", "cinephile")
        .processing()
        .completed(TaskResult(user_id=7, logs_count=20))
    )
    service = _RouteService(submission=TaskSubmission(task=task, reused=True))
    response = Response()

    payload = asyncio.run(
        submit_profile_sync(
            username="cinephile",
            response=response,
            service=service,
            force=False,
        )
    )

    assert response.status_code == 200
    assert payload.status == TaskStatus.COMPLETED
    assert payload.reused


@pytest.mark.parametrize(
    "task",
    [
        TaskMetadata.queued("queued", "cinephile"),
        TaskMetadata.queued("processing", "cinephile").processing(),
        TaskMetadata.queued("completed", "cinephile")
        .processing()
        .completed(TaskResult(user_id=7, logs_count=20)),
        TaskMetadata.queued("failed", "cinephile").failed(
            TaskError(code="profile_sync_failed", message="Safe failure")
        ),
    ],
)
def test_polling_maps_all_public_states(task: TaskMetadata) -> None:
    payload = asyncio.run(
        get_task_state(task_id=task.task_id, service=_RouteService(task=task))
    )

    assert payload.task_id == task.task_id
    assert payload.status == task.status
    assert payload.result is None if task.result is None else payload.result is not None
    assert payload.error is None if task.error is None else payload.error is not None


def test_unknown_task_returns_404() -> None:
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(get_task_state(task_id="unknown", service=_RouteService(task=None)))

    assert exc_info.value.status_code == 404


@pytest.mark.parametrize("route", ["submit", "poll"])
def test_task_infrastructure_failure_returns_503(route: str) -> None:
    service = _RouteService(error=TaskInfrastructureError())

    with pytest.raises(HTTPException) as exc_info:
        if route == "submit":
            asyncio.run(
                submit_profile_sync(
                    username="cinephile",
                    response=Response(),
                    service=service,
                    force=False,
                )
            )
        else:
            asyncio.run(get_task_state(task_id="task-1", service=service))

    assert exc_info.value.status_code == 503
