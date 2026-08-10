"""Durable background-task submission and polling routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, status

from app.api.schemas.tasks import TaskStateResponse, TaskSubmissionResponse
from app.domain.task_state import TaskStatus
from app.services.task_service import (
    TaskInfrastructureError,
    TaskService,
    get_task_service,
)

router = APIRouter()


@router.post(
    "/users/{username}/sync-logs",
    response_model=TaskSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_profile_sync(
    username: Annotated[str, Path(min_length=1, max_length=15)],
    response: Response,
    service: Annotated[TaskService, Depends(get_task_service)],
    force: Annotated[bool, Query()] = False,
) -> TaskSubmissionResponse:
    """Create or reuse a durable Letterboxd profile-sync task."""
    response.status_code = status.HTTP_202_ACCEPTED
    normalized_username = username.strip()
    if not normalized_username:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Username cannot be blank.",
        )
    try:
        submission = await service.submit_profile_sync(
            normalized_username,
            force=force,
        )
    except TaskInfrastructureError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Profile synchronization is temporarily unavailable.",
        ) from exc

    if submission.reused and submission.task.status == TaskStatus.COMPLETED:
        response.status_code = status.HTTP_200_OK

    return TaskSubmissionResponse(
        task_id=submission.task.task_id,
        username=submission.task.username,
        status=submission.task.status,
        reused=submission.reused,
    )


@router.get("/tasks/{task_id}", response_model=TaskStateResponse)
async def get_task_state(
    task_id: Annotated[str, Path(min_length=1, max_length=100)],
    service: Annotated[TaskService, Depends(get_task_service)],
) -> TaskStateResponse:
    """Return application-owned task state without Celery state inference."""
    try:
        task = await service.get_task(task_id)
    except TaskInfrastructureError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task status is temporarily unavailable.",
        ) from exc
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found.",
        )
    return TaskStateResponse.model_validate(task)
