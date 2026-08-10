"""Transport schemas for durable background tasks."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.domain.task_state import TaskStatus


class TaskResultResponse(BaseModel):
    """Successful profile synchronization result."""

    model_config = ConfigDict(from_attributes=True)

    user_id: int | None
    logs_count: int


class TaskErrorResponse(BaseModel):
    """Safe public task failure."""

    model_config = ConfigDict(from_attributes=True)

    code: str
    message: str


class TaskSubmissionResponse(BaseModel):
    """Profile-sync task creation or reuse response."""

    task_id: str
    username: str
    status: TaskStatus
    reused: bool


class TaskStateResponse(BaseModel):
    """Complete application-owned task state."""

    model_config = ConfigDict(from_attributes=True)

    task_id: str
    type: Literal["profile_sync"]
    username: str
    status: TaskStatus
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    attempt: int
    result: TaskResultResponse | None
    error: TaskErrorResponse | None
