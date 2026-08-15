"""Application-owned background-task state."""

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal


class TaskStatus(StrEnum):
    """Public task states supported by NexdWatch."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Successful profile synchronization result."""

    user_id: int | None
    logs_count: int


@dataclass(frozen=True, slots=True)
class TaskError:
    """Safe public task failure."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class TaskMetadata:
    """Immutable, Redis-persisted state for one pollable profile-sync task.

    Celery delivery state is deliberately not authoritative. API and worker replace
    this value through explicit queued, processing, completed, and failed transitions;
    only ``TaskError`` values safe for public polling are stored.
    """

    task_id: str
    type: Literal["profile_sync"]
    username: str
    status: TaskStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempt: int = 0
    result: TaskResult | None = None
    error: TaskError | None = None

    @classmethod
    def queued(cls, task_id: str, username: str) -> "TaskMetadata":
        """Create initial queued task metadata."""
        return cls(
            task_id=task_id,
            type="profile_sync",
            username=username,
            status=TaskStatus.QUEUED,
            created_at=datetime.now(UTC),
        )

    def processing(self) -> "TaskMetadata":
        """Return processing state with a fresh start time and incremented attempt.

        Prior terminal payloads are cleared so a retry cannot expose stale result or
        error data while work is active.
        """
        return replace(
            self,
            status=TaskStatus.PROCESSING,
            started_at=datetime.now(UTC),
            finished_at=None,
            attempt=self.attempt + 1,
            result=None,
            error=None,
        )

    def completed(self, result: TaskResult) -> "TaskMetadata":
        """Return terminal success with a finish time and no prior public error."""
        return replace(
            self,
            status=TaskStatus.COMPLETED,
            finished_at=datetime.now(UTC),
            result=result,
            error=None,
        )

    def failed(self, error: TaskError) -> "TaskMetadata":
        """Return terminal failure with product-safe error and no stale result."""
        return replace(
            self,
            status=TaskStatus.FAILED,
            finished_at=datetime.now(UTC),
            result=None,
            error=error,
        )

    def to_json(self) -> str:
        """Serialize task metadata to compact, Celery-independent Redis JSON."""
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "type": self.type,
            "username": self.username,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "attempt": self.attempt,
            "result": (
                {"user_id": self.result.user_id, "logs_count": self.result.logs_count}
                if self.result
                else None
            ),
            "error": (
                {"code": self.error.code, "message": self.error.message}
                if self.error
                else None
            ),
        }
        return json.dumps(payload, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str | bytes) -> "TaskMetadata":
        """Deserialize and type-normalize task metadata read from Redis.

        Raises:
            ValueError: If JSON is invalid, the task type/status is unsupported, or a
                required value cannot be normalized.
            KeyError: If required metadata fields are absent.
        """
        payload = json.loads(value)
        if payload.get("type") != "profile_sync":
            raise ValueError("unsupported task type")
        result_payload = payload.get("result")
        error_payload = payload.get("error")
        return cls(
            task_id=str(payload["task_id"]),
            type="profile_sync",
            username=str(payload["username"]),
            status=TaskStatus(payload["status"]),
            created_at=datetime.fromisoformat(payload["created_at"]),
            started_at=(
                datetime.fromisoformat(payload["started_at"])
                if payload.get("started_at")
                else None
            ),
            finished_at=(
                datetime.fromisoformat(payload["finished_at"])
                if payload.get("finished_at")
                else None
            ),
            attempt=int(payload.get("attempt", 0)),
            result=(
                TaskResult(
                    user_id=result_payload.get("user_id"),
                    logs_count=int(result_payload["logs_count"]),
                )
                if result_payload
                else None
            ),
            error=(
                TaskError(
                    code=str(error_payload["code"]),
                    message=str(error_payload["message"]),
                )
                if error_payload
                else None
            ),
        )
