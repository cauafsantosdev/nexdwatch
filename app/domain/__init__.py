"""Domain representations used across application boundaries."""

from .profiles import ScrapedProfile, ScrapedWatch
from .recommendations import Recommendation, RecommendationResult
from .task_state import TaskError, TaskMetadata, TaskResult, TaskStatus

__all__ = [
    "Recommendation",
    "RecommendationResult",
    "ScrapedProfile",
    "ScrapedWatch",
    "TaskError",
    "TaskMetadata",
    "TaskResult",
    "TaskStatus",
]
