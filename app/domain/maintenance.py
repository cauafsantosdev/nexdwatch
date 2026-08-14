"""Typed operational results for recommendation maintenance."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class RetrainingReason(StrEnum):
    LEGACY_MODEL_BOOTSTRAP = "LEGACY_MODEL_BOOTSTRAP"
    NEW_USERS_THRESHOLD = "NEW_USERS_THRESHOLD"
    NEW_FILMS_THRESHOLD = "NEW_FILMS_THRESHOLD"
    MODEL_AGE_THRESHOLD = "MODEL_AGE_THRESHOLD"
    FORCED = "FORCED"
    NONE = "NONE"


@dataclass(frozen=True, slots=True)
class TrainingStatistics:
    measured_at: datetime
    eligible_user_count: int
    rated_interaction_count: int
    rated_film_ids: tuple[int, ...]

    @property
    def model_film_count(self) -> int:
        return len(self.rated_film_ids)


@dataclass(frozen=True, slots=True)
class TrainedModelStatistics:
    trained_at: datetime
    eligible_user_count: int
    rated_interaction_count: int
    model_film_count: int
    model_film_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RetrainingDeltas:
    eligible_users: int
    rated_interactions: int
    new_model_films: int
    model_age_days: float


@dataclass(frozen=True, slots=True)
class RetrainingDecision:
    should_retrain: bool
    reasons: tuple[RetrainingReason, ...]
    current_stats: TrainingStatistics
    trained_stats: TrainedModelStatistics | None
    deltas: RetrainingDeltas


@dataclass(frozen=True, slots=True)
class FilmQueueRunResult:
    pending_count: int
    processed_count: int
    success_count: int
    filtered_count: int
    failed_count: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class CatalogRefreshResult:
    target_years: tuple[int, ...]
    selected_count: int
    updated_count: int
    failed_count: int
    dry_run: bool
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class PromotionResult:
    model_version: str
    previous_version: str | None
    serving_reload_required: bool = True
