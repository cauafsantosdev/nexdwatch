"""Typed operational results for recommendation maintenance."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class RetrainingReason(StrEnum):
    """Explicit operational reasons that can make production retraining eligible."""

    LEGACY_MODEL_BOOTSTRAP = "LEGACY_MODEL_BOOTSTRAP"
    NEW_USERS_THRESHOLD = "NEW_USERS_THRESHOLD"
    NEW_FILMS_THRESHOLD = "NEW_FILMS_THRESHOLD"
    MODEL_AGE_THRESHOLD = "MODEL_AGE_THRESHOLD"
    FORCED = "FORCED"
    NONE = "NONE"


@dataclass(frozen=True, slots=True)
class TrainingStatistics:
    """Current deduplicated PostgreSQL training-universe measurements."""

    measured_at: datetime
    eligible_user_count: int
    rated_interaction_count: int
    rated_film_ids: tuple[int, ...]

    @property
    def model_film_count(self) -> int:
        """Return the number of distinct rated films in the current snapshot."""
        return len(self.rated_film_ids)


@dataclass(frozen=True, slots=True)
class TrainedModelStatistics:
    """Authoritative baseline counters from a selected versioned manifest."""

    trained_at: datetime
    eligible_user_count: int
    rated_interaction_count: int
    model_film_count: int
    model_film_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RetrainingDeltas:
    """Differences between current snapshot state and the selected baseline."""

    eligible_users: int
    rated_interactions: int
    new_model_films: int
    model_age_days: float


@dataclass(frozen=True, slots=True)
class RetrainingDecision:
    """Typed threshold decision used by diagnostics and scheduled maintenance."""

    should_retrain: bool
    reasons: tuple[RetrainingReason, ...]
    current_stats: TrainingStatistics
    trained_stats: TrainedModelStatistics | None
    deltas: RetrainingDeltas


@dataclass(frozen=True, slots=True)
class FilmQueueRunResult:
    """Bounded film-backlog outcome with isolated success and failure counts."""

    pending_count: int
    processed_count: int
    success_count: int
    filtered_count: int
    failed_count: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class CatalogRefreshResult:
    """Aggregate-only catalog refresh result for one scheduled year selection."""

    target_years: tuple[int, ...]
    selected_count: int
    updated_count: int
    failed_count: int
    dry_run: bool
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class PromotionResult:
    """Atomic pointer transition and previous-selection identity."""

    model_version: str
    previous_version: str | None
    serving_reload_required: bool = True
