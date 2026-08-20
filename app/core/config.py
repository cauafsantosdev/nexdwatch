"""Loads validated environment settings for API, workers, and model lifecycle."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated process configuration shared by every NexdWatch component."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    PROJECT_NAME: str = "nexdwatch"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: int = 5432

    @property
    def DATABASE_URL(self) -> str:
        """Build the async SQLAlchemy URL from authoritative PostgreSQL settings."""
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    BACKEND_CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])

    MODEL_PATH: str = "models/"
    SCRAPER_MAX_WORKERS: int = 5
    ZENROWS_API_KEY: SecretStr | None = None

    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    TASK_STATE_REDIS_URL: str = "redis://redis:6379/1"
    PROFILE_SYNC_FRESHNESS_SECONDS: int = 900
    PROFILE_SYNC_ACTIVE_TTL_SECONDS: int = 600
    TASK_RESULT_TTL_SECONDS: int = 86_400
    PROFILE_SYNC_SOFT_TIME_LIMIT_SECONDS: int = 300
    PROFILE_SYNC_HARD_TIME_LIMIT_SECONDS: int = 330
    CELERY_BROKER_VISIBILITY_TIMEOUT_SECONDS: int = 21_600
    MAINTENANCE_REDIS_URL: str = "redis://redis:6379/2"
    MAINTENANCE_LOCK_TTL_SECONDS: int = 21_600
    MAINTENANCE_SOFT_TIME_LIMIT_SECONDS: int = 18_000
    MAINTENANCE_HARD_TIME_LIMIT_SECONDS: int = 21_000
    FILM_QUEUE_BATCH_SIZE: int = Field(default=100, ge=1, le=1_000)

    ARTIFACT_ROOT: Path = Path("data")
    MODEL_VERSION: str = "svd-current"

    PROFILE_CACHE_TTL_SECONDS: int = 86_400
    RECOMMENDATION_RESULT_TTL_SECONDS: int = 86_400

    RETRIEVAL_TOP_K: int = 500
    MIN_PROFILE_FILMS: int = 5

    # Observable lifecycle triggers; these are operational policy, not learned
    # hyperparameters, and a forced rebuild remains available to operators.
    NEW_ELIGIBLE_USERS_THRESHOLD: int = Field(default=100, ge=1)
    NEW_MODEL_FILMS_THRESHOLD: int = Field(default=250, ge=1)
    MAX_MODEL_AGE_DAYS: int = Field(default=180, ge=1)
    MODEL_RETENTION_PREVIOUS: int = Field(default=2, ge=1)
    MODEL_POINTER_CHECK_INTERVAL_SECONDS: float = Field(default=30.0, ge=5.0)

    @model_validator(mode="after")
    def validate_task_timeouts(self) -> "Settings":
        """Keep locks and broker visibility longer than hard task execution."""
        if (
            self.PROFILE_SYNC_ACTIVE_TTL_SECONDS
            <= self.PROFILE_SYNC_HARD_TIME_LIMIT_SECONDS
        ):
            raise ValueError("profile sync active TTL must exceed the hard time limit")
        if self.CELERY_BROKER_VISIBILITY_TIMEOUT_SECONDS <= max(
            self.PROFILE_SYNC_HARD_TIME_LIMIT_SECONDS,
            self.MAINTENANCE_HARD_TIME_LIMIT_SECONDS,
        ):
            raise ValueError(
                "broker visibility timeout must exceed every hard time limit"
            )
        if (
            self.MAINTENANCE_LOCK_TTL_SECONDS
            <= self.MAINTENANCE_HARD_TIME_LIMIT_SECONDS
        ):
            raise ValueError("maintenance lock TTL must exceed the hard time limit")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return Settings()
