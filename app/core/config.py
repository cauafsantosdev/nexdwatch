from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
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
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    BACKEND_CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])

    MODEL_PATH: str = "models/"
    SCRAPER_MAX_WORKERS: int = 5

    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    TASK_STATE_REDIS_URL: str = "redis://redis:6379/1"
    PROFILE_SYNC_FRESHNESS_SECONDS: int = 900
    PROFILE_SYNC_ACTIVE_TTL_SECONDS: int = 600
    TASK_RESULT_TTL_SECONDS: int = 86_400
    PROFILE_SYNC_SOFT_TIME_LIMIT_SECONDS: int = 300
    PROFILE_SYNC_HARD_TIME_LIMIT_SECONDS: int = 330
    CELERY_BROKER_VISIBILITY_TIMEOUT_SECONDS: int = 900

    ARTIFACT_ROOT: Path = Path("data")
    MODEL_VERSION: str = "svd-current"

    PROFILE_CACHE_TTL_SECONDS: int = 86_400
    RECOMMENDATION_RESULT_TTL_SECONDS: int = 86_400

    RETRIEVAL_TOP_K: int = 500
    MIN_PROFILE_FILMS: int = 5

    @model_validator(mode="after")
    def validate_task_timeouts(self) -> "Settings":
        """Keep locks and broker visibility longer than hard task execution."""
        if (
            self.PROFILE_SYNC_ACTIVE_TTL_SECONDS
            <= self.PROFILE_SYNC_HARD_TIME_LIMIT_SECONDS
        ):
            raise ValueError("profile sync active TTL must exceed the hard time limit")
        if (
            self.CELERY_BROKER_VISIBILITY_TIMEOUT_SECONDS
            <= self.PROFILE_SYNC_HARD_TIME_LIMIT_SECONDS
        ):
            raise ValueError(
                "broker visibility timeout must exceed the hard time limit"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
