from functools import lru_cache
from pathlib import Path

from pydantic import Field
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

    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    ARTIFACT_ROOT: Path = Path("data")
    MODEL_VERSION: str = "svd-current"

    PROFILE_CACHE_TTL_SECONDS: int = 86_400
    RECOMMENDATION_RESULT_TTL_SECONDS: int = 86_400

    RETRIEVAL_TOP_K: int = 500
    MIN_PROFILE_FILMS: int = 5

    NCF_EMBEDDING_DIM: int = 64


@lru_cache
def get_settings() -> Settings:
    return Settings()
