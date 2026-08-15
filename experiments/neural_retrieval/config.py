"""Environment-backed configuration isolated to the neural retrieval experiment."""

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class NeuralRetrievalSettings(BaseSettings):
    """Validated hyperparameters used only by experimental commands."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    NCF_EMBEDDING_DIM: int = 64
    NCF_RATING_EMBEDDING_DIM: int = 8
    NCF_POSITIVE_RATING_THRESHOLD: float = 3.5
    NCF_NEGATIVE_RATING_THRESHOLD: float = 2.5
    NCF_MAX_CONTEXT_ITEMS: int = 256
    NCF_TARGETS_PER_USER_PER_EPOCH: int = 32
    NCF_NEGATIVES_PER_POSITIVE: int = 8
    NCF_TEMPERATURE: float = 0.1
    NCF_BATCH_SIZE: int = 256
    NCF_MAX_EPOCHS: int = 20
    NCF_LEARNING_RATE: float = 0.001
    NCF_WEIGHT_DECAY: float = 0.00001
    NCF_DROPOUT: float = 0.1
    NCF_EARLY_STOPPING_PATIENCE: int = 3
    NCF_RANDOM_SEED: int = 42
    NCF_TRAINING_DEVICE: Literal["cpu", "cuda"] = "cpu"
    NCF_EXACT_VALIDATION_INTERVAL: int = 1

    @model_validator(mode="after")
    def validate_hyperparameters(self) -> "NeuralRetrievalSettings":
        """Validate rating semantics, dimensions, and optimizer settings."""
        if not 0.5 <= self.NCF_NEGATIVE_RATING_THRESHOLD < 3.0:
            raise ValueError("NCF negative threshold must be between 0.5 and 3.0")
        if not 3.0 < self.NCF_POSITIVE_RATING_THRESHOLD <= 5.0:
            raise ValueError("NCF positive threshold must be between 3.0 and 5.0")
        bucket_values = {value / 2 for value in range(1, 11)}
        if self.NCF_NEGATIVE_RATING_THRESHOLD not in bucket_values:
            raise ValueError("NCF negative threshold must be a half-star value")
        if self.NCF_POSITIVE_RATING_THRESHOLD not in bucket_values:
            raise ValueError("NCF positive threshold must be a half-star value")
        positive_integers = {
            "NCF_EMBEDDING_DIM": self.NCF_EMBEDDING_DIM,
            "NCF_RATING_EMBEDDING_DIM": self.NCF_RATING_EMBEDDING_DIM,
            "NCF_MAX_CONTEXT_ITEMS": self.NCF_MAX_CONTEXT_ITEMS,
            "NCF_TARGETS_PER_USER_PER_EPOCH": self.NCF_TARGETS_PER_USER_PER_EPOCH,
            "NCF_NEGATIVES_PER_POSITIVE": self.NCF_NEGATIVES_PER_POSITIVE,
            "NCF_BATCH_SIZE": self.NCF_BATCH_SIZE,
            "NCF_MAX_EPOCHS": self.NCF_MAX_EPOCHS,
            "NCF_EARLY_STOPPING_PATIENCE": self.NCF_EARLY_STOPPING_PATIENCE,
            "NCF_EXACT_VALIDATION_INTERVAL": self.NCF_EXACT_VALIDATION_INTERVAL,
        }
        invalid = [name for name, value in positive_integers.items() if value <= 0]
        if invalid:
            raise ValueError(f"NCF settings must be positive: {', '.join(invalid)}")
        if self.NCF_TEMPERATURE <= 0:
            raise ValueError("NCF temperature must be positive")
        if not 0 <= self.NCF_DROPOUT < 1:
            raise ValueError("NCF dropout must be in [0, 1)")
        if self.NCF_LEARNING_RATE <= 0 or self.NCF_WEIGHT_DECAY < 0:
            raise ValueError("NCF optimizer settings are invalid")
        return self


@lru_cache
def get_neural_retrieval_settings() -> NeuralRetrievalSettings:
    """Return isolated neural-research settings without changing app configuration."""
    return NeuralRetrievalSettings()
