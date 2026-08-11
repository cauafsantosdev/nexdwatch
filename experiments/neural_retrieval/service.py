"""Runtime inductive neural collaborative retrieval backend."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import torch
from numpy.typing import NDArray
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.domain.recommendations import Recommendation, RecommendationResult
from app.ml.faiss_index import prepare_faiss_inputs, validate_faiss_index
from app.ml.ratings import rating_to_bucket
from app.repositories.films import FilmRepository
from app.repositories.interactions import InteractionRepository
from app.services.recommendation_backend import (
    NO_USABLE_RATINGS_INFO,
    NO_WATCHED_FILMS_INFO,
    ModelUnavailableError,
)
from experiments.neural_retrieval.artifacts import (
    NCFArtifactMetadata,
    generate_candidate_vectors,
    read_ncf_metadata,
)
from experiments.neural_retrieval.model import InductiveNCFModel

logger = logging.getLogger(__name__)
NCF_RECOMMENDATION_STRATEGY = "Inductive_NCF"


@dataclass(frozen=True, slots=True)
class _NCFArtifacts:
    """Cross-validated neural artifacts held by the API process."""

    model: InductiveNCFModel
    metadata: NCFArtifactMetadata
    item_vectors: NDArray[np.float32]
    film_index: tuple[int, ...]
    id_to_position: dict[int, int]
    retrieval_index: faiss.IndexIDMap2


class NCFRecommendationService:
    """Serve zero-shot recommendations from rated-film histories."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = SessionLocal,
        artifact_root: str | Path | None = None,
    ) -> None:
        settings = get_settings()
        self._session_factory = session_factory
        self._artifact_root = Path(artifact_root or (settings.ARTIFACT_ROOT / "ncf"))
        self._retrieval_top_k = settings.RETRIEVAL_TOP_K
        self._artifacts: _NCFArtifacts | None = None

    @property
    def is_model_loaded(self) -> bool:
        """Return whether a complete valid NCF artifact set is loaded."""
        return self._artifacts is not None

    def load_artifacts(self) -> bool:
        """Load all neural artifacts safely on CPU and validate consistency."""
        try:
            metadata = read_ncf_metadata(self._artifact_root / "metadata.json")
            item_vectors = np.load(
                self._artifact_root / "item_vectors.npy",
                allow_pickle=False,
            )
            with (self._artifact_root / "film_index.json").open(
                encoding="utf-8"
            ) as index_file:
                raw_film_ids = json.load(index_file)
            retrieval_index = faiss.read_index(
                str(self._artifact_root / "retrieval.faiss")
            )
            vectors, validated_ids = prepare_faiss_inputs(
                item_vectors,
                raw_film_ids,
            )
            if vectors.shape != (metadata.item_count, metadata.embedding_dim):
                raise ValueError("NCF item-vector shape does not match metadata")
            validate_faiss_index(retrieval_index, vectors.shape, validated_ids)

            model = InductiveNCFModel(
                metadata.item_count,
                embedding_dim=metadata.embedding_dim,
                rating_embedding_dim=metadata.rating_embedding_dim,
                hidden_dim=metadata.hidden_dim,
                dropout=metadata.dropout,
            )
            state = torch.load(
                self._artifact_root / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(state, dict):
                raise TypeError("NCF model artifact must contain a state dictionary")
            model.load_state_dict(state, strict=True)
            if model.item_embedding.num_embeddings != len(validated_ids):
                raise ValueError("NCF embedding table does not match film index")
            model.eval()
            regenerated_vectors = generate_candidate_vectors(
                model,
                device=torch.device("cpu"),
            )
            if not np.allclose(
                regenerated_vectors,
                vectors,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise ValueError("NCF candidate vectors do not match model weights")
            film_index = tuple(int(value) for value in validated_ids)
            self._artifacts = _NCFArtifacts(
                model=model,
                metadata=metadata,
                item_vectors=vectors,
                film_index=film_index,
                id_to_position={
                    film_id: position for position, film_id in enumerate(film_index)
                },
                retrieval_index=retrieval_index,
            )
            logger.info(
                "Loaded Inductive_NCF artifacts for %d films with dimension %d",
                len(film_index),
                vectors.shape[1],
            )
            return True
        except FileNotFoundError:
            logger.warning("NCF artifacts were not found in %s", self._artifact_root)
        except Exception:
            logger.exception("NCF artifacts are invalid")
        self._artifacts = None
        return False

    def unload_artifacts(self) -> None:
        """Release neural model, vectors, and index from memory."""
        self._artifacts = None
        logger.info("Inductive_NCF artifacts unloaded")

    async def recommend(self, user_id: int) -> RecommendationResult:
        """Encode a product user's complete usable history and retrieve films."""
        artifacts = self._artifacts
        if artifacts is None:
            raise ModelUnavailableError

        async with self._session_factory() as session:
            interactions = InteractionRepository(session)
            watched_film_ids = await interactions.get_watched_film_ids(user_id)
            if not watched_film_ids:
                return RecommendationResult(
                    user_id=user_id,
                    info=NO_WATCHED_FILMS_INFO,
                    recommendations=(),
                )
            rated = await interactions.get_rated_interactions(user_id)
            history_rows: list[int] = []
            history_ratings: list[int] = []
            for interaction in rated:
                position = artifacts.id_to_position.get(interaction.film_id)
                if position is None:
                    continue
                try:
                    rating_bucket = rating_to_bucket(interaction.rating)
                except ValueError:
                    logger.warning(
                        "Ignoring invalid persisted rating user_id=%d film_id=%d",
                        user_id,
                        interaction.film_id,
                    )
                    continue
                history_rows.append(position)
                history_ratings.append(rating_bucket)

            if not history_rows:
                return RecommendationResult(
                    user_id=user_id,
                    info=NO_USABLE_RATINGS_INFO,
                    recommendations=(),
                )

            model = artifacts.model
            model.eval()
            with torch.inference_mode():
                rows = torch.tensor([history_rows], dtype=torch.long)
                ratings = torch.tensor([history_ratings], dtype=torch.long)
                mask = torch.ones_like(rows, dtype=torch.bool)
                user_vector = model.encode_history(rows, ratings, mask)
            query = np.ascontiguousarray(user_vector.numpy(), dtype=np.float32)

            watched_set = set(watched_film_ids)
            indexed_watched_count = sum(
                film_id in artifacts.id_to_position for film_id in watched_set
            )
            requested_k = min(
                int(artifacts.retrieval_index.ntotal),
                self._retrieval_top_k + indexed_watched_count,
            )
            faiss_scores, faiss_ids = artifacts.retrieval_index.search(
                query,
                requested_k,
            )
            candidates: list[tuple[int, float]] = []
            for raw_film_id, raw_score in zip(
                faiss_ids[0], faiss_scores[0], strict=True
            ):
                film_id = int(raw_film_id)
                if film_id < 0 or film_id in watched_set:
                    continue
                candidates.append((film_id, float(raw_score)))
                if len(candidates) == self._retrieval_top_k:
                    break
            films = await FilmRepository(session).get_by_ids(
                [film_id for film_id, _ in candidates]
            )

        films_by_id = {film.id: film for film in films}
        recommendations: list[Recommendation] = []
        for film_id, score in candidates:
            film = films_by_id.get(film_id)
            if film is None:
                continue
            director: str | list[str] = film.directors[0].name if film.directors else []
            recommendations.append(
                Recommendation(
                    id=film.id,
                    title=film.title,
                    director=director,
                    year=film.year,
                    match_score=round(score, 4),
                )
            )
            if len(recommendations) == 10:
                break
        return RecommendationResult(
            user_id=user_id,
            strategy=NCF_RECOMMENDATION_STRATEGY,
            recommendations=tuple(recommendations),
        )
