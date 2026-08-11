"""Artifact and zero-shot serving tests for the neural backend."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.ml.faiss_index import build_faiss_index
from app.repositories.interactions import RatedInteraction
from app.services.recommendation_backend import ModelUnavailableError
from experiments.neural_retrieval import service as ncf_module
from experiments.neural_retrieval.artifacts import (
    NCF_ARTIFACT_SCHEMA,
    NCF_EVALUATION_PROTOCOL,
    NCF_MODEL_TYPE,
    NCFArtifactMetadata,
    generate_candidate_vectors,
    write_ncf_artifacts,
)
from experiments.neural_retrieval.model import InductiveNCFModel
from experiments.neural_retrieval.service import NCFRecommendationService


class _SessionContext:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *args: object) -> None:
        return None


class _SessionFactory:
    def __call__(self) -> _SessionContext:
        return _SessionContext()


def _metadata(item_count: int, dimension: int = 8) -> NCFArtifactMetadata:
    return NCFArtifactMetadata(
        artifact_schema=NCF_ARTIFACT_SCHEMA,
        model_type=NCF_MODEL_TYPE,
        embedding_dim=dimension,
        rating_embedding_dim=3,
        hidden_dim=12,
        dropout=0.0,
        item_count=item_count,
        positive_rating_threshold=3.5,
        negative_rating_threshold=2.5,
        training_seed=42,
        best_epoch=1,
        sampled_best_epoch=1,
        device="cpu",
        checkpoint_selection_metric="exact_validation_ndcg_at_10",
        evaluation_protocol=NCF_EVALUATION_PROTOCOL,
        data_summary={"csv_rows": 20},
        training_metrics={"ranking_loss": 0.5},
        sampled_validation_metrics={"loss": 1.0},
        exact_validation_metrics={"recall_at_10": 0.5},
        test_metrics={"recall_at_10": 0.5},
        svd_test_metrics={"recall_at_10": 0.4},
        popularity_test_metrics={"recall_at_10": 0.2},
    )


def _write_valid_artifacts(tmp_path, item_count: int = 15):
    torch.manual_seed(5)
    model = InductiveNCFModel(
        item_count,
        embedding_dim=8,
        rating_embedding_dim=3,
        hidden_dim=12,
        dropout=0,
    )
    model.eval()
    vectors = generate_candidate_vectors(model, device=torch.device("cpu"))
    film_ids = list(range(101, 101 + item_count))
    write_ncf_artifacts(
        tmp_path,
        model=model,
        metadata=_metadata(item_count),
        film_ids=film_ids,
        item_vectors=vectors,
    )
    return model, vectors, film_ids


def test_metadata_rejects_pre_corrected_evaluation_schema() -> None:
    payload = _metadata(5).to_dict()
    payload["artifact_schema"] = 1

    with pytest.raises(ValueError, match="unsupported NCF artifact schema"):
        NCFArtifactMetadata.from_dict(payload)


def _film(film_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=film_id,
        title=f"Film {film_id}",
        year=2000,
        directors=[SimpleNamespace(name="Director")],
    )


def test_artifact_round_trip_reproduces_eval_embedding(tmp_path) -> None:
    model, _, _ = _write_valid_artifacts(tmp_path, 5)
    service = NCFRecommendationService(_SessionFactory(), tmp_path)

    assert service.load_artifacts()
    assert service._artifacts is not None
    rows = torch.tensor([[0, 2, 4]])
    ratings = torch.tensor([[0, 5, 9]])
    mask = torch.ones_like(rows, dtype=torch.bool)
    with torch.inference_mode():
        expected = model.encode_history(rows, ratings, mask)
        actual = service._artifacts.model.encode_history(rows, ratings, mask)
    torch.testing.assert_close(actual, expected)


def test_zero_shot_runtime_uses_history_excludes_all_watched_and_is_deterministic(
    tmp_path, monkeypatch
) -> None:
    _write_valid_artifacts(tmp_path)
    service = NCFRecommendationService(_SessionFactory(), tmp_path)
    assert service.load_artifacts()
    assert service._artifacts is not None
    service._retrieval_top_k = 12
    watched = [101, 102, 103]
    rated = [RatedInteraction(101, 5.0), RatedInteraction(102, 1.0)]
    watched_lookup = AsyncMock(return_value=watched)
    rated_lookup = AsyncMock(return_value=rated)
    monkeypatch.setattr(
        ncf_module.InteractionRepository,
        "get_watched_film_ids",
        watched_lookup,
    )
    monkeypatch.setattr(
        ncf_module.InteractionRepository,
        "get_rated_interactions",
        rated_lookup,
    )

    artifacts = service._artifacts
    with torch.inference_mode():
        query = artifacts.model.encode_history(
            torch.tensor([[0, 1]]),
            torch.tensor([[9, 1]]),
            torch.ones((1, 2), dtype=torch.bool),
        ).numpy()
    _, labels = artifacts.retrieval_index.search(
        np.ascontiguousarray(query, dtype=np.float32),
        artifacts.retrieval_index.ntotal,
    )
    expected_order = [int(label) for label in labels[0] if int(label) not in watched]
    missing = expected_order[0]
    films = [
        _film(film_id) for film_id in reversed(expected_order) if film_id != missing
    ]
    film_lookup = AsyncMock(return_value=films)
    monkeypatch.setattr(ncf_module.FilmRepository, "get_by_ids", film_lookup)
    monkeypatch.setattr(
        artifacts.model,
        "encode_candidates",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("runtime must retrieve candidates through FAISS")
        ),
    )

    first = asyncio.run(service.recommend(999_999))
    second = asyncio.run(service.recommend(999_999))

    expected = [film_id for film_id in expected_order if film_id != missing][:10]
    assert [item.id for item in first.recommendations] == expected
    assert first == second
    assert first.strategy == "Inductive_NCF"
    assert len(first.recommendations) == 10
    assert set(watched).isdisjoint(item.id for item in first.recommendations)
    assert all(np.isfinite(item.match_score) for item in first.recommendations)
    watched_lookup.assert_awaited_with(999_999)
    rated_lookup.assert_awaited_with(999_999)


def test_unrated_watch_is_not_encoded_but_remains_excluded(
    tmp_path, monkeypatch
) -> None:
    _write_valid_artifacts(tmp_path)
    service = NCFRecommendationService(_SessionFactory(), tmp_path)
    assert service.load_artifacts()
    monkeypatch.setattr(
        ncf_module.InteractionRepository,
        "get_watched_film_ids",
        AsyncMock(return_value=[101, 103]),
    )
    monkeypatch.setattr(
        ncf_module.InteractionRepository,
        "get_rated_interactions",
        AsyncMock(return_value=[RatedInteraction(101, 5.0)]),
    )
    monkeypatch.setattr(
        ncf_module.FilmRepository,
        "get_by_ids",
        AsyncMock(return_value=[_film(film_id) for film_id in range(104, 116)]),
    )

    result = asyncio.run(service.recommend(7))

    assert 103 not in [item.id for item in result.recommendations]


def test_missing_artifacts_do_not_fall_back(tmp_path) -> None:
    service = NCFRecommendationService(_SessionFactory(), tmp_path)
    assert not service.load_artifacts()
    assert not service.is_model_loaded
    with pytest.raises(ModelUnavailableError):
        asyncio.run(service.recommend(1))


@pytest.mark.parametrize(
    "corruption",
    ["metadata", "vector_count", "vector_dimension", "faiss_ids", "weights"],
)
def test_inconsistent_or_corrupt_artifact_set_is_unavailable(
    tmp_path, corruption: str
) -> None:
    _, vectors, film_ids = _write_valid_artifacts(tmp_path, 5)
    if corruption == "metadata":
        (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    elif corruption == "vector_count":
        np.save(tmp_path / "item_vectors.npy", vectors[:4])
    elif corruption == "vector_dimension":
        invalid = np.pad(vectors, ((0, 0), (0, 1)))
        invalid /= np.linalg.norm(invalid, axis=1, keepdims=True)
        np.save(tmp_path / "item_vectors.npy", invalid)
    elif corruption == "faiss_ids":
        build_faiss_index(vectors, [*film_ids[:-1], 999], tmp_path / "retrieval.faiss")
    else:
        (tmp_path / "model.pt").write_bytes(b"not torch weights")

    service = NCFRecommendationService(_SessionFactory(), tmp_path)

    assert not service.load_artifacts()
    assert not service.is_model_loaded


def test_model_weights_are_loaded_with_weights_only(tmp_path, monkeypatch) -> None:
    _write_valid_artifacts(tmp_path, 4)
    original_load = ncf_module.torch.load
    observed: dict[str, object] = {}

    def recording_load(*args, **kwargs):
        observed.update(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(ncf_module.torch, "load", recording_load)
    service = NCFRecommendationService(_SessionFactory(), tmp_path)

    assert service.load_artifacts()
    assert observed["weights_only"] is True
    assert observed["map_location"] == "cpu"
