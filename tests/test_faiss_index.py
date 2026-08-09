"""Tests for exact FAISS index construction and rebuilding."""

import json
from types import SimpleNamespace

import faiss
import numpy as np
import pandas as pd
import pytest

from app.ml import train as train_module
from app.ml.faiss_index import (
    build_faiss_index,
    get_faiss_ids,
    rebuild_faiss_index,
)


def _normalized_vectors(count: int = 4, dimension: int = 3) -> np.ndarray:
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(count, dimension))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_builder_writes_exact_id_mapped_index_and_round_trips(tmp_path) -> None:
    vectors = _normalized_vectors().astype(np.float64)
    film_ids = np.array([90, 15, 73, 44], dtype=np.int64)
    output_path = tmp_path / "retrieval.faiss"

    result = build_faiss_index(vectors, film_ids, output_path)
    index = faiss.read_index(str(output_path))

    assert isinstance(index, faiss.IndexIDMap2)
    assert isinstance(faiss.downcast_index(index.index), faiss.IndexFlatIP)
    assert index.ntotal == 4
    assert index.d == 3
    np.testing.assert_array_equal(get_faiss_ids(index), film_ids)
    assert result.film_count == 4
    assert result.dimension == 3
    assert result.output_path == output_path

    scores, labels = index.search(
        np.ascontiguousarray(vectors[[0]], dtype=np.float32),
        len(vectors),
    )
    assert labels[0, 0] == 90
    assert scores.dtype == np.float32


@pytest.mark.parametrize(
    ("vectors", "film_ids", "message"),
    [
        (np.array([1.0, 0.0]), [1], "two-dimensional"),
        (np.empty((0, 2)), [], "at least one"),
        (np.empty((1, 0)), [1], "dimension"),
        (_normalized_vectors(2), [1], "count differ"),
        (_normalized_vectors(2), [1, 1], "unique"),
        (_normalized_vectors(1), [2**80], "int64"),
        (np.array([[np.nan, 0.0]]), [1], "finite"),
        (np.array([[np.inf, 0.0]]), [1], "finite"),
        (np.array([[2.0, 0.0]]), [1], "L2-normalized"),
    ],
)
def test_builder_rejects_invalid_inputs(
    tmp_path,
    vectors: np.ndarray,
    film_ids: list[int],
    message: str,
) -> None:
    output_path = tmp_path / "retrieval.faiss"

    with pytest.raises((TypeError, ValueError), match=message):
        build_faiss_index(vectors, film_ids, output_path)

    assert not output_path.exists()


def test_rebuild_uses_existing_numpy_and_json_artifacts(tmp_path) -> None:
    vectors = _normalized_vectors(3, 2)
    film_ids = [101, 205, 999]
    np.save(tmp_path / "item_embeddings.npy", vectors)
    (tmp_path / "film_index.json").write_text(json.dumps(film_ids), encoding="utf-8")

    result = rebuild_faiss_index(tmp_path)
    index = faiss.read_index(str(tmp_path / "retrieval.faiss"))

    assert result.film_count == 3
    assert result.dimension == 2
    np.testing.assert_array_equal(get_faiss_ids(index), film_ids)


def test_exact_faiss_results_match_numpy_oracle(tmp_path) -> None:
    vectors = _normalized_vectors(30, 8)
    film_ids = np.arange(100, 130, dtype=np.int64)
    index_path = tmp_path / "retrieval.faiss"
    build_faiss_index(vectors, film_ids, index_path)
    index = faiss.read_index(str(index_path))

    rated_positions = [1, 7, 13]
    watched_ids = {int(film_ids[position]) for position in [1, 2, 7, 13]}
    user_vector = np.mean(vectors[rated_positions], axis=0)
    numpy_scores = vectors @ user_vector
    oracle = sorted(
        (
            (int(film_id), float(numpy_scores[position]))
            for position, film_id in enumerate(film_ids)
            if int(film_id) not in watched_ids
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    faiss_scores, faiss_ids = index.search(
        np.ascontiguousarray(user_vector.reshape(1, -1), dtype=np.float32),
        len(film_ids),
    )
    retrieved = [
        (int(film_id), float(score))
        for film_id, score in zip(faiss_ids[0], faiss_scores[0], strict=True)
        if int(film_id) not in watched_ids
    ]

    assert [film_id for film_id, _ in retrieved] == [film_id for film_id, _ in oracle]
    np.testing.assert_allclose(
        [score for _, score in retrieved],
        [score for _, score in oracle],
        rtol=1e-5,
        atol=1e-6,
    )


def test_training_writes_all_three_consistent_artifacts(tmp_path, monkeypatch) -> None:
    rows = [
        {
            "user_id": user_id,
            "film_id": film_id,
            "rating": float((user_id + film_id) % 5 + 1),
        }
        for user_id in range(1, 41)
        for film_id in range(101, 136)
    ]
    frame = pd.DataFrame(rows)
    engine = SimpleNamespace(dispose=lambda: None)
    monkeypatch.setattr(train_module, "create_engine", lambda _: engine)
    monkeypatch.setattr(train_module.pd, "read_sql", lambda *_: frame)

    result = train_module.train_svd_model(tmp_path)

    assert result is not None
    assert (tmp_path / "item_embeddings.npy").exists()
    assert (tmp_path / "film_index.json").exists()
    assert (tmp_path / "retrieval.faiss").exists()
    vectors = np.load(tmp_path / "item_embeddings.npy")
    film_ids = json.loads((tmp_path / "film_index.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(tmp_path / "retrieval.faiss"))
    assert index.ntotal == len(vectors) == len(film_ids)
    assert index.d == vectors.shape[1]
    assert set(get_faiss_ids(index).tolist()) == set(film_ids)
