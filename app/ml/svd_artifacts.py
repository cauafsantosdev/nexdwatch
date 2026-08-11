"""Validated in-memory SVD and exact-FAISS artifact loading."""

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

from app.ml.faiss_index import prepare_faiss_inputs, validate_faiss_index


@dataclass(frozen=True, slots=True)
class SVDArtifacts:
    """SVD item vectors, catalog identity mapping, and exact retrieval index."""

    item_vectors: NDArray[np.floating]
    film_index: tuple[int, ...]
    id_to_position: dict[int, int]
    retrieval_index: faiss.IndexIDMap2


def load_svd_artifacts(artifact_root: str | Path) -> SVDArtifacts:
    """Load and validate the complete production SVD artifact set."""
    root = Path(artifact_root)
    item_vectors = np.load(root / "item_embeddings.npy", allow_pickle=False)
    with (root / "film_index.json").open(encoding="utf-8") as index_file:
        raw_film_ids = json.load(index_file)
    retrieval_index = faiss.read_index(str(root / "retrieval.faiss"))
    validated_vectors, validated_ids = prepare_faiss_inputs(item_vectors, raw_film_ids)
    validate_faiss_index(retrieval_index, validated_vectors.shape, validated_ids)
    film_index = tuple(int(value) for value in validated_ids)
    return SVDArtifacts(
        item_vectors=item_vectors,
        film_index=film_index,
        id_to_position={film_id: row for row, film_id in enumerate(film_index)},
        retrieval_index=retrieval_index,
    )
