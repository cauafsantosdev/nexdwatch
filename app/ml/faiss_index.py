"""Build and validate the exact FAISS recommendation index."""

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class FaissIndexBuildResult:
    """Summary of a successfully written retrieval index."""

    film_count: int
    dimension: int
    output_path: Path


def prepare_faiss_inputs(
    item_vectors: ArrayLike,
    film_ids: Sequence[object] | NDArray[np.generic],
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Validate vectors and stable film identities for exact FAISS indexing.

    Vectors must be a non-empty real matrix whose non-zero rows are already
    L2-normalized. IDs must be unique, one-dimensional, exact int64 values with one
    identity per vector row.

    Returns:
        A contiguous float32 vector matrix and matching contiguous int64 identities.

    Raises:
        ValueError: If shape, finiteness, normalization, count, or identity values
            violate the retrieval artifact contract.
        TypeError: If a film identity is boolean rather than an integer.
    """
    # Validate the source dtype before conversion so complex or nonnumeric values
    # cannot be silently coerced into a plausible float32 matrix.
    raw_vectors = np.asarray(item_vectors)
    if raw_vectors.ndim != 2:
        raise ValueError("item embeddings must be a two-dimensional array")
    if raw_vectors.shape[0] == 0:
        raise ValueError("item embeddings must contain at least one vector")
    if raw_vectors.shape[1] == 0:
        raise ValueError("item embedding dimension must be positive")
    if not np.issubdtype(raw_vectors.dtype, np.number) or np.iscomplexobj(raw_vectors):
        raise ValueError("item embeddings must contain real numeric values")

    try:
        vectors = np.ascontiguousarray(raw_vectors, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("item embeddings cannot be converted to float32") from exc
    if not np.isfinite(vectors).all():
        raise ValueError("item embeddings must contain only finite values")

    # Identity mapping is part of the model contract, not row-position metadata.
    ids = _prepare_film_ids(film_ids)
    if vectors.shape[0] != len(ids):
        raise ValueError("embedding rows and film ID count differ")

    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    nonzero_norms = norms[norms > 1e-12]
    if nonzero_norms.size and not np.allclose(
        nonzero_norms,
        1.0,
        rtol=1e-4,
        atol=1e-5,
    ):
        raise ValueError("non-zero item embeddings must be L2-normalized")

    return vectors, ids


def build_faiss_index(
    item_vectors: ArrayLike,
    film_ids: Sequence[object] | NDArray[np.generic],
    output_path: str | Path,
) -> FaissIndexBuildResult:
    """Build and atomically publish an exact ID-mapped inner-product index.

    The complete index is written to a sibling temporary file, permissioned, and
    replaced into the destination only after FAISS serialization succeeds. Cleanup
    removes abandoned temporary output without touching an existing index.

    Returns:
        FaissIndexBuildResult: Persisted film count, vector dimension, and path.

    Raises:
        ValueError: If vector or film-identity validation fails.
        OSError: If the destination cannot be created or atomically replaced.
    """
    index = create_faiss_index(item_vectors, film_ids)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    # A same-directory temporary file keeps os.replace atomic on the target filesystem.
    try:
        faiss.write_index(index, str(temporary_path))
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    return FaissIndexBuildResult(
        film_count=int(index.ntotal),
        dimension=int(index.d),
        output_path=destination,
    )


def create_faiss_index(
    item_vectors: ArrayLike,
    film_ids: Sequence[object] | NDArray[np.generic],
) -> faiss.IndexIDMap2:
    """Build and cross-check an exact ``IndexIDMap2(IndexFlatIP)`` in memory.

    Actual database film IDs are stored as FAISS labels; callers never depend on
    vector row positions when retrieving candidates.
    """
    vectors, ids = prepare_faiss_inputs(item_vectors, film_ids)
    base_index = faiss.IndexFlatIP(vectors.shape[1])
    index = faiss.IndexIDMap2(base_index)
    index.add_with_ids(vectors, ids)
    validate_faiss_index(index, vectors.shape, ids)
    return index


def rebuild_faiss_index(
    artifact_root: str | Path,
) -> FaissIndexBuildResult:
    """Rebuild ``retrieval.faiss`` from an existing vector/mapping pair.

    This maintenance path preserves current SVD vectors and film identities, while
    applying the same validation and atomic write guarantees as model training.
    """
    artifact_directory = Path(artifact_root)
    vectors = np.load(
        artifact_directory / "item_embeddings.npy",
        allow_pickle=False,
    )
    with (artifact_directory / "film_index.json").open(encoding="utf-8") as index_file:
        film_ids = json.load(index_file)
    return build_faiss_index(
        vectors,
        film_ids,
        artifact_directory / "retrieval.faiss",
    )


def validate_faiss_index(
    index: faiss.Index,
    expected_shape: tuple[int, int],
    expected_ids: NDArray[np.int64],
) -> None:
    """Validate exact index structure, shape, and stored film identities.

    Raises:
        TypeError: If the index is not an ID map wrapping exact flat IP search.
        ValueError: If metric, dimensions, counts, or stored IDs disagree with the
            vector and mapping artifacts.
    """
    expected_count, expected_dimension = expected_shape
    if not isinstance(index, faiss.IndexIDMap2):
        raise TypeError("retrieval index must be an IndexIDMap2")
    base_index = faiss.downcast_index(index.index)
    if not isinstance(base_index, faiss.IndexFlatIP):
        raise TypeError("retrieval index must wrap an IndexFlatIP")
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("retrieval index must use inner-product search")
    if index.d != expected_dimension:
        raise ValueError("retrieval index dimension does not match item embeddings")
    if index.ntotal != expected_count:
        raise ValueError("retrieval index count does not match item embeddings")

    stored_ids = get_faiss_ids(index)
    if len(stored_ids) != expected_count or set(stored_ids.tolist()) != set(
        expected_ids.tolist()
    ):
        raise ValueError("retrieval index film IDs do not match film_index.json")


def get_faiss_ids(index: faiss.IndexIDMap2) -> NDArray[np.int64]:
    """Return a copy of IDs stored by an IndexIDMap2."""
    return np.asarray(faiss.vector_to_array(index.id_map), dtype=np.int64).copy()


def _prepare_film_ids(
    film_ids: Sequence[object] | NDArray[np.generic],
) -> NDArray[np.int64]:
    """Convert unique exact integer identities to a contiguous int64 array.

    Raises:
        TypeError: If an identity is boolean.
        ValueError: If the sequence is not one-dimensional, contains non-integral or
            out-of-range values, or repeats a film identity.
    """
    raw_ids = np.asarray(film_ids, dtype=object)
    if raw_ids.ndim != 1:
        raise ValueError("film IDs must be a one-dimensional sequence")

    int64_info = np.iinfo(np.int64)
    converted: list[int] = []
    for raw_id in raw_ids.tolist():
        if isinstance(raw_id, (bool, np.bool_)):
            raise TypeError("film IDs must be integers")
        try:
            film_id = int(raw_id)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("film IDs must be representable as int64") from exc
        try:
            if raw_id != film_id:
                raise ValueError("film IDs must be integers")
        except TypeError as exc:
            raise ValueError("film IDs must be integers") from exc
        if not int64_info.min <= film_id <= int64_info.max:
            raise ValueError("film IDs must be representable as int64")
        converted.append(film_id)

    ids = np.ascontiguousarray(converted, dtype=np.int64)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("film IDs must be unique")
    return ids
