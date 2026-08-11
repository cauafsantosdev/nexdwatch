"""Safe neural-model artifact serialization and exact index rebuilding."""

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from app.ml.faiss_index import FaissIndexBuildResult, build_faiss_index
from experiments.neural_retrieval.model import InductiveNCFModel

NCF_ARTIFACT_SCHEMA = 2
NCF_MODEL_TYPE = "inductive_ncf"
NCF_EVALUATION_PROTOCOL = "exact_holdout_v2"


@dataclass(frozen=True, slots=True)
class NCFArtifactMetadata:
    """Validated parameters required to reconstruct a neural retrieval model."""

    artifact_schema: int
    model_type: str
    embedding_dim: int
    rating_embedding_dim: int
    hidden_dim: int
    dropout: float
    item_count: int
    positive_rating_threshold: float
    negative_rating_threshold: float
    training_seed: int
    best_epoch: int
    sampled_best_epoch: int
    device: str
    checkpoint_selection_metric: str
    evaluation_protocol: str
    data_summary: Mapping[str, int]
    training_metrics: Mapping[str, float | int]
    sampled_validation_metrics: Mapping[str, float | int]
    exact_validation_metrics: Mapping[str, float | int]
    test_metrics: Mapping[str, float | int]
    svd_test_metrics: Mapping[str, float | int]
    popularity_test_metrics: Mapping[str, float | int]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NCFArtifactMetadata":
        """Parse metadata while rejecting incompatible artifact schemas."""
        metadata = cls(
            artifact_schema=int(payload["artifact_schema"]),
            model_type=str(payload["model_type"]),
            embedding_dim=int(payload["embedding_dim"]),
            rating_embedding_dim=int(payload["rating_embedding_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            dropout=float(payload["dropout"]),
            item_count=int(payload["item_count"]),
            positive_rating_threshold=float(payload["positive_rating_threshold"]),
            negative_rating_threshold=float(payload["negative_rating_threshold"]),
            training_seed=int(payload["training_seed"]),
            best_epoch=int(payload["best_epoch"]),
            sampled_best_epoch=int(payload["sampled_best_epoch"]),
            device=str(payload["device"]),
            checkpoint_selection_metric=str(payload["checkpoint_selection_metric"]),
            evaluation_protocol=str(payload["evaluation_protocol"]),
            data_summary=dict(payload["data_summary"]),
            training_metrics=dict(payload.get("training_metrics", {})),
            sampled_validation_metrics=dict(payload["sampled_validation_metrics"]),
            exact_validation_metrics=dict(payload["exact_validation_metrics"]),
            test_metrics=dict(payload["test_metrics"]),
            svd_test_metrics=dict(payload["svd_test_metrics"]),
            popularity_test_metrics=dict(payload["popularity_test_metrics"]),
        )
        if metadata.artifact_schema != NCF_ARTIFACT_SCHEMA:
            raise ValueError("unsupported NCF artifact schema")
        if metadata.model_type != NCF_MODEL_TYPE:
            raise ValueError("invalid NCF model type")
        if metadata.evaluation_protocol != NCF_EVALUATION_PROTOCOL:
            raise ValueError("unsupported NCF evaluation protocol")
        if (
            min(
                metadata.embedding_dim,
                metadata.rating_embedding_dim,
                metadata.hidden_dim,
                metadata.item_count,
            )
            <= 0
        ):
            raise ValueError("NCF architecture dimensions must be positive")
        if not 0 <= metadata.dropout < 1:
            raise ValueError("NCF artifact dropout is invalid")
        return metadata

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata."""
        return asdict(self)


def generate_candidate_vectors(
    model: InductiveNCFModel,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> NDArray[np.float32]:
    """Generate normalized candidate vectors for every learned item row."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    batches: list[NDArray[np.float32]] = []
    with torch.inference_mode():
        for start in range(0, model.item_count, batch_size):
            stop = min(model.item_count, start + batch_size)
            rows = torch.arange(start, stop, dtype=torch.long, device=device)
            batches.append(
                model.encode_candidates(rows).detach().cpu().numpy().astype(np.float32)
            )
    return np.ascontiguousarray(np.concatenate(batches, axis=0), dtype=np.float32)


def write_ncf_artifacts(
    artifact_root: str | Path,
    *,
    model: InductiveNCFModel,
    metadata: NCFArtifactMetadata,
    film_ids: Sequence[int] | NDArray[np.int64],
    item_vectors: NDArray[np.float32],
) -> FaissIndexBuildResult:
    """Write state dict, metadata, vectors, IDs, and exact FAISS index."""
    destination = Path(artifact_root)
    destination.mkdir(parents=True, exist_ok=True)
    ids = [int(film_id) for film_id in film_ids]
    if metadata.item_count != len(ids) or len(item_vectors) != len(ids):
        raise ValueError("NCF artifact counts are inconsistent")

    _atomic_torch_save(
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        destination / "model.pt",
    )
    _atomic_numpy_save(item_vectors, destination / "item_vectors.npy")
    _atomic_json_write(ids, destination / "film_index.json")
    index_result = build_faiss_index(
        item_vectors,
        ids,
        destination / "retrieval.faiss",
    )
    _atomic_json_write(metadata.to_dict(), destination / "metadata.json")
    return index_result


def rebuild_ncf_index(artifact_root: str | Path) -> FaissIndexBuildResult:
    """Rebuild only the NCF exact index from candidate vectors and film IDs."""
    artifact_directory = Path(artifact_root)
    vectors = np.load(artifact_directory / "item_vectors.npy", allow_pickle=False)
    with (artifact_directory / "film_index.json").open(encoding="utf-8") as index_file:
        film_ids = json.load(index_file)
    return build_faiss_index(
        vectors,
        film_ids,
        artifact_directory / "retrieval.faiss",
    )


def read_ncf_metadata(path: str | Path) -> NCFArtifactMetadata:
    """Load and validate metadata JSON."""
    with Path(path).open(encoding="utf-8") as metadata_file:
        payload = json.load(metadata_file)
    if not isinstance(payload, dict):
        raise TypeError("NCF metadata must be a JSON object")
    return NCFArtifactMetadata.from_dict(payload)


def _atomic_json_write(payload: object, destination: Path) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_numpy_save(array: NDArray[np.float32], destination: Path) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".npy",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.save(temporary_path, array, allow_pickle=False)
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_torch_save(
    state_dict: Mapping[str, torch.Tensor], destination: Path
) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".pt",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(dict(state_dict), temporary_path)
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
