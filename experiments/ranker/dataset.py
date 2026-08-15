"""Build and persist reproducible fold ranking matrices and group metadata."""

import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from app.ml.historical_interactions import UserSplit
from app.ml.svd_profiles import build_svd_profile
from experiments.ranker.artifacts import FoldArtifacts
from experiments.ranker.candidates import generate_fold_candidates, target_source
from experiments.ranker.catalog import RankerCatalog
from experiments.ranker.config import (
    AFFINITY_FEATURES,
    FEATURE_NAMES,
    PREFERENCE_FEATURES,
    RANKER_PROTOCOL,
    SAMPLING_STRATA,
)
from experiments.ranker.features import (
    build_feature_matrix,
    build_personalized_feature_matrix,
    build_user_feature_profile,
)
from experiments.ranker.protocol import (
    RankerTrainingHoldouts,
    relevance_label_from_bucket,
)
from experiments.ranker.sampling import (
    build_full_evaluation_group,
    sample_ranker_group,
)

PartitionName = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class RankerUserExample:
    """One user's context, labels, and known-positive exclusions for a partition."""

    user_id: int
    context_item_rows: NDArray[np.int64]
    context_rating_buckets: NDArray[np.int64]
    positive_labels: dict[int, int]
    forbidden_positive_ids: set[int]
    designated_target_id: int | None
    designated_target_label: int
    target_stratum: str


@dataclass(frozen=True, slots=True)
class QueryAudit:
    """Per-group facts needed for global denominators and segment reports."""

    user_id: int
    requested_positive_count: int
    retrieved_positive_count: int
    designated_target_id: int | None
    designated_target_retrieved: bool
    target_source: str
    target_stratum: str
    history_depth_bucket: int
    full_candidate_count: int
    ranking_row_count: int
    ranking_inventory: str


@dataclass(frozen=True, slots=True)
class PartitionDataset:
    """Contiguous LightGBM rows plus shuffled-personalization control features."""

    features: NDArray[np.float32]
    shuffled_personalized_features: NDArray[np.float32]
    labels: NDArray[np.int8]
    query_ids: NDArray[np.int64]
    film_ids: NDArray[np.int64]
    sampling_strata: NDArray[np.int8]
    baseline_scores: NDArray[np.float32]
    group_sizes: NDArray[np.int32]
    queries: tuple[QueryAudit, ...]
    all_queries: tuple[QueryAudit, ...]
    eligible_user_count: int
    candidate_retrieved_user_count: int
    failed_training_group_count: int

    def validate(self) -> None:
        """Reject misaligned feature rows, groups, and query audit metadata."""
        row_count = len(self.labels)
        if self.features.shape != (row_count, len(FEATURE_NAMES)):
            raise ValueError("ranker feature matrix has an unexpected shape")
        if len(self.query_ids) != row_count or len(self.film_ids) != row_count:
            raise ValueError("ranker audit columns do not match feature rows")
        if self.baseline_scores.shape != (row_count, 4):
            raise ValueError("ranker baseline score matrix has an unexpected shape")
        if int(self.group_sizes.sum()) != row_count:
            raise ValueError("ranker group sizes do not sum to row count")
        if len(self.group_sizes) != len(self.queries):
            raise ValueError("ranker group metadata count differs from group sizes")
        boundaries = np.cumsum(self.group_sizes)
        starts = np.concatenate((np.asarray([0]), boundaries[:-1]))
        for start, stop in zip(starts, boundaries, strict=True):
            if len(set(self.query_ids[start:stop].tolist())) != 1:
                raise ValueError("ranker query rows are not contiguous")

    def missing_rates(self) -> dict[str, float]:
        """Report per-feature missing-value rates for benchmark diagnostics."""
        return {
            name: float(np.isnan(self.features[:, index]).mean())
            for index, name in enumerate(FEATURE_NAMES)
        }


def build_user_examples(
    splits: tuple[UserSplit, ...],
    film_ids: NDArray[np.int64],
    user_ids: set[int],
    partition: PartitionName,
    training_holdouts: dict[int, RankerTrainingHoldouts],
    target_stratum_by_user: dict[int, str],
) -> tuple[RankerUserExample, ...]:
    """Translate canonical splits into partition-specific contexts and labels."""
    examples: list[RankerUserExample] = []
    for split in splits:
        if split.cohort_user_id not in user_ids:
            continue
        canonical_ids = {
            int(film_ids[row])
            for row in (split.validation_target, split.test_target)
            if row is not None
        }
        if partition == "train":
            holdouts = training_holdouts[split.cohort_user_id]
            positive_labels = {
                int(film_ids[row]): relevance_label_from_bucket(int(bucket))
                for row, bucket in zip(
                    holdouts.item_rows,
                    holdouts.rating_buckets,
                    strict=True,
                )
            }
            context_rows = holdouts.context_item_rows
            context_buckets = holdouts.context_rating_buckets
            forbidden = canonical_ids
            target_id = None
            target_label = 0
        else:
            target_row = (
                split.validation_target
                if partition == "validation"
                else split.test_target
            )
            target_id = int(film_ids[target_row]) if target_row is not None else None
            bucket_by_row = {
                int(row): int(bucket)
                for row, bucket in zip(
                    split.all_item_rows,
                    split.all_rating_buckets,
                    strict=True,
                )
            }
            target_label = (
                relevance_label_from_bucket(bucket_by_row[target_row])
                if target_row is not None
                else 0
            )
            positive_labels = {target_id: target_label} if target_id is not None else {}
            alternate_row = (
                split.test_target
                if partition == "validation"
                else split.validation_target
            )
            forbidden = (
                {int(film_ids[alternate_row])} if alternate_row is not None else set()
            )
            context_rows = split.context_item_rows
            context_buckets = split.context_rating_buckets
        examples.append(
            RankerUserExample(
                user_id=split.cohort_user_id,
                context_item_rows=np.ascontiguousarray(context_rows, dtype=np.int64),
                context_rating_buckets=np.ascontiguousarray(
                    context_buckets, dtype=np.int64
                ),
                positive_labels=positive_labels,
                forbidden_positive_ids=forbidden,
                designated_target_id=target_id,
                designated_target_label=target_label,
                target_stratum=target_stratum_by_user[split.cohort_user_id],
            )
        )
    return tuple(examples)


def build_partition_dataset(
    examples: tuple[RankerUserExample, ...],
    artifacts: FoldArtifacts,
    catalog: RankerCatalog,
    *,
    partition: PartitionName,
    seed: int,
    fold: int,
    history_depth_thresholds: tuple[float, float, float],
) -> PartitionDataset:
    """Sample train groups but retain full validation/test candidate inventories."""
    profiles = {
        example.user_id: build_user_feature_profile(
            example.context_item_rows,
            example.context_rating_buckets,
            artifacts,
            catalog,
            history_depth_thresholds=history_depth_thresholds,
        )
        for example in examples
    }
    donor_by_user = _shuffled_profile_donors(tuple(profiles), seed=seed, fold=fold)
    feature_blocks: list[NDArray[np.float32]] = []
    shuffled_blocks: list[NDArray[np.float32]] = []
    labels: list[NDArray[np.int8]] = []
    query_ids: list[NDArray[np.int64]] = []
    film_ids: list[NDArray[np.int64]] = []
    stratum_codes: list[NDArray[np.int8]] = []
    baseline_blocks: list[NDArray[np.float32]] = []
    group_sizes: list[int] = []
    queries: list[QueryAudit] = []
    all_queries: list[QueryAudit] = []
    eligible = 0
    candidate_retrieved = 0
    failed_training_groups = 0
    personalized_indexes = np.asarray(
        [
            index
            for index, name in enumerate(FEATURE_NAMES)
            if name in AFFINITY_FEATURES or name in PREFERENCE_FEATURES
        ],
        dtype=np.int64,
    )
    stratum_to_code = {name: index for index, name in enumerate(SAMPLING_STRATA)}
    for example in examples:
        if example.positive_labels:
            eligible += 1
        result = generate_fold_candidates(
            example.user_id,
            example.context_item_rows,
            example.context_rating_buckets,
            artifacts,
        )
        candidate_by_id = {
            candidate.film_id: candidate for candidate in result.candidates
        }
        designated_candidate = (
            candidate_by_id.get(example.designated_target_id)
            if example.designated_target_id is not None
            else None
        )
        designated_retrieved = designated_candidate is not None
        if example.designated_target_id is not None and designated_retrieved:
            candidate_retrieved += 1
        if partition == "train":
            group = sample_ranker_group(
                result.candidates,
                example.positive_labels,
                forbidden_positive_ids=example.forbidden_positive_ids,
                seed=seed,
                user_id=example.user_id,
            )
            ranking_inventory = "sampled_512"
        else:
            group = build_full_evaluation_group(
                result.candidates,
                example.positive_labels,
                forbidden_positive_ids=example.forbidden_positive_ids,
            )
            ranking_inventory = "full_candidate_inventory"
        profile = profiles[example.user_id]
        audit = QueryAudit(
            user_id=example.user_id,
            requested_positive_count=group.requested_positive_count,
            retrieved_positive_count=group.retrieved_positive_count,
            designated_target_id=example.designated_target_id,
            designated_target_retrieved=designated_retrieved,
            target_source=target_source(designated_candidate),
            target_stratum=example.target_stratum,
            history_depth_bucket=int(profile.base_features["history_depth_bucket"]),
            full_candidate_count=result.unique_candidate_count,
            ranking_row_count=len(group.candidates),
            ranking_inventory=ranking_inventory,
        )
        all_queries.append(audit)
        if group.retrieved_positive_count == 0 and partition == "train":
            if example.positive_labels:
                failed_training_groups += 1
            continue
        matrix = build_feature_matrix(
            group.candidates,
            profile,
            artifacts,
            catalog,
            svd_profile_available=result.svd_profile_available,
        )
        donor_profile = profiles[donor_by_user[example.user_id]]
        donor_matrix = build_personalized_feature_matrix(
            group.candidates,
            donor_profile,
            artifacts,
            catalog,
        )
        feature_blocks.append(matrix)
        shuffled_blocks.append(donor_matrix)
        labels.append(group.labels)
        query_ids.append(
            np.full(len(group.candidates), example.user_id, dtype=np.int64)
        )
        film_ids.append(
            np.asarray(
                [candidate.film_id for candidate in group.candidates],
                dtype=np.int64,
            )
        )
        stratum_codes.append(
            np.asarray(
                [stratum_to_code[value] for value in group.sampling_strata],
                dtype=np.int8,
            )
        )
        baseline_blocks.append(
            _baseline_scores(
                group.candidates,
                example.context_item_rows,
                example.context_rating_buckets,
                artifacts,
            )
        )
        group_sizes.append(len(group.candidates))
        queries.append(audit)
    if not feature_blocks:
        empty = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        shuffled_empty = np.empty((0, len(personalized_indexes)), dtype=np.float32)
        dataset = PartitionDataset(
            features=empty,
            shuffled_personalized_features=shuffled_empty,
            labels=np.empty(0, dtype=np.int8),
            query_ids=np.empty(0, dtype=np.int64),
            film_ids=np.empty(0, dtype=np.int64),
            sampling_strata=np.empty(0, dtype=np.int8),
            baseline_scores=np.empty((0, 4), dtype=np.float32),
            group_sizes=np.empty(0, dtype=np.int32),
            queries=(),
            all_queries=tuple(all_queries),
            eligible_user_count=eligible,
            candidate_retrieved_user_count=candidate_retrieved,
            failed_training_group_count=failed_training_groups,
        )
        dataset.validate()
        return dataset
    dataset = PartitionDataset(
        features=np.ascontiguousarray(np.concatenate(feature_blocks), dtype=np.float32),
        shuffled_personalized_features=np.ascontiguousarray(
            np.concatenate(shuffled_blocks), dtype=np.float32
        ),
        labels=np.ascontiguousarray(np.concatenate(labels), dtype=np.int8),
        query_ids=np.ascontiguousarray(np.concatenate(query_ids), dtype=np.int64),
        film_ids=np.ascontiguousarray(np.concatenate(film_ids), dtype=np.int64),
        sampling_strata=np.ascontiguousarray(
            np.concatenate(stratum_codes), dtype=np.int8
        ),
        baseline_scores=np.ascontiguousarray(
            np.concatenate(baseline_blocks), dtype=np.float32
        ),
        group_sizes=np.ascontiguousarray(group_sizes, dtype=np.int32),
        queries=tuple(queries),
        all_queries=tuple(all_queries),
        eligible_user_count=eligible,
        candidate_retrieved_user_count=candidate_retrieved,
        failed_training_group_count=failed_training_groups,
    )
    dataset.validate()
    return dataset


def write_partition_dataset(
    dataset: PartitionDataset,
    destination: Path,
    *,
    seed: int,
    fold: int,
    partition: PartitionName,
) -> dict[str, object]:
    """Atomically persist numeric arrays and auditable metadata off-production."""
    dataset.validate()
    destination.mkdir(parents=True, exist_ok=True)
    arrays_path = destination / f"{partition}.npz"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{partition}.", suffix=".npz", dir=destination
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary_path,
            features=dataset.features,
            shuffled_personalized_features=dataset.shuffled_personalized_features,
            labels=dataset.labels,
            query_ids=dataset.query_ids,
            film_ids=dataset.film_ids,
            sampling_strata=dataset.sampling_strata,
            baseline_scores=dataset.baseline_scores,
            group_sizes=dataset.group_sizes,
        )
        os.replace(temporary_path, arrays_path)
        arrays_path.chmod(0o644)
    finally:
        temporary_path.unlink(missing_ok=True)
    metadata = {
        "protocol": RANKER_PROTOCOL,
        "seed": seed,
        "fold": fold,
        "partition": partition,
        "feature_names": list(FEATURE_NAMES),
        "row_count": len(dataset.labels),
        "group_count": len(dataset.group_sizes),
        "eligible_user_count": dataset.eligible_user_count,
        "candidate_retrieved_user_count": dataset.candidate_retrieved_user_count,
        "failed_training_group_count": dataset.failed_training_group_count,
        "group_construction": (
            "sampled_512" if partition == "train" else "full_candidate_inventory"
        ),
        "label_distribution": {
            str(label): int(count)
            for label, count in zip(
                *np.unique(dataset.labels, return_counts=True), strict=True
            )
        },
        "sampling_distribution": dict(
            Counter(SAMPLING_STRATA[int(code)] for code in dataset.sampling_strata)
        ),
        "missing_rates": dataset.missing_rates(),
        "queries": [asdict(query) for query in dataset.queries],
        "all_queries": [asdict(query) for query in dataset.all_queries],
    }
    _atomic_json_write(metadata, destination / f"{partition}_metadata.json")
    return metadata


def load_partition_dataset(
    destination: Path, partition: PartitionName
) -> PartitionDataset:
    """Load a persisted numeric partition without Python-object serialization."""
    with np.load(destination / f"{partition}.npz", allow_pickle=False) as arrays:
        values = {name: arrays[name].copy() for name in arrays.files}
    metadata = json.loads(
        (destination / f"{partition}_metadata.json").read_text(encoding="utf-8")
    )
    dataset = PartitionDataset(
        features=values["features"],
        shuffled_personalized_features=values["shuffled_personalized_features"],
        labels=values["labels"],
        query_ids=values["query_ids"],
        film_ids=values["film_ids"],
        sampling_strata=values["sampling_strata"],
        baseline_scores=values["baseline_scores"],
        group_sizes=values["group_sizes"],
        queries=tuple(QueryAudit(**query) for query in metadata["queries"]),
        all_queries=tuple(QueryAudit(**query) for query in metadata["all_queries"]),
        eligible_user_count=int(metadata["eligible_user_count"]),
        candidate_retrieved_user_count=int(metadata["candidate_retrieved_user_count"]),
        failed_training_group_count=int(metadata["failed_training_group_count"]),
    )
    dataset.validate()
    return dataset


def _shuffled_profile_donors(
    user_ids: tuple[int, ...], *, seed: int, fold: int
) -> dict[int, int]:
    ordered = np.asarray(sorted(user_ids), dtype=np.int64)
    if len(ordered) <= 1:
        return {int(user_id): int(user_id) for user_id in ordered}
    rng = np.random.default_rng(np.random.SeedSequence([seed, fold, 991]))
    shift = int(rng.integers(1, len(ordered)))
    donors = np.roll(ordered, shift)
    return {
        int(user_id): int(donor) for user_id, donor in zip(ordered, donors, strict=True)
    }


def _baseline_scores(
    candidates: tuple,
    context_item_rows: NDArray[np.int64],
    context_rating_buckets: NDArray[np.int64],
    artifacts: FoldArtifacts,
) -> NDArray[np.float32]:
    """Build popularity, weighted-SVD, RRF, and mean-SVD scores per row."""
    mean_query = build_svd_profile(
        artifacts.item_vectors,
        context_item_rows,
        context_rating_buckets,
        "svd_mean",
    )
    matrix = np.empty((len(candidates), 4), dtype=np.float32)
    id_to_row = {int(film_id): row for row, film_id in enumerate(artifacts.film_ids)}
    for index, candidate in enumerate(candidates):
        svd_rank = candidate.svd_rank
        popularity_rank = candidate.popularity_rank
        matrix[index, 0] = (
            -float(popularity_rank) if popularity_rank is not None else -np.inf
        )
        matrix[index, 1] = (
            float(candidate.svd_score) if candidate.svd_score is not None else -np.inf
        )
        matrix[index, 2] = (1.0 / (60 + svd_rank) if svd_rank is not None else 0.0) + (
            1.0 / (60 + popularity_rank) if popularity_rank is not None else 0.0
        )
        film_row = id_to_row[candidate.film_id]
        matrix[index, 3] = (
            float(artifacts.item_vectors[film_row] @ mean_query)
            if mean_query is not None
            else -np.inf
        )
    return matrix


def _atomic_json_write(payload: object, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary_path, destination)
        destination.chmod(0o644)
    finally:
        temporary_path.unlink(missing_ok=True)
