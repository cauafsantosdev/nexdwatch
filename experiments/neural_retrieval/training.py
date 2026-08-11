"""Training and leakage-free evaluation for inductive neural retrieval."""

import copy
import logging
import random
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from app.core.config import Settings, get_settings
from app.ml.catalog import load_catalog_slug_mapping
from app.ml.evaluation import (
    build_evaluation_svd_training_matrix,
    popularity_order_rows,
    training_positive_counts,
)
from app.ml.faiss_index import create_faiss_index
from app.ml.historical_interactions import (
    PreparedInteractions,
    UserSplit,
    build_interaction_splits,
    load_historical_interactions,
)
from experiments.neural_retrieval.artifacts import (
    NCF_ARTIFACT_SCHEMA,
    NCF_EVALUATION_PROTOCOL,
    NCF_MODEL_TYPE,
    NCFArtifactMetadata,
    generate_candidate_vectors,
    write_ncf_artifacts,
)
from experiments.neural_retrieval.config import (
    NeuralRetrievalSettings,
    get_neural_retrieval_settings,
)
from experiments.neural_retrieval.data import (
    TrainingExample,
    iter_training_examples,
    make_evaluation_example,
)
from experiments.neural_retrieval.model import (
    InductiveNCFModel,
    multi_negative_bpr_loss,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NCFTrainingConfig:
    """Typed neural training and evaluation settings."""

    embedding_dim: int
    rating_embedding_dim: int
    hidden_dim: int
    dropout: float
    positive_rating_threshold: float
    negative_rating_threshold: float
    max_context_items: int
    targets_per_user_per_epoch: int
    negatives_per_positive: int
    temperature: float
    batch_size: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    random_seed: int
    training_device: Literal["cpu", "cuda"]
    exact_validation_interval: int

    @classmethod
    def from_settings(cls, settings: NeuralRetrievalSettings) -> "NCFTrainingConfig":
        """Build configuration from validated environment-backed settings."""
        return cls(
            embedding_dim=settings.NCF_EMBEDDING_DIM,
            rating_embedding_dim=settings.NCF_RATING_EMBEDDING_DIM,
            hidden_dim=128,
            dropout=settings.NCF_DROPOUT,
            positive_rating_threshold=settings.NCF_POSITIVE_RATING_THRESHOLD,
            negative_rating_threshold=settings.NCF_NEGATIVE_RATING_THRESHOLD,
            max_context_items=settings.NCF_MAX_CONTEXT_ITEMS,
            targets_per_user_per_epoch=settings.NCF_TARGETS_PER_USER_PER_EPOCH,
            negatives_per_positive=settings.NCF_NEGATIVES_PER_POSITIVE,
            temperature=settings.NCF_TEMPERATURE,
            batch_size=settings.NCF_BATCH_SIZE,
            max_epochs=settings.NCF_MAX_EPOCHS,
            learning_rate=settings.NCF_LEARNING_RATE,
            weight_decay=settings.NCF_WEIGHT_DECAY,
            early_stopping_patience=settings.NCF_EARLY_STOPPING_PATIENCE,
            random_seed=settings.NCF_RANDOM_SEED,
            training_device=settings.NCF_TRAINING_DEVICE,
            exact_validation_interval=settings.NCF_EXACT_VALIDATION_INTERVAL,
        )


@dataclass(frozen=True, slots=True)
class SampledValidationMetrics:
    """Diagnostic ranking against one positive and 99 sampled negatives."""

    loss: float
    recall_at_10: float
    ndcg_at_10: float
    evaluated_users: int


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Exact held-out retrieval measurements over the full candidate catalog."""

    recall_at_10: float
    recall_at_50: float
    ndcg_at_10: float
    mrr_at_10: float
    evaluated_users: int
    candidate_catalog_size: int


@dataclass(frozen=True, slots=True)
class EpochValidation:
    """One measured epoch's sampled and exact validation diagnostics."""

    epoch: int
    sampled: SampledValidationMetrics
    exact: RetrievalMetrics
    exact_runtime_seconds: float


@dataclass(frozen=True, slots=True)
class NCFTrainingResult:
    """Final training, evaluation, baseline, and artifact summary."""

    seed: int
    training_users: int
    training_interactions: int
    candidate_films: int
    best_epoch: int
    sampled_best_epoch: int
    device: str
    training_loss: float
    sampled_validation_metrics: SampledValidationMetrics
    exact_validation_metrics: RetrievalMetrics
    test_metrics: RetrievalMetrics
    svd_test_metrics: RetrievalMetrics
    popularity_test_metrics: RetrievalMetrics
    sampled_exact_ndcg_correlation: float | None
    exact_validation_runtime_seconds: float
    training_runtime_seconds: float
    total_runtime_seconds: float
    validation_history: tuple[EpochValidation, ...]
    artifact_root: Path | None


def train_ncf_model(
    *,
    csv_path: str | Path | None = None,
    artifact_root: str | Path | None = None,
    settings: Settings | None = None,
    experiment_settings: NeuralRetrievalSettings | None = None,
    config: NCFTrainingConfig | None = None,
    slug_to_film_id: Mapping[str, int] | None = None,
    prepared_data: PreparedInteractions | None = None,
    persist_artifacts: bool = True,
) -> NCFTrainingResult:
    """Train, select, evaluate, and optionally serialize one NCF run."""
    active_settings = settings or get_settings()
    active_config = config or NCFTrainingConfig.from_settings(
        experiment_settings or get_neural_retrieval_settings()
    )
    started = time.perf_counter()
    data = prepared_data
    if data is None:
        source = Path(csv_path or (active_settings.ARTIFACT_ROOT / "users_data.csv"))
        mapping = (
            dict(slug_to_film_id)
            if slug_to_film_id is not None
            else load_catalog_slug_mapping(active_settings)
        )
        data = load_historical_interactions(source, mapping)
    output_root = Path(artifact_root or (active_settings.ARTIFACT_ROOT / "ncf"))
    return _run_ncf_experiment(
        data,
        config=active_config,
        output_root=output_root,
        persist_artifacts=persist_artifacts,
        started=started,
    )


def benchmark_ncf_models(
    seeds: Sequence[int],
    *,
    csv_path: str | Path | None = None,
    settings: Settings | None = None,
    experiment_settings: NeuralRetrievalSettings | None = None,
    slug_to_film_id: Mapping[str, int] | None = None,
) -> tuple[NCFTrainingResult, ...]:
    """Run isolated, non-persisted experiments over independently split seeds."""
    if not seeds:
        raise ValueError("at least one benchmark seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("benchmark seeds must be unique")
    active_settings = settings or get_settings()
    source = Path(csv_path or (active_settings.ARTIFACT_ROOT / "users_data.csv"))
    mapping = (
        dict(slug_to_film_id)
        if slug_to_film_id is not None
        else load_catalog_slug_mapping(active_settings)
    )
    data = load_historical_interactions(source, mapping)
    base_config = NCFTrainingConfig.from_settings(
        experiment_settings or get_neural_retrieval_settings()
    )
    results: list[NCFTrainingResult] = []
    for seed in seeds:
        results.append(
            train_ncf_model(
                settings=active_settings,
                config=replace(base_config, random_seed=int(seed)),
                prepared_data=data,
                persist_artifacts=False,
            )
        )
    return tuple(results)


def _run_ncf_experiment(
    data: PreparedInteractions,
    *,
    config: NCFTrainingConfig,
    output_root: Path,
    persist_artifacts: bool,
    started: float,
) -> NCFTrainingResult:
    _set_reproducibility(config.random_seed)
    device = _resolve_training_device(config.training_device)
    summary = data.summary
    logger.info(
        "Controlled NCF data: csv_rows=%d resolved=%d unresolved=%d users=%d "
        "films=%d duplicates=%d seed=%d device=%s",
        summary.csv_rows,
        summary.resolved_rows,
        summary.unresolved_rows,
        summary.unique_users,
        summary.unique_films,
        summary.duplicate_rows,
        config.random_seed,
        device.type,
    )
    splits = build_interaction_splits(
        data,
        positive_rating_threshold=config.positive_rating_threshold,
        negative_rating_threshold=config.negative_rating_threshold,
        seed=config.random_seed,
    )
    model = InductiveNCFModel(
        len(data.film_ids),
        embedding_dim=config.embedding_dim,
        rating_embedding_dim=config.rating_embedding_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_exact: RetrievalMetrics | None = None
    best_sampled_loss = float("inf")
    best_training_loss = float("inf")
    best_sampled = SampledValidationMetrics(float("inf"), 0.0, 0.0, 0)
    sampled_best_epoch = 0
    sampled_best_score = float("-inf")
    sampled_best_loss = float("inf")
    stale_measurements = 0
    validation_history: list[EpochValidation] = []
    exact_runtime_total = 0.0
    training_started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        training_loss = _train_epoch(
            model,
            optimizer,
            splits,
            config=config,
            epoch=epoch,
            device=device,
        )
        sampled = evaluate_sampled_validation(
            model,
            splits,
            config=config,
            device=device,
        )
        if sampled.ndcg_at_10 > sampled_best_score + 1e-8 or (
            abs(sampled.ndcg_at_10 - sampled_best_score) <= 1e-8
            and sampled.loss < sampled_best_loss
        ):
            sampled_best_epoch = epoch
            sampled_best_score = sampled.ndcg_at_10
            sampled_best_loss = sampled.loss

        measure_exact = (
            epoch == 1
            or epoch % config.exact_validation_interval == 0
            or epoch == config.max_epochs
        )
        if not measure_exact:
            logger.info(
                "NCF epoch=%d train_loss=%.6f sampled_val_loss=%.6f "
                "sampled_val_recall@10=%.4f sampled_val_ndcg@10=%.4f",
                epoch,
                training_loss,
                sampled.loss,
                sampled.recall_at_10,
                sampled.ndcg_at_10,
            )
            continue

        exact_started = time.perf_counter()
        item_vectors = generate_candidate_vectors(model, device=device)
        retrieval_index = create_faiss_index(item_vectors, data.film_ids)
        exact = evaluate_exact_retrieval(
            model,
            splits,
            data.film_ids,
            retrieval_index,
            target_kind="validation",
            device=device,
        )
        exact_runtime = time.perf_counter() - exact_started
        exact_runtime_total += exact_runtime
        validation_history.append(EpochValidation(epoch, sampled, exact, exact_runtime))
        improved = is_better_checkpoint(
            exact,
            best_exact,
            sampled_loss=sampled.loss,
            best_sampled_loss=best_sampled_loss,
        )
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_exact = exact
            best_sampled_loss = sampled.loss
            best_training_loss = training_loss
            best_sampled = sampled
            stale_measurements = 0
        else:
            stale_measurements += 1
        logger.info(
            "NCF epoch=%d train_loss=%.6f sampled_val_loss=%.6f "
            "sampled_val_recall@10=%.4f sampled_val_ndcg@10=%.4f "
            "exact_val_recall@10=%.4f exact_val_recall@50=%.4f "
            "exact_val_ndcg@10=%.4f exact_val_mrr@10=%.4f exact_seconds=%.3f",
            epoch,
            training_loss,
            sampled.loss,
            sampled.recall_at_10,
            sampled.ndcg_at_10,
            exact.recall_at_10,
            exact.recall_at_50,
            exact.ndcg_at_10,
            exact.mrr_at_10,
            exact_runtime,
        )
        if stale_measurements >= config.early_stopping_patience:
            logger.info("NCF early stopping at epoch=%d", epoch)
            break

    if best_state is None or best_exact is None:
        raise RuntimeError("NCF training produced no exact-validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    item_vectors = generate_candidate_vectors(model, device=device)
    retrieval_index = create_faiss_index(item_vectors, data.film_ids)
    benchmark_started = time.perf_counter()
    (
        test_metrics,
        svd_test_metrics,
        popularity_test_metrics,
        ncf_test_runtime,
    ) = evaluate_test_benchmarks(
        model,
        splits,
        data.film_ids,
        retrieval_index,
        seed=config.random_seed,
        device=device,
    )
    training_runtime = benchmark_started - training_started + ncf_test_runtime
    correlation = sampled_exact_ndcg_correlation(validation_history)

    if persist_artifacts:
        metadata = NCFArtifactMetadata(
            artifact_schema=NCF_ARTIFACT_SCHEMA,
            model_type=NCF_MODEL_TYPE,
            embedding_dim=config.embedding_dim,
            rating_embedding_dim=config.rating_embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            item_count=len(data.film_ids),
            positive_rating_threshold=config.positive_rating_threshold,
            negative_rating_threshold=config.negative_rating_threshold,
            training_seed=config.random_seed,
            best_epoch=best_epoch,
            sampled_best_epoch=sampled_best_epoch,
            device=device.type,
            checkpoint_selection_metric="exact_validation_ndcg_at_10",
            evaluation_protocol=NCF_EVALUATION_PROTOCOL,
            data_summary=asdict(summary),
            training_metrics={
                "ranking_loss": best_training_loss,
                "training_runtime_seconds": training_runtime,
                "exact_validation_runtime_seconds": exact_runtime_total,
                "sampled_exact_ndcg_correlation": (
                    correlation if correlation is not None else 0.0
                ),
            },
            sampled_validation_metrics=asdict(best_sampled),
            exact_validation_metrics=asdict(best_exact),
            test_metrics=asdict(test_metrics),
            svd_test_metrics=asdict(svd_test_metrics),
            popularity_test_metrics=asdict(popularity_test_metrics),
        )
        write_ncf_artifacts(
            output_root,
            model=model,
            metadata=metadata,
            film_ids=data.film_ids,
            item_vectors=item_vectors,
        )

    return NCFTrainingResult(
        seed=config.random_seed,
        training_users=summary.unique_users,
        training_interactions=summary.resolved_rows - summary.duplicate_rows,
        candidate_films=summary.unique_films,
        best_epoch=best_epoch,
        sampled_best_epoch=sampled_best_epoch,
        device=device.type,
        training_loss=best_training_loss,
        sampled_validation_metrics=best_sampled,
        exact_validation_metrics=best_exact,
        test_metrics=test_metrics,
        svd_test_metrics=svd_test_metrics,
        popularity_test_metrics=popularity_test_metrics,
        sampled_exact_ndcg_correlation=correlation,
        exact_validation_runtime_seconds=exact_runtime_total,
        training_runtime_seconds=training_runtime,
        total_runtime_seconds=time.perf_counter() - started,
        validation_history=tuple(validation_history),
        artifact_root=output_root if persist_artifacts else None,
    )


def _resolve_training_device(requested: Literal["cpu", "cuda"]) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "NCF_TRAINING_DEVICE=cuda was requested but CUDA is unavailable; "
            "standard training defaults to CPU"
        )
    return torch.device(requested)


def is_better_checkpoint(
    exact: RetrievalMetrics,
    best_exact: RetrievalMetrics | None,
    *,
    sampled_loss: float,
    best_sampled_loss: float,
) -> bool:
    """Select by exact NDCG@10, exact Recall@10, then sampled loss."""
    if best_exact is None:
        return True
    if exact.ndcg_at_10 > best_exact.ndcg_at_10 + 1e-8:
        return True
    if abs(exact.ndcg_at_10 - best_exact.ndcg_at_10) > 1e-8:
        return False
    if exact.recall_at_10 > best_exact.recall_at_10 + 1e-8:
        return True
    if abs(exact.recall_at_10 - best_exact.recall_at_10) > 1e-8:
        return False
    return sampled_loss < best_sampled_loss


def collate_examples(
    examples: Sequence[TrainingExample],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length histories and move one ranking batch to a device."""
    if not examples:
        raise ValueError("cannot collate an empty example batch")
    max_history = max(len(example.context_rows) for example in examples)
    history_rows = np.zeros((len(examples), max_history), dtype=np.int64)
    history_ratings = np.zeros_like(history_rows)
    history_mask = np.zeros_like(history_rows, dtype=np.bool_)
    for index, example in enumerate(examples):
        length = len(example.context_rows)
        history_rows[index, :length] = example.context_rows
        history_ratings[index, :length] = example.context_ratings
        history_mask[index, :length] = True
    positive_rows = np.asarray(
        [example.positive_row for example in examples], dtype=np.int64
    )
    negative_rows = np.stack([example.negative_rows for example in examples]).astype(
        np.int64, copy=False
    )
    return (
        torch.as_tensor(history_rows, device=device),
        torch.as_tensor(history_ratings, device=device),
        torch.as_tensor(history_mask, device=device),
        torch.as_tensor(positive_rows, device=device),
        torch.as_tensor(negative_rows, device=device),
    )


def evaluate_sampled_validation(
    model: InductiveNCFModel,
    splits: tuple[UserSplit, ...],
    *,
    config: NCFTrainingConfig,
    device: torch.device,
) -> SampledValidationMetrics:
    """Measure diagnostic validation against 99 deterministic negatives."""
    examples = [
        example
        for user in splits
        if user.validation_target is not None
        for example in [
            make_evaluation_example(
                user,
                target=user.validation_target,
                item_count=model.item_count,
                seed=config.random_seed,
                negative_count=99,
            )
        ]
        if example is not None
    ]
    if not examples:
        return SampledValidationMetrics(float("inf"), 0.0, 0.0, 0)
    model.eval()
    losses: list[float] = []
    ranks: list[int] = []
    with torch.inference_mode():
        for batch in _batched(examples, min(config.batch_size, 32)):
            tensors = collate_examples(batch, device=device)
            user_vectors, positive_vectors, negative_vectors = model(*tensors)
            loss = multi_negative_bpr_loss(
                user_vectors,
                positive_vectors,
                negative_vectors,
                temperature=config.temperature,
            )
            losses.append(float(loss.item()) * len(batch))
            positive_scores = (user_vectors * positive_vectors).sum(dim=-1)
            negative_scores = torch.einsum("bd,bnd->bn", user_vectors, negative_vectors)
            ranks.extend(
                int(value)
                for value in (
                    (negative_scores >= positive_scores.unsqueeze(1)).sum(dim=1) + 1
                ).cpu()
            )
    return SampledValidationMetrics(
        loss=sum(losses) / len(examples),
        recall_at_10=_recall_at(ranks, 10),
        ndcg_at_10=_ndcg_at(ranks, 10),
        evaluated_users=len(ranks),
    )


def evaluate_exact_retrieval(
    model: InductiveNCFModel,
    splits: tuple[UserSplit, ...],
    film_ids: NDArray[np.int64],
    retrieval_index: faiss.IndexIDMap2,
    *,
    target_kind: Literal["validation", "test"],
    device: torch.device,
    batch_size: int = 64,
) -> RetrievalMetrics:
    """Evaluate one untouched target per user over the full exact catalog."""
    eligible: list[tuple[UserSplit, int]] = []
    for user in splits:
        target = (
            user.validation_target if target_kind == "validation" else user.test_target
        )
        if target is not None and len(user.context_item_rows):
            eligible.append((user, target))
    ranks: list[int | None] = []
    model.eval()
    with torch.inference_mode():
        for batch in _batched_pairs(eligible, batch_size):
            max_history = max(len(user.context_item_rows) for user, _ in batch)
            rows = np.zeros((len(batch), max_history), dtype=np.int64)
            ratings = np.zeros_like(rows)
            mask = np.zeros_like(rows, dtype=np.bool_)
            exclusions: list[set[int]] = []
            targets: list[int] = []
            for position, (user, target_row) in enumerate(batch):
                length = len(user.context_item_rows)
                rows[position, :length] = user.context_item_rows
                ratings[position, :length] = user.context_rating_buckets
                mask[position, :length] = True
                exclusions.append(
                    {int(film_ids[row]) for row in user.context_item_rows}
                )
                targets.append(int(film_ids[target_row]))
            query = np.ascontiguousarray(
                model.encode_history(
                    torch.as_tensor(rows, device=device),
                    torch.as_tensor(ratings, device=device),
                    torch.as_tensor(mask, device=device),
                )
                .cpu()
                .numpy(),
                dtype=np.float32,
            )
            requested_k = min(
                int(retrieval_index.ntotal),
                50 + max(len(excluded) for excluded in exclusions),
            )
            _, labels = retrieval_index.search(query, requested_k)
            for row_labels, excluded, target_film_id in zip(
                labels, exclusions, targets, strict=True
            ):
                filtered = [
                    int(label)
                    for label in row_labels
                    if int(label) >= 0 and int(label) not in excluded
                ][:50]
                ranks.append(
                    filtered.index(target_film_id) + 1
                    if target_film_id in filtered
                    else None
                )
    return _retrieval_metrics(ranks, int(retrieval_index.ntotal))


def evaluate_leakage_free_svd(
    splits: tuple[UserSplit, ...],
    film_ids: NDArray[np.int64],
    *,
    seed: int,
) -> RetrievalMetrics:
    """Fit temporary SVD from shared training splits and evaluate shared tests."""
    matrix = build_evaluation_svd_training_matrix(splits, len(film_ids))
    svd = TruncatedSVD(n_components=32, random_state=seed)
    svd.fit(matrix)
    item_vectors = np.ascontiguousarray(
        normalize(svd.components_.T, axis=1).astype(np.float32), dtype=np.float32
    )
    retrieval_index = create_faiss_index(item_vectors, film_ids)
    queries: list[NDArray[np.float32]] = []
    exclusions: list[set[int]] = []
    targets: list[int] = []
    for user in splits:
        if user.test_target is None or not len(user.context_item_rows):
            continue
        queries.append(item_vectors[user.context_item_rows].mean(axis=0))
        exclusions.append({int(film_ids[row]) for row in user.context_item_rows})
        targets.append(int(film_ids[user.test_target]))
    if not queries:
        return _retrieval_metrics([], len(film_ids))
    query_matrix = np.ascontiguousarray(np.stack(queries), dtype=np.float32)
    requested_k = min(
        int(retrieval_index.ntotal),
        50 + max(len(excluded) for excluded in exclusions),
    )
    _, labels = retrieval_index.search(query_matrix, requested_k)
    ranks = _ranks_from_search(labels, exclusions, targets)
    return _retrieval_metrics(ranks, int(retrieval_index.ntotal))


def evaluate_test_benchmarks(
    model: InductiveNCFModel,
    splits: tuple[UserSplit, ...],
    film_ids: NDArray[np.int64],
    retrieval_index: faiss.IndexIDMap2,
    *,
    seed: int,
    device: torch.device,
) -> tuple[RetrievalMetrics, RetrievalMetrics, RetrievalMetrics, float]:
    """Evaluate all systems against the identical shared test split."""
    ncf_started = time.perf_counter()
    ncf = evaluate_exact_retrieval(
        model,
        splits,
        film_ids,
        retrieval_index,
        target_kind="test",
        device=device,
    )
    ncf_runtime = time.perf_counter() - ncf_started
    svd = evaluate_leakage_free_svd(splits, film_ids, seed=seed)
    popularity = evaluate_popularity_baseline(splits, film_ids)
    return ncf, svd, popularity, ncf_runtime


def build_popularity_ranking(
    splits: tuple[UserSplit, ...], film_ids: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Order all films by training-positive count, breaking ties by film ID."""
    counts = training_positive_counts(splits, len(film_ids))
    order = popularity_order_rows(counts, film_ids)
    return np.ascontiguousarray(film_ids[order], dtype=np.int64)


def evaluate_popularity_baseline(
    splits: tuple[UserSplit, ...], film_ids: NDArray[np.int64]
) -> RetrievalMetrics:
    """Evaluate deterministic training-positive popularity on shared tests."""
    ranking = build_popularity_ranking(splits, film_ids)
    ranks: list[int | None] = []
    for user in splits:
        if user.test_target is None or not len(user.context_item_rows):
            continue
        excluded = {int(film_ids[row]) for row in user.context_item_rows}
        target = int(film_ids[user.test_target])
        candidates = [int(value) for value in ranking if int(value) not in excluded][
            :50
        ]
        ranks.append(candidates.index(target) + 1 if target in candidates else None)
    return _retrieval_metrics(ranks, len(film_ids))


def sampled_exact_ndcg_correlation(
    history: Sequence[EpochValidation],
) -> float | None:
    """Return Pearson correlation across epochs when it is mathematically defined."""
    if len(history) < 2:
        return None
    sampled = np.asarray([entry.sampled.ndcg_at_10 for entry in history])
    exact = np.asarray([entry.exact.ndcg_at_10 for entry in history])
    if np.std(sampled) <= 1e-12 or np.std(exact) <= 1e-12:
        return None
    return float(np.corrcoef(sampled, exact)[0, 1])


def _train_epoch(
    model: InductiveNCFModel,
    optimizer: torch.optim.Optimizer,
    splits: tuple[UserSplit, ...],
    *,
    config: NCFTrainingConfig,
    epoch: int,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    example_count = 0
    examples = iter_training_examples(
        splits,
        item_count=model.item_count,
        epoch=epoch,
        seed=config.random_seed,
        targets_per_user=config.targets_per_user_per_epoch,
        max_context_items=config.max_context_items,
        negatives_per_positive=config.negatives_per_positive,
    )
    for batch in _batched(examples, config.batch_size):
        tensors = collate_examples(batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        user_vectors, positive_vectors, negative_vectors = model(*tensors)
        loss = multi_negative_bpr_loss(
            user_vectors,
            positive_vectors,
            negative_vectors,
            temperature=config.temperature,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("NCF training loss became non-finite")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(batch)
        example_count += len(batch)
    if not example_count:
        raise RuntimeError("controlled data produced no NCF training examples")
    return total_loss / example_count


def _ranks_from_search(
    labels: NDArray[np.int64],
    exclusions: Sequence[set[int]],
    targets: Sequence[int],
) -> list[int | None]:
    ranks: list[int | None] = []
    for row_labels, excluded, target in zip(labels, exclusions, targets, strict=True):
        filtered = [
            int(label)
            for label in row_labels
            if int(label) >= 0 and int(label) not in excluded
        ][:50]
        ranks.append(filtered.index(target) + 1 if target in filtered else None)
    return ranks


def _retrieval_metrics(
    ranks: Sequence[int | None], candidate_count: int
) -> RetrievalMetrics:
    return RetrievalMetrics(
        recall_at_10=_recall_at(ranks, 10),
        recall_at_50=_recall_at(ranks, 50),
        ndcg_at_10=_ndcg_at(ranks, 10),
        mrr_at_10=_mrr_at(ranks, 10),
        evaluated_users=len(ranks),
        candidate_catalog_size=candidate_count,
    )


def _recall_at(ranks: Sequence[int | None], cutoff: int) -> float:
    return (
        sum(rank is not None and rank <= cutoff for rank in ranks) / len(ranks)
        if ranks
        else 0.0
    )


def _ndcg_at(ranks: Sequence[int | None], cutoff: int) -> float:
    return (
        sum(
            1.0 / np.log2(rank + 1)
            for rank in ranks
            if rank is not None and rank <= cutoff
        )
        / len(ranks)
        if ranks
        else 0.0
    )


def _mrr_at(ranks: Sequence[int | None], cutoff: int) -> float:
    return (
        sum(1.0 / rank for rank in ranks if rank is not None and rank <= cutoff)
        / len(ranks)
        if ranks
        else 0.0
    )


def _batched(
    values: Iterable[TrainingExample], batch_size: int
) -> Iterator[list[TrainingExample]]:
    batch: list[TrainingExample] = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _batched_pairs(
    values: Sequence[tuple[UserSplit, int]], batch_size: int
) -> Iterator[list[tuple[UserSplit, int]]]:
    for start in range(0, len(values), batch_size):
        yield list(values[start : start + batch_size])


def _set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
