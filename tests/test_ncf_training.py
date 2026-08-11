"""Focused tests for neural training configuration and evaluation helpers."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError

torch = pytest.importorskip("torch")

from app.core.config import get_settings
from app.ml import catalog
from app.ml.faiss_index import create_faiss_index
from app.ml.historical_interactions import UserSplit
from experiments.neural_retrieval import training as ncf_training
from experiments.neural_retrieval.config import NeuralRetrievalSettings
from experiments.neural_retrieval.data import TrainingExample
from experiments.neural_retrieval.training import (
    RetrievalMetrics,
    build_evaluation_svd_training_matrix,
    build_popularity_ranking,
    collate_examples,
    evaluate_exact_retrieval,
    evaluate_popularity_baseline,
    evaluate_test_benchmarks,
    is_better_checkpoint,
    load_catalog_slug_mapping,
)


def test_experimental_configuration_rejects_invalid_rating_threshold() -> None:
    payload = NeuralRetrievalSettings().model_dump()
    payload["NCF_POSITIVE_RATING_THRESHOLD"] = 3.0
    with pytest.raises(ValidationError):
        NeuralRetrievalSettings(**payload)


def test_standard_ncf_training_defaults_to_cpu_and_exact_validation_each_epoch() -> (
    None
):
    settings = NeuralRetrievalSettings()

    assert settings.NCF_TRAINING_DEVICE == "cpu"
    assert settings.NCF_EXACT_VALIDATION_INTERVAL == 1


def test_catalog_slug_mapping_uses_one_batched_query(monkeypatch) -> None:
    execute = SimpleNamespace(all=lambda: [("a", 10), ("b", 20)])
    execute_query = Mock(return_value=execute)
    connection = SimpleNamespace(execute=execute_query)

    class _ConnectionContext:
        def __enter__(self):
            return connection

        def __exit__(self, *args: object) -> None:
            return None

    engine = SimpleNamespace(
        connect=lambda: _ConnectionContext(),
        dispose=lambda: None,
    )
    monkeypatch.setattr(catalog, "create_engine", lambda _: engine)

    mapping = load_catalog_slug_mapping(get_settings())

    assert mapping == {"a": 10, "b": 20}
    execute_query.assert_called_once()


def test_collate_masks_padding_without_changing_variable_histories() -> None:
    examples = [
        TrainingExample(
            cohort_user_id=1,
            context_rows=np.array([0, 1, 2], dtype=np.int64),
            context_ratings=np.array([1, 2, 3], dtype=np.int64),
            positive_row=3,
            negative_rows=np.array([4, 5], dtype=np.int64),
        ),
        TrainingExample(
            cohort_user_id=2,
            context_rows=np.array([2], dtype=np.int64),
            context_ratings=np.array([9], dtype=np.int64),
            positive_row=4,
            negative_rows=np.array([0, 1], dtype=np.int64),
        ),
    ]

    rows, ratings, mask, positives, negatives = collate_examples(
        examples, device=torch.device("cpu")
    )

    assert rows.shape == ratings.shape == mask.shape == (2, 3)
    assert mask.tolist() == [[True, True, True], [True, False, False]]
    assert positives.tolist() == [3, 4]
    assert negatives.tolist() == [[4, 5], [0, 1]]


def _split(
    *,
    user_id: int,
    context: list[int],
    training_positives: list[int],
    validation: int | None,
    test: int | None,
) -> UserSplit:
    held_out = [target for target in (validation, test) if target is not None]
    all_rows = np.asarray(context + held_out, dtype=np.int64)
    return UserSplit(
        cohort_user_id=user_id,
        all_item_rows=all_rows,
        all_rating_buckets=np.full(len(all_rows), 7, dtype=np.int64),
        context_item_rows=np.asarray(context, dtype=np.int64),
        context_rating_buckets=np.full(len(context), 7, dtype=np.int64),
        training_positive_rows=np.asarray(training_positives, dtype=np.int64),
        explicit_negative_rows=np.array([], dtype=np.int64),
        validation_target=validation,
        test_target=test,
    )


def test_evaluation_svd_matrix_excludes_validation_and_test_targets() -> None:
    split = _split(
        user_id=1,
        context=[0, 3],
        training_positives=[0, 3],
        validation=1,
        test=2,
    )

    matrix = build_evaluation_svd_training_matrix((split,), item_count=4)

    assert matrix[0, 0] == 4.0
    assert matrix[0, 3] == 4.0
    assert matrix[0, split.validation_target] == 0.0
    assert matrix[0, split.test_target] == 0.0


def test_popularity_is_training_only_deterministic_and_excludes_context() -> None:
    film_ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    evaluated = _split(
        user_id=1,
        context=[0],
        training_positives=[0],
        validation=4,
        test=2,
    )
    contributor = _split(
        user_id=2,
        context=[1],
        training_positives=[1],
        validation=3,
        test=None,
    )

    ranking = build_popularity_ranking((evaluated, contributor), film_ids)
    metrics = evaluate_popularity_baseline((evaluated, contributor), film_ids)

    # Film 10 and 20 tie at one training positive, so actual film ID breaks the tie.
    # Held-out films 30/40/50 remain at zero and do not inflate their own scores.
    assert ranking.tolist() == [10, 20, 30, 40, 50]
    # For user 1, known film 10 is removed: film 30 is exactly rank two.
    assert metrics.recall_at_10 == 1.0
    assert metrics.recall_at_50 == 1.0
    assert metrics.ndcg_at_10 == pytest.approx(1 / np.log2(3))
    assert metrics.mrr_at_10 == pytest.approx(1 / 2)


class _ExactValidationModel:
    item_count = 4

    def eval(self) -> "_ExactValidationModel":
        return self

    def encode_history(
        self, rows: torch.Tensor, ratings: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        del ratings, mask
        return torch.tensor([[1.0, 0.0]]).repeat(len(rows), 1)


def test_exact_validation_uses_full_catalog_and_never_excludes_target() -> None:
    film_ids = np.array([100, 200, 300, 400], dtype=np.int64)
    vectors = np.array(
        [[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    index = create_faiss_index(vectors, film_ids)
    first = _split(
        user_id=1,
        context=[0],
        training_positives=[0],
        validation=1,
        test=2,
    )
    different_test_target = _split(
        user_id=1,
        context=[0],
        training_positives=[0],
        validation=1,
        test=3,
    )

    metrics = evaluate_exact_retrieval(
        _ExactValidationModel(),
        (first,),
        film_ids,
        index,
        target_kind="validation",
        device=torch.device("cpu"),
    )
    repeated = evaluate_exact_retrieval(
        _ExactValidationModel(),
        (different_test_target,),
        film_ids,
        index,
        target_kind="validation",
        device=torch.device("cpu"),
    )

    assert metrics == repeated
    assert metrics.candidate_catalog_size == 4
    assert metrics.recall_at_10 == 1.0
    assert metrics.ndcg_at_10 == 1.0


def test_checkpoint_selection_uses_exact_metrics_not_sampled_ndcg() -> None:
    epoch_a_exact = RetrievalMetrics(0.2, 0.4, 0.1, 0.08, 10, 100)
    epoch_b_exact = RetrievalMetrics(0.3, 0.5, 0.2, 0.12, 10, 100)
    epoch_a_sampled_ndcg = 0.9
    epoch_b_sampled_ndcg = 0.1

    selected_b = is_better_checkpoint(
        epoch_b_exact,
        epoch_a_exact,
        sampled_loss=0.5,
        best_sampled_loss=0.4,
    )

    assert epoch_a_sampled_ndcg > epoch_b_sampled_ndcg
    assert selected_b


def test_all_test_evaluators_receive_the_same_split_targets(monkeypatch) -> None:
    splits = (
        _split(
            user_id=1,
            context=[0],
            training_positives=[0],
            validation=1,
            test=2,
        ),
    )
    film_ids = np.array([10, 20, 30], dtype=np.int64)
    seen: list[tuple[int, tuple[int | None, ...]]] = []
    metrics = RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 1, 3)

    def record(*args: object, **__: object):
        received = args[0] if args[0] is splits else args[1]
        assert received is splits
        seen.append((id(received), tuple(user.test_target for user in received)))
        return metrics

    monkeypatch.setattr(ncf_training, "evaluate_exact_retrieval", record)
    monkeypatch.setattr(ncf_training, "evaluate_leakage_free_svd", record)
    monkeypatch.setattr(ncf_training, "evaluate_popularity_baseline", record)

    result = evaluate_test_benchmarks(
        Mock(),
        splits,
        film_ids,
        Mock(),
        seed=42,
        device=torch.device("cpu"),
    )

    assert result[:3] == (metrics, metrics, metrics)
    assert result[3] >= 0
    assert seen == [(id(splits), (2,))] * 3
