"""Tests for the inductive neural retrieval architecture."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.ml.ratings import rating_to_bucket
from experiments.neural_retrieval.model import (
    InductiveNCFModel,
    multi_negative_bpr_loss,
)


@pytest.mark.parametrize(
    ("rating", "bucket"),
    [(value / 2, value - 1) for value in range(1, 11)],
)
def test_rating_bucket_mapping(rating: float, bucket: int) -> None:
    assert rating_to_bucket(rating) == bucket


@pytest.mark.parametrize("rating", [0.0, 5.5, 2.25, float("nan"), float("inf")])
def test_invalid_rating_bucket_is_rejected(rating: float) -> None:
    with pytest.raises(ValueError):
        rating_to_bucket(rating)


def test_model_outputs_are_normalized_and_mask_supports_variable_histories() -> None:
    model = InductiveNCFModel(8, embedding_dim=6, rating_embedding_dim=3, dropout=0)
    model.eval()
    rows = torch.tensor([[0, 1, 2], [3, 0, 0]])
    ratings = torch.tensor([[0, 5, 9], [4, 0, 0]])
    mask = torch.tensor([[True, True, True], [True, False, False]])

    with torch.inference_mode():
        users = model.encode_history(rows, ratings, mask)
        candidates = model.encode_candidates(torch.tensor([0, 4, 7]))

    assert users.shape == (2, 6)
    assert candidates.shape == (3, 6)
    torch.testing.assert_close(torch.linalg.vector_norm(users, dim=1), torch.ones(2))
    torch.testing.assert_close(
        torch.linalg.vector_norm(candidates, dim=1), torch.ones(3)
    )


def test_history_encoding_is_permutation_invariant_and_deterministic_in_eval() -> None:
    torch.manual_seed(7)
    model = InductiveNCFModel(8, embedding_dim=6, rating_embedding_dim=3)
    model.eval()
    rows = torch.tensor([[1, 4, 6]])
    ratings = torch.tensor([[0, 5, 9]])
    mask = torch.ones_like(rows, dtype=torch.bool)
    permutation = torch.tensor([2, 0, 1])

    with torch.inference_mode():
        first = model.encode_history(rows, ratings, mask)
        repeated = model.encode_history(rows, ratings, mask)
        permuted = model.encode_history(
            rows[:, permutation], ratings[:, permutation], mask
        )

    torch.testing.assert_close(first, repeated)
    torch.testing.assert_close(first, permuted, rtol=1e-5, atol=1e-6)


def test_identical_histories_encode_identically_without_user_identity() -> None:
    model = InductiveNCFModel(6, embedding_dim=4, rating_embedding_dim=2, dropout=0)
    model.eval()
    rows = torch.tensor([[1, 3], [1, 3]])
    ratings = torch.tensor([[2, 8], [2, 8]])
    mask = torch.ones_like(rows, dtype=torch.bool)

    with torch.inference_mode():
        vectors = model.encode_history(rows, ratings, mask)

    torch.testing.assert_close(vectors[0], vectors[1])
    parameter_names = [name for name, _ in model.named_parameters()]
    assert not any(
        "user" in name and "projection" not in name for name in parameter_names
    )
    assert not hasattr(model, "user_embedding")


def test_rating_embedding_changes_interaction_semantics() -> None:
    model = InductiveNCFModel(3, embedding_dim=4, rating_embedding_dim=2, dropout=0)
    model.eval()
    rows = torch.tensor([[1], [1]])
    ratings = torch.tensor([[rating_to_bucket(1.0)], [rating_to_bucket(5.0)]])
    mask = torch.ones_like(rows, dtype=torch.bool)

    with torch.inference_mode():
        vectors = model.encode_history(rows, ratings, mask)

    assert not torch.allclose(vectors[0], vectors[1])


def test_ranking_loss_is_finite_and_optimizer_improves_positive_margin() -> None:
    torch.manual_seed(11)
    model = InductiveNCFModel(5, embedding_dim=8, rating_embedding_dim=3, dropout=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    history_rows = torch.tensor([[0, 1]])
    history_ratings = torch.tensor([[8, 9]])
    history_mask = torch.ones_like(history_rows, dtype=torch.bool)
    positive = torch.tensor([2])
    negatives = torch.tensor([[3, 4]])

    def margin() -> float:
        model.eval()
        with torch.inference_mode():
            user, positive_vector, negative_vectors = model(
                history_rows,
                history_ratings,
                history_mask,
                positive,
                negatives,
            )
            return float(
                (user * positive_vector).sum()
                - torch.einsum("bd,bnd->bn", user, negative_vectors).mean()
            )

    initial_margin = margin()
    initial_weights = model.item_embedding.weight.detach().clone()
    for _ in range(40):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        vectors = model(
            history_rows,
            history_ratings,
            history_mask,
            positive,
            negatives,
        )
        loss = multi_negative_bpr_loss(*vectors, temperature=0.1)
        assert np.isfinite(float(loss.item()))
        loss.backward()
        optimizer.step()

    assert not torch.equal(initial_weights, model.item_embedding.weight)
    assert margin() > initial_margin
