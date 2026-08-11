"""Inductive neural collaborative retrieval model used by the experiment."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from app.ml.ratings import RATING_BUCKET_COUNT


class InductiveNCFModel(nn.Module):
    """Build user vectors exclusively from rated-film histories."""

    def __init__(
        self,
        item_count: int,
        *,
        embedding_dim: int = 64,
        rating_embedding_dim: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if item_count <= 0:
            raise ValueError("item_count must be positive")
        self.item_count = item_count
        self.embedding_dim = embedding_dim
        self.rating_embedding_dim = rating_embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.item_embedding = nn.Embedding(item_count, embedding_dim)
        self.rating_embedding = nn.Embedding(
            RATING_BUCKET_COUNT,
            rating_embedding_dim,
        )
        self.interaction_encoder = nn.Sequential(
            nn.Linear(embedding_dim + rating_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
        )
        self.user_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.candidate_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def encode_history(
        self,
        item_rows: Tensor,
        rating_buckets: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Encode padded, variable-length histories with masked mean pooling."""
        if item_rows.shape != rating_buckets.shape or item_rows.shape != mask.shape:
            raise ValueError("history tensors must have matching shapes")
        if item_rows.ndim != 2:
            raise ValueError("history tensors must have shape [batch, history]")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("every history must contain at least one interaction")

        item_vectors = self.item_embedding(item_rows)
        rating_vectors = self.rating_embedding(rating_buckets)
        interactions = self.interaction_encoder(
            torch.cat((item_vectors, rating_vectors), dim=-1)
        )
        float_mask = mask.unsqueeze(-1).to(dtype=interactions.dtype)
        pooled = (interactions * float_mask).sum(dim=1) / float_mask.sum(
            dim=1
        ).clamp_min(1.0)
        return F.normalize(self.user_projection(pooled), p=2, dim=-1)

    def encode_candidates(self, item_rows: Tensor) -> Tensor:
        """Encode candidate rows into normalized retrieval vectors."""
        return F.normalize(
            self.candidate_projection(self.item_embedding(item_rows)),
            p=2,
            dim=-1,
        )

    def forward(
        self,
        history_rows: Tensor,
        history_ratings: Tensor,
        history_mask: Tensor,
        positive_rows: Tensor,
        negative_rows: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return retrieval-compatible user, positive, and negative vectors."""
        return (
            self.encode_history(history_rows, history_ratings, history_mask),
            self.encode_candidates(positive_rows),
            self.encode_candidates(negative_rows),
        )


def multi_negative_bpr_loss(
    user_vectors: Tensor,
    positive_vectors: Tensor,
    negative_vectors: Tensor,
    *,
    temperature: float,
) -> Tensor:
    """Compute multi-negative Bayesian personalized ranking loss."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    positive_scores = (user_vectors * positive_vectors).sum(dim=-1, keepdim=True)
    negative_scores = torch.einsum("bd,bnd->bn", user_vectors, negative_vectors)
    return -F.logsigmoid((positive_scores - negative_scores) / temperature).mean()
