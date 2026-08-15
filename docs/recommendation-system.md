# Recommendation system

This document describes the frozen production methodology. Research alternatives
are preserved under [`experiments/`](../experiments/README.md) and are not selectable
application backends.

## Training universe

Production training reads explicit, non-null PostgreSQL ratings and deduplicates
`(user_id, film_id)`. An eligible user has at least one row in that universe. The
snapshot produces both the collaborative model and production popularity so the two
sources share the same film vocabulary and point in time.

The controlled historical research cohort contains 4,300,105 resolved interactions
from 1,976 users over 46,990 films. It supports reproducible offline comparisons but
is not the source of scheduled production retraining.

## SVD representation

TruncatedSVD fits 32 latent dimensions. Item vectors are normalized and stored with
their actual `Film.id` mapping. Exact retrieval uses FAISS
`IndexIDMap2(IndexFlatIP)`, so returned IDs remain stable database identifiers rather
than matrix row positions.

Production users need not have appeared during training. Their request-time vector
is built from currently rated films with:

```text
weight = max(rating - 3.0, 0)
profile = weighted vector sum / absolute weight sum
```

Ratings at or below 3.0 contribute no positive preference. The positive-weighted
profile does not silently fall back to mean SVD when it has no positive evidence.

## Candidate generation

Two deterministic sources are retrieved:

1. exact FAISS top 2,000 from the positive-weighted SVD profile;
2. controlled-popularity top 2,000.

Production popularity counts snapshot ratings `>= 3.5`, then orders by positive
count descending and actual `Film.id` ascending. It is not `Film.total_logs`, average
rating, or mutable product traffic. Research popularity uses the frozen historical
CSV and remains reproducible separately.

Every watched film is excluded, including unrated, liked, and disliked watches. The
source lists are combined without refill into one deterministic union that preserves
source membership and source ranks.

## Global ordering

The union is ranked by equal-weight reciprocal-rank fusion:

```text
RRF score = 0.5 / (60 + SVD rank) + 0.5 / (60 + popularity rank)
```

Missing-source terms contribute zero. Deterministic film-ID tie-breaking preserves
stable output. A bounded validation-only calibration found no reliable improvement
over the fixed 50/50, `k=60` configuration.

## Category policy V1.1

The finalized RRF ordering enters a portfolio policy with up to ten rows:

- Top Picks
- Hidden Gems
- Brazilian Cinema
- Because You Liked
- Directors You Love
- Favorite Genre
- Favorite Decade
- World Cinema
- Outside Your Usual Picks
- Classic Cinema

The policy uses explicit-rating preference evidence, candidate metadata, category
eligibility rules, diversity constraints, and global allocation. It does not retrain
or rescore the collaborative model. Public items include structured product-safe
reasons; internal scores, raw affinities, and policy diagnostics are not exposed.
`outside_usual` is explicitly marked experimental because offline novelty is not
evidence of user satisfaction.

## Why simpler production models were selected

The inductive neural retriever was designed for unseen users but underperformed both
controlled popularity and leakage-free SVD in the evaluated cohort, with limited
catalog coverage. See [neural retrieval](../experiments/neural_retrieval/README.md).

LightGBM LambdaRank was evaluated with corrected full candidate pools and strict
out-of-user folds. It underperformed conventional RRF on global NDCG@20 and showed
unstable checkpoint selection. See [ranking results](../experiments/ranker/RESULTS.md).

These negative results are retained as model-selection evidence. PyTorch and
LightGBM are not production dependencies.

## Public contracts

`GET /recommendations/{user_id}/feed` serves category policy V1.1.

`GET /users/{user_id}/recommendations` retains the original top-10
`SVD_Mean_Pooling` behavior for backward compatibility. The two endpoints should not
be interpreted as aliases.
