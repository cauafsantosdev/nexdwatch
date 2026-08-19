# Recommendation system

This document defines the production ranking and category policy. Research alternatives remain under [`experiments/`](../experiments/README.md).

## Training universe

Production training reads explicit, non-null PostgreSQL ratings and deduplicates `(user_id, film_id)`. One snapshot produces both the collaborative model and controlled popularity, so they share a film vocabulary and measurement time.

The historical research cohort has 4,300,105 resolved interactions from 1,976 users over 46,990 catalog films. It supports offline comparison but is not the source for scheduled production retraining.

## SVD representation and zero-shot profiles

TruncatedSVD fits **32 dimensions**. Normalized item vectors retain their actual `Film.id` mapping. Exact retrieval uses FAISS `IndexIDMap2(IndexFlatIP)`.

Users need not appear in the training cohort. NexdWatch builds a request-time vector from their rated, model-known films:

```text
weight = max(rating - 3.0, 0)
profile = weighted item-vector sum / absolute weight sum
```

Ratings at or below 3.0 add no positive preference. If all weights are zero, the SVD source is unavailable; popularity can still supply candidates.

## Candidate generation

Each source fills its nominal budget after watched exclusion:

1. up to 2,000 films from exact positive-weighted SVD retrieval;
2. up to 2,000 films from controlled popularity.

Popularity counts ratings `>= 3.5` in the same production snapshot and sorts by count descending, then `Film.id` ascending. It does not use `Film.total_logs`, average rating, or current product traffic.

Every watched film is excluded, including unrated and negatively rated watches. The two lists merge SVD-first into unique `RecommendationCandidate` values. Overlap is deduplicated without refill, so the union can contain fewer than 4,000 films.

## Global RRF order

Candidates receive equal-weight reciprocal-rank fusion with `k=60`:

```text
score = [1 / (60 + SVD rank), if present]
      + [1 / (60 + popularity rank), if present]
```

A missing term contributes zero. Scores sort descending and exact ties sort by film ID. The resulting `RankedCandidate` inventory also records source membership and a HEAD/MID/TAIL stratum from the full popularity universe.

RRF is fixed, not learned. Validation-only calibration did not improve reliably on equal source weights and `k=60`.

## User category profile

The category layer builds one `UserCategoryProfile` from explicit ratings and the loaded policy catalog. Unrated watches affect exclusion and history depth but add no preference evidence.

For directors, genres, decades, countries, and languages:

```text
user_mean = mean(all explicit ratings)
raw_preference = mean(rating - user_mean for rated films containing entity)
confidence = support / (support + smoothing)
affinity = raw_preference * confidence
```

Qualification also checks configured support, positive/high-rating evidence, positive affinity, and limited contradictory evidence for directors. Thresholds in `app/policy/config.py` are product policy, not learned parameters.

## V1.2 category policy

| Key | Display concept | Main eligibility or order |
| --- | --- | --- |
| `top_picks` | Top Picks | Global RRF order |
| `hidden_gems` | Hidden Gems for You | Strong non-head SVD neighborhood with enough indexed positive evidence |
| `brazilian_cinema` | Brazilian Cinema for You | Brazil/Brasil, Portuguese or no spoken language, and positive SVD evidence |
| `because_you_liked` | Because You Liked _Film_ | Top-100 local similarity neighborhood around a high-rated anchor |
| `directors_you_love` | From Directors You Love | Strongest supported directors with per-director caps |
| `favorite_genre` | _Genre_ Picks for You | Strongest qualified genre affinity |
| `favorite_decade` | _Decade_ Films for You | Strongest qualified decade affinity |
| `world_cinema` | World Cinema for You | Non-core-English country and non-English language; any English association excludes the film |
| `outside_usual` | Outside Your Usual Picks | Positive SVD match outside familiar metadata and the Hidden Gems neighborhood |
| `classic_cinema` | Classic Cinema for You | Year `<= 1969` with positive SVD evidence |

Most categories filter or reorder the RRF inventory. `because_you_liked` instead compares existing candidates with eligible anchors in bounded vector batches, then orders one top-100 neighborhood by similarity, RRF rank, and film ID. It does not search the full catalog.

### Cultural eligibility

**Brazilian Cinema** requires a `Brazil` or `Brasil` country association, a `Portuguese` or `No spoken language` association, and positive SVD evidence. This can exclude Brazilian films spoken entirely in another language because the schema does not identify a primary production country.

**World Cinema** requires a country outside the configured core-English set, at least one genuine non-English language, and no English association. Missing language metadata and `No spoken language` alone do not qualify. The policy does not infer user locale or primary language.

These V1.2 predicates postdate the V1.1 portfolio and serving benchmarks. Those reports retain their historical labels and measurements.

## Portfolio allocation

Allocation starts with Top Picks and reserves its leading ten films from later repetition. It then chooses proposals using history-depth role, evidence tier, novelty, overlap, support, and stable key ties. The result has at most ten categories and no more than two appearances per film. Focused shelves may reuse non-reserved films when needed. Hard director, country, category-specific HEAD, and appearance caps remain in force; only generic decade and genre caps may relax. Heavily overlapping or undersized proposals are dropped.

Public explanations contain stable reason codes and optional anchor or preference context. Internal ranks, scores, support counters, roles, diagnostics, and artifact paths stay internal. `outside_usual` remains explicitly experimental.

## Public endpoints

`GET /recommendations/{user_id}/feed` serves the categorized V1.2 feed.

`GET /users/{user_id}/recommendations` retains the legacy top-ten `SVD_Mean_Pooling` contract. It averages model-known rated vectors without rating weights and is separate from the positive-weighted RRF and category path.

## Experiment evidence

The [inductive neural retriever](../experiments/neural_retrieval/README.md) underperformed simpler baselines and remained research-only. The [LambdaRank benchmark](../experiments/ranker/RESULTS.md) underperformed RRF on the corrected full candidate pools. PyTorch and LightGBM are not production runtime dependencies.
