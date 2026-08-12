# Categorized recommendation policy V1

This package evaluates NexdWatch's first internal portfolio policy. It does not
participate in FastAPI startup or replace the live `SVD_Mean_Pooling` service.
The frozen input is the existing 2,000 positive-weighted-SVD candidates plus
2,000 controlled-popularity candidates, followed by exact equal-weight RRF with
`k=60`.

## Runtime design

One request loads user history once, reuses it for candidate exclusion and one
`UserCategoryProfile`, applies RRF, builds all viable proposals in memory,
allocates at most ten rows, and batch-loads final display films. Film metadata
is held in a service-scoped immutable snapshot populated by five bounded SQL
queries: one scalar film query and one query for each of directors, genres,
countries, and languages.

The profile computes every entity preference from explicit ratings only:

```text
user_mean = mean(all explicit ratings)
raw_preference = mean(rating - user_mean for films containing entity)
confidence = support / (support + lambda)
affinity = raw_preference * confidence
```

`lambda=5` for directors and `lambda=4` for genres, decades, countries, and
languages. Unrated watches contribute only to exclusion and history depth.
Qualification additionally requires support, positive/high-rating evidence,
positive affinity, and (for directors) bounded contradictory negative evidence.

## V1 semantic choices

- Brazilian cinema means any film whose available country metadata contains
  `Brazil` or `Brasil`. The schema does not identify a primary production
  country, so international co-productions can qualify.
- World Cinema requires both a non-core-English country association and a
  non-English language association. Core English countries are configured
  explicitly; user locale is not inferred.
- Outside Your Usual Picks excludes HEAD films and requires positive SVD
  retrieval within rank 500, RRF rank at most 1,000, and no match against the
  strongest qualifying director, genre, decade, country, or language entities.
- Classic Cinema uses `year <= 1969` and positive personalized SVD evidence.
- Anchor rows prefer ratings at least 4.5. A 4.0 fallback is used only when no
  4.5/5.0 indexed anchor exists. Candidate similarities are computed only over
  the existing inventory in bounded vector batches.

Every default and diversity limit is centralized in
`app/policy/config.py`. These are transparent product-policy choices, not
test-optimized thresholds.

## Strict offline evaluation

Run the complete read-only protocol with:

```bash
python manage.py evaluate-categories
```

The evaluator uses the existing three seeds and five strict out-of-user folds.
Fold artifacts are built from training users only. A test user's designated
target is never inserted into their context, profile evidence, anchor choice,
or candidate inventory. Semantic recall denominators are category-specific;
ordinary held-out recall is secondary for serendipity.

The sensitivity sample checks neighboring Hidden Gems depths, classic-year
boundaries, minimum row sizes, overlap cutoffs, broad support, director-pool
size, and the Outside-Usual HEAD exclusion. It is a diagnostic comparison, not
a test-label optimization procedure.

For a persisted read-only preview:

```bash
python manage.py preview-categories 3953
```

Offline portfolio metrics are proxies for behavior and inventory quality. They
must not be interpreted as CTR, conversion, retention, or user satisfaction.
