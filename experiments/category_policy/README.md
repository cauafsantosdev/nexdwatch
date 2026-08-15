# Categorized recommendation policy V1.1

> **Current status:** Policy V1.1 is now the production categorized-feed policy at
> `GET /recommendations/{user_id}/feed`. The experiment runners remain offline and do
> not participate in FastAPI startup. See
> [the production methodology](../../docs/recommendation-system.md).

This package evaluates NexdWatch's internal portfolio policy. It does not
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

## Semantic choices

- Brazilian cinema means any film whose available country metadata contains
  `Brazil` or `Brasil`. The schema does not identify a primary production
  country, so international co-productions can qualify.
- World Cinema requires both a non-core-English country association and a
  non-English language association. Core English countries are configured
  explicitly; user locale is not inferred.
- Outside Your Usual Picks requires positive SVD retrieval, no match against
  the strongest qualifying director, genre, decade, country, or language
  entities, and explicit exclusion of the Hidden-Gems neighborhood. The V1.1
  depth is SVD rank 750/RRF rank 1,250. At most four HEAD items may enter, and
  only when they are SVD-only and within SVD rank 100; all other HEAD items are
  excluded.
- Classic Cinema uses `year <= 1969` and positive personalized SVD evidence.
  V1.1 caps HEAD at 12 items per row while preserving the relative RRF order of
  selected candidates.
- World Cinema retains its broad discovery semantics and maximum five films per
  country, while V1.1 caps HEAD at 12 items per row.
- Anchor rows prefer ratings at least 4.5. A 4.0 fallback is used only when no
  4.5/5.0 indexed anchor exists. V1.1 defines the local neighborhood as the top
  100 cosine neighbors in the existing candidate inventory. Similarities are
  computed only over that inventory in bounded vector batches; this is not
  full-catalog anchor retrieval.

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
size, and unrestricted Outside-Usual HEAD admission. It is a diagnostic
comparison, not a test-label optimization procedure.

The V1.1 thresholds are selected using a context-only bounded comparison:

```bash
python manage.py analyze-category-refinement
```

Historical V1 evidence remains in `RESULTS.md`; V1.1 evidence is written to
`RESULTS_V1_1.md` and a separate ignored machine-readable report.

Deterministic paired qualitative previews and the warm loaded-service benchmark
are available with:

```bash
python manage.py preview-category-refinement
python manage.py benchmark-categories --user-ids 3318,3569,3155,2825,3953,2474,2994,3724 --repetitions 3
```

For a persisted read-only preview:

```bash
python manage.py preview-categories 3953
```

Offline portfolio metrics are proxies for behavior and inventory quality. They
must not be interpreted as CTR, conversion, retention, or user satisfaction.
