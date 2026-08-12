# Categorized recommendation policy V1.1 results

Evaluation date: 2026-08-11. Historical V1 conclusions remain in
`RESULTS.md`. The ignored machine-readable reports are
`data/analysis/category_policy_v1_1.json`,
`data/analysis/category_policy_refinement.json`,
`data/analysis/category_policy_qualitative.json`, and
`data/analysis/category_policy_benchmark.json`.

The strict result uses seeds `42,43,44` and all five out-of-user folds: 5,898
user appearances. Fold artifacts use training users only. Held-out targets do
not enter profiles, anchors, preferences, retrieval, or semantic eligibility.
Threshold selection used context-only diagnostics, not test targets.

## Selected corrections

- Anchor alternatives: legacy positive cosine, top 50, top 100, top 200, top
  5%, and top 10% of the existing candidate inventory. Top 100 was the smallest
  fixed rule with materially more viable capacity than top 50 while remaining
  genuinely local. It never performs full-catalog anchor retrieval.
- Classic alternatives: unconstrained, maximum 12 HEAD, and maximum 10 HEAD.
  The selected maximum 12 is the weakest constraint with a material discovery
  improvement.
- World alternatives: unconstrained, maximum 12 HEAD, and maximum 10 HEAD. The
  selected maximum 12 preserves broad activation and country diversity while
  retaining recognizable gateway films.
- Outside alternatives: V1, depth 1,000, explicit Hidden exclusion at depths
  600/750/1,000, the strict lower-HEAD bridge, and the combined orthogonal
  depth-750 bridge. The selected rule excludes the structural Hidden-Gems
  neighborhood, requires at least 12 viable orthogonal candidates in the
  primary rank window, uses SVD depth 750/RRF depth 1,250, and permits at most
  four HEAD items only when SVD-only and SVD rank <=100.
- Director alternatives: support 2/high-rating 1, support 3/high-rating 1, and
  support 2/high-rating 2. The current threshold is retained.

On the fixed 79-user context sample, the selected Outside alternative activated
for 83.54%, averaged 18.89 films, had zero raw/final Hidden overlap and zero Top
overlap, and had mean SVD rank 303.14. Its mix was 13.79% HEAD, 6.82% MID, and
79.39% TAIL. The all-non-HEAD depth-1,000 alternative was less recognizable,
with mean SVD rank 351.54 and 92.15% TAIL.

## Strict V1 versus V1.1

| Category | Activation V1 -> V1.1 | Mean row V1 -> V1.1 | HEAD/MID/TAIL V1 | HEAD/MID/TAIL V1.1 |
|---|---:|---:|---:|---:|
| Top Picks | 100.0 -> 100.0% | 20.00 -> 20.00 | 92.5/6.4/1.1 | 92.5/6.4/1.1 |
| Hidden Gems | 98.6 -> 98.6% | 18.42 -> 18.44 | 0.0/69.1/30.9 | 0.0/69.0/31.0 |
| Brazilian Cinema | 73.9 -> 73.4% | 12.95 -> 12.93 | 25.1/25.8/49.1 | 25.3/25.8/49.0 |
| Because You Liked | 99.9 -> 99.9% | 20.00 -> 16.12 | 42.7/29.8/27.5 | 44.7/30.8/24.5 |
| Directors You Love | 98.4 -> 98.4% | 16.12 -> 16.17 | 80.9/16.3/2.8 | 81.0/16.3/2.7 |
| Favorite Genre | 79.8 -> 79.8% | 20.00 -> 20.00 | 73.7/16.0/10.3 | 73.9/15.8/10.3 |
| Favorite Decade | 84.9 -> 84.9% | 19.74 -> 19.76 | 88.7/9.5/1.8 | 89.3/8.9/1.8 |
| World Cinema | 99.8 -> 99.8% | 19.17 -> 18.33 | 72.0/11.1/16.8 | 59.8/15.2/25.0 |
| Outside Usual | 29.1 -> 84.0% | 17.40 -> 18.80 | 0.0/53.5/46.5 | 13.1/7.8/79.1 |
| Classic Cinema | 97.2 -> 97.2% | 17.82 -> 16.32 | 83.1/12.9/4.0 | 65.1/27.1/7.8 |

Healthy categories changed only indirectly through global allocation. Top Picks
is bit-for-bit identical in the aggregate. Hidden, Brazil, Genre, Decade, and
Director activation/strata remain effectively stable. Brazilian Cinema still
uses any Brazil/Brasil country association because the schema has no primary
country; co-productions may qualify.

World Cinema averages 17.27 distinct country associations per row versus 16.85
in V1. Mean maximum-country share is 27.29% versus 26.18%. Preference-supported
items are 54.59%; the remainder are intentional discovery beyond established
country/language preferences. Hidden-versus-Outside raw proposal Jaccard falls
from 0.7845 to 0.0; final overlap is also zero. Outside has zero Top overlap,
zero familiar-metadata matches, mean SVD rank 317.75, and 99.998% SVD-only
source membership (two items across all rows also appeared in popularity).

The portfolio rises from 8.617 to 9.161 categories/user and from 157.801 to
162.367 unique films/response. Duplicate-slot rate remains negligible at
0.0175% mean, with the existing maximum-two-appearances rule intact.

## Leakage-safe semantic targets

| Category | Eligible V1 -> V1.1 | Candidate recall V1 -> V1.1 | Row recall V1 -> V1.1 |
|---|---:|---:|---:|
| Because You Liked | 3,701 -> 194 | 86.33 -> 97.42% | 0.92 -> 13.40% |
| Classic Cinema | 576 -> 576 | 77.95 -> 77.95% | 12.85 -> 11.98% |
| World Cinema | 1,188 -> 1,188 | 77.10 -> 77.10% | 9.26 -> 8.67% |
| Directors You Love | 364 -> 364 | 95.88 -> 95.88% | 39.29 -> 39.01% |

The anchor denominator changed intentionally: V1 asked whether any positive
cosine target was selected, while V1.1 asks whether a target inside the chosen
top-100 local boundary was selected. The two recall percentages are therefore
not directly comparable. Classic and World accept small row-recall reductions
in exchange for the preselected discovery-balance objective. Outside ordinary
held-out recall is not used as a serendipity objective; its semantic denominator
also changes with the new policy and is not a V1/V1.1 optimization signal.

Anchors activate for 5,895/5,898 appearances. Every selected neighborhood is
top 100; mean rating is 4.999, mean top-20 similarity 0.8270, mean local cutoff
0.5875, and mean Top overlap 0.969%. For user 3953, both versions select the
5.0-rated `The Amulet of Ogum` and return the same strongest recommendations,
but the claimed neighborhood changes from 2,629 positive-cosine candidates to
top 100 (cutoff 0.9267, mean top-20 similarity 0.9761).

## Director audit

Across all strict appearances, selected directors average support 6.55 (median
5), rating 4.492 (median 4.5), high-rating count 4.02 (median 3), positive
fraction 0.963 (median 1.0), and affinity 0.539 (median 0.519). Of 86,280
selected records, 61,588 are strong-tier and 24,692 minimum-tier.

Qualifying-but-not-selected directors average support 4.66, rating 4.136,
high-rating count 1.86, positive fraction 0.896, and affinity 0.275. The selected
top 15 is therefore materially stronger than the permissive boundary. On the
context threshold audit, requiring two high ratings only raised selected mean
affinity from 0.546 to 0.552 while reducing the mean qualifying pool from
114.03 to 60.54. Minimum support three produced weaker selected rating and
affinity. No director threshold changes are justified.

## Qualitative review

Deterministic paired previews cover histories with 10, 31, 102, 501, 662,
1,000, 3,001, 4,994, and 9,892 watched entries. Examples include:

- User 3953: Classic changes from 18 HEAD/0 MID/2 TAIL to 12/5/3, retaining
  canonical films while adding less obvious historical cinema. World was
  already balanced at 11/5/4 and remains so. Outside now starts with the strict
  SVD-only gateways `Claire's Knee`, `Céline and Julie Go Boating`, `La
  Ciénaga`, and `Red Desert`, followed by deeper unfamiliar films.
- User 2504: Classic changes from 20/0/0 to 12/5/1 and World from 16/0/4 to
  12/3/5. Outside fills from deeper SVD evidence but remains highly obscure;
  this is a useful warning against claiming satisfaction from offline review.
- User 3318 (10 entries): the `Society` anchor contracts from 2,793 candidates
  to top 100 and yields nine films; Outside is omitted rather than forced.
- Deep genre users receive coherent cult/horror or cinephile rows rather than a
  single globally uniform interpretation of discovery.

The samples support semantic coherence for Anchor, Classic, and World. Outside
is materially more orthogonal and its gateway items improve recognizability,
but deep-tail quality should remain observable in a controlled rollout. This
inspection is not evidence of user satisfaction.

## Warm service resources

The benchmark loads one service and makes 24 uninstrumented sequential requests
after warmup, followed by a separate profiler-instrumented memory pass:

- artifact load: 118.71 ms and +54.55 MiB RSS;
- policy-catalog load: 1,362.72 ms and +63.49 MiB RSS;
- RSS after load/GC: 346.00 MiB;
- RSS after 24 warm requests/GC: 458.13 MiB;
- warm latency: mean 1,578.61 ms, median 1,274.60 ms, range 610.44-4,128.61 ms;
- temporary Python request allocation: mean 92.24 MiB, median 75.56 MiB,
  maximum 202.11 MiB in the separate tracemalloc pass.

Repeated relation entities are now interned in the service catalog without
changing its five-query shape. The profiler-instrumented 835.80 MiB process
high-water mark is not a warm-request RSS figure. The strict offline evaluator
took 1,368.79 seconds and peaked at 1,081.50 MiB while repeatedly constructing
fold artifacts; it is also not a production-request measurement.

## Readiness

Top Picks, Hidden Gems, Brazilian Cinema, Because You Liked, Directors You Love,
Favorite Genre, Favorite Decade, World Cinema, and Classic Cinema are
semantically suitable for a future versioned API contract. Outside Usual is
suitable only as an explicitly experimental/observable category until real
impression feedback and further editorial review establish that its deep-tail
results are surprising rather than random.

The V1.1 internal policy correction is ready to commit after validation. It is
not authorization to add a public route. Remaining serving work includes a
versioned response contract, cache/freshness lifecycle, observability,
category-level rollout controls, and an explicit latency/memory budget for the
target VPS. No public schema, route, model, candidate policy, RRF calibration,
affinity formula, migration, dependency, or production artifact changed here.
