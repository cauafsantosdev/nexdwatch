# Categorized recommendation policy V1 results

> **Historical experiment report:** This V1 baseline predates the V1.1 refinement and
> public categorized feed. Measurements and readiness conclusions below are preserved
> in their original context. See [V1.1 results](RESULTS_V1_1.md) and
> [current production behavior](../../docs/recommendation-system.md).

Evaluation date: 2026-08-11. The ignored machine-readable report is
`data/analysis/category_policy.json`.

Protocol: three seeds (`42,43,44`) × five strict out-of-user test folds, 5,898
user appearances. Fold artifacts use training users only; category profiles,
anchors, and target-personalization eligibility use pre-target context only.

## Portfolio

- Categories/user: mean 8.617, median 9, range 3–10.
- Count distribution: 3: 2, 4: 1, 5: 13, 6: 169, 7: 608, 8: 1,589,
  9: 2,395, 10: 1,121.
- Unique films/response: mean 157.801, median 160, range 60–200.
- Duplicate-slot rate: mean 0.0212%, median 0%, maximum 5.369%; no film can
  exceed two appearances and Top Picks positions 1–10 are reserved.
- Mean policy-path runtime was 144.9 ms/user. The full evaluator took 1,132.5 s
  and peaked at 893.6 MiB including fold artifact construction.

`SVD/pop/both` below means source-only membership after candidate union.

| Category | Activation | Row min/median/max | HEAD/MID/TAIL | SVD/pop/both | Catalog coverage |
|---|---:|---:|---:|---:|---:|
| Top Picks | 100.0% | 20/20/20 | 92.5/6.4/1.1 | 9.4/12.5/78.1 | 16.47% |
| Hidden Gems | 98.6% | 12/20/20 | 0.0/69.1/30.9 | 99.0/0.0/1.0 | 37.95% |
| Brazilian Cinema | 73.9% | 8/12/20 | 25.1/25.8/49.1 | 81.0/0.0/19.0 | 1.67% |
| Because You Liked | 99.9% | 20/20/20 | 42.7/29.8/27.5 | 61.0/17.1/21.9 | 44.69% |
| Directors You Love | 98.4% | 8/16/20 | 80.9/16.3/2.8 | 23.8/23.1/53.1 | 12.01% |
| Favorite Genre | 79.8% | 16/20/20 | 73.7/16.0/10.3 | 30.4/23.1/46.5 | 15.39% |
| Favorite Decade | 84.9% | 12/20/20 | 88.7/9.5/1.8 | 14.2/34.3/51.4 | 11.09% |
| World Cinema | 99.8% | 12/20/20 | 72.0/11.1/16.8 | 32.1/14.5/53.4 | 15.86% |
| Outside Usual | 29.1% | 12/18/20 | 0.0/53.5/46.5 | 99.6/0.0/0.4 | 23.41% |
| Classic Cinema | 97.2% | 8/20/20 | 83.1/12.9/4.0 | 21.7/0.0/78.3 | 8.37% |

Mean raw top-row proposal overlap is low except Hidden Gems versus Outside
Usual (Jaccard 0.785). The next largest means are Top Picks/World Cinema 0.100,
Classic/Favorite Decade 0.091, Hidden/World 0.081, and Outside/World 0.071.
The allocator suppresses weak proposals over the configured 0.70 threshold;
final allocations then prefer globally unseen films.

World Cinema averaged 16.85 distinct country associations per row with mean
maximum-country share 26.18% (maximum 41.67%). Outside Usual contained zero
HEAD films, zero Top-Picks overlap, and zero matches to its familiar-metadata
profile; selected films had mean SVD rank 222.5.

## Leakage-safe semantic targets

Recall is zero-inclusive among independently eligible category targets. It is
not a satisfaction or engagement metric.

| Category | Eligible | Candidate recall | Category recall@row |
|---|---:|---:|---:|
| Top Picks | 5,898 | 83.89% | 14.06% |
| Hidden Gems | 1,417 | 49.47% | 3.95% |
| Brazilian Cinema | 43 | 90.70% | 39.53% |
| Because You Liked | 3,701 | 86.33% | 0.92% |
| Directors You Love | 364 | 95.88% | 39.29% |
| Favorite Genre | 272 | 77.21% | 20.22% |
| Favorite Decade | 256 | 83.59% | 27.34% |
| World Cinema | 1,188 | 77.10% | 9.26% |
| Outside Usual (secondary) | 1,194 | 49.75% | 1.01% |
| Classic Cinema | 576 | 77.95% | 12.85% |

Anchors activated for 5,895/5,898 appearances. Selected ratings averaged
4.999; usable positive-similarity inventory averaged 2,163 films and Top-Picks
overlap averaged 0.97%. This makes the current `similarity > 0` eligibility very
broad. Qualifying directors averaged 114.75/user; the top-15 pool averaged
14.63 selected directors and activated a viable row for 98.4% of users.

## Sensitivity and readiness

On the fixed 236-appearance diagnostic sample, the default averaged 8.737
categories. Most neighboring settings moved this between 8.62 and 8.86.
Notable effects:

- Brazilian activation moved from 84.7% at minimum-size −2 to 66.5% at +2.
- Classic activation moved 94.1% at a 1959 cutoff to 98.3% at 1979.
- Director activation moved 95.3% for a pool of 10 to 99.2% for 20.
- Outside activation moved 24.6%/39.4% at overlap 0.60/0.80 and jumped from
  30.5% to 89.4% when HEAD was allowed. HEAD exclusion should remain.

The internal engine is suitable to commit as an offline/internal V1, but no
category should be exposed through the public endpoint in this change.
Before public serving:

1. separate Outside Usual more clearly from Hidden Gems;
2. add MID/TAIL balancing to Classic Cinema, which is currently 83.1% HEAD;
3. calibrate a more selective anchor-neighborhood threshold on validation-only
   evidence and perform qualitative review;
4. audit near-universal World/Classic/Director activation and World Cinema's
   72.0% HEAD share;
5. disambiguate primary versus co-production country metadata where possible;
6. reduce/measure steady-state catalog memory on the target VPS;
7. define a versioned internal-to-public schema, cache/freshness lifecycle,
   observability, load tests, and controlled rollout before adding a route.
