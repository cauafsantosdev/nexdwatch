# Categorized recommendation serving performance

> **Historical experiment report:** The measurements below justified the production
> categorized feed's V1.1 serving architecture. V1.2 later changed two cultural
> eligibility predicates, so the timings and exact fingerprints below remain V1.1
> measurements rather than a new V1.2 benchmark. Original pre-API recommendations
> are retained as historical context. See
> [the current architecture](../../docs/architecture.md).

This note records the semantics-preserving V1.1 serving pass run on 2026-08-12.
The generated JSON reports are intentionally ignored:

- `data/analysis/category_serving_performance_baseline.json`
- `data/analysis/category_serving_performance_optimized.json`

Reproduce the loaded-service profile with:

```bash
python manage.py profile-category-serving \
  --repetitions 5 \
  --rss-requests 100 \
  --report-path data/analysis/category_serving_performance.json
```

## Method and environment

The latency, stage, SQL, cProfile, tracemalloc, long-RSS, and concurrency passes
are isolated from one another. The latency pass is warm and uninstrumented; it
does not include artifact or catalog startup. The sample is the fixed persisted
user set `3318, 3569, 3155, 2825, 3953, 2504, 2474, 2994, 3724`.

- CPU: AMD Ryzen 5 5600X, 6 physical cores / 12 logical CPUs
- OS: Linux 7.1.6-arch1-1, x86_64, glibc 2.44
- Python 3.14.6
- NumPy 2.5.2, FAISS 1.14.3, SQLAlchemy 2.0.52
- Baseline samples: 27 requests (3 repetitions per user)
- Optimized samples: 45 requests (5 repetitions per user)

## Result

| Warm latency (ms) | Baseline | Optimized | Change |
| --- | ---: | ---: | ---: |
| Mean | 1,544.86 | 115.78 | -92.5% |
| Median | 1,285.26 | 122.57 | -90.5% |
| p90 | 2,629.58 | 180.62 | -93.1% |
| p95 | 2,913.09 | 192.22 | -93.4% |
| Min | 577.27 | 34.26 | -94.1% |
| Max | 3,066.49 | 255.23 | -91.7% |

The instrumented baseline was dominated by the final ORM display fetch: mean
1,366.23 ms, median 1,157.49 ms, p95 2,529.78 ms, and 87.95% of total request
time. The equivalent in-memory lookup is now mean 0.025 ms and p95 0.034 ms.

| Instrumented stage (ms) | Baseline mean | Optimized mean | Optimized median | Optimized p95 | Optimized mean share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Total request | 1,553.43 | 121.31 | 115.86 | 208.04 | 100.0% |
| History read | 8.49 | 8.36 | 2.72 | 24.86 | 6.9% |
| Candidate profile | 1.24 | 1.49 | 0.53 | 5.99 | 1.2% |
| Exact SVD retrieval | 4.56 | 21.37 | 22.36 | 39.27 | 17.6% |
| Popularity merge | 7.96 | 10.62 | 9.26 | 14.63 | 8.8% |
| RRF | 8.95 | 7.98 | 7.42 | 10.22 | 6.6% |
| User profile | 17.47 | 16.91 | 6.11 | 67.61 | 13.9% |
| Because You Liked | 25.58 | 12.03 | 14.46 | 22.40 | 9.9% |
| Allocation | 8.14 | 8.05 | 7.23 | 15.68 | 6.6% |
| Display lookup | 1,366.23 | 0.025 | 0.026 | 0.034 | <0.1% |
| Result materialization | 1.01 | 0.54 | 0.56 | 0.66 | 0.4% |

The detailed profile/proposal sub-stages were:

| Sub-stage (ms) | Baseline mean / median / p95 / share | Optimized mean / median / p95 / share |
| --- | ---: | ---: |
| Rating aggregation | 0.40 / 0.18 / 1.23 / <0.1% | 0.30 / 0.14 / 0.98 / 0.2% |
| Director preferences | 6.26 / 1.69 / 23.49 / 0.4% | 5.91 / 2.08 / 21.35 / 4.9% |
| Genre preferences | 3.04 / 0.88 / 13.20 / 0.2% | 2.96 / 0.87 / 12.59 / 2.4% |
| Decade preferences | 2.54 / 0.77 / 11.02 / 0.2% | 2.55 / 0.90 / 10.79 / 2.1% |
| Country preferences | 1.94 / 0.73 / 8.39 / 0.1% | 1.91 / 0.78 / 8.01 / 1.6% |
| Language preferences | 2.06 / 0.79 / 8.97 / 0.1% | 2.06 / 0.87 / 8.63 / 1.7% |
| Top Picks | 4.80 / 4.69 / 5.94 / 0.3% | 10.94 / 4.69 / 74.03 / 9.0% |
| Hidden Gems | 0.75 / 0.76 / 0.98 / <0.1% | 0.76 / 0.76 / 0.93 / 0.6% |
| Brazilian Cinema | 2.48 / 2.42 / 2.96 / 0.2% | 2.37 / 2.34 / 2.97 / 2.0% |
| Because You Liked | 25.58 / 24.27 / 69.31 / 1.6% | 12.03 / 14.46 / 22.40 / 9.9% |
| Directors You Love | 1.75 / 2.20 / 2.55 / 0.1% | 1.72 / 2.16 / 2.49 / 1.4% |
| Favorite Genre | 2.08 / 2.02 / 6.79 / 0.1% | 2.12 / 1.97 / 6.55 / 1.7% |
| Favorite Decade | 3.61 / 3.86 / 5.76 / 0.2% | 3.72 / 3.75 / 5.74 / 3.1% |
| World Cinema | 4.34 / 4.03 / 5.84 / 0.3% | 4.33 / 4.08 / 5.53 / 3.6% |
| Outside Usual | 4.91 / 5.58 / 6.40 / 0.3% | 4.89 / 5.51 / 6.14 / 4.0% |
| Classic Cinema | 0.97 / 0.84 / 1.54 / 0.1% | 0.94 / 0.82 / 1.48 / 0.8% |

The optimized Top Picks p95 contains one scheduler outlier; its median and
cProfile cost stayed at the baseline level, and no code in that proposal changed.

The SVD baseline/optimized difference is run-to-run timing variance rather than
an algorithm change. Candidate depths and exact retrieval are unchanged.

## Finding and optimization evidence

The baseline request for user 3953 issued 14 SQL queries in 194.57 ms: one
history query, then the display film query and eager relationship queries. The
eager graph loaded 53,290 log rows, 5,312 actor rows, and four user batches in
addition to metadata that the result did not need. There was no per-candidate
N+1, but the broad `Film` relationship defaults made the one repository call
materialize a large unrelated object graph. Baseline cProfile accordingly spent
17.05 seconds cumulatively in SQLAlchemy relationship loading for the nine-user
CPU pass. Its largest self-time entries were ORM instance construction (3.59 s,
597,487 calls), state initialization (1.82 s, 583,388 calls), row equality
(1.00 s, 1,653,909 calls), and instrumentation/new-instance work (0.98 s,
583,388 calls).

The service now materializes title, year, and director names from the immutable
`PolicyCatalog` already loaded for the same request. It also creates the policy
engine and popularity-rank mapping once when resources load. A request now
issues exactly one SQL query: user 3953 returned 662 history rows in 1.11 ms.
Startup remains five bounded catalog queries (baseline observed SQL time
259.52 ms), not request work.

Anchor selection first sorts by descending user rating. Lower-rated eligible
anchors therefore cannot win. Only maximum-rating anchors proceed through
neighborhood ranking. Matrix batches retain their legacy shape, with padding
where required, because changing a float32 BLAS matrix shape caused a deterministic
diagnostic difference of about 6e-8 in one deep-history fingerprint. This keeps
all exact outputs stable while reducing computed anchor rows from 489 to 256
for user 2474, 475 to 256 for 2994, and 819 to 256 for 3724. Competitive maximum
rating anchors for those users were 254, 131, and 179 respectively.

Profile construction remains a family-by-family traversal. Its mean was only
17.47 ms at baseline, and changing aggregation order risks exact floating-point
preference output, so the measurements did not justify a single-pass rewrite.

## History-size variation

| User | Watched/rated | Eligible anchors | Maximum-rating anchors | Candidates | Categories | Baseline mean ms | Optimized mean ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3318 | 10 | 1 | 1 | 3,648 | 6 | 1,025.88 | 53.39 |
| 3569 | 31 | 24 | 17 | 3,696 | 9 | 2,348.43 | 56.79 |
| 3155 | 102 | 55 | 16 | 3,312 | 10 | 2,951.61 | 91.81 |
| 2825 | 501 | 220 | 149 | 3,897 | 10 | 1,283.49 | 123.47 |
| 3953 | 662 | 465 | 164 | 3,769 | 10 | 1,440.49 | 121.47 |
| 2504 | 1,000 | 529 | 178 | 3,399 | 9 | 2,429.38 | 100.17 |
| 2474 | 3,001 | 1,082 | 254 | 3,983 | 9 | 1,110.06 | 154.98 |
| 2994 | 4,994 | 1,398 | 131 | 3,885 | 8 | 595.70 | 139.97 |
| 3724 | 9,892 | 2,326 | 179 | 3,701 | 8 | 718.66 | 199.98 |

## Memory

| Memory measurement | Baseline | Optimized |
| --- | ---: | ---: |
| RSS after artifact + catalog load/GC | 350.30 MiB | 348.77 MiB |
| Temporary Python allocation, mean | 100.87 MiB | 7.23 MiB |
| Temporary Python allocation, median | 80.97 MiB | 7.59 MiB |
| Temporary Python allocation, p95 | 189.60 MiB | 10.08 MiB |
| Temporary Python allocation, max | 202.40 MiB | 10.33 MiB |
| RSS at 0 / 25 / 50 / 75 / 100 requests | 625.91 / 626.91 / 627.11 / 626.11 / 625.91 MiB | 399.63 / 399.63 / 399.63 / 399.63 / 399.63 MiB |

The old display ORM graph, not candidate-policy objects, caused the large
temporary peak and process allocator high-water mark. Both 100-request runs
ended at their starting RSS after GC; this is a bounded plateau, with no evidence
of a retained-object leak. After optimization, the largest retained snapshot
locations are small profile/ranking collections (roughly 0.1–0.3 MiB each).

The 46,990-film catalog adds about 65.9–67.8 MiB RSS. Measured shallow/shared
components include 3.94 MiB of `PolicyFilm` objects, 2.61 MiB of title strings,
2.50 MiB for the film dictionary, 2.00 MiB for the artifact-ID set, 2.09 MiB for
21,084 interned entities and names, and relation tuples of 2.54 MiB directors,
2.96 MiB genres, 2.63 MiB countries, and 2.61 MiB languages. The remainder is
dictionary entries, integer/referent storage, allocator overhead, and load-time
temporaries. No catalog redesign is warranted by the serving result.

## Concurrency smoke test

This is a service-level `asyncio.gather` smoke test, not throughput evidence.
The workload is CPU-heavy and mostly synchronous, so requests do not scale
linearly within one event loop.

| Concurrent requests | Baseline batch wall | Optimized batch wall | Optimized RSS after GC |
| ---: | ---: | ---: | ---: |
| 1 | 1,045.90 ms | 50.31 ms | 399.63 MiB |
| 2 | 3,330.32 ms | 124.15 ms | 399.63 MiB |
| 4 | 7,992.71 ms | 338.33 ms | 399.63 MiB |

Shared catalog, engine, popularity mapping, and model artifacts remained safe
because they are immutable during requests.

## Exact semantics and serving recommendation

Canonical fingerprints include every category, item, reason and safe evidence,
source/stratum, selected preference/anchor, and deterministic diagnostics. All
nine baseline fingerprints equal their optimized counterpart byte-for-byte,
covering shallow, established, and deep histories. Synthetic tests additionally
compare pruned and exhaustive anchor evaluation across mixed 5.0/4.5 ratings,
only 4.5 ratings, 4.0 fallback, equal-rating ties, randomized neighborhood
quality, and Top-Picks overlap.

An initial loaded-worker budget of median <=250 ms, p95 <=500 ms, ordinary max
<=750 ms, and steady RSS <=450 MiB is realistic on this host and leaves useful
headroom over the observed 122.57 / 192.22 / 255.23 ms and 399.63 MiB. The
internal categorized service is ready for versioned API design. Synchronous
serving is sufficient initially; a future hybrid can cache or precompute after
profile sync only when lifecycle/invalidation requirements and production load
measurements justify it. No caching, precompute, endpoint, public schema, or
persistent infrastructure is introduced here.

The remaining measured costs are exact FAISS retrieval, deep-history preference
construction, proposal loops, and allocation. Their absolute cost is small
enough that further synchronous optimization is not currently worth the semantic
risk. Before public API/cache design, production-like multi-process capacity and
database-pool/load tests remain operational follow-up, not correctness blockers.
