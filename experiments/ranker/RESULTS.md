# Full-pool LambdaRank benchmark results

## Protocol

These results use `strict_out_of_user_lambdarank_full_pool_v2`: five strict
out-of-user folds for each of seeds 42, 43, and 44. Fold artifacts use training
users only. Ranker training retains up to eight held-out positives per user and
the deterministic 512-row hard-negative groups. Validation and test retain the
complete deduplicated 2,000 positive-weighted-SVD + 2,000 controlled-popularity
candidate inventory, exclude the alternate canonical positive, and never inject
missed targets. The frozen 115-feature schema, labels, candidate policy, and
LightGBM model parameters are otherwise unchanged.

Checkpoint selection uses exact full-pool global NDCG@20. Candidate misses are
zero; users without an eligible canonical target are excluded; score ties use
the same film-ID ordering as final metrics. All values below are mean ±
population standard deviation over 15 seed/fold evaluations unless stated
otherwise.

The historical `strict_out_of_user_lambdarank_v1` result used 512-row sampled
validation and test groups. It is preserved separately under
`notebooks/data/ranker` and is not directly comparable to these ranking metrics.

## Candidate inventory

- Test inventory before alternate-positive exclusion: 3,509.12 candidates/user
  on average (min 2,000; max 3,999).
- Actually ranked test inventory: 3,508.29 candidates/user on average (min
  2,000; max 3,998), totaling 20,797,129 scored rows across repeated folds.
- Candidate recall: 0.83893 ± 0.00976 by fold; pooled recall is 4,948 / 5,898 =
  0.83893.
- The candidate oracle therefore reaches global Recall/NDCG of 0.83893. The 950
  retrieval misses remain zero globally.

## Full-pool ranking metrics

| Ranker / view | R@10 | R@20 | R@50 | NDCG@10 | NDCG@20 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM, candidate-conditional | 0.10347 | 0.13840 | 0.18424 | 0.06041 | 0.06930 | 0.04737 |
| LightGBM, global | 0.08681 | 0.11614 | 0.15461 | 0.05067 | 0.05813 | 0.03972 |
| RRF, candidate-conditional | 0.11581 | 0.16753 | 0.26318 | 0.06759 | 0.08057 | 0.05296 |
| RRF, global | 0.09715 | 0.14056 | 0.22077 | 0.05667 | 0.06757 | 0.04439 |
| Positive-weighted SVD, candidate-conditional | 0.10695 | 0.15571 | 0.23193 | 0.06095 | 0.07327 | 0.04706 |
| Positive-weighted SVD, global | 0.08971 | 0.13056 | 0.19449 | 0.05112 | 0.06144 | 0.03946 |

For global NDCG@20, LightGBM is **0.00944 lower than RRF (−13.97%)** and
0.00331 lower than positive-weighted SVD (−5.39%). For global Recall@20 it is
0.02442 lower than RRF (−17.38%). The candidate-conditional conclusion is the
same: LightGBM NDCG@20 is 13.99% below RRF.

After aggregating each historical user's delta across repeated seed appearances
and bootstrapping 1,966 unique users, the LightGBM-minus-RRF global NDCG@20
delta is −0.00944 with a 95% interval of **[−0.01311, −0.00587]**. Against
positive-weighted SVD the delta is −0.00331 with **[−0.00667, 0.00032]**.

## Segment diagnosis

| Target stratum | Eligible | Candidate recall | Conditional NDCG@20 | Global NDCG@20 | Global R@20 |
|---|---:|---:|---:|---:|---:|
| HEAD | 4,422 | 0.96043 | 0.08059 | 0.07744 | 0.15448 |
| MID | 1,292 | 0.50387 | 0.00085 | 0.00040 | 0.00154 |
| TAIL | 184 | 0.27174 | 0.00000 | 0.00000 | 0.00000 |

The failure is both retrieval and ranking: MID/TAIL candidate recall is poor,
and the model almost never ranks a retrieved MID/TAIL target into the top 20.
By source, targets retrieved by both generators have conditional/global
NDCG@20 0.10927 and R@20 0.21674; SVD-only and popularity-only targets have
conditional NDCG@20 0.00303 and 0.00280 respectively. There are 950 misses.

History-depth segments further separate the effects:

| History depth | Eligible | Candidate recall | Conditional NDCG@20 | Global NDCG@20 | Global R@20 |
|---|---:|---:|---:|---:|---:|
| Bottom quartile | 1,443 | 0.88773 | 0.06223 | 0.05551 | 0.12478 |
| Middle 50% | 2,965 | 0.84823 | 0.07968 | 0.06757 | 0.13080 |
| Top quartile | 1,490 | 0.77315 | 0.05421 | 0.04180 | 0.07848 |

## Checkpoint stability and controls

| Seed | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|---:|
| 42 | 1 | 9 | 1 | 2 | 2 |
| 43 | 1 | 1 | 25 | 1 | 1 |
| 44 | 13 | 1 | 1 | 1 | 13 |

Nine of 15 full-pool folds select iteration 1, versus eight of 15 under sampled
validation. Ten checkpoints changed; the mean absolute change is 115.87
iterations (median 24), largely because several sampled checkpoints in the
146–396 range collapse to 1–13. Sampled validation distorted individual
checkpoint depths, but iteration-1 instability was **not** mostly a sampling
artifact: it persists and is slightly more frequent on the correct validation
pool. No hyperparameter sweep was started.

The requested gate for expensive full-pool controls was not met because the full
model underperforms both RRF and positive-weighted SVD. Therefore source-only,
source + global metadata, and shuffled-personalization were not rerun on full
pools. For historical context only, the sampled-v1 global NDCG@20 values were
0.10774 for full and 0.08374 for shuffled personalization; those sampled values
must not be interpreted as full-pool control evidence.

## Runtime, limitations, and conclusion

The complete corrected benchmark consumed 4,743.61 aggregate fold-seconds
(1.32 hours), including the pre-optimization correctness checkpoint. Generated
NPZ matrices, models, and JSON reports remain ignored research artifacts and
are not part of the commit.

Limitations include one canonical target per validation/test user appearance,
frozen untuned ranker hyperparameters, offline-only evaluation, repeated users
across seeds (handled in the primary clustered interval), and no production
latency or calibration evidence. Retrieval quality is itself a major ceiling,
especially for MID and TAIL, but the finalized candidate policy was intentionally
not tuned in this pass.

**Conclusion:** LightGBM does not provide a practically meaningful personalized
ranking improvement over reciprocal-rank fusion when every retrieved candidate
competes. It is significantly worse on global NDCG@20, retains severe
checkpoint instability, and is not ready for production serving. The corrected
offline experiment is suitable to commit as reproducible negative evidence.
