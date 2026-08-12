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

## Weighted-RRF calibration

The follow-up `strict_out_of_user_lambdarank_full_pool_v2_rrf_calibration_v1`
pass kept the same folds, train-only SVD/popularity artifacts, full test pool,
miss semantics, and film-ID tie-breaking. It built no LightGBM feature matrices.
For each fold, all 28 configurations were evaluated on validation, one was
selected by global NDCG@20 (then global R@20, conditional NDCG@20, and default
nearness), and test was evaluated only after the selections were frozen.

### Complete validation grid

Values are validation-fold mean global NDCG@20 and population standard
deviation. `Near-best` counts folds within 0.001 absolute NDCG@20 of that fold's
winner. The mean-regret column is relative to the independently best
configuration on each fold.

| SVD/pop | k | Global NDCG@20 | Global R@20 | Conditional NDCG@20 | Mean regret | Near-best |
|---:|---:|---:|---:|---:|---:|---:|
| 50/50 | 60 | 0.06776 ± 0.01067 | 0.14006 | 0.08134 | 0.00282 | 5/15 |
| 50/50 | 20 | 0.06756 ± 0.01149 | 0.13785 | 0.08113 | 0.00303 | 4/15 |
| 50/50 | 100 | 0.06677 ± 0.01082 | 0.13751 | 0.08016 | 0.00382 | 3/15 |
| 60/40 | 20 | 0.06602 ± 0.01059 | 0.13972 | 0.07926 | 0.00457 | 3/15 |
| 50/50 | 200 | 0.06577 ± 0.01093 | 0.13581 | 0.07895 | 0.00481 | 4/15 |
| 60/40 | 100 | 0.06519 ± 0.01055 | 0.13649 | 0.07825 | 0.00540 | 1/15 |
| 60/40 | 60 | 0.06508 ± 0.01061 | 0.13734 | 0.07812 | 0.00551 | 0/15 |
| 60/40 | 200 | 0.06501 ± 0.01126 | 0.13446 | 0.07804 | 0.00558 | 3/15 |
| 70/30 | 200 | 0.06440 ± 0.01059 | 0.13378 | 0.07730 | 0.00618 | 2/15 |
| 70/30 | 20 | 0.06343 ± 0.00992 | 0.13107 | 0.07616 | 0.00715 | 1/15 |
| 70/30 | 100 | 0.06319 ± 0.01052 | 0.13293 | 0.07586 | 0.00740 | 0/15 |
| 40/60 | 200 | 0.06308 ± 0.01045 | 0.12836 | 0.07575 | 0.00751 | 0/15 |
| 80/20 | 200 | 0.06307 ± 0.01033 | 0.13073 | 0.07571 | 0.00752 | 0/15 |
| 80/20 | 100 | 0.06301 ± 0.01012 | 0.12887 | 0.07565 | 0.00757 | 0/15 |
| 80/20 | 20 | 0.06298 ± 0.01052 | 0.12768 | 0.07559 | 0.00760 | 2/15 |
| 70/30 | 60 | 0.06289 ± 0.01061 | 0.13022 | 0.07552 | 0.00769 | 0/15 |
| 80/20 | 60 | 0.06278 ± 0.01028 | 0.12768 | 0.07537 | 0.00781 | 0/15 |
| 40/60 | 100 | 0.06256 ± 0.01049 | 0.12548 | 0.07512 | 0.00802 | 0/15 |
| 40/60 | 60 | 0.06132 ± 0.01040 | 0.12141 | 0.07365 | 0.00926 | 0/15 |
| 40/60 | 20 | 0.06081 ± 0.01088 | 0.12548 | 0.07304 | 0.00977 | 0/15 |
| 30/70 | 200 | 0.06025 ± 0.00977 | 0.12056 | 0.07235 | 0.01033 | 0/15 |
| 30/70 | 100 | 0.05828 ± 0.01038 | 0.11395 | 0.07000 | 0.01230 | 0/15 |
| 30/70 | 60 | 0.05719 ± 0.01059 | 0.11090 | 0.06869 | 0.01339 | 0/15 |
| 20/80 | 200 | 0.05647 ± 0.00981 | 0.10970 | 0.06781 | 0.01411 | 0/15 |
| 20/80 | 100 | 0.05512 ± 0.01003 | 0.10547 | 0.06621 | 0.01546 | 0/15 |
| 30/70 | 20 | 0.05307 ± 0.00968 | 0.10683 | 0.06375 | 0.01751 | 0/15 |
| 20/80 | 60 | 0.05267 ± 0.00991 | 0.10326 | 0.06327 | 0.01792 | 0/15 |
| 20/80 | 20 | 0.04872 ± 0.00877 | 0.09953 | 0.05854 | 0.02186 | 0/15 |

The aggregate validation leader is exactly the current 50/50, `k=60` default.
Equal-weight `k=20`, `k=60`, and `k=100` form the declared 0.001 validation
plateau. Popularity-heavy configurations degrade monotonically enough to reject
a shift in that direction; modest SVD-heavy configurations sometimes win a
fold but do not improve the aggregate.

### Fold choices and test behavior

| Seed | Fold | Validation-selected SVD/pop, k | Validation NDCG@20 | Selected test NDCG@20 | Default test NDCG@20 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0 | 60/40, 200 | 0.07632 | 0.06809 | 0.06622 |
| 42 | 1 | 60/40, 20 | 0.05985 | 0.08013 | 0.07879 |
| 42 | 2 | 70/30, 20 | 0.04671 | 0.07690 | 0.07888 |
| 42 | 3 | 50/50, 100 | 0.07735 | 0.06001 | 0.05983 |
| 42 | 4 | 50/50, 200 | 0.07824 | 0.06875 | 0.07115 |
| 43 | 0 | 50/50, 60 | 0.08073 | 0.05866 | 0.05866 |
| 43 | 1 | 50/50, 60 | 0.07541 | 0.06100 | 0.06100 |
| 43 | 2 | 60/40, 20 | 0.09404 | 0.07247 | 0.08484 |
| 43 | 3 | 60/40, 20 | 0.06427 | 0.06645 | 0.06438 |
| 43 | 4 | 50/50, 20 | 0.06369 | 0.06132 | 0.05927 |
| 44 | 0 | 50/50, 100 | 0.06076 | 0.05410 | 0.05310 |
| 44 | 1 | 50/50, 200 | 0.07141 | 0.06005 | 0.05793 |
| 44 | 2 | 70/30, 200 | 0.06648 | 0.06305 | 0.06361 |
| 44 | 3 | 70/30, 200 | 0.06342 | 0.08284 | 0.08550 |
| 44 | 4 | 50/50, 20 | 0.08008 | 0.06960 | 0.07036 |

Eight configurations win at least one fold: 60/40 `k=20` wins three; 50/50 at
each of `k=20,60,100,200` and 70/30 `k=200` win two each; 60/40 `k=200` and
70/30 `k=20` win once each. This is fold-dependent selection, not evidence for
a precise weighted optimum.

| Ranker / view | R@10 | R@20 | R@50 | NDCG@10 | NDCG@20 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| Fold-selected RRF, conditional | 0.11503 | 0.16554 | 0.25612 | 0.06701 | 0.07976 | 0.05237 |
| Fold-selected RRF, global | 0.09648 | 0.13887 | 0.21483 | 0.05619 | 0.06690 | 0.04391 |
| Fixed/default RRF, conditional | 0.11581 | 0.16753 | 0.26318 | 0.06759 | 0.08057 | 0.05296 |
| Fixed/default RRF, global | 0.09715 | 0.14056 | 0.22077 | 0.05667 | 0.06757 | 0.04439 |
| Positive-weighted SVD, conditional | 0.10695 | 0.15571 | 0.23193 | 0.06095 | 0.07327 | 0.04706 |
| Positive-weighted SVD, global | 0.08971 | 0.13056 | 0.19449 | 0.05112 | 0.06144 | 0.03946 |
| Popularity, conditional | 0.06930 | 0.10564 | 0.17335 | 0.04097 | 0.05011 | 0.03240 |
| Popularity, global | 0.05815 | 0.08867 | 0.14548 | 0.03436 | 0.04203 | 0.02716 |
| Mean SVD, conditional | 0.05927 | 0.08837 | 0.14559 | 0.03307 | 0.04038 | 0.02509 |
| Mean SVD, global | 0.04968 | 0.07409 | 0.12207 | 0.02771 | 0.03385 | 0.02103 |

Fold-specific validation selection loses 0.00067 global NDCG@20 versus the
default (−1.00%) and remains 0.00546 above positive-weighted SVD (+8.88%). The
historical-user-clustered selected-minus-default delta is −0.00067 with 95% CI
**[−0.00273, 0.00137]**; selected-minus-SVD is +0.00546 with
**[0.00183, 0.00925]**. Thus weighting does not beat conventional RRF. The
validation-only fixed recommendation is the default itself, so its delta and CI
against conventional RRF are exactly zero.

### Fixed/default RRF segments

| Target stratum | Eligible | Candidate recall | Conditional NDCG@20 | Global NDCG@20 | Global R@20 |
|---|---:|---:|---:|---:|---:|
| HEAD | 4,422 | 0.96043 | 0.09246 | 0.08880 | 0.18408 |
| MID | 1,292 | 0.50387 | 0.00897 | 0.00452 | 0.01161 |
| TAIL | 184 | 0.27174 | 0.00000 | 0.00000 | 0.00000 |

| History depth | Eligible | Candidate recall | Conditional NDCG@20 | Global NDCG@20 | Global R@20 |
|---|---:|---:|---:|---:|---:|
| Bottom quartile | 1,443 | 0.88773 | 0.06227 | 0.05528 | 0.12751 |
| Middle 50% | 2,965 | 0.84823 | 0.08588 | 0.07284 | 0.15514 |
| Top quartile | 1,490 | 0.77315 | 0.08921 | 0.06897 | 0.12416 |

| Target source | Eligible | Candidate recall | Global NDCG@20 | Global R@20 |
|---|---:|---:|---:|---:|
| Both | 3,086 | 1.00000 | 0.12291 | 0.25113 |
| SVD only | 1,010 | 1.00000 | 0.00543 | 0.01881 |
| Popularity only | 852 | 1.00000 | 0.01613 | 0.04108 |
| Missed | 950 | 0.00000 | 0.00000 | 0.00000 |

The 172 KiB ignored report took 241.52 seconds and contains all metrics and
bootstrap results needed for this conclusion. No large artifacts were produced.

**RRF recommendation:** retain one fixed **50/50, `k=60`** configuration. The
aggregate validation result selects it directly, the nearby equal-weight `k`
values form a broad plateau, and unstable fold-specific tuning loses on test.
There is no evidence that weighted RRF or a more complex global ranker is worth
shipping. The offline global-ranking decision is finalized at conventional RRF;
production wiring and future category/policy work remain explicitly out of
scope.
