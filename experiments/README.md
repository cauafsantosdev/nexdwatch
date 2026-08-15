# Experiment index

NexdWatch keeps completed model-selection and serving investigations separate from
production runtime code. Production never imports `experiments`; experiments may
reuse production/shared SVD, FAISS, policy, and repository primitives so comparisons
measure the actual application semantics.

| Experiment | Question | Outcome | Production status |
| --- | --- | --- | --- |
| [Neural retrieval / NCF](neural_retrieval/README.md) | Does an inductive two-tower retriever outperform simpler unseen-user baselines? | Lower ranking metrics and limited catalog coverage in the controlled cohort | Rejected |
| [Candidate retrieval](retrieval/candidate_analysis.py) | Which bounded SVD/popularity allocation provides useful recall and coverage? | 2,000 + 2,000 selected as a pragmatic serving cap | Selected |
| [LambdaRank](ranker/RESULTS.md) | Does a 115-feature LightGBM ranker beat the fixed RRF ordering? | Worse global full-pool NDCG@20 and unstable checkpoints | Rejected |
| [RRF calibration](ranker/RESULTS.md#weighted-rrf-calibration) | Do tuned weights or `k` improve generalization? | Equal-weight `k=60` remained the validation leader; fold tuning lost on test | Selected |
| [Category policy V1/V1.1](category_policy/README.md) | Can one global ranking become a coherent multi-row feed? | V1.1 improved portfolio balance and preserved deterministic semantics | Selected |
| [Serving optimization](category_policy/SERVING_PERFORMANCE.md) | Can V1.1 serve synchronously without semantic drift? | Exact fingerprints preserved; warm requests reduced to roughly 100–200 ms | Production |

## Reproducibility boundary

The controlled historical cohort uses 4,300,105 resolved ratings, 1,976 users,
46,990 catalog films, and deterministic seeds 42, 43, and 44. Source CSVs and
generated matrices/checkpoints are intentionally ignored and must be supplied or
reconstructed locally. Human-readable results remain tracked because negative
experiments are part of the engineering decision record.

Neural retrieval requires the optional dependencies in
`neural_retrieval/requirements.txt`. LambdaRank training requires
`../requirements-ranker.txt`. Neither dependency set is installed by the normal API
or worker image.

Historical reports retain the language and measurements appropriate to the phase in
which they were produced. Status banners explain later production changes without
rewriting those original conclusions.
