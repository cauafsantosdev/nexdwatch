# Offline LambdaRank benchmark

This package is an isolated research pipeline. It does not alter the production
candidate policy or FastAPI recommendation path. Each fold builds temporary SVD,
exact FAISS, controlled-popularity, and film-aggregate artifacts from ranker
training users only. Validation and test histories are used only as request-time
user profiles. LightGBM is deliberately absent from the application dependency
set and container image.

The current `strict_out_of_user_lambdarank_full_pool_v2` protocol uses two
different group constructions on purpose:

- training retains the deterministic hard-negative sampler capped at 512 rows;
- validation retains every deduplicated 2,000 SVD + 2,000 controlled-popularity
  candidate (except the alternate canonical positive), and early stopping uses
  NDCG@20 on those full groups;
- test uses that same full-candidate construction and ranks the designated target
  at its true position. Missed targets are never injected and remain zero in
  global metrics.

The historical `strict_out_of_user_lambdarank_v1` output under
`notebooks/data/ranker` used 512-row sampled validation and test groups. It is
preserved for comparison and must not be mixed with the full-pool results.

Create an isolated ranker environment with:

```bash
python -m venv /tmp/nexdwatch-ranker-venv
/tmp/nexdwatch-ranker-venv/bin/pip install -r requirements-ranker.txt
```

Smoke-test seed 42/fold 0 first:

```bash
/tmp/nexdwatch-ranker-venv/bin/python -m experiments.ranker \
  --csv-path notebooks/data/users_data.csv \
  --output-root notebooks/data/ranker_full_pool_v2 \
  --seeds 42 --folds 0
```

The command defaults to the `full` model only. After inspecting that result, run
all 15 models by omitting `--seeds` and `--folds`. If the lift remains material,
rerun only `source_only`, `source_plus_global_metadata`, and
`shuffled_personalization` from the persisted matrices. Outputs are numeric NPZ
datasets, JSON audit metadata, text LightGBM models, and aggregate reports under
the ignored research output root. Never commit those generated artifacts.

The completed full-pool result and the decision not to pass the expensive-control
gate are documented in [RESULTS.md](RESULTS.md).
