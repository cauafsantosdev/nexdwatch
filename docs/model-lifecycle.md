# Model lifecycle

NexdWatch trains, validates, selects, and activates recommendation artifacts without changing resources already serving requests.

## Production snapshot and retraining policy

Training reads explicit PostgreSQL ratings, deduplicates `(user_id, film_id)`, and builds one in-memory snapshot. The snapshot produces:

* normalized 32-dimensional TruncatedSVD item vectors;
* the mapping to actual `Film.id` values;
* an exact FAISS `IndexIDMap2(IndexFlatIP)` index;
* popularity counts from ratings `>= 3.5`.

Production popularity therefore shares the SVD vocabulary and measurement time. It does not use mutable `Film.avg_rating` or `Film.total_logs` values. Historical research popularity comes from the frozen CSV cohort and cannot be packaged as a production source.

Celery Beat evaluates retraining each Sunday. A build is eligible after any of:

* 100 new eligible users;
* 250 newly rated films outside the selected model vocabulary;
* 180 days since selected-model training.

Catalog-only and unrated-only additions do not count as new model films. Operators can inspect the decision with `training-status`, run `retrain`, or bypass thresholds with `retrain --force`.

A complete legacy-flat installation lacks the manifest counters needed for this comparison. Its first evaluation reports `LEGACY_MODEL_BOOTSTRAP` and requests one baseline versioned retrain.

## Bundle structure

```text
data/models/
├── current.json
└── 20260812T181500Z-ab12cd34/
    ├── item_embeddings.npy
    ├── film_index.json
    ├── retrieval.faiss
    ├── popularity.json
    └── manifest.json
```

Training writes each candidate in an isolated `.building-*` directory. The manifest records model identity and times, SVD dimension, snapshot counts, training and popularity semantics, runtime library versions, and the SHA-256 checksum of every serving artifact.

After all files are written, the directory is renamed to its version and reopened for validation. A candidate cannot be selected before validation succeeds.

## Cross-artifact validation

`validate_model_bundle` checks:

1. the directory and manifest model identities match;
2. the manifest schema and compatibility constants are supported;
3. all payload checksums match;
4. vector dimensions and film counts match the manifest;
5. FAISS stores the ordered IDs from `film_index.json`;
6. popularity declares the production source and covers exactly the model universe.

This gate checks integrity and serving compatibility. Offline experiments own model and policy quality decisions.

## Promotion, retention, and rollback

`current.json` is the model selector. Promotion validates the full candidate and atomically replaces this pointer. The pointer records both the selected version and the previous selection used for activation recovery.

Retention keeps the current model plus `MODEL_RETENTION_PREVIOUS` valid older rollback candidates. Invalid and incomplete directories do not count as history.

Rollback selects the newest valid bundle trained strictly before the current model. Repeated rollback continues backward. Dry-run selection and validation finish before the pointer changes, so failure leaves the current selection intact.

## Activation

FastAPI startup resolves one model location and configures both recommendation services from it. The process then loads its SVD vectors, film mapping, FAISS index, popularity data, and policy catalog. These resources remain fixed for the process lifetime.

The pointer watcher reads `current.json` every 30 seconds by default. When it finds a new version, it validates the full target and rechecks the pointer identity. It then sends graceful `SIGTERM` to its own process. Uvicorn runs lifespan cleanup, and Compose's `restart: unless-stopped` policy starts a process that loads the selected generation. No Docker socket or in-process model hot swap is involved.

## Startup recovery

A malformed pointer or invalid target does not stop a healthy running process. If a selected target fails startup validation, the recorded lineage permits one restore of the previous valid selection. The first versioned transition may restore a complete legacy-flat layout. Versioned serving never combines flat and versioned files.

If neither the selected model nor its predecessor is valid, startup fails.

The supported Compose service runs one Uvicorn process per API container. Each process observes the pointer independently and activates a selection through its own restart.

See [Operations](operations.md) for lifecycle commands and recovery checks.
