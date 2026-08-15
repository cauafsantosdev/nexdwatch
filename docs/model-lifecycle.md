# Model lifecycle

NexdWatch trains and activates production recommendation artifacts without mutating
resources already serving requests.

## Production snapshot

Training reads current PostgreSQL rows with explicit ratings, deduplicates
`(user_id, film_id)`, and derives one in-memory snapshot. From that same snapshot it
builds:

- normalized 32-dimensional TruncatedSVD item vectors;
- actual film-ID mapping;
- exact FAISS `IndexIDMap2(IndexFlatIP)` retrieval;
- production popularity from ratings `>= 3.5`.

Historical research popularity is separately generated from the frozen
`data/users_data.csv` cohort. Aggregate catalog fields such as `avg_rating` and
`total_logs` are not production popularity inputs.

## Retraining policy

Retraining is eligible when any configured operational threshold is reached:

- 100 new eligible users;
- 250 newly rated films absent from the selected model vocabulary;
- 180 days since selected-model training.

Catalog-only or unrated-only additions do not count as new model films. A complete
legacy-flat installation has unknown historical user and interaction counts, so its
first evaluation reports `LEGACY_MODEL_BOOTSTRAP` and requires one baseline retrain.
No historical counts are invented.

## Immutable bundles

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

Construction starts in a `.building-*` directory. The manifest records model and
snapshot identity, training statistics, artifact metadata, and SHA-256 checksums.
Validation loads SVD and FAISS resources, checks dimensions and ordered IDs, verifies
popularity compatibility, and verifies every checksum before the directory can be
promoted.

A failed build or validation removes only the incomplete candidate. The existing
pointer and the model loaded by the API remain untouched.

## Promotion, retention, and rollback

Promotion atomically replaces `current.json` only after complete validation. The
pointer records the selected version and activation lineage. Retention keeps the
current version plus the configured number of valid older rollback candidates.

Rollback selects the newest fully valid bundle whose training time is strictly older
than the current selection. Repeated rollback therefore moves backward and never
bounces to a newer bundle. Pointer replacement remains atomic and failure leaves the
old pointer authoritative.

## Automatic activation

The API resolves exactly one model location during lifespan startup and loads both
the legacy recommendation service and categorized service from that location. A
background watcher remembers the loaded version and periodically reads only the tiny
pointer file. The default interval is 30 seconds.

When a different pointer is observed, the watcher fully validates the target and
sends the current API process `SIGTERM`. Uvicorn performs normal lifespan cleanup,
and Compose `restart: unless-stopped` starts a new process that loads the selected
bundle. No Docker socket or host-control command is exposed to the application.

Training therefore does not interrupt serving: the old process keeps its immutable
resources throughout construction and validation. Only a successful pointer
transition can request recycling.

Malformed pointers or invalid targets do not kill a healthy API. If a newly promoted
target unexpectedly fails startup validation, activation lineage restores the
previous known-valid bundle once before resources load. The first versioned bootstrap
may restore the still-complete legacy-flat layout; versioned serving never mixes flat
and versioned files.

The supported production model is one normal Uvicorn process per API container.
Multiple containers independently observe the pointer and converge.
