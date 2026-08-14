# Backend maintenance lifecycle

NexdWatch keeps request serving and maintenance separate. Profile imports run on
the `profile_sync` Celery queue and persist known interactions immediately.
Unknown slugs remain represented by the existing `FilmQueue` plus dependent
`LogPending` rows. No separate log queue exists.

The `maintenance` queue defaults to one worker process. One UTC Celery Beat
instance publishes four schedules from `app/workers/schedules.py`:

- Sunday 02:00: process at most `FILM_QUEUE_BATCH_SIZE` pending films in
  `created_at, id` order;
- Sunday 04:00: evaluate retraining thresholds;
- January 15 03:00: refresh films from the previous release year;
- July 15 03:00: refresh films from the previous and current release years.

Production should run exactly one Beat instance. Redis locks with finite TTLs
also prevent concurrent film-queue, period-specific catalog, evaluation, and
retraining work after duplicate delivery. A transient whole-batch queue failure
leaves rows pending for a bounded Celery retry. Existing per-film success,
filtered, failed, relationship, and `LogPending` resolution semantics remain in
the queue processor. Catalog refreshes are sequential and update only valid
`Film.avg_rating` and `Film.total_logs` values. One malformed or failed film does
not erase existing values or stop other updates.

## Retraining policy

The measurement universe is the production SVD query: explicit non-null ratings,
deduplicated by `(user_id, film_id)`. An eligible user has at least one row in
that universe. A new model film is a distinct rated film ID in the current
universe that is absent from the promoted `film_index.json`; catalog-only and
unrated-only additions do not count.

Retraining is eligible when at least 100 new eligible users, 250 new model films,
or 180 days of model age is reached. These defaults are configurable. Aggregate
catalog refreshes do not trigger training. `training-status` returns the typed
decision's statistics, deltas, and reasons without mutation.

A complete legacy-flat installation has no trustworthy historical user or
interaction counts. Its first evaluation therefore returns
`LEGACY_MODEL_BOOTSTRAP` and requires one validated retrain. The flat model keeps
serving until that first versioned promotion succeeds; subsequent evaluations
use normal manifest deltas and age thresholds.

Research popularity remains generated from the controlled historical CSV.
Production retraining reads PostgreSQL rated interactions once, then derives
both frozen 32-dimensional normalized SVD factors and popularity (`rating >=
3.5`, positive count descending, `Film.id` ascending) from that same in-memory
snapshot.

## Bundles, promotion, and serving

Production artifacts are immutable bundles:

```text
data/models/
  current.json
  20260812T181500Z-ab12cd34/
    item_embeddings.npy
    film_index.json
    retrieval.faiss
    popularity.json
    manifest.json
```

A build uses a `.building-*` directory and does not touch `current.json`.
Vectors/mapping, exact FAISS structure and ordered IDs, production popularity,
manifest counts, identities, and SHA-256 checksums must all validate. The
completed directory is renamed to its final version and only then is a temporary
pointer file atomically replaced over `current.json`. Any failure before or
during promotion leaves the previous pointer authoritative. Retention runs after
promotion and keeps current plus two valid rollback candidates.

FastAPI resolves the pointer once per lifespan and binds both the legacy SVD
service and categorized feed to that same directory. If no pointer exists, the
complete legacy flat development layout is supported; once a pointer exists,
flat and versioned files are never mixed. New catalog films become
recommendation-eligible only after a retrain, promotion, and automatic process
recycle. Aggregate refreshes alone require no recycle.

Promotion deliberately does not mutate live NumPy/FAISS resources. Each API
lifespan checks the tiny pointer every
`MODEL_POINTER_CHECK_INTERVAL_SECONDS` (30 seconds by default). A changed pointer
is fully validated once, then the process sends itself SIGTERM so Uvicorn performs
normal lifespan cleanup. Compose `restart: unless-stopped` brings the API back and
the new lifespan loads the promoted bundle. No Docker socket or host-runtime
privilege is exposed. Training occurs while the API continues serving its old
in-memory bundle.

The supported production layout is one Uvicorn serving process per API container.
Multiple containers each run the same watcher and converge independently. If a
promoted bundle unexpectedly fails startup validation, activation lineage in
`current.json` is used to atomically restore the previous valid version (or the
still-complete legacy layout for the first bootstrap) once before resources load,
preventing a restart loop.

## Operations

```bash
python manage.py process-film-queue [--batch-size N] [--dry-run]
python manage.py refresh-catalog [--execution-date YYYY-MM-DD] [--dry-run]
python manage.py training-status
python manage.py retrain [--force] [--dry-run]
python manage.py validate-model [--model-version VERSION]
python manage.py list-models
python manage.py current-model
python manage.py rollback-model [--dry-run]
```

Rollback selects the newest valid bundle strictly older than the selected model,
validates it, and atomically switches the pointer. Repeated rollback therefore
moves backward and never bounces to a newer bundle. The watcher activates it
automatically. The application does not invoke Docker or systemd. NCF and
LightGBM remain research-only, and no recommendation cache or precomputed feed is
part of this lifecycle.
