# Operations

## Services

The default Compose deployment expects:

- `api`: one normal Uvicorn process with automatic model-pointer watching;
- `worker`: one `profile_sync` Celery worker;
- `maintenance_worker`: one `maintenance` Celery worker;
- `beat`: the UTC schedule publisher;
- `redis`: broker, task state, freshness, and maintenance locks;
- `db`: PostgreSQL 17.

Run exactly one Beat instance. Redis locks protect maintenance work from duplicate
delivery, but multiple Beat publishers would create needless duplicate messages.
`pgadmin` is optional administration tooling, and `notebook` is behind the `dev`
profile.

## Scheduled maintenance

| UTC schedule | Task |
| --- | --- |
| Sunday 02:00 | Process at most `FILM_QUEUE_BATCH_SIZE` pending films |
| Sunday 04:00 | Evaluate retraining thresholds and enqueue training when eligible |
| January 15 03:00 | Refresh previous-year film aggregates |
| July 15 03:00 | Refresh previous- and current-year film aggregates |

Catalog refresh changes only `Film.avg_rating` and `Film.total_logs`. It does not
itself make a film recommendation-eligible or trigger retraining.

## Operational CLI

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

Dry runs inspect state without mutation. Maintenance and retraining commands acquire
the same Redis locks used by Celery so manual diagnostics cannot overlap scheduled
work.

After `retrain` or `rollback-model`, the CLI may report that API activation is
pending. No manual restart is required: the API watcher validates the pointer and
recycles the process automatically.

Initial data and legacy compatibility commands remain available:

```bash
python manage.py load-films
python manage.py load-logs
python manage.py load-all
python manage.py train
python manage.py build-index
```

`train` writes the established flat artifacts for development/backward
compatibility. `retrain` is the versioned production lifecycle command.

Research and diagnostic commands remain in `manage.py` for reproducibility; their
status and optional requirements are indexed in
[`experiments/README.md`](../experiments/README.md). A future cleanup may split CLI
implementation into internal command modules without changing this public command
surface.

## Failure boundaries

- A failed profile task records a safe public error and releases active ownership.
- A whole FilmQueue scrape failure leaves rows pending for retry.
- One film persistence failure does not roll back other completed films.
- Expiring Redis lock TTLs prevent crashed workers from holding maintenance forever.
- Failed training or validation leaves the selected model untouched.
- Malformed or invalid model pointer changes do not terminate a healthy API.
- Failed activation restores a previous valid selection rather than looping.

Health is available at `GET /` and reports `model_status` plus the version loaded by
the current API process. Logs record promoted, loaded, and transition versions
without exposing paths or checksums publicly.
