# Operations

## Compose runtime

The single-host deployment defines these services:

| Service | Responsibility |
| --- | --- |
| `frontend` | Next.js UI and same-origin BFF |
| `api` | One Uvicorn process, model resources, and pointer watcher |
| `worker` | Celery worker for `profile_sync` |
| `maintenance_worker` | Celery worker for `maintenance` |
| `beat` | UTC maintenance schedule publisher |
| `redis` | Broker, task state and freshness, and maintenance locks |
| `db` | PostgreSQL 17 |
| `pgadmin` | Optional database administration UI |
| `notebook` | Optional development-profile notebook environment |

Run exactly one Beat instance. The API uses `restart: unless-stopped` because a validated model-pointer change causes graceful process shutdown; Compose starts the process that loads the next generation.

## Scheduled maintenance

| UTC schedule | Task | Scope |
| --- | --- | --- |
| Sunday 02:00 | Process film queue | At most `FILM_QUEUE_BATCH_SIZE` pending films |
| Sunday 04:00 | Evaluate retraining | Build only when a lifecycle threshold is met |
| January 15 03:00 | Refresh recent catalog | Previous-year film aggregates |
| July 15 03:00 | Refresh recent catalog | Previous- and current-year film aggregates |

Catalog refresh updates `Film.avg_rating` and `Film.total_logs`; it does not change model vocabulary or retraining counters. Scheduled tasks and matching mutating CLI commands use the same Redis locks. Their TTLs exceed the corresponding hard task limits.

## Operational CLI

Run these commands through the API image in Docker-first environments:

```bash
docker compose run --rm api python manage.py training-status
```

### Model inspection, retraining, and rollback

```bash
python manage.py training-status
python manage.py retrain --dry-run
python manage.py retrain --force
python manage.py validate-model
python manage.py validate-model --model-version VERSION
python manage.py list-models
python manage.py current-model
python manage.py rollback-model --dry-run
python manage.py rollback-model
```

Status and dry-run commands do not build or promote a model. Validation checks checksums and cross-artifact identities. Retraining and rollback update the model pointer; the API watcher then starts the normal process transition.

### Catalog maintenance

```bash
python manage.py process-film-queue --dry-run
python manage.py process-film-queue --batch-size 100
python manage.py refresh-catalog --dry-run
python manage.py refresh-catalog --execution-date YYYY-MM-DD
```

Film-queue dry run selects the bounded pending batch. Catalog-refresh dry run applies the January or July selection policy without scraping or writing.

### Initial data and legacy artifacts

```bash
python manage.py load-films
python manage.py load-logs
python manage.py load-all
python manage.py train
python manage.py build-index
python manage.py build-popularity
```

`train` writes flat SVD compatibility artifacts. Production uses the versioned `retrain` lifecycle. `build-popularity` reads the frozen historical CSV for research or legacy use; production popularity is built from PostgreSQL by `retrain`.

Research and diagnostic commands are indexed in [`experiments/README.md`](../experiments/README.md).

## Letterboxd monitoring

Username synchronization depends on public-page scraping through the pinned `letterboxdpy` adapter. Two GitHub automations monitor that boundary:

* `.github/dependabot.yml` checks only `letterboxdpy` each Monday at 09:00 `America/Sao_Paulo`, opens at most one PR, and does not auto-merge;
* `.github/workflows/letterboxd-smoke.yml` runs Monday at 10:17 UTC and supports manual dispatch. It is separate from normal CI.

The smoke workflow exercises NexdWatch's profile and film adapters. It checks stable profile, watch, rating, title, slug, and positive TMDB identity invariants. The command timeout is five minutes and the job timeout is ten minutes.

Configure these GitHub Actions repository variables:

* `LETTERBOXD_SMOKE_USERNAME`: a stable public profile with a watched film;
* `LETTERBOXD_SMOKE_FILM_SLUG`: a stable public film above the 1,000-rating catalog gate with a TMDB identity.

Missing variables, provider outages, and upstream parser drift fail the canary.

## Failure and recovery

* Failed profile work records a final Redis task state and releases ownership only when the task still owns it.
* Whole-batch film scraping failures leave selected rows pending; one film failure does not roll back other successful films.
* Expiring maintenance locks release ownership after a crashed worker.
* Failed training, validation, or pointer replacement leaves the selected model unchanged.
* An invalid pointer change does not stop a healthy API process.
* Startup can restore one previous valid selection. It fails when neither selected model nor predecessor validates.

`GET /` reports `model_status` and the model generation loaded by the current API process. Logs identify built, promoted, loaded, and transition versions.

See [Model lifecycle](model-lifecycle.md) for activation details and [Data ingestion](data-ingestion.md) for task and reconciliation behavior.
