# NexdWatch

NexdWatch is a production-oriented movie recommendation backend built around
Letterboxd profiles. It combines durable FastAPI/Celery ingestion, PostgreSQL,
32-dimensional TruncatedSVD, exact FAISS retrieval, deterministic reciprocal-rank
fusion, and a categorized recommendation feed. The project also preserves the
offline evidence used to reject neural retrieval and LambdaRank for this data and
operating envelope.

## Engineering highlights

- Evaluated on a controlled historical cohort of 4,300,105 resolved interactions
  across 46,990 films with strict out-of-user holdouts.
- Builds recommendations for users unseen during model training from their current
  rating history; no learned production user-ID embedding is required.
- Combines 2,000 positive-weighted SVD candidates with 2,000 controlled-popularity
  candidates, excludes watched films, and applies equal-weight RRF with `k=60`.
- Serves category policy V1.1 with structured product-safe reasons; warm categorized
  requests measured roughly 100–200 ms on the documented profiling host.
- Uses Redis/Celery for durable profile synchronization and scheduled maintenance.
- Retrains conditionally from PostgreSQL, validates immutable artifact bundles with
  SHA-256 checksums, promotes atomically, supports rollback, and activates selected
  models without operator restarts.

## Production architecture

```text
Letterboxd username ──> FastAPI ──> Redis task state ──> Celery profile worker
                              │                                │
Letterboxd export ZIP ────────┘                                v
                                                       PostgreSQL
                                                           │
                     FilmQueue + LogPending <──────── unknown films
                              │
                    scheduled maintenance
                              │
                              v
PostgreSQL snapshot ──> SVD + popularity ──> validated model bundle
                                                  │
                                      atomic current.json promotion
                                                  │
                                      graceful API self-recycle
                                                  v
                                          categorized feed
```

See [Architecture](docs/architecture.md) for component and ownership details.

## Recommendation pipeline

```text
persisted user history
        │
        ├── positive-weighted SVD ──> exact FAISS top 2,000
        └── controlled popularity ──> top 2,000
                                      │
                       deterministic union + watched exclusion
                                      │
                          equal-weight RRF, k=60
                                      │
                           category policy V1.1
                                      │
                         categorized recommendation feed
```

The SVD profile weights explicit ratings with
`max(rating - 3.0, 0)`. Controlled production popularity counts ratings `>= 3.5`
from the same PostgreSQL training snapshot, ordered by count descending and actual
`Film.id` ascending. Full formulas, category behavior, and reason semantics are in
[Recommendation system](docs/recommendation-system.md).

## Model-selection evidence

| Evaluated approach | Outcome | Production status |
| --- | --- | --- |
| Positive-weighted SVD | Strong, compact personalized retriever | Selected |
| Controlled popularity | Complementary catalog coverage | Selected |
| Equal-weight RRF, `k=60` | Stable validation leader; tuning did not improve test | Selected |
| [Inductive neural retrieval / NCF](experiments/neural_retrieval/) | Underperformed simpler baselines and had limited coverage | Rejected |
| LightGBM LambdaRank | Underperformed RRF on corrected full-pool evaluation | Rejected |
| Category policy V1.1 | Improved portfolio balance with exact serving semantics | Selected |

These are completed model-selection decisions, not unfinished integrations. The
protocols, limitations, measurements, and negative results remain under
[Experiments](experiments/README.md).

## Automated model lifecycle

Celery Beat evaluates retraining every Sunday. Training is eligible after 100 new
eligible users, 250 newly rated model films, or 180 model-age days. A complete
legacy-flat installation receives one explicit `LEGACY_MODEL_BOOTSTRAP` retrain.

Training builds in isolation while the API continues serving its loaded model. A
bundle becomes selectable only after SVD, FAISS, popularity, manifest, identity,
and checksum validation succeeds. Promotion atomically changes `current.json`; the
API watcher validates the change, sends its own process graceful `SIGTERM`, and
Compose restarts it through `restart: unless-stopped`. Startup failure restores the
previous known-valid version rather than entering an endless restart loop.

See [Model lifecycle](docs/model-lifecycle.md) and
[Operations](docs/operations.md).

## API surfaces

- `POST /users/{username}/sync-logs` submits durable username synchronization.
- `GET /tasks/{task_id}` returns queued, processing, completed, or failed state.
- `POST /users/{username}/import` synchronously imports an official Letterboxd
  export ZIP as the offline fallback.
- `GET /recommendations/{user_id}/feed` returns the categorized V1.1 feed.
- The legacy `SVD_Mean_Pooling` endpoint remains available at
  `GET /users/{user_id}/recommendations`.
- `GET /` reports model health and the version loaded by the current API lifespan.

The feed exposes up to ten possible categories. `outside_usual` is explicitly
marked experimental. Internal scores, raw policy diagnostics, and filesystem
artifact details are not part of the public response.

## Development quickstart

Requirements:

- Docker with the Compose plugin;
- Python 3.14.6 for an equivalent local environment;
- a configured `.env` based on `.env.example`;
- external source data or prebuilt artifacts, because large datasets and generated
  models are intentionally not distributed through Git.

Start infrastructure and apply migrations:

```bash
cp .env.example .env
docker compose up -d db redis
docker compose run --rm api alembic upgrade head
```

For the full local bootstrap, place compatible catalog and historical-interaction
CSVs at `data/films_data.csv` and `data/users_data.csv`, then run:

```bash
docker compose run --rm api python manage.py load-all
docker compose run --rm api python manage.py retrain --force
docker compose up -d api worker maintenance_worker beat
```

The repository does not provide the 4.3M-interaction research dataset. API startup
requires either a complete legacy-flat artifact set or a valid selected bundle
under `data/models/`. See [Development](docs/development.md) and
[`data/README.md`](data/README.md) before attempting a fresh-clone bootstrap.

## Repository map

```text
app/          production API, ingestion, recommendation, tasks, and lifecycle
experiments/  finalized offline protocols and model-selection evidence
tests/        production and optional research coverage
docs/         architecture, methodology, lifecycle, operations, and development
data/         ignored local inputs and generated artifacts; README is tracked
alembic/      immutable PostgreSQL migration history
manage.py     current Typer operations and research entrypoint
```

## Documentation

- [Architecture](docs/architecture.md)
- [Recommendation system](docs/recommendation-system.md)
- [Data ingestion](docs/data-ingestion.md)
- [Model lifecycle](docs/model-lifecycle.md)
- [Operations](docs/operations.md)
- [Development and reproducibility](docs/development.md)
- [Experiment index](experiments/README.md)

## Limitations

- Offline ranking metrics and qualitative review are not evidence of user
  satisfaction, retention, or business impact.
- Tail retrieval remains materially harder than head retrieval.
- The supported production process model is one Uvicorn process per API container.
- The historical datasets are large, local, and not licensed or distributed by this
  repository; full research reproduction requires supplying compatible source data.
- `outside_usual` remains an explicitly experimental category pending real product
  feedback.

## License

NexdWatch is available under the [MIT License](LICENSE).
