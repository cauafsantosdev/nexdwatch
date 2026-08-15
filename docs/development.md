# Development and reproducibility

## What Git contains

Git contains application source, migrations, tests, experiment implementations,
human-readable experiment results, Docker configuration, and documentation.

Git does not contain the large historical datasets, local Letterboxd exports,
generated serving artifacts, versioned production bundles, trained NCF/LightGBM
artifacts, or fold matrices. `data/README.md` is the tracked inventory inside the
otherwise ignored `data/` directory.

Consequently, a fresh clone can build images and start PostgreSQL/Redis, but the API
cannot complete startup until it can resolve a complete legacy-flat artifact set or
a valid selected versioned bundle.

## Environment

The current container runtime is Python 3.14.6. Copy `.env.example` to `.env` and
provide PostgreSQL credentials plus a secret key. Redis service URLs and operational
defaults already match Compose service names.

```bash
cp .env.example .env
docker compose build
docker compose up -d db redis
docker compose run --rm api alembic upgrade head
```

## Full local bootstrap

The historical source files expected by the existing bootstrap commands are:

- `data/films_data.csv`: catalog and metadata relationships;
- `data/users_data.csv`: historical user/film/rating interactions.

These files are not distributed by Git. A developer must supply compatible data and
is responsible for its provenance and permitted use.

```bash
docker compose run --rm api python manage.py load-all
docker compose run --rm api python manage.py retrain --force
docker compose up -d api worker maintenance_worker beat
```

`retrain --force` builds the first versioned production baseline from PostgreSQL.
The legacy `train` command remains available when flat compatibility artifacts are
specifically required. Once a model is selected, scheduled retraining and activation
require no operator restart.

## What works without the full historical cohort

- Static imports, compilation, linting, and most unit tests use synthetic fixtures.
- API tests construct temporary model artifacts.
- ZIP parser and task-state tests do not require the historical CSV.
- Docker images can be built without the ignored local research outputs.

Starting the real API, loading the full catalog, running production retraining, or
reproducing offline metrics requires a database and/or external source data.

## Tests and optional research dependencies

The standard environment excludes PyTorch and LightGBM. Their tests use
`pytest.importorskip`, so the production-oriented suite remains runnable without
research dependencies.

```bash
python -m pytest
python -m compileall -q app experiments manage.py
```

Neural retrieval additionally requires
`experiments/neural_retrieval/requirements.txt`. Ranker feature training requires
`requirements-ranker.txt`. Reproduction commands and exact protocols live in each
experiment README.

## Historical notebooks

`notebooks/fetch_films.ipynb` and `notebooks/fetch_users.ipynb` use the removed
`scrapxd` workflow. They are retained only as dataset-provenance evidence and are not
current ingestion instructions. Production ingestion uses `letterboxdpy` through the
profile worker or the official export ZIP fallback.

## Local artifact hygiene

Do not commit source datasets, export files, model bundles, fold matrices, or model
checkpoints. Large reproducible outputs may be archived or deleted only after their
compact manifests and reported conclusions have been verified. No local data is
deleted by normal tests or maintenance commands.
