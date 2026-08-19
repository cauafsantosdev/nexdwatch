# Development and reproducibility

## Repository and data boundary

Git contains source, migrations, tests, experiment code and reports, Docker and CI configuration, scripts, and documentation. Large source datasets, Letterboxd exports, trained models, model bundles, fold matrices, and checkpoints stay local. [`data/README.md`](../data/README.md) lists the tracked contents of `data/`.

A fresh clone can build images and run isolated tests. Starting the API requires a complete legacy artifact set or a selected versioned model bundle.

## Docker-first setup

The backend image uses Python 3.14.6. The frontend image uses Node 24 and Next.js 16.3.1. Docker Compose is the main local workflow.

```bash
cp .env.example .env
docker compose up -d db redis
docker compose run --rm api alembic upgrade head
```

Set `SECRET_KEY`, `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD`. Compose uses `POSTGRES_HOST=db` and `POSTGRES_PORT=5432`. `TMDB_API_READ_TOKEN` is optional; without it, the frontend uses poster placeholders.

### Full bootstrap

The bootstrap loaders expect external `data/films_data.csv` and `data/users_data.csv` files. Verify their provenance and permitted use before loading them.

```bash
docker compose run --rm api python manage.py load-all
docker compose run --rm api python manage.py retrain --force
docker compose up -d --build api frontend worker maintenance_worker beat
```

`retrain --force` creates and selects the first production bundle. The legacy `train` command writes flat compatibility artifacts. Open `http://localhost:3000` for the product; the BFF reaches FastAPI at `http://api:8000` inside Compose.

Most tests, static checks, frontend builds, and Docker image builds need neither the historical datasets nor live providers. Full catalog loading, production retraining, the real API lifespan, and historical metric reproduction need external data or artifacts.

## Backend validation

Run the CI checks in the API image:

```bash
docker compose run --rm --no-deps api ruff check .
docker compose run --rm --no-deps api python -m compileall -q app experiments manage.py
docker compose run --rm --no-deps api python -m pytest
```

Tests use temporary resources, fakes, and dependency injection for PostgreSQL, Redis, Celery publication, external scraping, and model files.

## Frontend development and validation

Start the frontend and its product dependencies from the repository root:

```bash
docker compose up -d --build db redis api worker frontend
```

Source is bind-mounted for hot reload, while dependencies stay in the `frontend_node_modules` volume. Run the frontend checks with:

```bash
docker compose run --rm --no-deps frontend npm run lint
docker compose run --rm --no-deps frontend npm run typecheck
docker compose run --rm --no-deps frontend npm run build
```

Host-based frontend development is optional:

```bash
cd frontend
cp .env.example .env.local
npm ci
npm run dev
```

The tracked frontend environment template sets `NEXDWATCH_API_URL=http://localhost:8000` for a host-exposed API. `TMDB_API_READ_TOKEN` remains optional. Both values are server-only, and neither is required for production build validation.

## Continuous integration

`.github/workflows/ci.yml` runs for pushes and pull requests targeting `main`. The Python 3.14 job installs `requirements.txt`, runs `pip check`, Ruff, compilation, and pytest. The Node 24 job runs `npm ci`, ESLint, TypeScript checking, and the production build with lockfile-based npm caching. Normal CI does not call Letterboxd. The scheduled live canary is covered in [Operations](operations.md).

## Research dependencies and history

PyTorch and LightGBM are excluded from production requirements. Tests that need them use `pytest.importorskip`. Neural retrieval has its own `experiments/neural_retrieval/requirements.txt`; LambdaRank uses its separate ranker requirements. Reproduction commands and frozen protocols live with each experiment:

* [Inductive neural retrieval](../experiments/neural_retrieval/README.md)
* [Offline ranking research](../experiments/ranker/README.md)
* [Category policy research](../experiments/category_policy/README.md)

`notebooks/fetch_films.ipynb` and `notebooks/fetch_users.ipynb` preserve the retired `scrapxd` collection workflow for provenance. Current synchronization uses `letterboxdpy`; official export ZIPs provide the offline fallback.

## Artifact hygiene

Do not commit source datasets, official export files, model bundles, fold matrices, or checkpoints. Preserve the experiment protocol, compact manifest, and conclusion before archiving or deleting generated research output.
