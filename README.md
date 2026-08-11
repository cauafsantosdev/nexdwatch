
# NexdWatch: The Recommendation Engine for Cinephiles

[![Python Version](https://img.shields.io/badge/Python-3.13-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/FastAPI-0.115-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Infrastructure](https://img.shields.io/badge/Docker-Container-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Database](https://img.shields.io/badge/Postgres-17-316192.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Model](https://img.shields.io/badge/Model-Scikit__Learn-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NexdWatch** is a recommendation engine designed to discover latent user tastes by analyzing **Letterboxd** profiles. Built with a focus on scalable architecture, it features an asynchronous ingestion pipeline, reproducible training workflows, and a high-performance inference API with low latency.

---

## System Architecture

The system follows a **decoupled architecture pattern**, designed to handle high-throughput data ingestion while maintaining low-latency inference. The architecture is divided into three distinct layers:

### 1. Data Ingestion & Storage Layer

* **Gateway:** A **FastAPI** service submits durable username synchronization tasks and serves synchronous imports and recommendations.
* **Ingestion:** Public profiles are synchronized by a Redis/Celery worker through `letterboxdpy`. Official Letterboxd export ZIPs remain a synchronous offline fallback.
* **Persistence:** All relational data (users, films, logs) is stored in **PostgreSQL 17**, ensuring data integrity via **SQLAlchemy** ORM.

### 2. Offline Training Pipeline

* **Execution:** Managed via a custom CLI (`manage.py`), ensuring reproducible runs within the Docker environment.
* **Processing:** The pipeline fetches historical data from Postgres using a high-performance sync driver (`psycopg2`), handles duplicate removal, and pivots the data into a sparse matrix.
* **Modeling:** **Scikit-learn** performs TruncatedSVD factorization to reduce dimensionality.
* **Artifacts:** Training writes normalized SVD item embeddings, their film-ID mapping, and an exact FAISS retrieval index to the shared data volume.
* **Candidate Research:** Offline analysis evaluates positive-weighted SVD plus controlled historical popularity for a future pre-ranker. Inductive neural retrieval remains isolated research under `experiments/neural_retrieval/`.

### 3. Online Inference Engine

* **Startup Strategy:** During the application's `lifespan` startup event, the API loads the lightweight model artifacts directly into **RAM**.
* **Real-Time Computation:** The live `SVD_Mean_Pooling` service mean-pools SVD vectors and uses exact FAISS `IndexFlatIP` retrieval.
* **Result:** This "In-Memory" approach eliminates disk I/O latency, delivering recommendations in **milliseconds**.

---

## Key Features

* **Letterboxd Integration:** Redis/Celery runs durable `letterboxdpy` username synchronization with per-user deduplication; official export ZIP ingestion resolves watch history and ratings synchronously and offline.
* **Collaborative Retrieval:** Serves the established TruncatedSVD mean-pooling baseline and maintains a separately evaluated broad candidate layer for a future ranker.
* **In-Memory Inference:** The inference engine serves the model entirely from RAM, eliminating disk I/O during requests to ensure real-time performance.
* **Reproducible Operations:** Includes a custom CLI (`manage.py`) inside Docker to standardize data loading, model retraining, and migrations.
* **History-Based Profiles:** Live SVD inference derives user vectors from imported ratings and does not require a learned product-user identity.

---

## Tech Stack

* **Core:** Python 3.14.6
* **Web Framework:** FastAPI + Uvicorn
* **Data Engineering:** PostgreSQL 17, AsyncPG, SQLAlchemy 2.0 (Async), Alembic
* **Machine Learning:** Scikit-learn (TruncatedSVD), FAISS, NumPy, Pandas
* **Infrastructure:** Docker, Docker Compose
* **Tooling:** Typer (CLI)

PyTorch is used only by the separate `experiments/neural_retrieval/` research
environment and is not installed in the standard API or worker image.

---

## 📂 Project Structure

```
nexdwatch/
├── app/
│   ├── core/           # Configuration and Database setup
│   ├── db/             # Loaders and CRUD operations
│   ├── ml/             # Training pipeline logic (Offline)
│   ├── models/         # SQLAlchemy ORM models
│   ├── scraper/        # Async scraper module
│   └── main.py         # FastAPI Inference Endpoints (Online)
├── data/               # Model artifacts (.npy) and raw CSVs
├── experiments/        # Isolated neural-retrieval research
├── alembic/            # Database migrations
├── docker-compose.yml  # Service orchestration
├── manage.py           # CLI entrypoint for Ops tasks
└── README.md
```

---

## ⚙️ How to Run Locally

Prerequisites: **Docker** and  **Docker Compose** .

### 1. Clone and Configure

```bash
git clone [https://github.com/cauafsantosdev/nexdwatch](https://github.com/your-username/nexdwatch)
cd nexdwatch

# Create the .env file
cp .env.example .env
```

### 2. Launch Environment

```bash
docker-compose up -d --build
```

### 3. Initialize System

Use the integrated CLI to prepare the database and train the model inside the container:

```bash
# 1. Apply Database Migrations
docker-compose exec api alembic upgrade head

# 2. Load Initial Datasets (Populate DB)
docker-compose exec api python manage.py load-all

# 3. Train the SVD Model (Generate Artifacts)
docker-compose exec api python manage.py train

# Optional: rebuild only retrieval.faiss from existing NumPy/JSON artifacts
docker-compose exec api python manage.py build-index

# 4. Restart API (To load the new model into memory)
docker-compose restart api
```

---

## API Usage & Demo

Interactive documentation is available at: **http://localhost:8000/docs**

### Workflow Example

**Step 1: Ingest Profile**
Submit a Letterboxd username synchronization task, then poll it until completion.

* `POST /users/{username}/sync-logs`
* *Optional:* `?force=true` bypasses only the recent-success freshness window
* *Returns:* `202 Accepted` with a task ID, or a recently completed task
* `GET /tasks/{task_id}`
* *Returns:* `queued`, `processing`, `completed`, or `failed`; completed tasks include `user_id` and `logs_count`

If live synchronization is unavailable, download your official Letterboxd export
and upload the ZIP as multipart form data under the `file` field:

* `POST /users/{username}/import`
* *Input:* the unmodified Letterboxd export ZIP
* *Returns:* the user ID plus watched, rated, imported, and unresolved counts

Export ingestion performs no Letterboxd or metadata-provider requests. It matches
`watched.csv` rows only to films already in the NexdWatch catalog by exact,
case-insensitive, whitespace-normalized title and year (including original title).
Unknown or ambiguous films are reported as unresolved and are not queued.
They can be resolved by a later import after the NexdWatch catalog is updated.

Username task metadata is retained in Redis for 24 hours, while successful
synchronizations are reused for 15 minutes by default. The worker uses
at-least-once delivery with idempotent profile persistence. Configure the broker,
task-state Redis database, freshness, retention, active-lock TTL, and task limits
through the `CELERY_BROKER_URL`, `TASK_STATE_REDIS_URL`, and `PROFILE_SYNC_*`
environment settings documented in `.env.example`.

**Step 2: Get Recommendations**
Use the returned ID to generate recommendations. The engine calculates the user vector on-the-fly.

* `GET /users/5/recommendations`
* *Returns:* A JSON list of movies ranked by collaborative similarity.

### Recommendation architecture

The current live endpoint remains unchanged:

```text
rated history
    ↓
SVD mean pooling
    ↓
exact FAISS retrieval
    ↓
top 10 (`SVD_Mean_Pooling`)
```

The finalized internal pre-ranker boundary is broader:

```text
catalog
    ↓
positive-weighted SVD + controlled historical popularity
    ↓
~3,591 deduplicated candidates on average
    ↓
future personalized ranker
    ↓
future category/policy layer
    ↓
up to ~10 categories × ~20 films
```

Candidate depth is intentionally much larger than the possible ~200 displayed
slots so later ranking and category policy retain useful alternatives. Candidate
generation is global and personalized retrieval only; it does not create
category-specific pools.

The ranker, feature construction, category policy, and public hybrid serving are
not implemented. `GET /users/{user_id}/recommendations` continues to use direct
SVD mean pooling and the strategy string `SVD_Mean_Pooling`.

### Candidate-generation evidence

The maintained `exact_holdout_v2` analysis uses the controlled
`data/users_data.csv` cohort, deterministic seeds 42/43/44, identical held-out
targets, leakage-free temporary SVD fitting, exact FAISS retrieval, and
training-only popularity counts over the same 46,990-film universe.

```bash
python manage.py analyze-candidates --seeds 42,43,44
python manage.py build-popularity
python manage.py preview-candidates 3953
```

The final bounded allocation sweep tested nominal budgets 500, 750, 1,000,
1,500, 2,000, 2,500, 3,000, 4,000, and 5,000. Every budget evaluated the full
two-source 80/20, 70/30, 60/40, 50/50, 40/60, 30/70, and 20/80 grid. Mean SVD
was not re-tested after its earlier rejection, and NCF was not included. Source
depths sum exactly to the nominal budget before deduplication.

| Nominal source budget | Mean unique candidates | Recall | Marginal gain |
| ---: | ---: | ---: | ---: |
| 500 | 458.07 | 0.485927 ± 0.004849 | — |
| 750 | 682.39 | 0.554086 ± 0.005193 | +0.068159 |
| 1,000 | 908.18 | 0.605459 ± 0.005275 | +0.051373 |
| 1,500 | 1,358.98 | 0.682435 ± 0.006571 | +0.076975 |
| 2,000 | 1,815.85 | 0.733130 ± 0.006438 | +0.050695 |
| 2,500 | 2,256.55 | 0.770770 ± 0.007064 | +0.037640 |
| 3,000 | 2,702.70 | 0.800780 ± 0.005577 | +0.030010 |
| 4,000 | 3,590.94 | 0.843337 ± 0.005399 | +0.042557 |
| 5,000 | 4,494.34 | 0.874873 ± 0.005187 | +0.031536 |

The ratio winners are interior points at every budget: 60/40 at 500, 2,000,
and 5,000; 50/50 at 750, 1,000, 1,500, 2,500, 3,000, and 4,000. Thus the ratio
boundary effect is resolved. The complete 63-configuration results remain in
`data/analysis/candidate_analysis.json`.

The 4,000→5,000 step still adds 3.15 absolute recall points, so this tested range
does not establish a mathematical plateau. The finalized 4,000-source setting is
an explicit pragmatic production cap: it retains 84.33% candidate recall and
roughly 18 candidate alternatives per possible displayed slot while avoiding
about 903 additional pre-ranker inputs per user:

```text
2,000 positive-weighted SVD
+ 2,000 controlled-popularity
→ deduplicate without refill
```

At 4,000, target recall was 0.957062 HEAD, 0.520588 MID, and 0.377423 TAIL.
Collective catalog coverage across all evaluated users was 99.9936%; this is not
per-user catalog coverage. Retrieved popularity percentile had mean 0.7780 and
median 0.9372.

A local steady-state benchmark loaded artifacts before timing and ran 20
repetitions for each of four persisted users with 210, 662, 1,004, and 14,946
watches. Each budget used its measured winning ratio:

| Nominal budget | Mean live unique candidates | p50 | p95 |
| ---: | ---: | ---: | ---: |
| 2,000 (60/40) | 1,790.75 | 24.77 ms | 124.44 ms |
| 3,000 (50/50) | 2,674.00 | 26.62 ms | 135.25 ms |
| 4,000 (50/50) | 3,571.00 | 27.97 ms | 133.61 ms |
| 5,000 (60/40) | 4,478.75 | 29.56 ms | 150.56 ms |

All 320 measured calls were deterministic; watched overlap was zero, including
at 5,000. Artifact loading was excluded from these timings.

The positive-weighted profile uses `weight = max(rating - 3.0, 0)` and divides
the weighted vector sum by the absolute-weight sum. It does not fall back to
mean SVD when no positive profile exists. Every source excludes all watched
films, including rated, unrated, liked, and disliked watches.

Controlled popularity is the count of resolved ratings `>= 3.5` in
`data/users_data.csv`, ordered by count descending and film ID ascending. It is
not `Film.total_logs`, a Letterboxd aggregate, an average rating, or mutable
product traffic. `build-popularity` atomically writes validated film IDs,
counts, ranks, threshold, schema, and source metadata to
`data/candidates/popularity.json`.

### Neural retrieval research

The evaluated inductive neural retriever is no longer a supported application
backend and FastAPI/Celery do not load its artifacts. Its Python implementation,
tests, CPU dependency file, reproduction commands, corrected multi-seed metrics,
and rejection rationale remain in
[`experiments/neural_retrieval`](experiments/neural_retrieval/README.md). Local
`data/ncf/` outputs remain ignored research artifacts. The normal application
image does not install PyTorch.

---

## Engineering Decisions

### 1. Why keep SVD (TruncatedSVD)?

SVD mean pooling is the only supported live recommendation service because it is
compact, fast to train, straightforward to operate, and remains the known-good
baseline. The evaluated inductive neural retriever is research-only and is not
selectable by the application.

### 2. Why In-Memory Loading?

To achieve real-time performance, the embedding matrix (`numpy.load`) is loaded into RAM during the application startup (`lifespan` event). This design eliminates network latency associated with external feature stores (like Redis) for this scale, ensuring the API responds in milliseconds.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

Cauã Santos – [My LinkedIn Profile](https://www.linkedin.com/in/cauafsantosdev/) – [cauafsantosdev@gmail.com](mailto:cauafsantosdev@gmail.com)

Project Link: [https://github.com/cauafsantosdev/nexdwatch](https://github.com/cauafsantosdev/nexdwatch)
