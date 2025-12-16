
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

* **Gateway:** A **FastAPI** service acts as the entry point, handling asynchronous requests for user synchronization.
* **Scraping:** Background workers utilize `scrapxd` to fetch raw logs from Letterboxd without blocking the main thread.
* **Persistence:** All relational data (users, films, logs) is stored in **PostgreSQL 17**, ensuring data integrity via **SQLAlchemy** ORM.

### 2. Offline Training Pipeline

* **Execution:** Managed via a custom CLI (`manage.py`), ensuring reproducible runs within the Docker environment.
* **Processing:** The pipeline fetches historical data from Postgres using a high-performance sync driver (`psycopg2`), handles duplicate removal, and pivots the data into a sparse matrix.
* **Modeling:** **Scikit-learn** performs TruncatedSVD factorization to reduce dimensionality.
* **Artifacts:** The trained model is serialized not as a heavy pickle file, but as optimized NumPy arrays (`.npy`) and JSON indices, saved to a shared volume.

### 3. Online Inference Engine

* **Startup Strategy:** During the application's `lifespan` startup event, the API loads the lightweight model artifacts directly into **RAM**.
* **Real-Time Computation:** When a recommendation request arrives, the engine avoids database reads for feature extraction. Instead, it computes the user's vector profile on-the-fly using **NumPy** operations (Mean Pooling & Dot Product).
* **Result:** This "In-Memory" approach eliminates disk I/O latency, delivering recommendations in **milliseconds**.

---

## Key Features

* **Letterboxd Integration:** A robust, asynchronous scraping pipeline (`asyncio` + `scrapxd`) that extracts watch history and ratings from public profiles.
* **Collaborative Filtering:** Uses Matrix Factorization (TruncatedSVD) trained on over **4.3 million** interaction logs to map users and items into a dense vector space.
* **In-Memory Inference:** The inference engine serves the model entirely from RAM, eliminating disk I/O during requests to ensure real-time performance.
* **Reproducible Operations:** Includes a custom CLI (`manage.py`) inside Docker to standardize data loading, model retraining, and migrations.
* **Cold Start Mitigation:** Implements a Mean Pooling strategy to generate vector profiles for new users instantly based on their imported history.

---

## Tech Stack

* **Core:** Python 3.13
* **Web Framework:** FastAPI + Uvicorn
* **Data Engineering:** PostgreSQL 17, AsyncPG, SQLAlchemy 2.0 (Async), Alembic
* **Machine Learning:** Scikit-learn (TruncatedSVD), NumPy, Pandas
* **Infrastructure:** Docker, Docker Compose
* **Tooling:** Typer (CLI)

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

# 4. Restart API (To load the new model into memory)
docker-compose restart api
```

---

## API Usage & Demo

Interactive documentation is available at: **http://localhost:8000/docs**

### Workflow Example

**Step 1: Ingest Profile**
Send a Letterboxd username to the sync endpoint. The system will scrape logs and store them.

* `POST /users/{username}/sync-logs`
* *Input:* `cauafsantosdev`
* *Returns:* `user_id: 5`

**Step 2: Get Recommendations**
Use the returned ID to generate recommendations. The engine calculates the user vector on-the-fly.

* `GET /users/5/recommendations`
* *Returns:* A JSON list of movies ranked by collaborative similarity.

---

## Engineering Decisions

### 1. Why SVD (TruncatedSVD)?

While Deep Learning (NCF) is powerful, SVD was chosen for this MVP due to its efficiency/latency ratio. It handles sparse matrices (common in movie ratings) exceptionally well and allows for fast CPU training, removing the need for heavy GPU dependencies in the initial deployment.

### 2. Why In-Memory Loading?

To achieve real-time performance, the embedding matrix (`numpy.load`) is loaded into RAM during the application startup (`lifespan` event). This design eliminates network latency associated with external feature stores (like Redis) for this scale, ensuring the API responds in milliseconds.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

Cauã Santos – [My LinkedIn Profile](https://www.linkedin.com/in/cauafsantosdev/) – [cauafsantosdev@gmail.com](mailto:cauafsantosdev@gmail.com)

Project Link: [https://github.com/cauafsantosdev/nexdwatch](https://github.com/cauafsantosdev/nexdwatch)
