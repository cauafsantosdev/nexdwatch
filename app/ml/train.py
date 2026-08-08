import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sqlalchemy import create_engine

from app.core.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


def train_svd_model() -> None:
    """Train the current SVD model and write its established artifacts."""
    logger.info("Starting SVD training pipeline...")

    sync_db_url = f"postgresql+psycopg2://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    engine = create_engine(sync_db_url)

    logger.info("Reading logs from database...")
    query = "SELECT user_id, film_id, rating FROM logs WHERE rating IS NOT NULL"

    try:
        df = pd.read_sql(query, engine)
        df = df.drop_duplicates(subset=["user_id", "film_id"])
    except Exception:
        logger.exception("Unable to read SVD training interactions")
        return
    finally:
        engine.dispose()

    if df.empty:
        logger.warning("No logs found. Model can't be trained.")
        return

    logger.info("Data loaded: %d logs", len(df))

    logger.info("Creating User-Item Matrix...")
    matrix = df.pivot(index="user_id", columns="film_id", values="rating").fillna(0)
    film_ids = list(matrix.columns)

    logger.info("Training TruncatedSVD...")
    svd = TruncatedSVD(n_components=32, random_state=42)
    svd.fit(matrix)
    item_factors = svd.components_.T

    logger.info("Normalizing vectors...")
    item_factors_norm = normalize(item_factors, axis=1)

    logger.info("Saving artifacts...")
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    np.save(f"{output_dir}/item_embeddings.npy", item_factors_norm)

    with open(f"{output_dir}/film_index.json", "w", encoding="utf-8") as index_file:
        json.dump(film_ids, index_file)

    logger.info("Artifacts saved successfully!")
