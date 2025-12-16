import json
import logging
import asyncio
import contextlib
import numpy as np
from sqlalchemy import select
from fastapi import FastAPI, HTTPException
from app.core.database import SessionLocal
from app.scraper import scrape_user_logs
from app.db.loaders import sync_user_logs
from app.models import User, Log, Film


# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Embeddings and indexes storage
ml_artifacts = {}

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Executed on API startup. Loads artifacts.
    """
    try:
        logger.info("Loading artifacts into memory...")
        
        # Loads film embeddings 
        ml_artifacts["item_vectors"] = np.load("data/item_embeddings.npy")
        
        # Loads index mapping
        with open("data/film_index.json", "r") as f:
            film_ids = json.load(f)
            ml_artifacts["film_index"] = film_ids
            ml_artifacts["id_to_pos"] = {id: i for i, id in enumerate(film_ids)}
            
        logger.info(f"Artifacts loaded! {len(film_ids)} films indexed.")
    except FileNotFoundError:
        logger.warning("Artifacts file not found in 'data/'. Run train notebook.")
        ml_artifacts["item_vectors"] = None
    except Exception as e:
        logger.error(f"Critical error loading artifacts: {e}")
        ml_artifacts["item_vectors"] = None
        
    yield
    # Shutdown: clear artifacts
    ml_artifacts.clear()
    logger.info("Artifacts unloaded from memory.")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"health": "check", 
            "model_status": "loaded" if ml_artifacts.get("item_vectors") is not None else "missing"
            }

@app.post("/users/{username}/sync-logs")
async def sync_logs(username: str):
    """
    Scrapes and syncs user logs into the database.
    """
    logging.info(f"Syncing logs for user: {username}")

    # Scrapes user logs
    user_logs = await asyncio.to_thread(scrape_user_logs, username)

    if not user_logs:
        raise HTTPException(status_code=404, detail="User not found or no logs available.")

    # Inserts scraped logs into DB
    await sync_user_logs(user_logs)

    # Gets user ID from username
    async with SessionLocal() as session:
        result = await session.execute(select(User).where(User.username == username))
        user = result.scalar_one_or_none()
        user_id = user.id if user else None

    return {"status": "ok", 
            "user_id": user_id,
            "logs_count": len(user_logs)
            }

@app.get("/users/{user_id}/recommendations")
async def recommendations(user_id: int):
    """
    Provides film recommendations for a given user based on SVD embeddings.
    """
    # Verifies if model is loaded
    if ml_artifacts.get("item_vectors") is None:
        raise HTTPException(status_code=503, detail="ML model not loaded.")
    
    logging.info(f"Generating recommendations for user ID: {user_id}")

    async with SessionLocal() as session:
        # Searches for user's watched films
        result = await session.execute(select(Log.film_id).where(Log.user_id == user_id))
        watched_film_ids = result.scalars().all()
        
        # If there aren't watched films, return cold start message
        if not watched_film_ids:
            return {
                "user_id": user_id, 
                "info": "No watched films found for this user. Cannot provide recommendations.", 
                "recommendations": []
            }

        # Gets artifacts
        item_matrix = ml_artifacts["item_vectors"]
        id_map = ml_artifacts["id_to_pos"]
        
        # Filters watched films to those present in the embeddings
        valid_indexes = [id_map[film_id] for film_id in watched_film_ids if film_id in id_map]
        
        if not valid_indexes:
             return {"user_id": user_id, 
                     "recommendations": []
                    }

        # User Vector Construction (Mean Pooling)
        user_vector = np.mean(item_matrix[valid_indexes], axis=0)

        # Score Calculation (Dot Product)
        scores = np.dot(item_matrix, user_vector)

        # Remove watched films from recommendations
        for idx in valid_indexes:
            scores[idx] = -1.0

        # Gets Top 10 recommendations
        top_indices = np.argsort(scores)[-10:][::-1]
        
        # Converts indexes back to film IDs
        top_ids = [ml_artifacts["film_index"][i] for i in top_indices]

        # Gets film details from DB
        films_query = await session.execute(select(Film).where(Film.id.in_(top_ids)))
        films_details = films_query.scalars().all()
        
        # Orders recommendations as per scores
        films_map = {f.id: f for f in films_details}
        ordered_recommendations = []
        
        for film_id in top_ids:
            if film_id in films_map:
                film = films_map[film_id]
                score_val = float(scores[id_map[film_id]]) 
                
                ordered_recommendations.append({
                    "id": film.id,
                    "title": film.title,
                    "director": film.directors[0].name if len(film.directors) > 0 else film.directors,
                    "year": film.year,
                    "match_score": round(score_val, 4) # Similarity score
                })

        return {
            "user_id": user_id,
            "strategy": "SVD_Mean_Pooling",
            "recommendations": ordered_recommendations
        }