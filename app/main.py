import asyncio
from fastapi import FastAPI
from scraper import scrape_user_logs
from db.loaders import sync_user_logs


app = FastAPI()

@app.get("/")
async def root():
    return {"health": "check"}

@app.post("/users/{username}/sync-logs")
async def sync_logs(username: str):
    # Scrapes user logs
    user_logs = await asyncio.to_thread(scrape_user_logs, username)
    # Inserts scraped logs into DB
    await sync_user_logs(user_logs)

    return {"status": "ok"}

@app.get("/users/{user_id}/recommendations")
async def reccomendations(user_id: int):
    pass