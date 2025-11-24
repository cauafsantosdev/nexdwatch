import logging
from sqlalchemy import select, insert

from app.models import Film, User, Log, LogPending, FilmQueue
from app.core.database import SessionLocal


logger = logging.getLogger(__file__)

async def sync_user_logs(user_logs: list):
    """
    Syncs user logs.
    - Logs of existing films are inserted into logs table.
    - Logs of non-existing films are inserted into FilmQueue and LogPending tables.
    """
    if not user_logs:
        return

    username = str(user_logs[0]["username"]).strip()
    if not username:
        return

    async with SessionLocal() as session:
        async with session.begin():
            # User Cache
            result_user = await session.execute(
                select(User).where(User.username == username)
            )
            user = result_user.scalar_one_or_none()

            if user is None:
                user = User(username=username)
                session.add(user)
                await session.flush()  # generates user.id

            user_id = user.id

            # Film Cache
            result_films = await session.execute(select(Film.slug, Film.id))
            FILM_CACHE = {slug: id for slug, id in result_films.all()}

            # User existing logs cache
            result_existing_logs = await session.execute(
                select(Log.film_id).where(Log.user_id == user_id)
            )
            EXISTING_LOGS = {film_id for (film_id,) in result_existing_logs.all()}

            # FilmQueue Cache
            result_filmqueue = await session.execute(select(FilmQueue.film_slug))
            FILMQUEUE_CACHE = {slug for (slug,) in result_filmqueue.all()}

            logs_to_insert = []
            pending_to_insert = []
            films_to_queue = []

            # Logs processing
            for log in user_logs:
                slug = str(log["slug"]).strip()
                rating = log.get("rating")

                if not slug or rating is None:
                    continue

                film_id = FILM_CACHE.get(slug)

                if film_id:
                    # Film exists on DB
                    if film_id in EXISTING_LOGS:
                        # Already logged on DB, skips
                        continue

                    logs_to_insert.append({
                        "user_id": user_id,
                        "film_id": film_id,
                        "rating": float(rating)
                    })

                else:
                    # Film doesn't exists on DB
                    if slug not in FILMQUEUE_CACHE:
                        # Adds Film to FilmQueue
                        films_to_queue.append({"film_slug": slug})
                        FILMQUEUE_CACHE.add(slug)
                    
                    # Adds Log to LogPending
                    pending_to_insert.append({
                        "username": username,
                        "film_slug": slug,
                        "rating": float(rating)
                    })

            # Bulk Inserts
            if films_to_queue:
                await session.execute(insert(FilmQueue), films_to_queue)

            if pending_to_insert:
                await session.execute(insert(LogPending), pending_to_insert)

            if logs_to_insert:
                await session.execute(insert(Log), logs_to_insert)

            logger.info(
                f"[{username}] Inserts: "
                f"{len(logs_to_insert)} logs, "
                f"{len(pending_to_insert)} pending, "
                f"{len(films_to_queue)} new films."
            )