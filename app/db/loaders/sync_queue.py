import asyncio
import logging
from sqlalchemy import select, insert
from sqlalchemy.orm import Session as AsyncSession

from app.models import Film, User, Log, LogPending, FilmQueue
from app.models import Director, Actor, Genre, Language, Country, Studio, Theme
from app.models.film_queue import Status
from app.scraper import scrape_film_queue
from app.core.database import SessionLocal


logger = logging.getLogger(__file__)

async def sync_film_queue():
    """
    Processes pending films in the FilmQueue:
    - Scrapes metadata for each film slug.
    - Inserts new films and their relationships.
    - Updates FilmQueue status (PROCESSED or FAILED).
    - Moves matching LogPending entries into Log after the film is created.
    """
    async with SessionLocal() as session:
        async with session.begin():
            # Select all films on queue
            queue_result = await session.execute(select(FilmQueue.film_slug).where(FilmQueue.status != Status.PROCESSED))
            film_queue = queue_result.scalars.all()

            # Scrapes metadata for each film
            films_data = scrape_film_queue(film_queue)

            relation_map = [
                (Director, "director", "directors"),
                (Actor, "actors", "actors"),
                (Genre, "genre", "genres"),
                (Language, "language", "languages"),
                (Country, "country", "countries"),
                (Studio, "studio", "studios"),
                (Theme, "themes", "themes"),
            ]
            
            # Cache Dictionary: {Model: {Name: ORM Object}}
            caches = {}

            for model, _, _ in relation_map:
                # Selects all existent objects for this model in one query
                result = await session.execute(select(model))
                # Creates a cache for the model
                caches[model] = {obj.name: obj for obj in result.scalars().all()}

            # User Cache
            result_users = await session.execute(select(User.username, User.id))
            USER_CACHE = {username: id for username, id in result_users.all()}
            
            def get_or_create(model, name, session: AsyncSession):
                """Gets in cache or creates ORM object."""
                cache = caches[model]

                if name in cache:
                    obj = cache[name]
                    return obj 
                
                # If it's a new object, it's created and added to session and cache
                obj = model(name=name)
                session.add(obj)
                cache[name] = obj
                return obj
            
            for item in films_data:
                film_queue_item = await session.scalar(
                    select(FilmQueue).where(FilmQueue.film_slug == item["slug"])
                )
                
                try:
                    # If original_title_value is None, uses 'title' as fallback
                    original_title_value = item["original_title"] or item["title"]             

                    # Creates Film object
                    film = Film(
                        tmdb_id=int(item["tmdb_id"]),
                        slug=str(item["slug"]).strip(),
                        title=str(item["title"]).strip(),
                        original_title=str(original_title_value).strip(),
                        year=int(item["year"]),
                        runtime=int(item["runtime"]),
                        synopsis=str(item["synopsis"]).strip(),
                        tagline=str(item["tagline"]).strip(),
                        avg_rating=float(item["avg_rating"]),
                        total_logs=int(item["total_logs"])
                    )
                    session.add(film)

                    # Adds relationships using cache
                    for model, column, relation_list in relation_map:
                        items = item[column]
                        unique_names = set(items)

                        for name in unique_names:
                            obj = get_or_create(model, name, session)
                            getattr(film, relation_list).append(obj)

                    # Updates object Status
                    film_queue_item.status = Status.PROCESSED
                except Exception as e:
                    logger.error(f"Failed to process film {item['slug']}: {e}")
                    film_queue_item.status = Status.FAILED
                    continue

                # Flush session to assign film.id before inserting logs
                await session.flush()
                logger.info(f"Film {item['slug']} inserted.")

                # Select all pending logs for this film
                logs_result = await session.scalars(
                    select(LogPending).where(
                        (LogPending.status != Status.PROCESSED) &
                        (LogPending.film_slug == item["slug"])
                    )
                )
                pending_logs = logs_result.all()

                # Prepare logs for bulk insert into Log table
                logs_to_insert = []
                for log in pending_logs:
                    logs_to_insert.append({
                        "user_id": USER_CACHE.get(log.username),
                        "film_id": film.id,
                        "rating": log.rating,
                    })
                    log.status = Status.PROCESSED

                # Bulk insert
                if logs_to_insert:
                    await session.execute(insert(Log), logs_to_insert)
                    logger.info(f"{len(logs_to_insert)} logs of {item["slug"]} inserted.")
                else:
                    logger.info(f"No logs of {item["slug"]} available for insert.")


if __name__ == "__main__":
    asyncio.run(sync_film_queue())        