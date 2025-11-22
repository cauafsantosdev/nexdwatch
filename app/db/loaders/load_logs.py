import typer
import pandas as pd
from sqlalchemy import select, insert

from app.models.film import Film
from app.models.user import User
from app.models.logs import Log
from app.core.database import SessionLocal


def safe_convert(value, dtype):
    """
    Converts Pandas values safely.
    """
    if pd.isna(value):
        return None
    
    try:
        if dtype == float:
            return float(value)
        if dtype == str:
            clean_val = str(value).strip()
            return clean_val if clean_val else None
    except (ValueError, TypeError):
        return None
    
    return str(value)

async def load_logs_data(csv_path):
    """Loads a CSV of user logs into the database"""
    async with SessionLocal() as session:
        df = pd.read_csv(csv_path, sep=";")
        total_logs = len(df)
        
        async with session.begin():    
            typer.echo("  > Loading existing data in cache...")

            # Film Cache
            result_films = await session.execute(select(Film.slug, Film.id))
            FILM_CACHE = {slug: id for slug, id in result_films.all()}
            
            # User Cache
            result_users = await session.execute(select(User.username, User.id))
            USER_CACHE = {username: id for username, id in result_users.all()}
            
            typer.echo(f"  > Cache of {len(USER_CACHE)} users and {len(FILM_CACHE)} films loaded.")
                       
            def get_film_id(slug):
                """Helper for getting film ID in cache"""
                clean_slug = safe_convert(slug, str)
                if not clean_slug:
                    return None
                return FILM_CACHE.get(clean_slug) # Returns ID or None
            
            logs_to_insert = []
            new_users_to_insert = []
            new_usernames = set()
                
            typer.echo(f"  > Starting collection of {total_logs} logs and new users...")

            for index, row in df.iterrows():
                username = safe_convert(row["username"], str)
                
                # Searches for username in cache
                user_id = USER_CACHE.get(username)
                
                # Creates User object if it's new and adds to new_usernames
                if user_id is None and username and username not in new_usernames:
                    new_user = User(username=username)
                    session.add(new_user)
                    new_users_to_insert.append(new_user)
                    new_usernames.add(username) 
                
                # Preparing log
                film_id = get_film_id(row["slug"])
                rating = safe_convert(row["rating"], float)

                if film_id is None or rating is None or not username:
                    continue
                
                # Colecting data for bulk insert 
                logs_to_insert.append({
                    "username": username, # username as temporary key
                    "film_id": film_id,
                    "rating": rating
                })
                
                if (index + 1) % 100000 == 0:
                    typer.echo(f"  > Processed logs: {index + 1}/{total_logs}")
            
            if new_users_to_insert:
                typer.echo(f"  > Creating {len(new_users_to_insert)} news users...")
                # Saves new users to DB with IDs
                await session.flush()

                # Updates ID caching
                for user in new_users_to_insert:
                    USER_CACHE[user.username] = user.id

            typer.echo(f"  > Preparing {len(logs_to_insert)} logs for bulk insert...")

            # Mapping all logs with the real IDs
            final_logs_to_insert = [
                {
                    "user_id": USER_CACHE.get(log["username"]),
                    "film_id": log["film_id"],
                    "rating": log["rating"],
                }
                for log in logs_to_insert if USER_CACHE.get(log["username"]) is not None
            ]
            
            # Inserts all logs at once
            if final_logs_to_insert:
                await session.execute(
                    insert(Log),
                    final_logs_to_insert
                )
                typer.echo(f"  > Insertion of {len(final_logs_to_insert)} logs finished.")
            else:
                typer.echo("  > No valid logs for insertion.")