"""Import externally supplied historical user ratings into PostgreSQL.

The administrative loader caches catalog and user identities, creates missing
users, and bulk-inserts only rows whose film slug and rating resolve. The complete
import runs in one transaction and is separate from live profile synchronization.
"""

import pandas as pd
import typer
from sqlalchemy import insert, select

from app.core.database import SessionLocal
from app.models.film import Film
from app.models.logs import Log
from app.models.user import User


def safe_convert(value, dtype):
    """Normalize one pandas cell without failing the surrounding bulk import.

    Args:
        value: Raw CSV cell, including pandas missing-value sentinels.
        dtype: Requested scalar type; this loader explicitly supports floats and
            trimmed, non-empty strings.

    Returns:
        The normalized value, ``None`` when missing or invalid, or a string
        fallback for an unsupported target type.
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
    """Persist resolvable historical ratings with batched identity lookup.

    Existing films and users are loaded into dictionaries before scanning the CSV.
    Missing users are flushed once to obtain primary keys; rows with unknown films,
    missing usernames, or invalid ratings are skipped. Valid rows are inserted in
    bulk inside the same all-or-nothing transaction.

    Args:
        csv_path: Semicolon-delimited historical interaction export.

    Returns:
        None: New users and valid ratings are committed directly to PostgreSQL.

    Raises:
        Exception: Propagates CSV, database, and integrity failures after rollback.
    """
    async with SessionLocal() as session:
        # Keep file parsing outside the transaction; the frame is the immutable
        # source for all identity resolution below.
        df = pd.read_csv(csv_path, sep=";")
        total_logs = len(df)
        
        async with session.begin():    
            typer.echo("  > Loading existing data in cache...")

            # Resolve all foreign-key identities up front to avoid per-row queries.
            result_films = await session.execute(select(Film.slug, Film.id))
            FILM_CACHE = {slug: id for slug, id in result_films.all()}
            
            result_users = await session.execute(select(User.username, User.id))
            USER_CACHE = {username: id for username, id in result_users.all()}
            
            typer.echo(f"  > Cache of {len(USER_CACHE)} users and {len(FILM_CACHE)} films loaded.")
                       
            def get_film_id(slug):
                """Return the cached film identity for a normalized source slug."""
                clean_slug = safe_convert(slug, str)
                if not clean_slug:
                    return None
                return FILM_CACHE.get(clean_slug) # Returns ID or None
            
            logs_to_insert = []
            new_users_to_insert = []
            new_usernames = set()
                
            typer.echo(f"  > Starting collection of {total_logs} logs and new users...")

            for index, row in df.iterrows():
                # Stage missing users separately because their generated IDs are
                # required before log payloads can be materialized.
                username = safe_convert(row["username"], str)
                
                # Searches for username in cache
                user_id = USER_CACHE.get(username)
                
                # Creates User object if it's new and adds to new_usernames
                if user_id is None and username and username not in new_usernames:
                    new_user = User(username=username)
                    session.add(new_user)
                    new_users_to_insert.append(new_user)
                    new_usernames.add(username) 
                
                film_id = get_film_id(row["slug"])
                rating = safe_convert(row["rating"], float)

                if film_id is None or rating is None or not username:
                    continue
                
                # Retain usernames temporarily until the single user flush assigns
                # every new primary key.
                logs_to_insert.append({
                    "username": username, # username as temporary key
                    "film_id": film_id,
                    "rating": rating
                })
                
                if (index + 1) % 100000 == 0:
                    typer.echo(f"  > Processed logs: {index + 1}/{total_logs}")
            
            if new_users_to_insert:
                typer.echo(f"  > Creating {len(new_users_to_insert)} news users...")
                await session.flush()

                # Extend the lookup with database-assigned identities before the
                # final bulk payload is constructed.
                for user in new_users_to_insert:
                    USER_CACHE[user.username] = user.id

            typer.echo(f"  > Preparing {len(logs_to_insert)} logs for bulk insert...")

            # Convert staged usernames to stable foreign keys and write valid logs
            # in one database round trip.
            final_logs_to_insert = [
                {
                    "user_id": USER_CACHE.get(log["username"]),
                    "film_id": log["film_id"],
                    "rating": log["rating"],
                }
                for log in logs_to_insert if USER_CACHE.get(log["username"]) is not None
            ]
            
            if final_logs_to_insert:
                await session.execute(
                    insert(Log),
                    final_logs_to_insert
                )
                typer.echo(f"  > Insertion of {len(final_logs_to_insert)} logs finished.")
            else:
                typer.echo("  > No valid logs for insertion.")
