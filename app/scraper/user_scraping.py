from scrapxd import Scrapxd
from typing import List, Dict, Union


def scrape_user_logs(username: str) -> List[Dict[str, Union[str, float]]]:
    """Scrapes all logs of a specific user."""
    client = Scrapxd()
    user = client.get_user(username=username)
    logs = user.logs
    logs_list = [{"username": username, "slug": entry.film.slug, "rating": entry.rating} for entry in logs.entries
                 if entry.film and entry.rating is not None]

    return logs_list