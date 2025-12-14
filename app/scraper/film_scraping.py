from scrapxd import Scrapxd


def scrape_film_queue(films: list):
    """Scrapes metadata of films on FilmQueue."""
    client = Scrapxd()

    films_data = []
    for slug in films:
        film = client.get_film(slug)

        total_logs = film.total_logs

        if total_logs >= 1000:
            
            films_data.append({
                "id": film.id,
                "slug": slug,
                "title": film.title,
                "original_title": film.original_title,
                "year": film.year,
                "runtime": film.runtime,
                "director": film.director,
                "genre": film.genre,
                "country": film.country,
                "language": film.language,
                "actors": film.actors,
                "studio": film.studio,
                "synopsis": film.synopsis,
                "tagline": film.tagline,
                "themes": film.themes,
                "avg_rating": film.avg_rating,
                "total_logs": film.total_logs
            })

    return films_data