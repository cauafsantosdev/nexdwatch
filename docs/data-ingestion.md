# Data ingestion

NexdWatch supports durable username synchronization and synchronous official-export
ingestion. Both paths converge on the same interaction persistence rules.

## Username synchronization

```text
POST /users/{username}/sync-logs
        │
        ├── create/reuse Redis task metadata
        └── enqueue primitive username + task ID on profile_sync
                                      │
                              Celery profile worker
                                      │
                              Letterboxd profile scrape
                                      │
                                  PostgreSQL
```

The API returns `202 Accepted` with a task ID. Clients poll
`GET /tasks/{task_id}` for `queued`, `processing`, `completed`, or `failed` state.
Successful task metadata is retained for 24 hours by default and reused for 15
minutes unless `force=true` bypasses the freshness window.

Active ownership prevents concurrent scrapes of the same username. Celery uses
late acknowledgement and bounded retry for transient scrape failures. Persistence
is idempotent for existing users and `(user_id, film_id)` interactions, so broker
redelivery cannot create duplicate recommendation history.

## Known and unknown films

For a known film slug, synchronization creates or updates its `Log` immediately.

For an unknown slug:

1. one `FilmQueue` database row records the metadata backlog;
2. a `LogPending` row preserves the username, slug, and rating;
3. Sunday maintenance selects a bounded oldest-first batch;
4. successful metadata scraping creates the `Film` and relationships;
5. dependent pending rows become resolved `Log` rows;
6. filtered or failed films move their pending rows to the same terminal status.

A whole-batch scraper failure leaves selected rows pending so Celery can retry the
same work. An individual persistence failure is isolated to that film and does not
erase already completed results.

`FilmQueue` is a PostgreSQL backlog model. The Celery `maintenance` queue is a
separate execution channel; the shared word “queue” does not imply shared storage.

## Official Letterboxd export fallback

`POST /users/{username}/import` accepts an unmodified Letterboxd export ZIP. The
parser combines `watched.csv` and `ratings.csv`, validates bounded archive contents,
and resolves films by normalized title and year against the current catalog.

The fallback performs no Letterboxd or metadata-provider network request. Unknown or
ambiguous films are returned as unresolved and are not inserted into `FilmQueue`.
A later import can resolve them after catalog maintenance adds the corresponding
film.

## Sparse and empty users

Sparse histories remain valid recommendation inputs; category activation depends on
available evidence. A scraped profile with no valid watches fails synchronization
rather than creating misleading completion state. The categorized policy may still
produce popularity-supported rows for an existing user with an empty usable rating
profile.

Configuration for task retention, freshness, active locks, retries, and time limits
is listed in [Operations](operations.md) and `.env.example`.
