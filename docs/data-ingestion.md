# Data ingestion

NexdWatch supports asynchronous username synchronization and synchronous official export import. Both paths create a `ScrapedProfile` and use the same transactional interaction reconciliation.

## Username synchronization

```text
POST /users/{username}/sync-logs
        -> Redis task state and username ownership
        -> profile_sync Celery queue
        -> profile worker
        -> Letterboxd adapter
        -> ScrapedProfile
        -> sync_user_logs
        -> PostgreSQL
```

New work returns `202 Accepted` with a task ID. Clients poll `GET /tasks/{task_id}` for `queued`, `processing`, `completed`, or `failed`. A successful task within the 15-minute freshness window can be reused with `200 OK`. `force=true` bypasses freshness but not active ownership.

Redis owns public task state, profile freshness, and the active task for each username. Task metadata is retained for 24 hours by default. Username ownership has a shorter crash-recovery TTL.

The API `TaskService` applies freshness and ownership rules, creates queued metadata atomically in Redis, and publishes Celery work off the API event loop.

### Worker ownership and retries

The API publishes only the username and task ID. The profile worker:

1. claims or refreshes ownership for that task;
2. records `processing` in Redis;
3. runs the blocking Letterboxd adapter through its async bridge;
4. retries classified transient failures with bounded exponential delay;
5. records a final task state;
6. releases the username lock only if it still owns it.

Late acknowledgement and terminal-redelivery checks handle broker redelivery. Persistence is idempotent for existing users and user/film interactions.

`app/scraper/user_scraping.py` wraps `letterboxdpy`. It classifies provider errors, validates the response, skips malformed entries, and preserves rated and unrated watches. Provider access completes before the database transaction starts. A profile with no valid watches fails synchronization.

## Shared reconciliation

`sync_user_logs` reconciles one complete `ScrapedProfile` in one transaction. Duplicate watches in a payload collapse by film identity.

### Known films

For a slug already in the catalog, NexdWatch inserts the user's `Log` or updates its rating. It also updates and marks processed any pending row for the same username/slug. The resolved interaction is immediately available to recommendation history.

### Unknown online films

For a canonical slug missing from the catalog:

1. one global `FilmQueue` row records the catalog backlog;
2. one username/slug `LogPending` row preserves the interaction and optional rating;
3. Sunday maintenance selects a bounded, oldest-first film batch;
4. each slug is scraped and processed independently;
5. successful metadata creates the film and normalized relationships;
6. dependent pending rows become resolved `Log` rows;
7. filtered or failed films move their pending rows to the same terminal status.

The catalog gate requires at least 1,000 Letterboxd ratings and a valid TMDB identity. A whole-batch failure leaves the selected rows pending for retry. A failure for one film does not roll back successful films from the batch.

`FilmQueue` is a PostgreSQL backlog shared by users. It is unrelated to the Celery `maintenance` queue.

## Official Letterboxd export fallback

`POST /users/{username}/import` accepts an original Letterboxd export ZIP. This fallback does not call Letterboxd.

The parser:

* requires `watched.csv` and accepts optional `ratings.csv`;
* validates UTF-8 headers, duplicate or conflicting rows, and half-star ratings from 0.5 through 5.0;
* rejects archives larger than 20 MiB, archives with more than 1,000 members, and archives with more than 50 MiB of uncompressed content;
* reads members in memory without extracting them;
* merges watched and rating rows by Letterboxd URI.

Rows are matched to the local catalog by normalized display title and year. Original title is used only when display title finds no candidate, and resolution requires exactly one film ID. Missing and ambiguous rows remain explicit in the response, with a bounded public sample.

Export short URLs do not provide trusted canonical slugs, so unresolved ZIP rows do not enter `FilmQueue`. A later import may resolve them after catalog maintenance. At least one film must resolve before persistence; otherwise the request writes no profile state.

## Sparse and unrated histories

Unrated watches remain part of watched exclusion but do not contribute to SVD or category preference evidence. Sparse rated histories can still produce a feed. Users without a positive SVD profile can receive popularity candidates, though personalized shelves may be unavailable.

Task, retry, batch, and lock settings are listed in [Operations](operations.md) and `.env.example`.
