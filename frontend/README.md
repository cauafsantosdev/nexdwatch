# NexdWatch frontend

The consumer-facing NexdWatch V1 is a Next.js App Router application built with strict TypeScript, React, Tailwind CSS, TanStack Query, Lucide icons, and Sonner notifications. It exposes two product routes: onboarding at `/` and the categorized film feed at `/recommendations`.

## Docker Compose development

The complete development stack runs without host Node.js or npm. From the repository root, start PostgreSQL, Redis, FastAPI, the `profile_sync` worker, and the frontend with:

```bash
docker compose up -d --build db redis api worker frontend
```

Open `http://localhost:3000`. The browser talks only to same-origin Next.js Route Handlers; inside the Compose network those handlers reach FastAPI at `http://api:8000` through the server-only `NEXDWATCH_API_URL` variable.

Frontend source is bind-mounted from `./frontend` for hot reload. Dependencies remain inside the `frontend_node_modules` named volume, so the host does not need to install or maintain `node_modules`.

`TMDB_API_READ_TOKEN` is optional. Add it to the repository-root `.env` file to enable poster lookups; Compose passes it only to the Next.js server. When absent, film cards retain their designed placeholders.

Run frontend checks inside the container:

```bash
docker compose run --rm --no-deps frontend npm run lint
docker compose run --rm --no-deps frontend npm run typecheck
docker compose run --rm --no-deps frontend npm run build
```

## Optional standalone development

Host-based development remains available when Node.js and npm are already installed. Copy `.env.example` to `.env.local`, install with `npm ci`, and run `npm run dev`. In that mode, `NEXDWATCH_API_URL=http://localhost:8000` reaches the host-exposed API.

## Environment

```dotenv
NEXDWATCH_API_URL=http://localhost:8000
TMDB_API_READ_TOKEN=
```

Both values are server-only. `NEXDWATCH_API_URL` is required for profile and recommendation data. `TMDB_API_READ_TOKEN` is optional: when absent, film cards retain their designed poster placeholders while the rest of the application continues to work.

## Product flow

The preferred onboarding path submits a Letterboxd username, polls the real durable task until `completed` or `failed`, stores the returned user reference in `localStorage`, and opens the categorized feed. An official Letterboxd export ZIP can be imported inline as the synchronous offline fallback. Recommendation category and film ordering come directly from the public backend feed without client-side reranking.

TMDB movie metadata is fetched and cached only by the server-side poster Route Handler, which redirects valid poster requests to the TMDB image CDN. Film cards link directly to the canonical Letterboxd slug returned by the backend.

The production Docker target builds Next.js standalone output and runs it with `node server.js`; the Compose service intentionally selects the `development` target for hot reload.
