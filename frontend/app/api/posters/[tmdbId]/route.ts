import { NextResponse } from "next/server";

const TMDB_API_ROOT = "https://api.themoviedb.org/3/movie";
const TMDB_IMAGE_ROOT = "https://image.tmdb.org/t/p/w500";
const POSTER_REVALIDATE_SECONDS = 60 * 60 * 24 * 7;

interface TmdbMovieDetails {
  poster_path?: unknown;
}

export async function GET(
  _request: Request,
  context: { params: Promise<{ tmdbId: string }> },
): Promise<Response> {
  const token = process.env.TMDB_API_READ_TOKEN?.trim();
  if (!token) {
    return Response.json({ detail: "Poster unavailable." }, { status: 404 });
  }

  const { tmdbId } = await context.params;
  const parsedTmdbId = Number(tmdbId);
  if (!Number.isInteger(parsedTmdbId) || parsedTmdbId <= 0) {
    return Response.json({ detail: "Poster unavailable." }, { status: 404 });
  }

  try {
    const response = await fetch(`${TMDB_API_ROOT}/${parsedTmdbId}`, {
      headers: {
        Accept: "application/json",
        Authorization: `Bearer ${token}`,
      },
      next: { revalidate: POSTER_REVALIDATE_SECONDS },
    });
    if (!response.ok) {
      return Response.json({ detail: "Poster unavailable." }, { status: 404 });
    }
    const movie = (await response.json()) as TmdbMovieDetails;
    if (typeof movie.poster_path !== "string" || !movie.poster_path.startsWith("/")) {
      return Response.json({ detail: "Poster unavailable." }, { status: 404 });
    }

    const redirect = NextResponse.redirect(`${TMDB_IMAGE_ROOT}${movie.poster_path}`, 307);
    redirect.headers.set(
      "Cache-Control",
      `public, max-age=${POSTER_REVALIDATE_SECONDS}, stale-while-revalidate=86400`,
    );
    return redirect;
  } catch (error) {
    console.error("TMDB poster lookup failed", error);
    return Response.json({ detail: "Poster unavailable." }, { status: 404 });
  }
}
