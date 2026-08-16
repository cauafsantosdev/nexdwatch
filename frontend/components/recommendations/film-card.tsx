"use client";

import { ExternalLink, Film } from "lucide-react";
import { useState } from "react";

import { BrandMark } from "@/components/ui/brand-mark";
import { Skeleton } from "@/components/ui/skeleton";
import type { RecommendationFilm } from "@/types/recommendations";

export function FilmCard({ film }: { film: RecommendationFilm }) {
  const [posterFailed, setPosterFailed] = useState(false);
  const [posterLoaded, setPosterLoaded] = useState(false);
  const hasPoster =
    typeof film.tmdb_id === "number" &&
    Number.isInteger(film.tmdb_id) &&
    film.tmdb_id > 0 &&
    !posterFailed;
  const directors = film.directors.slice(0, 2).join(", ");
  const letterboxdUrl = `https://letterboxd.com/film/${encodeURIComponent(film.slug)}/`;

  return (
    <article className="film-card">
      <a
        href={letterboxdUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="film-card-link"
        aria-label={`Open ${film.title} on Letterboxd`}
      >
        <div className="poster-frame">
          {hasPoster ? (
            <>
              {!posterLoaded && <Skeleton className="absolute inset-0" />}
              {/* The same-origin BFF redirects to the final poster asset. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`/api/posters/${film.tmdb_id}`}
                alt={`${film.title}${film.year ? ` (${film.year})` : ""} poster`}
                loading="lazy"
                decoding="async"
                className={posterLoaded ? "poster-image poster-image-loaded" : "poster-image"}
                onLoad={() => setPosterLoaded(true)}
                onError={() => setPosterFailed(true)}
              />
            </>
          ) : (
            <div className="poster-fallback">
              <div className="poster-fallback-mark" aria-hidden="true">
                <Film />
                <BrandMark className="poster-brand-mark" />
              </div>
              <p>{film.title}</p>
              {film.year && <span>{film.year}</span>}
            </div>
          )}
          <span className="external-chip" aria-hidden="true"><ExternalLink /></span>
        </div>
        <div className="film-card-copy">
          <div className="flex items-baseline gap-2">
            <h3>{film.title}</h3>
            {film.year && <span className="film-year">{film.year}</span>}
          </div>
          {directors && <p className="director-line">{directors}</p>}
        </div>
      </a>
    </article>
  );
}
