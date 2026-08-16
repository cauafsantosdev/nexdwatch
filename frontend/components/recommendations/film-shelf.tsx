"use client";

import { ArrowLeft, ArrowRight, FlaskConical } from "lucide-react";
import { useRef } from "react";

import { FilmCard } from "@/components/recommendations/film-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { RecommendationCategory } from "@/types/recommendations";

function preferenceContextCopy(category: RecommendationCategory): string | null {
  if (category.key !== "favorite_genre" && category.key !== "favorite_decade") {
    return null;
  }
  const context = category.preference_context;
  if (
    !context ||
    !Number.isFinite(context.average_rating) ||
    context.average_rating <= 0 ||
    context.average_rating > 5 ||
    !Number.isInteger(context.rated_count) ||
    context.rated_count <= 0
  ) {
    return null;
  }

  const family = category.key === "favorite_genre" ? "genre" : "decade";
  const entityName = category.items.find(
    (film) => film.reason.entity?.type === family,
  )?.reason.entity?.name;
  const average = context.average_rating.toFixed(1);
  const filmNoun = context.rated_count === 1 ? "film" : "films";

  if (category.key === "favorite_genre") {
    return entityName
      ? `You average ${average}★ across ${context.rated_count} ${entityName} ${filmNoun}`
      : `You average ${average}★ across ${context.rated_count} ${filmNoun} in this genre`;
  }
  return entityName
    ? `You average ${average}★ across ${context.rated_count} ${filmNoun} from the ${entityName}`
    : `You average ${average}★ across ${context.rated_count} ${filmNoun} from this decade`;
}

export function FilmShelf({ category, index }: { category: RecommendationCategory; index: number }) {
  const shelfRef = useRef<HTMLDivElement>(null);
  const preferenceCopy = preferenceContextCopy(category);

  function scroll(direction: -1 | 1) {
    const shelf = shelfRef.current;
    if (!shelf) return;
    shelf.scrollBy({ left: direction * shelf.clientWidth * 0.78, behavior: "smooth" });
  }

  return (
    <section className="film-shelf" aria-labelledby={`shelf-${index}`}>
      <div className="shelf-heading">
        <div>
          <p className="shelf-index">{String(index + 1).padStart(2, "0")} / CURATED ROW</p>
          <div className="mt-1 flex flex-wrap items-center gap-3">
            <h2 id={`shelf-${index}`}>{category.title}</h2>
            {category.experimental && (
              <Badge tone="blue"><FlaskConical size={12} aria-hidden="true" /> EXPERIMENTAL</Badge>
            )}
          </div>
          {preferenceCopy && <p className="shelf-preference-context">{preferenceCopy}</p>}
        </div>
        <div className="shelf-controls" aria-label={`Scroll ${category.title}`}>
          <Button variant="ghost" onClick={() => scroll(-1)} aria-label={`Scroll ${category.title} left`}>
            <ArrowLeft size={18} aria-hidden="true" />
          </Button>
          <Button variant="ghost" onClick={() => scroll(1)} aria-label={`Scroll ${category.title} right`}>
            <ArrowRight size={18} aria-hidden="true" />
          </Button>
        </div>
      </div>
      <div ref={shelfRef} className="shelf-track" tabIndex={0} aria-label={`${category.title} films`}>
        {category.items.map((film) => (
          <FilmCard key={film.film_id} film={film} />
        ))}
      </div>
    </section>
  );
}
