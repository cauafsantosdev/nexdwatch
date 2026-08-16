import { FilmShelf } from "@/components/recommendations/film-shelf";
import type { RecommendationFeed } from "@/types/recommendations";

export function Feed({ feed }: { feed: RecommendationFeed }) {
  if (feed.categories.length === 0) {
    return (
      <section className="feed-empty">
        <p className="eyebrow">NO ACTIVE SHELVES</p>
        <h2>We don’t have a confident set of picks yet.</h2>
        <p>Your history is loaded. Try again after the recommendation catalog is refreshed.</p>
      </section>
    );
  }

  return (
    <div className="feed-shelves">
      {feed.categories.map((category, index) => (
        <FilmShelf key={category.key} category={category} index={index} />
      ))}
    </div>
  );
}
