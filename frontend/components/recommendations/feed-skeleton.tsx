import { Skeleton } from "@/components/ui/skeleton";
import { Wordmark } from "@/components/ui/wordmark";

export function FeedSkeleton() {
  return (
    <main className="recommendations-page" aria-busy="true" aria-label="Loading recommendations">
      <header className="feed-nav"><Wordmark compact /><Skeleton className="h-10 w-40" /></header>
      <div className="feed-intro">
        <div><Skeleton className="h-4 w-44" /><Skeleton className="mt-5 h-20 w-[min(520px,80vw)]" /></div>
      </div>
      <div className="feed-shelves">
        {[0, 1, 2].map((row) => (
          <section className="film-shelf" key={row}>
            <Skeleton className="mb-5 h-9 w-64" />
            <div className="shelf-track overflow-hidden">
              {[0, 1, 2, 3, 4, 5, 6].map((card) => (
                <div className="film-card" key={card}>
                  <Skeleton className="aspect-[2/3] w-full" />
                  <Skeleton className="mt-3 h-4 w-4/5" />
                  <Skeleton className="mt-2 h-3 w-1/2" />
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>
    </main>
  );
}
