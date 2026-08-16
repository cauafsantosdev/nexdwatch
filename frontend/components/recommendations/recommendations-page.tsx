"use client";

import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, ArrowLeft, RefreshCw } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { Feed } from "@/components/recommendations/feed";
import { FeedHeader } from "@/components/recommendations/feed-header";
import { FeedSkeleton } from "@/components/recommendations/feed-skeleton";
import { SiteFooter } from "@/components/site-footer";
import { Button } from "@/components/ui/button";
import { ApiError, getRecommendationFeed } from "@/lib/api/client";
import { clearStoredProfile, loadStoredProfile } from "@/lib/storage";
import type { StoredProfile } from "@/types/profile";

const FEED_STALE_TIME_MS = 5 * 60 * 1000;

export function RecommendationsPage() {
  const router = useRouter();
  const [profile, setProfile] = useState<StoredProfile | null | undefined>(undefined);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const stored = loadStoredProfile();
      setProfile(stored);
      if (!stored) router.replace("/");
    }, 0);
    return () => window.clearTimeout(timer);
  }, [router]);

  const feedQuery = useQuery({
    queryKey: ["recommendation-feed", profile?.userId],
    queryFn: () => getRecommendationFeed(profile!.userId),
    enabled: Boolean(profile),
    staleTime: FEED_STALE_TIME_MS,
    gcTime: 30 * 60 * 1000,
  });

  const profileMissing =
    feedQuery.error instanceof ApiError && feedQuery.error.status === 404;

  useEffect(() => {
    if (profileMissing) clearStoredProfile();
  }, [profileMissing]);

  if (profile === undefined || feedQuery.isPending || !profile) {
    return <FeedSkeleton />;
  }

  function changeProfile() {
    clearStoredProfile();
    router.push("/");
  }

  if (feedQuery.isError) {
    const unavailable =
      feedQuery.error instanceof ApiError && feedQuery.error.status === 503;
    return (
      <main className="recommendations-page min-h-screen">
        <FeedHeader username={profile.username} onChangeProfile={changeProfile} />
        <section className="feed-error" role="alert">
          <span className="status-icon"><AlertTriangle aria-hidden="true" /></span>
          <p className="eyebrow text-[var(--orange)]">
            {profileMissing ? "PROFILE NOT FOUND" : unavailable ? "MODEL UNAVAILABLE" : "FEED INTERRUPTED"}
          </p>
          <h1>
            {profileMissing
              ? "This saved profile is no longer available."
              : "Your recommendations couldn’t be loaded."}
          </h1>
          <p>
            {profileMissing
              ? "Return to onboarding to synchronize or import it again."
              : feedQuery.error instanceof ApiError
                ? feedQuery.error.message
                : "NexdWatch hit an unexpected problem. Please try again."}
          </p>
          <div className="mt-7 flex flex-col gap-3 sm:flex-row">
            {!profileMissing && (
              <Button onClick={() => feedQuery.refetch()}>
                <RefreshCw size={17} aria-hidden="true" /> Try again
              </Button>
            )}
            <Button variant="secondary" onClick={changeProfile}>
              <ArrowLeft size={17} aria-hidden="true" /> Return to onboarding
            </Button>
          </div>
        </section>
        <SiteFooter />
      </main>
    );
  }

  return (
    <main className="recommendations-page">
      <FeedHeader username={profile.username} onChangeProfile={changeProfile} />
      <Feed feed={feedQuery.data} />
      <SiteFooter />
    </main>
  );
}
