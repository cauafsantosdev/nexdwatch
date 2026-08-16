"use client";

import { AlertTriangle, RefreshCw } from "lucide-react";
import { useEffect } from "react";

import { Button } from "@/components/ui/button";
import { Wordmark } from "@/components/ui/wordmark";

export default function ErrorPage({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Recommendations route failed", error);
  }, [error]);

  return (
    <main className="recommendations-page min-h-screen">
      <header className="feed-nav"><Wordmark compact /></header>
      <section className="feed-error" role="alert">
        <span className="status-icon"><AlertTriangle aria-hidden="true" /></span>
        <p className="eyebrow text-[var(--orange)]">UNEXPECTED INTERRUPTION</p>
        <h1>The projector stopped.</h1>
        <p>Your profile is safe. Retry this screen to load the feed again.</p>
        <Button className="mt-7" onClick={reset}>
          <RefreshCw size={17} aria-hidden="true" /> Try again
        </Button>
      </section>
    </main>
  );
}
