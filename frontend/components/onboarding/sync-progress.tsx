"use client";

import { AlertTriangle, Check, Clock3, LoaderCircle, RotateCcw, Upload } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import type { TaskStatus } from "@/types/tasks";

const statusCopy: Record<TaskStatus, { title: string; detail: string }> = {
  queued: {
    title: "Your profile is queued.",
    detail: "NexdWatch will begin reading your public Letterboxd history shortly.",
  },
  processing: {
    title: "Reading your Letterboxd history…",
    detail: "Finding patterns in your taste and looking beyond the obvious picks.",
  },
  completed: {
    title: "Your recommendations are ready.",
    detail: "Taking you to your personalized shelves now.",
  },
  failed: {
    title: "We couldn’t load this profile.",
    detail: "Letterboxd may be unavailable or the profile may not be public.",
  },
};

interface SyncProgressProps {
  username: string;
  status: TaskStatus;
  errorMessage?: string;
  onRetry: () => void;
  onImport: () => void;
}

export function SyncProgress({
  username,
  status,
  errorMessage,
  onRetry,
  onImport,
}: SyncProgressProps) {
  const failed = status === "failed" || Boolean(errorMessage);
  const copy = failed
    ? {
        title: "We couldn’t load this profile.",
        detail: errorMessage || statusCopy.failed.detail,
      }
    : statusCopy[status];

  return (
    <section className="sync-panel" aria-live="polite" aria-busy={!failed && status !== "completed"}>
      <div className="flex items-center justify-between gap-4">
        <Badge tone={failed ? "orange" : status === "completed" ? "green" : "blue"}>
          {failed ? "NEEDS ATTENTION" : status.toUpperCase()}
        </Badge>
        <span className="font-mono text-xs text-[var(--muted)]">@{username}</span>
      </div>

      <div className="mt-8 flex items-start gap-4">
        <span className="status-icon" aria-hidden="true">
          {failed ? (
            <AlertTriangle />
          ) : status === "queued" ? (
            <Clock3 />
          ) : status === "completed" ? (
            <Check />
          ) : (
            <LoaderCircle className="motion-safe:animate-spin" />
          )}
        </span>
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-[var(--ink)] sm:text-3xl">
            {copy.title}
          </h2>
          <p className="mt-2 max-w-md leading-7 text-[var(--muted)]">{copy.detail}</p>
        </div>
      </div>

      {!failed && status !== "completed" && (
        <div className="indeterminate-track mt-8" aria-hidden="true">
          <span />
        </div>
      )}

      {failed && (
        <div className="mt-8 flex flex-col gap-3 sm:flex-row">
          <Button onClick={onRetry} className="justify-center">
            <RotateCcw size={17} aria-hidden="true" /> Try again
          </Button>
          <Button variant="secondary" onClick={onImport} className="justify-center">
            <Upload size={17} aria-hidden="true" /> Import Letterboxd export
          </Button>
        </div>
      )}

      <p className="mt-7 text-xs leading-5 text-[var(--quiet)]">
        Profile synchronization can take a moment for longer film histories. This status reflects the real backend task state.
      </p>
    </section>
  );
}
