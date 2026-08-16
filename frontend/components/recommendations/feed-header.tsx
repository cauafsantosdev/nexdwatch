"use client";

import { ArrowUpRight, UserRoundPen } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Wordmark } from "@/components/ui/wordmark";

interface FeedHeaderProps {
  username: string;
  onChangeProfile: () => void;
}

export function FeedHeader({ username, onChangeProfile }: FeedHeaderProps) {
  return (
    <>
      <header className="feed-nav">
        <Wordmark compact />
        <div className="feed-profile-actions">
          <span className="current-profile"><span className="status-dot" /> @{username}</span>
          <Button variant="ghost" onClick={onChangeProfile}>
            <UserRoundPen size={16} aria-hidden="true" /> Change profile
          </Button>
        </div>
      </header>
      <div className="feed-intro">
        <div>
          <p className="eyebrow"><span /> CURATED FOR YOUR TASTE</p>
          <h1>Recommendations<br />for <em>@{username}</em></h1>
        </div>
        <p>
          Built from your Letterboxd history to surface films that fit your taste without only repeating the obvious choices.
          <ArrowUpRight aria-hidden="true" />
        </p>
      </div>
    </>
  );
}
