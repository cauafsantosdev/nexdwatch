import type { Metadata } from "next";

import { RecommendationsPage } from "@/components/recommendations/recommendations-page";

export const metadata: Metadata = {
  title: "Your recommendations",
  description: "Personalized film shelves from your Letterboxd history.",
};

export default function Page() {
  return <RecommendationsPage />;
}
