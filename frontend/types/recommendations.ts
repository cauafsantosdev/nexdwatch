export type RecommendationReasonCode =
  | "GLOBAL_RRF"
  | "NON_HEAD_TASTE_MATCH"
  | "BRAZILIAN_CINEMA_DISCOVERY"
  | "ANCHOR_SIMILARITY"
  | "DIRECTOR_AFFINITY"
  | "GENRE_AFFINITY"
  | "DECADE_AFFINITY"
  | "WORLD_CINEMA_DISCOVERY"
  | "LATENT_MATCH_METADATA_NOVELTY"
  | "CLASSIC_CINEMA_DISCOVERY";

export interface RecommendationReason {
  code: RecommendationReasonCode;
  anchor?: { film_id: number; title: string };
  entity?: {
    type: "director" | "genre" | "decade" | "country" | "language";
    name: string;
  };
}

export interface RecommendationFilm {
  film_id: number;
  title: string;
  year: number | null;
  directors: string[];
  tmdb_id: number | null;
  slug: string;
  reason: RecommendationReason;
}

export interface RecommendationCategory {
  key: string;
  title: string;
  experimental: boolean;
  preference_context?: {
    average_rating: number;
    rated_count: number;
  } | null;
  items: RecommendationFilm[];
}

export interface RecommendationFeed {
  user_id: number;
  categories: RecommendationCategory[];
}
