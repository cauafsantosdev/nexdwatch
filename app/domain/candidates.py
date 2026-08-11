"""Ranking-independent candidate-generation domain types."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecommendationCandidate:
    """One deduplicated film with independent retrieval-source provenance."""

    film_id: int
    svd_score: float | None = None
    svd_rank: int | None = None
    popularity_score: int | None = None
    popularity_rank: int | None = None
    retrieved_by_svd: bool = False
    retrieved_by_popularity: bool = False

    @property
    def source_count(self) -> int:
        """Return the number of independent sources that retrieved this film."""
        return int(self.retrieved_by_svd) + int(self.retrieved_by_popularity)


@dataclass(frozen=True, slots=True)
class CandidateGenerationResult:
    """Variable-size broad candidate inventory for a future ranker."""

    user_id: int
    candidates: tuple[RecommendationCandidate, ...]
    nominal_budget: int
    svd_depth: int
    popularity_depth: int
    svd_profile_available: bool

    @property
    def unique_candidate_count(self) -> int:
        return len(self.candidates)
