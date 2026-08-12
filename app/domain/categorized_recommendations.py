"""Internal domain contracts for categorized recommendation policy."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from app.domain.candidates import RecommendationCandidate

PopularityStratum = Literal["HEAD", "MID", "TAIL"]
EntityFamily = Literal["director", "genre", "decade", "country", "language"]
EvidenceTier = Literal["minimum", "strong"]


class CategoryRole(StrEnum):
    GENERAL = "general"
    PERSONALIZED = "personalized"
    DISCOVERY = "discovery"
    CULTURAL = "cultural"
    SERENDIPITY = "serendipity"


class RecommendationReasonCode(StrEnum):
    GLOBAL_RRF = "GLOBAL_RRF"
    SOURCE_AGREEMENT = "SOURCE_AGREEMENT"
    NON_HEAD_TASTE_MATCH = "NON_HEAD_TASTE_MATCH"
    BRAZILIAN_CINEMA_DISCOVERY = "BRAZILIAN_CINEMA_DISCOVERY"
    ANCHOR_SIMILARITY = "ANCHOR_SIMILARITY"
    DIRECTOR_AFFINITY = "DIRECTOR_AFFINITY"
    GENRE_AFFINITY = "GENRE_AFFINITY"
    DECADE_AFFINITY = "DECADE_AFFINITY"
    WORLD_CINEMA_DISCOVERY = "WORLD_CINEMA_DISCOVERY"
    LATENT_MATCH_METADATA_NOVELTY = "LATENT_MATCH_METADATA_NOVELTY"
    CLASSIC_CINEMA_DISCOVERY = "CLASSIC_CINEMA_DISCOVERY"


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """One candidate after exact finalized RRF ordering."""

    film_id: int
    svd_score: float | None
    svd_rank: int | None
    popularity_count: int | None
    popularity_rank: int | None
    retrieved_by_svd: bool
    retrieved_by_popularity: bool
    rrf_score: float
    rrf_rank: int
    popularity_stratum: PopularityStratum

    @classmethod
    def from_candidate(
        cls,
        candidate: RecommendationCandidate,
        *,
        rrf_score: float,
        rrf_rank: int,
        popularity_stratum: PopularityStratum,
    ) -> "RankedCandidate":
        return cls(
            film_id=candidate.film_id,
            svd_score=candidate.svd_score,
            svd_rank=candidate.svd_rank,
            popularity_count=candidate.popularity_score,
            popularity_rank=candidate.popularity_rank,
            retrieved_by_svd=candidate.retrieved_by_svd,
            retrieved_by_popularity=candidate.retrieved_by_popularity,
            rrf_score=rrf_score,
            rrf_rank=rrf_rank,
            popularity_stratum=popularity_stratum,
        )

    @property
    def retrieved_by_both(self) -> bool:
        return self.retrieved_by_svd and self.retrieved_by_popularity

    @property
    def source_membership(self) -> str:
        if self.retrieved_by_both:
            return "both"
        if self.retrieved_by_svd:
            return "svd_only"
        return "popularity_only"


@dataclass(frozen=True, slots=True)
class EntityPreferenceRecord:
    """Support-aware preference evidence for one catalog entity."""

    family: EntityFamily
    entity_id: int
    name: str
    support_count: int
    mean_rating: float
    positive_count: int
    high_rating_count: int
    negative_count: int
    positive_fraction: float
    high_rating_fraction: float
    raw_preference: float
    confidence: float
    affinity: float


@dataclass(frozen=True, slots=True)
class AnchorPreference:
    film_id: int
    title: str
    rating: float


@dataclass(frozen=True, slots=True)
class UserCategoryProfile:
    """One request-scoped profile shared by every category proposal."""

    user_id: int
    watched_count: int
    rated_count: int
    user_mean_rating: float | None
    positive_count: int
    indexed_positive_count: int
    high_count: int
    negative_count: int
    anchors: tuple[AnchorPreference, ...]
    preferences: dict[EntityFamily, tuple[EntityPreferenceRecord, ...]]
    history_depth_band: Literal["sparse", "established", "deep"]


@dataclass(frozen=True, slots=True)
class RecommendationReason:
    """Structured internal explanation without raw research scores."""

    code: RecommendationReasonCode
    additional_codes: tuple[RecommendationReasonCode, ...] = ()
    anchor_film_id: int | None = None
    anchor_title: str | None = None
    entity_family: EntityFamily | None = None
    entity_name: str | None = None
    support_count: int | None = None
    high_rating_count: int | None = None
    popularity_stratum: PopularityStratum | None = None
    evidence_tier: EvidenceTier | None = None
    retrieved_by_both: bool = False


@dataclass(frozen=True, slots=True)
class CategoryProposal:
    """A semantically eligible ordered pool before portfolio allocation."""

    key: str
    family: str
    role: CategoryRole
    title_template: str
    title_parameters: dict[str, str | int]
    evidence_tier: EvidenceTier
    evidence_support: int
    ordered_candidate_ids: tuple[int, ...]
    minimum_size: int
    maximum_size: int
    reasons: dict[int, RecommendationReason]
    policy_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CategorizedRecommendation:
    film_id: int
    title: str
    year: int | None
    directors: tuple[str, ...]
    reason: RecommendationReason
    rrf_rank: int
    popularity_stratum: PopularityStratum
    source_membership: str


@dataclass(frozen=True, slots=True)
class RecommendationCategory:
    key: str
    family: str
    role: CategoryRole
    title: str
    items: tuple[CategorizedRecommendation, ...]
    evidence_tier: EvidenceTier
    evidence_support: int


@dataclass(frozen=True, slots=True)
class AllocatedCategory:
    """Allocation result before display metadata is batch-resolved."""

    proposal: CategoryProposal
    film_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CategoryPolicyResult:
    user_id: int
    ranked_candidates: tuple[RankedCandidate, ...]
    proposals: tuple[CategoryProposal, ...]
    allocated_categories: tuple[AllocatedCategory, ...]
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CategorizedRecommendationResult:
    user_id: int
    categories: tuple[RecommendationCategory, ...]
    diagnostics: dict[str, Any]
