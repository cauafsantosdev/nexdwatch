"""Evidence-driven category proposal generation over one ranked inventory."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.domain.categorized_recommendations import (
    CategoryProposal,
    CategoryRole,
    EntityPreferenceRecord,
    RankedCandidate,
    RecommendationReason,
    RecommendationReasonCode,
    UserCategoryProfile,
)
from app.policy.catalog import PolicyCatalog, PolicyFilm
from app.policy.config import DEFAULT_POLICY_CONFIG, CategoryPolicyConfig
from app.policy.profile import preference_evidence_tier, qualifying_preferences

CATEGORY_KEYS = (
    "top_picks",
    "hidden_gems",
    "brazilian_cinema",
    "because_you_liked",
    "directors_you_love",
    "favorite_genre",
    "favorite_decade",
    "world_cinema",
    "outside_usual",
    "classic_cinema",
)


@dataclass(frozen=True, slots=True)
class ProposalBuildResult:
    proposals: tuple[CategoryProposal, ...]
    diagnostics: dict[str, Any]


def build_category_proposals(
    ranked_candidates: tuple[RankedCandidate, ...],
    profile: UserCategoryProfile,
    catalog: PolicyCatalog,
    item_vectors: NDArray[np.floating],
    id_to_position: dict[int, int],
    *,
    config: CategoryPolicyConfig = DEFAULT_POLICY_CONFIG,
) -> ProposalBuildResult:
    """Build all viable V1 category proposals exactly once for one user."""
    candidate_by_id = {candidate.film_id: candidate for candidate in ranked_candidates}
    proposals: list[CategoryProposal] = []
    diagnostics: dict[str, Any] = {}
    builders = (
        _top_picks,
        _hidden_gems,
        _brazilian_cinema,
        _directors_you_love,
        _favorite_genre,
        _favorite_decade,
        _world_cinema,
        _outside_usual,
        _classic_cinema,
    )
    for builder in builders:
        proposal = builder(ranked_candidates, profile, catalog, config)
        if proposal is not None:
            proposals.append(proposal)

    anchor, anchor_diagnostics = _because_you_liked(
        ranked_candidates,
        profile,
        catalog,
        item_vectors,
        id_to_position,
        config,
    )
    diagnostics["anchor"] = anchor_diagnostics
    if anchor is not None:
        proposals.append(anchor)
    diagnostics["director_pool"] = _director_diagnostics(profile, config)
    diagnostics["proposal_sizes"] = {
        proposal.key: len(proposal.ordered_candidate_ids) for proposal in proposals
    }
    diagnostics["unavailable_category_keys"] = sorted(
        set(CATEGORY_KEYS) - {proposal.key for proposal in proposals}
    )
    if set(candidate_by_id) != {candidate.film_id for candidate in ranked_candidates}:
        raise RuntimeError("candidate identity changed during category proposal build")
    return ProposalBuildResult(tuple(proposals), diagnostics)


def _top_picks(
    ranked: tuple[RankedCandidate, ...],
    profile: UserCategoryProfile,
    catalog: PolicyCatalog,
    config: CategoryPolicyConfig,
) -> CategoryProposal | None:
    del profile, catalog
    if not ranked:
        return None
    reasons = {
        candidate.film_id: _candidate_reason(
            RecommendationReasonCode.GLOBAL_RRF, candidate
        )
        for candidate in ranked
    }
    return CategoryProposal(
        key="top_picks",
        family="general",
        role=CategoryRole.GENERAL,
        title_template="Top Picks",
        title_parameters={},
        evidence_tier="strong",
        evidence_support=len(ranked),
        ordered_candidate_ids=tuple(candidate.film_id for candidate in ranked),
        minimum_size=1,
        maximum_size=config.top_picks_maximum,
        reasons=reasons,
    )


def _hidden_gems(
    ranked: tuple[RankedCandidate, ...],
    profile: UserCategoryProfile,
    catalog: PolicyCatalog,
    config: CategoryPolicyConfig,
) -> CategoryProposal | None:
    del catalog
    if profile.indexed_positive_count < config.hidden_minimum_indexed_positives:
        return None
    eligible = [
        candidate
        for candidate in ranked
        if _is_hidden_neighborhood_candidate(candidate, profile, config)
    ]
    return _generic_proposal(
        key="hidden_gems",
        family="discovery",
        role=CategoryRole.DISCOVERY,
        title="Hidden Gems for You",
        candidates=eligible,
        minimum=config.hidden_minimum,
        maximum=config.default_maximum,
        reason_code=RecommendationReasonCode.NON_HEAD_TASTE_MATCH,
        evidence_support=profile.indexed_positive_count,
        evidence_tier="strong" if profile.indexed_positive_count >= 6 else "minimum",
    )


def _brazilian_cinema(
    ranked: tuple[RankedCandidate, ...],
    profile: UserCategoryProfile,
    catalog: PolicyCatalog,
    config: CategoryPolicyConfig,
) -> CategoryProposal | None:
    del profile
    eligible = [
        candidate
        for candidate in ranked
        if _is_brazilian(catalog.film(candidate.film_id), config)
        and _positive_svd_evidence(candidate)
    ]
    eligible.sort(
        key=lambda value: (
            value.rrf_rank,
            value.svd_rank if value.svd_rank is not None else 10**9,
            value.film_id,
        )
    )
    return _generic_proposal(
        key="brazilian_cinema",
        family="cultural",
        role=CategoryRole.CULTURAL,
        title="Brazilian Cinema for You",
        candidates=eligible,
        minimum=config.brazilian_minimum,
        maximum=config.default_maximum,
        reason_code=RecommendationReasonCode.BRAZILIAN_CINEMA_DISCOVERY,
        evidence_support=len(eligible),
    )


def _directors_you_love(
    ranked: tuple[RankedCandidate, ...],
    profile: UserCategoryProfile,
    catalog: PolicyCatalog,
    config: CategoryPolicyConfig,
) -> CategoryProposal | None:
    directors = qualifying_preferences(profile, "director", config=config)[
        : config.director_pool_size
    ]
    if not directors:
        return None
    preference_by_id = {preference.entity_id: preference for preference in directors}
    ordered_ids: list[int] = []
    reasons: dict[int, RecommendationReason] = {}
    counts: dict[int, int] = {}
    for candidate in ranked:
        film = catalog.film(candidate.film_id)
        matches = _matching_preferences(film, "director", preference_by_id)
        if not matches:
            continue
        preference = matches[0]
        if counts.get(preference.entity_id, 0) >= config.director_film_cap:
            continue
        counts[preference.entity_id] = counts.get(preference.entity_id, 0) + 1
        ordered_ids.append(candidate.film_id)
        reasons[candidate.film_id] = _entity_reason(
            RecommendationReasonCode.DIRECTOR_AFFINITY, candidate, preference, config
        )
    if len(ordered_ids) < config.directors_minimum:
        return None
    return CategoryProposal(
        key="directors_you_love",
        family="director",
        role=CategoryRole.PERSONALIZED,
        title_template="From Directors You Love",
        title_parameters={},
        evidence_tier=(
            "strong"
            if any(
                preference_evidence_tier(value, config=config) == "strong"
                for value in directors
            )
            else "minimum"
        ),
        evidence_support=sum(value.support_count for value in directors),
        ordered_candidate_ids=tuple(ordered_ids),
        minimum_size=config.directors_minimum,
        maximum_size=config.default_maximum,
        reasons=reasons,
        policy_metadata={
            "directors": [_preference_summary(value) for value in directors]
        },
    )


def _favorite_genre(
    ranked: tuple[RankedCandidate, ...],
    profile: UserCategoryProfile,
    catalog: PolicyCatalog,
    config: CategoryPolicyConfig,
) -> CategoryProposal | None:
    preferences = qualifying_preferences(profile, "genre", config=config)
    return _single_entity_proposal(
        ranked,
        catalog,
        preferences[0] if preferences else None,
        key="favorite_genre",
        title_template="{entity} Picks for You",
        minimum=config.genre_minimum,
        reason_code=RecommendationReasonCode.GENRE_AFFINITY,
        config=config,
    )


def _favorite_decade(
    ranked: tuple[RankedCandidate, ...],
    profile: UserCategoryProfile,
    catalog: PolicyCatalog,
    config: CategoryPolicyConfig,
) -> CategoryProposal | None:
    preferences = qualifying_preferences(profile, "decade", config=config)
    return _single_entity_proposal(
        ranked,
        catalog,
        preferences[0] if preferences else None,
        key="favorite_decade",
        title_template="{entity} Films for You",
        minimum=config.decade_minimum,
        reason_code=RecommendationReasonCode.DECADE_AFFINITY,
        config=config,
    )


def _world_cinema(
    ranked: tuple[RankedCandidate, ...],
    profile: UserCategoryProfile,
    catalog: PolicyCatalog,
    config: CategoryPolicyConfig,
) -> CategoryProposal | None:
    country_preferences = tuple(
        preference
        for preference in qualifying_preferences(profile, "country", config=config)
        if preference.name.casefold() not in config.english_core_country_names
    )[: config.world_supported_country_pool]
    language_preferences = tuple(
        preference
        for preference in qualifying_preferences(profile, "language", config=config)
        if preference.name.casefold() not in config.english_language_names
        and preference.name.casefold() not in config.metadata_none_names
    )
    supported = {
        (value.family, value.entity_id): value
        for value in (*country_preferences, *language_preferences)
    }
    eligible: list[RankedCandidate] = []
    reasons: dict[int, RecommendationReason] = {}
    country_counts: dict[int, int] = {}
    for candidate in ranked:
        film = catalog.film(candidate.film_id)
        if film is None or not _is_world_cinema(film, config):
            continue
        matches = [
            supported[(family, entity.id)]
            for family in ("country", "language")
            for entity in film.entities(family)
            if (family, entity.id) in supported
        ]
        if not matches and not (
            _positive_svd_evidence(candidate)
            and candidate.svd_rank is not None
            and candidate.svd_rank <= config.world_discovery_svd_rank
        ):
            continue
        country_ids = [entity.id for entity in film.countries]
        if country_ids and all(
            country_counts.get(entity_id, 0) >= config.world_country_cap
            for entity_id in country_ids
        ):
            continue
        if country_ids:
            chosen_country = min(
                country_ids,
                key=lambda entity_id: (country_counts.get(entity_id, 0), entity_id),
            )
            country_counts[chosen_country] = country_counts.get(chosen_country, 0) + 1
        eligible.append(candidate)
        preference = (
            min(
                matches,
                key=lambda value: (
                    -value.affinity,
                    -value.support_count,
                    value.entity_id,
                ),
            )
            if matches
            else None
        )
        reasons[candidate.film_id] = (
            _entity_reason(
                RecommendationReasonCode.WORLD_CINEMA_DISCOVERY,
                candidate,
                preference,
                config,
            )
            if preference is not None
            else _candidate_reason(
                RecommendationReasonCode.WORLD_CINEMA_DISCOVERY, candidate
            )
        )
    if len(eligible) < config.world_minimum:
        return None
    return CategoryProposal(
        key="world_cinema",
        family="world",
        role=CategoryRole.CULTURAL,
        title_template="World Cinema for You",
        title_parameters={},
        evidence_tier="strong" if country_preferences else "minimum",
        evidence_support=sum(value.support_count for value in country_preferences),
        ordered_candidate_ids=tuple(value.film_id for value in eligible),
        minimum_size=config.world_minimum,
        maximum_size=config.default_maximum,
        reasons=reasons,
        policy_metadata={
            "countries": [_preference_summary(value) for value in country_preferences],
            "languages": [_preference_summary(value) for value in language_preferences],
            **_head_balance_metadata(eligible, config.world_head_cap),
        },
    )


def _outside_usual(
    ranked: tuple[RankedCandidate, ...],
    profile: UserCategoryProfile,
    catalog: PolicyCatalog,
    config: CategoryPolicyConfig,
) -> CategoryProposal | None:
    familiar = {
        family: {
            value.entity_id
            for value in qualifying_preferences(profile, family, config=config)[
                : config.outside_familiar_entities_per_family
            ]
        }
        for family in ("director", "genre", "decade", "country", "language")
    }
    if (
        sum(bool(values) for values in familiar.values())
        < config.outside_minimum_familiar_families
    ):
        return None
    eligible = []
    primary_eligible_count = 0
    for candidate in ranked:
        is_lower_head = candidate.popularity_stratum == "HEAD"
        if not _positive_svd_evidence(candidate) or candidate.svd_rank is None:
            continue
        film = catalog.film(candidate.film_id)
        if film is None or any(
            familiar[family].intersection(entity.id for entity in film.entities(family))
            for family in familiar
        ):
            continue
        if candidate.rrf_rank > config.outside_rrf_rank:
            continue
        if is_lower_head:
            if config.outside_exclude_head:
                continue
            if config.outside_lower_head_cap > 0 and (
                candidate.retrieved_by_popularity
                or candidate.svd_rank > config.outside_lower_head_svd_rank
            ):
                continue
        elif candidate.svd_rank > config.outside_svd_rank:
            continue
        if config.outside_exclude_hidden_neighborhood and (
            _is_hidden_neighborhood_candidate(candidate, profile, config)
        ):
            continue
        if (
            not is_lower_head
            and candidate.svd_rank <= config.outside_primary_svd_rank
            and candidate.rrf_rank <= config.outside_primary_rrf_rank
        ):
            primary_eligible_count += 1
        eligible.append(candidate)
    if (
        config.outside_require_primary_viability
        and primary_eligible_count < config.outside_minimum
    ):
        return None
    eligible.sort(
        key=lambda value: (
            value.svd_rank if value.svd_rank is not None else 10**9,
            value.rrf_rank,
            value.film_id,
        )
    )
    return _generic_proposal(
        key="outside_usual",
        family="serendipity",
        role=CategoryRole.SERENDIPITY,
        title="Outside Your Usual Picks",
        candidates=eligible,
        minimum=config.outside_minimum,
        maximum=config.default_maximum,
        reason_code=RecommendationReasonCode.LATENT_MATCH_METADATA_NOVELTY,
        evidence_support=sum(len(values) for values in familiar.values()),
        policy_metadata={
            "familiar_entities": {
                family: sorted(values) for family, values in familiar.items()
            },
            "excluded_hidden_neighborhood": (
                config.outside_exclude_hidden_neighborhood
            ),
            "primary_eligible_count": primary_eligible_count,
            **_head_balance_metadata(
                eligible,
                (
                    config.outside_lower_head_cap
                    if config.outside_lower_head_cap > 0
                    else None
                ),
            ),
        },
    )


def _classic_cinema(
    ranked: tuple[RankedCandidate, ...],
    profile: UserCategoryProfile,
    catalog: PolicyCatalog,
    config: CategoryPolicyConfig,
) -> CategoryProposal | None:
    eligible = [
        candidate
        for candidate in ranked
        if (film := catalog.film(candidate.film_id)) is not None
        and film.year is not None
        and film.year <= config.classic_year_boundary
        and _positive_svd_evidence(candidate)
    ]
    eligible.sort(
        key=lambda value: (
            value.rrf_rank,
            value.svd_rank if value.svd_rank is not None else 10**9,
            value.film_id,
        )
    )
    return _generic_proposal(
        key="classic_cinema",
        family="classic",
        role=CategoryRole.CULTURAL,
        title="Classic Cinema for You",
        candidates=eligible,
        minimum=config.classic_minimum,
        maximum=config.default_maximum,
        reason_code=RecommendationReasonCode.CLASSIC_CINEMA_DISCOVERY,
        evidence_support=profile.indexed_positive_count,
        policy_metadata={
            "year_boundary": config.classic_year_boundary,
            **_head_balance_metadata(eligible, config.classic_head_cap),
        },
    )


def _because_you_liked(
    ranked: tuple[RankedCandidate, ...],
    profile: UserCategoryProfile,
    catalog: PolicyCatalog,
    item_vectors: NDArray[np.floating],
    id_to_position: dict[int, int],
    config: CategoryPolicyConfig,
) -> tuple[CategoryProposal | None, dict[str, Any]]:
    del catalog
    high_anchors = [
        anchor for anchor in profile.anchors if anchor.rating >= config.anchor_rating
    ]
    anchors = high_anchors or [
        anchor
        for anchor in profile.anchors
        if anchor.rating >= config.anchor_fallback_rating
    ]
    candidate_ids = [
        candidate.film_id for candidate in ranked if candidate.film_id in id_to_position
    ]
    if not anchors or not candidate_ids:
        return None, {"eligible_anchors": len(anchors), "selected": None}
    candidate_matrix = np.ascontiguousarray(
        item_vectors[[id_to_position[film_id] for film_id in candidate_ids]],
        dtype=np.float32,
    )
    candidate_by_id = {candidate.film_id: candidate for candidate in ranked}
    candidate_id_array = np.asarray(candidate_ids, dtype=np.int64)
    candidate_rrf_ranks = np.asarray(
        [candidate_by_id[film_id].rrf_rank for film_id in candidate_ids],
        dtype=np.int64,
    )
    top_picks = {candidate.film_id for candidate in ranked[: config.top_picks_maximum]}
    valid_anchors = tuple(
        anchor for anchor in anchors if anchor.film_id in id_to_position
    )
    neighborhood_size = _anchor_neighborhood_size(config, len(candidate_ids))
    neighborhoods: list[tuple[Any, float, int, float, float]] = []
    for start in range(0, len(valid_anchors), config.anchor_similarity_batch_size):
        batch = valid_anchors[start : start + config.anchor_similarity_batch_size]
        anchor_matrix = np.ascontiguousarray(
            item_vectors[[id_to_position[anchor.film_id] for anchor in batch]],
            dtype=np.float32,
        )
        batch_similarities = anchor_matrix @ candidate_matrix.T
        for anchor, similarities in zip(batch, batch_similarities, strict=True):
            finite_indices = np.flatnonzero(np.isfinite(similarities))
            usable_indices = (
                _top_neighbor_indices(
                    similarities,
                    finite_indices,
                    candidate_id_array,
                    candidate_rrf_ranks,
                    neighborhood_size,
                )
                if neighborhood_size is not None
                else finite_indices[
                    similarities[finite_indices] > config.anchor_similarity_threshold
                ]
            )
            if len(usable_indices) < config.anchor_minimum:
                continue
            top_indices = _top_neighbor_indices(
                similarities,
                usable_indices,
                candidate_id_array,
                candidate_rrf_ranks,
                config.anchor_maximum,
            )
            mean_similarity = float(np.mean(similarities[top_indices]))
            overlap = _jaccard(set(candidate_id_array[top_indices].tolist()), top_picks)
            similarity_cutoff = float(np.min(similarities[usable_indices]))
            neighborhoods.append(
                (
                    anchor,
                    mean_similarity,
                    len(usable_indices),
                    overlap,
                    similarity_cutoff,
                )
            )
    if not neighborhoods:
        return None, {"eligible_anchors": len(anchors), "selected": None}
    selected = min(
        neighborhoods,
        key=lambda value: (
            -value[0].rating,
            -value[1],
            -value[2],
            value[3],
            value[0].film_id,
        ),
    )
    anchor, mean_similarity, usable_count, overlap, similarity_cutoff = selected
    similarities = candidate_matrix @ item_vectors[id_to_position[anchor.film_id]]
    finite_indices = np.flatnonzero(np.isfinite(similarities))
    usable_indices = (
        _top_neighbor_indices(
            similarities,
            finite_indices,
            candidate_id_array,
            candidate_rrf_ranks,
            neighborhood_size,
        )
        if neighborhood_size is not None
        else finite_indices[
            similarities[finite_indices] > config.anchor_similarity_threshold
        ]
    )
    order = np.lexsort(
        (
            candidate_id_array[usable_indices],
            candidate_rrf_ranks[usable_indices],
            -similarities[usable_indices],
        )
    )
    ordered_ids = tuple(candidate_id_array[usable_indices[order]].tolist())
    reasons = {
        film_id: _candidate_reason(
            RecommendationReasonCode.ANCHOR_SIMILARITY,
            candidate_by_id[film_id],
            anchor_film_id=anchor.film_id,
            anchor_title=anchor.title,
        )
        for film_id in ordered_ids
    }
    proposal = CategoryProposal(
        key="because_you_liked",
        family="anchor",
        role=CategoryRole.PERSONALIZED,
        title_template="Because You Liked {anchor}",
        title_parameters={"anchor": anchor.title},
        evidence_tier="strong" if anchor.rating >= config.anchor_rating else "minimum",
        evidence_support=1,
        ordered_candidate_ids=ordered_ids,
        minimum_size=config.anchor_minimum,
        maximum_size=config.anchor_maximum,
        reasons=reasons,
        policy_metadata={
            "anchor_film_id": anchor.film_id,
            "anchor_title": anchor.title,
            "anchor_rating": anchor.rating,
            "mean_top_20_similarity": mean_similarity,
            "usable_neighbor_count": usable_count,
            "top_picks_overlap": overlap,
            "neighborhood_rule": _anchor_neighborhood_rule(config),
            "local_similarity_cutoff": similarity_cutoff,
        },
    )
    return proposal, {
        "eligible_anchors": len(anchors),
        "selected": proposal.policy_metadata,
    }


def _anchor_neighborhood_size(
    config: CategoryPolicyConfig, candidate_count: int
) -> int | None:
    if config.anchor_neighborhood_limit is not None:
        return min(config.anchor_neighborhood_limit, candidate_count)
    if config.anchor_neighborhood_fraction is not None:
        return min(
            candidate_count,
            max(
                config.anchor_minimum,
                int(np.ceil(candidate_count * config.anchor_neighborhood_fraction)),
            ),
        )
    return None


def _anchor_neighborhood_rule(config: CategoryPolicyConfig) -> str:
    if config.anchor_neighborhood_limit is not None:
        return f"top_{config.anchor_neighborhood_limit}"
    if config.anchor_neighborhood_fraction is not None:
        percentage = round(config.anchor_neighborhood_fraction * 100)
        return f"top_{percentage}_percent"
    return "legacy_positive_similarity"


def _top_neighbor_indices(
    similarities: NDArray[np.floating],
    usable_indices: NDArray[np.integer],
    candidate_ids: NDArray[np.integer],
    rrf_ranks: NDArray[np.integer],
    maximum: int,
) -> NDArray[np.integer]:
    """Return exact top neighbors without fully sorting every anchor row."""
    if len(usable_indices) <= maximum:
        contenders = usable_indices
    else:
        usable_scores = similarities[usable_indices]
        partial = np.argpartition(-usable_scores, maximum - 1)[:maximum]
        cutoff = float(np.min(usable_scores[partial]))
        contenders = usable_indices[usable_scores >= cutoff]
    order = np.lexsort(
        (
            candidate_ids[contenders],
            rrf_ranks[contenders],
            -similarities[contenders],
        )
    )
    return contenders[order[:maximum]]


def _single_entity_proposal(
    ranked: tuple[RankedCandidate, ...],
    catalog: PolicyCatalog,
    preference: EntityPreferenceRecord | None,
    *,
    key: str,
    title_template: str,
    minimum: int,
    reason_code: RecommendationReasonCode,
    config: CategoryPolicyConfig,
) -> CategoryProposal | None:
    if preference is None:
        return None
    candidates = [
        candidate
        for candidate in ranked
        if _film_has_entity(catalog.film(candidate.film_id), preference)
    ]
    if len(candidates) < minimum:
        return None
    reasons = {
        candidate.film_id: _entity_reason(reason_code, candidate, preference, config)
        for candidate in candidates
    }
    return CategoryProposal(
        key=key,
        family=preference.family,
        role=CategoryRole.PERSONALIZED,
        title_template=title_template,
        title_parameters={"entity": preference.name},
        evidence_tier=preference_evidence_tier(preference, config=config),
        evidence_support=preference.support_count,
        ordered_candidate_ids=tuple(value.film_id for value in candidates),
        minimum_size=minimum,
        maximum_size=config.default_maximum,
        reasons=reasons,
        policy_metadata={"selected_entity": _preference_summary(preference)},
    )


def _generic_proposal(
    *,
    key: str,
    family: str,
    role: CategoryRole,
    title: str,
    candidates: list[RankedCandidate],
    minimum: int,
    maximum: int,
    reason_code: RecommendationReasonCode,
    evidence_support: int,
    evidence_tier: str = "minimum",
    policy_metadata: dict[str, Any] | None = None,
) -> CategoryProposal | None:
    if len(candidates) < minimum:
        return None
    return CategoryProposal(
        key=key,
        family=family,
        role=role,
        title_template=title,
        title_parameters={},
        evidence_tier=evidence_tier,
        evidence_support=evidence_support,
        ordered_candidate_ids=tuple(candidate.film_id for candidate in candidates),
        minimum_size=minimum,
        maximum_size=maximum,
        reasons={
            candidate.film_id: _candidate_reason(reason_code, candidate)
            for candidate in candidates
        },
        policy_metadata=policy_metadata or {},
    )


def _candidate_reason(
    code: RecommendationReasonCode,
    candidate: RankedCandidate,
    *,
    anchor_film_id: int | None = None,
    anchor_title: str | None = None,
) -> RecommendationReason:
    return RecommendationReason(
        code=code,
        additional_codes=(
            (RecommendationReasonCode.SOURCE_AGREEMENT,)
            if candidate.retrieved_by_both
            else ()
        ),
        anchor_film_id=anchor_film_id,
        anchor_title=anchor_title,
        popularity_stratum=candidate.popularity_stratum,
        retrieved_by_both=candidate.retrieved_by_both,
    )


def _entity_reason(
    code: RecommendationReasonCode,
    candidate: RankedCandidate,
    preference: EntityPreferenceRecord,
    config: CategoryPolicyConfig,
) -> RecommendationReason:
    base = _candidate_reason(code, candidate)
    return RecommendationReason(
        code=base.code,
        additional_codes=base.additional_codes,
        entity_family=preference.family,
        entity_name=preference.name,
        support_count=preference.support_count,
        high_rating_count=preference.high_rating_count,
        popularity_stratum=base.popularity_stratum,
        evidence_tier=preference_evidence_tier(preference, config=config),
        retrieved_by_both=base.retrieved_by_both,
    )


def _matching_preferences(
    film: PolicyFilm | None,
    family: str,
    preference_by_id: dict[int, EntityPreferenceRecord],
) -> list[EntityPreferenceRecord]:
    if film is None:
        return []
    matches = [
        preference_by_id[entity.id]
        for entity in film.entities(family)
        if entity.id in preference_by_id
    ]
    return sorted(
        matches,
        key=lambda value: (-value.affinity, -value.support_count, value.entity_id),
    )


def _film_has_entity(
    film: PolicyFilm | None, preference: EntityPreferenceRecord
) -> bool:
    return film is not None and any(
        entity.id == preference.entity_id for entity in film.entities(preference.family)
    )


def _positive_svd_evidence(candidate: RankedCandidate) -> bool:
    return (
        candidate.retrieved_by_svd
        and candidate.svd_rank is not None
        and candidate.svd_score is not None
        and candidate.svd_score > 0
    )


def _is_hidden_neighborhood_candidate(
    candidate: RankedCandidate,
    profile: UserCategoryProfile,
    config: CategoryPolicyConfig,
) -> bool:
    return (
        profile.indexed_positive_count >= config.hidden_minimum_indexed_positives
        and _positive_svd_evidence(candidate)
        and candidate.svd_rank is not None
        and (
            candidate.popularity_stratum == "MID"
            and candidate.svd_rank <= config.hidden_mid_svd_rank
            or candidate.popularity_stratum == "TAIL"
            and candidate.svd_rank <= config.hidden_tail_svd_rank
        )
    )


def _head_balance_metadata(
    candidates: list[RankedCandidate], head_cap: int | None
) -> dict[str, Any]:
    if head_cap is None:
        return {}
    return {
        "maximum_head_count": head_cap,
        "head_candidate_ids": frozenset(
            candidate.film_id
            for candidate in candidates
            if candidate.popularity_stratum == "HEAD"
        ),
    }


def _is_brazilian(film: PolicyFilm | None, config: CategoryPolicyConfig) -> bool:
    return film is not None and any(
        entity.name.casefold() in config.brazilian_country_names
        for entity in film.countries
    )


def _is_world_cinema(film: PolicyFilm, config: CategoryPolicyConfig) -> bool:
    non_core_country = any(
        entity.name.casefold() not in config.english_core_country_names
        and entity.name.casefold() not in config.metadata_none_names
        for entity in film.countries
    )
    non_english_language = any(
        entity.name.casefold() not in config.english_language_names
        and entity.name.casefold() not in config.metadata_none_names
        for entity in film.languages
    )
    return non_core_country and non_english_language


def _preference_summary(preference: EntityPreferenceRecord) -> dict[str, Any]:
    return {
        "family": preference.family,
        "name": preference.name,
        "support_count": preference.support_count,
        "mean_rating": preference.mean_rating,
        "positive_fraction": preference.positive_fraction,
        "high_rating_count": preference.high_rating_count,
        "affinity": preference.affinity,
    }


def _director_diagnostics(
    profile: UserCategoryProfile, config: CategoryPolicyConfig
) -> dict[str, Any]:
    qualified = qualifying_preferences(profile, "director", config=config)
    selected = qualified[: config.director_pool_size]
    return {
        "qualifying_count": len(qualified),
        "selected_count": len(selected),
        "selected": [_preference_summary(value) for value in selected],
    }


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0
