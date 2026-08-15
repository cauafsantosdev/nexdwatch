"""Deterministic category portfolio selection, diversity, and duplicate policy."""

from collections import Counter
from itertools import combinations
from typing import Any

from app.domain.categorized_recommendations import (
    AllocatedCategory,
    CategoryProposal,
    CategoryRole,
    UserCategoryProfile,
)
from app.policy.catalog import PolicyCatalog, PolicyFilm
from app.policy.config import DEFAULT_POLICY_CONFIG, CategoryPolicyConfig

_FOCUSED_REPEAT_KEYS = {
    "because_you_liked",
    "directors_you_love",
    "favorite_genre",
    "favorite_decade",
    "world_cinema",
    "classic_cinema",
}


def allocate_categories(
    proposals: tuple[CategoryProposal, ...],
    profile: UserCategoryProfile,
    catalog: PolicyCatalog,
    *,
    config: CategoryPolicyConfig = DEFAULT_POLICY_CONFIG,
) -> tuple[tuple[AllocatedCategory, ...], dict[str, Any]]:
    """Select viable proposals and allocate films under bounded relaxation.

    Top Picks is allocated first and reserves its leading films. Later rows prefer
    globally unseen films, then relax configured soft diversity caps only when a
    semantically viable proposal would otherwise miss its minimum size.

    Args:
        proposals: Viable category inventories in deterministic proposal order.
        profile: User evidence whose depth band controls portfolio role priority.
        catalog: Immutable metadata used to enforce within-row diversity.
        config: Frozen category count, overlap, repeat, and diversity limits.

    Returns:
        A tuple containing allocated categories and duplicate/overlap diagnostics.
    """
    # Top Picks is mandatory when available and reserves its leading identities from
    # all later repetition, establishing the portfolio's general-purpose baseline.
    top = next(
        (proposal for proposal in proposals if proposal.key == "top_picks"), None
    )
    if top is None:
        return (), _allocation_diagnostics(())
    top_ids = tuple(top.ordered_candidate_ids[: top.maximum_size])
    allocated: list[AllocatedCategory] = [AllocatedCategory(top, top_ids)]
    appearances = Counter(top_ids)
    reserved = set(top_ids[: config.top_picks_reserved])
    selected_raw_sets = [set(top.ordered_candidate_ids[: top.maximum_size])]
    remaining = [proposal for proposal in proposals if proposal.key != "top_picks"]

    # Recompute priority after every allocation because global novelty and overlap
    # depend on films already assigned to the portfolio.
    while remaining and len(allocated) < config.maximum_categories:
        ordered = sorted(
            remaining,
            key=lambda proposal: _proposal_priority(
                proposal,
                profile,
                appearances,
                selected_raw_sets,
            ),
        )
        proposal = ordered[0]
        remaining.remove(proposal)
        max_overlap = max(
            (_jaccard(_top_set(proposal), values) for values in selected_raw_sets),
            default=0.0,
        )
        if (
            max_overlap > config.category_overlap_threshold
            and proposal.evidence_tier != "strong"
        ):
            continue
        film_ids = _allocate_one(
            proposal,
            appearances,
            reserved,
            catalog,
            config,
        )
        if len(film_ids) < proposal.minimum_size:
            continue
        allocated.append(AllocatedCategory(proposal, film_ids))
        appearances.update(film_ids)
        selected_raw_sets.append(_top_set(proposal))

    result = tuple(allocated)
    return result, _allocation_diagnostics(result)


def _proposal_priority(
    proposal: CategoryProposal,
    profile: UserCategoryProfile,
    appearances: Counter[int],
    selected_sets: list[set[int]],
) -> tuple[Any, ...]:
    """Order proposals by history-band role, evidence, novelty, and stable ties."""
    role_order = {
        "sparse": {
            CategoryRole.CULTURAL: 0,
            CategoryRole.PERSONALIZED: 1,
            CategoryRole.DISCOVERY: 2,
            CategoryRole.SERENDIPITY: 3,
        },
        "established": {
            CategoryRole.PERSONALIZED: 0,
            CategoryRole.DISCOVERY: 1,
            CategoryRole.CULTURAL: 2,
            CategoryRole.SERENDIPITY: 3,
        },
        "deep": {
            CategoryRole.DISCOVERY: 0,
            CategoryRole.CULTURAL: 1,
            CategoryRole.SERENDIPITY: 2,
            CategoryRole.PERSONALIZED: 3,
        },
    }
    top = proposal.ordered_candidate_ids[: proposal.maximum_size]
    globally_new = sum(not appearances[film_id] for film_id in top)
    overlap = max(
        (_jaccard(set(top), selected) for selected in selected_sets), default=0.0
    )
    return (
        role_order[profile.history_depth_band].get(proposal.role, 4),
        0 if proposal.evidence_tier == "strong" else 1,
        -globally_new,
        overlap,
        -proposal.evidence_support,
        proposal.key,
    )


def _allocate_one(
    proposal: CategoryProposal,
    appearances: Counter[int],
    reserved: set[int],
    catalog: PolicyCatalog,
    config: CategoryPolicyConfig,
) -> tuple[int, ...]:
    """Fill one proposal through the policy's four bounded relaxation passes.

    The allocator first requires globally new films under all caps, then relaxes
    only soft decade/genre caps. Focused categories may subsequently reuse
    non-reserved films, first with and then without those soft caps; hard director,
    country, head-balance, and maximum-appearance limits always remain.
    """
    selected: list[int] = []
    local_seen: set[int] = set()
    # Pass 1: globally novel films satisfying both hard and soft diversity caps.
    _scan_candidates(
        proposal,
        selected,
        local_seen,
        appearances,
        reserved,
        catalog,
        config,
        allow_repeats=False,
        relax_soft_caps=False,
    )
    if len(selected) < proposal.minimum_size:
        # Pass 2: retain global novelty while relaxing generic decade/genre caps.
        _scan_candidates(
            proposal,
            selected,
            local_seen,
            appearances,
            reserved,
            catalog,
            config,
            allow_repeats=False,
            relax_soft_caps=True,
        )
    if len(selected) < proposal.minimum_size and proposal.key in _FOCUSED_REPEAT_KEYS:
        # Passes 3–4 permit bounded cross-category reuse only for semantically
        # focused shelves, never for reserved Top Picks identities.
        _scan_candidates(
            proposal,
            selected,
            local_seen,
            appearances,
            reserved,
            catalog,
            config,
            allow_repeats=True,
            relax_soft_caps=False,
        )
        if len(selected) < proposal.minimum_size:
            _scan_candidates(
                proposal,
                selected,
                local_seen,
                appearances,
                reserved,
                catalog,
                config,
                allow_repeats=True,
                relax_soft_caps=True,
            )
    return tuple(selected[: proposal.maximum_size])


def _scan_candidates(
    proposal: CategoryProposal,
    selected: list[int],
    local_seen: set[int],
    appearances: Counter[int],
    reserved: set[int],
    catalog: PolicyCatalog,
    config: CategoryPolicyConfig,
    *,
    allow_repeats: bool,
    relax_soft_caps: bool,
) -> None:
    """Append candidates allowed by the current repeat and diversity pass."""
    for film_id in proposal.ordered_candidate_ids:
        if len(selected) >= proposal.maximum_size or film_id in local_seen:
            continue
        prior = appearances[film_id]
        if allow_repeats:
            if (
                prior == 0
                or film_id in reserved
                or prior >= config.maximum_film_appearances
            ):
                continue
        elif prior:
            continue
        if not _passes_diversity(
            film_id,
            selected,
            proposal,
            catalog,
            config,
            relax_soft_caps=relax_soft_caps,
        ):
            continue
        selected.append(film_id)
        local_seen.add(film_id)


def _passes_diversity(
    film_id: int,
    selected_ids: list[int],
    proposal: CategoryProposal,
    catalog: PolicyCatalog,
    config: CategoryPolicyConfig,
    *,
    relax_soft_caps: bool,
) -> bool:
    """Enforce permanent hard caps and optionally relax generic soft caps."""
    film = catalog.film(film_id)
    if film is None:
        return False
    maximum_head_count = proposal.policy_metadata.get("maximum_head_count")
    head_candidate_ids = proposal.policy_metadata.get("head_candidate_ids")
    if (
        isinstance(maximum_head_count, int)
        and isinstance(head_candidate_ids, frozenset)
        and film_id in head_candidate_ids
        and sum(value in head_candidate_ids for value in selected_ids)
        >= maximum_head_count
    ):
        return False
    selected = [value for film_id in selected_ids if (value := catalog.film(film_id))]
    director_cap = (
        config.director_film_cap
        if proposal.key == "directors_you_love"
        else config.generic_director_cap
    )
    if _would_exceed_entity_cap(film, selected, "director", director_cap):
        return False
    if proposal.key == "world_cinema" and _would_exceed_entity_cap(
        film, selected, "country", config.world_country_cap
    ):
        return False
    if relax_soft_caps:
        return True
    if proposal.key != "favorite_decade" and _would_exceed_decade_cap(
        film, selected, config.generic_decade_cap
    ):
        return False
    return not (
        proposal.key != "favorite_genre"
        and _would_exceed_entity_cap(film, selected, "genre", config.generic_genre_cap)
    )


def _would_exceed_entity_cap(
    film: PolicyFilm,
    selected: list[PolicyFilm],
    family: str,
    cap: int,
) -> bool:
    """Return whether adding a film would exceed any shared entity's row cap."""
    for entity in film.entities(family):
        count = sum(
            any(value.id == entity.id for value in other.entities(family))
            for other in selected
        )
        if count >= cap:
            return True
    return False


def _would_exceed_decade_cap(
    film: PolicyFilm, selected: list[PolicyFilm], cap: int
) -> bool:
    return (
        film.decade is not None
        and sum(value.decade == film.decade for value in selected) >= cap
    )


def _allocation_diagnostics(
    categories: tuple[AllocatedCategory, ...],
) -> dict[str, Any]:
    """Summarize cross-category duplication and pairwise allocated overlap."""
    appearances = Counter(
        film_id for category in categories for film_id in category.film_ids
    )
    overlaps = {
        f"{left.proposal.key}|{right.proposal.key}": _jaccard(
            set(left.film_ids), set(right.film_ids)
        )
        for left, right in combinations(categories, 2)
    }
    total = sum(len(category.film_ids) for category in categories)
    return {
        "category_count": len(categories),
        "unique_film_count": len(appearances),
        "total_slots": total,
        "duplicate_slots": total - len(appearances),
        "duplicate_rate": (total - len(appearances)) / total if total else 0.0,
        "maximum_appearances": max(appearances.values(), default=0),
        "pairwise_jaccard": overlaps,
    }


def _top_set(proposal: CategoryProposal) -> set[int]:
    return set(proposal.ordered_candidate_ids[: proposal.maximum_size])


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0
