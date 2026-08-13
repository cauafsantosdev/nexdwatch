import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import numpy as np

from app.domain.candidates import RecommendationCandidate
from app.domain.categorized_recommendations import (
    AnchorPreference,
    CategoryProposal,
    CategoryRole,
    EntityPreferenceRecord,
    RankedCandidate,
    RecommendationReason,
    RecommendationReasonCode,
    UserCategoryProfile,
)
from app.policy.allocation import allocate_categories
from app.policy.catalog import (
    PolicyCatalog,
    PolicyEntity,
    PolicyFilm,
    load_policy_catalog,
)
from app.policy.config import DEFAULT_POLICY_CONFIG, V1_POLICY_CONFIG
from app.policy.profile import (
    build_user_category_profile,
    qualifying_preferences,
)
from app.policy.proposals import _because_you_liked, build_category_proposals
from app.policy.ranking import rank_candidates_by_rrf
from app.repositories.interactions import (
    RatedInteraction,
    RecommendationHistory,
)


def _entity(entity_id: int, name: str) -> PolicyEntity:
    return PolicyEntity(entity_id, name)


def _catalog(films: list[PolicyFilm]) -> PolicyCatalog:
    values = {film.film_id: film for film in films}
    return PolicyCatalog(values, frozenset(values))


def _ranked(
    film_id: int,
    rank: int,
    *,
    stratum: str = "MID",
    svd_rank: int | None = None,
    popularity_rank: int | None = None,
    score: float = 0.5,
) -> RankedCandidate:
    return RankedCandidate(
        film_id=film_id,
        svd_score=score if svd_rank is not None else None,
        svd_rank=svd_rank,
        popularity_count=100 if popularity_rank is not None else None,
        popularity_rank=popularity_rank,
        retrieved_by_svd=svd_rank is not None,
        retrieved_by_popularity=popularity_rank is not None,
        rrf_score=1 / (60 + rank),
        rrf_rank=rank,
        popularity_stratum=stratum,
    )


def _preference(
    family: str,
    entity_id: int,
    name: str,
    *,
    support: int = 6,
    affinity: float = 0.2,
) -> EntityPreferenceRecord:
    return EntityPreferenceRecord(
        family=family,
        entity_id=entity_id,
        name=name,
        support_count=support,
        mean_rating=4.5,
        positive_count=support,
        high_rating_count=max(2, support // 2),
        negative_count=0,
        positive_fraction=1.0,
        high_rating_fraction=0.5,
        raw_preference=0.3,
        confidence=0.7,
        affinity=affinity,
    )


def _profile(
    *,
    preferences: dict | None = None,
    anchors: tuple[AnchorPreference, ...] = (),
    band: str = "established",
) -> UserCategoryProfile:
    empty = {
        "director": (),
        "genre": (),
        "decade": (),
        "country": (),
        "language": (),
    }
    empty.update(preferences or {})
    return UserCategoryProfile(
        user_id=7,
        watched_count=50,
        rated_count=20,
        user_mean_rating=3.5,
        positive_count=10,
        indexed_positive_count=10,
        high_count=6,
        negative_count=2,
        anchors=anchors,
        preferences=empty,
        history_depth_band=band,
    )


def test_rrf_scoring_missing_sources_provenance_and_film_id_tie() -> None:
    candidates = (
        RecommendationCandidate(20, svd_rank=1, retrieved_by_svd=True),
        RecommendationCandidate(10, svd_rank=1, retrieved_by_svd=True),
        RecommendationCandidate(
            30,
            popularity_score=4,
            popularity_rank=1,
            retrieved_by_popularity=True,
        ),
        RecommendationCandidate(
            40,
            svd_rank=2,
            popularity_score=3,
            popularity_rank=2,
            retrieved_by_svd=True,
            retrieved_by_popularity=True,
        ),
    )
    ranked = rank_candidates_by_rrf(candidates, {10: 10, 20: 20, 30: 30, 40: 40}, 100)

    assert [candidate.film_id for candidate in ranked] == [40, 10, 20, 30]
    assert ranked[1].rrf_score == 1 / 61
    assert ranked[0].retrieved_by_both
    assert ranked[0].source_membership == "both"


def test_support_shrunk_user_centered_affinity_resists_one_observation() -> None:
    france = _entity(1, "France")
    russia = _entity(2, "Russia")
    other = _entity(3, "Other")
    films = [
        PolicyFilm(film_id, str(film_id), 2000, countries=(france,))
        for film_id in range(1, 6)
    ]
    films.append(PolicyFilm(6, "6", 2000, countries=(russia,)))
    films.extend(
        PolicyFilm(film_id, str(film_id), 2000, countries=(other,))
        for film_id in range(7, 13)
    )
    catalog = _catalog(films)
    ratings = [
        *(RatedInteraction(film_id, 4.5) for film_id in range(1, 6)),
        RatedInteraction(6, 5.0),
        *(RatedInteraction(film_id, 2.0) for film_id in range(7, 13)),
    ]
    profile = build_user_category_profile(
        7,
        RecommendationHistory(tuple(range(1, 13)), tuple(ratings)),
        catalog,
    )
    by_name = {value.name: value for value in profile.preferences["country"]}

    assert by_name["France"].raw_preference == 4.5 - profile.user_mean_rating
    assert by_name["France"].confidence > by_name["Russia"].confidence
    assert by_name["France"].affinity > by_name["Russia"].affinity
    assert "Russia" not in {
        value.name for value in qualifying_preferences(profile, "country")
    }


def test_top_picks_and_hidden_gems_apply_exact_semantics() -> None:
    films = [PolicyFilm(film_id, str(film_id), 2000) for film_id in range(1, 31)]
    ranked = tuple(
        _ranked(
            film_id,
            film_id,
            stratum="HEAD" if film_id <= 5 else "MID" if film_id <= 25 else "TAIL",
            svd_rank=film_id * 10,
            popularity_rank=film_id if film_id <= 5 else None,
        )
        for film_id in range(1, 31)
    )
    config = replace(DEFAULT_POLICY_CONFIG, hidden_minimum=3)
    result = build_category_proposals(
        ranked,
        _profile(),
        _catalog(films),
        np.eye(30, dtype=np.float32),
        {film_id: film_id - 1 for film_id in range(1, 31)},
        config=config,
    )
    by_key = {proposal.key: proposal for proposal in result.proposals}

    assert by_key["top_picks"].ordered_candidate_ids[:20] == tuple(range(1, 21))
    hidden = by_key["hidden_gems"]
    assert all(film_id > 5 for film_id in hidden.ordered_candidate_ids)
    assert hidden.reasons[6].code == RecommendationReasonCode.NON_HEAD_TASTE_MATCH


def test_brazil_world_outside_and_classic_eligibility_are_constrained() -> None:
    brazil = _entity(1, "Brazil")
    japan = _entity(2, "Japan")
    usa = _entity(3, "USA")
    portuguese = _entity(10, "Portuguese")
    japanese = _entity(11, "Japanese")
    english = _entity(12, "English")
    familiar_genre = _entity(20, "Drama")
    unfamiliar_genre = _entity(21, "Horror")
    films = []
    for film_id in range(1, 31):
        films.append(
            PolicyFilm(
                film_id,
                str(film_id),
                1960 if film_id <= 10 else 2000,
                genres=(familiar_genre if film_id <= 15 else unfamiliar_genre,),
                countries=(
                    brazil if film_id <= 10 else japan if film_id <= 20 else usa,
                ),
                languages=(
                    portuguese
                    if film_id <= 10
                    else japanese
                    if film_id <= 20
                    else english,
                ),
            )
        )
    ranked = tuple(
        _ranked(
            film_id,
            film_id,
            stratum="MID" if film_id <= 25 else "HEAD",
            svd_rank=film_id,
            popularity_rank=film_id if film_id == 30 else None,
        )
        for film_id in range(1, 31)
    )
    preferences = {
        "genre": (_preference("genre", 20, "Drama"),),
        "country": (_preference("country", 2, "Japan"),),
        "language": (_preference("language", 11, "Japanese"),),
    }
    config = replace(
        DEFAULT_POLICY_CONFIG,
        brazilian_minimum=3,
        world_minimum=3,
        world_country_cap=5,
        classic_minimum=3,
        outside_minimum=3,
        outside_exclude_hidden_neighborhood=False,
    )
    result = build_category_proposals(
        ranked,
        _profile(preferences=preferences),
        _catalog(films),
        np.eye(30, dtype=np.float32),
        {film_id: film_id - 1 for film_id in range(1, 31)},
        config=config,
    )
    by_key = {proposal.key: proposal for proposal in result.proposals}

    assert set(by_key["brazilian_cinema"].ordered_candidate_ids) == set(range(1, 11))
    assert len(by_key["world_cinema"].ordered_candidate_ids) >= 10
    world_country_counts = {}
    catalog = _catalog(films)
    for film_id in by_key["world_cinema"].ordered_candidate_ids:
        country_id = catalog.films[film_id].countries[0].id
        world_country_counts[country_id] = world_country_counts.get(country_id, 0) + 1
    assert max(world_country_counts.values()) <= config.world_country_cap
    assert set(by_key["classic_cinema"].ordered_candidate_ids) == set(range(1, 11))
    outside = by_key["outside_usual"].ordered_candidate_ids
    assert outside
    assert all(film_id > 15 for film_id in outside)
    assert 30 not in outside
    assert all(
        candidate.popularity_stratum != "HEAD"
        or (
            candidate.retrieved_by_svd
            and not candidate.retrieved_by_popularity
            and candidate.svd_rank <= config.outside_lower_head_svd_rank
        )
        for candidate in (ranked[film_id - 1] for film_id in outside)
    )
    assert by_key["outside_usual"].policy_metadata["maximum_head_count"] == 4


def test_anchor_selection_prefers_lower_top_picks_overlap_after_equal_quality() -> None:
    films = [PolicyFilm(film_id, str(film_id), 2000) for film_id in range(1, 43)]
    ranked = tuple(
        _ranked(film_id, film_id, svd_rank=film_id) for film_id in range(3, 43)
    )
    vectors = np.zeros((42, 2), dtype=np.float32)
    vectors[0] = [1, 0]
    vectors[1] = [0, 1]
    vectors[2:22] = [1, 0]
    vectors[22:42] = [0, 1]
    profile = _profile(
        anchors=(
            AnchorPreference(1, "Top-overlap anchor", 5.0),
            AnchorPreference(2, "Novel anchor", 5.0),
        )
    )
    config = replace(DEFAULT_POLICY_CONFIG, anchor_minimum=8)
    result = build_category_proposals(
        ranked,
        profile,
        _catalog(films),
        vectors,
        {film_id: film_id - 1 for film_id in range(1, 43)},
        config=config,
    )
    anchor = next(
        value for value in result.proposals if value.key == "because_you_liked"
    )

    assert anchor.policy_metadata["anchor_film_id"] == 2
    assert anchor.ordered_candidate_ids[:3] == (23, 24, 25)
    assert anchor.reasons[23].anchor_title == "Novel anchor"


def test_director_pool_is_top_15_and_limits_each_director() -> None:
    directors = tuple(
        _preference(
            "director", entity_id, f"D{entity_id}", support=4, affinity=1 / entity_id
        )
        for entity_id in range(1, 17)
    )
    films = [
        PolicyFilm(
            film_id,
            str(film_id),
            2000,
            directors=(_entity((film_id - 1) // 4 + 1, f"D{(film_id - 1) // 4 + 1}"),),
        )
        for film_id in range(1, 65)
    ]
    ranked = tuple(
        _ranked(film_id, film_id, svd_rank=film_id) for film_id in range(1, 65)
    )
    config = replace(DEFAULT_POLICY_CONFIG, directors_minimum=8)
    result = build_category_proposals(
        ranked,
        _profile(preferences={"director": directors}),
        _catalog(films),
        np.eye(64, dtype=np.float32),
        {film_id: film_id - 1 for film_id in range(1, 65)},
        config=config,
    )
    proposal = next(
        value for value in result.proposals if value.key == "directors_you_love"
    )
    counts = {}
    for film_id in proposal.ordered_candidate_ids:
        director_id = (film_id - 1) // 4 + 1
        counts[director_id] = counts.get(director_id, 0) + 1

    assert proposal.policy_metadata["directors"][-1]["name"] == "D15"
    assert 16 not in counts
    assert max(counts.values()) == DEFAULT_POLICY_CONFIG.director_film_cap


def test_favorite_genre_and_decade_select_strongest_supported_entities() -> None:
    drama = _entity(1, "Drama")
    comedy = _entity(2, "Comedy")
    films = [
        PolicyFilm(
            film_id,
            str(film_id),
            1995 if film_id <= 15 else 2005,
            genres=(drama if film_id <= 15 else comedy,),
        )
        for film_id in range(1, 31)
    ]
    preferences = {
        "genre": (
            _preference("genre", 1, "Drama", affinity=0.4),
            _preference("genre", 2, "Comedy", affinity=0.2),
        ),
        "decade": (
            _preference("decade", 1990, "1990s", affinity=0.3),
            _preference("decade", 2000, "2000s", affinity=0.1),
        ),
    }
    result = build_category_proposals(
        tuple(_ranked(film_id, film_id, svd_rank=film_id) for film_id in range(1, 31)),
        _profile(preferences=preferences),
        _catalog(films),
        np.eye(30, dtype=np.float32),
        {film_id: film_id - 1 for film_id in range(1, 31)},
    )
    by_key = {proposal.key: proposal for proposal in result.proposals}

    assert by_key["favorite_genre"].title_parameters == {"entity": "Drama"}
    assert by_key["favorite_genre"].ordered_candidate_ids == tuple(range(1, 16))
    assert by_key["favorite_decade"].title_parameters == {"entity": "1990s"}
    assert by_key["favorite_decade"].ordered_candidate_ids == tuple(range(1, 16))


def test_anchor_uses_documented_four_star_fallback_only_when_needed() -> None:
    films = [PolicyFilm(film_id, str(film_id), 2000) for film_id in range(1, 12)]
    ranked = tuple(
        _ranked(film_id, film_id, svd_rank=film_id) for film_id in range(2, 12)
    )
    vectors = np.ones((11, 2), dtype=np.float32)
    result = build_category_proposals(
        ranked,
        _profile(anchors=(AnchorPreference(1, "Fallback", 4.0),)),
        _catalog(films),
        vectors,
        {film_id: film_id - 1 for film_id in range(1, 12)},
    )
    proposal = next(
        value for value in result.proposals if value.key == "because_you_liked"
    )

    assert proposal.evidence_tier == "minimum"
    assert proposal.policy_metadata["anchor_rating"] == 4.0


def test_anchor_v1_1_uses_exact_top_100_inventory_neighborhood() -> None:
    films = [PolicyFilm(film_id, str(film_id), 2000) for film_id in range(1, 152)]
    ranked = tuple(
        _ranked(film_id, film_id - 1, svd_rank=film_id - 1) for film_id in range(2, 152)
    )
    similarities = np.linspace(0.99, 0.01, 150, dtype=np.float32)
    vectors = np.zeros((151, 2), dtype=np.float32)
    vectors[0] = [1.0, 0.0]
    vectors[1:, 0] = similarities
    vectors[1:, 1] = np.sqrt(1 - similarities**2)

    result = build_category_proposals(
        ranked,
        _profile(anchors=(AnchorPreference(1, "Anchor", 5.0),)),
        _catalog(films),
        vectors,
        {film_id: film_id - 1 for film_id in range(1, 152)},
    )
    proposal = next(
        value for value in result.proposals if value.key == "because_you_liked"
    )

    assert len(proposal.ordered_candidate_ids) == 100
    assert proposal.ordered_candidate_ids == tuple(range(2, 102))
    assert proposal.policy_metadata["neighborhood_rule"] == "top_100"
    assert proposal.policy_metadata["usable_neighbor_count"] == 100


def test_maximum_rating_anchor_pruning_is_exactly_equivalent_to_full_evaluation() -> (
    None
):
    rng = np.random.default_rng(42)
    raw_vectors = rng.normal(size=(124, 8)).astype(np.float32)
    vectors = raw_vectors / np.linalg.norm(raw_vectors, axis=1, keepdims=True)
    films = [PolicyFilm(film_id, str(film_id), 2000) for film_id in range(1, 125)]
    ranked = tuple(
        _ranked(film_id, film_id - 4, svd_rank=film_id - 4) for film_id in range(5, 125)
    )
    catalog = _catalog(films)
    positions = {film_id: film_id - 1 for film_id in range(1, 125)}
    scenarios = (
        (5.0, 4.5, 5.0, 4.5),
        (4.5, 4.5, 4.5, 4.5),
        (4.0, 4.0, 4.0, 4.0),
        (5.0, 5.0, 5.0, 5.0),
    )

    for ratings in scenarios:
        profile = _profile(
            anchors=tuple(
                AnchorPreference(film_id, f"Anchor {film_id}", rating)
                for film_id, rating in enumerate(ratings, start=1)
            )
        )
        optimized = _because_you_liked(
            ranked,
            profile,
            catalog,
            vectors,
            positions,
            DEFAULT_POLICY_CONFIG,
        )
        exhaustive = _because_you_liked(
            ranked,
            profile,
            catalog,
            vectors,
            positions,
            DEFAULT_POLICY_CONFIG,
            maximum_rating_only=False,
        )

        assert optimized == exhaustive


def test_balanced_category_hard_caps_head_while_filling_with_mid_tail() -> None:
    films = [PolicyFilm(film_id, str(film_id), 2000) for film_id in range(1, 61)]
    reason = RecommendationReason(RecommendationReasonCode.GLOBAL_RRF)
    top = CategoryProposal(
        "top_picks",
        "general",
        CategoryRole.GENERAL,
        "Top",
        {},
        "strong",
        20,
        tuple(range(41, 61)),
        1,
        20,
        {film_id: reason for film_id in range(41, 61)},
    )
    balanced = CategoryProposal(
        "classic_cinema",
        "classic",
        CategoryRole.CULTURAL,
        "Classic",
        {},
        "strong",
        40,
        tuple(range(1, 41)),
        8,
        20,
        {film_id: reason for film_id in range(1, 41)},
        {
            "maximum_head_count": 12,
            "head_candidate_ids": frozenset(range(1, 21)),
        },
    )

    allocated, _ = allocate_categories((top, balanced), _profile(), _catalog(films))
    selected = allocated[1].film_ids

    assert len(selected) == 20
    assert sum(film_id <= 20 for film_id in selected) == 12
    assert selected == (*range(1, 13), *range(21, 29))


def test_outside_v1_1_uses_viable_deeper_pool_outside_hidden_neighborhood() -> None:
    drama = _entity(1, "Drama")
    horror = _entity(2, "Horror")
    usa = _entity(10, "USA")
    japan = _entity(11, "Japan")
    films = [
        PolicyFilm(
            film_id,
            str(film_id),
            2000,
            genres=(horror,),
            countries=(japan,),
        )
        for film_id in range(1, 1001)
    ]
    ranked = tuple(
        _ranked(film_id, film_id, stratum="TAIL", svd_rank=film_id)
        for film_id in range(1, 1001)
    )
    profile = _profile(
        preferences={
            "genre": (_preference("genre", drama.id, drama.name),),
            "country": (_preference("country", usa.id, usa.name),),
        }
    )
    result = build_category_proposals(
        ranked,
        profile,
        _catalog(films),
        np.eye(1, dtype=np.float32),
        {},
    )
    by_key = {value.key: value for value in result.proposals}

    assert by_key["hidden_gems"].ordered_candidate_ids[:20] == tuple(range(1, 21))
    assert by_key["outside_usual"].ordered_candidate_ids[:20] == tuple(range(251, 271))
    assert set(by_key["hidden_gems"].ordered_candidate_ids).isdisjoint(
        by_key["outside_usual"].ordered_candidate_ids
    )


def test_policy_catalog_interns_repeated_relation_entities() -> None:
    session = type("Session", (), {})()
    session.execute = AsyncMock(
        side_effect=[
            [(1, "One", 2000), (2, "Two", 2001)],
            [(1, 7, "Director"), (2, 7, "Director")],
            [],
            [],
            [],
        ]
    )

    catalog = asyncio.run(load_policy_catalog(session, (1, 2)))

    assert catalog.films[1].directors[0] is catalog.films[2].directors[0]
    assert session.execute.await_count == 5


def test_v1_baseline_is_retained_only_for_explicit_comparison() -> None:
    assert DEFAULT_POLICY_CONFIG.anchor_neighborhood_limit == 100
    assert DEFAULT_POLICY_CONFIG.classic_head_cap == 12
    assert DEFAULT_POLICY_CONFIG.world_head_cap == 12
    assert DEFAULT_POLICY_CONFIG.outside_exclude_hidden_neighborhood
    assert not DEFAULT_POLICY_CONFIG.outside_exclude_head
    assert DEFAULT_POLICY_CONFIG.outside_lower_head_cap == 4
    assert DEFAULT_POLICY_CONFIG.outside_svd_rank == 750
    assert V1_POLICY_CONFIG.anchor_neighborhood_limit is None
    assert V1_POLICY_CONFIG.classic_head_cap is None
    assert V1_POLICY_CONFIG.world_head_cap is None
    assert not V1_POLICY_CONFIG.outside_exclude_hidden_neighborhood
    assert V1_POLICY_CONFIG.outside_exclude_head
    assert V1_POLICY_CONFIG.outside_lower_head_cap == 0
    assert not V1_POLICY_CONFIG.outside_require_primary_viability


def test_allocation_reserves_top_ten_allows_two_appearances_and_relaxes_soft_caps() -> (
    None
):
    same_genre = _entity(1, "Drama")
    films = [
        PolicyFilm(
            film_id,
            str(film_id),
            1995,
            directors=(_entity(film_id, f"D{film_id}"),),
            genres=(same_genre,),
        )
        for film_id in range(1, 31)
    ]
    reason = RecommendationReason(RecommendationReasonCode.GLOBAL_RRF)
    top = CategoryProposal(
        "top_picks",
        "general",
        CategoryRole.GENERAL,
        "Top Picks",
        {},
        "strong",
        30,
        tuple(range(1, 31)),
        1,
        20,
        {film_id: reason for film_id in range(1, 31)},
    )
    focused = CategoryProposal(
        "favorite_genre",
        "genre",
        CategoryRole.PERSONALIZED,
        "{entity}",
        {"entity": "Drama"},
        "strong",
        10,
        tuple(range(1, 31)),
        20,
        20,
        {film_id: reason for film_id in range(1, 31)},
    )
    allocated, diagnostics = allocate_categories(
        (top, focused), _profile(), _catalog(films)
    )
    focused_ids = allocated[1].film_ids

    assert set(range(1, 11)).isdisjoint(focused_ids)
    assert set(range(11, 21)).issubset(focused_ids)
    assert diagnostics["maximum_appearances"] == 2
    assert len(focused_ids) == 20


def test_generic_minimum_omission_does_not_relax_semantics() -> None:
    films = [PolicyFilm(film_id, str(film_id), 2000) for film_id in range(1, 25)]
    reason = RecommendationReason(RecommendationReasonCode.GLOBAL_RRF)
    top = CategoryProposal(
        "top_picks",
        "general",
        CategoryRole.GENERAL,
        "Top Picks",
        {},
        "strong",
        24,
        tuple(range(1, 25)),
        1,
        20,
        {film_id: reason for film_id in range(1, 25)},
    )
    hidden = CategoryProposal(
        "hidden_gems",
        "discovery",
        CategoryRole.DISCOVERY,
        "Hidden",
        {},
        "minimum",
        3,
        (21, 22, 23),
        4,
        20,
        {film_id: reason for film_id in (21, 22, 23)},
    )
    allocated, _ = allocate_categories((top, hidden), _profile(), _catalog(films))

    assert [value.proposal.key for value in allocated] == ["top_picks"]


def test_established_proposal_order_prefers_personalized_and_omits_weak_overlap() -> (
    None
):
    films = [PolicyFilm(film_id, str(film_id), 2000) for film_id in range(1, 61)]
    reason = RecommendationReason(RecommendationReasonCode.GLOBAL_RRF)
    top = CategoryProposal(
        "top_picks",
        "general",
        CategoryRole.GENERAL,
        "Top Picks",
        {},
        "strong",
        60,
        tuple(range(1, 61)),
        1,
        20,
        {film_id: reason for film_id in range(1, 61)},
    )
    weak_overlap = CategoryProposal(
        "brazilian_cinema",
        "cultural",
        CategoryRole.CULTURAL,
        "Brazilian",
        {},
        "minimum",
        20,
        tuple(range(1, 21)),
        8,
        20,
        {film_id: reason for film_id in range(1, 21)},
    )
    personalized = CategoryProposal(
        "favorite_genre",
        "genre",
        CategoryRole.PERSONALIZED,
        "Drama",
        {},
        "strong",
        8,
        tuple(range(21, 41)),
        12,
        20,
        {film_id: reason for film_id in range(21, 41)},
    )
    config = replace(DEFAULT_POLICY_CONFIG, maximum_categories=2)

    first, _ = allocate_categories(
        (top, weak_overlap, personalized),
        _profile(),
        _catalog(films),
        config=config,
    )
    second, _ = allocate_categories(
        (top, weak_overlap, personalized),
        _profile(),
        _catalog(films),
        config=config,
    )

    assert [value.proposal.key for value in first] == ["top_picks", "favorite_genre"]
    assert first == second


def test_structured_reason_adds_source_agreement_without_raw_scores() -> None:
    candidate = RecommendationCandidate(
        10,
        svd_score=0.75,
        svd_rank=2,
        popularity_score=100,
        popularity_rank=3,
        retrieved_by_svd=True,
        retrieved_by_popularity=True,
    )
    ranked = rank_candidates_by_rrf((candidate,), {10: 3}, 100)
    proposal = build_category_proposals(
        ranked,
        _profile(),
        _catalog([PolicyFilm(10, "Film", 2000)]),
        np.ones((1, 1), dtype=np.float32),
        {10: 0},
    ).proposals[0]
    reason = proposal.reasons[10]

    assert reason.code == RecommendationReasonCode.GLOBAL_RRF
    assert reason.additional_codes == (RecommendationReasonCode.SOURCE_AGREEMENT,)
    assert not hasattr(reason, "svd_score")
    assert not hasattr(reason, "rrf_score")
