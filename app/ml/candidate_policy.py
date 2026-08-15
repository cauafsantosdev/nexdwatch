"""Freeze the production candidate inventory at equal 2,000-film source depths.

SVD and controlled popularity each contribute up to half of the nominal 4,000-film
budget before stable union deduplication. The assertion prevents configuration drift
between the advertised budget and the depths consumed by serving.
"""

FINAL_CANDIDATE_NOMINAL_BUDGET = 4000
FINAL_WEIGHTED_SVD_DEPTH = 2000
FINAL_POPULARITY_DEPTH = 2000

if FINAL_WEIGHTED_SVD_DEPTH + FINAL_POPULARITY_DEPTH != FINAL_CANDIDATE_NOMINAL_BUDGET:
    raise RuntimeError("final candidate source depths must sum to the nominal budget")
