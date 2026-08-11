"""Final source-depth policy for the future pre-ranker candidate inventory."""

FINAL_CANDIDATE_NOMINAL_BUDGET = 4000
FINAL_WEIGHTED_SVD_DEPTH = 2000
FINAL_POPULARITY_DEPTH = 2000

if FINAL_WEIGHTED_SVD_DEPTH + FINAL_POPULARITY_DEPTH != FINAL_CANDIDATE_NOMINAL_BUDGET:
    raise RuntimeError("final candidate source depths must sum to the nominal budget")
