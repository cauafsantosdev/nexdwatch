# Local data and artifacts

Everything under `data/` is ignored by Git except this inventory. The directory may
contain private inputs, generated production resources, and reproducible research
outputs at the same time. Paths remain unchanged because several are compatibility
contracts.

| Path | Classification | Purpose and retention |
| --- | --- | --- |
| `films_data.csv` | SOURCE INPUT | Historical catalog bootstrap input; external and not distributed |
| `users_data.csv` | SOURCE INPUT | Controlled historical interaction cohort; external and not distributed |
| `ratings.csv`, `watched.csv` | LOCAL/PRIVATE EXPORT | Loose Letterboxd export material; verify ownership before archiving or deleting |
| `item_embeddings.npy` | GENERATED PRODUCTION ARTIFACT | Legacy-flat normalized SVD vectors retained for compatibility |
| `film_index.json` | GENERATED PRODUCTION ARTIFACT | Legacy-flat actual film-ID mapping |
| `retrieval.faiss` | GENERATED PRODUCTION ARTIFACT | Legacy-flat exact FAISS index |
| `candidates/popularity.json` | GENERATED PRODUCTION/RESEARCH ARTIFACT | Flat controlled-popularity artifact; source metadata identifies its universe |
| `models/current.json` | GENERATED PRODUCTION ARTIFACT | Atomic authoritative version pointer |
| `models/<version>/` | GENERATED PRODUCTION ARTIFACT | Immutable validated SVD/FAISS/popularity/manifest bundles |
| `analysis/*.json` | HISTORICAL EVIDENCE | Compact machine-readable candidate, policy, and performance reports |
| `ncf/` | GENERATED RESEARCH ARTIFACT | Rejected neural model, vectors, index, and metadata |
| `ranker/` | DISPOSABLE REPRODUCIBLE OUTPUT | Sampled-v1 fold matrices/models; locally about 2.7 GB at audit time |
| `ranker_full_pool_v2/` | DISPOSABLE REPRODUCIBLE OUTPUT | Corrected full-pool matrices/models; locally about 8.8 GB at audit time |

## Production versus research popularity

Scheduled production training derives popularity from the same current PostgreSQL
rated-interaction snapshot used for SVD. Ratings `>= 3.5` are counted, then films are
ordered by count descending and actual `Film.id` ascending.

Research commands derive popularity from the frozen historical `users_data.csv`
cohort. Artifact metadata records the source, and model-bundle validation rejects a
research-sourced popularity file in a production versioned bundle.

## Git and Docker policy

Git does not track datasets, model files, export files, analysis JSON, checkpoints,
or matrices. This prevents large or private data from entering repository history.
Selected legacy-flat artifacts are admitted to the Docker build context by
`.dockerignore` for current compatibility; changing that packaging contract requires
a separate deployment review.

Compose bind-mounts the repository for the current single-VPS layout. A fresh clone
therefore needs externally supplied source data or prebuilt compatible artifacts
before the real API can start.

## Future archive/deletion candidates

No data is deleted by the repository cleanup. Before reclaiming space:

1. verify the protocol in each manifest;
2. preserve compact summaries and human-readable conclusions;
3. confirm the expensive run is reproducible from retained code and source data;
4. verify local export files are no longer needed;
5. delete only explicit resolved paths, never the entire artifact root.

The strongest candidates are sampled/full-pool fold matrices and model files. NCF
outputs and local Letterboxd export CSVs also appear removable after the checks above.
