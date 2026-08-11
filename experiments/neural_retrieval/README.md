# Inductive neural retrieval experiment

This package preserves the two-tower neural retrieval experiment that informed
NexdWatch's candidate-source decision. It is reproducible research, not a
supported application recommendation backend.

The model encodes a variable-length set of rated film IDs and rating buckets
into a normalized user vector, and encodes catalog film IDs into normalized
candidate vectors. It intentionally has no user-ID embedding, allowing the
experiment to encode users absent from the historical cohort.

The controlled cohort contains 4,300,105 resolved ratings from 1,976 users over
46,990 films. Evaluation uses `exact_holdout_v2`: deterministic per-user
validation/test positives are absent from neural context and temporary SVD
training, popularity uses only positive training interactions (`rating >= 3.5`),
and all systems use the same exact full-catalog targets.

Population mean ± standard deviation over seeds 42, 43, and 44:

| Metric | Popularity | Leakage-free SVD | Inductive neural |
| --- | ---: | ---: | ---: |
| Recall@10 | 0.059851 ± 0.001199 | 0.043744 ± 0.004395 | 0.025263 ± 0.005159 |
| Recall@50 | 0.145303 ± 0.001045 | 0.105120 ± 0.005531 | 0.081214 ± 0.007293 |
| NDCG@10 | 0.034470 ± 0.001073 | 0.024066 ± 0.001911 | 0.012667 ± 0.002288 |
| MRR@10 | 0.026817 ± 0.001394 | 0.018060 ± 0.001522 | 0.008887 ± 0.001501 |

The seed-42 neural artifact covered 2,837 films, or 6.04% of the catalog, at
depth 500. Its MID/TAIL target retrieval was nearly zero, and adding it to the
fixed candidate budget improved seed-42 recall by only about 0.0036 absolute.
Those measured results did not justify production dependency and serving
complexity. This conclusion applies to this architecture, cohort, and training
setup; it is not a general claim about neural recommenders.

Create an experimental environment with the application dependencies and CPU
PyTorch, then run:

```bash
python -m venv .venv-neural
.venv-neural/bin/pip install -r requirements.txt
.venv-neural/bin/pip install -r experiments/neural_retrieval/requirements.txt
.venv-neural/bin/python -m experiments.neural_retrieval train
.venv-neural/bin/python -m experiments.neural_retrieval benchmark --seeds 42,43,44
.venv-neural/bin/python -m experiments.neural_retrieval build-index
```

Local outputs remain under ignored `data/ncf/`. They are experimental artifacts
and are not loaded by FastAPI or Celery startup.
