# Music-CRS 2026: official devset recommendation metrics

The [organizer's evaluator](https://github.com/nlp4musa/music-crs-evaluator) defines a session-turn recommendation task. We evaluated all 8,000 official devset turns with 20 unique catalog tracks per turn. OpenRec Hot and ItemBasedI2I were fitted on the 15,199 public training sessions. The evaluator's `compute_recsys_metrics`, `compute_catalog_diversity`, and `compute_lexical_diversity` functions scored the predictions. Re-scoring the organizer's published Popularity predictions reproduced its unrounded published scores exactly (absolute tolerance 1e-12).

| Method | nDCG@1 | nDCG@10 | nDCG@20 | Catalog diversity | Lexical diversity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random (official) | 0.0000 | 0.0001 | 0.0001 | 0.9652 | 0.0000 |
| Popularity (official) | 0.0005 | 0.0018 | 0.0024 | 0.0004 | 0.0000 |
| LLaMA-1B + BM25 (official) | 0.0098 | 0.0627 | 0.0815 | 0.3795 | 0.2558 |
| OpenRec Hot + ItemBasedI2I | **0.0121** | 0.0429 | 0.0522 | **0.4055** | 0.0000 |

OpenRec exceeds the published Popularity baseline on nDCG@20 and falls below LLaMA-1B + BM25 on that metric. Responses are empty, so lexical diversity is zero; this run does not measure conversational response quality or the LLM-judge portion of the challenge.

Reproduce (using the official evaluator checkout and data paths in the config):

```bash
.venv/bin/python -m openrec_experiments.cli official-music \
  --config studies/baselines/music-crs-2026-official-devset.json \
  --evaluator /tmp/openrec-music-crs-evaluator \
  --output runs/music-crs-2026/official-devset-v1
```

The ignored run directory contains predictions, exact scores, data hashes, and the official evaluator commit.
This run used evaluator commit `3dd7455179fe69a0396f6007c89c078824498415`.
