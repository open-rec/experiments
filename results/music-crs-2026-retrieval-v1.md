# Music-CRS 2026: local OpenRec retrieval study

Run: `runs/music-crs-2026/retrieval-v4` (ignored artifact), 2026-09-22.
Companion CSV: `music-crs-2026-retrieval-v1.csv`.
Prepared SHA-256: `1151aebd7ad86485fb247725509133a105534657a40494f09fec2b8d32998307`.
This is a single deterministic local run, not an official Challenge submission.

The adapter used the organizer's TalkPlayData-Challenge train/test conversations
and 47.1k-track catalog. It extracted recorded `music` turns, retaining
session dates and order. The training set was the earliest 80% of train dates
(97,264 music turns; 37,545 distinct training tracks). Later train dates are
local validation; the provided test split is local test. The model contains
only training events. Each query uses at most five earlier music turns in the
same session as observed triggers. The study tests 5,000 evenly spaced eligible
queries per split. Previously played tracks are excluded from recommendations.

| Split | Method | Recall@20 | MRR@20 | Queries |
| --- | --- | ---: | ---: | ---: |
| Validation | OpenRec Hot | 0.0080 | 0.00179 | 5,000 |
| Validation | OpenRec ItemBasedI2I, then Hot | 0.2282 | 0.06723 | 5,000 |
| Test | OpenRec Hot | 0.0050 | 0.00106 | 5,000 |
| Test | OpenRec ItemBasedI2I, then Hot | 0.1334 | 0.03629 | 5,000 |

The test gain over Hot is 0.1284 absolute Recall@20 on this proxy task.
The recorded track is the previous system's output, not an explicit listener
preference label. The pipeline does not read dialog text, produce response text,
or use challenge embeddings. Its full training-item catalog evaluation and
metrics are not comparable with the official Music-CRS nDCG/composite score.
The generated conversation dates are used only for a deterministic local split.

## Official baseline reference

The [organizer's evaluator](https://github.com/nlp4musa/music-crs-evaluator)
reports the following **development-set** baselines. A machine-readable copy
is in `datasets/music_crs_2026/official_baselines.csv`.

| Method | nDCG@20 | Catalog diversity | Lexical diversity |
| --- | ---: | ---: | ---: |
| Random | 0.0001 | 0.9652 | 0.0000 |
| Popularity | 0.0024 | 0.0004 | 0.0000 |
| LLaMA-1B + BM25 | 0.0815 | 0.3795 | 0.2558 |

These are organizer-published reference numbers, not OpenRec runs. Their
ground truth, split, and nDCG/diversity metrics differ from the local proxy
task above. No gain against the official baseline is claimed.
