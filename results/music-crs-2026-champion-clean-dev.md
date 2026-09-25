# Music-CRS 2026 champion recipe on the OpenRec dev split

This run retrains the published winner recipe on the same public data used by
the OpenRec experiment. The 15,199 official train sessions are the only source
of labels for candidate generators and the XGBoost reranker. The 1,000 official
dev sessions remain a strict 8,000-turn holdout. Session IDs have zero overlap
between train and dev.

Published candidate-generator, RRF, and XGBoost hyperparameters are frozen.
All model weights and OOF predictions are regenerated. XGBoost trains on the
OOF train pool and uses the disjoint internal reranker-validation pool for
early stopping; official dev labels are read only by the final evaluator.

| Method | nDCG@1 | nDCG@10 | nDCG@20 | Catalog diversity |
| --- | ---: | ---: | ---: | ---: |
| OpenRec Session + Entity + Context | 0.04175 | 0.10931 | 0.12576 | 0.44847 |
| Champion clean XGB | **0.16625** | **0.29367** | **0.32371** | 0.53292 |
| Champion heuristic | 0.05275 | 0.14586 | 0.16957 | 0.52614 |
| Champion final graft | 0.05688 | 0.24630 | 0.27723 | **0.53379** |

The clean champion XGB exceeds OpenRec by 0.19795 absolute nDCG@20, or
157.40% relative (2.574x). At nDCG@1 the gap is 0.12450 absolute (3.982x).

The original submission's final heuristic and graft were tuned for the
Blind-B last-turn distribution. Applying the same rules to every official dev
turn reduces nDCG relative to the clean XGB output, so the clean XGB score is
the meaningful like-for-like champion comparison on this split.

The official evaluator functions score all 8,000 turns. Empty generated
responses make lexical diversity zero for these recommendation-only runs.
The XGBoost model stopped at iteration 186 with internal validation nDCG@20
0.5181. Candidate-level dev recall is 0.57238 at 20 and 0.82075 at 200.

The dev heuristic adapts the published splitK Qwen embeddings by selecting
the exact 8,000 `(session_id, user_id, turn_number)` keys. These embeddings are
unlabelled inputs. Their query template differs from the Blind-B heuristic
template, so heuristic and graft scores are reported separately and do not
replace the strict XGB comparison.
