# Music-CRS 2026 OpenRec query embedding recall

This ablation adds a generic query-to-entity cosine recall channel to the
existing OpenRec Session + Entity + Context system. It uses the frozen Qwen3
8B query and track vectors as request and catalog embeddings. The query channel
retrieves 100 candidates from all 47,071 tracks, excludes prior session tracks,
and contributes cosine similarity with a fixed weight of 1.0. The existing
TF-IDF, transition, artist, tag, and popularity weights are unchanged.

| Method | nDCG@1 | nDCG@10 | nDCG@20 | Catalog diversity |
|---|---:|---:|---:|---:|
| OpenRec Session + Entity + Context | 0.041750 | 0.109315 | 0.125764 | 0.448471 |
| OpenRec + Query Embedding Recall | **0.061625** | **0.144792** | **0.165771** | **0.483270** |
| Absolute change | +0.019875 | +0.035477 | +0.040006 | +0.034798 |
| Relative change | +47.60% | +32.45% | +31.81% | +7.76% |
| Clean champion XGB | 0.166250 | 0.293670 | 0.323710 | 0.532920 |

The query embedding channel closes 20.21% of the previous nDCG@20 gap to the
clean champion (`0.197946` down to `0.157939`). OpenRec now reaches 51.21% of
the champion nDCG@20. The remaining gap shows that candidate generation was a
material limitation, while trained cross-feature ranking and the champion's
additional retrieval channels still account for most of the difference.

The run covers all 8,000 official development turns with the organizer evaluator
at commit `3dd7455179fe69a0396f6007c89c078824498415`. Query-cache metadata includes
ground-truth identifiers for alignment, but OpenRec reads only `session_id`,
`turn_number`, and the embedding matrix; labels do not enter retrieval or scoring.
