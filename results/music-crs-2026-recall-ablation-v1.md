# Music-CRS 2026 OpenRec recall-channel ablation

All runs use the same 8,000 official development turns, organizer evaluator,
Session + Entity + Context ranker, and frozen Qwen3 8B query embedding recall.
Each additional channel uses 100 candidates. Additional channel weights were
fixed before reading official dev scores: `0.5` for ItemCF, Content-I2I, UserCF,
and sequence embedding; `0.1` for New. Query embedding remains `1.0`.

| Recall configuration | nDCG@1 | nDCG@10 | nDCG@20 | Catalog diversity | Δ nDCG@20 vs query |
|---|---:|---:|---:|---:|---:|
| Query embedding | 0.061625 | 0.144792 | 0.165771 | 0.483270 | — |
| Query + ItemCF | 0.061125 | 0.142786 | 0.164727 | 0.495039 | -0.001044 |
| **Query + Content-I2I** | **0.066500** | **0.153098** | **0.174250** | 0.502709 | **+0.008479** |
| Query + UserCF | 0.060625 | 0.143079 | 0.164426 | 0.481443 | -0.001344 |
| Query + sequence embedding | 0.061750 | 0.144702 | 0.165582 | 0.483036 | -0.000189 |
| Query + New | 0.061625 | 0.144786 | 0.165736 | 0.483355 | -0.000034 |
| Query + all recall channels | 0.063125 | 0.148854 | 0.170606 | **0.505959** | +0.004836 |
| Clean champion XGB | 0.166250 | 0.293670 | 0.323710 | 0.532920 | — |

Content-I2I is the only individually positive added channel. It improves
nDCG@20 by 5.11% over query-only and 38.56% over the earlier OpenRec
Session + Entity + Context score (`0.125764`). It reaches 53.83% of champion
nDCG@20 while preserving high catalog coverage.

The all-channel run improves query-only nDCG@20 by 2.92% and produces the best
OpenRec catalog diversity, but underperforms Content-I2I alone. This demonstrates
that candidate richness helps coverage while equal fixed fusion lets weaker
history, user, and freshness signals dilute current-query intent. The next
optimization should learn channel weights and gates on the training split,
especially by turn, warm-user availability, and intent continuation versus
intent shift.

Implementation notes:

- ItemCF uses OpenRec `ItemBasedI2I` over session music events.
- Content-I2I uses the same catalog entity TF-IDF representation as the generic
  content rank features, with per-trigger top-100 cosine neighbours cached.
- UserCF uses OpenRec `UserBasedCF`; 371 of 500 test users occur in training.
- Sequence embedding uses OpenRec `EventEmbedding` Word2Vec with 64 dimensions.
- New filters `release_date` at each request's `session_date`, so future tracks
  never enter the candidate set.
- Query cache ground-truth columns are never loaded; only session/turn keys and
  embeddings are used.
