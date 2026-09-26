# Music-CRS 2026 champion recall ablation

## Protocol

This experiment reuses the clean champion XGB score for all 1,600,000 dev
candidates. Each ablation changes only candidate eligibility according to the
14 recall-channel membership columns. The ranker is therefore evaluated once;
no model is retrained and no score is recomputed using dev labels.

The official 8,000-turn devset remains the evaluation set. The baseline
reproduces the clean champion exactly: nDCG@20 `0.32371267`, Recall@20
`0.572375`, and candidate Recall@200 `0.82075`.

`single` retains candidates supplied by one channel. `LOO` removes one channel
and retains candidates supplied by any other channel. Raw recall is measured
directly on each channel's original top-200 output. Missing candidates receive
zero gain rather than being backfilled from another route.

This is a frozen-score candidate-support ablation. It measures the marginal
value of each recall route under the deployed champion ranker. It does not
measure how much a retrained ranker could compensate for a removed channel.

## Leave-one-channel-out and single-channel results

| Channel | Single nDCG@20 | Raw Recall@200 | Exclusive Recall@200 | LOO nDCG@20 loss |
| --- | ---: | ---: | ---: | ---: |
| `tower_a_oneshot` | **0.239255** | **0.590750** | **0.018750** | **0.009569** |
| `query_full_multibehav_alltext_softor_session` | 0.233973 | 0.565500 | 0.003875 | 0.005982 |
| `query_full_multibehav_alltext_softor_session_ndcg` | 0.226673 | 0.551000 | 0.005625 | 0.004940 |
| `split_*_noiserobust_session` | 0.220975 | 0.532125 | 0.004500 | 0.003823 |
| `split_*_nova_session_ndcg` | 0.217087 | 0.524375 | 0.004625 | 0.003703 |
| `split_*_dif_session_ndcg` | 0.220738 | 0.531250 | 0.005250 | 0.003378 |
| `bm25_colisten_oneshot` | 0.224447 | 0.522000 | 0.005000 | 0.003076 |
| `emb_item_knn_8b_session_dro` | 0.207012 | 0.516625 | 0.001125 | 0.002882 |
| `split_hidim_xattn_hardneg_session_ndcg` | 0.220359 | 0.528000 | 0.004250 | 0.002668 |
| `heuristic_v2_hybrid_session_dro` | 0.180771 | 0.390000 | 0.001625 | 0.002258 |
| `hybrid_all_qwen_session_dro` | 0.228914 | 0.571125 | 0.000875 | 0.001727 |
| `tower_ensemble_session_dro` | 0.189922 | 0.427625 | 0.000250 | 0.000375 |
| `heuristic_v3_session_dro` | 0.228405 | 0.588500 | 0.000000 | 0.000203 |
| `tower_cf_ensemble_session_dro` | 0.189831 | 0.428500 | 0.000000 | 0.000000 |

`tower_a_oneshot` is the Qwen 8B query-to-track text tower. It is the strongest
channel by single-channel score, exclusive target coverage, and removal loss.

## Greedy cumulative result

| Routes | Newly added channel | nDCG@20 | Recall@20 | Fraction of full nDCG@20 |
| ---: | --- | ---: | ---: | ---: |
| 1 | `tower_a_oneshot` | 0.239255 | 0.452500 | 73.91% |
| 2 | `query_full_multibehav_alltext_softor_session` | 0.278181 | 0.516250 | 85.94% |
| 3 | `query_full_multibehav_alltext_softor_session_ndcg` | 0.289367 | 0.530375 | 89.39% |
| 4 | `split_*_noiserobust_session` | 0.297243 | 0.540375 | 91.82% |
| 5 | `split_*_dif_session_ndcg` | 0.303168 | 0.547125 | 93.65% |
| 6 | `split_*_nova_session_ndcg` | 0.308012 | 0.552375 | 95.15% |
| 7 | `bm25_colisten_oneshot` | 0.312165 | 0.558875 | 96.43% |
| 8 | `emb_item_knn_8b_session_dro` | 0.315762 | 0.563125 | 97.54% |
| 11 | through `hybrid_all_qwen_session_dro` | 0.322675 | 0.570875 | 99.68% |
| 13 | through `heuristic_v3_session_dro` | 0.323713 | 0.572375 | 100.00% |

The final `tower_cf_ensemble_session_dro` route adds no candidate-level value
under the frozen score. The last two ensemble/heuristic routes together add
only `0.00104` nDCG@20 after the first eleven routes.

## OpenRec implementation priority

1. Improve the generic query-to-entity embedding recall contract and query
   construction. OpenRec already has this route, but the champion's equivalent
   channel alone reaches `0.239255` nDCG@20 under the frozen XGB score.
2. Add a generic BM25/full-text recall provider. Its removal costs `0.003076`
   and it contributes lexical candidates that dense towers miss.
3. Add embedding ItemKNN as a first-class generic recall route. Its removal
   costs `0.002882`; it maps naturally to OpenRec's existing I2I abstraction.
4. Generalize request-aware sequential towers that combine query, prior
   interactions, and entity metadata. The two query-full channels provide the
   second and third largest removal losses.
5. Treat the several `split_hidim` variants as one model family initially.
   They are individually useful but expensive and mutually correlated.
6. Defer the CF tower ensemble and late heuristic ensemble until the simpler
   generic routes are present; their frozen marginal gains are near zero.

The complete machine-readable results include every cutoff, all 91 pairwise
channel overlaps, per-channel candidate share, and every greedy step:

- `music-crs-2026-champion-recall-ablation-v1.json`
- `music-crs-2026-champion-recall-ablation-channels-v1.csv`
- `music-crs-2026-champion-recall-ablation-cumulative-v1.csv`
