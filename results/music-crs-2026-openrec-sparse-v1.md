# Music-CRS 2026 OpenRec sparse recall experiment

This controlled experiment adds OpenRec BM25 sparse recall to the existing
LightGBM v4 candidate pool. It keeps the official clean devset, Qwen3 query and
track embeddings, 40,000 training requests, 12,000 validation requests, 300 dev
candidates, seed, rank objectives, and official evaluator unchanged. BM25 uses
generic entity text fields: title 2.0, primary entity 1.5, collection 1.0, and
tags 0.5.

| Method | nDCG@1 | nDCG@10 | nDCG@20 | Recall@20 | Candidate Recall | Catalog diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenRec LightGBM v4 | 0.07350 | 0.16177 | 0.18362 | 0.36225 | 0.60175 @300 | 0.47809 |
| OpenRec LightGBM + sparse | **0.07688** | **0.16935** | **0.19091** | **0.37263** | **0.60900 @300** | **0.47832** |
| Champion clean XGB | 0.16625 | 0.29367 | 0.32371 | 0.57238 | 0.82075 @200 | 0.53292 |

Sparse recall raises candidate Recall@300 by 0.00725 absolute and Recall@20 by
0.01038. nDCG@20 rises by 0.00729 absolute, or 3.97% relative. The validation
selector chooses the 62-feature fusion set at iteration 197; its validation
nDCG@20 is 0.39048, versus 0.38549 for the base feature set.

The remaining nDCG@20 gap to the clean champion is 0.13280. Sparse closes 5.20%
of the previous 0.14009 gap. Candidate recall remains 0.21175 below the champion,
so recall and ranking both remain material sources of the difference.

Artifacts are under `runs/music-crs-2026-openrec-lightgbm-sparse-v1/`. The
tracked configuration is
`studies/baselines/music-crs-2026-openrec-lightgbm-sparse-v1.json`.
