# Music-CRS 2026 OpenRec LightGBM ranking experiment

The experiment trains OpenRec's `LightGBMRankModel` on 40,000 OOF requests
and uses 12,000 disjoint reranker-validation requests for early stopping and
feature-set selection. Official dev labels are read only by the final
organizer evaluator.

The rank input contract is domain-neutral. It consumes candidate IDs, generic
entity and tag attributes, item/query vectors, request history, timestamps,
and named recall-channel evidence. The Music-CRS adapter maps track metadata
onto that contract and adds no competition-category rules or label-derived
features.

| Method | nDCG@1 | nDCG@10 | nDCG@20 | Recall@20 | Candidate recall | Catalog diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenRec fixed rank, query + content I2I | 0.06650 | 0.15310 | 0.17425 | 0.34950 | — | 0.50271 |
| OpenRec LightGBM v1 | **0.07350** | **0.16177** | **0.18362** | **0.36225** | 0.60175 @300 | 0.47809 |
| Champion clean XGB | 0.16625 | 0.29367 | 0.32371 | 0.57238 | 0.82075 @200 | 0.53292 |

LightGBM improves nDCG@20 by `0.00937` absolute (`5.38%`) over the strongest
previous OpenRec fixed-rank run. Its mean rank among top-20 hits is `6.57`
and its median hit rank is `5`.

The remaining nDCG@20 gap to the clean champion is `0.14009` absolute. The
candidate pools are still materially different: OpenRec reaches `0.60175`
recall at 300 candidates while the champion reaches `0.82075` at 200. This
experiment therefore does not support attributing the remaining gap to
LightGBM versus XGBoost. The champion also uses many more recall channels,
per-channel calibration, audio/image/text towers, and a much broader generic
candidate feature table.

## Validation decisions

- `lambdarank` validation nDCG@20: `0.38973`
- `rank_xendcg` validation nDCG@20: `0.38276` in the objective ablation
- base 33-feature set validation nDCG@20: `0.38973`
- expanded 55-feature fusion set validation nDCG@20: `0.38874`
- selected model: `lambdarank`, base feature set, iteration `118`

The expanded fusion features remain available in the generic feature builder,
but the experiment artifact records and serves only the validation-selected
33 features.

Artifacts are generated under `runs/music-crs-2026-openrec-lightgbm-v4/` and
are intentionally excluded from Git. The tracked configuration is
`studies/baselines/music-crs-2026-openrec-lightgbm-v4.json`.
