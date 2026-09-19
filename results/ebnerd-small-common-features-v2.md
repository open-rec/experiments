# EB-NeRD-small common-feature model comparison v2

Status: preliminary local temporal holdout; not an official leaderboard result.

This experiment reports Popularity, LR, FM, and LightGBM in the same form as
the KuaiRand result summary. It applies the current strongest causal OpenRec
feature configuration to LR and FM, then compares them with the existing
pruned LightGBM result. Data, temporal boundaries, frozen feedback policy, and
test impressions are shared across all four models. LR, FM, and LightGBM share
the same logical feature configuration; Popularity consumes only training
labels and item identities.

| Model | Seeds | Encoded dimensions | Macro AUC | Macro MRR | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 42 | n/a | 0.56467 | 0.32585 | 0.37074 | 0.45081 |
| OpenRec LR (pointwise BCE) | 42/43/44 | 192 | 0.78679 ± 0.00029 | 0.56669 ± 0.00050 | 0.63449 ± 0.00044 | 0.66250 ± 0.00042 |
| OpenRec FM (pointwise BCE) | 42/43/44 | 192 | 0.78720 ± 0.00088 | 0.56858 ± 0.00094 | 0.63622 ± 0.00089 | 0.66398 ± 0.00063 |
| OpenRec LightGBM (LambdaRank) | 42 | 192 | **0.82203** | **0.61311** | **0.68157** | **0.70229** |

Values after `±` are sample standard deviations. Popularity is deterministic
and therefore has one run. LightGBM currently has one completed run for this
pruned configuration, so no variance is reported for it. The per-run values,
wall times, and selected LR/FM epochs are retained in
`ebnerd-small-common-features-v2.csv`.

## Feature ownership audit

The input is a common logical feature configuration, not a LightGBM-private
feature set. The 124 recorded encoded feature names map without omissions to
122 logical feature IDs in `rec-algorithm`'s global
`feature.catalog.json`; device one-hot columns and cyclical request-time columns
collapse to their parent catalog definitions. The logical inputs cover content,
request context, user-candidate affinity, relative freshness, long-term user
statistics, topic/entity affinity, candidate-relative ranks, recent session
exposure, fixed pre-window history, and causal past exposure/engagement.

Each model family builds its own fitted `FeatureSpace`, and all three produce
192 numeric columns for this configuration. Their learned normalization,
categorical mappings, and model parameters remain family-specific even though
the ordered logical feature inputs and encoded widths match.

Catalog membership also does not yet mean production serving parity. Of the 122
logical catalog features, 10 are stable with both online and offline
materialization, while 112 remain experimental and offline-only. All experiment
features obey causal cutoffs and exclude the current request's outcome, but the
offline-only families must receive matching incremental online implementations
before this feature stack can be activated in rank-engine.

## Comparison with the stage-7 matrix

| Model | Stage-7 Macro AUC | Current common features | Gain |
|---|---:|---:|---:|
| LR | 0.69623 | 0.78679 | +0.09055 |
| FM | 0.69943 | 0.78720 | +0.08777 |
| LightGBM | 0.72167 | 0.82203 | +0.10036 |

The richer causal history improves all three model families substantially, so
the earlier LightGBM gain was not only a tree-model artifact. FM exceeds LR by
0.00041 Macro AUC under the current configuration, while LightGBM seed 42 still
exceeds the FM three-seed mean by 0.03483. This remaining difference is
consistent with LambdaRank's impression-level objective and tree interactions,
but it is not yet a three-seed LightGBM estimate.

LR has the best probability calibration under the current runs: mean global
LogLoss is 0.22821 for LR, 0.22996 for FM, and 0.31618 for the uncalibrated
LightGBM ranking score. The primary ranking metric remains impression Macro AUC.

## Run provenance

- `runs/ebnerd-192-shared-popularity-seed42`
- `runs/ebnerd-192-shared-lr-seed{42,43,44}`
- `runs/ebnerd-192-shared-fm-seed{42,43,44}`
- `runs/ebnerd-stage14-past-engagement-pruned-rerun-lightgbm-seed42`

The report exporter verified every artifact against its manifest SHA-256. All
eight runs share protocol SHA-256
`da9a4263b1e04d9def540c7cf8d0dbfc943621b16c2e3133f369adb57030b689`.
