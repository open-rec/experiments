# EB-NeRD-small OpenRec model comparison v1

Status: preliminary local temporal holdout; not an official leaderboard result.

This report compares OpenRec LR, FM, and LightGBM under the same data, split,
feature set, and three seeds. It is the EB-NeRD counterpart to the KuaiRand
multi-model table and is derived from the completed immutable run manifests
listed below.

The prepared dataset contains 5,514,689 candidates in 477,534 impressions. The
first four days of the official training bundle are local training, its final
three days are local validation, and the complete official validation bundle is
the local test set. Evaluation covers 2,928,942 test candidates in 244,647
impressions. Feedback is frozen at the training boundary with the configured
one-hour delay and daily feature snapshots.

All three models consume the same 131-dimensional causal feature matrix from
`ebnerd-feature-stage7-candidate-rank.json`. It includes article content,
request context, visible user-item/category interactions, relative freshness,
long-term user statistics, topic/entity affinity, and candidate-relative ranks.
It excludes future impressions, future engagement, and current-impression
outcomes.

| Model | Seeds | Macro AUC | Macro MRR | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| OpenRec LR (pointwise BCE) | 42/43/44 | 0.69623 ± 0.00097 | 0.46325 ± 0.00156 | 0.52238 ± 0.00150 | 0.57149 ± 0.00098 |
| OpenRec FM (pointwise BCE) | 42/43/44 | 0.69943 ± 0.00144 | 0.46267 ± 0.00087 | 0.52302 ± 0.00111 | 0.57223 ± 0.00119 |
| OpenRec LightGBM (LambdaRank) | 42/43/44 | **0.72167 ± 0.00058** | **0.48266 ± 0.00067** | **0.54742 ± 0.00075** | **0.59176 ± 0.00071** |

Values after `±` are sample standard deviations across seeds. The corresponding
per-seed metrics, wall times, and LR/FM selected epochs are in
`ebnerd-small-model-comparison-v1.csv`.

LightGBM improves mean Macro AUC by 0.02543 over LR and 0.02224 over FM. It also
leads on MRR and both NDCG cutoffs. This is consistent with two differences:
LambdaRank optimizes ordering within an impression, and trees capture nonlinear
thresholds and interactions among counts, recencies, affinities, and relative
ranks without requiring those crosses to be enumerated for a linear model.

FM improves Macro AUC by 0.00319 and NDCG@10 by 0.00074 over LR, but its mean MRR
is 0.00058 lower. The small and mixed change suggests that generic second-order
interactions add less value than the explicit causal interaction and
candidate-relative features already present in the matrix.

The test global metrics provide a useful calibration diagnostic but are not the
primary ranking result:

| Model | Global AUC | Global LogLoss |
|---|---:|---:|
| OpenRec LR | 0.75635 ± 0.00135 | **0.25657 ± 0.00032** |
| OpenRec FM | **0.76043 ± 0.00110** | 0.25720 ± 0.00032 |
| OpenRec LightGBM | 0.73526 ± 0.00042 | 0.45298 ± 0.00162 |

LightGBM's LambdaRank output is an ordering score rather than a calibrated click
probability, so its LogLoss must not be interpreted as contradicting its better
impression ranking. Probability calibration would require a separate
training-only calibration fit and evaluation on untouched data.

The comparison is deliberately limited to the shared stage-7 features. The
later 192-dimensional, LightGBM-specific causal feature stack reaches 0.82203
test Macro AUC for seed 42, but it is not inserted into this table because LR
and FM have not been evaluated on that same matrix. Mixing it into the shared
table would confound model-family and feature-set effects.

## Run provenance

- `runs/ebnerd-stage7-candidate-rank-lr-seed{42,43,44}`
- `runs/ebnerd-stage7-candidate-rank-fm-seed{42,43,44}`
- `runs/ebnerd-stage7-candidate-rank-lightgbm-seed{42,43,44}`

The report exporter verified every artifact against the SHA-256 recorded in its
run manifest. All nine runs share protocol SHA-256
`a912acc51170e1f2628ed882243e43d8cf8073f43096719a6f4c90dd0e173765`.
