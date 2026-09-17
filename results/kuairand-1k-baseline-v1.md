# KuaiRand-1K baseline v1

Status: preliminary local temporal holdout; LR/FM results cover seeds 42, 43,
and 44.

The prepared dataset contains 11,756,073 exposures, 1,000 users, 4,371,868
items, 15 scenes, and 4,437,332 positive labels. Its SHA-256 is
`0df2329e4b8dbd878ab0e176cde2e156bc96337607db5ddf35b73e63f19f520c`.
Standard and random-policy observations are always reported separately. Random
observations are excluded from training, validation selection, and feature
history.

| Model | Seeds | Standard AUC | Standard LogLoss | Random AUC | Random LogLoss |
|---|---:|---:|---:|---:|---:|
| Popularity | 42 | 0.52956 | 0.65869 | 0.52949 | **0.57823** |
| OpenRec LR | 42/43/44 | 0.74603 ± 0.00016 | 0.57106 ± 0.00025 | **0.66376 ± 0.00020** | 0.66958 ± 0.00460 |
| OpenRec FM | 42/43/44 | **0.74970 ± 0.00017** | **0.56836 ± 0.00016** | 0.65923 ± 0.00101 | 0.68172 ± 0.00223 |

The values after `±` are sample standard deviations across seeds. Popularity is
deterministic and therefore has one run. FM improves the standard-policy metrics
over LR, while LR performs better on the randomly exposed population. This is
evidence of a policy-distribution shift, not proof of causal or online uplift.
Stronger external baselines and statistical inference remain required for a
model-family or SOTA claim.
