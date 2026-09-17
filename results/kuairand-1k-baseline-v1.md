# KuaiRand-1K baseline v1

Status: preliminary local temporal holdout; LR/FM results cover seeds 42, 43,
and 44.

The prepared dataset contains 11,756,073 exposures, 1,000 users, 4,371,868
items, 15 scenes, and 4,437,332 positive labels. The behavior-only projection
has SHA-256
`0df2329e4b8dbd878ab0e176cde2e156bc96337607db5ddf35b73e63f19f520c`;
the content projection preserves the same ordered exposures and has SHA-256
`936b2607eb6b76b8b5176532ec7cb2f59c03c03adf6ca4c2f27309a6151a4aa1`.
Standard and random-policy observations are always reported separately. Random
observations are excluded from training, validation selection, and feature
history.

| Model | Seeds | Standard AUC | Standard LogLoss | Random AUC | Random LogLoss |
|---|---:|---:|---:|---:|---:|
| Popularity | 42 | 0.52956 | 0.65869 | 0.52949 | **0.57823** |
| OpenRec LR | 42/43/44 | 0.74603 ± 0.00016 | 0.57106 ± 0.00025 | **0.66376 ± 0.00020** | 0.66958 ± 0.00460 |
| OpenRec LR + content | 42/43/44 | 0.75119 ± 0.00011 | 0.56764 ± 0.00025 | 0.66158 ± 0.00048 | 0.68669 ± 0.00584 |
| OpenRec FM | 42/43/44 | 0.74970 ± 0.00017 | 0.56836 ± 0.00016 | 0.65923 ± 0.00101 | 0.68172 ± 0.00223 |
| OpenRec FM + content | 42/43/44 | **0.75687 ± 0.00005** | **0.56218 ± 0.00012** | 0.66101 ± 0.00094 | 0.70305 ± 0.00368 |

The values after `±` are sample standard deviations across seeds. Popularity is
deterministic and therefore has one run. The content variants add video type,
upload type, tags, and point-in-time age from the basic video table. They
exclude user snapshots and aggregate video statistics whose historical
availability is unknown. Content improves LR by 0.00516 Standard AUC and FM by
0.00718; FM + content is the strongest model on the standard-policy population.

The randomly exposed population gives a more cautious result. LR + content
loses 0.00217 AUC and adds 0.01711 LogLoss. FM + content gains 0.00178 AUC but
adds 0.02133 LogLoss, indicating worse probability calibration despite a small
ranking gain. For the seed-42 diagnostic, Standard cold-item AUC improves from
0.74450 to 0.74990 for LR and from 0.74802 to 0.75574 for FM. Standard warm-item
AUC also improves from 0.75213 to 0.75420 and from 0.75519 to 0.75935,
respectively. The test set contains 5,700,209 cold and 954,500 warm Standard
exposures, plus 13,414 cold and 29,614 warm Random exposures.

These results support the usefulness of content features under the logged
production policy. The Random results show policy-distribution shift and a
calibration problem, not proof of causal or online uplift. Stronger external
baselines and statistical inference remain required for a model-family or SOTA
claim.
