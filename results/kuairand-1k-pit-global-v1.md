# KuaiRand-1K OpenRec global PIT features v1

Status: completed local temporal holdout. Popularity is deterministic; LR, FM,
and LightGBM cover seeds 42, 43, and 44. The companion CSV retains every run.

This experiment answers whether the shared OpenRec point-in-time feature set
also improves KuaiRand. It uses 33 logical features from the global catalog: 16
user event aggregates and 17 item/scene event aggregates. Categorical encoding
produces 46 model inputs. All selected catalog entries are `stable`,
`online=true`, and `offline=true`; no KuaiRand-only content, user snapshot,
EB-NeRD history, semantic, or candidate-list feature is used.

| Model | Seeds | Standard AUC | Standard LogLoss | Random AUC | Random LogLoss |
|---|---:|---:|---:|---:|---:|
| Popularity | 42 | 0.52956 | 0.65869 | 0.52949 | **0.57823** |
| OpenRec LR + global PIT | 42/43/44 | 0.75142 ± 0.00003 | 0.56651 ± 0.00001 | 0.66737 ± 0.00036 | 0.68590 ± 0.00062 |
| OpenRec FM + global PIT | 42/43/44 | 0.75169 ± 0.00022 | 0.56609 ± 0.00030 | 0.67316 ± 0.00113 | 0.69849 ± 0.00584 |
| OpenRec LightGBM + global PIT | 42/43/44 | **0.75410 ± 0.00046** | **0.56289 ± 0.00057** | **0.67690 ± 0.00079** | 0.71215 ± 0.00397 |

Values after `±` are sample standard deviations. Every learned-model row uses
the same prepared data, split, visibility rules, and feature selection. The
protocol SHA-256 is
`f7a44d1222b70a7ae6064f93c1974c1397a149150edd0d42349707865de09220`;
the behavior projection SHA-256 is
`0df2329e4b8dbd878ab0e176cde2e156bc96337607db5ddf35b73e63f19f520c`.
Random-policy observations are excluded from training, validation selection,
and feature history. Evaluation features are frozen at the training boundary
with a one-hour feedback delay.

Compared with the earlier behavior-only baseline, the global PIT selection
raises mean Standard AUC from 0.74603 to 0.75142 for LR (+0.00539) and from
0.74970 to 0.75169 for FM (+0.00199). Random AUC rises from 0.66376 to 0.66737
for LR (+0.00361) and from 0.65923 to 0.67316 for FM (+0.01393). This confirms
that the shared OpenRec PIT additions transfer to KuaiRand, with the largest
unbiased-policy ranking gain appearing for FM.

LightGBM uses a binary objective because KuaiRand records independent exposure
rows; its per-row group IDs do not provide candidate pairs for LambdaRank.
Under the common features it leads LR/FM by 0.00241/0.00267 Standard AUC and
0.00374/0.00953 Random AUC. Its Random LogLoss is worse despite better AUC, so
the random-policy probabilities require calibration and must not be interpreted
as evidence of online uplift.

Run manifests and artifacts are under `runs/kuairand-pit-global-*`. The exact
feature and training configuration is
`studies/baselines/kuairand-pit-global.json`.
