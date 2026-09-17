# EB-NeRD-small baseline v1

Status: preliminary local temporal holdout; not an official leaderboard result.

The prepared dataset contains 5,514,689 candidates in 477,534 impressions,
18,827 users, 5,776 candidate articles, and 479,385 positive candidates. Its
SHA-256 is `7835be6d67f6bdcfe4b627a573df2b4cc69276f794d625b19cd62d474edf59fc`.

The local test set is the complete official validation bundle. The first four
days of the official training bundle are local training and its final three days
are local validation. Boundaries follow the observed 07:00 UTC dataset day.

| Model | Seeds | Macro AUC | Macro MRR | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Popularity | 42 | 0.56467 | 0.32585 | 0.37074 | 0.45081 |
| OpenRec LR | 42/43/44 | 0.55039 ± 0.00177 | 0.33952 ± 0.00196 | 0.37962 ± 0.00216 | 0.45630 ± 0.00198 |
| OpenRec FM | 42/43/44 | 0.55864 ± 0.00058 | 0.34696 ± 0.00074 | 0.38779 ± 0.00082 | 0.46372 ± 0.00072 |

The values after `±` are sample standard deviations across seeds. Popularity
has one deterministic run. FM improves top-ranked-list metrics over popularity
but does not improve impression-macro AUC. Current LR/FM use only historical
count/click-rate features plus scene and article category. History-aware and
semantic news baselines remain required before drawing model-quality claims.
