# EB-NeRD-small baseline v1

Status: preliminary local temporal holdout; not an official leaderboard result.

The prepared dataset contains 5,514,689 candidates in 477,534 impressions,
18,827 users, 5,776 candidate articles, and 479,385 positive candidates. The
behavior-only projection used by the first three baselines has SHA-256
`7835be6d67f6bdcfe4b627a573df2b4cc69276f794d625b19cd62d474edf59fc`;
the content projection preserves those rows and adds article metadata.

The local test set is the complete official validation bundle. The first four
days of the official training bundle are local training and its final three days
are local validation. Boundaries follow the observed 07:00 UTC dataset day.

| Model | Seeds | Macro AUC | Macro MRR | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Popularity | 42 | 0.56467 | 0.32585 | 0.37074 | 0.45081 |
| OpenRec LR | 42/43/44 | 0.55039 ± 0.00177 | 0.33952 ± 0.00196 | 0.37962 ± 0.00216 | 0.45630 ± 0.00198 |
| OpenRec FM | 42/43/44 | 0.55864 ± 0.00058 | 0.34696 ± 0.00074 | 0.38779 ± 0.00082 | 0.46372 ± 0.00072 |
| OpenRec FM + content | 42/43/44 | **0.57292 ± 0.00297** | **0.35098 ± 0.00282** | **0.39328 ± 0.00275** | **0.46889 ± 0.00255** |

The values after `±` are sample standard deviations across seeds. Popularity
has one deterministic run. The content variant adds fixed-width signed hashes
of title, topic tags and subcategories plus point-in-time content age. Its
prepared-data SHA-256 is
`7e05c3223952a5ac3e3ede8cd8f3e7d10482c0b953f08d7d7ccd99b587b81d8f`.
Against behavior-only FM, content improves mean Macro AUC by 0.01427, MRR by
0.00403, NDCG@5 by 0.00549 and NDCG@10 by 0.00517.

For the seed-42 diagnostic, content improves Macro AUC from 0.56312 to 0.57547
on the 239,799 impressions whose clicked articles are all unseen in training.
It reduces Macro AUC from 0.32318 to 0.28451 on the much smaller 4,825
all-warm-click impressions. This supports the cold-start value of content but
also shows that a warm-item gate or richer hybrid needs evaluation. History-aware
and semantic news baselines remain required before drawing model-quality or SOTA
claims.
