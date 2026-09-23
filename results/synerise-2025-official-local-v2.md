# Synerise 2025: expanded OpenRec official local comparison

This run evaluates the expanded OpenRec conversion, category, and price feature
foundation with the organizer's fixed six-task model and local challenge split.
The v2 profile contains the public 320-dimensional organizer representation and
71 appended OpenRec features. This setup holds the public input representation
constant and measures whether the added OpenRec features improve it.

All profiles cover the same 1,000,000 relevant clients and passed the official
validator. The evaluator uses three epochs and retains the best validation score
for each task.

| Task | OpenRec v1 | OpenRec v2 | Official baseline | v2 vs v1 | v2 vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Churn AUROC | 0.7083 | 0.7084 | 0.6983 | +0.0001 | +0.0101 |
| Category propensity | 0.7409 | 0.7646 | 0.7601 | +0.0237 | +0.0045 |
| SKU propensity | 0.7446 | 0.7610 | 0.7488 | +0.0164 | +0.0122 |
| Conversion AUROC | 0.6758 | 0.7153 | 0.7057 | +0.0395 | +0.0096 |
| New SKU propensity | 0.7230 | 0.7734 | 0.7109 | +0.0504 | +0.0625 |
| Price propensity | 0.7494 | 0.7720 | 0.7658 | +0.0226 | +0.0062 |
| **Sum of six task scores** | **4.3420** | **4.4947** | **4.3896** | **+0.1527** | **+0.1051** |

The three targeted weaknesses all improved. Conversion has the largest direct
gain at 0.0395, while category and price improve by 0.0237 and 0.0226. The
largest overall gain is new-SKU propensity at 0.0504. The v2 profile exceeds
the example baseline on every local task.

The result is a feature-addition ablation against the public baseline. It does
not attribute the full v2 score to OpenRec alone, because the first 320 profile
dimensions are the organizer representation. A repeated-seed run is still
needed to estimate variance in the official neural evaluator. The published
leaderboard uses hidden target windows, so this local result does not imply a
leaderboard rank.

The run uses official evaluator commit
`e91c1a62f84611151a399ce157e6031a4a73618f`. Reproducibility hashes for the
profile arrays and all score files are stored in the ignored
`runs/synerise-2025/official-local-comparison-v2/manifest.json`.
