# Synerise 2025: official local task comparison

The [organizer's evaluator](https://github.com/Synerise/recsys2025) runs a fixed
downstream model for six tasks on submitted user profiles. Both profile sets
cover the same 1,000,000 relevant clients and passed the organizer's validator.
OpenRec provides 21 dimensions from its event-feature aggregation on purchases
and cart changes. The organizer's example aggregated-feature baseline provides
320 dimensions from purchases, cart changes, page visits, and search queries.

The two profiles use the same organizer-provided `challenge_dataset` input and
target split, with successive 14-day train and validation target windows, and
the same 3-epoch downstream trainer. Churn and conversion use
AUROC. Each propensity task scores `0.8 × AUROC + 0.1 × diversity + 0.1 ×
novelty`. The organizer selects the best of the three validation epochs for
each task. Local score files are under `runs/synerise-2025/`.

| Task | OpenRec | Official example baseline | Difference |
| --- | ---: | ---: | ---: |
| Churn AUROC | 0.7083 | 0.6983 | +0.0100 |
| Category propensity | 0.7409 | 0.7601 | −0.0192 |
| SKU propensity | 0.7446 | 0.7488 | −0.0042 |
| Conversion AUROC | 0.6758 | 0.7057 | −0.0299 |
| New SKU propensity | 0.7230 | 0.7109 | +0.0121 |
| Price propensity | 0.7494 | 0.7658 | −0.0164 |
| **Sum of six task scores** | **4.3420** | **4.3896** | **−0.0476** |

OpenRec scores higher on churn and new-SKU propensity. The organizer's example
scores higher on category, SKU, conversion, and price; its six-task sum is
0.0476 higher on this local split. The two profiles use different input event
sets and dimensions, so this comparison measures the complete representation
methods, not an isolated feature ablation. The official final leaderboard's
Borda count cannot be reconstructed from two local runs.

The [published final leaderboard](https://recsys.synerise.com/results/final-leaderboard)
lists its `baseline` at 4.2019 summed task score and 0.6947 churn. The
leaderboard uses undisclosed final target windows, so those numbers are shown
only as a published reference and are not used to calculate local differences.

The local comparison uses official evaluator commit
`e91c1a62f84611151a399ce157e6031a4a73618f`. The wrapper caches exact
official labels once, checks samples against the official target calculators,
disables unused checkpoints, and otherwise runs the fixed model and metric
implementation. The official baseline script uses its default 1, 7, and 30 day
windows and top-10 values. Baseline profiles needed no zero filling when
aligned to the relevant-client list.

The ignored `runs/synerise-2025/official-local-comparison-v1/` directory
contains the score CSV and a manifest with the evaluator commit and hashes of
both submitted profile arrays and every score file. The tracked companion CSV
is `synerise-2025-official-local-v1.csv`.
