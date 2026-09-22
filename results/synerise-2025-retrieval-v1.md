# Synerise 2025: local OpenRec purchase retrieval study

Run: `runs/synerise-2025/retrieval-v4` (ignored artifact), 2026-09-22.
Companion CSV: `synerise-2025-retrieval-v1.csv`.
This uses the organizer's **raw** `synerise_dataset.tar.gz`, specifically its
`product_buy.parquet` (2,318,502 purchases). The challenge's preprocessed
dataset contains a different event population and uses a different task.

Purchases are sorted by UTC timestamp. The earliest 70% are training
(1,622,947 events, 487,289 distinct SKUs); the next 15% are local validation
and the final 15% local test. Both recall models are fitted only to training
purchases. Each query uses at most five earlier purchases by the same client
as observed triggers, strictly earlier than the target timestamp. Repeated recent SKUs are excluded from the query set
and recommendations. The study tests 5,000 evenly spaced eligible queries per
holdout split, over the full training SKU catalog.

| Split | Method | Recall@20 | MRR@20 | Queries |
| --- | --- | ---: | ---: | ---: |
| Validation | OpenRec Hot | 0.0076 | 0.00165 | 5,000 |
| Validation | OpenRec ItemBasedI2I, then Hot | 0.0140 | 0.00433 | 5,000 |
| Test | OpenRec Hot | 0.0036 | 0.00082 | 5,000 |
| Test | OpenRec ItemBasedI2I, then Hot | 0.0060 | 0.00209 | 5,000 |

The test gain over Hot is 0.0024 absolute Recall@20 on this local task.
Only purchases are used. This does not evaluate Universal Behavioral Profiles,
cart/page/search behavior, the challenge's target SKU subset, hidden tasks,
or its official AUROC/novelty/diversity aggregate. Results cannot be compared
with the official leaderboard. Cold test SKUs absent from training cannot be
returned by either model.

## Official baseline reference

The [organizer's final leaderboard](https://recsys.synerise.com/results/final-leaderboard)
has a `baseline` row. A machine-readable copy is in
`datasets/synerise_2025/official_baselines.csv`.

| Method | Borda count | Sum of task scores | Churn | Category | SKU | Hidden 1 | Hidden 2 | Hidden 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Official baseline | 143 | 4.2019 | 0.6947 | 0.6985 | 0.6919 | 0.6579 | 0.7382 | 0.7207 |

The challenge scores measure Universal Behavioral Profiles on its processed
data and downstream tasks. They do not measure next-SKU Recall@20 on the raw
purchase file; no numerical comparison with the local table is valid.
