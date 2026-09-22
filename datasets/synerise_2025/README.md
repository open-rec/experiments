# RecSys 2025 Synerise

Official dataset and terms: https://recsys.synerise.com/data-set
Official evaluation code: https://github.com/Synerise/recsys2025

Obtain the organizer's 1.9 GB `synerise_dataset.tar.gz` from the linked data
page, then extract `product_buy.parquet` to
`data/raw/synerise_2025/product_buy.parquet`. The older summary page has a
stale 404 link; use the data-set page. The raw and prepared data are ignored by
Git. The dataset is CC BY-NC 4.0.

The organizer's [final leaderboard](https://recsys.synerise.com/results/final-leaderboard)
contains a `baseline` row. Its six task scores and Borda count are transcribed
in `official_baselines.csv`. They come from the challenge's Universal Behavioral
Profiles evaluation and are not comparable with local purchase Recall@20.

This adapter uses observed purchases only and retains client, SKU and UTC
timestamp. The local task is next new-to-recent purchased SKU from prior purchases, with a
chronological holdout. Only purchases strictly earlier than the target timestamp
can be query triggers; items bought by the same client in the same second are
not treated as prior history. This is a subset study for OpenRec's recall algorithms,
not the official Universal Behavioral Profiles task. It does not use cart,
page-view, search, or hidden competition targets, so its Recall@20 cannot be
compared with the official leaderboard.

Run from `experiments/` after obtaining the file:

```bash
.venv/bin/python -m openrec_experiments.cli prepare \
  --config datasets/synerise_2025/prepare.json \
  --output data/processed/synerise-2025-v2.parquet
.venv/bin/python -m openrec_experiments.cli run \
  --config studies/baselines/synerise-2025-retrieval.json \
  --output runs/synerise-2025/retrieval-v4
```

The in-memory OpenRec item co-occurrence implementation is quadratic in each
client's purchase history. For the full dataset, use a bounded training sample
or replace it with a scalable implementation before running this study.

For the challenge's distinct official task, obtain the organizer's
`challenge_dataset.tar.gz` and extract its `input/` and `target/` directories
under `data/raw/synerise_2025/challenge/`. Generate 21-dimensional OpenRec
behavioral profiles for the exact 1,000,000 relevant clients, validate their
submission format, then run the organizer's fixed downstream trainer.

Install the optional trainer dependencies with
`.venv/bin/python -m pip install --index-url https://mirrors.aliyun.com/pypi/simple/ -e '.[synerise-official]'`.
Check out the organizer's evaluator at commit
`e91c1a62f84611151a399ce157e6031a4a73618f` at the path below.

```bash
.venv/bin/python -m openrec_experiments.cli official-synerise-profiles \
  --config studies/baselines/synerise-2025-official-profiles.json \
  --output runs/synerise-2025/official-profiles-v1
PYTHONPATH=/tmp/openrec-synerise-recsys2025 .venv/bin/python -m validator.run \
  --data-dir data/raw/synerise_2025/challenge \
  --embeddings-dir runs/synerise-2025/official-profiles-v1
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 .venv/bin/python \
  -m openrec_experiments.datasets.synerise_2025.train \
  --evaluator /tmp/openrec-synerise-recsys2025 \
  --data-dir data/raw/synerise_2025/challenge \
  --embeddings-dir runs/synerise-2025/official-profiles-v1 \
  --tasks churn propensity_category propensity_sku conversion \
          propensity_new_sku propensity_price \
  --log-name openrec-profiles --num-workers 0 --accelerator cpu \
  --devices auto --score-dir runs/synerise-2025/official-task-scores \
  --hidden-logging-mode
```

The trainer wrapper only replaces per-row pandas label filtering with an
indexed array containing the same labels; it spot-checks against the official
calculator. The model, splits, epochs, and metrics come from the official
trainer. The local `challenge_dataset` split and published final leaderboard
use different target windows, so even matching task metrics cannot establish
an equal-split leaderboard ranking.

The organizer also supplies an aggregated-feature profile baseline. Generate
it using its default `[1, 7, 30]` day windows and top-10 values, then align
its profiles to the same million clients with zero vectors for any client
missing from the local input window. This alignment permits a same-split,
same-client comparison with OpenRec using the trainer above:

```bash
PYTHONPATH=/tmp/openrec-synerise-recsys2025 .venv/bin/python \
  -m baseline.aggregated_features_baseline.create_embeddings \
  --data-dir data/raw/synerise_2025/challenge \
  --embeddings-dir runs/synerise-2025/official-baseline-profiles-v1
.venv/bin/python -m openrec_experiments.datasets.synerise_2025.align \
  --baseline-profiles runs/synerise-2025/official-baseline-profiles-v1 \
  --relevant-clients data/raw/synerise_2025/challenge/input/relevant_clients.npy \
  --output runs/synerise-2025/official-baseline-aligned-v1
OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 .venv/bin/python \
  -m openrec_experiments.datasets.synerise_2025.train \
  --evaluator /tmp/openrec-synerise-recsys2025 \
  --data-dir data/raw/synerise_2025/challenge \
  --embeddings-dir runs/synerise-2025/official-baseline-aligned-v1 \
  --tasks churn propensity_category propensity_sku conversion \
          propensity_new_sku propensity_price \
  --log-name official-example-baseline --num-workers 0 \
  --accelerator cpu --devices auto \
  --score-dir runs/synerise-2025/official-baseline-scores \
  --hidden-logging-mode
.venv/bin/python -m openrec_experiments.datasets.synerise_2025.compare \
  --openrec-scores runs/synerise-2025/official-task-scores/scores.json \
  --baseline-scores runs/synerise-2025/official-baseline-scores/scores.json \
  --openrec-profiles runs/synerise-2025/official-profiles-v1 \
  --baseline-profiles runs/synerise-2025/official-baseline-aligned-v1 \
  --evaluator /tmp/openrec-synerise-recsys2025 \
  --output runs/synerise-2025/official-local-comparison-v1
```

In the downloaded `challenge_dataset` split, the example baseline covers all
1,000,000 relevant clients, so alignment filled zero missing profiles. Its
profile has 320 dimensions and uses purchases, cart changes, page visits, and
search queries. The 21-dimensional OpenRec profile uses purchases and cart
changes. These are different representation methods evaluated by the same
fixed downstream model.

The [official local comparison](../../results/synerise-2025-official-local-v1.md)
reports all six task scores for both profile sets on the same split. OpenRec
totals 4.3420 versus 4.3896 for the organizer's example baseline. This local
sum is separate from the published final leaderboard's hidden target windows.
