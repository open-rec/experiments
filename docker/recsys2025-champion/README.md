# RecSys 2025 rec2 champion container

This runs the published [rec2 first-place source](https://github.com/yukia18/recsys-challenge-2025-1st-place)
at commit `fb17bd991d3ab3cc246f529485f7ba662749d390` and the
[organizer evaluator](https://github.com/Synerise/recsys2025) at commit
`e91c1a62f84611151a399ce157e6031a4a73618f`. The source trees and
licensed data are mounted read-only. The container makes a writable copy of
the winning code under the output directory because upstream stages expect to
write checkpoints and intermediate data beside their scripts.

From the `experiments/` directory, set absolute paths to the two checked-out
repositories, the extracted organizer `challenge_dataset` directory, the
official `synerise_dataset` raw archive, and a writable output directory:

```bash
git clone https://github.com/yukia18/recsys-challenge-2025-1st-place.git /tmp/recsys2025-rec2
git -C /tmp/recsys2025-rec2 checkout fb17bd991d3ab3cc246f529485f7ba662749d390
git clone https://github.com/Synerise/recsys2025.git /tmp/openrec-synerise-recsys2025
git -C /tmp/openrec-synerise-recsys2025 checkout e91c1a62f84611151a399ce157e6031a4a73618f
export RECSYS2025_WINNER_SOURCE=/tmp/recsys2025-rec2
export RECSYS2025_OFFICIAL_SOURCE=/tmp/openrec-synerise-recsys2025
export RECSYS2025_CHALLENGE_DATA=$PWD/data/raw/synerise_2025/challenge
export RECSYS2025_RAW_ARCHIVE=$PWD/data/raw/synerise_2025/synerise_dataset_direct.tar.gz
export RECSYS2025_OUTPUT=/ssd/2/xsank.mz/recsys2025-rec2-output
export RECSYS2025_UID=$(id -u)
export RECSYS2025_GID=$(id -g)
export RECSYS2025_GPU=0
mkdir -p "$RECSYS2025_OUTPUT"
docker compose -f docker/recsys2025-champion/compose.yaml build
docker compose -f docker/recsys2025-champion/compose.yaml run --rm champion check
docker compose -f docker/recsys2025-champion/compose.yaml run --rm champion prepare
docker compose -f docker/recsys2025-champion/compose.yaml run --rm champion all
```

`RECSYS2025_GPU` selects a host GPU index; use another free GPU when running
independent stages concurrently.

`all` runs the organizer's feature engineering, MTL transformer, contrastive
transformer, and stacking stages before the six-task organizer evaluator.
Stages can be resumed individually with `feature`, `mtl`, `cl`, `baseline`, `stack`, and
`evaluate`. Completed stages are recorded under `stages/`; raw data and model
artifacts stay under the output directory and must not be committed. `all`
also writes `comparison.json` with the six local organizer scores and the
published hidden-final reference, clearly marked as different evaluation
splits.

The organizer CLI accepts `conversion`, `propensity_new_sku`, and
`propensity_price` for the three hidden tasks. The adapter enables its
`--hidden-logging-mode`, which labels their score entries `hidden1`, `hidden2`,
and `hidden3` in the same order as the final leaderboard.

The original full-event parquet files encode `timestamp` as text. The published
MTL code compares those columns to Python `datetime` values, which fails under
Polars 1.30. Before neural stages, the adapter parses the same timestamp text
into native `Datetime` columns in its output copy. It records the source and
converted digests in `timestamp-normalization.json` and leaves the provider
files untouched.

The organizer's challenge split omits SKUs that appear in later full-event
history. Submission feature engineering joins that history to product
properties, so the adapter extracts the complete properties table from the
raw archive for that stage. MTL continues to use the challenge-split table:
its SKU ID mapping and published embedding size depend on that table.
`submission-properties.json` records the extracted table digest.

The published MTL code also hard-codes a URL vocabulary of 373,500, while the
released train split contains URL IDs through 437,993. The adapter scans the
generated train and validation arrays and expands the URL embedding table in
its writable copy of the two MTL scripts to the observed maximum plus one.
`mtl-url-vocabulary.json` records the observed range and source hashes. This
is a compatibility change to the model parameter count and should be reported
alongside any local result.

MTL saves weight-only top-three checkpoints. The adapter reads the 25
`valid/sum_score` values from the matching offline W&B run, verifies that the
saved epochs are its top three, and chooses the best. If all 25 epochs finish
but W&B fails while staging artifacts during trainer teardown, the adapter
records that recovery and uses the already saved checkpoints. W&B artifact and
cache directories are mounted under the writable output directory.

The published winner explicitly warns that its local validation pipeline has
leakage. Therefore, its local six-task scores are a pipeline check, not a
reproduction of the hidden final leaderboard score. The hidden competition
window and original submitted profiles are not public; report the local task
scores and the organizer's final score as separate measurements.

The [official final leaderboard](https://recsys.synerise.com/results/final-leaderboard)
records rec2 at Borda count 662 and six-task score sum 4.7949. These are
reference values for the hidden final evaluation only.

The Docker base image is the existing local `openrec/experiments-gpu:local`
(PyTorch 2.8/CUDA 12.9). It differs from the three upstream environments.
Record dependency versions and GPU details with each result; numerical parity
is not guaranteed by the common container alone.
