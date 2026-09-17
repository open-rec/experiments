# OpenRec Experiments

Reproducible recommender-system research built on OpenRec. The initial datasets
are **EB-NeRD** and **KuaiRand-1K**. The project first establishes trustworthy
baselines, then studies training-serving feature consistency, feature freshness,
and the trade-off between recommendation quality and system cost.

**Current status: preliminary real-data baselines are available; no official
leaderboard or SOTA claims are available yet.** The first version reuses the
feature aggregation, `FeatureSpace`, LR, and FM implementations from
`rec-algorithm`. Experiment checkpoints use OpenRec model formats but are never
published to a serving environment automatically.

## Installation

Python 3.10 or newer is required; Python 3.12 is recommended. Check out this
repository next to the other OpenRec repositories:

```text
openrec/
├── experiments/
└── rec-algorithm/
```

```bash
python -m venv .venv
.venv/bin/python -m pip install --index-url https://mirrors.aliyun.com/pypi/simple/ -e '.[test]'
.venv/bin/python -m pytest -q
```

All PyPI downloads use the Aliyun mirror. The runner explicitly loads the
`rec-algorithm` source tree specified by each experiment configuration, verifies
the actual import path, and records repository commits, working-tree status, a
diff digest, and environment versions. Versions used in the current workspace
are listed in `requirements-tested.txt`; this file is not a complete
cross-platform lock file.

Use the Aliyun mirror explicitly when installing an individual package as well:

```bash
.venv/bin/python -m pip install \
  --index-url https://mirrors.aliyun.com/pypi/simple/ <package>
```

## Data preparation

Review each provider's terms and obtain the files from its official entry point.
Download instructions and expected file layouts are documented under
[datasets/ebnerd](datasets/ebnerd/README.md) and
[datasets/kuairand](datasets/kuairand/README.md). Datasets are not included in
this repository or in the OpenRec distribution.

Access overseas data hosts through caller-provided `HTTPS_PROXY` and
`HTTP_PROXY` settings. Download scripts must not contain personal proxy
addresses. Prefer a provider's domestic mirror when one is available, and
always verify the provider's published digest.

Run all commands from the repository root. Relative paths in configuration
files are resolved from the current working directory.

```bash
.venv/bin/python -m openrec_experiments.cli prepare \
  --config datasets/ebnerd/prepare.json --output data/processed/ebnerd-small-content.parquet
.venv/bin/python -m openrec_experiments.cli prepare \
  --config datasets/kuairand/prepare.json --output data/processed/kuairand-1k-content.parquet
```

The adapters preserve observed exposure labels and candidate identities; they
do not fabricate negative exposures. Each prepared dataset receives a companion
manifest containing source-file SHA-256 digests, row counts, the observed time
range, and the output digest. Preparation fails if the output already exists,
which prevents accidental replacement of experiment inputs.

## Baseline experiments

```bash
.venv/bin/python -m openrec_experiments.cli run \
  --config studies/baselines/ebnerd.json --output runs/ebnerd/fm-seed42
.venv/bin/python -m openrec_experiments.cli run \
  --config studies/baselines/kuairand.json --output runs/kuairand/fm-seed42
```

The `model` setting accepts `popularity`, `lr`, or `fm`. The popularity
baseline is a smoothed item click-rate estimate fitted on the training split.
`studies/baselines/ebnerd-content.json` enables OpenRec's cold-start content
features (hashed title/topic/subcategory and point-in-time content age) while
keeping the baseline split, labels and optimizer fixed for a controlled
ablation. Run it with both `--model lr` and `--model fm` to reproduce the full
behavior/content model matrix.
`studies/baselines/kuairand-content.json` provides the equivalent controlled
ablation using static basic video metadata. It deliberately excludes aggregate
video statistics and static user snapshots.
Formal baseline runs use seeds 42, 43, and 44, a separate output directory for
every run, and a preserved copy of the effective configuration.

The dates in the sample configurations define a local temporal holdout and must
be checked against the acquired dataset version before the first real-data run.
For EB-NeRD, the official validation data acts as the local test split and the
tail of the official training data acts as local validation. These local scores
must not be compared directly with the official hidden-test leaderboard.

LR and FM use OpenRec historical event counts, click rates, and scene features;
EB-NeRD additionally uses article category. They do not consume future
popularity aggregates, post-impression outcomes, static user-statistic
snapshots, or candidate position. Normalization statistics and categorical
vocabularies are fitted only on training samples. The selected epoch minimizes
validation LogLoss.

Every run produces:

- `manifest.json`: configuration, data and code versions, environment, split
  sizes, run status, and artifact digests.
- `metrics.json`: validation and test metrics, with KuaiRand results stratified
  by exposure policy and interface.
- `predictions.parquet`: validation and test predictions with sample and group
  identities.
- `learning_curve.json`: validation results for every epoch.
- `environment.txt`: the effective Python dependency environment.
- An LR/FM checkpoint and fitted `FeatureSpace`, or popularity parameters.

## Scope and research rules

The full protocol is defined in
[protocols/baseline-v1.md](protocols/baseline-v1.md). The current implementation
is an in-memory CPU reference runner. It calls the production feature
aggregation logic for each time bucket; it is not an incremental feature engine
or a large-scale training implementation.

Confirm the memory budget before running the full KuaiRand-1K dataset. Evaluation
feedback is frozen by default. Delayed replay represents an assumed feedback
visibility policy over public logs and must not be interpreted as an online CTR
measurement or as a simulation of user response.

Small synthetic datasets in the test suite validate pipeline behavior only;
they are not research results. Real-data acquisition, official scorer parity,
strong external baselines, statistical inference, and serving replay remain on
the [research plan](docs/RESEARCH_PLAN.md). A SOTA claim requires evidence under
the same protocol against strong baselines on an untouched test set.

Reviewable summaries from completed real-data runs are published under
[results](results/README.md). Each summary states its protocol and maturity;
preliminary local results are kept separate from official leaderboard claims.

## Exporting result tables

Use options such as `run --model lr --seed 43` to override the model and seed.
The effective values are still recorded in the run manifest.

```bash
.venv/bin/python -m openrec_experiments.cli report \
  --runs runs/ebnerd/fm-seed42 runs/kuairand/fm-seed42 \
  --output artifacts/baselines.csv
```

The report command verifies artifact digests before export. Every run occupies
one row and retains its protocol and dataset identity. Results from different
datasets or incompatible evaluation protocols are never averaged into a single
score automatically.
