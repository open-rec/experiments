# OpenRec Experiments

Reproducible recommender-system research built on OpenRec. The initial datasets
are **EB-NeRD**, **KuaiRand-1K**, **RecSys 2025 Synerise**, and **RecSys 2026 Music-CRS**. The project first establishes trustworthy
baselines, then studies training-serving feature consistency, feature freshness,
and the trade-off between recommendation quality and system cost.

**Current status: Music-CRS recommendation metrics have been evaluated on the
official devset; Synerise user profiles have been compared with the organizer's
example baseline on its official local six-task pipeline. Neither result is an
official leaderboard submission.** The first version reuses the
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

Local installation commands use the Aliyun mirror; GitHub Actions uses default
PyPI access. The runner explicitly loads the
`rec-algorithm` source tree specified by each experiment configuration, verifies
the actual import path, and records repository commits, working-tree status, a
diff digest, and environment versions. Versions used in the current workspace
are listed in `requirements-tested.txt`; this file is not a complete
cross-platform lock file.

### CI installation

CI pins `rec-algorithm` to `9003733844c4ea84e2024bea8e5631e2ec3bb084`,
which includes the content-feature, LightGBM and Transformer modules used by
the current experiment runner. Use this revision to reproduce CI locally;
update the pinned revision together with new algorithm-contract dependencies.

CI follows the rank-engine installation pattern: cache pip downloads, install
recorded dependencies before the editable project, and upgrade pip first. The
cache key includes `pyproject.toml` and `requirements-tested.txt`; the latter
keeps major dependency versions stable without changing the experiment runtime
to CPU-only PyTorch. CI uses pip's default PyPI index rather than the Aliyun
mirror. `requirements-tested.txt` contains versions only; local installations
can select Aliyun with `--index-url` as shown above.

Installation steps have separate names, verbose project-install output, bounded
network retries and disabled pip version checks. Superseded runs are cancelled.
These changes reduce repeat downloads and make stalled installation phases easier
to identify; cold-cache timing and mirror connectivity must be verified on the
GitHub runner, not inferred from local tests.

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

The two newer datasets have a separate positive-only local retrieval protocol.
See [Synerise 2025](datasets/synerise_2025/README.md) and
[Music-CRS 2026](datasets/music_crs_2026/README.md) for acquisition, commands,
and the limits of comparison with each official challenge. Their runner uses
OpenRec's `Hot` and `ItemBasedI2I` recall implementations, with train-only
models and observed prior events as triggers. It reports full train-catalog
Recall@20 and MRR@20; no unobserved item is labeled a negative exposure.

```bash
.venv/bin/python -m openrec_experiments.cli run \
  --config studies/baselines/ebnerd.json --output runs/ebnerd/fm-seed42
.venv/bin/python -m openrec_experiments.cli run \
  --config studies/baselines/kuairand.json --output runs/kuairand/fm-seed42
```

The `model` setting accepts `popularity`, `lr`, `fm`, `lightgbm`, or `transformer`. The popularity
baseline is a smoothed item click-rate estimate fitted on the training split.
`studies/baselines/ebnerd-content.json` enables OpenRec's cold-start content
features (hashed title/topic/subcategory and point-in-time content age) while
keeping the baseline split, labels and optimizer fixed for a controlled
ablation. Run it with both `--model lr` and `--model fm` to reproduce the full
behavior/content model matrix.
`studies/baselines/ebnerd-semantic.json` replaces the title hash with frozen
multilingual E5 title vectors for LR/FM. `studies/baselines/ebnerd-transformer.json`
uses those vectors for both candidates and the official click history, applies
candidate-aware Transformer attention, and fuses OpenRec's existing global
features. Generate the ignored embedding artifact with:

```bash
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python -m openrec_experiments.cli \
  embed-titles --articles data/raw/ebnerd_small/articles.parquet \
  --output data/processed/ebnerd-small-title-e5.parquet
```
`studies/baselines/kuairand-content.json` provides the equivalent controlled
ablation using static basic video metadata. It deliberately excludes aggregate
video statistics and static user snapshots. KuaiRand has no suitable text field
for a semantic-language ablation. `studies/baselines/kuairand-transformer.json`
therefore represents items with OpenRec's structured video type, upload type,
and tag features, and attends to the last 50 eligible clicks. Random-policy
clicks never enter history, and frozen evaluation exposes no post-training
feedback.
Formal baseline runs use seeds 42, 43, and 44, a separate output directory for
every run, and a preserved copy of the effective configuration.

The earlier body-embedding and large-to-small body-embedding runs remain in the
results directory as historical evidence, but their active study configurations
and embedding command have been removed. EB-NeRD body coverage and measured
gain did not justify carrying that feature into the production roadmap.

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
- An LR/FM/Transformer checkpoint and fitted `FeatureSpace`, or popularity
  parameters.

## Scope and research rules

The full protocol is defined in
[protocols/baseline-v1.md](protocols/baseline-v1.md). The current implementation
is an in-memory reference runner with optional CUDA training. It calls the
production feature aggregation logic for each time bucket; it is not an
incremental feature engine or a large-scale training implementation.

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
The feature-engineering path that raises the causal EB-NeRD LightGBM baseline
from 0.680 to 0.822 impression-macro AUC is documented in
[EB-NeRD ranking optimization](docs/EBNERD_OPTIMIZATION.md).

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
