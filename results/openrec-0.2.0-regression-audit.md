# OpenRec 0.2.0 experiment regression audit

Audit date: 2026-10-20. This checks local experiment compatibility and
reproducibility, not online serving performance or official leaderboard results.

## Outcome

Ranking quality is stable in the checked runs, with one reproducibility caveat:
EB-NeRD LambdaRank now uses early stopping and its probability LogLoss worsened
slightly. This is not a claim that every historical result reproduces exactly.

All six full ranking reruns use seed 42. Values below are individual runs,
not the historical three-seed means.

| Dataset / model | Test ranking metric | Historical | OpenRec 0.2.0 |
| --- | --- | ---: | ---: |
| EB-NeRD LR | Macro AUC | 0.7871249375 | 0.7871249375 |
| EB-NeRD FM | Macro AUC | 0.7865647379 | 0.7865647379 |
| EB-NeRD LightGBM | Macro AUC | 0.8220263857 | 0.8220467417 |
| KuaiRand LR | Standard / Random AUC | 0.7514223518 / 0.6677718897 | 0.7514223518 / 0.6677718897 |
| KuaiRand FM | Standard / Random AUC | 0.7517418126 / 0.6719491768 | 0.7517418126 / 0.6719491768 |
| KuaiRand LightGBM | Standard / Random AUC | 0.7545105062 / 0.6778074273 | 0.7545105062 / 0.6778074273 |

All validation/test metrics and all prediction-table columns match exactly for
the four LR/FM reruns. Each EB-NeRD prediction table contains **4,006,529**
rows; each KuaiRand table contains **7,721,457** rows.
KuaiRand LightGBM's largest score difference is `1.4654943925052066e-14`,
and the largest reported metric difference is `1.1102230246251565e-16`.
Its ranking AUC values are unchanged.

Machine-readable before/after metrics, prediction comparisons, manifest hashes,
and timings are in [openrec-0.2.0-regression-audit.json](openrec-0.2.0-regression-audit.json).
All new ranking artifacts passed their manifest checksums; the verified export
is `runs/release-0.2.0-audit/rerun-verified.csv`. Concurrent rerun timings are
recorded for provenance and are not a controlled performance benchmark.

### EB-NeRD LightGBM reproducibility caveat

- Test Macro AUC increased by `0.000020355957892292942`; MRR and NDCG also
  increased slightly.
- Test LogLoss increased from `0.31617593092541846` to
  `0.3192562418800183` (`+0.0030803109545998186`). LambdaRank probabilities are
  uncalibrated, but this remains a real difference in a reported metric.
- The largest per-sample score difference is `0.19108850874063954`; stable
  aggregate ranking metrics do not establish stable scores or threshold-based
  downstream behavior.
- The historical model has **500 trees**; the new model has **375**.
  The recorded configuration omits `early_stopping_rounds` and `eval_at`.
- Algorithm commit `92f8c68d8800e597f06fc5bae4612b2aa71c9960` introduced default
  early stopping of 50 rounds and changed evaluation cutoffs from `[5, 10]`
  to `[1, 5, 10, 20]`. That change predates the old experiments CI pin as well
  as the tested release. Thus unchanged JSON configuration does not imply
  unchanged effective training policy.
- Fitted user/item encoder specifications are identical. The first tree
  leaf-value difference occurs at zero-based tree 43 and is only
  `2.3245294578089215e-16`; the first split-feature/threshold difference is at
  tree 154. Floating-point training variation is also present, so early
  stopping should not be presented as a proven sole cause of every difference.
- Future controlled comparisons should explicitly record these training
  parameters. Reproducing the old fixed-500-tree policy requires disabling
  early stopping and retaining its evaluation cutoffs, followed by a separate
  rerun. This audit preserves the original configuration to expose the default
  change; it does not silently retune or replace the published baseline.

## Version and environment

- Distribution authority: `../openrec/release/openrec.json`, version `0.2.0`,
  status `released`.
- Tested `rec-algorithm`: `8cd58b8491470027a5e24ed1e4dd7cfd99d569fb`,
  matching the release manifest, with a clean working tree.
- Experiments CI previously pinned
  `25276f5a9b59d0e1267da76e6bd9823ba1aecccb`. The workflow and README now pin
  the released revision.
- Ranking reruns preserve each historical manifest's complete configuration,
  prepared data, split, seed 42, and CPU execution. Recorded package versions
  match the corresponding historical runs.
- New artifacts are isolated under `runs/release-0.2.0-audit/`; historical
  results and checkpoints are unchanged.

## Completed checks

- Experiments unit and synthetic integration suite: **32 passed**.
- Related algorithm feature, rank, and recall contract tests: **82 passed**.
- `pip check`: no broken requirements.
- **16 historical learned-model checkpoints** and their saved `FeatureSpace`
  files load under the release and produce finite outputs on a zero-input
  forward smoke check. This checks serialization compatibility only; real-data
  prediction comparison is separate. Details:
  `runs/release-0.2.0-audit/checkpoint-compatibility.json`.
- SHA-256 verification of all recorded artifacts in **17 historical ranking
  runs** passed: eight EB-NeRD common-feature runs and nine KuaiRand learned
  model runs. Export: `runs/release-0.2.0-audit/historical-verified.csv`.
- Prediction and score SHA-256 verification passed for **14 historical
  Music-CRS official-devset runs**, including candidate-reranking variants.
  This verifies retained artifact integrity, not a new official evaluation.
  Details: `runs/release-0.2.0-audit/historical-music-integrity.json`.
- Music-CRS and Synerise train-only retrieval reruns match the historical
  `retrieval-v4` results exactly, including all validation/test Hot and I2I+Hot
  metrics and item-coverage counts. Each split evaluates 5,000 queries.

| Dataset | I2I+Hot test Recall@20, before / after | Test MRR@20, before / after |
| --- | --- | --- |
| Music-CRS 2026 | 0.1334 / 0.1334 | 0.03628962650782311 / 0.03628962650782311 |
| Synerise 2025 | 0.0060 / 0.0060 | 0.002090295435558594 / 0.002090295435558594 |

## Commands

The six seed-42 ranking baseline directories are:

```text
ebnerd-192-shared-lr-seed42
ebnerd-192-shared-fm-seed42
ebnerd-stage14-past-engagement-pruned-rerun-lightgbm-seed42
kuairand-pit-global-lr-seed42
kuairand-pit-global-fm-seed42
kuairand-pit-global-lightgbm-binary-seed42
```

For each baseline, its `manifest.json` `config` object was copied without
overrides to `runs/release-0.2.0-audit/<baseline>.config.json`. The exact rerun
command, from `experiments/`, was:

```bash
.venv/bin/python -m openrec_experiments.cli run \
  --config runs/release-0.2.0-audit/<baseline>.config.json \
  --output runs/release-0.2.0-audit/<baseline>
```

Existing output directories are intentionally rejected by the runner. Choose
a fresh audit directory for a subsequent audit.

From `experiments/`:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
.venv/bin/python -m openrec_experiments.cli run \
  --config studies/baselines/music-crs-2026-retrieval.json \
  --output runs/release-0.2.0-audit/music-crs-2026-retrieval
.venv/bin/python -m openrec_experiments.cli run \
  --config studies/baselines/synerise-2025-retrieval.json \
  --output runs/release-0.2.0-audit/synerise-2025-retrieval
```

From `rec-algorithm/`:

```bash
../experiments/.venv/bin/python -m pytest -q \
  test/algorithm/feature \
  test/algorithm/rank/test_lightgbm.py \
  test/algorithm/rank/test_training.py \
  test/algorithm/rank/test_transformer.py \
  test/algorithm/rank/test_temporal_split.py \
  test/algorithm/recall/test_query_qwen_emb.py \
  test/algorithm/recall/test_query_embedding.py \
  test/algorithm/recall/test_bm25.py
```

## Limits and unavailable checks

- The broad algorithm test command
  `../experiments/.venv/bin/python -m pytest -q test/algorithm/feature
  test/algorithm/rank test/algorithm/recall` stopped during collection because
  `rec-algorithm/data/test/item.csv` is absent. The focused 82-test command above
  does not require that legacy fixture.
- Official Music-CRS devset evaluation was attempted with
  `official-music --config studies/baselines/music-crs-2026-official-devset.json
  --evaluator /tmp/openrec-music-crs-evaluator
  --output runs/release-0.2.0-audit/music-crs-2026-official-devset`.
  It stopped because the evaluator's `exp/ground_truth/devset.json` is absent.
  The retrieval result above uses its separate positive-only local protocol.
- The official Synerise six-task downstream evaluation, Music-CRS embedding
  and candidate-reranking experiments, Transformer experiments, and online
  rank-engine/cluster replay are outside the completed rerun coverage.
- Seed 42 reruns check release reproducibility; they do not re-estimate the
  historical three-seed variance or establish online uplift.
