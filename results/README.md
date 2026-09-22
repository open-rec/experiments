# Published experiment summaries

This directory contains small, reviewable summaries derived from immutable run
manifests. Raw datasets, predictions, checkpoints, and complete run directories
remain under Git-ignored paths.

Results are preliminary until the protocol document removes its draft status,
official scorer parity is established, and all declared strong baselines have
been reproduced.

`music-crs-2026-official-devset-v1.md` compares OpenRec Hot + ItemBasedI2I with
the organizer's published devset baselines using the same recommendation
metrics. The official Popularity prediction artifact passes scorer parity.

`synerise-2025-official-local-v1.md` compares OpenRec user profiles with the
organizer's example aggregated-feature profiles on the same official local
split and fixed downstream trainer. Its local scores remain separate from the
published final leaderboard's undisclosed target windows.

`music-crs-2026-retrieval-v1.md` records a separate positive-only OpenRec recall
study over organizer conversation turns. Its target is the logged music turn,
not the official challenge's listener-preference and response-quality target.

`synerise-2025-retrieval-v1.md` records the same local OpenRec recall comparison
on the organizer's raw purchase data. It is separate from the official
Universal Behavioral Profiles challenge.

Both newer study reports include separately sourced official baseline data.
The corresponding `datasets/*/official_baselines.csv` files preserve the
published figures without merging incompatible evaluation protocols.

`kuairand-1k-pit-global-v1.md` reports Popularity, LR, FM, and binary LightGBM
using only the stable OpenRec global point-in-time feature catalog. Its
companion CSV preserves per-seed metrics and excludes dataset-specific content
or EB-NeRD-only features.

The EB-NeRD-small summary also includes a clearly separated validation reference
from the supplied RecSys Challenge 2024 first-place source tree. External rows
are not added to the run-derived CSV and are not treated as protocol-matched
OpenRec experiments.

`ebnerd-large-to-small-v1.md` records the separate scale experiment that trains
on the large population with deterministic negative sampling and predicts the
complete small validation candidate sets. Its protocol differences are stated
next to the result and it is not merged into the baseline-v1 comparison table.
The body-embedding input used by that historical run has been retired because
EB-NeRD does not provide reliable body text for the entity population and its
measured gain did not justify the storage and inference cost.

`ebnerd-small-recsys2024-reference.md` records an isolated execution of the
RecSys Challenge 2024 winner repository's supplied Kami small configurations.
It reports complete-validation impression AUC for LightGBM, CatBoost, and their
tree-only weighted mean, and separates these results from the repository's
large-data neural ensemble reference.

`ebnerd-small-lightgbm-v1.md` is the first protocol-matched OpenRec tree
baseline. It compares the existing content selection with an expanded selection
of already implemented causal aggregates.

`ebnerd-small-context-v1.md` records the first categorized feature-family
ablation across LR, FM, and LightGBM. Context is useful for the tree ranker but
is not enabled universally because it does not improve all three models.

`ebnerd-small-model-comparison-v1.md` is the three-seed, same-feature comparison
of OpenRec LR, FM, and LightGBM on EB-NeRD-small. Its companion CSV retains each
individual run instead of publishing only aggregate values.

`ebnerd-small-common-features-v2.md` reports Popularity, LR, FM, and LightGBM
under one EB-NeRD-small protocol, matching the structure of the KuaiRand
summary. It applies the current strongest common causal feature configuration
to the learned models and separates global catalog membership from production
online materialization readiness.
