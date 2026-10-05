# experiments v0.1.0

Released: 2026-10-05

Reproducible recommendation research. First coordinated OpenRec source release.

## Features

- Dataset adapters and experiment protocols for EB-NeRD, KuaiRand-1K, RecSys 2025 Synerise and RecSys 2026 Music-CRS.
- Shared ranking/retrieval evaluation, temporal splits, feature builders and baseline comparisons.
- Popularity, LR, FM, LightGBM and Transformer experiments with optional semantic features.
- Per-run configuration, environment, source/data provenance, artifact hashes and report generation.
- Documented real-data results and official-protocol comparisons with explicit evaluation boundaries.

## Installation and compatibility

Requires Python 3.10+ (3.12 recommended) and a compatible sibling rec-algorithm checkout. This release pins the tested algorithm source in CI.

Datasets are obtained separately under their providers' terms. Experimental models are not automatically published to serving. Local/devset metrics are not official leaderboard submissions or production CTR measurements; keep dataset protocols and splits separate.

## Validation and known boundaries

See this repository's README for build/test commands and deployment requirements. The coordinated release's [validation record](https://github.com/open-rec/openrec/blob/v0.1.0/release/VALIDATION.md) distinguishes checks executed for this release from historical integration evidence.

This initial release establishes a versioned source baseline. Source archives and checksums are published; external package registries and container registries are not populated by the source-release workflow. Upgrade the complete compatible distribution, retain data/checkpoints/artifacts, and preserve prior component refs for rollback.
