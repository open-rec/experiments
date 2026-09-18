# Published experiment summaries

This directory contains small, reviewable summaries derived from immutable run
manifests. Raw datasets, predictions, checkpoints, and complete run directories
remain under Git-ignored paths.

Results are preliminary until the protocol document removes its draft status,
official scorer parity is established, and all declared strong baselines have
been reproduced.

The EB-NeRD-small summary also includes a clearly separated validation reference
from the supplied RecSys Challenge 2024 first-place source tree. External rows
are not added to the run-derived CSV and are not treated as protocol-matched
OpenRec experiments.

`ebnerd-large-to-small-v1.md` records the separate scale experiment that trains
on the large population with deterministic negative sampling and predicts the
complete small validation candidate sets. Its protocol differences are stated
next to the result and it is not merged into the baseline-v1 comparison table.
