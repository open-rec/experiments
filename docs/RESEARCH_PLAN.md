# OpenRec research plan

## Objective and scope

Develop publishable, reproducible evidence on EB-NeRD and KuaiRand-1K using
OpenRec as the engineering foundation. Seek competitive results and an original,
generalizable contribution; no SOTA or acceptance claim before evidence.
Candidate question: how should feature updates be scheduled under a fixed budget
while preserving training-serving consistency? This is a hypothesis to investigate,
not an already implemented novel method.

## Milestones and acceptance gates

| Stage | Work | Acceptance evidence | Status |
|---|---|---|---|
| M0 | Dataset adapters, explicit protocol, OpenRec LR/FM and popularity runner, lineage | Synthetic tests for schema, splitting, leakage, metrics, training artifacts | Implemented; see local test results |
| M1 | Acquire real data, audit label/time semantics, history adapter, official scorer parity | Data report, frozen splits, reference scorer agreement including ties/multiclick | Pending |
| M2 | Three-seed baseline suite, stronger external models, fair tuning budget | Reproducible baseline table and learning curves, independent final test | Pending |
| M3 | Fixed-period/frozen/update-budget ablations; propose adaptive method | Quality/cost tradeoff, equal-budget comparisons and paired intervals | Pending |
| M4 | OpenRec serving replay, incremental scaling and fault experiments | Feature/score parity, QPS/P95/P99/error rate, recovery and memory | Pending |
| M5 | Paper and reproducible artifact | Tables/figures rebuilt from manifests, clean pinned refs, documented limitations | Pending |

## Baseline matrix

EB-NeRD: smoothed popularity, OpenRec LR/FM, same-feature LightGBM, and a
history/content-aware news baseline (e.g. NRMS). KuaiRand: popularity, LR/FM,
LightGBM, DeepFM; add DIN only if sequence modeling is part of the contribution.
External models are NOT implemented in M0. Separate model comparisons with shared
features from full-method comparisons with different representations. Investigate
current task-specific strong baselines during M1 before freezing M2's comparison.

Three seeds initially: 42/43/44. Hyperparameters selected on validation only.
Every run records the budget, features and code/data identities. Do not tune by
repeatedly reading final test. The current runner emits test results for baseline
verification; a locked final-test release workflow must precede method search.

## Scaling and resources

Begin with EB-NeRD-small and a clearly labeled development subset for debugging;
final 1K claims require the full selected protocol. Full 1K contains millions of
observations and the reference materializer recomputes snapshots in memory.
Measure RAM before full runs. EB-NeRD-large is a later scale experiment.
Hardware, GPU availability, disk quota and runtime budget are not yet confirmed.
Do not start an unbounded hyperparameter search or paid cloud jobs implicitly.

## Research risks

- Novelty: a conventional stack and baseline metric table are not a research contribution.
- Temporal validity: metadata and feedback arrival times are not complete revision logs.
- Selection bias: random exposure is a separate evaluation population, not proof of universal causal uplift.
- Serving claims: reference offline code does not establish online latency or HA.
- Dataset limitations: neither dataset supplies full entity mutation/delete histories;
  fault/late-event injections must be labeled controlled synthetic interventions.
- Reproducibility: working-tree changes and untracked source are recorded, but paper
  releases require clean immutable companion refs and archived source/environment.

## Immediate next actions

1. Acquire licensed datasets, run prepare and inspect row/time/class/policy reports.
2. Verify official EB-NeRD metric formulas and tie handling against pinned scorer.
3. Add pre-window click histories without fabricating exposure denominators.
4. Run popularity/LR/FM using real data under the confirmed memory budget.
5. Establish strong external baselines, uncertainty estimates, then choose the
   research contribution based on measured weaknesses and update-cost profiles.
