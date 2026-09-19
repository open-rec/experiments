# EB-NeRD ranking optimization: from 0.680 to 0.822 AUC

## Purpose

This document records the feature-engineering path that raised OpenRec's
EB-NeRD-small LightGBM ranking baseline from **0.680412** to a peak test
impression-macro AUC of **0.822243**. The active pruned feature set scores
**0.822026** with five fewer inputs.

The main lesson is that the largest gains did not come from a larger model or
raw semantic embeddings. They came from representing what happened before the
request at the same granularity as the ranking decision: candidate exposure,
within-request relative statistics, complete user history, and engagement on
completed past impressions.

These are local temporal-holdout results, not official leaderboard scores. The
public EB-NeRD validation split is used as the local test set. The winning-code
small LightGBM reproduction scores 0.838472, but its 103 inputs include future
impression and transductive statistics that are unavailable to a causal online
ranker. Comparisons must preserve that distinction.

## Evaluation protocol

All results below use seed 42 and the same candidate-complete evaluation:

- Train: 2023-05-18 07:00 UTC to 2023-05-22 07:00 UTC.
- Validation: 2023-05-22 07:00 UTC to 2023-05-25 07:00 UTC.
- Test: 2023-05-25 07:00 UTC to 2023-06-01 07:00 UTC.
- Primary metric: ROC AUC inside each impression, averaged across impressions.
- Test coverage: 244,647 impressions and 2,928,942 displayed candidates.
- Ranker: LightGBM with LambdaRank, 500 trees, learning rate 0.05, 63 leaves.
- Label-dependent aggregates use the declared feedback cutoff.
- Exposure features read only requests with timestamps strictly before the
  current request. Requests at the same timestamp are updated as one batch.
- Engagement from the current impression is never used to score that
  impression. It becomes visible only to later requests.

Run manifests preserve data hashes, source state, parameters, environment, and
the exact feature names. The active configuration is
[`ebnerd-feature-stage14-past-engagement.json`](../studies/baselines/ebnerd-feature-stage14-past-engagement.json).

## Improvement sequence

| Stage | Added information | Test macro AUC | Increment |
|---|---|---:|---:|
| LightGBM content baseline | OpenRec article attributes and basic aggregates | 0.680412 | — |
| Request context | Position, candidate count, device and request time | 0.696485 | +0.016073 |
| User-candidate interaction | Visible category and item affinity | 0.709451 | +0.012966 |
| Relative freshness | Candidate age and freshness relative to the request | 0.720666 | +0.011215 |
| Long-term profile | User activity and consumed-content statistics | 0.721310 | +0.000644 |
| Topic/entity affinity | Candidate overlap with visible interests | 0.722018 | +0.000708 |
| Candidate-relative ranks | Within-request ranks and centered values | 0.722214 | +0.000196 |
| Session exposure | Recent request load and repeated candidates | 0.741945 | +0.019731 |
| Official fixed history | Candidate/category/topic history affinity | 0.754131 | +0.012186 |
| Causal past exposure | Candidate exposure over 5m, 1h and all history | 0.796136 | +0.042005 |
| 24h exposure and momentum | 24h state and short/long exposure change | 0.796423 | +0.000287 |
| Causal past engagement | Past `read_time / candidate_count` state | **0.822243** | **+0.025821** |
| Pruned active matrix | Five zero-use redundant columns removed | **0.822026** | -0.000217 |

The cumulative gain over the original LightGBM content baseline is **+0.141614**
for the active matrix and **+0.141831** at the measured peak.

## Why the effective features worked

### Request context and relative features

Impression AUC evaluates ordering inside one candidate set. Features that are
constant for the request may predict whether an impression is generally easy
or click-heavy without helping order its candidates. Candidate position,
relative freshness, and within-request ranks convert raw signals into direct
comparisons among candidates in the same impression.

This was especially important for LR and FM, but it also gave LightGBM a useful
inductive bias. The model no longer had to infer a candidate's freshness rank
from independent rows.

### Explicit user-candidate affinity

Category and item history counts express whether this candidate matches the
user's visible interests. FM can represent pairwise interactions, but sparse
IDs and coarse aggregates do not automatically produce a reliable causal
history statistic. Explicit affinity features improved LR, FM, and LightGBM,
confirming that the signal was in the data rather than an artifact of one
model family.

### Session exposure

Users in EB-NeRD frequently issue several impressions in a short period.
Recent impression count, elapsed time since the previous impression, and
candidate recurrence over the previous 1/2/5/10/20 impressions raised AUC by
0.019731. These features capture fatigue, repeated recommendation, and short
session continuity that static user profiles cannot represent.

### Official fixed history

The official history snapshots cover activity before the experiment window.
They prevent the model from treating every user as cold at the beginning of the
local train period. Candidate recurrence, category share, and topic share from
these snapshots added 0.012186 AUC.

This result is also an engineering lesson: reconstructing history only from the
current training slice loses useful serving-time state even when the model and
optimizer are correct.

### Causal past candidate exposure

This was the largest single improvement. For each candidate, OpenRec now
tracks exposure count at both global and user-candidate scope over 5 minutes,
1 hour, 24 hours, and all previous requests. Each signal produces:

- a log-scaled absolute value;
- its share inside the current candidate set;
- its percentile rank inside the current candidate set.

Short/long momentum compares 5-minute with hourly activity and hourly with
daily activity. The implementation excludes every request at the current
timestamp until all candidates at that timestamp have been scored.

These features work because news recommendation is driven by fast-changing
attention. Static article metadata says what an article is; causal exposure
state says how the platform and user have interacted with it immediately
before ranking.

### Engagement on completed past impressions

Exposure alone cannot distinguish passive repetition from meaningful reading.
For completed past requests, candidate engagement is weighted as:

```text
read_time / candidate_count
```

Only impressions with scroll feedback contribute, matching the winning-code
feature semantics. The same 5-minute, 1-hour, and all-history global/user
statistics are transformed into log values, within-request shares, and ranks.

This family added 0.025821 AUC on top of the already strong exposure model. It
shows that the most useful behavioral signal is not the current label or current
read time, but properly delayed engagement from requests that have already
finished.

## Experiments that did not help

| Attempt | Result | Interpretation and action |
|---|---:|---|
| Existing OpenRec aggregate expansion | 0.681071 | Only +0.000659; highly correlated coarse statistics |
| Frozen daily news trends | 0.719381 | -0.001285; windows became stale under daily frozen snapshots; removed |
| Previous-click transition | 0.740863 | -0.001082; sparse first-order transitions; removed |
| Champion-sized trees | 0.732986 | -0.008959; capacity could not replace missing information; config removed |
| Fixed-history time decay | 0.752433 | -0.001698; simple exponential decay added noise; removed |
| 20% EB-NeRD-large sample | 0.710047 | Sampling broke continuous histories and query-size parity; not a valid scale proxy |
| Fixed-history title semantic similarity | 0.822132 | -0.000111 test and -0.002214 validation; removed |
| Five-column importance pruning | 0.822026 | -0.000217 accepted for a smaller 192-dimensional active matrix |

Raw title sentence embeddings had little effect because semantic meaning alone
does not encode current popularity, repetition, user exposure, or delayed
engagement. Candidate-to-history semantic similarity was a more appropriate
interaction, but it still did not improve the strong behavioral model.

## Feature cleanup and active contract

Failed feature families are removed from executable code and the shared
catalog; their measured results remain in reports to prevent accidental repeat
work. The active catalog is version 13 with 160 features. The stage-14 model
uses 192 encoded inputs after removing these redundant zero-use columns:

- `interaction.user_category_seen`;
- `interaction.user_item_seen`;
- `temporal.is_freshest`;
- `temporal.published_within_6h`;
- `interaction.candidate_fixed_history_seen`.

The corresponding continuous counts, recencies, and ages remain available, so
the information loss is small. Retraining changed test AUC by -0.000217.

## Reusable OpenRec lessons

1. Optimize the feature cutoff before optimizing model capacity. A deeper tree
   did not compensate for missing request-time state.
2. Match features to the metric. Impression AUC rewards candidate-relative
   values, not request-level constants.
3. Keep exposure and feedback semantics separate. Exposure is immediately
   observable; click/read feedback requires a declared delay.
4. Preserve pre-window history. Otherwise established users appear cold at the
   beginning of every training slice.
5. Use absolute, relative, and trend views together. Counts, candidate-set
   ranks, and short/long momentum answer different questions.
6. Treat semantic features as interactions with user history, not as isolated
   item vectors. Even then, verify incremental value after behavioral features.
7. Remove failed families from the active contract while preserving their
   ablation results.
8. Cache point-in-time state. The current reference runner scans 5.5 million
   candidates separately for exposure and engagement; production and rapid
   experimentation should materialize both in one partitioned pass.

## Remaining gap to the reference winner

The active causal model scores 0.822026 versus 0.838472 for the reproduced
winning-code small LightGBM path, a gap of 0.016446. The reference pipeline also
uses future-impression and transductive global statistics, plus richer
TF-IDF/SVD similarities across title, subtitle, body, topics, categories,
entities, and NER clusters.

The next defensible steps are causal category/entity similarity, consolidated
and cached past-state materialization, multi-seed validation, and full
EB-NeRD-large training with complete or representative ranking groups. Future
impression features should remain a separately labeled challenge-only protocol,
not enter the online OpenRec baseline.

Detailed metric tables are retained in
[`results/ebnerd-small-lightgbm-v1.md`](../results/ebnerd-small-lightgbm-v1.md),
and the winner reproduction is documented in
[`results/ebnerd-small-recsys2024-reference.md`](../results/ebnerd-small-recsys2024-reference.md).
