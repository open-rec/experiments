# Baseline protocol v1 (draft until real-data audit)

## Samples and targets

Canonical fields: sample_id, group_id, user_id, item_id (strings), timestamp
(integer UTC milliseconds), label (0/1), policy, scene. A group has exactly one
user, timestamp and policy. Duplicate sample identities and missing fields fail.

EB-NeRD: expand each labeled impression into its original in-view candidates.
A clicked candidate has label 1; remaining candidates have label 0. Do not sample
negatives, filter hard candidates, use impression position as a feature, or accept
unlabeled hidden test as negative data. Article category is treated as available
metadata (a simplifying assumption, not a reconstructed metadata revision log).
History parquet is not yet ingested; the initial history is empty and grows from
supplied behavior logs. This limits comparisons against history-aware baselines.

KuaiRand: each log row is one observation, not an invented ranking request.
is_click is used as supplied: its interpretation differs by tab (click vs valid
play). Report policy and policy/tab strata; do not call the pooled target CTR.
Keep random observations for test, excluded from training, validation selection,
and feature history. This is a specified protocol, not a universal unbiased OPE
estimator. Do not deduplicate legitimate repeat events by user/item pair.

## Time and feedback

Global UTC boundaries define [train_start, validation_start),
[validation_start, test_start), [test_start, test_end). Equal timestamps never
cross a boundary. Older supplied observations may warm features, but do not train
model parameters. Random observations never contribute history.

EB-NeRD-small uses the observed 07:00 UTC dataset boundary: the first four days
of official training data are local training, its final three days are local
validation, and the official validation bundle is local test. This preserves
the provider split instead of mixing the last training hours into test.

KuaiRand-1K uses the first eleven days of the earlier standard-log bundle for
training and its final three days for validation. The later standard and random
bundles form test from 2022-04-22 local calendar time onward. Random exposure is
therefore retained only for final evaluation and never participates in model or
epoch selection. The small number of records outside the declared local-day
boundaries is excluded and must be reported in the data audit.

Feature visibility cutoff for timestamp t is:
`floor(t / update_interval_ms) * update_interval_ms - feedback_delay_ms`.
Only history strictly earlier than this cutoff can contribute. Frozen mode caps
cutoffs at validation_start - feedback_delay_ms. Delayed replay allows previously
observed validation/test feedback only after its configured delay and snapshot
boundary; current and future labels remain excluded. Real feedback/ingestion time
is not fully observable in these datasets: delay is an experimental assumption.

The baseline maps each observation to one mutually exclusive click/expose label
for OpenRec aggregation. This matches its negative-label convention; it is not
an attempt to reconstruct a separate raw impression plus click event stream.
Compare serving features only after applying this same mapping.

## Features, fitting and selection

Call OpenRec aggregate_event_features, FeatureSpace.for_model, LRModel/FMModel.
Initial selected features: user/item event_count and event_click_rate, item.scene,
and item.category when available. No identity one-hot, static user-stat snapshot,
full-period video statistics, future content aggregates or post-outcome fields.
FeatureSpace fits on training rows only. Adam/BCE, deterministic CPU execution;
checkpoint selection uses validation LogLoss. Do not select seeds/configurations
using test metrics. Popularity uses only training labels with prior smoothing 10.

The EB-NeRD content ablation adds the item schema's title, topic tags,
subcategory and publication time. Text and unbounded multi-value fields use the
fixed-width signed BLAKE2b feature hash persisted by OpenRec FeatureSpace;
`content_age_hours` is calculated from publication time at each impression time.
No full-period article engagement statistics are used. The same content
materializer and fitted sidecar are consumed by offline training and rank-engine.

## Evaluation

Global AUC and LogLoss are named separately from impression-macro AUC.
EB-NeRD: macro impression AUC, NDCG@5/10 and mean positive reciprocal rank
(the average reciprocal rank of all clicked candidates, not first-positive RR).
Each impression must have both classes; fail instead of silently discarding groups.
Ties use ascending sample_id. This differs potentially from the official scorer's
sorting tie semantics: official-scorer parity is a release gate, not yet certified.
KuaiRand: global and policy/tab AUC and LogLoss; single-class AUC is null with
sample/positive counts retained. No artificial grouped ranking metrics.

## Publication gates

Freeze data hashes, protocol, code revisions and tuning budget before final test.
Compare same-feature model baselines separately from feature-rich task baselines.
Run at least three seeds for stochastic models; report all seeds, mean/std, and
paired user-cluster bootstrap intervals (implementation pending). Preserve one
untouched final evaluation set. Report exclusions, OOV/cold-start rates and data
coverage. Official hidden test scores require the official scoring/submission
protocol. Never compare small vs large, validation vs hidden test, or differing
candidate sets as if they were the same benchmark.
