# Feature Experiment Roadmap

OpenRec classifies features independently by entity ownership (`group`), user-facing
purpose (`family`), and applicable recommendation scene (`scenes`). The catalog is
the source of truth; experiment configurations record the exact selected IDs.

The research sequence is cumulative and each stage retains an ablation against
the preceding stage:

1. **Context**: candidate position, impression hour/day, device, subscription,
   traffic source, and request platform.
2. **User interests**: causal category/topic/entity preferences over short and long
   windows, plus interest diversity and recent-history length.
3. **Interactions**: user-category, user-topic, user-entity, freshness-preference,
   and device-position crosses. Tree models consume paired values; LR/FM get
   explicit bounded crosses where their representation requires them.
4. **History representations**: clicked-title/entity pooling and
   candidate-to-history similarity, followed by sequence models using the same
   global feature set.

Every feature must declare online/offline materialization, its event-time cutoff,
default behavior, family, and scenes. A feature enters the shared model matrix only
after point-in-time parity tests pass. Primary comparisons use the unchanged
temporal split and EB-NeRD impression macro AUC; random or transductive features
must be reported separately.

The staged targets are 0.70 from contextual and temporal features, 0.75 from user
interest and interaction features, and then 0.80-0.85 with richer histories,
large-data training, and model ensembles. These are research gates rather than
promised scores.

## Completed families

### Contextual v1

The first family includes candidate position/count, relative position, cyclical
request time, device type, subscription state, and authentication state. On the
seed-42 EB-NeRD-small test split it changes macro AUC by -0.002071 for LR,
+0.001034 for FM, and +0.016074 for LightGBM. It remains an optional family.
Post-request read time, scroll percentage, and future behavior are explicitly
excluded as leakage.

### Interaction v1

This family contains visible user-category click count/share/seen/recency and
visible user-item click count/seen. Added cumulatively after contextual v1, it
raises seed-42 EB-NeRD-small test macro AUC to 0.606023 for LR, 0.597893 for FM,
and 0.709451 for LightGBM. Because all three improve, it is the first candidate
for full incremental online materialization after the experiment incubation
stage.

### Relative temporal v1

This request-candidate family contains log age, within-request freshness rank,
gap from the freshest candidate, tied-freshest state, and six/twenty-four-hour
publication flags. Cumulatively it raises seed-42 test macro AUC to 0.690096 for
LR, 0.686220 for FM, and 0.720666 for LightGBM. It is label-free and can be
computed from request time, candidate publication time, and the candidate set.

### Champion-derived statistical families

News traffic/CTR trends, long-term user profiles, Topic/entity affinity, and
candidate statistical ranks have now been evaluated sequentially. Their
LightGBM test macro AUCs are 0.719381, 0.721310, 0.722018, and 0.722214 against
the previous 0.720666 best. The trend family regresses under daily frozen
snapshots, so its implementation and catalog entries were removed. The other
three remain optional;
their combined small-data gain is modest, while candidate ranks materially help
LR/FM.

### Causal session exposure

This family includes user impression counts over one and twenty-four hours,
elapsed time since the previous impression, and candidate recurrence over the
previous 1/2/5/10/20 impressions. It raises the cumulative LightGBM test macro
AUC from 0.722214 to **0.741945**. A prior-click transition extension scores
0.740863; its implementation and catalog entries were removed.

### Fixed request-time user history

The official EB-NeRD history snapshots supply candidate recurrence and
category/topic affinity that predate each evaluated request. Added to the
session branch, these features raise LightGBM test macro AUC from 0.741945 to
**0.754131**. A time-decay extension scores 0.752433 and was removed. This
result makes complete history materialization the next
production feature priority.

### Causal past candidate state

Candidate exposure count/share/rank over five minutes, one hour, twenty-four
hours, and full history raises test macro AUC from 0.754138 to 0.796423.
Accumulated engagement from completed past impressions raises it to
**0.822243**. These features replay exposure and feedback strictly before the
current timestamp. A title-semantic history extension scores 0.822132 and was
removed.

The active matrix was pruned from 197 to 192 dimensions by removing five
zero-use redundant columns. Its retrained test macro AUC is 0.822026 versus
0.822243 before pruning.
