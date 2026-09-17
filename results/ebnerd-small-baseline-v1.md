# EB-NeRD-small baseline v1

Status: preliminary local temporal holdout; not an official leaderboard result.

The prepared dataset contains 5,514,689 candidates in 477,534 impressions,
18,827 users, 5,776 candidate articles, and 479,385 positive candidates. The
behavior-only projection used by the first three baselines has SHA-256
`7835be6d67f6bdcfe4b627a573df2b4cc69276f794d625b19cd62d474edf59fc`;
the content projection preserves those rows and adds article metadata.

The local test set is the complete official validation bundle. The first four
days of the official training bundle are local training and its final three days
are local validation. Boundaries follow the observed 07:00 UTC dataset day.

| Model | Seeds | Macro AUC | Macro MRR | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Popularity | 42 | 0.56467 | 0.32585 | 0.37074 | 0.45081 |
| OpenRec LR | 42/43/44 | 0.55039 ± 0.00177 | 0.33952 ± 0.00196 | 0.37962 ± 0.00216 | 0.45630 ± 0.00198 |
| OpenRec LR + content | 42/43/44 | 0.58197 ± 0.00095 | 0.35812 ± 0.00115 | 0.40192 ± 0.00137 | 0.47571 ± 0.00122 |
| OpenRec LR + semantic title | 42/43/44 | 0.58132 ± 0.00131 | 0.35870 ± 0.00162 | 0.40119 ± 0.00144 | 0.47573 ± 0.00141 |
| OpenRec LR + semantic body / title-hash fallback | 42/43/44 | 0.58396 ± 0.00126 | 0.36032 ± 0.00149 | 0.40260 ± 0.00157 | 0.47758 ± 0.00121 |
| OpenRec FM | 42/43/44 | 0.55864 ± 0.00058 | 0.34696 ± 0.00074 | 0.38779 ± 0.00082 | 0.46372 ± 0.00072 |
| OpenRec FM + content | 42/43/44 | 0.57292 ± 0.00297 | 0.35098 ± 0.00282 | 0.39328 ± 0.00275 | 0.46889 ± 0.00255 |
| OpenRec FM + semantic title | 42/43/44 | 0.57741 ± 0.00251 | 0.35635 ± 0.00311 | 0.39892 ± 0.00355 | 0.47391 ± 0.00267 |
| OpenRec FM + semantic body / title-hash fallback | 42/43/44 | 0.58603 ± 0.00285 | 0.36283 ± 0.00184 | 0.40661 ± 0.00267 | 0.47996 ± 0.00216 |
| OpenRec history Transformer | 42/43/44 | **0.61403 ± 0.00743** | **0.38534 ± 0.00675** | **0.43267 ± 0.00719** | **0.50209 ± 0.00564** |

## RecSys Challenge 2024 winner reference

The first-place Team :D repository checked out at
`../tmp/recsys-challenge-2024-1st-place` records the following independently
produced validation result:

| System | Evaluation slice | Macro AUC | Macro MRR | NDCG@5 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| Team :D four-model weighted ensemble | Official validation impressions marked `in_small` | **0.87910** | not reported | not reported | not reported |

The exact AUC in
`kfujikawa/src/exp/v8xxx_ensemble/v8004_015_016_v1170_v1174.py` is
`0.8791009544954949`. The script averages per-impression
`sklearn.metrics.roc_auc_score`, then filters evaluation to `in_small`. It
combines two Kami tree models (LightGBM LambdaRank and CatBoost) and two
Kfujikawa neural rankers. This is the strongest small-slice validation number
recorded in the supplied source tree, but it is an intermediate ensemble rather
than the final challenge submission: the final pipeline additionally uses more
neural predictions, pseudo-label training, optimized weighted averaging and a
LightGBM LambdaRank stacker. The repository does not record a final hidden-test
score, MRR, or NDCG values, so none are inferred here.

This row is an external reference, not another baseline-v1 run. It evaluates
the small validation slice, but its models are trained with the EB-NeRD-large
pipeline and use a different feature-availability policy. The OpenRec row is
trained only on EB-NeRD-small with a frozen, serving-oriented temporal cutoff.
Consequently, the raw AUC difference must not be interpreted as an isolated
model-architecture effect or as an official leaderboard comparison.

### Gap analysis

The recorded winner reference exceeds the OpenRec Transformer mean by
`0.26507` Macro AUC (`0.87910 - 0.61403`), a 43.2% relative increase or a 68.7%
reduction of the distance to perfect AUC. The source comparison indicates the
following main causes, in approximate order of importance:

1. **Training scale and supervision.** Team :D trains the referenced
   `large067` tree models and large neural models on EB-NeRD-large, while only
   restricting evaluation to impressions marked `in_small`. OpenRec trains on
   the much smaller EB-NeRD-small training bundle. The external system therefore
   has substantially more users, impressions and article interactions from
   which to estimate popularity, user preferences and article statistics.
2. **Feature breadth and visibility.** OpenRec uses a 384-dimensional frozen E5
   title vector, up to 50 clicked titles, and 46 point-in-time global features.
   Team :D builds hundreds of article, user, impression and cross features,
   including click ratios, read-time/page-view statistics, TF-IDF/SVD similarity
   over title, subtitle, body, topics, categories, entities and NER clusters,
   publish-time features, transition features and within-impression ranks. Its
   neural configuration also enables future-impression and future-article
   statistics. Those transductive signals are useful under the challenge rules
   but are intentionally excluded by OpenRec's frozen cutoff because they are
   not available in a causal online-serving setting.
3. **Impression-aware learning.** OpenRec scores candidates independently with
   pointwise binary cross-entropy; its attention query sees one candidate and a
   user history, but not the other candidates in the same impression. Team :D's
   LightGBM model uses `lambdarank`, and its neural model jointly encodes the
   in-view candidate list before producing all scores. This directly models the
   within-impression ordering measured by Macro AUC.
4. **Capacity and representation.** OpenRec uses two 128-dimensional Transformer
   layers over title embeddings. The referenced Kfujikawa model uses eight
   128-dimensional layers and embeds many numerical, categorical, history and
   pretrained semantic inputs. The tree branch captures threshold and feature
   interactions that a small title-history network and its compact global vector
   cannot express as easily.
5. **Ensembling, pseudo-labeling and tuning.** The reported `0.87910` already
   blends four diverse tree/neural predictors with unequal weights. The final
   code expands to five base predictions, trains a pseudo-labeled neural model,
   tunes blend weights with Optuna, and trains a four-fold LambdaRank stacker on
   prediction ranks, normalized scores and pairwise model differences. OpenRec
   reports one model per seed without cross-model ensembling and uses only five
   epochs with validation LogLoss selection.

The comparison therefore identifies a stack gap more than a Transformer-only
gap. A defensible next ablation is to retain OpenRec's causal cutoff and small
training set, then add (a) impression-list batching plus a listwise loss, (b)
causal versions of the winner's content/user/cross features, (c) a same-feature
LightGBM LambdaRank baseline, and finally (d) a validation-only ensemble. Moving
to EB-NeRD-large should be reported as a separate scale experiment, and
future-derived challenge features should remain a separate transductive track.

The values after `±` are sample standard deviations across seeds. Popularity
has one deterministic run. The content variant adds fixed-width signed hashes
of title, topic tags and subcategories plus point-in-time content age. Its
prepared-data SHA-256 is
`7e05c3223952a5ac3e3ede8cd8f3e7d10482c0b953f08d7d7ccd99b587b81d8f`.
Against behavior-only FM, content improves mean Macro AUC by 0.01427, MRR by
0.00403, NDCG@5 by 0.00549 and NDCG@10 by 0.00517.
Against behavior-only LR, content improves mean Macro AUC by 0.03158, MRR by
0.01860, NDCG@5 by 0.02230 and NDCG@10 by 0.01941. LR + content also exceeds
FM + content by 0.00905 Macro AUC under this protocol. This indicates that the
useful signal in the current hashed metadata is largely linear; the present FM
interactions do not improve on it.

The semantic variants replace the 32-dimensional signed title hash with a
384-dimensional, L2-normalized title embedding from
`intfloat/multilingual-e5-small`; the embedding artifact SHA-256 is
`5571b19f877359d7b2d67de9b761ec06f1f9fecbfde748f9d7511917e73db37e`.
Semantic title vectors do not improve LR on average, while FM gains 0.00449
Macro AUC over hashed-content FM. This shows that semantic vectors become more
useful when the model can interact dimensions, but candidate semantics alone
remain a weak substitute for personalized history matching.

The body-semantic variants use the article `body` field when it is nonempty and
enable the 32-dimensional title hash only for articles whose body is absent.
The raw article table has 19,196 nonempty bodies among 20,738 articles (92.56%
coverage); only 18 articles lack both body and title. Body vectors use the same
frozen multilingual E5 encoder, its 512-token truncation limit, and L2
normalization. The artifact SHA-256 is
`de48c171c571ece8e28871956d8b31b6842c8da4b24c93e4099864f3b102b72c`.
Against hashed content, semantic body plus fallback improves mean Macro AUC by
0.00199 for LR and 0.01312 for FM. Against semantic title, it improves LR by
0.00264 and FM by 0.00863. This controlled result supports adding semantic
vectors for richer body content while retaining title hashing as a missing-body
fallback; it does not support replacing inexpensive title hashes with title-only
sentence embeddings.

The history Transformer projects the same semantic vectors, encodes the most
recent 50 official history clicks with two Transformer layers, applies
candidate-aware multi-head attention, and fuses the result with the existing
46-dimensional point-in-time global feature vector. Train and local-validation
rows use `train/history.parquet`; official-validation test rows use its supplied
`validation/history.parquet`. It improves Macro AUC by 0.03206 over LR +
content and by 0.03662 over semantic FM. Its larger seed variance and remaining
gap to challenge systems make listwise objectives, richer article text, context,
and GBDT ensembling the next controlled experiments.

For the seed-42 diagnostic, LR + content improves Macro AUC from 0.55440 to
0.58833 on the 239,799 impressions whose clicked articles are all unseen in
training. It reduces Macro AUC from 0.33420 to 0.24364 on the much smaller 4,825
all-warm-click impressions. FM + content shows the same direction: 0.56312 to
0.57547 for all-cold-click impressions and 0.32318 to 0.28451 for
all-warm-click impressions. There are another 23 mixed impressions. This
supports the cold-start value of content but also shows that a warm-item gate,
calibration, or richer hybrid needs evaluation. These local-small results are
not directly comparable with the full hidden-test Challenge leaderboard and do
not support a SOTA claim.
