# EB-NeRD-large training to EB-NeRD-small prediction v1

Status: preliminary content-only scale experiment; not an official leaderboard
result and not a single-variable comparison with the baseline-v1 table.

The experiment trains OpenRec FM from the complete 12,063,890-impression
EB-NeRD-large training population and evaluates on all 2,928,942 candidates in
the 244,647 impressions of EB-NeRD-small validation. The first six large-train
days supply 20,807,457 training samples. Its final day supplies 3,364,254 local
validation samples and is used only for epoch selection. EB-NeRD-small
validation is untouched until final prediction.

To keep the large source tractable and class-balanced, every large impression
retains all clicked candidates and one deterministic unclicked candidate. The
small test retains its complete candidate list. Randomness in negative selection
is derived only from the impression id and is therefore identical across model
seeds. The prepared-data SHA-256 is
`cf8a6214fdb53a3a76b4a41446da799b47f8371e59cd0b1de7fa7b340e34edd6`.

| Model | Seeds | Macro AUC | Macro MRR | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| OpenRec FM + semantic body / title-hash fallback, large train | 42/43/44 | **0.56731 ± 0.00028** | **0.35238 ± 0.00116** | **0.39249 ± 0.00140** | **0.47084 ± 0.00077** |

The model uses the same 384-dimensional multilingual-E5 body representation and
enables OpenRec's 32-dimensional title hash only when body text is absent. The
large article table contains 125,541 articles, of which 119,883 have nonempty
bodies (95.49%). The embedding artifact SHA-256 is
`e38e465b3ca7605d44f2ab50b78a0cebfe1a2b828ee493df3a28008881584dd1`.
The fitted OpenRec feature vector has 462 dimensions and combines semantic body
with user age, title fallback, category, subcategory, tags, and scene. It omits
behavioral counts and click rates because those values would be distorted by
one-negative sampling.

The score is lower than the earlier small-trained FM body-semantic result
(`0.58603 ± 0.00285`), but the two rows differ in more than training scale: the
small baseline uses complete candidates and point-in-time behavioral features,
whereas this scale run uses sampled training candidates and content-only
features. The result establishes a reproducible large-to-small execution path;
it does not show that more training data is harmful. A causal scale conclusion
requires either streaming full-candidate training or a matched small-train
content-only control under the same negative-sampling protocol.

All three seeds selected epoch 1. This, together with the large validation/test
distribution difference, suggests that probability calibration and the
negative-sampling correction should be addressed before increasing epochs or
model capacity.
