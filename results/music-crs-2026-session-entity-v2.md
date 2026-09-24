# Music-CRS 2026 OpenRec session/entity result

This run uses the official 8,000-turn development set and the organizer's
evaluator at commit `3dd7455179fe69a0396f6007c89c078824498415`.

| Method | nDCG@1 | nDCG@10 | nDCG@20 | Catalog diversity |
|---|---:|---:|---:|---:|
| Official LLaMA-1B + BM25 baseline | 0.009800 | 0.062700 | 0.081500 | 0.379500 |
| OpenRec Hot + ItemBasedI2I | 0.012125 | 0.042880 | 0.052182 | 0.405515 |
| OpenRec Session + Entity + Context | **0.041750** | **0.109315** | **0.125764** | **0.448471** |
| Absolute change | +0.029625 | +0.066435 | +0.073582 | +0.042956 |
| Relative change | +244.33% | +154.93% | +141.01% | +10.59% |

The new run retains OpenRec's Hot and ItemBasedI2I recall foundation and adds
the generic feature roles introduced in catalog v18:

- candidate entities: track name, artist, album, tags, and popularity;
- session state: previously recommended tracks and recent user turns;
- context: current user request, conversation goal, and musical culture;
- interaction: transition strength plus artist and tag affinity to session history.

Feature weights were selected by grid search using the final 400 training
sessions (`Recall@20 = 0.379063`, discounted gain `0.160128`). Official
development targets were evaluated once after selection. For turn `t`, the
ranker only reads user messages through `t` and music events before `t`.

Responses remain empty, so lexical diversity is zero and conversational text
quality is outside this comparison. The verified champion's published hidden
test nDCG@20 is `0.618486308`; that result uses a different hidden split, so it
is included as a directional gap rather than a directly comparable score.
