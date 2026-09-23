# RecSys 2026 Hallucinated champion final assembly

The [first-place Hallucinated source](https://github.com/remaplab/recsys-challenge-2026-music-crs)
is pinned to commit `3bc859dd716d0891f12311e6547d41a57b041b8e`.
Its [published Hugging Face artifact set](https://huggingface.co/datasets/NicoloLocatelli/RecSys_ACM_2026_Hallucinated)
includes the final Blind-B `src/heuristic/LAST.json` and two intermediate
outputs at revision `a187aa513434c5169aec8ecfffeda8c6b04aaf45`. The
Docker adapter fetches them through `hf-mirror.com`, checks their SHA-256 hashes,
and reruns the winner's `96_graft_turn1.py`, `98_top1_graft.py`, and
`merge_submission.py` with its committed response file. The reconstructed
submission must be byte-identical to `LAST.json`. It independently computes two
public-data metrics:

Only the three JSON artifacts needed for final assembly are downloaded; the
full published artifact repository is substantially larger.

| Metric | Recomputed from final artifact | [Official leaderboard](https://nlp4musa.github.io/music-crs-challenge/results.html) |
| --- | ---: | ---: |
| Catalog Diversity | 0.027490387 | 0.027490387 |
| Lexical Diversity | 0.957074025 | 0.957074025 |

Blind-B relevance labels and the official LLM judge outputs are private, so
nDCG@20, the LLM-as-a-Judge score, and the composite cannot be independently
recomputed from the published files. The leaderboard reports 0.618486308,
4.75, and 0.688949595 respectively. This adapter reproduces the winner's final
assembly from its published stage outputs. The upstream retrieval, reranker
training, and 26B language-model generation are not rerun here.

Run from `experiments/` with a checkout of the pinned winning commit:

```bash
git clone https://github.com/remaplab/recsys-challenge-2026-music-crs.git /tmp/recsys2026-hallucinated
git -C /tmp/recsys2026-hallucinated checkout 3bc859dd716d0891f12311e6547d41a57b041b8e
export RECSYS2026_WINNER_SOURCE=/tmp/recsys2026-hallucinated
export RECSYS2026_OUTPUT=$PWD/runs/recsys2026-champion
export RECSYS2026_UID=$(id -u)
export RECSYS2026_GID=$(id -g)
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p "$RECSYS2026_OUTPUT"
docker compose -f docker/recsys2026-champion/compose.yaml build
docker compose -f docker/recsys2026-champion/compose.yaml run --rm champion verify
```

The output contains `LAST.json`, the two intermediate outputs, the recreated
`prediction.json`, and `verification.json` with source and artifact hashes. All outputs are generated
under the chosen directory; do not commit them.
