# RecSys 2026 Music-CRS

Official challenge: https://www.recsyschallenge.com/2026/

Download these files from the organizer's `talkpl-ai` Hugging Face repositories.
Set `HF_ENDPOINT=https://hf-mirror.com` if using the mirror; retain the original
file bytes and use the following local names:

- `data/raw/music_crs/train.parquet`: `TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet`
- `data/raw/music_crs/test.parquet`: `TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet`
- `data/raw/music_crs/tracks.parquet`: `TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet`

The adapter extracts only the recorded `music` turn track IDs. It validates
them against the official track catalog and preserves session, turn, user, and
session date. This local study predicts a new-to-recent recorded track from earlier
music turns in the same session. It does not model the user text or generate a
response. The recorded recommendation is a proxy target, not evidence that the
listener preferred the track. Thus its Recall@20 is not the challenge's nDCG
or composite leaderboard score. The official data license is CC BY-NC 4.0;
do not commit or redistribute the files.

The organizer's [evaluator](https://github.com/nlp4musa/music-crs-evaluator)
publishes development-set Random, Popularity, and LLaMA-1B + BM25 baselines.
Their reported metrics are transcribed in `official_baselines.csv`. Those
numbers use the official development set and nDCG/diversity protocol; they are
not comparable with this repository's local next-recorded-track Recall@20.
The separate [official devset result](../../results/music-crs-2026-official-devset-v1.md)
uses the organizer's evaluator, its 8,000 session turns, and 20 catalog tracks
per turn. It is comparable on the recommendation metrics shown there. Empty
responses leave conversational quality unmeasured.
Check out the organizer's evaluator at commit
`3dd7455179fe69a0396f6007c89c078824498415` at the path used below.

Run from `experiments/`:

```bash
.venv/bin/python -m openrec_experiments.cli prepare \
  --config datasets/music_crs_2026/prepare.json \
  --output data/processed/music-crs-2026.parquet
.venv/bin/python -m openrec_experiments.cli run \
  --config studies/baselines/music-crs-2026-retrieval.json \
  --output runs/music-crs-2026/retrieval-v4
.venv/bin/python -m openrec_experiments.cli official-music \
  --config studies/baselines/music-crs-2026-official-devset.json \
  --evaluator /tmp/openrec-music-crs-evaluator \
  --output runs/music-crs-2026/official-devset-v1

# Session/entity/context feature version (weights selected on training sessions)
.venv/bin/python -m openrec_experiments.cli official-music \
  --config studies/baselines/music-crs-2026-session-entity-v2.json \
  --evaluator /tmp/openrec-music-crs-evaluator \
  --output runs/music-crs-2026/session-entity-v2
```
