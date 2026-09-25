"""Build the winner heuristic's aligned AudioCLAP NumPy cache from official parquet."""

from pathlib import Path

import numpy as np
import polars as pl


DATA = Path(
    "/workspace/data/talkpl-ai/TalkPlayData-Challenge-Track-Embeddings/data"
)
OUTPUT = Path("/workspace/models/track_tower_cache")


parts = [pl.read_parquet(path, columns=["track_id", "audio-laion_clap"])
         for path in sorted(DATA.glob("all_tracks-*-of-00004.parquet"))]
frame = pl.concat(parts).unique(subset=["track_id"], keep="first")
track_ids = frame["track_id"].cast(pl.Utf8).to_list()
vectors = frame["audio-laion_clap"].to_list()
dim = next(len(value) for value in vectors if value is not None and len(value))
embeddings = np.zeros((len(vectors), dim), dtype=np.float32)
mask = np.zeros(len(vectors), dtype=bool)
for index, value in enumerate(vectors):
    if value is None or len(value) != dim:
        continue
    vector = np.asarray(value, dtype=np.float32)
    if not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-12:
        continue
    embeddings[index] = vector
    mask[index] = True
OUTPUT.mkdir(parents=True, exist_ok=True)
np.save(OUTPUT / "track_ids.npy", np.asarray(track_ids, dtype=object))
np.save(OUTPUT / "audio-laion_clap.npy", embeddings)
np.save(OUTPUT / "audio-laion_clap__mask.npy", mask)
print(f"wrote {len(track_ids)} rows, dim={dim}, coverage={mask.mean():.4f}")
