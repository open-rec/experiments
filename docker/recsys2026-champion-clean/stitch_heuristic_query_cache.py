"""Select official dev query embeddings from the winner's splitK cache shards."""

from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path("/workspace/models/retrieval_text_towers/Qwen__Qwen3-Embedding-8B")
RAW = Path(
    "/workspace/data/talkpl-ai/TalkPlayData-Challenge-Dataset/"
    "data/test-00000-of-00001.parquet"
)
OUTPUT = ROOT / "dense_clean_dev_all_turns_query_len512_poollast"


raw = pl.read_parquet(RAW, columns=["session_id", "user_id"])
wanted = {
    (str(row["session_id"]), str(row["user_id"]), turn)
    for row in raw.iter_rows(named=True)
    for turn in range(1, 9)
}
selected = {}
for cache in sorted(ROOT.glob("dense_splitk_*_query_len512_poollast")):
    meta_path = cache / "query_meta.parquet"
    emb_path = cache / "query_embeddings.npy"
    if not meta_path.is_file() or not emb_path.is_file():
        continue
    meta = pl.read_parquet(meta_path)
    embeddings = np.load(emb_path, mmap_mode="r")
    for index, row in enumerate(meta.iter_rows(named=True)):
        key = (str(row["session_id"]), str(row["user_id"]), int(row["turn_number"]))
        if key not in wanted or key in selected:
            continue
        selected[key] = (row, np.asarray(embeddings[index]).copy())

missing = wanted - selected.keys()
if missing:
    raise RuntimeError(f"missing {len(missing)}/{len(wanted)} dev query embeddings")
ordered = sorted(selected)
out_meta = pl.DataFrame([selected[key][0] for key in ordered])
out_embeddings = np.stack([selected[key][1] for key in ordered])
OUTPUT.mkdir(parents=True, exist_ok=True)
out_meta.write_parquet(OUTPUT / "query_meta.parquet")
np.save(OUTPUT / "query_embeddings.npy", out_embeddings)
print(f"wrote {len(ordered)} rows, dim={out_embeddings.shape[1]} to {OUTPUT}")
