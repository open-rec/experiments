"""Expand official dev sessions into one leakage-free raw row per target turn."""

from pathlib import Path

import polars as pl


SOURCE = Path(
    "/workspace/data/talkpl-ai/TalkPlayData-Challenge-Dataset/"
    "data/test-00000-of-00001.parquet"
)
OUTPUT = Path("/workspace/data/splitK/holdout_all_turns_raw.parquet")


rows = []
for raw in pl.read_parquet(SOURCE).iter_rows(named=True):
    for target_turn in range(1, 9):
        row = dict(raw)
        row["conversations"] = [
            event
            for event in raw["conversations"]
            if int(event["turn_number"]) < target_turn
            or (
                int(event["turn_number"]) == target_turn
                and event["role"] == "user"
            )
        ]
        rows.append(row)

out = pl.DataFrame(rows, schema=pl.read_parquet_schema(SOURCE))
if out.height != 8000:
    raise RuntimeError(f"expected 8000 target views, got {out.height}")
for row in out.iter_rows(named=True):
    target = max(
        int(event["turn_number"])
        for event in row["conversations"]
        if event["role"] == "user"
    )
    if any(
        event["role"] == "music" and int(event["turn_number"]) >= target
        for event in row["conversations"]
    ):
        raise RuntimeError("target or future music leaked into heuristic input")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
out.write_parquet(OUTPUT)
print(f"wrote {out.height} leakage-free target views to {OUTPUT}")
