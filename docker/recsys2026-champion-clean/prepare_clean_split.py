"""Create champion-compatible folds with the official devset as strict holdout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl


ROOT = Path("/workspace")
SOURCE_SPLITS = ROOT / "src" / "splits"
sys.path.insert(0, str(SOURCE_SPLITS))

from launchers.splitK_crossvalidation import (  # noqa: E402
    _assemble,
    _assign_groups,
    _split_group_two,
)

TRAIN = ROOT / "data/talkpl-ai/TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet"
DEV = ROOT / "data/talkpl-ai/TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet"
OUT = ROOT / "data/splitK"
N_FOLDS = 5
SEED = 42


def session_digest(values):
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def main():
    train = _assemble(pl.read_parquet(TRAIN), is_submission=False)
    dev = _assemble(pl.read_parquet(DEV), is_submission=False)
    train_sessions = set(train["session_id"].unique().to_list())
    dev_sessions = set(dev["session_id"].unique().to_list())
    if train_sessions & dev_sessions:
        raise ValueError("official train/dev session overlap")

    user_to_group = _assign_groups(train, N_FOLDS, SEED)
    groups = [[] for _ in range(N_FOLDS)]
    for user, group in user_to_group.items():
        groups[group].append(user)
    counts = dict(train.group_by("user_id").agg(
        pl.col("session_id").n_unique().alias("n")
    ).select("user_id", "n").iter_rows())

    assignments = {session: "holdout" for session in dev_sessions}
    for fold in range(N_FOLDS):
        cg_users, rr_users = _split_group_two(groups[fold], counts, SEED + fold)
        for session in train.filter(pl.col("user_id").is_in(cg_users))["session_id"].unique():
            assignments[session] = f"fold_{fold}_cg_val"
        for session in train.filter(pl.col("user_id").is_in(rr_users))["session_id"].unique():
            assignments[session] = f"fold_{fold}_reranker_val"

    if set(assignments) != train_sessions | dev_sessions:
        raise ValueError("assignment does not cover train and dev exactly")
    OUT.mkdir(parents=True, exist_ok=True)
    assignment = pl.DataFrame({
        "session_id": sorted(assignments),
        "bucket": [assignments[key] for key in sorted(assignments)],
    })
    assignment.write_parquet(OUT / "splitK_assignment.parquet")
    dev.write_parquet(OUT / "holdout_test.parquet")

    tagged = train.join(assignment.filter(pl.col("bucket") != "holdout"),
                        on="session_id", how="inner")
    counts_out = {}
    for fold in range(N_FOLDS):
        train_labels = [f"fold_{other}_{part}" for other in range(N_FOLDS)
                        if other != fold for part in ("cg_val", "reranker_val")]
        frames = {
            "cg_train": tagged.filter(pl.col("bucket").is_in(train_labels)),
            "cg_val": tagged.filter(pl.col("bucket") == f"fold_{fold}_cg_val"),
            "reranker_val": tagged.filter(pl.col("bucket") == f"fold_{fold}_reranker_val"),
        }
        for name, frame in frames.items():
            clean = frame.drop("bucket")
            clean.write_parquet(OUT / f"fold_{fold}_{name}.parquet")
            counts_out[f"fold_{fold}_{name}"] = {
                "rows": clean.height,
                "sessions": clean["session_id"].n_unique(),
            }

    train_written = set()
    for path in OUT.glob("fold_*_*.parquet"):
        train_written.update(pl.read_parquet(path, columns=["session_id"])["session_id"].to_list())
    leaked = sorted(dev_sessions & train_written)
    if leaked:
        raise ValueError(f"dev leakage into training folds: {leaked[:5]}")
    manifest = {
        "schema": 1,
        "protocol": "official-dev-as-strict-holdout",
        "seed": SEED,
        "folds": N_FOLDS,
        "train_sessions": len(train_sessions),
        "dev_sessions": len(dev_sessions),
        "train_rows": train.height,
        "dev_rows": dev.height,
        "train_session_sha256": session_digest(train_sessions),
        "dev_session_sha256": session_digest(dev_sessions),
        "dev_train_overlap": 0,
        "splits": counts_out,
    }
    (OUT / "clean_dev_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
