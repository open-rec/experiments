"""Shared prepared-data schema and dataset dispatch."""
from pathlib import Path

import numpy as np

from .datasets import get_dataset
from .provenance import digest, write_json

REQUIRED = ["sample_id", "group_id", "user_id", "item_id", "timestamp", "label", "policy", "scene"]


def validate(frame):
    if frame.empty or not set(REQUIRED).issubset(frame):
        raise ValueError("empty data or missing required columns")
    if frame[REQUIRED].isna().any().any() or frame.sample_id.duplicated().any():
        raise ValueError("null identity/label or duplicate sample_id")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(frame.timestamp).all() or not frame.timestamp.mod(1).eq(0).all():
        raise ValueError("timestamps must be finite UTC milliseconds")
    for column in ["user_id", "item_id", "sample_id", "group_id"]:
        if frame[column].astype(str).str.strip().eq("").any():
            raise ValueError("empty identity")
    if (frame.groupby("group_id")[["timestamp", "user_id", "policy"]].nunique() > 1).any().any():
        raise ValueError("a group spans multiple timestamps, users or policies")


def prepare(config, output):
    if config.get("task") == "implicit_retrieval_v1":
        from .retrieval import prepare_retrieval
        return prepare_retrieval(config, output)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if config.get("protocol") == "large_train_small_test_v1":
        if config["dataset"] != "ebnerd":
            raise ValueError("large-to-small protocol requires EB-NeRD")
        return get_dataset(config["dataset"]).prepare_large_to_small(config, output)
    paths = [Path(p) for p in config["inputs"]]
    content_path = Path(config["video_features"]) if config.get("video_features") else None
    source_paths = paths + ([content_path] if content_path else [])
    if not paths or len({p.resolve() for p in source_paths}) != len(source_paths):
        raise ValueError("inputs must be nonempty and distinct")
    frame = get_dataset(config["dataset"]).prepare_frame(config, paths, content_path)
    frame = frame.sort_values(["timestamp", "sample_id"], kind="stable").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    write_json(str(output) + ".manifest.json", {
        "schema": 1, "dataset": config["dataset"], "config": config,
        "inputs": [{"path": str(p), "sha256": digest(p)} for p in source_paths],
        "output_sha256": digest(output), "rows": len(frame),
        "min_time": int(frame.timestamp.min()), "max_time": int(frame.timestamp.max()),
    })
