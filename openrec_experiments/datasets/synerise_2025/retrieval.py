"""Synerise purchase event projection and temporal split."""
from pathlib import Path

import numpy as np
import pandas as pd

from ...temporal import utc_milliseconds


def prepare_events(config):
    sources = [Path(config["train_events"])]
    buys = pd.read_parquet(sources[0], columns=["client_id", "sku", "timestamp"])
    if buys[["client_id", "sku", "timestamp"]].isna().any().any():
        raise ValueError("null Synerise purchase field")
    frame = pd.DataFrame({
        "event_id": [f"buy:{i}" for i in range(len(buys))],
        "session_id": buys.client_id.astype(str),
        "user_id": buys.client_id.astype(str),
        "item_id": buys.sku.astype(str),
        "timestamp": utc_milliseconds(buys.timestamp),
        "source_split": "all",
    })
    frame = frame.sort_values(["user_id", "timestamp", "event_id"], kind="stable")
    frame["step"] = frame.groupby("user_id").cumcount() + 1
    return frame, sources


def split(frame, config):
    times = frame.timestamp
    first = times.quantile(float(config.get("train_fraction", 0.7)))
    second = times.quantile(float(config.get("validation_fraction", 0.85)))
    return np.where(times.lt(first), "train",
                    np.where(times.lt(second), "validation", "test"))
