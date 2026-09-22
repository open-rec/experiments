"""Music-CRS session event projection and official train/test split."""
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_events(config):
    sources = [Path(config["train_events"])]
    sources.extend([Path(config["test_events"]), Path(config["tracks"])])
    catalog = pd.read_parquet(sources[-1], columns=["track_id"])
    valid_tracks = set(catalog.track_id.astype(str))
    rows = []
    for split, path in [("train", sources[0]), ("test", sources[1])]:
        sessions = pd.read_parquet(path, columns=[
            "session_id", "user_id", "session_date", "conversations"
        ])
        if sessions.session_id.duplicated().any():
            raise ValueError("duplicate Music-CRS session_id")
        for session in sessions.itertuples(index=False):
            day = pd.Timestamp(session.session_date, tz="UTC")
            for message in session.conversations:
                if message["role"] != "music":
                    continue
                track = str(message["content"])
                if track not in valid_tracks:
                    raise ValueError(f"music track absent from catalog: {track}")
                turn = int(message["turn_number"])
                rows.append((f"{session.session_id}:{turn}", str(session.session_id),
                             str(session.user_id), track, day.value // 1_000_000,
                             turn, split))
    frame = pd.DataFrame(rows, columns=[
        "event_id", "session_id", "user_id", "item_id", "timestamp", "step", "source_split"
    ])
    return frame, sources


def split(frame, config):
    train = frame.source_split.eq("train")
    boundary = frame.loc[train, "timestamp"].quantile(float(config.get("train_fraction", 0.8)))
    return np.where(frame.source_split.eq("test"), "test",
                    np.where(frame.timestamp.lt(boundary), "train", "validation"))
