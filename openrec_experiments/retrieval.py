"""Positive-only, full-catalog local retrieval studies using OpenRec recall code."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import digest, git_state, write_json


def _utc_ms(values):
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    if parsed.isna().any():
        raise ValueError("missing event timestamp")
    return parsed.astype("datetime64[ns, UTC]").astype("int64") // 1_000_000


def prepare_retrieval(config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    dataset = config["dataset"]
    sources = [Path(config["train_events"])]
    if dataset == "music-crs-2026":
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
    elif dataset == "synerise-2025":
        buys = pd.read_parquet(sources[0], columns=["client_id", "sku", "timestamp"])
        if buys[["client_id", "sku", "timestamp"]].isna().any().any():
            raise ValueError("null Synerise purchase field")
        frame = pd.DataFrame({
            "event_id": [f"buy:{i}" for i in range(len(buys))],
            "session_id": buys.client_id.astype(str),
            "user_id": buys.client_id.astype(str),
            "item_id": buys.sku.astype(str),
            "timestamp": _utc_ms(buys.timestamp),
            "source_split": "all",
        })
        frame = frame.sort_values(["user_id", "timestamp", "event_id"], kind="stable")
        frame["step"] = frame.groupby("user_id").cumcount() + 1
    else:
        raise ValueError("unknown retrieval dataset")
    if frame.empty or frame.event_id.duplicated().any():
        raise ValueError("empty or duplicate retrieval events")
    frame = frame.sort_values(["timestamp", "session_id", "step", "event_id"], kind="stable")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    write_json(str(output) + ".manifest.json", {
        "schema": 1, "dataset": dataset, "task": "implicit_retrieval_v1",
        "config": config, "inputs": [{"path": str(p), "sha256": digest(p)} for p in sources],
        "output_sha256": digest(output), "rows": len(frame),
        "min_time": int(frame.timestamp.min()), "max_time": int(frame.timestamp.max()),
    })


def _split(frame, config):
    if config["dataset"] == "music-crs-2026":
        train = frame.source_split.eq("train")
        boundary = frame.loc[train, "timestamp"].quantile(float(config.get("train_fraction", 0.8)))
        labels = np.where(frame.source_split.eq("test"), "test",
                          np.where(frame.timestamp.lt(boundary), "train", "validation"))
    else:
        times = frame.timestamp
        first = times.quantile(float(config.get("train_fraction", 0.7)))
        second = times.quantile(float(config.get("validation_fraction", 0.85)))
        labels = np.where(times.lt(first), "train",
                          np.where(times.lt(second), "validation", "test"))
    if len(set(labels)) != 3:
        raise ValueError("train, validation and test must all be nonempty")
    return labels


def _queries(frame, labels, max_queries, strict_timestamps=False):
    histories = defaultdict(list)
    pending = defaultdict(list)
    last_time = {}
    queries = []
    for row, split in zip(frame.itertuples(index=False), labels):
        key = row.session_id
        if strict_timestamps and last_time.get(key) != row.timestamp:
            histories[key].extend(pending.pop(key, ()))
            last_time[key] = row.timestamp
        prior = histories[key]
        # OpenRec's I2I recall omits triggers. Evaluate new-to-recent items only.
        if split != "train" and prior and row.item_id not in prior[-5:]:
            queries.append((split, row.event_id, row.item_id, prior[-5:].copy()))
        if strict_timestamps:
            pending[key].append(row.item_id)
        else:
            prior.append(row.item_id)
    result = {}
    for split in ("validation", "test"):
        subset = [query for query in queries if query[0] == split]
        if max_queries and len(subset) > max_queries:
            indices = np.linspace(0, len(subset) - 1, max_queries, dtype=int)
            subset = [subset[i] for i in indices]
        result[split] = subset
    return result


def _score(recommender, hot_items, queries, k):
    hits, reciprocal, covered = 0, 0.0, set()
    for _, _, target, history in queries:
        excluded = set(history)
        ranked = []
        if recommender is not None:
            merged = {}
            for trigger in history:
                for item, score in recommender.get(trigger, ()):
                    if item not in excluded:
                        merged[item] = max(merged.get(item, 0), score)
            ranked.extend(item for item, _ in sorted(
                merged.items(), key=lambda pair: (-pair[1], pair[0])
            )[:k])
        ranked.extend(item for item in hot_items[:k + len(excluded)] if item not in excluded)
        ranked = list(dict.fromkeys(ranked))[:k]
        covered.update(ranked)
        if target in ranked:
            hits += 1
            reciprocal += 1 / (ranked.index(target) + 1)
    count = len(queries)
    return {"queries": count, f"recall@{k}": hits / count if count else None,
            f"mrr@{k}": reciprocal / count if count else None,
            "recommended_items": len(covered)}


def run_retrieval(config, output):
    """Static train-only recall models; validation/test histories are observed prefixes."""
    import json
    from .runner import load_openrec

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    source = Path(config["data"])
    manifest = json.loads(Path(str(source) + ".manifest.json").read_text())
    if manifest["dataset"] != config["dataset"] or manifest["output_sha256"] != digest(source):
        raise ValueError("prepared data manifest mismatch")
    root = load_openrec(config["openrec_algorithm"])
    from algorithm.recall.hot import Hot
    from algorithm.recall.item_cf_i2i import ItemBasedI2I
    frame = pd.read_parquet(source)
    labels = _split(frame, config)
    train = frame.loc[labels == "train"].copy()
    events = pd.DataFrame({
        "id": train.event_id, "user_id": train.session_id,
        "item_id": train.item_id, "time": train.timestamp // 1000,
        "type": "click", "value": 1.0,
    })
    k = int(config.get("k", 20))
    hot_items = [result.item for result in Hot(events=events, recall_size=len(train.item_id.unique())).recall()]
    i2i = ItemBasedI2I(events=events, recall_size=k, cut_size=int(config.get("neighbor_size", 50)))
    neighbors = i2i.dump_i2i(cut_size=int(config.get("neighbor_size", 50)))
    queries = _queries(
        frame, labels, int(config.get("max_queries_per_split", 0)),
        strict_timestamps=config["dataset"] == "synerise-2025",
    )
    metrics = {
        split: {"hot": _score(None, hot_items, subset, k),
                "openrec_i2i_hot": _score(neighbors, hot_items, subset, k)}
        for split, subset in queries.items()
    }
    output.mkdir(parents=True)
    write_json(output / "metrics.json", metrics)
    write_json(output / "manifest.json", {
        "schema": 1, "task": "implicit_retrieval_v1", "dataset": config["dataset"],
        "protocol": "train-only static recall; observed prior positive triggers; full train item catalog",
        "config": config, "prepared_sha256": digest(source),
        "openrec_algorithm": str(root),
        "openrec_git": git_state(root),
        "experiments_git": git_state(Path(__file__).resolve().parents[1]),
        "retrieval_source_sha256": digest(__file__),
        "train_events": len(train),
        "catalog_items": len(hot_items), "metrics_sha256": digest(output / "metrics.json"),
    })
    return metrics
