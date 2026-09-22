"""Positive-only, full-catalog local retrieval studies using OpenRec recall code."""
from collections import defaultdict
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import get_dataset
from .provenance import digest, git_state, write_json


def prepare_retrieval(config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    dataset = config["dataset"]
    frame, sources = get_dataset(dataset).prepare_events(config)
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
    labels = get_dataset(config["dataset"]).split(frame, config)
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
    from .openrec import load_openrec

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
        strict_timestamps=get_dataset(config["dataset"]).strict_retrieval_timestamps,
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
        "dataset_source_sha256": digest(inspect.getfile(
            get_dataset(config["dataset"]).prepare_events)),
        "train_events": len(train),
        "catalog_items": len(hot_items), "metrics_sha256": digest(output / "metrics.json"),
    })
    return metrics
