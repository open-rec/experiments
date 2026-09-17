"""Strict, label-preserving adapters. No fabricated exposures or profiles."""
from pathlib import Path

import numpy as np
import pandas as pd

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


def ebnerd(behaviors, articles):
    required = ["impression_id", "user_id", "impression_time", "article_ids_inview", "article_ids_clicked"]
    if behaviors[required].isna().any().any():
        raise ValueError("EB-NeRD requires labeled, non-null impressions")
    if behaviors.impression_id.duplicated().any() or articles.article_id.duplicated().any():
        raise ValueError("duplicate impression/article identity")
    metadata = articles.set_index("article_id")
    rows = []
    for row in behaviors.to_dict("records"):
        candidates = list(row["article_ids_inview"])
        clicks = set(row["article_ids_clicked"])
        if not candidates or len(candidates) != len(set(candidates)) or not clicks.issubset(candidates):
            raise ValueError("invalid candidate set")
        timestamp = pd.Timestamp(row["impression_time"])
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        for position, item in enumerate(candidates):
            if item not in metadata.index:
                raise ValueError("candidate missing article metadata")
            article = metadata.loc[item]
            subcategory = article.get("subcategory", [])
            topics = article.get("topics", [])
            subcategory = (
                ",".join(str(value) for value in subcategory)
                if isinstance(subcategory, (list, tuple, np.ndarray))
                else str(subcategory or "")
            )
            tags = (
                ",".join(str(value) for value in topics)
                if isinstance(topics, (list, tuple, np.ndarray))
                else str(topics or "")
            )
            published = pd.Timestamp(article.get("published_time"))
            pub_time = 0 if pd.isna(published) else int(
                (published.tz_localize("UTC") if published.tzinfo is None else
                 published.tz_convert("UTC")).timestamp()
            )
            rows.append({"sample_id": f"{row['impression_id']}:{item}",
                         "group_id": str(row["impression_id"]), "user_id": str(row["user_id"]),
                         "item_id": str(item), "timestamp": timestamp.value // 1_000_000,
                         "label": int(item in clicks), "policy": "observed", "scene": "news",
                         "position": position, "category": str(article["category"]),
                         "subcategory": subcategory, "tags": tags,
                         "title": str(article.get("title", "") or ""),
                         "pub_time": pub_time,
                         "age": row.get("age", np.nan)})
    result = pd.DataFrame(rows)
    validate(result)
    return result


def kuairand(logs):
    """Do not import static user snapshots or full-day video statistics as historical features."""
    required = ["user_id", "video_id", "time_ms", "is_click", "is_rand", "tab"]
    if logs[required].isna().any().any():
        raise ValueError("missing KuaiRand identity/label")
    if not logs.is_rand.isin([0, 1]).all():
        raise ValueError("invalid exposure policy")
    result = pd.DataFrame({
        "sample_id": [f"event:{i}" for i in range(len(logs))],
        "group_id": [f"event:{i}" for i in range(len(logs))],
        "user_id": logs.user_id.astype(str), "item_id": logs.video_id.astype(str),
        "timestamp": logs.time_ms, "label": logs.is_click,
        "policy": np.where(logs.is_rand.eq(1), "random", "standard"),
        "scene": logs.tab.astype(str),
    })
    validate(result)
    return result


def prepare(config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    paths = [Path(p) for p in config["inputs"]]
    if not paths or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("inputs must be nonempty and distinct")
    if config["dataset"] == "ebnerd":
        if len(paths) < 2:
            raise ValueError("EB-NeRD inputs: one or more labeled behaviors files, articles last")
        frame = ebnerd(pd.concat([pd.read_parquet(p) for p in paths[:-1]], ignore_index=True),
                       pd.read_parquet(paths[-1]))
    elif config["dataset"] == "kuairand-1k":
        columns = ["user_id", "video_id", "time_ms", "is_click", "is_rand", "tab"]
        frame = kuairand(pd.concat([pd.read_csv(p, usecols=columns) for p in paths], ignore_index=True))
    else:
        raise ValueError("unknown dataset")
    frame = frame.sort_values(["timestamp", "sample_id"], kind="stable").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    write_json(str(output) + ".manifest.json", {
        "schema": 1, "dataset": config["dataset"], "config": config,
        "inputs": [{"path": str(p), "sha256": digest(p)} for p in paths],
        "output_sha256": digest(output), "rows": len(frame),
        "min_time": int(frame.timestamp.min()), "max_time": int(frame.timestamp.max()),
    })
