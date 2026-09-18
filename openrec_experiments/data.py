"""Strict, label-preserving adapters. No fabricated exposures or profiles."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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


def kuairand(logs, videos=None):
    """Project exposures and optional static video metadata without behavior snapshots."""
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
    if videos is not None:
        required_video = ["video_id", "video_type", "upload_dt", "upload_type", "tag"]
        if videos[required_video].isna().all(axis=0).any():
            raise ValueError("KuaiRand video metadata is missing a required content field")
        if videos.video_id.isna().any() or videos.video_id.duplicated().any():
            raise ValueError("duplicate or missing KuaiRand video metadata identity")
        metadata = videos[required_video].copy()
        metadata["item_id"] = metadata.pop("video_id").astype(str)
        metadata["category"] = metadata.pop("video_type").fillna("").astype(str)
        metadata["subcategory"] = metadata.pop("upload_type").fillna("").astype(str)
        metadata["tags"] = metadata.pop("tag").fillna("").astype(str)
        published = pd.to_datetime(metadata.pop("upload_dt"), errors="coerce", utc=True)
        metadata["pub_time"] = (
            published.astype("datetime64[ns, UTC]").astype("int64") // 1_000_000_000
        ).where(published.notna(), 0)
        metadata["title"] = ""
        result = result.merge(metadata, on="item_id", how="left", validate="many_to_one")
        if result.category.isna().any():
            raise ValueError("KuaiRand exposure is missing video metadata")
    validate(result)
    return result


def _article_metadata(articles):
    """Normalize EB-NeRD article metadata once for chunked behavior projection."""
    metadata = articles[[
        "article_id", "title", "category", "subcategory", "topics", "published_time"
    ]].copy()
    metadata["item_id"] = metadata.pop("article_id").astype(str)
    metadata["title"] = metadata.title.fillna("").astype(str)
    metadata["category"] = metadata.category.fillna("").astype(str)
    for source, destination in [("subcategory", "subcategory"), ("topics", "tags")]:
        metadata[destination] = metadata.pop(source).map(
            lambda values: ",".join(str(value) for value in values)
            if isinstance(values, (list, tuple, np.ndarray)) else str(values or "")
        )
    published = pd.to_datetime(metadata.pop("published_time"), errors="coerce", utc=True)
    metadata["pub_time"] = (
        published.astype("datetime64[ns, UTC]").astype("int64") // 1_000_000_000
    ).where(published.notna(), 0)
    if metadata.item_id.duplicated().any():
        raise ValueError("duplicate article identity")
    return metadata


def _stable_negative(impression_id, candidates, clicks):
    negatives = [value for value in candidates if value not in clicks]
    if not negatives:
        return None
    number = int.from_bytes(
        hashlib.blake2b(str(impression_id).encode("utf-8"), digest_size=8).digest(),
        "little",
    )
    return negatives[number % len(negatives)]


def _project_behavior_batch(behaviors, metadata, sample_negative):
    rows = []
    for row in behaviors.itertuples(index=False):
        candidates = list(row.article_ids_inview)
        clicks = set(row.article_ids_clicked)
        if not candidates or len(candidates) != len(set(candidates)) or not clicks.issubset(candidates):
            raise ValueError("invalid candidate set")
        if sample_negative:
            negative = _stable_negative(row.impression_id, candidates, clicks)
            retained = clicks | ({negative} if negative is not None else set())
            candidates = [value for value in candidates if value in retained]
        timestamp = pd.Timestamp(row.impression_time)
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        for position, item in enumerate(candidates):
            rows.append({
                "sample_id": f"{row.impression_id}:{item}",
                "group_id": str(row.impression_id),
                "user_id": str(row.user_id),
                "item_id": str(item),
                "timestamp": timestamp.value // 1_000_000,
                "label": int(item in clicks),
                "policy": "observed",
                "scene": "news",
                "position": position,
                "age": getattr(row, "age", np.nan),
            })
    result = pd.DataFrame(rows)
    result = result.merge(metadata, on="item_id", how="left", validate="many_to_one")
    if result.category.isna().any():
        raise ValueError("candidate missing article metadata")
    validate(result)
    return result


def prepare_ebnerd_large_to_small(config, output):
    """Use every large-train impression with one deterministic negative; test on full small."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    train_path = Path(config["large_train_behaviors"])
    test_path = Path(config["small_test_behaviors"])
    articles_path = Path(config["articles"])
    metadata = _article_metadata(pd.read_parquet(articles_path))
    columns = [
        "impression_id", "impression_time", "article_ids_inview",
        "article_ids_clicked", "user_id", "age",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    counts = {"large_impressions": 0, "large_samples": 0,
              "small_impressions": 0, "small_samples": 0}
    try:
        for path, sampled, prefix in [
            (train_path, True, "large"), (test_path, False, "small")
        ]:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=int(config.get("projection_batch_size", 50_000)), columns=columns):
                behaviors = batch.to_pandas()
                projected = _project_behavior_batch(behaviors, metadata, sampled)
                counts[f"{prefix}_impressions"] += len(behaviors)
                counts[f"{prefix}_samples"] += len(projected)
                table = pa.Table.from_pandas(projected, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output, table.schema, compression="zstd")
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("scale preparation produced no samples")
    sources = [train_path, test_path, articles_path]
    write_json(str(output) + ".manifest.json", {
        "schema": 1, "dataset": "ebnerd", "protocol": "large_train_small_test_v1",
        "config": config, "inputs": [
            {"path": str(path), "sha256": digest(path)} for path in sources
        ], "output_sha256": digest(output), **counts,
        "negative_sampling": "all clicks plus one deterministic unclicked candidate per large impression",
        "small_test_candidates": "complete",
    })
    return counts


def prepare(config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if config.get("protocol") == "large_train_small_test_v1":
        return prepare_ebnerd_large_to_small(config, output)
    paths = [Path(p) for p in config["inputs"]]
    content_path = Path(config["video_features"]) if config.get("video_features") else None
    source_paths = paths + ([content_path] if content_path else [])
    if not paths or len({p.resolve() for p in source_paths}) != len(source_paths):
        raise ValueError("inputs must be nonempty and distinct")
    if config["dataset"] == "ebnerd":
        if len(paths) < 2:
            raise ValueError("EB-NeRD inputs: one or more labeled behaviors files, articles last")
        frame = ebnerd(pd.concat([pd.read_parquet(p) for p in paths[:-1]], ignore_index=True),
                       pd.read_parquet(paths[-1]))
    elif config["dataset"] == "kuairand-1k":
        columns = ["user_id", "video_id", "time_ms", "is_click", "is_rand", "tab"]
        logs = pd.concat([pd.read_csv(p, usecols=columns) for p in paths], ignore_index=True)
        videos = None if content_path is None else pd.read_csv(
            content_path,
            usecols=["video_id", "video_type", "upload_dt", "upload_type", "tag"],
            dtype={"tag": "string"},
        )
        frame = kuairand(logs, videos)
    else:
        raise ValueError("unknown dataset")
    frame = frame.sort_values(["timestamp", "sample_id"], kind="stable").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    write_json(str(output) + ".manifest.json", {
        "schema": 1, "dataset": config["dataset"], "config": config,
        "inputs": [{"path": str(p), "sha256": digest(p)} for p in source_paths],
        "output_sha256": digest(output), "rows": len(frame),
        "min_time": int(frame.timestamp.min()), "max_time": int(frame.timestamp.max()),
    })
