"""EB-NeRD projection and large-to-small preparation."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ...data import validate
from ...provenance import digest, write_json

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
                         "position": position, "candidate_count": len(candidates),
                         "device_type": str(row.get("device_type", "") or ""),
                         "is_subscriber": bool(row.get("is_subscriber", False)),
                         "is_authenticated": bool(row.get("is_sso_user", False)),
                         "read_time": row.get("read_time", np.nan),
                         "scroll_percentage": row.get("scroll_percentage", np.nan),
                         "session_id": str(row.get("session_id", "") or ""),
                         "category": str(article["category"]),
                         "subcategory": subcategory, "tags": tags,
                         "title": str(article.get("title", "") or ""),
                         "pub_time": pub_time,
                         "age": row.get("age", np.nan)})
    result = pd.DataFrame(rows)
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
        candidate_count = len(candidates)
        original_positions = {item: position for position, item in enumerate(candidates)}
        clicks = set(row.article_ids_clicked)
        if not candidates or len(candidates) != len(set(candidates)) or not clicks.issubset(candidates):
            raise ValueError("invalid candidate set")
        if sample_negative:
            negative = _stable_negative(row.impression_id, candidates, clicks)
            retained = clicks | ({negative} if negative is not None else set())
            candidates = [value for value in candidates if value in retained]
        timestamp = pd.Timestamp(row.impression_time)
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        for item in candidates:
            rows.append({
                "sample_id": f"{row.impression_id}:{item}",
                "group_id": str(row.impression_id),
                "user_id": str(row.user_id),
                "item_id": str(item),
                "timestamp": timestamp.value // 1_000_000,
                "label": int(item in clicks),
                "policy": "observed",
                "scene": "news",
                "position": original_positions[item],
                "candidate_count": candidate_count,
                "device_type": getattr(row, "device_type", None),
                "is_subscriber": getattr(row, "is_subscriber", False),
                "is_authenticated": getattr(row, "is_sso_user", False),
                "read_time": getattr(row, "read_time", np.nan),
                "scroll_percentage": getattr(row, "scroll_percentage", np.nan),
                "session_id": str(getattr(row, "session_id", "") or ""),
                "age": getattr(row, "age", np.nan),
            })
    result = pd.DataFrame(rows)
    result = result.merge(metadata, on="item_id", how="left", validate="many_to_one")
    if result.category.isna().any():
        raise ValueError("candidate missing article metadata")
    validate(result)
    return result


def prepare_large_to_small(config, output):
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
        "article_ids_clicked", "user_id", "age", "device_type",
        "is_subscriber", "is_sso_user", "read_time", "scroll_percentage",
        "session_id",
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
                if prefix == "large" and int(config.get("large_impression_sample_mod", 1)) > 1:
                    modulus = int(config["large_impression_sample_mod"])
                    remainder = int(config.get("large_impression_sample_remainder", 0))
                    if not 0 <= remainder < modulus:
                        raise ValueError("large impression sample remainder must be within modulus")
                    hashes = pd.util.hash_pandas_object(
                        behaviors["impression_id"], index=False
                    ).to_numpy(dtype="uint64")
                    behaviors = behaviors.loc[hashes % modulus == remainder]
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
        "large_impression_sample_mod": int(config.get("large_impression_sample_mod", 1)),
        "large_impression_sample_remainder": int(config.get("large_impression_sample_remainder", 0)),
        "small_test_candidates": "complete",
    })
    return counts



def prepare_frame(config, paths, content_path):
    if len(paths) < 2:
        raise ValueError("EB-NeRD inputs: one or more labeled behaviors files, articles last")
    return ebnerd(pd.concat([pd.read_parquet(p) for p in paths[:-1]], ignore_index=True),
                  pd.read_parquet(paths[-1]))
