"""KuaiRand-1K exposure and video metadata projection."""
import numpy as np
import pandas as pd

from ...data import validate

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



def prepare_frame(config, paths, content_path):
    columns = ["user_id", "video_id", "time_ms", "is_click", "is_rand", "tab"]
    logs = pd.concat([pd.read_csv(p, usecols=columns) for p in paths], ignore_index=True)
    videos = None if content_path is None else pd.read_csv(
        content_path,
        usecols=["video_id", "video_type", "upload_dt", "upload_type", "tag"],
        dtype={"tag": "string"},
    )
    return kuairand(logs, videos)
