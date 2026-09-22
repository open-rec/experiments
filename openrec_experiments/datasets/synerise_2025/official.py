"""Create official-format Synerise user profiles from OpenRec event features."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...provenance import digest, git_state, write_json
from ...openrec import load_openrec


EVENT_TYPES = ("product_buy", "add_to_cart", "remove_from_cart")


def create_profiles(config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    data_dir = Path(config["challenge_data_dir"])
    clients_path = data_dir / "input/relevant_clients.npy"
    clients = np.load(clients_path)
    if clients.ndim != 1 or clients.dtype != np.int64 or len(set(clients)) != len(clients):
        raise ValueError("invalid official relevant_clients")
    root = load_openrec(config["openrec_algorithm"])
    from algorithm.feature.event_feature import aggregate_event_features
    target_start = pd.to_datetime(pd.read_parquet(
        data_dir / "target/train_target.parquet", columns=["timestamp"]
    ).timestamp.min(), utc=True)
    as_of_time = int(target_start.timestamp()) - 1

    snapshots = []
    input_hashes = {"relevant_clients.npy": digest(clients_path)}
    for event_type in EVENT_TYPES:
        path = data_dir / "input" / f"{event_type}.parquet"
        source = pd.read_parquet(path, columns=["client_id", "timestamp", "sku"])
        source = source[source.client_id.isin(clients)].copy()
        if source.empty or source[["client_id", "timestamp", "sku"]].isna().any().any():
            raise ValueError(f"invalid {event_type} input events")
        dates = pd.to_datetime(source.timestamp, utc=True)
        frame = pd.DataFrame({
            "user_id": source.client_id.to_numpy(dtype=np.int64),
            "item_id": source.sku.to_numpy(dtype=np.int64),
            "time": dates.astype("datetime64[ns, UTC]").astype("int64") // 1_000_000_000,
            "type": event_type,
            "value": 1.0,
            "scene": "shop",
            "event_id": [f"{event_type}:{i}" for i in source.index],
        })
        aggregated = aggregate_event_features(
            frame, entity="user", as_of_time=as_of_time,
            event_types=(event_type,),
        )
        selected = ["user_id", "event_count", "event_active_days",
                    "event_unique_item_count", "event_recency_seconds",
                    f"event_{event_type}_count"]
        selected.extend(column for column in aggregated if column.startswith("event_")
                        and ("_7d" in column or "_30d" in column))
        selected = list(dict.fromkeys(column for column in selected if column in aggregated))
        snapshot = aggregated[selected].set_index("user_id").reindex(clients).fillna(0)
        snapshot.columns = [f"{event_type}.{column}" for column in snapshot.columns]
        snapshots.append(snapshot)
        input_hashes[str(path)] = digest(path)
    features = pd.concat(snapshots, axis=1).to_numpy(dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("non-finite OpenRec features")
    # Counts are heavy-tailed; official downstream model consumes float16 profiles.
    features = np.log1p(np.maximum(features, 0)).astype(np.float16)
    output.mkdir(parents=True)
    np.save(output / "client_ids.npy", clients)
    np.save(output / "embeddings.npy", features)
    write_json(output / "manifest.json", {
        "schema": 1, "task": "synerise_official_profiles",
        "config": config, "users": len(clients), "dimensions": features.shape[1],
        "event_types": EVENT_TYPES, "input_sha256": input_hashes,
        "as_of_time": as_of_time,
        "client_ids_sha256": digest(output / "client_ids.npy"),
        "embeddings_sha256": digest(output / "embeddings.npy"),
        "openrec_git": git_state(root), "source_sha256": digest(__file__),
    })
    return {"users": len(clients), "dimensions": features.shape[1]}
