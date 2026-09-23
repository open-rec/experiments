"""Create official-format Synerise user profiles from OpenRec event features."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...provenance import digest, git_state, write_json
from ...openrec import load_openrec


EVENT_TYPES = ("product_buy", "add_to_cart", "remove_from_cart")
ACTION_TYPES = {"product_buy": "buy", "add_to_cart": "collect", "remove_from_cart": "click"}
WINDOWS = (("1d", 86400), ("7d", 604800), ("30d", 2592000))


def _safe_ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float32),
                     where=denominator > 0)


def _expanded_profiles(config, data_dir, clients, as_of_time, root):
    """Build official-aligned profiles plus OpenRec conversion and commerce features.

    The organizer representation is used as the public, fixed benchmark input.  Appended columns
    implement OpenRec's global feature formulas, so the downstream model and official split stay
    unchanged and the experiment isolates the value of the expanded feature foundation.
    """
    from algorithm.feature.commerce_feature import CATEGORY_WEIGHTS

    base_dir = Path(config["official_base_embeddings"])
    base_clients = np.load(base_dir / "client_ids.npy", mmap_mode="r")
    if not np.array_equal(base_clients, clients):
        raise ValueError("official base embeddings are not aligned to relevant_clients")
    base = np.load(base_dir / "embeddings.npy", mmap_mode="r")
    if base.shape[0] != len(clients):
        raise ValueError("invalid official base embedding rows")

    client_index = pd.Index(clients)
    properties = pd.read_parquet(data_dir / "product_properties.parquet",
                                 columns=["sku", "category", "price"]).set_index("sku")
    action_counts = {}
    action_price_sum = {}
    action_price_sq_sum = {}
    action_price_count = {}
    window_counts = {}
    window_price_sum = {}
    window_price_count = {}
    category_bins = int(config.get("category_hash_bins", 32))
    category_profile = np.zeros((len(clients), category_bins), dtype=np.float32)

    for event_type, action in ACTION_TYPES.items():
        source = pd.read_parquet(data_dir / "input" / f"{event_type}.parquet",
                                 columns=["client_id", "timestamp", "sku"])
        rows = client_index.get_indexer(source.client_id.to_numpy())
        valid = rows >= 0
        source = source.loc[valid].copy()
        rows = rows[valid]
        seconds = pd.to_datetime(source.timestamp, utc=True).astype("int64").to_numpy() // 1_000_000_000
        item = properties.reindex(source.sku.to_numpy())
        prices = pd.to_numeric(item.price, errors="coerce").to_numpy(dtype=np.float64)
        categories = pd.to_numeric(item.category, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        price_valid = np.isfinite(prices) & (prices >= 0)

        action_counts[action] = np.bincount(rows, minlength=len(clients)).astype(np.float32)
        action_price_sum[action] = np.bincount(rows[price_valid], weights=prices[price_valid],
                                               minlength=len(clients)).astype(np.float32)
        action_price_sq_sum[action] = np.bincount(rows[price_valid], weights=prices[price_valid] ** 2,
                                                  minlength=len(clients)).astype(np.float32)
        action_price_count[action] = np.bincount(rows[price_valid], minlength=len(clients)).astype(np.float32)
        cat_valid = categories >= 0
        flat = rows[cat_valid] * category_bins + np.mod(categories[cat_valid], category_bins)
        category_profile += np.bincount(
            flat, weights=np.full(len(flat), CATEGORY_WEIGHTS[action], dtype=np.float32),
            minlength=len(clients) * category_bins).reshape(len(clients), category_bins)
        for suffix, duration in WINDOWS:
            recent = seconds >= as_of_time - duration
            window_counts[(action, suffix)] = np.bincount(rows[recent], minlength=len(clients)).astype(np.float32)
            recent_price = recent & price_valid
            window_price_sum[(action, suffix)] = np.bincount(
                rows[recent_price], weights=prices[recent_price], minlength=len(clients)).astype(np.float32)
            window_price_count[(action, suffix)] = np.bincount(
                rows[recent_price], minlength=len(clients)).astype(np.float32)

    search = pd.read_parquet(data_dir / "input/search_query.parquet", columns=["client_id"])
    search_rows = client_index.get_indexer(search.client_id.to_numpy())
    search_count = np.bincount(search_rows[search_rows >= 0], minlength=len(clients)).astype(np.float32)
    # Official StatsFeaturesCalculator puts page_visit event_count at offset 273.
    page_count = np.asarray(base[:, 273], dtype=np.float32)

    scalar = []
    scalar.extend(action_counts[action] for action in ("click", "collect", "buy"))
    scalar.extend((search_count, page_count))
    scalar.extend((_safe_ratio(action_counts["collect"], page_count),
                   _safe_ratio(action_counts["buy"], action_counts["collect"]),
                   _safe_ratio(action_counts["buy"], search_count)))
    for suffix, _ in WINDOWS:
        scalar.extend(window_counts[(action, suffix)] for action in ("click", "collect", "buy"))
        scalar.extend((_safe_ratio(window_counts[("collect", suffix)], page_count),
                       _safe_ratio(window_counts[("buy", suffix)], window_counts[("collect", suffix)])))
    for action in ("click", "collect", "buy"):
        mean = _safe_ratio(action_price_sum[action], action_price_count[action])
        variance = np.maximum(_safe_ratio(action_price_sq_sum[action], action_price_count[action]) - mean ** 2, 0)
        scalar.extend((mean, np.sqrt(variance)))
    for suffix, _ in WINDOWS:
        scalar.extend(_safe_ratio(window_price_sum[(action, suffix)],
                                  window_price_count[(action, suffix)])
                      for action in ("click", "collect", "buy"))
    buy_mean = _safe_ratio(action_price_sum["buy"], action_price_count["buy"])
    collect_mean = _safe_ratio(action_price_sum["collect"], action_price_count["collect"])
    scalar.append(_safe_ratio(buy_mean, collect_mean))
    category_profile = _safe_ratio(category_profile, category_profile.sum(axis=1, keepdims=True))
    expanded = np.column_stack(scalar).astype(np.float32)
    expanded[:, :len(scalar)] = np.log1p(np.maximum(expanded, 0))
    return np.concatenate((np.asarray(base, dtype=np.float32), expanded, category_profile), axis=1), {
        "official_base_embeddings": digest(base_dir / "embeddings.npy"),
        "official_base_client_ids": digest(base_dir / "client_ids.npy"),
        "product_properties.parquet": digest(data_dir / "product_properties.parquet"),
        "feature_groups": ["conversion", "category", "price"],
    }


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

    if config.get("profile_version", 1) == 2:
        features, expanded_manifest = _expanded_profiles(
            config, data_dir, clients, as_of_time, root)
        if not np.isfinite(features).all():
            raise ValueError("non-finite expanded OpenRec features")
        features = features.astype(np.float16)
        output.mkdir(parents=True)
        np.save(output / "client_ids.npy", clients)
        np.save(output / "embeddings.npy", features)
        write_json(output / "manifest.json", {
            "schema": 2, "task": "synerise_official_profiles",
            "config": config, "users": len(clients), "dimensions": features.shape[1],
            "event_types": list(ACTION_TYPES), "as_of_time": as_of_time,
            "expanded_features": expanded_manifest,
            "client_ids_sha256": digest(output / "client_ids.npy"),
            "embeddings_sha256": digest(output / "embeddings.npy"),
            "openrec_git": git_state(root), "source_sha256": digest(__file__),
        })
        return {"users": len(clients), "dimensions": features.shape[1]}

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
