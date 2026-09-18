"""Snapshot features computed by OpenRec, with explicit feedback visibility."""
import numpy as np
import pandas as pd


def materialize(frame, config):
    from algorithm.feature.event_feature import aggregate_event_features

    interval = int(config["update_interval_ms"])
    delay = int(config["feedback_delay_ms"])
    if interval <= 0 or delay < 0:
        raise ValueError("update interval must be positive and feedback delay nonnegative")
    train_end = int(pd.Timestamp(config["validation_start"]).value // 1_000_000)
    # Both standard and random observations are retained in prepared data, but
    # random observations never feed this baseline's training or feature history.
    selected = config.get("feature_selection", {})
    needs_behavior = any(
        ".event_" in feature
        for features in selected.values()
        for feature in features
    ) if selected else True
    if needs_behavior:
        history = frame[frame.policy.ne("random")].copy()
        history["time"] = history.timestamp // 1000
        history["type"] = np.where(history.label.eq(1), "click", "expose")
        history["event_id"] = history.sample_id
        history["value"] = 0.0
    cutoffs = (frame.timestamp // interval) * interval - delay
    if config["evaluation_feedback"] == "frozen":
        cutoffs = cutoffs.clip(upper=train_end - delay)
    elif config["evaluation_feedback"] != "delayed_replay":
        raise ValueError("evaluation_feedback must be frozen or delayed_replay")
    users = pd.DataFrame(index=frame.index)
    items = pd.DataFrame(index=frame.index)
    users["id"] = frame.user_id
    items["id"] = frame.item_id
    items["scene"] = frame.scene
    for name in ["category", "subcategory", "tags", "title", "pub_time"]:
        if name in frame:
            items[name] = frame[name]
    if "age" in frame:
        users["age"] = frame.age
    # Recompute with the product implementation for correctness. This reference
    # materializer is intentionally not a claim of incremental scalability.
    if needs_behavior:
        for cutoff, index in cutoffs.groupby(cutoffs).groups.items():
            visible = history[history.timestamp < cutoff]
            for entity, destination in [("user", users), ("item", items)]:
                aggregate = aggregate_event_features(visible, entity, as_of_time=cutoff // 1000)
                aggregate = aggregate.set_index(f"{entity}_id")
                for column in ["event_count", "event_click_rate"]:
                    destination.loc[index, column] = destination.loc[index, "id"].map(aggregate[column]).fillna(0)
    if "pub_time" in items:
        from algorithm.feature.content_feature import enrich_item_content_features
        items = enrich_item_content_features(items, frame.timestamp / 1000)
    return users, items
