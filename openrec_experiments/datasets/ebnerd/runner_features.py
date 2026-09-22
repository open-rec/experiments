"""EB-NeRD article, history, context and exposure features."""
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from ...temporal import timestamp


def history_inputs(frame, config, semantic_matrix, semantic_ids):
    max_history = int(config.get("max_history", 50))
    if max_history < 1:
        raise ValueError("max_history must be positive")
    article_lookup = {article_id: i for i, article_id in enumerate(semantic_ids)}
    tables, keys = [], {}
    for source_index, path in enumerate(config["history_files"]):
        history = pd.read_parquet(path, columns=["user_id", "article_id_fixed"])
        if history.user_id.duplicated().any():
            raise ValueError("history file contains duplicate users")
        for row in history.itertuples(index=False):
            article_indices = [article_lookup.get(str(value), -1)
                               for value in row.article_id_fixed[-max_history:]]
            valid = [value for value in article_indices if value >= 0]
            vector = np.zeros((max_history, semantic_matrix.shape[1]), dtype=np.float32)
            mask = np.ones(max_history, dtype=bool)
            if valid:
                vector[-len(valid):] = semantic_matrix[valid]
                mask[-len(valid):] = False
            keys[(source_index, str(row.user_id))] = len(tables)
            tables.append((vector, mask))
    if len(config["history_files"]) != 2:
        raise ValueError("EB-NeRD Transformer requires train and validation history files")
    test_start = timestamp(config["test_start"])
    history_indices = np.empty(len(frame), dtype=np.int64)
    for i, (user, time_value) in enumerate(zip(frame.user_id, frame.timestamp)):
        source_index = 0 if time_value < test_start else 1
        key = (source_index, str(user))
        if key not in keys:
            raise ValueError("sample user is missing from its point-in-time history file")
        history_indices[i] = keys[key]
    vectors = np.stack([value[0] for value in tables])
    masks = np.stack([value[1] for value in tables])
    return vectors, masks, history_indices


def fixed_history_features(frame, config):
    """Materialize request-time candidate affinity from EB-NeRD history snapshots."""
    paths = [Path(path) for path in config.get("history_files", [])]
    if len(paths) != 2:
        raise ValueError("fixed history features require train and validation history files")
    articles = pd.read_parquet(
        config["article_metadata"], columns=["article_id", "category", "topics"]
    )
    article_category = dict(zip(
        articles.article_id.astype(str), articles.category.fillna("").astype(str)
    ))
    article_topics = {
        str(article): tuple(str(value) for value in values)
        if isinstance(values, (list, tuple, np.ndarray)) else ()
        for article, values in zip(articles.article_id, articles.topics)
    }
    snapshots = {}
    for source_index, path in enumerate(paths):
        history = pd.read_parquet(path, columns=["user_id", "article_id_fixed"])
        if history.user_id.duplicated().any():
            raise ValueError("history file contains duplicate users")
        for row in history.itertuples(index=False):
            item_ids = [str(value) for value in row.article_id_fixed]
            item_counts = Counter(item_ids)
            last_positions = {
                item: offset for offset, item in enumerate(reversed(item_ids), 1)
            }
            categories = Counter(article_category.get(item, "") for item in item_ids)
            topics = Counter(
                topic for item in item_ids for topic in article_topics.get(item, ())
            )
            snapshots[(source_index, str(row.user_id))] = (
                len(item_ids), item_counts, last_positions, categories, topics
            )
    test_start = timestamp(config["test_start"])
    values = np.zeros((len(frame), 7), dtype=np.float32)
    for index, (user, item, time_value, category, tags) in enumerate(zip(
        frame.user_id.astype(str), frame.item_id.astype(str), frame.timestamp,
        frame.category.fillna("").astype(str), frame.tags.fillna("").astype(str),
    )):
        source_index = 0 if time_value < test_start else 1
        snapshot = snapshots.get((source_index, user))
        if snapshot is None:
            continue
        length, item_counts, last_positions, categories, topics = snapshot
        item_count = item_counts.get(item, 0)
        category_count = categories.get(category, 0)
        candidate_topics = [value for value in tags.split(",") if value]
        topic_count = sum(topics.get(value, 0) for value in candidate_topics)
        values[index] = (
            np.log1p(length), np.log1p(item_count),
            0.0 if item not in last_positions else 1.0 / last_positions[item],
            np.log1p(category_count), category_count / max(length, 1),
            np.log1p(topic_count), topic_count / max(sum(topics.values()), 1),
        )
    return values, [
        "statistical.user_fixed_history_length_log",
        "interaction.candidate_fixed_history_count_log",
        "interaction.candidate_fixed_history_recency_reciprocal",
        "interaction.fixed_history_category_count_log",
        "interaction.fixed_history_category_share",
        "interaction.fixed_history_topic_count_log",
        "interaction.fixed_history_topic_share",
    ]


def apply_semantic_title_fallback(items, semantic_present, candidate_indices):
    """Use title hash only for rows whose semantic source text is absent."""
    present = np.asarray(semantic_present, dtype=bool)
    indices = np.asarray(candidate_indices, dtype=np.int64)
    if len(items) != len(indices) or (indices < 0).any() or (indices >= len(present)).any():
        raise ValueError("semantic fallback inputs are not row-aligned")
    result = items.copy()
    result.loc[present[indices], "title"] = ""
    return result


def contextual_features(frame, train_mask):
    """Fit and materialize request/candidate context without label access."""
    required = {
        "position", "candidate_count", "timestamp", "device_type",
        "is_subscriber", "is_authenticated",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("contextual feature columns are missing: %s" % sorted(missing))
    train = frame.loc[train_mask]
    values, names = [], []
    for name in ("position", "candidate_count"):
        raw = pd.to_numeric(frame[name], errors="coerce")
        fitted = pd.to_numeric(train[name], errors="coerce")
        mean = float(fitted.mean()) if fitted.notna().any() else 0.0
        scale = float(fitted.std(ddof=0)) if fitted.notna().any() else 1.0
        scale = scale if scale > 1e-12 else 1.0
        values.append(np.clip((raw.fillna(mean).to_numpy() - mean) / scale, -3, 3))
        names.append("context." + name)
    denominator = pd.to_numeric(frame.candidate_count, errors="coerce").clip(lower=1)
    values.append(pd.to_numeric(frame.position, errors="coerce").fillna(0).to_numpy() / denominator)
    names.append("context.position_ratio")
    instant = pd.to_datetime(frame.timestamp, unit="ms", utc=True)
    for name, number, period in (
        ("hour", instant.dt.hour.to_numpy(), 24),
        ("weekday", instant.dt.weekday.to_numpy(), 7),
    ):
        values.extend([np.sin(2 * np.pi * number / period), np.cos(2 * np.pi * number / period)])
        names.extend(["context.%s_sin" % name, "context.%s_cos" % name])
    device = frame.device_type.fillna("").astype(str)
    for category in sorted(set(train.device_type.fillna("").astype(str))):
        values.append(device.eq(category).to_numpy(dtype=float))
        names.append("context.device_type=" + category)
    for name in ("is_subscriber", "is_authenticated"):
        values.append(frame[name].fillna(False).to_numpy(dtype=float))
        names.append("context." + name)
    return np.column_stack(values).astype(np.float32), names


def interaction_features(frame, config):
    """Causal user-candidate affinity features materialized at request cutoff."""
    required = {"user_id", "item_id", "category", "timestamp", "label", "policy"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("interaction feature columns are missing: %s" % sorted(missing))
    interval = int(config["update_interval_ms"])
    delay = int(config["feedback_delay_ms"])
    cutoffs = (frame.timestamp // interval) * interval - delay
    train_end = timestamp(config["validation_start"])
    if config["evaluation_feedback"] == "frozen":
        cutoffs = cutoffs.clip(upper=train_end - delay)
    elif config["evaluation_feedback"] != "delayed_replay":
        raise ValueError("evaluation_feedback must be frozen or delayed_replay")
    clicked = frame[
        frame.policy.ne("random") & frame.label.eq(1)
    ][["user_id", "item_id", "category", "timestamp"]]
    output = np.zeros((len(frame), 4), dtype=np.float32)
    missing_recency = np.log1p(24 * 30)
    for cutoff, index in cutoffs.groupby(cutoffs).groups.items():
        visible = clicked[clicked.timestamp < cutoff]
        rows = frame.loc[index]
        user_total = visible.groupby("user_id", sort=False).size()
        category_count = visible.groupby(
            ["user_id", "category"], sort=False
        ).size()
        category_last = visible.groupby(
            ["user_id", "category"], sort=False
        ).timestamp.max()
        item_count = visible.groupby(
            ["user_id", "item_id"], sort=False
        ).size()
        category_keys = pd.MultiIndex.from_arrays(
            [rows.user_id.to_numpy(), rows.category.astype(str).to_numpy()]
        )
        item_keys = pd.MultiIndex.from_arrays(
            [rows.user_id.to_numpy(), rows.item_id.to_numpy()]
        )
        category_values = category_count.reindex(category_keys, fill_value=0).to_numpy(dtype=float)
        totals = rows.user_id.map(user_total).fillna(0).to_numpy(dtype=float)
        last = category_last.reindex(category_keys).to_numpy(dtype=float)
        item_values = item_count.reindex(item_keys, fill_value=0).to_numpy(dtype=float)
        seen_category = category_values > 0
        recency = np.full(len(rows), missing_recency, dtype=float)
        recency[seen_category] = np.log1p(
            np.maximum(0, float(cutoff) - last[seen_category]) / 3_600_000
        )
        output[index, 0] = np.log1p(category_values)
        output[index, 1] = np.divide(
            category_values, totals, out=np.zeros_like(category_values), where=totals > 0
        )
        output[index, 2] = recency
        output[index, 3] = np.log1p(item_values)
    return output, [
        "interaction.user_category_click_count_log",
        "interaction.user_category_click_share",
        "interaction.user_category_recency_hours_log",
        "interaction.user_item_click_count_log",
    ]


def relative_temporal_features(frame):
    """Candidate freshness relative to the other items in the same request."""
    required = {"group_id", "timestamp", "pub_time"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("relative temporal columns are missing: %s" % sorted(missing))
    published_ms = pd.to_numeric(frame.pub_time, errors="coerce").fillna(0).to_numpy() * 1000
    age_hours = np.maximum(0, frame.timestamp.to_numpy(dtype=float) - published_ms) / 3_600_000
    valid = published_ms > 0
    age_hours[~valid] = 24 * 365 * 10
    age = pd.Series(age_hours, index=frame.index)
    grouped = age.groupby(frame.group_id, sort=False)
    freshest = grouped.transform("min").to_numpy()
    count = frame.groupby("group_id", sort=False).group_id.transform("size").to_numpy()
    rank = grouped.rank(method="average", ascending=True).to_numpy()
    values = np.column_stack([
        np.log1p(age_hours),
        (rank - 1) / np.maximum(1, count - 1),
        np.log1p(np.maximum(0, age_hours - freshest)),
        age_hours <= 24,
    ]).astype(np.float32)
    return values, [
        "temporal.candidate_age_hours_log",
        "temporal.freshness_rank_ratio",
        "temporal.hours_from_freshest_log",
        "temporal.published_within_24h",
    ]


def long_term_user_features(frame, config):
    """Causal long-horizon user activity and consumed-content profile."""
    interval, delay = int(config["update_interval_ms"]), int(config["feedback_delay_ms"])
    cutoffs = (frame.timestamp // interval) * interval - delay
    train_end = timestamp(config["validation_start"])
    if config["evaluation_feedback"] == "frozen":
        cutoffs = cutoffs.clip(upper=train_end - delay)
    clicked = frame[frame.policy.ne("random") & frame.label.eq(1)].copy()
    published_ms = pd.to_numeric(clicked.pub_time, errors="coerce").fillna(0) * 1000
    clicked["consumed_age_hours"] = np.maximum(0, clicked.timestamp - published_ms) / 3_600_000
    clicked["active_day"] = clicked.timestamp // 86_400_000
    output = np.zeros((len(frame), 9), dtype=np.float32)
    for cutoff, index in cutoffs.groupby(cutoffs).groups.items():
        visible = clicked[clicked.timestamp < cutoff]
        grouped = visible.groupby("user_id", sort=False)
        stats = grouped.agg(
            click_count=("item_id", "size"),
            active_days=("active_day", "nunique"),
            unique_items=("item_id", "nunique"),
            unique_categories=("category", "nunique"),
            age_mean=("consumed_age_hours", "mean"),
            age_std=("consumed_age_hours", "std"),
            last_click=("timestamp", "max"),
        )
        fresh_share = visible.assign(fresh=visible.consumed_age_hours.le(24)).groupby(
            "user_id", sort=False
        ).fresh.mean()
        repeat_share = 1 - stats.unique_items / stats.click_count.clip(lower=1)
        users = frame.loc[index, "user_id"]
        columns = [
            np.log1p(users.map(stats.click_count).fillna(0)),
            np.log1p(users.map(stats.active_days).fillna(0)),
            np.log1p(users.map(stats.unique_items).fillna(0)),
            np.log1p(users.map(stats.unique_categories).fillna(0)),
            np.log1p(users.map(stats.age_mean).fillna(0)),
            np.log1p(users.map(stats.age_std).fillna(0)),
            users.map(fresh_share).fillna(0),
            users.map(repeat_share).fillna(0),
            np.log1p(np.maximum(0, cutoff - users.map(stats.last_click).fillna(cutoff)) / 3_600_000),
        ]
        output[index] = np.column_stack(columns)
    return output, [
        "statistical.user_click_count_log", "statistical.user_active_days_log",
        "statistical.user_unique_items_log", "statistical.user_unique_categories_log",
        "statistical.user_consumed_age_mean_log", "statistical.user_consumed_age_std_log",
        "statistical.user_fresh_click_share", "statistical.user_repeat_click_share",
        "temporal.user_last_click_recency_hours_log",
    ]


def topic_entity_affinity_features(frame, config):
    """Visible user token preferences matched to candidate topics/entities."""
    from scipy import sparse

    articles = pd.read_parquet(
        config["article_metadata"], columns=["article_id", "topics", "entity_groups"]
    )
    articles["item_id"] = articles.article_id.astype(str)
    users = pd.Index(frame.user_id.astype(str).unique())
    items = pd.Index(frame.item_id.astype(str).unique())
    user_index = pd.Series(np.arange(len(users)), index=users)
    item_index = pd.Series(np.arange(len(items)), index=items)
    row_users = frame.user_id.astype(str).map(user_index).to_numpy(dtype=np.int64)
    row_items = frame.item_id.astype(str).map(item_index).to_numpy(dtype=np.int64)

    def item_token_matrix(column):
        token_lists = articles.set_index("item_id")[column].reindex(items).map(
            lambda value: [str(token) for token in value]
            if isinstance(value, (list, tuple, np.ndarray)) else []
        )
        vocabulary = {token: i for i, token in enumerate(sorted({
            token for values in token_lists for token in values
        }))}
        rows, columns = [], []
        for row, values in enumerate(token_lists):
            for token in set(values):
                rows.append(row); columns.append(vocabulary[token])
        return sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, columns)),
            shape=(len(items), len(vocabulary)),
        )

    matrices = [item_token_matrix("topics"), item_token_matrix("entity_groups")]
    interval, delay = int(config["update_interval_ms"]), int(config["feedback_delay_ms"])
    cutoffs = (frame.timestamp // interval) * interval - delay
    train_end = timestamp(config["validation_start"])
    if config["evaluation_feedback"] == "frozen":
        cutoffs = cutoffs.clip(upper=train_end - delay)
    click_positions = np.flatnonzero(
        frame.policy.ne("random").to_numpy() & frame.label.eq(1).to_numpy()
    )
    output = np.zeros((len(frame), 8), dtype=np.float32)
    for cutoff, index in cutoffs.groupby(cutoffs).groups.items():
        visible = click_positions[frame.timestamp.to_numpy()[click_positions] < cutoff]
        incidence = sparse.csr_matrix(
            (np.ones(len(visible), dtype=np.float32),
             (row_users[visible], row_items[visible])),
            shape=(len(users), len(items)),
        )
        family_columns = []
        for item_tokens in matrices:
            preferences = incidence @ item_tokens
            selected_preferences = preferences[row_users[index]]
            selected_items = item_tokens[row_items[index]]
            matched = selected_preferences.multiply(selected_items)
            count_sum = np.asarray(matched.sum(axis=1)).ravel()
            overlap = np.asarray(matched.sign().sum(axis=1)).ravel()
            token_count = np.asarray(selected_items.sum(axis=1)).ravel()
            user_totals = np.asarray(preferences.sum(axis=1)).ravel()[row_users[index]]
            family_columns.extend([
                overlap,
                overlap / np.maximum(1, token_count),
                np.log1p(count_sum),
                np.divide(count_sum, user_totals, out=np.zeros(len(index)), where=user_totals > 0),
            ])
        output[index] = np.column_stack(family_columns)
    names = []
    for family in ("topic", "entity"):
        names.extend([
            "interaction.user_%s_overlap_count" % family,
            "interaction.user_%s_overlap_ratio" % family,
            "interaction.user_%s_click_count_log" % family,
            "interaction.user_%s_click_share" % family,
        ])
    return output, names


def candidate_statistical_ranks(frame, sources):
    """Within-request percentile ranks and centered values for candidate signals."""
    values, names = [], []
    groups = frame.group_id
    for name, raw in sources.items():
        series = pd.Series(np.asarray(raw, dtype=float), index=frame.index)
        rank = series.groupby(groups, sort=False).rank(
            method="average", ascending=False, pct=True
        )
        centered = series - series.groupby(groups, sort=False).transform("mean")
        values.extend([rank.to_numpy(), centered.to_numpy()])
        names.extend([
            "context.%s_rank_pct" % name,
            "context.%s_minus_request_mean" % name,
        ])
    return np.column_stack(values).astype(np.float32), names


def session_exposure_features(frame):
    """Causal request frequency and repeated-candidate exposure features."""
    from collections import deque

    output = np.zeros((len(frame), 8), dtype=np.float32)
    impressions = frame.groupby("group_id", sort=False).agg(
        user_id=("user_id", "first"), timestamp=("timestamp", "first"),
        row_indices=("sample_id", lambda value: list(value.index)),
        item_ids=("item_id", list),
    ).reset_index()
    for _, user_impressions in impressions.groupby("user_id", sort=False):
        ordered = user_impressions.sort_values("timestamp", kind="stable")
        times = deque()
        recent = deque(maxlen=20)
        previous_time = None
        for row in ordered.itertuples(index=False):
            now = int(row.timestamp)
            while times and times[0] < now - 86_400_000:
                times.popleft()
            count_24h = len(times)
            count_1h = sum(value >= now - 3_600_000 for value in times)
            elapsed = 0 if previous_time is None else max(0, now - previous_time) / 60_000
            unions = {}
            for width in (1, 2, 5, 10, 20):
                selected = list(recent)[-width:]
                unions[width] = set().union(*selected) if selected else set()
            for index, item_id in zip(row.row_indices, row.item_ids):
                output[index] = [
                    np.log1p(count_1h), np.log1p(count_24h), np.log1p(elapsed),
                    *(float(item_id in unions[width]) for width in (1, 2, 5, 10, 20)),
                ]
            times.append(now)
            recent.append(set(row.item_ids))
            previous_time = now
    return output, [
        "statistical.user_impression_count_1h_log",
        "statistical.user_impression_count_24h_log",
        "temporal.user_previous_impression_minutes_log",
        "interaction.candidate_seen_previous_1",
        "interaction.candidate_seen_previous_2",
        "interaction.candidate_seen_previous_5",
        "interaction.candidate_seen_previous_10",
        "interaction.candidate_seen_previous_20",
    ]


def past_candidate_exposure_features(frame, extended=False):
    """Causal candidate exposure counts before the current request timestamp."""
    windows = ("5m", "1h", "24h") if extended else ("5m", "1h")
    global_counts = {window: Counter() for window in (*windows, "all")}
    user_counts = {window: Counter() for window in (*windows, "all")}
    global_expiry = {window: deque() for window in windows}
    user_expiry = {window: deque() for window in windows}
    widths = {"5m": 300_000, "1h": 3_600_000, "24h": 86_400_000}
    sources = [
        *(('global_' + window) for window in windows), "global_all",
        *(('user_' + window) for window in windows), "user_all",
    ]
    raw = np.zeros((len(frame), len(sources)), dtype=np.float32)
    impressions = frame.groupby("group_id", sort=False).agg(
        user_id=("user_id", "first"), timestamp=("timestamp", "first"),
        row_indices=("sample_id", lambda value: list(value.index)),
        item_ids=("item_id", list),
    ).reset_index().sort_values("timestamp", kind="stable")
    for now, simultaneous in impressions.groupby("timestamp", sort=False):
        now = int(now)
        for window in windows:
            while global_expiry[window] and global_expiry[window][0][0] <= now:
                _, item = global_expiry[window].popleft()
                global_counts[window][item] -= 1
            while user_expiry[window] and user_expiry[window][0][0] <= now:
                _, key = user_expiry[window].popleft()
                user_counts[window][key] -= 1
        pending = []
        for row in simultaneous.itertuples(index=False):
            user = str(row.user_id)
            for index, item_value in zip(row.row_indices, row.item_ids):
                item = str(item_value)
                key = (user, item)
                raw[index] = [
                    *(global_counts[window][item] for window in windows),
                    global_counts["all"][item],
                    *(user_counts[window][key] for window in windows),
                    user_counts["all"][key],
                ]
                pending.append((user, item, key))
        for user, item, key in pending:
            global_counts["all"][item] += 1
            user_counts["all"][key] += 1
            for window in windows:
                width = widths[window]
                global_counts[window][item] += 1
                user_counts[window][key] += 1
                global_expiry[window].append((now + width, item))
                user_expiry[window].append((now + width, key))
    values, names = [], []
    groups = frame.group_id
    for column, name in enumerate(sources):
        series = pd.Series(raw[:, column], index=frame.index)
        total = series.groupby(groups, sort=False).transform("sum")
        ratio = np.divide(
            series.to_numpy(), total.to_numpy(),
            out=np.zeros(len(frame), dtype=np.float32), where=total.to_numpy() > 0,
        )
        rank = series.groupby(groups, sort=False).rank(
            method="average", ascending=False, pct=True
        ).to_numpy()
        values.extend([np.log1p(series.to_numpy()), ratio, rank])
        names.extend([
            "statistical.candidate_past_exposure_%s_log" % name,
            "context.candidate_past_exposure_%s_ratio" % name,
            "context.candidate_past_exposure_%s_rank_pct" % name,
        ])
    if extended:
        positions = {name: index for index, name in enumerate(sources)}
        for scope in ("global", "user"):
            for short, long, scale in (("5m", "1h", 12), ("1h", "24h", 24)):
                short_values = raw[:, positions[scope + "_" + short]]
                long_values = raw[:, positions[scope + "_" + long]]
                momentum = np.log1p((short_values + 1) / (long_values / scale + 1))
                series = pd.Series(momentum, index=frame.index)
                rank = series.groupby(groups, sort=False).rank(
                    method="average", ascending=False, pct=True
                ).to_numpy()
                values.extend([momentum, rank])
                names.extend([
                    "statistical.candidate_past_exposure_%s_%s_%s_momentum_log"
                    % (scope, short, long),
                    "context.candidate_past_exposure_%s_%s_%s_momentum_rank_pct"
                    % (scope, short, long),
                ])
    return np.column_stack(values).astype(np.float32), names


def past_candidate_engagement_features(frame):
    """Causal candidate engagement accumulated from completed past requests."""
    required = {"read_time", "scroll_percentage", "candidate_count"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("past engagement columns are missing: %s" % sorted(missing))
    windows = ("5m", "1h")
    widths = {"5m": 300_000, "1h": 3_600_000}
    global_sums = {window: Counter() for window in (*windows, "all")}
    user_sums = {window: Counter() for window in (*windows, "all")}
    global_expiry = {window: deque() for window in windows}
    user_expiry = {window: deque() for window in windows}
    raw = np.zeros((len(frame), 6), dtype=np.float32)
    impressions = frame.groupby("group_id", sort=False).agg(
        user_id=("user_id", "first"), timestamp=("timestamp", "first"),
        read_time=("read_time", "first"),
        scroll_percentage=("scroll_percentage", "first"),
        candidate_count=("candidate_count", "first"),
        row_indices=("sample_id", lambda value: list(value.index)),
        item_ids=("item_id", list),
    ).reset_index().sort_values("timestamp", kind="stable")
    for now, simultaneous in impressions.groupby("timestamp", sort=False):
        now = int(now)
        for window in windows:
            while global_expiry[window] and global_expiry[window][0][0] <= now:
                _, item, weight = global_expiry[window].popleft()
                global_sums[window][item] -= weight
            while user_expiry[window] and user_expiry[window][0][0] <= now:
                _, key, weight = user_expiry[window].popleft()
                user_sums[window][key] -= weight
        pending = []
        for row in simultaneous.itertuples(index=False):
            user = str(row.user_id)
            weight = (
                0.0 if pd.isna(row.scroll_percentage)
                else max(float(row.read_time), 0.0) / max(int(row.candidate_count), 1)
            )
            for index, item_value in zip(row.row_indices, row.item_ids):
                item = str(item_value)
                key = (user, item)
                raw[index] = [
                    global_sums["5m"][item], global_sums["1h"][item],
                    global_sums["all"][item], user_sums["5m"][key],
                    user_sums["1h"][key], user_sums["all"][key],
                ]
                pending.append((item, key, weight))
        for item, key, weight in pending:
            global_sums["all"][item] += weight
            user_sums["all"][key] += weight
            for window in windows:
                global_sums[window][item] += weight
                user_sums[window][key] += weight
                expires = now + widths[window]
                global_expiry[window].append((expires, item, weight))
                user_expiry[window].append((expires, key, weight))
    values, names = [], []
    groups = frame.group_id
    sources = (
        "global_5m", "global_1h", "global_all",
        "user_5m", "user_1h", "user_all",
    )
    for column, name in enumerate(sources):
        series = pd.Series(raw[:, column], index=frame.index)
        total = series.groupby(groups, sort=False).transform("sum")
        ratio = np.divide(
            series.to_numpy(), total.to_numpy(),
            out=np.zeros(len(frame), dtype=np.float32), where=total.to_numpy() > 0,
        )
        rank = series.groupby(groups, sort=False).rank(
            method="average", ascending=False, pct=True
        ).to_numpy()
        values.extend([np.log1p(series.to_numpy()), ratio, rank])
        names.extend([
            "statistical.candidate_past_engagement_%s_log" % name,
            "context.candidate_past_engagement_%s_ratio" % name,
            "context.candidate_past_engagement_%s_rank_pct" % name,
        ])
    return np.column_stack(values).astype(np.float32), names
