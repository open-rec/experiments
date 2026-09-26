"""Music-CRS adapter for OpenRec's generic candidate LightGBM ranker."""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _ordered_union(channels, limit):
    scores = {}
    for values in channels.values():
        for rank, (item, _) in enumerate(values, 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (60.0 + rank)
    return [item for item, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]


def _load_queries(directories, limit=None):
    rows = []
    vectors = []
    for directory in directories:
        root = Path(directory)
        meta = pd.read_parquet(root / "query_meta.parquet")
        matrix = np.load(root / "query_embeddings.npy", mmap_mode="r")
        if len(meta) != len(matrix):
            raise ValueError("query metadata and embeddings are not aligned: %s" % root)
        remaining = None if limit is None else max(0, int(limit) - len(rows))
        if remaining == 0:
            break
        count = len(meta) if remaining is None else min(len(meta), remaining)
        rows.extend(meta.iloc[:count].to_dict("records"))
        vectors.append(np.asarray(matrix[:count], dtype=np.float32))
    if not vectors:
        raise ValueError("no query embeddings were loaded")
    return rows, np.concatenate(vectors)


def _channel_pairs(results):
    return [(str(value.item), float(value.score)) for value in results]


def _transition_indexes(sessions):
    global_counts = defaultdict(Counter)
    session_counts = {}
    for session in sessions.itertuples(index=False):
        music = [str(value["content"]) for value in session.conversations
                 if value["role"] == "music"]
        own = Counter(zip(music, music[1:]))
        session_counts[str(session.session_id)] = own
        for (left, right), count in own.items():
            global_counts[left][right] += count
    return global_counts, session_counts


def _transition_pairs(history, session_id, global_counts, session_counts, limit):
    scores = {}
    own = session_counts.get(str(session_id), {})
    for offset, left in enumerate(history[-5:]):
        recency = 1.0 / (len(history[-5:]) - offset)
        for right, count in global_counts.get(left, {}).items():
            adjusted = max(0, count - own.get((left, right), 0))
            if adjusted:
                scores[right] = max(scores.get(right, 0.0), math.log1p(adjusted) * recency)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]


def _history_pairs(builder, recall, history, count):
    positions = [builder.index[str(value)] for value in history if str(value) in builder.index]
    if not positions:
        return []
    query = np.mean(builder.vectors[positions[-5:]], axis=0, dtype=np.float32)
    return _channel_pairs(recall.recall(query, exclude=history, recall_size=count))


def _materialize(rows, query_vectors, query_recall, history_recall, builder,
                 candidate_count, history_count, hot, require_positive,
                 transitions, session_transitions, sparse_recall=None,
                 sparse_count=100, keep_identities=False,
                 batch_size=512):
    feature_blocks, labels, groups, identities = [], [], [], []
    retrieved_positive = 0
    group_number = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        batch_vectors = query_vectors[start:start + batch_size]
        histories = [[str(value) for value in row.get("prior_track_ids", ())]
                     for row in batch_rows]
        query_batches = query_recall.recall_many(
            batch_vectors, excludes=histories, recall_size=candidate_count,
            block_size=min(batch_size, 128))
        history_queries = np.zeros_like(batch_vectors, dtype=np.float32)
        for position, history in enumerate(histories):
            indices = [builder.index[value] for value in history[-5:] if value in builder.index]
            if indices:
                history_queries[position] = np.mean(builder.vectors[indices], axis=0)
        history_batches = history_recall.recall_many(
            history_queries, excludes=histories, recall_size=history_count,
            block_size=min(batch_size, 128))
        for row, query, history, query_result, history_result in zip(
                batch_rows, batch_vectors, histories, query_batches, history_batches):
            query_items = _channel_pairs(query_result)
            history_items = _channel_pairs(history_result)
            transition_items = _transition_pairs(
                history, row["session_id"], transitions, session_transitions, history_count)
            channels = {"query_embedding": query_items,
                        "history_embedding": history_items,
                        "sequence_transition": transition_items, "popular": hot}
            if sparse_recall is not None:
                channels["sparse"] = _channel_pairs(sparse_recall.recall(
                    row.get("query_text", ""), exclude=history,
                    recall_size=sparse_count))
            candidates = _ordered_union(channels, candidate_count)
            gt = str(row.get("gt_track_id", ""))
            hit = gt in candidates
            retrieved_positive += int(hit)
            if require_positive and not hit:
                continue
            ids, features = builder.transform(
                candidates, query_vector=query, history_ids=history,
                channel_results=channels, request_position=int(row["turn_number"]),
                request_time=row.get("session_date"), query_text=row.get("query_text"))
            feature_blocks.append(features)
            labels.extend(int(value == gt) for value in ids)
            groups.extend([group_number] * len(ids))
            if keep_identities:
                identities.append((row, ids, features))
            group_number += 1
    matrix = (np.concatenate(feature_blocks) if feature_blocks else
              np.empty((0, len(builder.feature_names)), dtype=np.float32))
    return (matrix, np.asarray(labels, dtype=np.int8), np.asarray(groups, dtype=np.int32), identities,
            {"requests": len(rows), "positive_retrieved": retrieved_positive,
             "recall": retrieved_positive / max(len(rows), 1),
             "trained_groups": len(feature_blocks), "candidate_rows": len(matrix)})


def _group_ndcg(labels, scores, groups, cutoff=20):
    values = []
    start = 0
    while start < len(groups):
        end = start + 1
        while end < len(groups) and groups[end] == groups[start]:
            end += 1
        order = np.argsort(-scores[start:end], kind="stable")[:cutoff]
        positives = np.flatnonzero(labels[start:end][order] > 0)
        values.append(0.0 if not len(positives) else 1.0 / np.log2(positives[0] + 2.0))
        start = end
    return float(np.mean(values))


def _feature_indices(names, feature_set):
    if feature_set == "fusion":
        return np.arange(len(names), dtype=np.int64)
    if feature_set != "base":
        raise ValueError("unknown candidate ranking feature set: %s" % feature_set)
    added = {"recall.rank_std", "recall.agreement_top10", "recall.agreement_top50",
             "recall.score_minmax_sum", "recall.score_minmax_mnz",
             "recall.rank_product_score"}
    return np.asarray([
        position for position, name in enumerate(names)
        if name not in added and not name.endswith((
            ".score_minmax", ".score_z", ".rank_pct", ".in_top10"))
    ], dtype=np.int64)


def train_and_predict(config, catalog, hot_ids, dev_query_dir, output,
                      train_sessions, dev_sessions):
    """Train with OOF requests and score dev requests without reading dev labels."""
    from algorithm.feature.candidate_ranking import CandidateRankingFeatureBuilder
    from algorithm.rank.lightgbm import LightGBMRankModel
    from algorithm.recall.bm25 import BM25Recall
    from algorithm.recall.query_embedding import QueryEmbeddingRecall

    track_root = Path(config["track_embedding_dir"])
    item_ids = np.load(track_root / "track_ids.npy", allow_pickle=True).astype(str)
    item_vectors = np.load(track_root / "embeddings.npy", mmap_mode="r")
    metadata = catalog.assign(track_id=catalog.track_id.astype(str)).set_index("track_id").reindex(item_ids)
    texts = (metadata.track_name.fillna("").astype(str) + " "
             + metadata.artist_name.fillna("").astype(str) + " "
             + metadata.album_name.fillna("").astype(str) + " "
             + metadata.tag_list.map(lambda value: " ".join(map(str, value))
                                     if isinstance(value, (list, tuple, np.ndarray)) else str(value)))
    sparse_enabled = bool(config.get("sparse_recall", False))
    channel_names = ["query_embedding", "history_embedding",
                     "sequence_transition", "popular"]
    if sparse_enabled:
        channel_names.append("sparse")
    builder = CandidateRankingFeatureBuilder(
        item_ids, item_vectors,
        primary_entities=metadata.artist_id.fillna("").astype(str).to_numpy(),
        tags=metadata.tag_list.to_numpy(),
        popularity=pd.to_numeric(metadata.popularity, errors="coerce").fillna(0).to_numpy(),
        release_times=pd.to_datetime(metadata.release_date, errors="coerce").to_numpy(),
        texts=texts.to_numpy(),
        channel_names=channel_names)
    candidate_count = int(config.get("rank_candidate_count", 200))
    train_candidate_count = int(config.get("rank_train_candidate_count", 120))
    history_count = int(config.get("rank_history_candidate_count", 80))
    sparse_count = int(config.get("rank_sparse_candidate_count", 100))
    query_recall = QueryEmbeddingRecall(item_ids, item_vectors, recall_size=candidate_count,
                                        device=config.get("query_embedding_device", "cpu"))
    history_recall = QueryEmbeddingRecall(item_ids, item_vectors, recall_size=history_count,
                                          device=config.get("query_embedding_device", "cpu"))
    sparse_recall = None
    if sparse_enabled:
        def text(value):
            if isinstance(value, (list, tuple, np.ndarray)):
                return " ".join(map(str, value))
            return "" if value is None else str(value)

        documents = {
            item_id: {
                "title": text(row.track_name),
                "artist": text(row.artist_name),
                "album": text(row.album_name),
                "tags": text(row.tag_list),
            }
            for item_id, row in metadata.iterrows()
        }
        sparse_recall = BM25Recall(
            documents, recall_size=sparse_count,
            field_weights=config.get("sparse_field_weights", {
                "title": 2.0, "artist": 1.5, "album": 1.0, "tags": 0.5,
            }),
            k1=float(config.get("sparse_k1", 1.2)),
            b=float(config.get("sparse_b", 0.75)))
    hot = [(str(value), 1.0 / rank) for rank, value in enumerate(hot_ids[:100], 1)]
    transitions, session_transitions = _transition_indexes(train_sessions)
    dates = dict(zip(train_sessions.session_id.astype(str), train_sessions.session_date))
    dates.update(zip(dev_sessions.session_id.astype(str), dev_sessions.session_date))

    train_rows, train_vectors = _load_queries(
        config["rank_train_query_dirs"], config.get("rank_max_train_requests"))
    val_rows, val_vectors = _load_queries(
        config["rank_validation_query_dirs"], config.get("rank_max_validation_requests"))
    for row in train_rows + val_rows:
        row["session_date"] = dates.get(str(row["session_id"]))
    train = _materialize(train_rows, train_vectors, query_recall, history_recall, builder,
                         train_candidate_count, history_count, hot, True,
                         transitions, session_transitions, sparse_recall, sparse_count)
    validation = _materialize(val_rows, val_vectors, query_recall, history_recall, builder,
                              train_candidate_count, history_count, hot, True,
                              transitions, session_transitions, sparse_recall, sparse_count)
    params = dict(config.get("lightgbm", {}))
    params.setdefault("random_state", int(config.get("seed", 42)))
    params.setdefault("n_jobs", int(config.get("threads", 8)))
    objectives = config.get("rank_objective_candidates", [params.get("objective", "lambdarank")])
    feature_sets = config.get("rank_feature_set_candidates", ["fusion"])
    trials = []
    for feature_set in feature_sets:
        indices = _feature_indices(builder.feature_names, feature_set)
        for objective in objectives:
            trial_params = dict(params, objective=objective)
            candidate_model = LightGBMRankModel(**trial_params).fit(
                train[0][:, indices], train[1], train[2],
                validation=(validation[0][:, indices], validation[1], validation[2]))
            validation_score = _group_ndcg(
                validation[1], candidate_model.predict_proba(validation[0][:, indices]),
                validation[2], 20)
            trials.append((validation_score, str(objective), str(feature_set),
                           candidate_model, indices))
    validation_score, selected_objective, selected_feature_set, model, selected = max(
        trials, key=lambda value: value[0])

    dev_rows, dev_vectors = _load_queries([dev_query_dir])
    for row in dev_rows:
        row["session_date"] = dates.get(str(row["session_id"]))
    dev = _materialize(dev_rows, dev_vectors, query_recall, history_recall, builder,
                       candidate_count, history_count, hot, False,
                       transitions, session_transitions, sparse_recall, sparse_count,
                       keep_identities=True)
    predictions = []
    for row, ids, features in dev[3]:
        scores = model.predict_proba(features[:, selected])
        order = np.argsort(-scores, kind="stable")
        ranked = [ids[position] for position in order[:20]]
        if len(ranked) < 20:
            ranked.extend(value for value in hot_ids if value not in ranked and value not in
                          set(str(item) for item in row.get("prior_track_ids", ())))
        predictions.append({
            "session_id": str(row["session_id"]), "user_id": str(row["user_id"]),
            "turn_number": int(row["turn_number"]), "predicted_track_ids": ranked[:20],
            "predicted_response": "",
        })
    output = Path(output)
    model.save(output / "lightgbm.txt")
    (output / "lightgbm.features.json").write_text(json.dumps({
        "schema": 1, "feature_names": [builder.feature_names[value] for value in selected],
        "channels": list(builder.channel_names),
    }, indent=2) + "\n")
    return predictions, {"train": train[4], "validation": validation[4], "dev": dev[4],
                         "channels": list(builder.channel_names),
                         "best_iteration": int(model.booster.best_iteration or 0),
                         "feature_count": len(selected),
                         "selected_objective": selected_objective,
                         "selected_feature_set": selected_feature_set,
                         "validation_ndcg_at_20": validation_score,
                         "validation_trials": {
                             "%s:%s" % (feature_set, objective): score
                             for score, objective, feature_set, _, _ in trials}}
