"""Music-CRS devset inference scored with the organizer's evaluator functions."""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ...provenance import digest, git_state, write_json
from ...openrec import load_openrec


def _score(predictions, ground_truth, catalog_size, evaluator):
    sys.path.insert(0, str(evaluator))
    from metrics import compute_catalog_diversity, compute_lexical_diversity, compute_recsys_metrics

    pred = {(row["session_id"], int(row["turn_number"])): row for row in predictions}
    truth = {(row["session_id"], int(row["turn_number"])): row for row in ground_truth}
    if len(pred) != len(predictions) or pred.keys() != truth.keys():
        raise ValueError("predictions must cover every official devset turn exactly once")
    per_turn = {turn: [] for turn in range(1, 9)}
    recommended, responses = [], []
    for key, row in truth.items():
        candidate = pred[key]
        ids = candidate["predicted_track_ids"]
        if len(ids) != 20 or len(set(ids)) != 20:
            raise ValueError("each turn requires 20 unique tracks")
        if candidate["user_id"] != row["user_id"]:
            raise ValueError("user identity mismatch")
        per_turn[key[1]].append(compute_recsys_metrics(
            ids, [row["ground_truth_track_id"]], [1, 10, 20]
        ))
        recommended.extend(ids)
        responses.append(candidate["predicted_response"])
    scores = {metric: float(np.mean([
        np.mean([entry[metric] for entry in per_turn[turn]])
        for turn in range(1, 9)
    ])) for metric in ("ndcg@1", "ndcg@10", "ndcg@20")}
    scores["catalog_diversity"] = compute_catalog_diversity(recommended, catalog_size)
    scores["lexical_diversity"] = compute_lexical_diversity(responses)
    scores["total_catalog_size"] = catalog_size
    return scores


def _rank(history, neighbors, hot, k=20):
    excluded = set(history)
    scores = {}
    for trigger in history[-5:]:
        for item, score in neighbors.get(trigger, ()):
            if item not in excluded:
                scores[item] = max(scores.get(item, 0), score)
    ranked = [item for item, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]
    ranked.extend(item for item in hot if item not in excluded and item not in scores)
    return ranked[:k]


def _text(value):
    if isinstance(value, np.ndarray):
        return " ".join(str(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return "" if value is None else str(value)


class _SessionEntityRanker:
    """Generic entity/context/interaction reranker over the OpenRec recalls."""

    def __init__(self, tracks, sequences, hot, config, query_recall=None,
                 query_vectors=None, item_neighbors=None, user_candidates=None,
                 sequence_recall=None, sequence_vectors=None):
        self.ids = tracks.track_id.astype(str).tolist()
        self.index = {value: position for position, value in enumerate(self.ids)}
        self.hot = [self.index[value] for value in hot if value in self.index]
        self.artists = [_text(value) for value in tracks.artist_id]
        self.tags = [set(_text(value).lower().split()) for value in tracks.tag_list]
        popularity = np.log1p(pd.to_numeric(
            tracks.popularity, errors="coerce").fillna(0).to_numpy())
        self.popularity = popularity / max(float(popularity.max()), 1.0)
        documents = [" ".join((_text(row.track_name), _text(row.artist_name),
                               _text(row.album_name), _text(row.tag_list)))
                     for row in tracks.itertuples(index=False)]
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2),
            min_df=int(config.get("text_min_df", 2)),
            max_features=int(config.get("text_max_features", 120000)),
            sublinear_tf=True, strip_accents="unicode")
        self.entities = self.vectorizer.fit_transform(documents)
        transitions = defaultdict(Counter)
        for sequence in sequences:
            for left, right in zip(sequence, sequence[1:]):
                transitions[left][right] += 1
        self.transitions = {key: dict(value) for key, value in transitions.items()}
        self.semantic_candidates = int(config.get("semantic_candidates", 80))
        self.query_recall = query_recall
        self.query_vectors = query_vectors or {}
        self.query_candidates = int(config.get("query_embedding_candidates", 100))
        self.query_dimension = (query_recall._vectors.shape[1]
                                if query_recall is not None else 0)
        self.item_neighbors = item_neighbors or {}
        self.user_candidates = user_candidates or {}
        self.sequence_recall = sequence_recall
        self.sequence_vectors = sequence_vectors or {}
        self.sequence_dimension = (sequence_recall._vectors.shape[1]
                                   if sequence_recall is not None else 0)
        self.recall_candidates = int(config.get("channel_candidates", 100))
        self.enabled = set(config.get("recall_channels", []))
        release = pd.to_datetime(tracks.release_date, errors="coerce")
        self.release_days = release.to_numpy(dtype="datetime64[D]")
        self.release_order = np.argsort(self.release_days)[::-1]
        self.content_neighbors = {}
        self.new_candidates = {}
        self.weights = config.get("feature_weights", {
            "semantic": 0.5, "transition": 2.0, "artist": 1.0,
            "tag": 0.25, "popularity": 0.1, "query_embedding": 0.0,
        })

    def rank_many(self, requests, k=20):
        result = []
        batch_size = 128
        for start in range(0, len(requests), batch_size):
            batch = requests[start:start + batch_size]
            queries = self.vectorizer.transform([item[0] for item in batch])
            similarities = (queries @ self.entities.T).toarray()
            if self.query_recall is not None:
                dense_queries = np.asarray([
                    self.query_vectors.get(item[2], np.zeros(self.query_dimension,
                                                             dtype=np.float32))
                    for item in batch], dtype=np.float32)
                query_results = self.query_recall.recall_many(
                    dense_queries, [item[1] for item in batch],
                    self.query_candidates, block_size=batch_size)
            else:
                query_results = [[] for _ in batch]
            if self.sequence_recall is not None:
                sequence_queries = []
                for _, history, _, _, _ in batch:
                    vectors = [self.sequence_vectors[value] for value in history[-5:]
                               if value in self.sequence_vectors]
                    sequence_queries.append(np.mean(vectors, axis=0) if vectors else
                                            np.zeros(self.sequence_dimension, dtype=np.float32))
                sequence_results = self.sequence_recall.recall_many(
                    np.asarray(sequence_queries), [item[1] for item in batch],
                    self.recall_candidates, block_size=batch_size)
            else:
                sequence_results = [[] for _ in batch]
            for row, (_, history, _, user_id, request_date), query_items, sequence_items in zip(
                    similarities, batch, query_results, sequence_results):
                count = min(self.semantic_candidates, len(row))
                candidates = set(np.argpartition(row, -count)[-count:])
                query_scores = {self.index[item.item]: item.score
                                for item in query_items if item.item in self.index}
                candidates.update(query_scores)
                channel_scores = {"item_cf": {}, "content_i2i": {},
                                  "user_cf": {}, "sequence_embedding": {}, "new": {}}
                if "item_cf" in self.enabled:
                    for trigger in history[-5:]:
                        for item, value in self.item_neighbors.get(trigger, ()):
                            position = self.index.get(item)
                            if position is not None:
                                channel_scores["item_cf"][position] = max(
                                    channel_scores["item_cf"].get(position, 0.0), value)
                if "content_i2i" in self.enabled:
                    for trigger in history[-5:]:
                        trigger_position = self.index.get(trigger)
                        if trigger_position is None:
                            continue
                        if trigger_position not in self.content_neighbors:
                            content = (self.entities[trigger_position]
                                       @ self.entities.T).toarray()[0]
                            count = min(self.recall_candidates, len(content))
                            positions = np.argpartition(content, -count)[-count:]
                            self.content_neighbors[trigger_position] = [
                                (int(position), float(content[position]))
                                for position in positions if position != trigger_position]
                        for position, value in self.content_neighbors[trigger_position]:
                            channel_scores["content_i2i"][position] = max(
                                channel_scores["content_i2i"].get(position, 0.0), value)
                if "user_cf" in self.enabled:
                    for item, value in self.user_candidates.get(user_id, ()):
                        position = self.index.get(item)
                        if position is not None:
                            channel_scores["user_cf"][position] = float(value)
                if "sequence_embedding" in self.enabled:
                    channel_scores["sequence_embedding"] = {
                        self.index[item.item]: item.score for item in sequence_items
                        if item.item in self.index}
                if "new" in self.enabled:
                    cutoff = np.datetime64(request_date, "D")
                    if request_date not in self.new_candidates:
                        available = []
                        for position in self.release_order:
                            if (not np.isnat(self.release_days[position])
                                    and self.release_days[position] <= cutoff):
                                available.append(position)
                                if len(available) == self.recall_candidates:
                                    break
                        self.new_candidates[request_date] = available
                    available = self.new_candidates[request_date]
                    if available:
                        oldest = self.release_days[available[-1]].astype("int64")
                        newest = self.release_days[available[0]].astype("int64")
                        span = max(1, newest - oldest)
                        channel_scores["new"] = {
                            int(position): float((self.release_days[position].astype("int64")
                                                  - oldest) / span)
                            for position in available}
                for values in channel_scores.values():
                    candidates.update(values)
                transition_scores = {}
                for offset, trigger in enumerate(history[-5:]):
                    recency = 1 + len(history[-5:]) - offset
                    for item, frequency in self.transitions.get(trigger, {}).items():
                        position = self.index.get(item)
                        if position is not None:
                            transition_scores[position] = max(
                                transition_scores.get(position, 0.0),
                                math.log1p(frequency) / recency)
                candidates.update(transition_scores)
                candidates.update(self.hot[:100])
                history_positions = [self.index[value] for value in history
                                     if value in self.index]
                excluded = set(history_positions)
                artist_counts = Counter(self.artists[value]
                                        for value in history_positions)
                tag_counts = Counter(tag for value in history_positions
                                     for tag in self.tags[value])
                tag_total = max(1, sum(tag_counts.values()))
                history_total = max(1, len(history_positions))

                def score(position):
                    artist = artist_counts.get(self.artists[position], 0) / history_total
                    tag = sum(tag_counts.get(value, 0)
                              for value in self.tags[position]) / tag_total
                    return (self.weights["semantic"] * row[position]
                            + self.weights["transition"]
                            * transition_scores.get(position, 0.0)
                            + self.weights["artist"] * artist
                            + self.weights["tag"] * tag
                            + self.weights["popularity"]
                            * self.popularity[position]
                            + self.weights.get("query_embedding", 0.0)
                            * query_scores.get(position, 0.0)
                            + sum(self.weights.get(channel, 0.0) * values.get(position, 0.0)
                                  for channel, values in channel_scores.items()))

                ranked = [position for position in sorted(
                    candidates, key=lambda value: (-score(value), self.ids[value]))
                    if position not in excluded][:k]
                if len(ranked) < k:
                    ranked.extend(position for position in self.hot
                                  if position not in excluded and position not in ranked)
                result.append([self.ids[position] for position in ranked[:k]])
        return result


def run(config, evaluator_path, output):
    evaluator = Path(evaluator_path).resolve()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    truth_path = evaluator / "exp/ground_truth/devset.json"
    baseline_path = evaluator / "exp/inference/devset/popularity.json"
    baseline_score_path = evaluator / "exp/scores/devset/popularity.json"
    for path in (truth_path, baseline_path, baseline_score_path):
        if not path.is_file():
            raise ValueError(f"missing official evaluator artifact: {path}")
    root = load_openrec(config["openrec_algorithm"])
    from algorithm.recall.hot import Hot
    from algorithm.recall.item_cf_i2i import ItemBasedI2I
    from algorithm.recall.query_embedding import QueryEmbeddingRecall

    train = pd.read_parquet(config["train_events"], columns=[
        "session_id", "user_id", "conversations", "session_date"
    ])
    test = pd.read_parquet(config["test_events"], columns=[
        "session_id", "user_id", "user_profile", "conversation_goal",
        "conversations", "session_date"
    ])
    catalog = pd.read_parquet(config["tracks"])
    valid = set(catalog.track_id.astype(str))
    rows = []
    for session in train.itertuples(index=False):
        for message in session.conversations:
            if message["role"] == "music":
                rows.append((str(session.session_id), str(session.user_id), str(message["content"]),
                             pd.Timestamp(session.session_date).timestamp(),
                             int(message["turn_number"])))
    events = pd.DataFrame(rows, columns=["session_id", "user_id", "item_id", "time", "turn"])
    events["id"] = events.session_id + ":" + events.turn.astype(str)
    events["type"] = "click"
    events["value"] = 1.0
    session_events = events.assign(user_id=events.session_id)
    hot = [value.item for value in Hot(events=session_events, recall_size=len(valid)).recall()]
    neighbors = ItemBasedI2I(events=session_events, recall_size=20, cut_size=int(
        config.get("neighbor_size", 50)
    )).dump_i2i(cut_size=int(config.get("neighbor_size", 50)))
    truth = json.loads(truth_path.read_text())
    train_sequences = []
    for session in train.itertuples(index=False):
        music = {int(message["turn_number"]): str(message["content"])
                 for message in session.conversations if message["role"] == "music"}
        train_sequences.append([music[turn] for turn in sorted(music)])
    query_recall = None
    query_vectors = None
    if config.get("query_embedding_recall", False):
        track_dir = Path(config["track_embedding_dir"])
        query_dir = Path(config["query_embedding_dir"])
        track_ids = np.load(track_dir / "track_ids.npy", allow_pickle=True)
        track_vectors = np.load(track_dir / "embeddings.npy", mmap_mode="r")
        query_meta = pd.read_parquet(query_dir / "query_meta.parquet",
                                     columns=["session_id", "turn_number"])
        query_matrix = np.load(query_dir / "query_embeddings.npy", mmap_mode="r")
        if len(query_meta) != len(query_matrix):
            raise ValueError("query embedding metadata and vectors are not aligned")
        query_vectors = {
            (str(row.session_id), int(row.turn_number)): query_matrix[position]
            for position, row in enumerate(query_meta.itertuples(index=False))
        }
        query_recall = QueryEmbeddingRecall(
            track_ids, track_vectors,
            recall_size=int(config.get("query_embedding_candidates", 100)),
            device=config.get("query_embedding_device", "cpu"))
    enabled_channels = set(config.get("recall_channels", []))
    user_candidates = None
    if "user_cf" in enabled_channels:
        from algorithm.recall.user_cf_u2i import UserBasedCF
        user_candidates = UserBasedCF(
            events=events, recall_size=int(config.get("channel_candidates", 100)),
            neighbour_size=int(config.get("user_neighbor_size", 50))).dump_user_recall()
    sequence_recall = None
    sequence_vectors = None
    if "sequence_embedding" in enabled_channels:
        from algorithm.recall.item_seq_emb import EventEmbedding
        embedding = EventEmbedding(
            events=session_events, recall_size=int(config.get("channel_candidates", 100)))
        embedding.train(embedding.gen_sentences(),
                        vector_size=int(config.get("sequence_embedding_dimension", 64)))
        dumped = embedding.dump_vectors(
            vector_size=int(config.get("sequence_embedding_dimension", 64)))
        if dumped:
            sequence_vectors = {str(item): np.asarray(vector, dtype=np.float32)
                                for item, vector in dumped}
            sequence_recall = QueryEmbeddingRecall(
                list(sequence_vectors), np.asarray(list(sequence_vectors.values())),
                recall_size=int(config.get("channel_candidates", 100)),
                device=config.get("sequence_embedding_device", "cpu"))
    feature_ranker = (_SessionEntityRanker(
        catalog, train_sequences, hot, config, query_recall, query_vectors,
        neighbors, user_candidates, sequence_recall, sequence_vectors)
                      if config.get("session_entity_features", False) else None)
    requests = []
    request_rows = []
    predictions = []
    rank_diagnostics = None
    if config.get("lightgbm_rank", False):
        from .lightgbm_rank import train_and_predict
        output.mkdir(parents=True, exist_ok=True)
        predictions, rank_diagnostics = train_and_predict(
            config, catalog, hot, config["query_embedding_dir"], output, train, test)
        # The generic rank adapter has already produced every test request.
        test = test.iloc[0:0]
    for session in test.itertuples(index=False):
        music = {int(m["turn_number"]): str(m["content"])
                 for m in session.conversations if m["role"] == "music"}
        user_text = {int(m["turn_number"]): str(m["content"])
                     for m in session.conversations if m["role"] == "user"}
        goal = _text(session.conversation_goal.get("listener_goal", ""))
        culture = _text(session.user_profile.get("preferred_musical_culture", ""))
        for turn in range(1, 9):
            history = [music[earlier] for earlier in range(1, turn)]
            if feature_ranker is None:
                ids = _rank(history, neighbors, hot)
            else:
                query = " ".join([user_text.get(earlier, "")
                                  for earlier in range(max(1, turn - 2), turn + 1)]
                                 + [goal, culture])
                requests.append((query, history, (str(session.session_id), turn),
                                 str(session.user_id), str(session.session_date)))
                request_rows.append((str(session.session_id), str(session.user_id), turn))
                continue
            if len(ids) != 20 or not set(ids).issubset(valid):
                raise ValueError("OpenRec produced invalid catalog recommendations")
            predictions.append({
                "session_id": str(session.session_id), "user_id": str(session.user_id),
                "turn_number": turn, "predicted_track_ids": ids,
                "predicted_response": "",
            })
    if feature_ranker is not None:
        for (session_id, user_id, turn), ids in zip(
                request_rows, feature_ranker.rank_many(requests)):
            if len(ids) != 20 or not set(ids).issubset(valid):
                raise ValueError("OpenRec produced invalid catalog recommendations")
            predictions.append({
                "session_id": session_id, "user_id": user_id,
                "turn_number": turn, "predicted_track_ids": ids,
                "predicted_response": "",
            })
    published = json.loads(baseline_score_path.read_text())
    parity = _score(json.loads(baseline_path.read_text()), truth, len(valid), evaluator)
    for metric in ("ndcg@1", "ndcg@10", "ndcg@20", "catalog_diversity", "lexical_diversity"):
        if abs(parity[metric] - published[metric]) > 1e-12:
            raise ValueError(f"official evaluator parity failed for {metric}")
    scores = _score(predictions, truth, len(valid), evaluator)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "predictions.json", predictions)
    write_json(output / "scores.json", scores)
    write_json(output / "manifest.json", {
        "schema": 1, "task": "music_crs_official_devset", "config": config,
        "evaluator_commit": git_state(evaluator)["commit"],
        "openrec_commit": git_state(root)["commit"],
        "source_sha256": digest(__file__),
        "train_sha256": digest(config["train_events"]),
        "test_sha256": digest(config["test_events"]),
        "catalog_sha256": digest(config["tracks"]),
        "ground_truth_sha256": digest(truth_path),
        "predictions_sha256": digest(output / "predictions.json"),
        "scores_sha256": digest(output / "scores.json"),
        "official_popularity_parity": parity,
        "response_policy": "empty text; recommendation metrics only",
        "feature_roles": (["session", "context", "interaction", "candidate"]
                          if feature_ranker is not None or rank_diagnostics else []),
        "recall_channels": (rank_diagnostics["channels"] if rank_diagnostics else
                            sorted(set(["hot", "query_embedding"]
                                       if query_recall is not None else ["hot"])
                                   | enabled_channels)),
        "rank_model": ("openrec_lightgbm_%s" % rank_diagnostics["selected_objective"]
                       if rank_diagnostics else None),
        "rank_diagnostics": rank_diagnostics,
    })
    return scores
