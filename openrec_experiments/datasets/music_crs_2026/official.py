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

    def __init__(self, tracks, sequences, hot, config):
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
        self.weights = config.get("feature_weights", {
            "semantic": 0.5, "transition": 2.0, "artist": 1.0,
            "tag": 0.25, "popularity": 0.1,
        })

    def rank_many(self, requests, k=20):
        result = []
        batch_size = 128
        for start in range(0, len(requests), batch_size):
            batch = requests[start:start + batch_size]
            queries = self.vectorizer.transform([item[0] for item in batch])
            similarities = (queries @ self.entities.T).toarray()
            for row, (_, history) in zip(similarities, batch):
                count = min(self.semantic_candidates, len(row))
                candidates = set(np.argpartition(row, -count)[-count:])
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
                            * self.popularity[position])

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

    train = pd.read_parquet(config["train_events"], columns=[
        "session_id", "conversations", "session_date"
    ])
    test = pd.read_parquet(config["test_events"], columns=[
        "session_id", "user_id", "user_profile", "conversation_goal",
        "conversations"
    ])
    catalog = pd.read_parquet(config["tracks"])
    valid = set(catalog.track_id.astype(str))
    rows = []
    for session in train.itertuples(index=False):
        for message in session.conversations:
            if message["role"] == "music":
                rows.append((str(session.session_id), str(message["content"]),
                             pd.Timestamp(session.session_date).timestamp(),
                             int(message["turn_number"])))
    events = pd.DataFrame(rows, columns=["user_id", "item_id", "time", "turn"])
    events["id"] = events.user_id + ":" + events.turn.astype(str)
    events["type"] = "click"
    events["value"] = 1.0
    hot = [value.item for value in Hot(events=events, recall_size=len(valid)).recall()]
    neighbors = ItemBasedI2I(events=events, recall_size=20, cut_size=int(
        config.get("neighbor_size", 50)
    )).dump_i2i(cut_size=int(config.get("neighbor_size", 50)))
    truth = json.loads(truth_path.read_text())
    train_sequences = []
    for session in train.itertuples(index=False):
        music = {int(message["turn_number"]): str(message["content"])
                 for message in session.conversations if message["role"] == "music"}
        train_sequences.append([music[turn] for turn in sorted(music)])
    feature_ranker = (_SessionEntityRanker(catalog, train_sequences, hot, config)
                      if config.get("session_entity_features", False) else None)
    requests = []
    request_rows = []
    predictions = []
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
                requests.append((query, history))
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
    output.mkdir(parents=True)
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
                          if feature_ranker is not None else []),
    })
    return scores
