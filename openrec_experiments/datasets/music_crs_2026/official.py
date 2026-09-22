"""Music-CRS devset inference scored with the organizer's evaluator functions."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
        "session_id", "user_id", "conversations"
    ])
    catalog = pd.read_parquet(config["tracks"], columns=["track_id"])
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
    predictions = []
    for session in test.itertuples(index=False):
        music = {int(m["turn_number"]): str(m["content"])
                 for m in session.conversations if m["role"] == "music"}
        for turn in range(1, 9):
            history = [music[earlier] for earlier in range(1, turn)]
            ids = _rank(history, neighbors, hot)
            if len(ids) != 20 or not set(ids).issubset(valid):
                raise ValueError("OpenRec produced invalid catalog recommendations")
            predictions.append({
                "session_id": str(session.session_id), "user_id": str(session.user_id),
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
    })
    return scores
