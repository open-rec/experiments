"""Ablate champion recall routes while reusing one frozen XGB score pass."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import yaml


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_metrics(frame, support, groups=8000):
    selected = frame.filter(support).sort(
        ["session_id", "turn_number", "score", "track_id"],
        descending=[False, False, True, False],
    ).with_columns(
        (pl.int_range(pl.len()).over("session_id", "turn_number") + 1).alias("_rank")
    )
    hits = selected.filter(
        (pl.col("track_id") == pl.col("gt_track_id")) & (pl.col("_rank") <= 200)
    )
    ranks = hits["_rank"].to_numpy()
    result = {}
    for cutoff in (1, 10, 20, 50, 100, 200):
        within = ranks[ranks <= cutoff]
        result["recall@%d" % cutoff] = float(len(within) / groups)
        result["ndcg@%d" % cutoff] = float(
            np.sum(1.0 / np.log2(within.astype(np.float64) + 1.0)) / groups
        )
    return result


def raw_recall(path):
    frame = pl.read_parquet(path)
    turn_column = "gt_turn_number" if "gt_turn_number" in frame.columns else "turn"
    ranks = {}
    for row in frame.iter_rows(named=True):
        ids = [str(value) for value in row["track_ids"]]
        gt = str(row["gt_track_id"])
        rank = ids.index(gt) + 1 if gt in ids else 0
        ranks[(str(row["session_id"]), int(row[turn_column]))] = rank
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.workspace
    config_path = root / "src/reranker_oof/configs/blind_no_filter/dataset.yaml"
    config = yaml.safe_load(config_path.read_text())
    channels = list(config["cgs"])
    scored_path = root / "models/clean_dev_champion/dev_scored.parquet"
    chunk_root = root / "models/reranker_oof/datasets/blind_no_filter/holdout"

    rank_columns = ["rank_%s" % channel for channel in channels]
    features = pl.concat([
        pl.read_parquet(path, columns=[
            "session_id", "turn_number", "track_id", "gt_track_id", *rank_columns
        ]) for path in sorted(chunk_root.glob("chunk_*.parquet"))
    ])
    scored = pl.read_parquet(scored_path).join(
        features, on=["session_id", "turn_number", "track_id"], how="inner"
    )
    if scored.height != features.height:
        raise RuntimeError("frozen score rows do not align with the assembled holdout pool")
    supports = [pl.col(column).is_not_null() for column in rank_columns]
    all_support = pl.any_horizontal(supports)
    baseline = rank_metrics(scored, all_support)

    raw_ranks = {}
    raw_stats = {}
    for channel in channels:
        path = root / "models/CG_crossvalidation" / channel / "datasets/holdout_candidates.parquet"
        ranks = raw_recall(path)
        raw_ranks[channel] = ranks
    reference_keys = sorted(set.union(*[set(value) for value in raw_ranks.values()]))
    if len(reference_keys) != 8000:
        raise RuntimeError("raw candidate union does not cover 8000 requests")
    for channel in channels:
        values = np.asarray([raw_ranks[channel].get(key, 0) for key in reference_keys],
                            dtype=np.int16)
        raw_stats[channel] = {
            "raw_request_coverage": float(len(raw_ranks[channel]) / len(reference_keys)),
            **{"raw_recall@%d" % cutoff:
               float(np.mean((values > 0) & (values <= cutoff)))
               for cutoff in (20, 50, 100, 200)},
        }
    hit_matrix = np.column_stack([
        np.asarray([raw_ranks[channel].get(key, 0) > 0 for key in reference_keys])
        for channel in channels
    ])

    rows = []
    for index, channel in enumerate(channels):
        single = rank_metrics(scored, supports[index])
        remaining = [value for position, value in enumerate(supports) if position != index]
        leave_one_out = rank_metrics(scored, pl.any_horizontal(remaining))
        row = {
            "channel": channel,
            **raw_stats[channel],
            "exclusive_gt_recall@200": float(np.mean(hit_matrix[:, index]
                                                      & (hit_matrix.sum(axis=1) == 1))),
            "fused_pool_candidate_share": float(
                scored.select(supports[index].mean()).item()),
            "single_ndcg@1": single["ndcg@1"],
            "single_ndcg@10": single["ndcg@10"],
            "single_ndcg@20": single["ndcg@20"],
            "single_recall@20": single["recall@20"],
            "loo_ndcg@20": leave_one_out["ndcg@20"],
            "loo_recall@20": leave_one_out["recall@20"],
            "ndcg@20_drop": baseline["ndcg@20"] - leave_one_out["ndcg@20"],
            "recall@20_drop": baseline["recall@20"] - leave_one_out["recall@20"],
        }
        rows.append(row)

    # Greedy cumulative order uses only the fixed XGB scores and candidate support.
    remaining = set(range(len(channels)))
    selected = []
    cumulative = []
    while remaining:
        best = None
        for candidate in remaining:
            trial = selected + [candidate]
            metrics = rank_metrics(scored, pl.any_horizontal([supports[value] for value in trial]))
            value = (metrics["ndcg@20"], metrics["recall@20"], channels[candidate])
            if best is None or value > best[0]:
                best = (value, candidate, metrics)
        _, chosen, metrics = best
        selected.append(chosen)
        remaining.remove(chosen)
        cumulative.append({"step": len(selected), "channel": channels[chosen], **metrics})

    # Global candidate overlap in the final fused pool.
    overlap = []
    for left in range(len(channels)):
        for right in range(left + 1, len(channels)):
            intersection = scored.select((supports[left] & supports[right]).sum()).item()
            union = scored.select((supports[left] | supports[right]).sum()).item()
            overlap.append({"left": channels[left], "right": channels[right],
                            "global_jaccard": float(intersection / max(union, 1))})

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "protocol": "single frozen XGB score pass; candidate support ablation",
        "channels": channels,
        "baseline": baseline,
        "channel_ablation": sorted(rows, key=lambda row: (-row["ndcg@20_drop"], row["channel"])),
        "greedy_cumulative": cumulative,
        "pair_overlap": overlap,
        "artifacts": {
            "dataset_config": str(config_path), "dataset_config_sha256": sha256(config_path),
            "scored_candidates": str(scored_path), "scored_candidates_sha256": sha256(scored_path),
            "scored_rows": scored.height, "requests": 8000,
        },
    }
    (args.output / "ablation.json").write_text(json.dumps(payload, indent=2) + "\n")
    with open(args.output / "channels.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (-row["ndcg@20_drop"], row["channel"])))
    with open(args.output / "cumulative.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cumulative[0]))
        writer.writeheader()
        writer.writerows(cumulative)
    print(json.dumps({"baseline": baseline, "channel_ablation": payload["channel_ablation"],
                      "greedy_cumulative": cumulative}, indent=2))


if __name__ == "__main__":
    main()
