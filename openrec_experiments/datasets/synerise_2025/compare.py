"""Compare Synerise profiles on the organizer's local task metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from ...provenance import digest, git_state, write_json


TASKS = (
    "churn", "propensity_category", "propensity_sku",
    "hidden1", "hidden2", "hidden3",
)


def _load_scores(paths: list[Path]) -> dict[str, float]:
    values = {}
    for path in paths:
        source = json.loads(path.read_text())
        for task, value in source.items():
            if task == "placeholder":
                continue
            if task in values:
                raise ValueError(f"duplicate official task score: {task}")
            values[task] = value
    missing = set(TASKS) - values.keys()
    if missing:
        raise ValueError(f"missing official task scores: {sorted(missing)}")
    result = {task: float(values[task]) for task in TASKS}
    if any(not math.isfinite(score) for score in result.values()):
        raise ValueError("non-finite official task score")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openrec-scores", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline-scores", type=Path, nargs="+", required=True)
    parser.add_argument("--openrec-profiles", type=Path, required=True)
    parser.add_argument("--baseline-profiles", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    openrec = _load_scores(args.openrec_scores)
    baseline = _load_scores(args.baseline_scores)
    args.output.mkdir(parents=True)
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("task", "openrec", "official_example_baseline", "delta"))
        for task in TASKS:
            writer.writerow((task, openrec[task], baseline[task],
                             round(openrec[task] - baseline[task], 4)))
        writer.writerow(("sum_of_task_scores", round(sum(openrec.values()), 4),
                         round(sum(baseline.values()), 4),
                         round(sum(openrec.values()) - sum(baseline.values()), 4)))
    write_json(args.output / "manifest.json", {
        "schema": 1,
        "protocol": "official Synerise local challenge_dataset, six tasks",
        "evaluator_commit": git_state(args.evaluator)["commit"],
        "comparison_source_sha256": digest(__file__),
        "openrec_scores_sha256": {str(path): digest(path) for path in args.openrec_scores},
        "official_example_baseline_scores_sha256": {
            str(path): digest(path) for path in args.baseline_scores},
        "openrec_profiles_sha256": {
            name: digest(args.openrec_profiles / name)
            for name in ("client_ids.npy", "embeddings.npy")},
        "official_example_baseline_profiles_sha256": {
            name: digest(args.baseline_profiles / name)
            for name in ("client_ids.npy", "embeddings.npy")},
        "note": "Published final leaderboard uses different hidden target windows; no ranking inference.",
    })


if __name__ == "__main__":
    main()
