"""Align the organizer's example profiles to the exact official client set."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-profiles", type=Path, required=True)
    parser.add_argument("--relevant-clients", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    clients = np.load(args.relevant_clients)
    baseline_clients = np.load(args.baseline_profiles / "client_ids.npy")
    baseline_embeddings = np.load(
        args.baseline_profiles / "embeddings.npy", mmap_mode="r"
    )
    if len(set(map(int, baseline_clients))) != len(baseline_clients):
        raise ValueError("official baseline has duplicate client IDs")
    if baseline_embeddings.shape[0] != len(baseline_clients):
        raise ValueError("official baseline profile rows do not match client IDs")
    index = {int(client_id): row for row, client_id in enumerate(baseline_clients)}
    args.output.mkdir(parents=True)
    aligned = np.lib.format.open_memmap(
        args.output / "embeddings.npy", mode="w+", dtype=np.float16,
        shape=(len(clients), baseline_embeddings.shape[1]),
    )
    missing = 0
    for row, client_id in enumerate(clients):
        source = index.get(int(client_id))
        if source is None:
            aligned[row] = 0
            missing += 1
        else:
            aligned[row] = baseline_embeddings[source]
    aligned.flush()
    np.save(args.output / "client_ids.npy", clients)
    print({"clients": len(clients), "missing_zero_filled": missing,
           "dimensions": baseline_embeddings.shape[1]})


if __name__ == "__main__":
    main()
