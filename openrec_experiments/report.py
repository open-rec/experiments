"""Export auditable result rows without pooling incompatible protocols."""
import csv
import hashlib
import json
from pathlib import Path

from .provenance import digest


def flatten(value, prefix=""):
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            yield from flatten(item, name)
        else:
            yield name, item


def report(run_paths, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for path in map(Path, run_paths):
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"incomplete run: {path}")
        for name, expected in manifest["artifacts"].items():
            if digest(path / name) != expected:
                raise ValueError(f"artifact checksum mismatch: {path / name}")
        config = manifest["config"]
        protocol = {key: config[key] for key in ["dataset", "train_start", "validation_start",
                    "test_start", "test_end", "update_interval_ms", "feedback_delay_ms", "evaluation_feedback"]}
        protocol["data_sha256"] = manifest["data"]["output_sha256"]
        protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        row = {"run": str(path), "dataset": config["dataset"], "model": config["model"],
               "seed": config["seed"], "protocol_sha256": protocol_id,
               "wall_seconds": manifest["wall_seconds"]}
        row.update(flatten(json.loads((path / "metrics.json").read_text())))
        rows.append(row)
    if not rows:
        raise ValueError("at least one completed run is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)
