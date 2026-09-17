import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import validate
from .evaluation import evaluate
from .features import materialize
from .provenance import digest, git_state, write_json


def timestamp(value):
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError("split boundaries must include a UTC offset")
    return result.value // 1_000_000


def split(frame, config):
    train_start, validation_start, test_start, test_end = [timestamp(config[k]) for k in
        ["train_start", "validation_start", "test_start", "test_end"]]
    if not train_start < validation_start < test_start < test_end:
        raise ValueError("split boundaries must be strictly increasing")
    # Do not tune on randomly exposed data: preserve it for separate evaluation.
    masks = {
        "train": frame.timestamp.ge(train_start) & frame.timestamp.lt(validation_start) & frame.policy.ne("random"),
        "validation": frame.timestamp.ge(validation_start) & frame.timestamp.lt(test_start) & frame.policy.ne("random"),
        "test": frame.timestamp.ge(test_start) & frame.timestamp.lt(test_end),
    }
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("every split must be nonempty")
    if frame.loc[masks["train"], "label"].nunique() != 2:
        raise ValueError("training requires both label classes")
    return masks


def load_openrec(root):
    root = Path(root).resolve()
    if not (root / "algorithm/rank/lr.py").is_file():
        raise ValueError("openrec_algorithm must point to the rec-algorithm source repository")
    sys.path.insert(0, str(root))
    import algorithm.rank.lr as lr
    if not Path(lr.__file__).resolve().is_relative_to(root):
        raise ValueError("a different rec-algorithm is already imported")
    return root


def run(config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    root = load_openrec(config["openrec_algorithm"])
    from algorithm.feature.feature_space import FeatureSpace
    from algorithm.rank.lr import LRModel
    from algorithm.rank.fm import FMModel

    source = Path(config["data"])
    manifest_path = Path(str(source) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset"] != config["dataset"] or manifest["output_sha256"] != digest(source):
        raise ValueError("prepared data does not match its manifest/config")
    frame = pd.read_parquet(source).reset_index(drop=True)
    validate(frame)
    masks = split(frame, config)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get("threads", 2)))
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    output.mkdir(parents=True)
    experiment_root = Path(__file__).resolve().parents[1]
    run_manifest = {
        "status": "running", "config": config, "data": manifest,
        "repositories": {"experiments": git_state(experiment_root), "rec-algorithm": git_state(root)},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "cpu_count": os.cpu_count(), "device": "cpu",
                        "packages": {name: importlib.metadata.version(name) for name in
                                     ["numpy", "pandas", "pyarrow", "torch", "scikit-learn"]}},
        "source_files": {str(p.relative_to(experiment_root)): digest(p) for p in
                         (experiment_root / "openrec_experiments").glob("*.py")},
        "split_rows": {name: int(mask.sum()) for name, mask in masks.items()},
        "result_scope": "local temporal holdout; not an official leaderboard result",
    }
    write_json(output / "manifest.json", run_manifest)
    (output / "environment.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True))
    try:
        model_type = config["model"]
        epoch_history = []
        if model_type == "popularity":
            train = frame[masks["train"]]
            prior = float(train.label.mean())
            counts = train.groupby("item_id").label.agg(["sum", "count"])
            probability = (counts["sum"] + 10 * prior) / (counts["count"] + 10)
            predictions = frame.item_id.map(probability).fillna(prior).to_numpy()
            write_json(output / "popularity.json", {"prior": prior, "smoothing": 10,
                                                    "items": probability.to_dict()})
        elif model_type in {"lr", "fm"}:
            if int(config["epochs"]) < 1 or int(config["batch_size"]) < 1 or float(config["learning_rate"]) <= 0:
                raise ValueError("epochs, batch size and learning rate must be positive")
            users, items = materialize(frame, config)
            selection = {"user": ["user.event_count", "user.event_click_rate"],
                         "candidate": ["item.scene", "item.event_count", "item.event_click_rate"]}
            if "category" in items:
                selection["candidate"].append("item.category")
            space = FeatureSpace.for_model(model_type, selection=selection)
            space.fit(users[masks["train"]], items[masks["train"]])
            features = np.concatenate([space.transform_users(users), space.transform_items(items)], axis=1).astype("float32")
            x = torch.from_numpy(features)
            y = torch.tensor(frame.label.to_numpy(), dtype=torch.float32).reshape(-1, 1)
            model = LRModel(features.shape[1]) if model_type == "lr" else FMModel(features.shape[1], config.get("factor_dim", 8))
            train_indices = torch.from_numpy(np.flatnonzero(masks["train"].to_numpy()))
            loader = DataLoader(TensorDataset(x[train_indices], y[train_indices]),
                                batch_size=int(config["batch_size"]), shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
            optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
            criterion = torch.nn.BCELoss()
            best = float("inf")
            batch_size = int(config["batch_size"])

            def predict(values):
                model.eval()
                with torch.no_grad():
                    return np.concatenate([model(chunk).numpy().ravel() for chunk in values.split(batch_size)])

            validation_indices = torch.from_numpy(np.flatnonzero(masks["validation"].to_numpy()))
            for epoch in range(int(config["epochs"])):
                model.train()
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    loss.backward()
                    optimizer.step()
                metrics = evaluate(frame[masks["validation"]], predict(x[validation_indices]), config["dataset"])
                epoch_history.append({"epoch": epoch + 1, "validation": metrics})
                if metrics["logloss"] < best:
                    best = metrics["logloss"]
                    torch.save(model.state_dict(), output / f"{model_type}.pth")
                    run_manifest["selected_epoch"] = epoch + 1
            model.load_state_dict(torch.load(output / f"{model_type}.pth", map_location="cpu", weights_only=True))
            space.save(output / f"{model_type}.features.json")
            predictions = predict(x)
            run_manifest["feature_selection"] = selection
            run_manifest["feature_dimension"] = features.shape[1]
        else:
            raise ValueError("model must be popularity, lr or fm")
        metrics = {name: evaluate(frame[mask], predictions[mask], config["dataset"])
                   for name, mask in masks.items() if name != "train"}
        write_json(output / "metrics.json", metrics)
        write_json(output / "learning_curve.json", epoch_history)
        heldout = masks["validation"] | masks["test"]
        prediction_frame = frame.loc[heldout, ["sample_id", "group_id", "user_id", "item_id", "label", "policy", "scene"]].copy()
        prediction_frame["split"] = np.where(masks["validation"][heldout], "validation", "test")
        prediction_frame["score"] = predictions[heldout]
        prediction_frame.to_parquet(output / "predictions.parquet", index=False)
        run_manifest.update(status="complete", wall_seconds=time.monotonic() - started)
        run_manifest["artifacts"] = {p.name: digest(p) for p in output.iterdir() if p.name != "manifest.json"}
        write_json(output / "manifest.json", run_manifest)
        return metrics
    except Exception as error:
        run_manifest.update(status="failed", error=str(error), wall_seconds=time.monotonic() - started)
        write_json(output / "manifest.json", run_manifest)
        raise
