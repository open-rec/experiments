import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import validate
from .evaluation import evaluate
from .features import materialize
from .provenance import digest, git_state, write_json
from .semantic import load_embeddings


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


def logged_history_inputs(frame, config, item_vectors, candidate_indices):
    """Build point-in-time click histories with the same visibility policy as globals."""
    max_history = int(config.get("max_history", 50))
    interval = int(config["update_interval_ms"])
    delay = int(config["feedback_delay_ms"])
    train_end = timestamp(config["validation_start"])
    cutoffs = (frame.timestamp.to_numpy(dtype=np.int64) // interval) * interval - delay
    if config["evaluation_feedback"] == "frozen":
        cutoffs = np.minimum(cutoffs, train_end - delay)
    elif config["evaluation_feedback"] != "delayed_replay":
        raise ValueError("evaluation_feedback must be frozen or delayed_replay")

    click_mask = frame.policy.ne("random") & frame.label.eq(1)
    click_positions = np.flatnonzero(click_mask.to_numpy())
    order = click_positions[np.argsort(frame.timestamp.to_numpy()[click_positions], kind="stable")]
    click_times = frame.timestamp.to_numpy()[order]
    click_users = frame.user_id.to_numpy()[order]
    click_items = candidate_indices[order]
    users = frame.user_id.astype(str).unique()
    histories = defaultdict(lambda: deque(maxlen=max_history))
    vectors, masks, keys = [], [], {}
    pointer = 0
    for cutoff in np.unique(cutoffs):
        while pointer < len(order) and click_times[pointer] < cutoff:
            histories[str(click_users[pointer])].append(int(click_items[pointer]))
            pointer += 1
        for user in users:
            values = list(histories[user])
            vector = np.zeros((max_history, item_vectors.shape[1]), dtype=np.float32)
            mask = np.ones(max_history, dtype=bool)
            if values:
                vector[-len(values):] = item_vectors[values]
                mask[-len(values):] = False
            keys[(int(cutoff), user)] = len(vectors)
            vectors.append(vector)
            masks.append(mask)
    history_indices = np.fromiter(
        (keys[(int(cutoff), str(user))] for cutoff, user in zip(cutoffs, frame.user_id)),
        dtype=np.int64,
        count=len(frame),
    )
    return np.stack(vectors), np.stack(masks), history_indices


def apply_semantic_title_fallback(items, semantic_present, candidate_indices):
    """Use title hash only for rows whose semantic source text is absent."""
    present = np.asarray(semantic_present, dtype=bool)
    indices = np.asarray(candidate_indices, dtype=np.int64)
    if len(items) != len(indices) or (indices < 0).any() or (indices >= len(present)).any():
        raise ValueError("semantic fallback inputs are not row-aligned")
    result = items.copy()
    result.loc[present[indices], "title"] = ""
    return result


def run(config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    root = load_openrec(config["openrec_algorithm"])
    from algorithm.feature.feature_space import FeatureSpace
    from algorithm.rank.lr import LRModel
    from algorithm.rank.fm import FMModel
    from algorithm.rank.transformer import CandidateAwareTransformerModel

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(config.get("threads", 2)))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    torch.use_deterministic_algorithms(device.type == "cpu")
    started = time.monotonic()
    output.mkdir(parents=True)
    experiment_root = Path(__file__).resolve().parents[1]
    run_manifest = {
        "status": "running", "config": config, "data": manifest,
        "repositories": {"experiments": git_state(experiment_root), "rec-algorithm": git_state(root)},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "cpu_count": os.cpu_count(), "device": str(device),
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
        elif model_type in {"lr", "fm", "transformer"}:
            if int(config["epochs"]) < 1 or int(config["batch_size"]) < 1 or float(config["learning_rate"]) <= 0:
                raise ValueError("epochs, batch size and learning rate must be positive")
            users, items = materialize(frame, config)
            selection = {"user": ["user.event_count", "user.event_click_rate"],
                         "candidate": ["item.scene", "item.event_count", "item.event_click_rate"]}
            if "category" in items:
                selection["candidate"].append("item.category")
            if config.get("content_features"):
                selection["candidate"] = [
                    "item.title", "item.category", "item.subcategory",
                    "item.tags", "item.content_age_hours", "item.scene",
                    "item.event_count", "item.event_click_rate",
                ]
            if config.get("feature_selection"):
                selection = config["feature_selection"]
            semantic_matrix = semantic_present = candidate_indices = semantic_manifest = None
            semantic_ids = None
            if config.get("semantic_embeddings"):
                semantic_ids, semantic_matrix, semantic_present, semantic_manifest = load_embeddings(
                    config["semantic_embeddings"]
                )
                lookup = pd.Series(np.arange(len(semantic_ids), dtype=np.int64), index=semantic_ids)
                mapped = frame.item_id.map(lookup)
                if mapped.isna().any():
                    raise ValueError("candidate article is missing a semantic embedding")
                candidate_indices = mapped.to_numpy(dtype=np.int64)
                if config.get("semantic_fallback_title_hash"):
                    items = apply_semantic_title_fallback(
                        items, semantic_present, candidate_indices
                    )
                else:
                    selection["candidate"] = [value for value in selection["candidate"]
                                              if value != "item.title"]
                run_manifest["semantic_embeddings"] = semantic_manifest
                run_manifest["semantic_fallback_title_hash"] = bool(
                    config.get("semantic_fallback_title_hash")
                )
            if model_type == "transformer" and config.get("structured_content_vectors"):
                content_selection = {"user": ["user.event_count"], "candidate": [
                    "item.category", "item.subcategory", "item.tags",
                ]}
                content_space = FeatureSpace.for_model("fm", selection=content_selection)
                content_space.fit(users=users[masks["train"]], items=items[masks["train"]])
                profiles = items[["id", "category", "subcategory", "tags"]].drop_duplicates("id")
                semantic_ids = profiles.id.astype(str).to_numpy()
                semantic_matrix = content_space.transform_items(profiles).astype(np.float32)
                semantic_present = np.ones(len(semantic_ids), dtype=bool)
                lookup = pd.Series(np.arange(len(semantic_ids), dtype=np.int64), index=semantic_ids)
                mapped = frame.item_id.map(lookup)
                if mapped.isna().any():
                    raise ValueError("candidate item is missing structured content")
                candidate_indices = mapped.to_numpy(dtype=np.int64)
                run_manifest["item_representation"] = {
                    "type": "openrec_structured_content",
                    "features": content_selection["candidate"],
                    "dimension": int(semantic_matrix.shape[1]),
                }
            if model_type == "transformer" and semantic_matrix is None:
                raise ValueError("Transformer requires semantic or structured item vectors")
            feature_model = "fm" if model_type == "transformer" else model_type
            space = FeatureSpace.for_model(feature_model, selection=selection)
            space.fit(users[masks["train"]], items[masks["train"]])
            global_features = np.concatenate(
                [space.transform_users(users), space.transform_items(items)], axis=1
            ).astype("float32")
            features = global_features
            if model_type in {"lr", "fm"} and semantic_matrix is not None:
                features = np.concatenate(
                    [global_features, semantic_matrix[candidate_indices]], axis=1
                ).astype("float32")
            x = torch.from_numpy(features)
            y = torch.tensor(frame.label.to_numpy(), dtype=torch.float32).reshape(-1, 1)
            if model_type == "lr":
                model = LRModel(features.shape[1])
            elif model_type == "fm":
                model = FMModel(features.shape[1], config.get("factor_dim", 8))
            else:
                model = CandidateAwareTransformerModel(
                    global_dim=global_features.shape[1],
                    semantic_dim=semantic_matrix.shape[1],
                    model_dim=int(config.get("model_dim", 128)),
                    num_heads=int(config.get("num_heads", 4)),
                    num_layers=int(config.get("num_layers", 2)),
                    max_history=int(config.get("max_history", 50)),
                    dropout=float(config.get("dropout", 0.1)),
                )
            model = model.to(device)
            train_indices = torch.from_numpy(np.flatnonzero(masks["train"].to_numpy()))
            optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
            criterion = torch.nn.BCELoss()
            best = float("inf")
            batch_size = int(config["batch_size"])

            history_vectors = history_masks = history_indices = semantic_tensor = None
            if model_type == "transformer":
                if config.get("history_files"):
                    history_vectors, history_masks, history_indices = history_inputs(
                        frame, config, semantic_matrix, semantic_ids
                    )
                    run_manifest["history_inputs"] = [
                        {"path": str(path), "sha256": digest(Path(path))}
                        for path in config["history_files"]
                    ]
                else:
                    history_vectors, history_masks, history_indices = logged_history_inputs(
                        frame, config, semantic_matrix, candidate_indices
                    )
                    run_manifest["history_inputs"] = [{
                        "path": str(source), "sha256": manifest["output_sha256"],
                        "policy": "standard clicks visible at feature cutoff",
                    }]
                semantic_tensor = torch.from_numpy(semantic_matrix).to(device)
                history_vectors = torch.from_numpy(history_vectors).to(device)
                history_masks = torch.from_numpy(history_masks).to(device)
                history_indices = torch.from_numpy(history_indices).to(device)
                candidate_indices = torch.from_numpy(candidate_indices).to(device)
                run_manifest["architecture"] = model.architecture()
                run_manifest["history_rows"] = int(len(history_vectors))

            x = x.to(device)
            y = y.to(device)
            train_indices = train_indices.to(device)

            def predict(indices):
                model.eval()
                with torch.no_grad():
                    chunks = []
                    for index in indices.split(batch_size):
                        if model_type == "transformer":
                            score = model(
                                x[index], semantic_tensor[candidate_indices[index]],
                                history_vectors[history_indices[index]],
                                history_masks[history_indices[index]],
                            )
                        else:
                            score = model(x[index])
                        chunks.append(score.detach().cpu().numpy().ravel())
                    return np.concatenate(chunks)

            validation_indices = torch.from_numpy(
                np.flatnonzero(masks["validation"].to_numpy())
            ).to(device)
            for epoch in range(int(config["epochs"])):
                model.train()
                order = train_indices[torch.randperm(len(train_indices), device=device,
                                                     generator=torch.Generator(device=device).manual_seed(seed + epoch))]
                for index in order.split(batch_size):
                    optimizer.zero_grad()
                    if model_type == "transformer":
                        score = model(
                            x[index], semantic_tensor[candidate_indices[index]],
                            history_vectors[history_indices[index]],
                            history_masks[history_indices[index]],
                        )
                    else:
                        score = model(x[index])
                    loss = criterion(score, y[index])
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    loss.backward()
                    optimizer.step()
                metrics = evaluate(frame[masks["validation"]], predict(validation_indices), config["dataset"])
                epoch_history.append({"epoch": epoch + 1, "validation": metrics})
                if metrics["logloss"] < best:
                    best = metrics["logloss"]
                    torch.save(model.state_dict(), output / f"{model_type}.pth")
                    run_manifest["selected_epoch"] = epoch + 1
            model.load_state_dict(torch.load(output / f"{model_type}.pth", map_location=device, weights_only=True))
            space.save(output / f"{model_type}.features.json")
            predictions = predict(torch.arange(len(frame), device=device))
            run_manifest["feature_selection"] = selection
            run_manifest["feature_dimension"] = features.shape[1]
        else:
            raise ValueError("model must be popularity, lr, fm or transformer")
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
