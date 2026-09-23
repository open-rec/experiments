"""Container adapter for the unmodified rec2 winning source tree."""

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile


WINNER_COMMIT = "fb17bd991d3ab3cc246f529485f7ba662749d390"
OFFICIAL_COMMIT = "e91c1a62f84611151a399ce157e6031a4a73618f"
ROOT = Path("/output")
WORK = ROOT / "winner"
DATA = ROOT / "ubc_data"
RAW = Path("/raw-events.tar.gz")
CHALLENGE = Path("/challenge")
EVENTS = ("add_to_cart", "page_visit", "product_buy", "remove_from_cart", "search_query")
EVALUATION_TASKS = ("churn", "propensity_category", "propensity_sku", "conversion",
                    "propensity_new_sku", "propensity_price")
SCORE_TASKS = ("churn", "propensity_category", "propensity_sku", "hidden1", "hidden2", "hidden3")
OFFICIAL_FINAL = {"churn": 0.7375, "propensity_category": 0.8179,
                  "propensity_sku": 0.8224, "hidden1": 0.7717,
                  "hidden2": 0.8293, "hidden3": 0.8161}


def git_commit(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def link(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if path.resolve() != target.resolve():
            raise ValueError(f"unexpected existing link: {path}")
    elif path.exists():
        raise FileExistsError(path)
    else:
        path.symlink_to(target, target_is_directory=target.is_dir())


def check():
    actual = {"winner": git_commit(Path("/source")), "official": git_commit(Path("/official"))}
    expected = {"winner": WINNER_COMMIT, "official": OFFICIAL_COMMIT}
    if actual != expected:
        raise ValueError(f"source revision mismatch: expected {expected}, got {actual}")
    needed = [CHALLENGE / "input" / "relevant_clients.npy",
              CHALLENGE / "target" / "train_target.parquet",
              CHALLENGE / "target" / "validation_target.parquet",
              CHALLENGE / "product_properties.parquet", RAW]
    for path in needed:
        if not path.is_file():
            raise FileNotFoundError(path)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for rec2 transformer training")
    print(json.dumps({"source_commits": actual, "gpu": torch.cuda.get_device_name(0),
                      "python": sys.version, "torch": torch.__version__,
                      "data_present": True}, indent=2))


def prepare():
    check()
    manifest_path = ROOT / "input-manifest.json"
    if manifest_path.is_file() and WORK.is_dir() and all(
        (DATA / f"{event}.parquet").is_file() for event in EVENTS
    ):
        print(f"already prepared: {DATA}")
        return
    if not WORK.exists():
        shutil.copytree("/source", WORK, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    DATA.mkdir(exist_ok=True)
    link(DATA / "input", CHALLENGE / "input")
    link(DATA / "target", CHALLENGE / "target")
    link(DATA / "product_properties.parquet", CHALLENGE / "product_properties.parquet")
    with tarfile.open(RAW, "r:gz") as archive:
        members = {member.name: member for member in archive if member.name in
                   {f"{event}.parquet" for event in EVENTS}}
        if len(members) != len(EVENTS):
            raise ValueError("raw Synerise archive lacks expected event tables")
        for name, member in members.items():
            target = DATA / name
            if target.exists():
                continue
            partial = target.with_suffix(".parquet.partial")
            with archive.extractfile(member) as source, partial.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            partial.replace(target)
    submission = ROOT / "submission_data"
    (submission / "input").mkdir(parents=True, exist_ok=True)
    link(submission / "target", CHALLENGE / "target")
    link(submission / "product_properties.parquet", CHALLENGE / "product_properties.parquet")
    link(submission / "input" / "relevant_clients.npy",
         CHALLENGE / "input" / "relevant_clients.npy")
    for event in EVENTS:
        link(submission / "input" / f"{event}.parquet", DATA / f"{event}.parquet")
    for relative, target in (
        ("feature_engineering/input/ubc_data", DATA),
        ("feature_engineering/sub_input/ubc_data", submission),
        ("mtl_transformer/dataset/ubc_data", DATA),
        ("cl_transformer/input/ubc_data", DATA),
        ("stacking/data/ubc_data", DATA),
        ("stacking/recsys2025", Path("/official")),
    ):
        link(WORK / relative, target)
    manifest = {"winner_commit": WINNER_COMMIT, "official_commit": OFFICIAL_COMMIT,
                "challenge_train_target_sha256": sha256(CHALLENGE / "target/train_target.parquet"),
                "challenge_validation_target_sha256": sha256(CHALLENGE / "target/validation_target.parquet"),
                "raw_archive_sha256": sha256(RAW), "data_root": str(DATA)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def normalize():
    """Preserve event times while adapting raw string columns for modern Polars."""
    import polars as pl

    manifest_path = ROOT / "timestamp-normalization.json"
    if manifest_path.is_file():
        print(f"already normalized: {DATA}")
        return
    changes = {}
    for event in EVENTS:
        path = DATA / f"{event}.parquet"
        before = sha256(path)
        dtype = pl.scan_parquet(path).collect_schema()["timestamp"]
        if dtype == pl.String:
            partial = path.with_suffix(".normalized.parquet")
            pl.scan_parquet(path).with_columns(
                pl.col("timestamp").str.strptime(
                    pl.Datetime("us"), format="%Y-%m-%d %H:%M:%S", strict=True
                )
            ).sink_parquet(partial)
            partial.replace(path)
        elif dtype.base_type() != pl.Datetime:
            raise ValueError(f"unexpected timestamp type in {path}: {dtype}")
        changes[event] = {"before_sha256": before, "after_sha256": sha256(path),
                          "timestamp_type": str(pl.scan_parquet(path).collect_schema()["timestamp"])}
    manifest_path.write_text(json.dumps({"source_archive_sha256":
        json.loads((ROOT / "input-manifest.json").read_text())["raw_archive_sha256"],
        "format": "%Y-%m-%d %H:%M:%S", "events": changes}, indent=2) + "\n")


def execute(command, cwd, name):
    marker = ROOT / "stages" / f"{name}.json"
    marker.parent.mkdir(exist_ok=True)
    if marker.exists():
        print(f"already complete: {name}")
        return
    log = ROOT / "logs" / f"{name}.log"
    log.parent.mkdir(exist_ok=True)
    print(f"running {name}; log: {log}", flush=True)
    with log.open("a") as stream:
        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ,
                                     "PYTHONPATH": f"{cwd}:{WORK / 'stacking/src'}:/official"})
    if result.returncode:
        with log.open(errors="replace") as stream:
            tail = "".join(deque(stream, maxlen=80))[-10000:]
        print(tail, file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, command)
    marker.write_text(json.dumps({"stage": name, "command": command,
                                  "winner_commit": WINNER_COMMIT}) + "\n")


def zip_profiles(directory, name):
    destination = directory / name
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in ("client_ids.npy", "embeddings.npy"):
            archive.write(directory / filename, filename)
    return destination


def prepare_full_properties():
    """Submission feature engineering needs the raw archive's full SKU table."""
    destination = DATA / "product_properties_full.parquet"
    if not destination.is_file():
        partial = destination.with_suffix(".parquet.partial")
        with tarfile.open(RAW, "r:gz") as archive:
            member = archive.getmember("product_properties.parquet")
            with archive.extractfile(member) as source, partial.open("wb") as stream:
                shutil.copyfileobj(source, stream)
        partial.replace(destination)
    properties = DATA / "product_properties.parquet"
    if properties.is_symlink() and properties.resolve() == destination:
        properties.unlink()
    link(properties, CHALLENGE / "product_properties.parquet")
    for properties in (ROOT / "submission_data/product_properties.parquet",):
        if properties.is_symlink() and properties.resolve() != destination:
            if properties.resolve() != (CHALLENGE / "product_properties.parquet").resolve():
                raise ValueError(f"unexpected properties link: {properties}")
            properties.unlink()
        link(properties, destination)
    (ROOT / "submission-properties.json").write_text(json.dumps({
        "archive_sha256": json.loads((ROOT / "input-manifest.json").read_text())["raw_archive_sha256"],
        "member": "product_properties.parquet", "sha256": sha256(destination),
        "used_for": "submission feature engineering; MTL keeps the published SKU vocabulary"}, indent=2) + "\n")


def feature():
    cwd = WORK / "feature_engineering"
    for data, output, name in (
        ("input", "result/feature_engineering", "feature_local"),
        ("sub_input", "result/submission_feature_engineering", "feature_submit"),
    ):
        execute([sys.executable, "-m", "feature_engineering.aggregated_features_baseline.create_embeddings",
                 "--data-dir", f"{data}/ubc_data", "--embeddings-dir", output,
                 "--top-n", "10", "--num-price-bins", "5", "--random-state", "42"], cwd, name)
        zip_profiles(cwd / output, "embeddings.zip")


def adapt_mtl_url_vocabulary():
    """Size the winner's URL embedding for IDs in the released event tables."""
    import numpy as np

    manifest_path = ROOT / "mtl-url-vocabulary.json"
    if manifest_path.is_file():
        return
    dataset = WORK / "mtl_transformer/dataset/local"
    highest = max(
        int(sequence.max())
        for split in ("train", "valid")
        for sequence in np.load(dataset / split / "url_id.npy", allow_pickle=True)
        if len(sequence)
    )
    required = highest + 1
    published = 373_500
    if required <= published:
        size = published
    else:
        size = required
        source_hashes = {}
        for filename in ("train.py", "create_embeddings.py"):
            path = WORK / "mtl_transformer/src" / filename
            source_hashes[filename] = {"before_sha256": sha256(path)}
            before = path.read_text()
            old = "num_url = 373_500"
            if before.count(old) != 1:
                raise ValueError(f"unexpected URL vocabulary declaration in {path}")
            path.write_text(before.replace(old, f"num_url = {size:_}"))
            source_hashes[filename]["after_sha256"] = sha256(path)
    manifest_path.write_text(json.dumps({"published_num_url": published,
        "maximum_observed_url_id": highest, "adapted_num_url": size,
        "source_hashes": source_hashes if required > published else {},
        "reason": "released dataset contains URL IDs beyond the published hard-coded size"},
        indent=2) + "\n")


def recover_completed_mtl_training():
    """Keep completed epochs when W&B fails while logging checkpoint artifacts."""
    marker = ROOT / "stages/mtl_train.json"
    log = ROOT / "logs/mtl_train.log"
    weights = sorted((WORK / "mtl_transformer/results/weights").glob("*.ckpt"))
    if marker.exists() or not log.is_file() or not weights:
        return
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 20000))
        tail = stream.read()
    if (b"`Trainer.fit` stopped: `max_epochs=25` reached." not in tail or
            b"Unable to write staging files" not in tail):
        return
    marker.write_text(json.dumps({"stage": "mtl_train", "winner_commit": WINNER_COMMIT,
        "training_completed_epochs": 25, "checkpoint_files": [p.name for p in weights],
        "recovered_after": "W&B artifact staging permission error during trainer teardown"}) + "\n")
    print("recovered completed 25-epoch MTL training from saved checkpoints", flush=True)


def mtl_validation_scores(weights):
    """Read the 25 monitored scores from the winner's offline W&B run."""
    epoch_paths = {}
    for path in weights:
        match = re.fullmatch(r"epoch=(\d+)-step=\d+\.ckpt", path.name)
        if not match:
            raise ValueError(f"unexpected MTL checkpoint name: {path.name}")
        epoch_paths[int(match.group(1))] = path
    for run in sorted((WORK / "mtl_transformer/wandb").glob("offline-run-*/run-*.wandb"),
                      key=lambda path: path.stat().st_mtime, reverse=True):
        raw = re.findall(rb"valid/sum_score\x82\x01.[0-9.eE+-]+", run.read_bytes())
        values = [float(value.split(b"\x82\x01", 1)[1][1:]) for value in raw]
        if len(values) != 50 or any(values[2*i] != values[2*i+1] for i in range(25)):
            continue
        epoch_scores = values[::2]
        top_epochs = {epoch for epoch, _ in sorted(enumerate(epoch_scores),
                      key=lambda pair: pair[1], reverse=True)[:len(weights)]}
        if set(epoch_paths) != top_epochs:
            continue
        return epoch_paths, epoch_scores, run
    raise ValueError("no complete offline W&B history matches the saved MTL top checkpoints")


def mtl():
    cwd = WORK / "mtl_transformer"
    execute([sys.executable, "src/create_dataset.py", "--dataset_type", "local", "--split_data"], cwd, "mtl_dataset_train")
    adapt_mtl_url_vocabulary()
    recover_completed_mtl_training()
    execute([sys.executable, "src/train.py", "--dataset_type", "local"], cwd, "mtl_train")
    weights = sorted((cwd / "results/weights").glob("*.ckpt"))
    if not weights:
        raise FileNotFoundError("MTL training produced no checkpoint")
    epoch_paths, epoch_scores, wandb_run = mtl_validation_scores(weights)
    best_epoch = max(epoch_paths, key=lambda epoch: epoch_scores[epoch])
    best_score, selected = epoch_scores[best_epoch], epoch_paths[best_epoch]
    (ROOT / "mtl-checkpoint.json").write_text(json.dumps({"selected": str(selected),
        "selection": "highest offline W&B valid/sum_score among saved top checkpoints",
        "epoch": best_epoch, "valid_sum_score": best_score,
        "saved_epoch_scores": {str(epoch): epoch_scores[epoch] for epoch in epoch_paths},
        "wandb_history": str(wandb_run), "sha256": sha256(selected)}) + "\n")
    for dataset_type in ("local", "sub"):
        execute([sys.executable, "src/create_dataset.py", "--dataset_type", dataset_type], cwd,
                f"mtl_dataset_{dataset_type}")
        execute([sys.executable, "src/create_embeddings.py", "--dataset_type", dataset_type,
                 "--model_path", str(selected)], cwd, f"mtl_profiles_{dataset_type}")
        zip_profiles(cwd / f"results/{dataset_type}_embeddings", "rintaro.zip")


def cl():
    cwd = WORK / "cl_transformer"
    execute([sys.executable, "-m", "scripts.preprocess_events"], cwd, "cl_events")
    execute([sys.executable, "-m", "scripts.create_relevant_proba"], cwd, "cl_relevant")
    for mode in ("submission", "local_by_submission_model"):
        execute([sys.executable, "-m", "scripts.create_embeddings", f"mode={mode}"], cwd,
                f"cl_{mode}")


def baseline():
    baseline_dir = WORK / "stacking/data/baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    execute([sys.executable, "-m", "baseline.aggregated_features_baseline.create_embeddings",
             "--data-dir", str(DATA), "--embeddings-dir", str(baseline_dir)],
            Path("/official"), "official_baseline_for_stack")


def stack():
    cwd = WORK / "stacking"
    destination = cwd / "data/embeddings"
    for part in ("local", "submit"):
        (destination / part).mkdir(parents=True, exist_ok=True)
    files = {
        "local/313038_local.zip": WORK / "feature_engineering/result/feature_engineering/embeddings.zip",
        "submit/313038.zip": WORK / "feature_engineering/result/submission_feature_engineering/embeddings.zip",
        "local/rintaro.zip": WORK / "mtl_transformer/results/local_embeddings/rintaro.zip",
        "submit/rintaro.zip": WORK / "mtl_transformer/results/sub_embeddings/rintaro.zip",
        "local/v4001_local_embeddings_by_sub_model.zip": WORK / "cl_transformer/output/create_embeddings/local_by_submission_model/embeddings.zip",
        "submit/v4001_sub_embeddings.zip": WORK / "cl_transformer/output/create_embeddings/submission/embeddings.zip",
    }
    for relative, source in files.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        link(destination / relative, source)
    baseline()
    execute([sys.executable, "-m", "run.stacking"], cwd, "stack")


def evaluate():
    profiles = WORK / "stacking/data/outputs/stacking/local"
    if not (profiles / "embeddings.npy").is_file():
        raise FileNotFoundError(profiles / "embeddings.npy")
    (ROOT / "scores").mkdir(exist_ok=True)
    command = [sys.executable, "-m", "training_pipeline.train", "--data-dir", str(DATA),
               "--embeddings-dir", str(profiles), "--tasks", *EVALUATION_TASKS,
               "--hidden-logging-mode", "--log-name", "rec2_local",
               "--devices", "auto", "--accelerator", "gpu", "--score-dir", str(ROOT / "scores"),
               "--disable-relevant-clients-check"]
    execute(command, Path("/official"), "official_local_evaluation")


def report():
    scores_path = ROOT / "scores/scores.json"
    if not scores_path.is_file():
        raise FileNotFoundError(scores_path)
    raw_scores = json.loads(scores_path.read_text())
    missing = set(SCORE_TASKS) - raw_scores.keys()
    if missing:
        raise ValueError(f"organizer evaluator omitted tasks: {sorted(missing)}")
    local_scores = {task: float(raw_scores[task]) for task in SCORE_TASKS}
    result = {"winner_commit": WINNER_COMMIT, "official_evaluator_commit": OFFICIAL_COMMIT,
              "local_public_validation": local_scores,
              "local_public_validation_sum": round(sum(local_scores.values()), 4),
              "hidden_final_leaderboard": OFFICIAL_FINAL,
              "hidden_final_leaderboard_sum": 4.7949,
              "hidden_final_borda": 662,
              "same_evaluation_split": False,
              "note": "Winner reports local leakage; local and hidden final scores are not directly comparable."}
    (ROOT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("check", "prepare", "normalize", "feature", "baseline",
                                          "mtl", "cl", "stack", "evaluate", "report", "all"))
    args = parser.parse_args()
    if args.stage == "check":
        check()
        return
    prepare()
    prepare_full_properties()
    if args.stage == "prepare":
        return
    if args.stage in {"normalize", "mtl", "cl", "baseline", "stack", "evaluate", "all"}:
        normalize()
    if args.stage == "normalize":
        return
    stages = (feature, mtl, cl, stack, evaluate, report) if args.stage == "all" else (globals()[args.stage],)
    for stage in stages:
        stage()


if __name__ == "__main__":
    main()
