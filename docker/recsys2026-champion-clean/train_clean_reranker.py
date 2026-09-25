"""Train the published winner XGB recipe without touching the official dev labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl
import yaml


PKG_ROOT = Path("/workspace/src/reranker_oof")
sys.path[:0] = [str(PKG_ROOT), str(PKG_ROOT / "src")]

from launchers_overfit_blind_b._common import load_gt  # noqa: E402
from launchers_overfit_blind_b._rerank import (  # noqa: E402
    _all_fold_chunks,
    build_infer_dmatrix,
    eval_scored,
    holdout_chunks,
)
from src.features.pipeline import fold_chunk_paths  # noqa: E402
from src.paths import active_dataset_dir, set_active_dataset  # noqa: E402
from src.rerankers.base import DatasetSpec  # noqa: E402
from src.rerankers.xgb_ranker import XGBReranker  # noqa: E402


# Best published winner trial (trial 31). Hyperparameters are frozen; only the
# weights are refit on clean OOF train data in this script.
WINNER_PARAMS = {
    "objective": "rank:map",
    "ndcg_exp_gain": False,
    "lambdarank_unbiased": False,
    "lambdarank_score_normalization": False,
    "learning_rate": 0.20563170845928097,
    "max_depth": 24,
    "max_leaves": 171,
    "grow_policy": "lossguide",
    "min_child_weight": 27,
    "gamma": 0.09667355975035413,
    "reg_lambda": 1.6733332553592581,
    "reg_alpha": 0.010008597407147489,
    "subsample": 0.8336504862087833,
    "sampling_method": "gradient_based",
    "colsample_bytree": 0.5895583190771366,
    "colsample_bylevel": 0.8739507474716727,
    "colsample_bynode": 0.6667047102796672,
    "lambdarank_pair_method": "topk",
    "lambdarank_num_pair_per_sample": 20,
    "eval_metric": ["ndcg@10", "ndcg@50", "ndcg@200", "map@20", "pre@20", "ndcg@20"],
    "tree_method": "hist",
    "num_boost_round": 5000,
    "seed": 42,
    "max_bin": 128,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=PKG_ROOT / "configs/blind_no_filter/xgb_v5.yaml")
    ap.add_argument("--output", type=Path, default=Path("/workspace/models/clean_dev_champion"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_active_dataset(cfg["dataset_name"])
    train_paths = _all_fold_chunks("train")
    val_paths = _all_fold_chunks("val")
    test_paths = holdout_chunks()
    if not train_paths or not val_paths or not test_paths:
        raise RuntimeError(f"missing assembled chunks: train={len(train_paths)} val={len(val_paths)} dev={len(test_paths)}")

    feature_cols = list(cfg["feat_cols_keep"])
    for label, paths in (("train", train_paths), ("val", val_paths), ("dev", test_paths)):
        cols = set(pl.read_parquet(paths[0], n_rows=1).columns)
        missing = sorted(set(feature_cols) - cols)
        if missing:
            raise RuntimeError(f"{label} is missing configured features: {missing}")

    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "xgb_cache"
    ds = DatasetSpec(train_paths=train_paths, val_paths=val_paths, feat_cols=feature_cols)
    model = XGBReranker()
    bundle = model.build_dmatrix(
        ds,
        device="cuda",
        cache_dir=cache_dir,
        max_bin=128,
        max_quantile_batches=8,
        cache_host_ratio=0.0,
        dtype="float32",
        use_cuda_async_pool=True,
        use_rmm=False,
    )
    model.fit(
        dtrain=bundle.dtrain,
        dval=bundle.dval,
        feat_cols=feature_cols,
        params=WINNER_PARAMS,
        device="cuda",
        early_stopping_rounds=50,
    )
    model._booster.save_model(args.output / "winner_clean.json")

    dtest, meta = build_infer_dmatrix(
        test_paths, feature_cols, "cuda", ref=bundle.dtrain, max_bin=128
    )
    scored = model.predict_dval(dtest, meta)
    scored.write_parquet(args.output / "dev_scored.parquet")
    metrics = eval_scored(scored, load_gt("holdout"), [1, 5, 10, 20, 50, 100, 200])
    (args.output / "candidate_metrics.json").write_text(json.dumps(metrics, indent=2))

    ranked = (
        scored.sort(["session_id", "turn_number", "score"], descending=[False, False, True])
        .group_by(["session_id", "turn_number"], maintain_order=True)
        .agg([
            pl.col("track_id").head(20).alias("predicted_track_ids"),
        ])
    )
    users = (
        pl.read_parquet("/workspace/data/splitK/holdout_test.parquet")
        .select(["session_id", "user_id"])
        .unique(subset=["session_id"])
    )
    ranked = ranked.join(users, on="session_id", how="left")
    records = [
        {
            "session_id": r["session_id"],
            "user_id": r["user_id"],
            "turn_number": int(r["turn_number"]),
            "predicted_track_ids": r["predicted_track_ids"],
            "predicted_response": "",
        }
        for r in ranked.iter_rows(named=True)
    ]
    (args.output / "dev_predictions.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"wrote {len(records)} dev predictions to {args.output}")


if __name__ == "__main__":
    main()
