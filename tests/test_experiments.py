import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openrec_experiments.data import ebnerd, kuairand, prepare, validate
from openrec_experiments.evaluation import binary_metrics, evaluate
from openrec_experiments.runner import load_openrec, run, split
from openrec_experiments.features import materialize

ROOT = Path(__file__).resolve().parents[2] / "rec-algorithm"


def config():
    return {"dataset": "ebnerd", "openrec_algorithm": str(ROOT), "seed": 42,
            "model": "lr", "epochs": 2, "batch_size": 8, "learning_rate": 0.01,
            "train_start": "2023-05-18T00:00:00Z", "validation_start": "2023-05-20T00:00:00Z",
            "test_start": "2023-05-21T00:00:00Z", "test_end": "2023-05-22T00:00:00Z",
            "update_interval_ms": 86400000, "feedback_delay_ms": 3600000,
            "evaluation_feedback": "frozen"}


def raw_ebnerd():
    behaviors = pd.DataFrame([
        {"impression_id": i, "user_id": i % 2, "impression_time": pd.Timestamp(f"2023-05-{18 + i // 2}T12:00:00"),
         "article_ids_inview": [10, 20, 30], "article_ids_clicked": [10 if i % 2 else 20]}
        for i in range(8)
    ])
    articles = pd.DataFrame({"article_id": [10, 20, 30], "category": [1, 2, 3]})
    return behaviors, articles


def test_ebnerd_retains_groups_and_candidates():
    frame = ebnerd(*raw_ebnerd())
    assert len(frame) == 24
    assert frame.groupby("group_id").size().eq(3).all()
    assert frame.groupby("group_id").label.sum().eq(1).all()
    masks = split(frame, config())
    assert [int(v.sum()) for v in masks.values()] == [12, 6, 6]
    group_sets = [set(frame[m].group_id) for m in masks.values()]
    assert not group_sets[0] & group_sets[1]
    assert not group_sets[1] & group_sets[2]


def test_bad_candidates_and_unlabeled_test_fail():
    behaviors, articles = raw_ebnerd()
    behaviors.at[0, "article_ids_clicked"] = [999]
    with pytest.raises(ValueError, match="candidate"):
        ebnerd(behaviors, articles)
    behaviors.at[0, "article_ids_clicked"] = None
    with pytest.raises(ValueError, match="labeled"):
        ebnerd(behaviors, articles)


def test_identity_and_probability_validation():
    frame = ebnerd(*raw_ebnerd())
    with pytest.raises(ValueError, match="duplicate"):
        validate(pd.concat([frame, frame]))
    for scores in [[np.nan, .5], [-.1, 1.1]]:
        with pytest.raises(ValueError):
            binary_metrics([0, 1], scores)


def test_metrics_known_multiclick_example():
    frame = pd.DataFrame({"group_id": ["a"] * 3, "sample_id": ["1", "2", "3"], "label": [1, 0, 1]})
    result = evaluate(frame, [.9, .8, .7], "ebnerd")
    assert result["impression_macro"]["auc"] == .5
    assert result["impression_macro"]["mrr"] == pytest.approx((1 + 1 / 3) / 2)
    assert binary_metrics([0, 1], [.5, .5])["auc"] == .5
    assert binary_metrics([1, 1], [.5, .5])["auc"] is None


def test_random_policy_is_not_training_or_validation():
    base = ebnerd(*raw_ebnerd())
    logs = pd.DataFrame({"user_id": base.user_id, "video_id": base.item_id,
                         "time_ms": base.timestamp, "is_click": base.label,
                         "is_rand": (np.arange(len(base)) // 3) % 2, "tab": 0})
    frame = kuairand(logs)
    masks = split(frame, config())
    for name in ["train", "validation"]:
        assert set(frame[masks[name]].policy) == {"standard"}
    metrics = evaluate(frame[masks["test"]], [.5] * int(masks["test"].sum()), "kuairand-1k")
    assert set(metrics["by_policy"]) == {"random", "standard"}


def test_features_ignore_same_time_and_future_labels():
    load_openrec(ROOT)
    frame = ebnerd(*raw_ebnerd())
    settings = config()
    users, items = materialize(frame, settings)
    assert items.iloc[:6].event_count.eq(0).all()
    assert users.iloc[:6].event_count.eq(0).all()
    changed = frame.copy()
    changed.loc[changed.timestamp.ge(pd.Timestamp(settings["validation_start"]).value // 1000000), "label"] ^= 1
    u2, i2 = materialize(changed, settings)
    pd.testing.assert_frame_equal(users, u2)
    pd.testing.assert_frame_equal(items, i2)


def test_same_timestamp_group_cannot_leak():
    load_openrec(ROOT)
    frame = ebnerd(*raw_ebnerd())
    settings = config()
    settings.update(update_interval_ms=1, feedback_delay_ms=0, evaluation_feedback="delayed_replay")
    users, items = materialize(frame, settings)
    assert items.iloc[:3].event_count.eq(0).all()
    changed = frame.copy()
    changed.loc[:2, "label"] ^= 1
    u2, i2 = materialize(changed, settings)
    pd.testing.assert_frame_equal(users.iloc[:3], u2.iloc[:3])
    pd.testing.assert_frame_equal(items.iloc[:3], i2.iloc[:3])


@pytest.mark.parametrize("model", ["popularity", "lr", "fm"])
def test_prepare_train_evaluate_artifacts(tmp_path, model):
    behaviors, articles = raw_ebnerd()
    paths = [tmp_path / "behaviors.parquet", tmp_path / "articles.parquet"]
    behaviors.to_parquet(paths[0])
    articles.to_parquet(paths[1])
    source = tmp_path / "prepared.parquet"
    prepare({"dataset": "ebnerd", "inputs": list(map(str, paths))}, source)
    settings = dict(config(), data=str(source), model=model)
    output = tmp_path / "run"
    result = run(settings, output)
    assert result["test"]["rows"] == 6
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert "predictions.parquet" in manifest["artifacts"]
    from openrec_experiments.report import report
    report([output], tmp_path / "table.csv")
    table = pd.read_csv(tmp_path / "table.csv")
    assert table.iloc[0]["test.rows"] == 6
    with pytest.raises(FileExistsError):
        run(settings, output)
    if model != "popularity":
        from algorithm.feature.feature_space import FeatureSpace
        assert FeatureSpace.load(output / f"{model}.features.json").dim == manifest["feature_dimension"]
    (output / "metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        report([output], tmp_path / "bad-table.csv")


def test_split_rejects_ambiguous_timezone():
    settings = config()
    settings["train_start"] = "2023-05-18"
    with pytest.raises(ValueError, match="UTC offset"):
        split(ebnerd(*raw_ebnerd()), settings)


def test_kuairand_csv_end_to_end_and_data_tamper(tmp_path):
    base = ebnerd(*raw_ebnerd())
    logs = pd.DataFrame({"user_id": base.user_id, "video_id": base.item_id,
                         "time_ms": base.timestamp, "is_click": base.label,
                         "is_rand": (np.arange(len(base)) // 3) % 2, "tab": 0})
    csv = tmp_path / "log.csv"
    logs.to_csv(csv, index=False)
    source = tmp_path / "prepared.parquet"
    prepare({"dataset": "kuairand-1k", "inputs": [str(csv)]}, source)
    settings = dict(config(), dataset="kuairand-1k", data=str(source), model="fm")
    result = run(settings, tmp_path / "first")
    repeat = run(settings, tmp_path / "second")
    assert result == repeat
    assert set(result["test"]["by_policy"]) == {"standard", "random"}
    source.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="manifest"):
        run(settings, tmp_path / "tampered")
