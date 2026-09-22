import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openrec_experiments.data import prepare, validate
from openrec_experiments.datasets.ebnerd.data import (
    _article_metadata,
    _project_behavior_batch,
    ebnerd,
)
from openrec_experiments.datasets.kuairand.data import kuairand
from openrec_experiments.evaluation import binary_metrics, evaluate
from openrec_experiments.datasets.ebnerd.runner_features import (
    apply_semantic_title_fallback,
    contextual_features,
    interaction_features,
    relative_temporal_features,
    long_term_user_features,
    candidate_statistical_ranks,
    session_exposure_features,
    past_candidate_exposure_features,
    past_candidate_engagement_features,
)
from openrec_experiments.runner import (
    logged_history_inputs,
    run,
    split,
)
from openrec_experiments.features import materialize
from openrec_experiments.openrec import load_openrec
from openrec_experiments.provenance import digest, write_json
from openrec_experiments.semantic import load_embeddings

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
    articles = pd.DataFrame({
        "article_id": [10, 20, 30],
        "category": [1, 2, 3],
        "subcategory": [[11], [21, 22], []],
        "topics": [["local"], ["sport"], ["culture"]],
        "title": ["First article", "Second article", "Third article"],
        "published_time": [pd.Timestamp("2023-05-17T12:00:00")] * 3,
    })
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
    assert set(frame[frame.item_id.eq("20")].subcategory) == {"21,22"}
    assert set(frame[frame.item_id.eq("20")].tags) == {"sport"}
    assert frame.title.str.len().gt(0).all()
    assert frame.pub_time.gt(0).all()
    assert frame.groupby("group_id").candidate_count.nunique().eq(1).all()
    assert frame.position.ge(0).all()


def test_contextual_features_are_label_independent_and_train_fitted():
    frame = ebnerd(*raw_ebnerd())
    mask = frame.timestamp.lt(frame.timestamp.sort_values().iloc[len(frame) // 2])
    first, names = contextual_features(frame, mask)
    changed = frame.copy()
    changed["label"] ^= 1
    second, repeated = contextual_features(changed, mask)
    assert names == repeated
    assert names[:3] == ["context.position", "context.candidate_count", "context.position_ratio"]
    np.testing.assert_array_equal(first, second)


def test_interaction_features_only_use_visible_clicks():
    frame = ebnerd(*raw_ebnerd()).sort_values(["timestamp", "sample_id"]).reset_index(drop=True)
    settings = config()
    settings.update(update_interval_ms=1, feedback_delay_ms=0,
                    evaluation_feedback="delayed_replay")
    values, names = interaction_features(frame, settings)
    assert len(names) == values.shape[1] == 4
    first_time = frame.timestamp.min()
    assert not values[frame.timestamp.eq(first_time), 3].any()
    changed = frame.copy()
    changed.loc[changed.timestamp.ge(pd.Timestamp(settings["validation_start"]).value // 1_000_000), "label"] ^= 1
    frozen = dict(settings, evaluation_feedback="frozen")
    first, _ = interaction_features(frame, frozen)
    second, _ = interaction_features(changed, frozen)
    np.testing.assert_array_equal(first, second)


def test_relative_temporal_features_are_group_local_and_label_free():
    frame = ebnerd(*raw_ebnerd())
    values, names = relative_temporal_features(frame)
    assert values.shape == (len(frame), 4)
    assert names[1] == "temporal.freshness_rank_ratio"
    assert ((values[:, 1] >= 0) & (values[:, 1] <= 1)).all()
    changed = frame.copy()
    changed["label"] ^= 1
    np.testing.assert_array_equal(values, relative_temporal_features(changed)[0])


def test_long_term_user_profile_is_frozen_and_finite():
    frame = ebnerd(*raw_ebnerd()).sort_values(["timestamp", "sample_id"]).reset_index(drop=True)
    settings = config()
    values, names = long_term_user_features(frame, settings)
    assert values.shape == (len(frame), 9) and len(names) == 9
    assert np.isfinite(values).all()
    changed = frame.copy()
    changed.loc[changed.timestamp.ge(pd.Timestamp(settings["validation_start"]).value // 1_000_000), "label"] ^= 1
    np.testing.assert_array_equal(values, long_term_user_features(changed, settings)[0])


def test_candidate_statistical_ranks_are_request_local():
    frame = ebnerd(*raw_ebnerd())
    raw = np.tile([3.0, 2.0, 1.0], len(frame) // 3)
    values, names = candidate_statistical_ranks(frame, {"signal": raw})
    assert names == ["context.signal_rank_pct", "context.signal_minus_request_mean"]
    np.testing.assert_allclose(values[:3, 0], [1 / 3, 2 / 3, 1])
    np.testing.assert_allclose(values[:3, 1], [1, 0, -1])


def test_session_exposure_features_use_only_previous_impressions():
    frame = ebnerd(*raw_ebnerd()).sort_values(["timestamp", "sample_id"]).reset_index(drop=True)
    values, names = session_exposure_features(frame)
    assert values.shape == (len(frame), 8) and len(names) == 8
    first = frame.groupby("user_id", sort=False).head(3).index
    assert not values[first, 3:].any()


def test_past_candidate_exposures_exclude_simultaneous_requests():
    frame = ebnerd(*raw_ebnerd()).sort_values(
        ["timestamp", "sample_id"]
    ).reset_index(drop=True)
    values, names = past_candidate_exposure_features(frame)
    assert values.shape == (len(frame), 18) and len(names) == 18
    first_time = frame.timestamp.min()
    assert not values[frame.timestamp.eq(first_time), 0].any()
    later = frame.timestamp.gt(first_time) & frame.item_id.eq("10")
    assert values[later, 6].max() > 0
    extended, extended_names = past_candidate_exposure_features(frame, True)
    assert extended.shape == (len(frame), 32) and len(extended_names) == 32


def test_past_candidate_engagement_excludes_current_feedback():
    frame = ebnerd(*raw_ebnerd()).sort_values(
        ["timestamp", "sample_id"]
    ).reset_index(drop=True)
    frame["read_time"] = 30.0
    frame["scroll_percentage"] = 100.0
    values, names = past_candidate_engagement_features(frame)
    assert values.shape == (len(frame), 18) and len(names) == 18
    first_time = frame.timestamp.min()
    assert not values[frame.timestamp.eq(first_time), 0].any()
    later = frame.timestamp.gt(first_time) & frame.item_id.eq("10")
    assert values[later, 6].max() > 0


def test_bad_candidates_and_unlabeled_test_fail():
    behaviors, articles = raw_ebnerd()
    behaviors.at[0, "article_ids_clicked"] = [999]
    with pytest.raises(ValueError, match="candidate"):
        ebnerd(behaviors, articles)


def test_large_scale_projection_keeps_all_clicks_and_one_stable_negative():
    behaviors, articles = raw_ebnerd()
    behaviors["read_time"] = 30.0
    behaviors["scroll_percentage"] = 100.0
    behaviors["session_id"] = "session"
    metadata = _article_metadata(articles)
    assert metadata.pub_time.eq(1_684_324_800).all()
    sampled = _project_behavior_batch(behaviors.iloc[:2], metadata, True)
    repeated = _project_behavior_batch(behaviors.iloc[:2], metadata, True)
    full = _project_behavior_batch(behaviors.iloc[:2], metadata, False)
    assert sampled.groupby("group_id").size().eq(2).all()
    assert sampled.groupby("group_id").label.sum().eq(1).all()
    assert sampled.sample_id.tolist() == repeated.sample_id.tolist()
    assert full.groupby("group_id").size().eq(3).all()
    assert {"read_time", "scroll_percentage", "session_id"}.issubset(sampled.columns)
    assert sampled.read_time.notna().all()
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


def test_vectorized_group_metrics_preserve_tie_semantics():
    frame = pd.DataFrame({
        "group_id": ["a"] * 4 + ["b"] * 3,
        "sample_id": ["2", "1", "4", "3", "7", "5", "6"],
        "label": [1, 0, 1, 0, 0, 1, 0],
    })
    scores = [.8, .8, .2, .2, .5, .5, .1]
    result = evaluate(frame, scores, "ebnerd")["impression_macro"]

    # Group a has one tied positive/negative pair at each score (AUC .5).
    # Group b ties its positive with one negative and beats the other (AUC .75).
    assert result["auc"] == pytest.approx(.625)
    assert result["mrr"] == pytest.approx(((1 / 2 + 1 / 4) / 2 + 1) / 2)
    assert result["ndcg@5"] <= 1
    assert result["ndcg@10"] <= 1


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


def test_kuairand_static_video_content_projection():
    base = ebnerd(*raw_ebnerd())
    logs = pd.DataFrame({"user_id": base.user_id, "video_id": base.item_id,
                         "time_ms": base.timestamp, "is_click": base.label,
                         "is_rand": 0, "tab": 0})
    videos = pd.DataFrame({
        "video_id": [10, 20, 30],
        "video_type": ["NORMAL", "NORMAL", "AD"],
        "upload_dt": ["2023-05-17"] * 3,
        "upload_type": ["Web", "ShortImport", "Web"],
        "tag": ["sport", "music", None],
    })
    frame = kuairand(logs, videos)
    assert set(frame.category) == {"NORMAL", "AD"}
    assert set(frame.subcategory) == {"Web", "ShortImport"}
    assert frame.loc[frame.item_id.eq("10"), "tags"].eq("sport").all()
    assert frame.pub_time.eq(int(pd.Timestamp("2023-05-17T00:00:00Z").timestamp())).all()
    assert frame.title.eq("").all()

    with pytest.raises(ValueError, match="duplicate"):
        kuairand(logs, pd.concat([videos, videos.iloc[[0]]]))


def test_features_ignore_same_time_and_future_labels():
    load_openrec(ROOT)
    frame = ebnerd(*raw_ebnerd())
    settings = config()
    users, items = materialize(frame, settings)
    assert items.iloc[:6].event_count.eq(0).all()
    assert users.iloc[:6].event_count.eq(0).all()
    assert items.iloc[:3].content_age_hours.eq(24).all()
    assert items.title.str.len().gt(0).all()
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


def test_logged_history_excludes_random_and_frozen_future_clicks():
    frame = pd.DataFrame({
        "user_id": ["u", "u", "u", "u"],
        "timestamp": [100, 200, 300, 1_100],
        "policy": ["standard", "random", "standard", "standard"],
        "label": [1, 1, 0, 1],
    })
    settings = {
        "max_history": 2,
        "update_interval_ms": 1,
        "feedback_delay_ms": 0,
        "validation_start": "1970-01-01T00:00:01Z",
        "evaluation_feedback": "frozen",
    }
    item_vectors = np.eye(4, dtype=np.float32)
    candidate_indices = np.arange(4, dtype=np.int64)
    vectors, masks, indices = logged_history_inputs(
        frame, settings, item_vectors, candidate_indices
    )

    assert masks[indices[0]].all()
    np.testing.assert_array_equal(vectors[indices[2], -1], item_vectors[0])
    np.testing.assert_array_equal(vectors[indices[3], -1], item_vectors[0])
    assert not np.any(np.all(vectors == item_vectors[1], axis=2))
    assert not np.any(np.all(vectors == item_vectors[3], axis=2))


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


def test_semantic_embedding_artifact_is_validated(tmp_path):
    path = tmp_path / "semantic.parquet"
    pd.DataFrame({"article_id": ["10", "20"],
                  "embedding": [np.array([1, 0], dtype=np.float32),
                                np.array([0, 1], dtype=np.float32)]}).to_parquet(path)
    write_json(str(path) + ".manifest.json", {
        "rows": 2, "dimension": 2, "output_sha256": digest(path)
    })
    ids, matrix, present, manifest = load_embeddings(path)
    assert ids.tolist() == ["10", "20"]
    assert matrix.shape == (2, 2)
    assert present.tolist() == [True, True]
    assert manifest["dimension"] == 2
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="manifest"):
        load_embeddings(path)


def test_semantic_content_uses_title_hash_only_as_missing_fallback():
    items = pd.DataFrame({"title": ["has body", "title fallback", "also has body"]})
    result = apply_semantic_title_fallback(
        items, np.array([True, False]), np.array([0, 1, 0])
    )
    assert result.title.tolist() == ["", "title fallback", ""]
    assert items.title.tolist() == ["has body", "title fallback", "also has body"]


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
