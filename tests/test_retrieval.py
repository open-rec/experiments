import json

import pandas as pd
import pytest

from openrec_experiments.retrieval import _queries, _split, prepare_retrieval


def test_music_adapter_uses_only_catalogued_music_turns(tmp_path):
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    tracks = tmp_path / "tracks.parquet"
    conversation = [
        {"role": "user", "content": "play jazz", "turn_number": 1},
        {"role": "music", "content": "a", "turn_number": 1},
        {"role": "assistant", "content": "playing a", "turn_number": 1},
        {"role": "music", "content": "b", "turn_number": 2},
    ]
    sessions = pd.DataFrame([{
        "session_id": "s1", "user_id": "u1", "session_date": "2020-01-01",
        "conversations": conversation,
    }])
    sessions.to_parquet(train)
    sessions.assign(session_id="s2", session_date="2021-01-01").to_parquet(test)
    pd.DataFrame({"track_id": ["a", "b"]}).to_parquet(tracks)
    config = {"dataset": "music-crs-2026", "train_events": str(train),
              "test_events": str(test), "tracks": str(tracks)}
    output = tmp_path / "prepared.parquet"
    prepare_retrieval(config, output)
    result = pd.read_parquet(output)
    assert len(result) == 4
    assert set(result.item_id) == {"a", "b"}
    assert set(result.source_split) == {"train", "test"}
    assert json.loads((tmp_path / "prepared.parquet.manifest.json").read_text())["rows"] == 4
    with pytest.raises(FileExistsError):
        prepare_retrieval(config, output)


def test_synerise_queries_are_prior_only_and_new_to_recent(tmp_path):
    source = tmp_path / "buy.parquet"
    pd.DataFrame({
        "client_id": [1, 1, 1, 1, 1, 1],
        "sku": [10, 20, 10, 30, 40, 50],
        "timestamp": [f"2020-01-0{i} 00:00:00" for i in range(1, 7)],
    }).to_parquet(source)
    output = tmp_path / "prepared.parquet"
    prepare_retrieval({"dataset": "synerise-2025", "train_events": str(source)}, output)
    frame = pd.read_parquet(output)
    assert frame.timestamp.min() == 1577836800000
    labels = _split(frame, {"dataset": "synerise-2025", "train_fraction": 0.5,
                            "validation_fraction": 0.9})
    queries = _queries(frame, labels, 0, strict_timestamps=True)
    assert queries["validation"][0][2:] == ("30", ["10", "20", "10"])
    assert queries["test"][0][2:] == ("50", ["10", "20", "10", "30", "40"])


def test_synerise_same_second_basket_is_not_visible_as_history():
    frame = pd.DataFrame({
        "event_id": ["a", "b", "c"], "session_id": ["u"] * 3,
        "item_id": ["1", "2", "3"], "timestamp": [1000, 2000, 2000],
    })
    queries = _queries(frame, ["train", "test", "test"], 0, strict_timestamps=True)
    assert [query[3] for query in queries["test"]] == [["1"], ["1"]]
