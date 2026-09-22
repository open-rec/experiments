import json

import pytest

from openrec_experiments.official_synerise_compare import TASKS, _load_scores


def test_merge_official_task_scores_requires_exact_coverage(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({task: 0.5 for task in TASKS[:3]} | {"placeholder": -1}))
    second.write_text(json.dumps({task: 0.6 for task in TASKS[3:]} | {"placeholder": -1}))

    merged = _load_scores([first, second])
    assert list(merged) == list(TASKS)
    assert merged["hidden3"] == 0.6

    with pytest.raises(ValueError, match="missing official task scores"):
        _load_scores([first])
    with pytest.raises(ValueError, match="duplicate official task score"):
        _load_scores([first, first, second])
