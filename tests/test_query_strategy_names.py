import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_music_qwen_configs_name_actual_encoder_and_keep_upstream_cache_paths():
    found = 0
    for path in (ROOT / 'studies/baselines').glob('music-crs-2026-*.json'):
        config = json.loads(path.read_text())
        assert not any(key.startswith('query_embedding_') for key in config), path
        assert 'query_embedding' not in config.get('feature_weights', {}), path
        if 'query_qwen_emb_dir' in config:
            found += 1
            assert 'Qwen3-Embedding-8B' in config['query_qwen_emb_dir']
            assert 'Qwen3-Embedding-8B' in config['track_embedding_dir']
    assert found >= 1


def test_music_ranker_uses_named_qwen_weight_and_candidate_count():
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    from openrec_experiments.datasets.music_crs_2026.official import _SessionEntityRanker

    calls = []
    class Recall:
        _vectors = np.zeros((3, 2))

        def recall_many(self, queries, excludes, count, **kwargs):
            calls.append(count)
            return [[SimpleNamespace(item='b', score=1.0)] for _ in queries]

    tracks = pd.DataFrame([{
        'track_id': item, 'artist_id': 'artist', 'tag_list': 'jazz',
        'popularity': 1, 'track_name': 'song ' + item,
        'artist_name': 'artist', 'album_name': 'album', 'release_date': '2020-01-01',
    } for item in ('a', 'b', 'c')])
    config = {'text_min_df': 1, 'query_qwen_emb_candidates': 7,
              'feature_weights': {'semantic': 0, 'transition': 0, 'artist': 0,
                                  'tag': 0, 'popularity': 0, 'query_qwen_emb': 1}}
    ranker = _SessionEntityRanker(tracks, [], ['a'], config, query_recall=Recall(),
                                  query_vectors={('s', 1): np.array([1, 0])})
    result = ranker.rank_many([('jazz', [], ('s', 1), 'u', None)], k=1)
    assert result == [['b']]
    assert calls == [7]
