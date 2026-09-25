"""Apply the winner's turn-1 and top-1 grafts to an all-turn dev prediction file."""

import json
from pathlib import Path


ROOT = Path("/workspace/models/clean_dev_champion")
reranker = json.loads((ROOT / "dev_predictions.json").read_text())
heuristic = json.loads((ROOT / "dev_predictions_heuristic.json").read_text())


def by_key(rows):
    return {(str(row["session_id"]), int(row["turn_number"])): row for row in rows}


xgb = by_key(reranker)
heur = by_key(heuristic)
if xgb.keys() != heur.keys() or len(xgb) != 8000:
    raise RuntimeError("reranker and heuristic must cover the same 8000 dev turns")

output = []
for key in sorted(xgb):
    base = xgb[key]
    if key[1] == 1:
        tracks = list(base["predicted_track_ids"])
    else:
        top1 = heur[key]["predicted_track_ids"][0]
        tracks = [top1]
        tracks.extend(track for track in base["predicted_track_ids"] if track != top1)
        tracks = tracks[:20]
    if len(tracks) != 20 or len(set(tracks)) != 20:
        raise RuntimeError(f"invalid grafted list at {key}")
    output.append({
        "session_id": base["session_id"],
        "user_id": base["user_id"],
        "turn_number": key[1],
        "predicted_track_ids": tracks,
        "predicted_response": "",
    })

(ROOT / "dev_predictions_final.json").write_text(json.dumps(output, indent=2))
print(f"wrote {len(output)} final grafted predictions")
