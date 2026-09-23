"""Rebuild Hallucinated's final Blind-B assembly from published stage outputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import urlopen


WINNER_COMMIT = "3bc859dd716d0891f12311e6547d41a57b041b8e"
RESPONSE_SHA256 = "efe776ea142aacd2937af939045166e0e23315e48e7884155c0cbe5294eb57ed"
ARTIFACT_REPO = "NicoloLocatelli/RecSys_ACM_2026_Hallucinated"
ARTIFACT_REVISION = "a187aa513434c5169aec8ecfffeda8c6b04aaf45"
ARTIFACTS = {
    "LAST.json": (
        "src/heuristic/LAST.json",
        "6308050cdfefbc8764bc1f8a517fbd003990eb005efc405de7c87779b82be1bd",
    ),
    "FINAL29.json": (
        "src/heuristic/FINAL29.json",
        "7cf0ba70a5194feeb9849b59e111acd28ff8a9fd37814f05e00f5771b7fddc6b",
    ),
    "reranker.json": (
        "models/reranker_oof/blind_b_retrain/no_filter_v5/v2_blind_last/submissions/"
        "blind_b_no_filter_v5_v2_blind_last.json",
        "95c7312108cd787b949b7308d2d92b0d95eaf22486fef7593b822818dcbf6ecd",
    ),
}
CATALOG_SIZE = 47_071
EXPECTED = {"catalog_diversity": 0.027490387, "lexical_diversity": 0.957074025}
SOURCE = Path("/source")
OUTPUT = Path("/output")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_source():
    commit = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"],
                                     text=True).strip()
    if commit != WINNER_COMMIT:
        raise ValueError(f"winner commit mismatch: {commit} != {WINNER_COMMIT}")
    if subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain",
                                "--untracked-files=no"], text=True).strip():
        raise ValueError("winner source checkout has modified tracked files")
    response = SOURCE / "data/blind_b_responses/final_pipeline/responses.json"
    merger = SOURCE / "src/merge_submission.py"
    if not response.is_file() or not merger.is_file():
        raise FileNotFoundError("published final responses or merger missing")
    response_sha256 = sha256(response)
    if response_sha256 != RESPONSE_SHA256:
        raise ValueError(f"winner response checksum mismatch: {response_sha256}")
    return {"winner_commit": commit, "response_sha256": response_sha256}


def download_artifact(name):
    path, expected_sha256 = ARTIFACTS[name]
    destination = OUTPUT / name
    if destination.is_file() and sha256(destination) == expected_sha256:
        return destination
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    if endpoint != "https://hf-mirror.com":
        raise ValueError("HF_ENDPOINT must use the requested https://hf-mirror.com proxy")
    url = f"{endpoint}/datasets/{ARTIFACT_REPO}/resolve/{ARTIFACT_REVISION}/{path}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".json.partial")
    with urlopen(url, timeout=90) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    digest = sha256(partial)
    if digest != expected_sha256:
        partial.unlink()
        raise ValueError(f"published artifact checksum mismatch: {digest}")
    partial.replace(destination)
    return destination


def assemble_submission():
    final_heuristic = download_artifact("FINAL29.json")
    reranker = download_artifact("reranker.json")
    official = download_artifact("LAST.json")
    heuristic_source = SOURCE / "src/heuristic"
    graft_turn_one = OUTPUT / "FINAL29_t1graft.json"
    subprocess.run([sys.executable, str(heuristic_source / "96_graft_turn1.py"),
                    "--fused-turn1", str(reranker), str(final_heuristic)], check=True)
    if not graft_turn_one.is_file():
        raise FileNotFoundError(graft_turn_one)
    rebuilt = OUTPUT / "prediction.json"
    subprocess.run([sys.executable, str(heuristic_source / "98_top1_graft.py"),
                    "--a", str(graft_turn_one), "--b", str(reranker),
                    "--out", str(rebuilt)], check=True)
    responses_path = SOURCE / "data/blind_b_responses/final_pipeline/responses.json"
    subprocess.run([sys.executable, str(SOURCE / "src/merge_submission.py"),
                    "--preds", str(rebuilt), "--response", str(responses_path)], check=True)
    published_rows = json.loads(official.read_text())
    rebuilt_rows = json.loads(rebuilt.read_text())
    published_by_session = {row["session_id"]: row for row in published_rows}
    rebuilt_by_session = {row["session_id"]: row for row in rebuilt_rows}
    if len(published_by_session) != len(published_rows) or rebuilt_by_session != published_by_session:
        raise ValueError("final assembly differs from the published winning submission")
    if sha256(rebuilt) != sha256(official):
        raise ValueError("final assembly content matches, but byte-level output differs")
    return official, rebuilt, published_rows


def verify():
    manifest = check_source()
    artifact, rebuilt, original = assemble_submission()
    responses_path = SOURCE / "data/blind_b_responses/final_pipeline/responses.json"
    responses = json.loads(responses_path.read_text())
    if len(original) != 80 or len(responses) != 80:
        raise ValueError("expected 80 Blind-B sessions")
    ids = set()
    all_tracks = set()
    for row in original:
        session = row["session_id"]
        tracks = row["predicted_track_ids"]
        if session in ids or session not in responses:
            raise ValueError(f"duplicate or unmatched session: {session}")
        if len(tracks) != 20 or len(set(tracks)) != 20:
            raise ValueError(f"session {session} has invalid top-20 tracks")
        ids.add(session)
        all_tracks.update(tracks)

    sys.path.insert(0, str(SOURCE / "src"))
    from merge_submission import compute_lexical_diversity

    lexical = compute_lexical_diversity([row["predicted_response"] for row in original])
    catalog = len(all_tracks) / CATALOG_SIZE
    if round(lexical, 9) != EXPECTED["lexical_diversity"]:
        raise ValueError(f"lexical diversity differs from leaderboard: {lexical}")
    if round(catalog, 9) != EXPECTED["catalog_diversity"]:
        raise ValueError(f"catalog diversity differs from leaderboard: {catalog}")
    official_ndcg = 0.618486308
    official_judge_raw = 4.75
    composite_from_official_hidden = (
        0.50 * official_ndcg + 0.10 * catalog + 0.10 * lexical
        + 0.30 * ((official_judge_raw - 1.0) / 4.0)
    )
    if round(composite_from_official_hidden, 9) != 0.688949595:
        raise ValueError("official composite inputs do not match the leaderboard")
    manifest.update({"artifact_repo": ARTIFACT_REPO, "artifact_revision": ARTIFACT_REVISION,
                     "artifact_sha256": sha256(artifact),
                     "intermediate_sha256": {name: sha256(OUTPUT / name)
                                             for name in ("FINAL29.json", "reranker.json")},
                     "rebuilt_submission_sha256": sha256(rebuilt),
                     "all_published_records_rebuilt": True, "sessions": len(original),
                     "tracks_per_session": 20, "distinct_tracks": len(all_tracks),
                     "catalog_size": CATALOG_SIZE, "catalog_diversity": catalog,
                     "lexical_diversity": lexical,
                     "official_ndcg_at_20": official_ndcg,
                     "official_llm_judge_raw": official_judge_raw,
                     "official_composite": 0.688949595,
                     "composite_from_official_hidden_metrics": composite_from_official_hidden,
                     "hidden_metrics_recomputed": False})
    (OUTPUT / "verification.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("check", "fetch", "verify"))
    stage = parser.parse_args().stage
    if stage == "check":
        print(json.dumps(check_source(), indent=2))
    elif stage == "fetch":
        check_source()
        for name in ARTIFACTS:
            print(download_artifact(name))
    else:
        verify()


if __name__ == "__main__":
    main()
