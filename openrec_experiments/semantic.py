"""Versioned semantic article embeddings for reproducible experiments."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import digest


def load_embeddings(path):
    path = Path(path)
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    if manifest["output_sha256"] != digest(path):
        raise ValueError("semantic embedding artifact does not match its manifest")
    frame = pd.read_parquet(path)
    if frame.article_id.duplicated().any() or len(frame) != manifest["rows"]:
        raise ValueError("invalid semantic embedding identities")
    matrix = np.stack(frame.embedding.to_numpy()).astype(np.float32)
    if matrix.shape != (manifest["rows"], manifest["dimension"]):
        raise ValueError("semantic embedding dimension does not match manifest")
    present = (
        frame.text_present.to_numpy(dtype=bool)
        if "text_present" in frame
        else np.linalg.norm(matrix, axis=1) > 0
    )
    if present.shape != (manifest["rows"],):
        raise ValueError("semantic embedding presence mask does not match manifest")
    return frame.article_id.astype(str).to_numpy(), matrix, present, manifest
