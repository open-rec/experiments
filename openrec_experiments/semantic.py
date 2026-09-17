"""Versioned semantic article embeddings for reproducible experiments."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import digest, write_json


def embed_article_text(articles_path, output, model_name, text_column, batch_size=256):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    articles_path = Path(articles_path)
    articles = pd.read_parquet(articles_path, columns=["article_id", text_column])
    if articles.article_id.isna().any() or articles.article_id.duplicated().any():
        raise ValueError("article ids must be unique and non-null")
    text = articles[text_column].fillna("").astype(str)
    present = text.str.strip().ne("").to_numpy()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    encoded = model.encode(
        ("passage: " + text[present]).tolist(), batch_size=int(batch_size),
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True,
    ).astype(np.float32)
    vectors = np.zeros((len(articles), encoded.shape[1]), dtype=np.float32)
    vectors[present] = encoded
    if vectors.ndim != 2 or len(vectors) != len(articles) or not np.isfinite(vectors).all():
        raise ValueError("semantic encoder returned invalid vectors")
    if not np.allclose(np.linalg.norm(vectors[present], axis=1), 1, atol=1e-4):
        raise ValueError("semantic embeddings must be L2-normalized")

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"article_id": articles.article_id.astype(str),
                  "text_present": present,
                  "embedding": list(vectors)}).to_parquet(output, index=False)
    write_json(str(output) + ".manifest.json", {
        "schema": 1, "model": model_name, "dimension": int(vectors.shape[1]),
        "normalized": True, "prefix": "passage: ", "articles": str(articles_path),
        "max_seq_length": int(model.max_seq_length),
        "articles_sha256": digest(articles_path), "rows": len(articles),
        "text_column": text_column, "nonempty_texts": int(present.sum()),
        "output_sha256": digest(output),
    })


def embed_titles(articles_path, output, model_name, batch_size=256):
    return embed_article_text(
        articles_path, output, model_name, "title", batch_size
    )


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
