"""Build the vector store as plain files: embeddings.npy + meta.json.

We deliberately do NOT use ChromaDB's HNSW index — it is an approximate index
that returned wrong neighbors when the store was copied to a cloud server.
Instead we save the raw (normalized) embeddings to a numpy file and the
medicine metadata to JSON. At query time the engine loads these (instant) and
does an EXACT brute-force cosine search. With ~5k vectors this is fast,
correct, and behaves identically on every machine.

Run once (or whenever the data changes):
    python -m src.build_vectors
"""
import os
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from . import config
from .data_loader import load_records


def build_store():
    records = load_records()
    print(f"Embedding {len(records)} medicines with '{config.EMBED_MODEL}'...")

    model = SentenceTransformer(config.EMBED_MODEL)
    texts = [r["doc_text"] for r in records]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    metas = [
        {
            "name": r["name"],
            "uses": r["uses"],
            "composition": r["composition"],
            "side_effects": r["side_effects"],
            "substitutes": r["substitutes"],
        }
        for r in records
    ]

    os.makedirs(config.CHROMA_DIR, exist_ok=True)
    emb_path = os.path.join(config.CHROMA_DIR, "embeddings.npy")
    meta_path = os.path.join(config.CHROMA_DIR, "meta.json")

    np.save(emb_path, embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)

    print(f"Saved {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]}) -> {emb_path}")
    print(f"Saved metadata -> {meta_path}")


if __name__ == "__main__":
    build_store()
