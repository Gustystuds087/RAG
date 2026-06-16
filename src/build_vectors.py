"""Build the ChromaDB vector store from medicine records.

Replaces the old FAISS index. Chroma stores the embedding AND the medicine
metadata together (and persists to a folder), so there is no separate
meta.pkl to keep in sync.

Run once (or whenever the data changes):
    python -m src.build_vectors
"""
import chromadb
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
    ).tolist()

    # Persistent client = data saved to disk in CHROMA_DIR (no server needed).
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)

    # Start clean so re-running doesn't duplicate records.
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:
        pass

    # cosine distance matches our normalized embeddings.
    # The HNSW index is approximate; the default search/construction effort is
    # too low for ~5k vectors and returns wrong neighbors. Raise ef + M so the
    # index is accurate (these values make recall effectively exact at this size).
    collection = client.create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,
            "hnsw:search_ef": 200,
            "hnsw:M": 32,
        },
    )

    # Chroma stores: id, embedding, the document text, and metadata.
    # Lists must be flattened to strings for metadata (Chroma allows only
    # str/int/float/bool), so we join lists with " | ".
    ids = [str(r["id"]) for r in records]
    metadatas = [
        {
            "name": r["name"],
            "uses": r["uses"],
            "composition": r["composition"],
            "side_effects": " | ".join(r["side_effects"]),
            "substitutes": " | ".join(r["substitutes"]),
        }
        for r in records
    ]

    # Add in batches (Chroma has a per-call cap).
    BATCH = 1000
    for i in range(0, len(records), BATCH):
        collection.add(
            ids=ids[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH],
            documents=texts[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
        )
        print(f"  added {min(i + BATCH, len(records))}/{len(records)}")

    print(f"Done. ChromaDB collection '{config.CHROMA_COLLECTION}' "
          f"has {collection.count()} vectors -> {config.CHROMA_DIR}")


if __name__ == "__main__":
    build_store()
