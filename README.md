# MedGraphy Clone — Hybrid RAG for Medicines

An intelligent drug-information app combining a **graph database** (Neo4j) and
a **vector database** (ChromaDB) with a **Groq LLM** to answer medical questions
about medicines, interactions, conditions, and side effects.

This is a hybrid **RAG** (Retrieval-Augmented Generation) pipeline:

```
question ──▶ ChromaDB (semantic search) ──▶ top-k medicines
                                               │
                                               ▼
                                    Neo4j (graph expansion:
                                    conditions, side effects,
                                    ingredients, substitutes)
                                               │
                                               ▼
                                context ──▶ Groq LLM ──▶ grounded answer
```

## Tech stack
| Layer | Tool |
|-------|------|
| Vector search | ChromaDB + Sentence-Transformers (`all-mpnet-base-v2`) |
| Knowledge graph | Neo4j |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| UI | Streamlit |

---

## Setup

### 1. Install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
pip install -r requirements.txt
```

### 2. Configure credentials
Copy `.env.example` to `.env` and fill in:

```bash
copy .env.example .env
```

- **Groq key** — free at https://console.groq.com/keys
- **Neo4j** — free cloud instance at https://neo4j.com/cloud/aura-free/
  (create a database, download/copy the URI + password)

ChromaDB needs **no account** — it runs locally and saves to `artifacts/chroma/`.

### 3. Get data
A small `data/sample_medicines.csv` is included so you can run immediately. For
the full experience, download the Kaggle "250k Medicines Usage, Side Effects and
Substitutes" dataset, save it as `data/medicine_dataset.csv`, and set
`DATA_CSV=data/medicine_dataset.csv` in `.env`.

### 4. Build the vector store and graph
```bash
python -m src.build_vectors   # embeds medicines -> ChromaDB (local)
python -m src.graph_loader     # loads the Neo4j knowledge graph (optional)
```

### 5. Run
```bash
python -m streamlit run app.py
```

---

## Project structure
```
RAG/
├── app.py                  # Streamlit UI
├── requirements.txt
├── .env.example
├── data/
│   └── sample_medicines.csv
└── src/
    ├── config.py           # env + schema
    ├── data_loader.py      # CSV -> clean records (samples across full dataset)
    ├── build_vectors.py    # embed + store in ChromaDB
    ├── graph_loader.py     # load Neo4j graph
    ├── rag_engine.py       # ChromaDB + Neo4j + Groq pipeline
    └── query_logger.py     # logs every query + Cypher to logs/
```

## Notes
- The app runs **vector-only** if Neo4j is unavailable, and shows
  **retrieved context only** if no Groq key is set — so you can build up
  incrementally.
- ChromaDB uses an approximate (HNSW) index; we tune `hnsw:search_ef`/`M` in
  `build_vectors.py` so results are accurate at this dataset size.
- ⚠️ This is an educational project. Output is **not medical advice**.
```
