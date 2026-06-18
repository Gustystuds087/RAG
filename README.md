# MediSage — Your Medicine Knowledge Assistant

An **agentic GraphRAG** app that answers questions about medicines — their uses,
side effects, conditions, and substitutes. It combines a **knowledge graph +
vector search (both in Neo4j)** with **LLM-generated Cypher queries** and
guardrails for safety.

> ⚠️ Educational/demo project. Output is **not medical advice.**

---

## What makes it different

Instead of a fixed retrieval pipeline, **the LLM writes the database query**.
For each question, Gemini generates a single Cypher query that combines
**vector (semantic) search** with **graph traversal** — then a safety layer
validates it is read-only before it runs.

Everything lives in **Neo4j** (graph **and** embeddings) — there is no separate
vector store to build, ship, or sync.

---

## Architecture

```
USER QUESTION
   │
   ▼  [1] GUARDRAILS                rules (instant) + Groq classifier
   │        emergency / self-harm / harm-others / hate / injection / off-topic
   │        → blocked here if unsafe
   ▼
   ▼  [2] EMBED the question        Sentence-Transformers (all-mpnet) → $queryVector
   │
   ▼  [3] GEMINI GENERATES CYPHER   given the graph schema, writes a query that
   │        combines vector search + graph hops:
   │          CALL db.index.vector.queryNodes('medicine_embeddings', $k, $queryVector)
   │          YIELD node AS m, score
   │          OPTIONAL MATCH (m)-[:TREATS]->(c) ...
   │
   ▼  [4] VALIDATE read-only        reject CREATE/DELETE/MERGE/SET/DROP/...
   │        → retry once → else FALL BACK to a safe plain vector search
   ▼
   ▼  [5] RUN ON NEO4J              vector search + graph traversal, one query
   │
   ▼  [6] LOW-CONFIDENCE GUARD      nothing relevant? → refuse instead of guessing
   │
   ▼  [7] GEMINI WRITES THE ANSWER  grounded ONLY in retrieved context + disclaimer
   │
   ▼  ANSWER  (shown in Streamlit with sources + raw-context expanders)
```

### Two-LLM split (free-tier friendly)
| Task | Provider | Why |
|------|----------|-----|
| Guardrail classification | **Groq** (Llama) | fast, simple, separate quota |
| Cypher generation + answers | **Gemini 2.0 Flash** | needs reasoning/quality |

Each provider **falls back to the other** if one is unavailable or rate-limited.

---

## Tech stack
| Layer | Tool |
|-------|------|
| Graph **and** vectors | **Neo4j** (native vector index) |
| Embeddings | Sentence-Transformers (`all-mpnet-base-v2`) |
| Query generation + answers | **Gemini** (`gemini-2.0-flash`) |
| Guardrail classifier | **Groq** (`llama-3.3-70b-versatile`) |
| UI | Streamlit |

---

## Setup

### 1. Install dependencies
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure credentials
Copy `.env.example` to `.env` and fill in:
- **Neo4j** — free instance at https://neo4j.com/cloud/aura-free/ (URI, user, password, database)
- **Gemini key** — free at https://aistudio.google.com/apikey
- **Groq key** — free at https://console.groq.com/keys (used for guardrails)

### 3. Get data
Download the Kaggle "250k Medicines Usage, Side Effects and Substitutes" dataset,
save it as `data/medicine_dataset.csv` (or use the included
`data/sample_medicines.csv`).

### 4. Load Neo4j (graph + embeddings) — run once
```powershell
python -m src.graph_loader
```
This builds the graph, embeds every medicine onto its node, and creates the
`medicine_embeddings` vector index. (~5 min for 5k medicines.)

### 5. Run
```powershell
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
    ├── graph_loader.py     # load Neo4j graph + embeddings + vector index
    ├── llm.py              # Gemini (primary) + Groq (fast) with fallback
    ├── cypher_agent.py     # schema + LLM Cypher generation + read-only validation
    ├── guardrails.py       # input safety (emergency/crisis/harm/hate/injection/…)
    ├── rag_engine.py       # the full agentic flow
    └── query_logger.py     # logs every query + Cypher to logs/
```

---

## Safety

- **Input guardrails** — medical emergencies route to help (not medicines),
  self-harm → crisis lines, threats/hate/profanity refused, off-topic redirected.
- **Prompt-injection defense** — hardened system prompts + a read-only Cypher
  validator: any generated query with `CREATE/DELETE/MERGE/SET/DROP/...` is
  rejected before it runs.
- **Grounding** — the answer LLM may use ONLY the retrieved context; if a fact
  (e.g. a dosage) isn't present, it says so instead of inventing one.
- **Disclaimer** — every answer carries a "not medical advice" notice.

## Deployment notes
- Vectors + graph both live in Neo4j → **nothing to upload** beyond the code;
  the deployed app just connects to Neo4j Aura.
- On Streamlit Cloud: set Python to **3.11** and put all keys in **Secrets**
  (TOML format, every value quoted).
- ⚠️ This is an educational project and **not** a medical device.
```
