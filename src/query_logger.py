"""Logging for the RAG engine.

Two outputs:
  - logs/queries.log     -> human-readable running log (every query + retrieval)
  - logs/queries.jsonl   -> one JSON object per query (easy to analyze later)

Each query records: timestamp, question, retrieved medicines + similarity
scores, whether the graph was used, and the generated answer.
"""
import json
import logging
import os
from datetime import datetime, timezone

LOG_DIR = "logs"
TEXT_LOG = os.path.join(LOG_DIR, "queries.log")
JSONL_LOG = os.path.join(LOG_DIR, "queries.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)

# Console + file text logger.
logger = logging.getLogger("medgraphy")
if not logger.handlers:  # avoid duplicate handlers on Streamlit reruns
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(TEXT_LOG, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)


def log_cypher(cypher: str, params: dict, result_summary: str = ""):
    """Log a Cypher (CQL) query sent to Neo4j, its parameters, and a result note."""
    one_line = " ".join(cypher.split())  # collapse whitespace for a tidy log line
    logger.info("CYPHER params=%s | %s", params, one_line)
    if result_summary:
        logger.info("CYPHER result: %s", result_summary)


def log_query(question: str, hits: list[dict], graph_used: bool, answer: str):
    """Write one query's full lifecycle to both the text and JSONL logs."""
    sources = [
        {"name": h.get("name"), "score": round(float(h.get("score", 0)), 4)}
        for h in hits
    ]

    # Readable text log
    logger.info("QUESTION: %s", question)
    logger.info(
        "RETRIEVED (graph=%s): %s",
        graph_used,
        ", ".join(f"{s['name']} ({s['score']})" for s in sources),
    )
    logger.info("ANSWER: %s", answer.replace("\n", " ")[:500])
    logger.info("-" * 60)

    # Structured JSONL log (one line per query)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "graph_used": graph_used,
        "sources": sources,
        "answer": answer,
    }
    with open(JSONL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
