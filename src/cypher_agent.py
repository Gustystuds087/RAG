"""Agentic retrieval: the LLM writes Cypher (vector search + graph hops) from
the graph schema, we validate it is READ-ONLY, run it, and on any failure fall
back to a plain vector search.

Safety: generated Cypher is rejected if it contains any write/admin keyword.
The query is also run under a read-only-intended session with a timeout.
"""
import re

from . import config
from . import llm

# ---- read-only validation ----
_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL\s+apoc\.\w+\.(create|delete)|"
    r"LOAD\s+CSV|FOREACH|CALL\s+db\.create|CALL\s+dbms)\b",
    re.IGNORECASE,
)
# the only db.* call we allow is the vector query
_ALLOWED_PROCS = ("db.index.vector.queryNodes",)


def is_read_only(cypher: str) -> bool:
    if _FORBIDDEN.search(cypher):
        return False
    # if it calls a procedure, it must be the allowed vector one
    for m in re.finditer(r"CALL\s+([\w.]+)", cypher, re.IGNORECASE):
        if not m.group(1).startswith(_ALLOWED_PROCS):
            return False
    return True


# ---- schema given to the LLM ----
SCHEMA = """
Graph schema (Neo4j):
  (:Medicine {name, embedding})       -- embedding is a vector; use the vector index
  (:Condition {name})
  (:SideEffect {name})
  (:Ingredient {name})
Relationships:
  (Medicine)-[:TREATS]->(Condition)
  (Medicine)-[:HAS_SIDE_EFFECT]->(SideEffect)
  (Medicine)-[:CONTAINS]->(Ingredient)
  (Medicine)-[:SUBSTITUTE_FOR]->(Medicine)

Vector search (for fuzzy / symptom / "something for X" questions) — EXAMPLE,
copy this shape exactly and adjust the traversal:

  CALL db.index.vector.queryNodes('medicine_embeddings', $k, $queryVector)
  YIELD node AS m, score
  OPTIONAL MATCH (m)-[:TREATS]->(c:Condition)
  OPTIONAL MATCH (m)-[:HAS_SIDE_EFFECT]->(se:SideEffect)
  OPTIONAL MATCH (m)-[:SUBSTITUTE_FOR]->(sub:Medicine)
  RETURN m.name AS name, score,
         collect(DISTINCT c.name) AS conditions,
         collect(DISTINCT se.name) AS side_effects,
         collect(DISTINCT sub.name) AS substitutes
  ORDER BY score DESC

Notes:
  - The index name is the literal string 'medicine_embeddings'.
  - Condition/SideEffect/Ingredient names are stored lowercased.
  - $queryVector and $k are provided as parameters — use them by those names.
  - Keep the query short and COMPLETE. Return ONLY the query text.
"""

_SYSTEM = (
    "You write a SINGLE read-only Cypher query for a medicines knowledge graph. "
    "Use the vector index for fuzzy/symptom questions (the user's embedded query "
    "is the $queryVector parameter, $k is the number of results). Use plain "
    "MATCH for exact/structured questions. You MAY combine vector search with "
    "graph traversal. Return ONLY the Cypher query, no markdown, no explanation. "
    "NEVER write to the database (no CREATE/MERGE/DELETE/SET/etc.)."
)


def generate_cypher(question: str) -> str:
    user = f"{SCHEMA}\n\nUser question: {question}\n\nCypher:"
    raw = llm.complete(_SYSTEM, user, temperature=0.0, max_tokens=1024)
    # strip markdown fences if the model added them
    raw = re.sub(r"```(?:cypher)?", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("```", "").strip()
    return raw
