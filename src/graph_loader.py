"""Load medicine records into Neo4j as a knowledge graph.

Graph model:
    (:Medicine {name})
    (:Condition {name})    <-[:TREATS]-          (Medicine)
    (:Ingredient {name})   <-[:CONTAINS]-         (Medicine)
    (:SideEffect {name})   <-[:HAS_SIDE_EFFECT]-  (Medicine)
    (Medicine) -[:SUBSTITUTE_FOR]-> (Medicine)

Each Medicine node also gets an `embedding` property (vector of its doc_text),
plus a native Neo4j VECTOR INDEX so the app can do semantic search inside Neo4j
— no separate vector store needed.

Run once after building data:
    python -m src.graph_loader
"""
import re

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from . import config
from .data_loader import load_records


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


class GraphLoader:
    def __init__(self):
        if not config.NEO4J_URI or not config.NEO4J_PASSWORD:
            raise RuntimeError("NEO4J_URI / NEO4J_PASSWORD not set in .env")
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def setup_constraints(self):
        stmts = [
            "CREATE CONSTRAINT med_name IF NOT EXISTS FOR (m:Medicine) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT cond_name IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT ingr_name IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE",
            "CREATE CONSTRAINT se_name IF NOT EXISTS FOR (s:SideEffect) REQUIRE s.name IS UNIQUE",
        ]
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            for s in stmts:
                session.run(s)

    def wipe(self):
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def load(self, records: list[dict], batch_size: int = 500):
        # Each record carries normalized lists; we let Cypher UNWIND do the work.
        rows = []
        for r in records:
            rows.append({
                "name": r["name"],
                "uses": [_norm(u) for u in re.split(r"[|,;]", r["uses"]) if u.strip()],
                "side_effects": [_norm(s) for s in r["side_effects"] if s.strip()],
                "substitutes": [s.strip() for s in r["substitutes"] if s.strip()],
                "composition": [_norm(c) for c in re.split(r"[|,;+]", r["composition"]) if c.strip()],
            })

        cypher = """
        UNWIND $rows AS row
        MERGE (m:Medicine {name: row.name})
        WITH m, row
        FOREACH (u IN row.uses |
            MERGE (c:Condition {name: u})
            MERGE (m)-[:TREATS]->(c))
        FOREACH (s IN row.side_effects |
            MERGE (se:SideEffect {name: s})
            MERGE (m)-[:HAS_SIDE_EFFECT]->(se))
        FOREACH (i IN row.composition |
            MERGE (ing:Ingredient {name: i})
            MERGE (m)-[:CONTAINS]->(ing))
        FOREACH (sub IN row.substitutes |
            MERGE (m2:Medicine {name: sub})
            MERGE (m)-[:SUBSTITUTE_FOR]->(m2))
        """
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                session.run(cypher, rows=batch)
                print(f"  loaded {min(i + batch_size, len(rows))}/{len(rows)}")

    def add_embeddings(self, records: list[dict], dim: int, batch_size: int = 200):
        """Embed each medicine's doc_text and store it on its Medicine node."""
        model = SentenceTransformer(config.EMBED_MODEL)
        texts = [r["doc_text"] for r in records]
        print(f"Embedding {len(records)} medicines (dim={dim})...")
        embs = model.encode(texts, batch_size=64, show_progress_bar=True,
                            normalize_embeddings=True).tolist()

        rows = [{"name": r["name"], "embedding": e} for r, e in zip(records, embs)]
        cypher = """
        UNWIND $rows AS row
        MATCH (m:Medicine {name: row.name})
        CALL db.create.setNodeVectorProperty(m, 'embedding', row.embedding)
        """
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            for i in range(0, len(rows), batch_size):
                session.run(cypher, rows=rows[i:i + batch_size])
                print(f"  embedded {min(i + batch_size, len(rows))}/{len(rows)}")

    def create_vector_index(self, dim: int):
        cypher = f"""
        CREATE VECTOR INDEX medicine_embeddings IF NOT EXISTS
        FOR (m:Medicine) ON (m.embedding)
        OPTIONS {{ indexConfig: {{
            `vector.dimensions`: {dim},
            `vector.similarity_function`: 'cosine'
        }} }}
        """
        with self.driver.session(database=config.NEO4J_DATABASE) as session:
            session.run(cypher)
        print("Vector index 'medicine_embeddings' created.")


def main():
    records = load_records()
    # dim from the embedding model (mpnet = 768)
    dim = SentenceTransformer(config.EMBED_MODEL).get_sentence_embedding_dimension()

    loader = GraphLoader()
    try:
        print("Setting up constraints...")
        loader.setup_constraints()
        print("Wiping existing graph...")
        loader.wipe()
        print(f"Loading {len(records)} medicines into Neo4j...")
        loader.load(records)
        print("Adding embeddings to Medicine nodes...")
        loader.add_embeddings(records, dim=dim)
        print("Creating vector index...")
        loader.create_vector_index(dim=dim)
        print("Done. Graph + vectors loaded.")
    finally:
        loader.close()


if __name__ == "__main__":
    main()
