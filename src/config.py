"""Central configuration. Reads from .env and defines the dataset schema."""
import os
from dotenv import load_dotenv

load_dotenv()

def _real(value: str) -> str:
    """Treat the .env.example placeholder values as 'not set'."""
    v = (value or "").strip()
    if not v or v.startswith(("your", "gsk_your", "neo4j+s://xxxx")):
        return ""
    return v


# ---- Neo4j ----
NEO4J_URI = _real(os.getenv("NEO4J_URI", ""))
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = _real(os.getenv("NEO4J_PASSWORD", ""))
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ---- Gemini (primary LLM) ----
GEMINI_API_KEY = _real(os.getenv("GEMINI_API_KEY", ""))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ---- Groq (legacy / fallback) ----
GROQ_API_KEY = _real(os.getenv("GROQ_API_KEY", ""))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---- Embeddings ----
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# ---- Paths ----
DATA_CSV = os.getenv("DATA_CSV", "data/medicines.csv")

# ---- ChromaDB (vector store; replaces FAISS) ----
# Chroma persists to a local folder — no server needed.
CHROMA_DIR = os.getenv("CHROMA_DIR", "artifacts/chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "medicines")

# URL to a zipped Chroma folder, used to bootstrap the store on cloud deploys
# where the folder isn't in the repo. Leave empty for local dev.
CHROMA_URL = os.getenv("CHROMA_URL", "")

# Bump this (via env/secret) whenever you re-upload a new Chroma zip, to force
# the cloud app to discard its cached store and re-download the new one.
CHROMA_VERSION = os.getenv("CHROMA_VERSION", "1")

# ---- Dataset schema ----
# The Kaggle "medicine dataset" columns we care about. The loader is tolerant:
# if a column is missing it is simply skipped.
COL_NAME = "name"
COL_USES = "uses"            # what conditions it treats
COL_SIDE_EFFECTS = "side_effects"
COL_SUBSTITUTES = "substitutes"  # alternative medicines
COL_COMPOSITION = "composition"  # active ingredients

# How many medicines to load (the Kaggle file has ~250k rows; cap for a demo).
MAX_ROWS = int(os.getenv("MAX_ROWS", "5000"))
