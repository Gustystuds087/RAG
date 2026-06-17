"""MedGraphy-style hybrid RAG app (ChromaDB + Neo4j + Groq).

Run locally:
    streamlit run app.py
"""
import os

import streamlit as st

# set_page_config MUST be the very first Streamlit command.
st.set_page_config(page_title="MedGraphy Clone", page_icon="💊", layout="wide")

# --- Bridge Streamlit Cloud secrets into env BEFORE importing config ---
# On Streamlit Cloud, secrets live in st.secrets. config.py reads os.environ at
# import time, so we copy any secrets into the environment first. Locally this
# is a no-op (st.secrets is empty) and .env is used instead.
try:
    if hasattr(st, "secrets"):
        for _k, _v in st.secrets.items():
            os.environ[_k] = str(_v)   # direct set (not setdefault) to win over stale env
except Exception:
    pass  # no secrets file locally — fine

from src.rag_engine import RagEngine
from src import config
from src.setup_data import ensure_chroma


@st.cache_resource(show_spinner="Loading models, vector store, and graph...")
def get_engine(cache_key: str):
    # Download the Chroma store if missing (cloud). cache_key (= CHROMA_VERSION)
    # busts this cache when bumped, so a stale engine is never reused.
    ensure_chroma()
    return RagEngine()


def main():
    st.title("💊 MedGraphy Clone — Hybrid RAG for Medicines")
    st.caption("ChromaDB (semantic) + Neo4j (graph) + Groq (LLM)")
    st.info(
        "⚠️ **Demo project — not a medical device.** Information here is from a "
        "public dataset, may be incomplete or wrong, and is **not medical advice.** "
        "Always consult a qualified doctor or pharmacist.",
        icon="⚠️",
    )

    engine = get_engine(config.CHROMA_VERSION)

    with st.sidebar:
        st.header("Settings")
        mode = st.radio(
            "Query mode",
            ["General Q&A", "Medicine lookup", "Find by condition", "Side effects"],
        )
        top_k = st.slider("Results to retrieve (k)", 1, 10, 5)
        st.divider()
        st.write("**Neo4j:**", "✅ connected" if engine.graph else "⚠️ vector-only")
        st.write("**Groq:**", "✅ set" if engine.llm else "⚠️ no key")

    # Tailor the prompt to the selected mode.
    placeholder = {
        "General Q&A": "What can I take for a headache?",
        "Medicine lookup": "Paracetamol",
        "Find by condition": "diabetes",
        "Side effects": "Ibuprofen",
    }[mode]

    query = st.text_input("Your question", placeholder=placeholder)
    if not query:
        return

    if mode == "Medicine lookup":
        question = f"Give detailed information about the medicine '{query}'."
    elif mode == "Find by condition":
        question = f"Which medicines are used to treat '{query}'? List them."
    elif mode == "Side effects":
        question = f"What are the side effects of '{query}'?"
    else:
        question = query

    with st.spinner("Retrieving and generating..."):
        result = engine.answer(question, k=top_k)

    # A guardrail blocked the query — show a warning, no sources/context.
    if result.get("blocked"):
        st.warning(result["answer"])
        st.caption(f"🛡️ Blocked by guardrail: {result.get('reason', '')}")
        return

    st.subheader("Answer")
    st.write(result["answer"])

    with st.expander("🔎 Retrieved medicines (sources)"):
        for h in result["sources"]:
            st.markdown(
                f"**{h['name']}** — similarity `{h['score']:.3f}`  \n"
                f"Uses: {h.get('uses') or '—'}"
            )

    with st.expander("🧩 Raw context sent to the LLM"):
        st.code(result["context"])


if __name__ == "__main__":
    main()
