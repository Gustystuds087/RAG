"""MediSage — agentic GraphRAG app (Neo4j graph+vectors + Gemini + Groq).

Run locally:
    streamlit run app.py
"""
import os

import streamlit as st

# set_page_config MUST be the very first Streamlit command.
st.set_page_config(
    page_title="MediSage",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Bridge Streamlit Cloud secrets into env BEFORE importing config ---
try:
    if hasattr(st, "secrets"):
        for _k, _v in st.secrets.items():
            os.environ[_k] = str(_v)
except Exception:
    pass  # no secrets file locally — fine

from src.rag_engine import RagEngine


# ─────────────────────────────────────────────────────────────────────────────
# Theme — dark + neon, with subtle animations (injected CSS)
# ─────────────────────────────────────────────────────────────────────────────
NEON = "#00e5ff"        # cyan
NEON2 = "#a855f7"       # purple
NEON3 = "#22ff88"       # green

CSS = """
<style>
/* ---- base dark canvas ---- */
.stApp {
    background: radial-gradient(circle at 20% 0%, #11142a 0%, #0a0b14 45%, #060710 100%);
    color: #e6e9ff;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1024 0%, #0a0b16 100%);
    border-right: 1px solid rgba(0,229,255,0.15);
}

/* ---- animated neon title ---- */
.medi-hero {
    text-align: center;
    padding: 1.4rem 0 0.4rem 0;
}
.medi-title {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: 1px;
    background: linear-gradient(90deg, #00e5ff, #a855f7, #22ff88, #00e5ff);
    background-size: 300% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 6s linear infinite;
}
@keyframes shimmer { to { background-position: 300% center; } }

.medi-sub {
    text-align: center;
    color: #8b93c7;
    font-size: 0.95rem;
    margin-top: -0.2rem;
}
.medi-pill {
    display: inline-block;
    margin-top: 0.6rem;
    padding: 0.25rem 0.9rem;
    border-radius: 999px;
    font-size: 0.78rem;
    color: #00e5ff;
    border: 1px solid rgba(0,229,255,0.4);
    background: rgba(0,229,255,0.06);
    box-shadow: 0 0 18px rgba(0,229,255,0.25);
    animation: pulse 2.8s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { box-shadow: 0 0 10px rgba(0,229,255,0.18); }
    50%     { box-shadow: 0 0 26px rgba(0,229,255,0.45); }
}

/* ---- input box glow ---- */
.stTextInput > div > div input {
    background: rgba(255,255,255,0.03) !important;
    color: #e6e9ff !important;
    border: 1px solid rgba(168,85,247,0.4) !important;
    border-radius: 12px !important;
}
.stTextInput > div > div input:focus {
    border-color: #00e5ff !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.25), 0 0 18px rgba(0,229,255,0.3) !important;
}

/* ---- answer card ---- */
.answer-card {
    background: rgba(168,85,247,0.06);
    border: 1px solid rgba(168,85,247,0.35);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 0 30px rgba(168,85,247,0.18);
    animation: rise 0.5s ease;
}
@keyframes rise { from { opacity:0; transform: translateY(12px);} to {opacity:1; transform:none;} }

/* ---- source chips ---- */
.src-chip {
    display:inline-block; margin:4px 6px 0 0; padding:6px 12px;
    border-radius:10px; font-size:0.85rem;
    background: rgba(34,255,136,0.07);
    border:1px solid rgba(34,255,136,0.35);
    color:#bafce0;
}
.src-score { color:#22ff88; font-weight:700; }

/* ---- status dots ---- */
.dot-ok  { color:#22ff88; }
.dot-off { color:#ff5c7c; }

/* expander + buttons */
.stExpander { border:1px solid rgba(0,229,255,0.15) !important; border-radius:12px !important; }

/* ---- thinking timeline ---- */
.tl-title { color:#00e5ff; font-weight:700; font-size:0.95rem; margin-bottom:0.6rem; }
.tl-step {
    position:relative; padding:8px 0 8px 28px; border-left:2px solid rgba(255,255,255,0.08);
    margin-left:8px;
}
.tl-dot {
    position:absolute; left:-9px; top:12px; width:16px; height:16px; border-radius:50%;
    background:#1a1d33; border:2px solid #2a2e4a;
}
.tl-run  .tl-dot { border-color:#00e5ff; background:#06283a;
    box-shadow:0 0 14px rgba(0,229,255,0.7); animation:blink 1s ease-in-out infinite; }
.tl-done .tl-dot { border-color:#22ff88; background:#0c2a1c; box-shadow:0 0 10px rgba(34,255,136,0.5); }
.tl-warn .tl-dot { border-color:#ffb020; background:#2a210c; box-shadow:0 0 10px rgba(255,176,32,0.5); }
.tl-block .tl-dot{ border-color:#ff5c7c; background:#2a0c14; box-shadow:0 0 10px rgba(255,92,124,0.5); }
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.45;} }
.tl-label { font-weight:600; color:#e6e9ff; font-size:0.9rem; }
.tl-run  .tl-label { color:#00e5ff; }
.tl-done .tl-label { color:#bafce0; }
.tl-detail { color:#8b93c7; font-size:0.78rem; margin-top:2px; word-break:break-word;
    max-height:3.2rem; overflow:hidden; }
</style>
"""

# the fixed pipeline stages we display in the timeline (in order)
TIMELINE_STEPS = [
    ("guardrails", "Safety check"),
    ("memory", "Use recent context"),
    ("embed", "Embed question"),
    ("generate", "Generate Cypher"),
    ("validate", "Validate (read-only)"),
    ("run", "Run on Neo4j"),
    ("fallback", "Fallback search"),
    ("nodes", "Relevant nodes"),
    ("confidence", "Confidence check"),
    ("answer", "Write answer"),
]


def render_timeline(state: dict) -> str:
    """state: key -> (label, status, detail). Build the timeline HTML."""
    html = "<div class='tl-title'>🧠 Thinking…</div>"
    for key, default_label in TIMELINE_STEPS:
        if key not in state:
            continue  # only show stages that actually fired
        label, status, detail = state[key]
        cls = {"run": "tl-run", "done": "tl-done", "warn": "tl-warn",
               "block": "tl-block"}.get(status, "")
        det = (detail or "")[:120]
        html += (f"<div class='tl-step {cls}'><div class='tl-dot'></div>"
                 f"<div class='tl-label'>{label}</div>"
                 f"<div class='tl-detail'>{det}</div></div>")
    return html


@st.cache_resource(show_spinner="Loading models and connecting to the graph...")
def get_engine():
    return RagEngine()


def main():
    st.markdown(CSS, unsafe_allow_html=True)

    # ---- animated hero header ----
    st.markdown(
        """
        <div class="medi-hero">
            <div class="medi-title">💊 MediSage</div>
            <div class="medi-sub">Your Medicine Knowledge Assistant — Neo4j GraphRAG · Gemini · Groq</div>
            <div class="medi-pill">⚡ agentic graph + vector retrieval</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "⚠️ **Demo project — not a medical device.** Information is from a public "
        "dataset, may be incomplete or wrong, and is **not medical advice.** "
        "Always consult a qualified doctor or pharmacist.",
        icon="⚠️",
    )

    engine = get_engine()

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        mode = st.radio(
            "Query mode",
            ["General Q&A", "Medicine lookup", "Find by condition", "Side effects"],
        )
        top_k = st.slider("Results to retrieve (k)", 1, 10, 5)
        st.divider()
        st.markdown("### 🔌 Status")
        neo = ("<span class='dot-ok'>● connected</span>" if engine.graph
               else "<span class='dot-off'>● offline</span>")
        llm = ("<span class='dot-ok'>● ready</span>" if engine.llm_ready
               else "<span class='dot-off'>● no key</span>")
        st.markdown(f"**Neo4j** &nbsp; {neo}", unsafe_allow_html=True)
        st.markdown(f"**LLM** &nbsp;&nbsp;&nbsp; {llm}", unsafe_allow_html=True)

        # ---- recent conversation memory (last 5, per-session) ----
        hist = st.session_state.get("history", [])
        if hist:
            st.divider()
            st.markdown("### 🕘 Recent")
            st.caption("last 5 turns — used as follow-up context")
            for h in reversed(hist):
                st.markdown(f"- **{h['q'][:40]}**")
            if st.button("🗑️ Clear memory"):
                st.session_state["history"] = []
                st.rerun()

    placeholder = {
        "General Q&A": "What can I take for a headache?",
        "Medicine lookup": "Paracetamol",
        "Find by condition": "diabetes",
        "Side effects": "Ibuprofen",
    }[mode]

    query = st.text_input("💬 Ask about a medicine", placeholder=placeholder)
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

    # ---- two columns: timeline (left) + answer (right) ----
    left, right = st.columns([1, 2], gap="large")
    timeline_box = left.empty()
    answer_box = right.empty()

    # live state — callback updates it and re-renders the animated timeline.
    # We keep the FULL detail (untruncated) AND the order steps fired, so we can
    # show clickable expanders afterwards.
    state: dict = {}
    order: list = []

    def on_step(key, label, status, detail):
        if key not in state:
            order.append(key)
        state[key] = (label, status, detail)
        timeline_box.markdown(render_timeline(state), unsafe_allow_html=True)

    answer_box.markdown(
        "<div class='answer-card' style='opacity:0.6;'>⏳ working on it…</div>",
        unsafe_allow_html=True,
    )
    # pass the last 5 turns as short conversation memory
    history = st.session_state.get("history", [])
    result = engine.answer(question, k=top_k, step=on_step, history=history)

    # ---- after the run: replace the animated timeline with CLICKABLE steps ----
    timeline_box.empty()
    with left:
        st.markdown("<div class='tl-title'>🧠 Reasoning steps</div>", unsafe_allow_html=True)
        st.caption("click a step to see its full detail")
        icon = {"done": "🟢", "warn": "🟡", "block": "🔴", "run": "🔵"}
        for key in order:
            label, status, detail = state[key]
            # auto-open the "Relevant nodes" step so the searched medicines show
            with st.expander(f"{icon.get(status,'⚪')} {label}", expanded=(key == "nodes")):
                if key == "generate" and detail.strip().upper().startswith("CALL"):
                    st.code(detail, language="cypher")        # full Cypher
                elif key == "nodes":
                    for line in (detail or "").split("\n"):
                        st.markdown(f"- {line}")               # one node per line
                else:
                    st.write(detail or "—")

    # ---- a guardrail blocked the query ----
    if result.get("blocked"):
        answer_box.markdown(
            f"<div class='answer-card' style='border-color:rgba(255,92,124,0.5);"
            f"box-shadow:0 0 30px rgba(255,92,124,0.18);'>{result['answer']}</div>",
            unsafe_allow_html=True,
        )
        right.caption(f"🛡️ Blocked by guardrail: `{result.get('reason', '')}`")
        return

    # ---- answer card (right column) ----
    mode_tag = result.get("mode", "")
    answer_html = "<div class='answer-card'>" + result["answer"].replace("\n", "<br>") + "</div>"
    answer_box.markdown(answer_html, unsafe_allow_html=True)
    if mode_tag:
        right.caption(f"retrieval mode: `{mode_tag}`")

    if result.get("sources"):
        chips = "".join(
            f"<span class='src-chip'>{h['name']} "
            f"<span class='src-score'>{h.get('score', 0.0):.2f}</span></span>"
            for h in result["sources"]
        )
        right.markdown("**🔎 Retrieved medicines**", unsafe_allow_html=True)
        right.markdown(chips, unsafe_allow_html=True)

    with right.expander("⚙️ Generated Cypher query"):
        st.code(result.get("cypher") or "(used fallback vector search — no LLM Cypher)", language="cypher")
    with right.expander("🧩 Raw context sent to the LLM"):
        st.code(result.get("context", ""))

    # ---- save this turn into per-session memory (keep only the last 5) ----
    # store the exact medicines retrieved, so "side effects of these" reuses them
    hist = st.session_state.get("history", [])
    hist.append({"q": question, "a": result["answer"], "meds": result.get("meds", [])})
    st.session_state["history"] = hist[-5:]


if __name__ == "__main__":
    main()