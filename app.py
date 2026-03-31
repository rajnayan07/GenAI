"""
GitLab Handbook AI Assistant — Streamlit Application

A RAG-powered chatbot that helps users explore GitLab's Handbook
and Direction pages through natural language conversation.

Stack: sentence-transformers (local embeddings) + Groq/Llama 3.3 (LLM)
"""

import sys
import os
import json
import logging
from datetime import datetime

from dotenv import load_dotenv
import streamlit as st

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")

from core.indexer import build_or_load_index
from core.retriever import retrieve, format_context, get_source_citations
from core.chatbot import create_chat_client, generate_response_stream
from core.guardrails import validate_input, check_relevance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GitLab Handbook AI Assistant",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — enhanced with animations, dark mode, feedback, polish
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ---- Base layout ---- */
    .main .block-container { max-width: 920px; padding-top: 1.5rem; }

    /* ---- Animated header ---- */
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(-12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .app-header {
        display: flex; align-items: center; gap: 18px;
        padding: 1.2rem 0; border-bottom: 2px solid #e5e0f1;
        margin-bottom: 1.2rem; animation: fadeSlideIn 0.6s ease-out;
    }
    .app-header .logo { font-size: 2.8rem; line-height: 1; }
    .app-header .title-block h1 {
        margin: 0; font-size: 1.7rem; font-weight: 700;
    }
    .app-header .title-block p {
        margin: 4px 0 0 0; font-size: 0.88rem; opacity: 0.65;
    }
    .app-header .live-badge {
        display: inline-block; margin-left: 10px;
        background: #22c55e; color: white; font-size: 0.65rem;
        padding: 2px 8px; border-radius: 10px; font-weight: 600;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    /* ---- Chat messages ---- */
    .stChatMessage { border-radius: 14px !important; margin-bottom: 0.5rem !important; }

    /* ---- Source citation cards ---- */
    .source-card {
        background: rgba(107,79,187,0.1);
        border-left: 4px solid #a78bfa; border-radius: 0 10px 10px 0;
        padding: 12px 16px; margin: 6px 0; font-size: 0.85rem;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .source-card:hover { transform: translateX(4px); box-shadow: 0 2px 8px rgba(107,79,187,0.2); }
    .source-card a { color: #a78bfa; font-weight: 600; text-decoration: none; }
    .source-card a:hover { text-decoration: underline; }
    .source-card .section-tag {
        background: #6b4fbb; color: white; padding: 2px 8px;
        border-radius: 10px; font-size: 0.72rem; margin-left: 8px;
    }
    .source-card .relevance { opacity: 0.6; font-size: 0.78rem; }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f6fc 0%, #ffffff 100%);
    }
    @media (prefers-color-scheme: dark) {
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e1b2e 0%, #252236 100%);
        }
    }
    [data-theme="dark"] [data-testid="stSidebar"],
    .stApp[data-testid="stAppViewContainer"][class*="dark"] [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b2e 0%, #252236 100%);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: inherit;
    }

    /* ---- Badges ---- */
    .transparency-badge {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-radius: 10px; padding: 14px 16px; margin: 8px 0;
        font-size: 0.82rem; color: #2e7d32 !important; border: 1px solid #a5d6a7;
    }
    .transparency-badge strong { color: #1b5e20 !important; }

    .info-box {
        background: #f0f4ff; border-radius: 10px; padding: 16px;
        margin: 12px 0; border: 1px solid #c5d5ff; font-size: 0.85rem;
        color: #333 !important;
    }
    .info-box a { color: #6b4fbb !important; }

    /* ---- Metrics row ---- */
    .metrics-row { display: flex; gap: 10px; margin: 12px 0; }
    .metric-card {
        flex: 1; background: rgba(107,79,187,0.08); border: 1px solid rgba(107,79,187,0.2);
        border-radius: 10px; padding: 14px 10px; text-align: center;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(107,79,187,0.15); }
    .metric-card .value { font-size: 1.4rem; font-weight: 700; color: #a78bfa; }
    .metric-card .label { font-size: 0.72rem; opacity: 0.6; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }

    /* ---- Buttons ---- */
    div.stButton > button {
        border-radius: 22px; border: 1px solid rgba(107,79,187,0.35);
        background: rgba(107,79,187,0.1); color: #a78bfa; font-size: 0.84rem;
        padding: 10px 20px; transition: all 0.2s;
    }
    div.stButton > button:hover {
        background: #6b4fbb; color: white !important; border-color: #6b4fbb;
        transform: translateY(-1px); box-shadow: 0 4px 12px rgba(107,79,187,0.35);
    }

    /* ---- Welcome card ---- */
    @keyframes welcomeFade { from { opacity: 0; transform: scale(0.97); } to { opacity: 1; transform: scale(1); } }
    .welcome-card {
        background: linear-gradient(135deg, #6b4fbb 0%, #8b6fd6 50%, #a78bfa 100%);
        border-radius: 16px; padding: 28px 32px; color: white;
        margin-bottom: 20px; animation: welcomeFade 0.7s ease-out;
        box-shadow: 0 8px 24px rgba(107,79,187,0.3);
    }
    .welcome-card h2 { margin: 0 0 8px 0; font-size: 1.4rem; font-weight: 700; }
    .welcome-card p { margin: 0; font-size: 0.92rem; opacity: 0.92; line-height: 1.5; }
    .welcome-card .features {
        display: flex; gap: 16px; margin-top: 16px; flex-wrap: wrap;
    }
    .welcome-card .feat {
        background: rgba(255,255,255,0.15); border-radius: 10px;
        padding: 8px 14px; font-size: 0.8rem; backdrop-filter: blur(4px);
    }

    /* ---- Feedback buttons ---- */
    .feedback-row { display: flex; gap: 6px; margin-top: 8px; }
    .feedback-btn {
        background: rgba(107,79,187,0.1); border: 1px solid rgba(107,79,187,0.3); border-radius: 8px;
        padding: 4px 12px; cursor: pointer; font-size: 0.82rem;
        transition: all 0.15s; color: #a78bfa;
    }
    .feedback-btn:hover { background: #6b4fbb; color: white; border-color: #6b4fbb; }
    .feedback-done { background: rgba(34,197,94,0.15); color: #4ade80 !important; border: 1px solid rgba(34,197,94,0.3); font-size: 0.82rem; padding: 4px 12px; border-radius: 8px; }

    /* ---- Export button ---- */
    .export-container { text-align: center; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------
SUGGESTED_QUESTIONS = [
    ("What are GitLab's core values (CREDIT)?"),
    ("How does all-remote work at GitLab?"),
    ("What is the hiring process at GitLab?"),
    ("Explain GitLab's product direction and AI strategy"),
    ("How does GitLab handle security testing?"),
    ("What is the onboarding experience like?"),
    ("How are meetings conducted in an all-remote company?"),
    ("What is TeamOps?"),
]


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_session():
    defaults = {
        "messages": [],
        "index": None,
        "chunks": None,
        "client": None,
        "query_count": 0,
        "index_ready": False,
        "client_ready": False,
        "feedback": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ---------------------------------------------------------------------------
# API key handling (Groq)
# ---------------------------------------------------------------------------
def get_api_key() -> str | None:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    return st.session_state.get("user_api_key")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🦊 GitLab AI Assistant")
    st.markdown("---")

    api_key = get_api_key()
    if not api_key:
        st.markdown("#### API Configuration")
        user_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get a free key at [console.groq.com](https://console.groq.com/keys)",
        )
        if user_key:
            st.session_state.user_api_key = user_key
            api_key = user_key
        st.markdown(
            '<div class="info-box">'
            ' Get a free API key from <a href="https://console.groq.com/keys" target="_blank">'
            "Groq Console</a>.<br>No credit card needed — free tier includes "
            "30 req/min and 14,400 req/day.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

    st.markdown("#### ℹ️ About")
    st.markdown(
        "This chatbot uses **RAG** (Retrieval-Augmented Generation) to answer "
        "questions about GitLab's Handbook & Direction pages.\n\n"
        "**Embeddings** run locally and **generation** uses Llama 3.3 70B via Groq."
    )

    st.markdown("---")
    st.markdown("#### 🔍 How it works")
    st.markdown(
        "1.  Your question is embedded locally\n"
        "2.  Similar handbook sections retrieved via FAISS\n"
        "3.  Context + question sent to Llama 3.3\n"
        "4.  Grounded answer with source citations"
    )

    st.markdown("---")
    st.markdown(
        '<div class="transparency-badge">'
        "<strong> Transparency & Safety</strong><br>"
        "• Answers grounded in GitLab's public handbook<br>"
        "• Source citations for verification<br>"
        "• PII detection & input sanitization<br>"
        "• Off-topic query guardrails<br>"
        "• Relevance scores visible"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button(" Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.session_state.feedback = {}
            st.rerun()
    with col_export:
        if st.session_state.messages:
            chat_export = []
            for m in st.session_state.messages:
                chat_export.append({"role": m["role"], "content": m["content"]})
            st.download_button(
                " Export",
                data=json.dumps(chat_export, indent=2),
                file_name=f"gitlab_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True,
            )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; font-size:0.72rem; color:#bbb;'>"
        "Built with Streamlit • Groq + Llama 3.3<br>"
        "Embeddings: sentence-transformers (local)<br>"
        "Data: GitLab Handbook (public)"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-header">'
    '<div class="logo">🦊</div>'
    '<div class="title-block">'
    '<h1>GitLab Handbook AI Assistant <span class="live-badge">LIVE</span></h1>'
    "<p>Ask anything about GitLab's culture, values, processes, and product direction</p>"
    "</div></div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Initialize index (local — no API key needed)
# ---------------------------------------------------------------------------
if not st.session_state.index_ready:
    with st.spinner("🔧 Building knowledge base locally..."):
        try:
            index, chunks = build_or_load_index()
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.index_ready = True
        except Exception as e:
            st.error(f"Failed to build index: {e}")
            st.stop()

# ---------------------------------------------------------------------------
# Initialize Groq client
# ---------------------------------------------------------------------------
api_key = get_api_key()

if api_key and not st.session_state.client_ready:
    try:
        st.session_state.client = create_chat_client(api_key)
        st.session_state.client_ready = True
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")

if not api_key:
    st.info(" Enter your **Groq API key** in the sidebar to start chatting. It's free — no credit card needed.")
    st.stop()

if not st.session_state.client_ready:
    st.stop()


# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------
n_chunks = len(st.session_state.chunks) if st.session_state.chunks else 0
positive_feedback = sum(1 for v in st.session_state.feedback.values() if v == "up")
st.markdown(
    f'<div class="metrics-row">'
    f'<div class="metric-card"><div class="value">{n_chunks}</div><div class="label">Knowledge Chunks</div></div>'
    f'<div class="metric-card"><div class="value">{st.session_state.query_count}</div><div class="label">Questions Asked</div></div>'
    f'<div class="metric-card"><div class="value">39</div><div class="label">Handbook Pages</div></div>'
    f'<div class="metric-card"><div class="value">{positive_feedback} 👍</div><div class="label">Helpful Answers</div></div>'
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("")


# ---------------------------------------------------------------------------
# Welcome card + suggested questions (only when conversation is empty)
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown(
        '<div class="welcome-card">'
        "<h2>Welcome!</h2>"
        "<p>I'm your AI guide to GitLab's Handbook and Direction pages. "
        "Ask me about company values, remote work culture, engineering practices, "
        "hiring, product strategy, or anything else from the handbook.</p>"
        '<div class="features">'
        '<div class="feat">39 Handbook Pages</div>'
        '<div class="feat">RAG-Powered Search</div>'
        '<div class="feat">Source Citations</div>'
        '<div class="feat">Guardrails Active</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Try asking about...")
    cols = st.columns(2)
    for i, (icon, question) in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 2]:
            if st.button(f"{icon}  {question}", key=f"suggest_{i}", use_container_width=True):
                st.session_state.pending_query = question
                st.rerun()


# ---------------------------------------------------------------------------
# Display chat history with feedback
# ---------------------------------------------------------------------------
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🦊" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if "citations" in msg and msg["citations"]:
                with st.expander("📚 Sources & Citations", expanded=False):
                    for cite in msg["citations"]:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<a href="{cite["url"]}" target="_blank">{cite["title"]}</a>'
                            f'<span class="section-tag">{cite["section"]}</span><br>'
                            f'<span class="relevance">Relevance: {cite["relevance"]:.0%}</span>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            feedback_key = f"fb_{idx}"
            if feedback_key in st.session_state.feedback:
                val = st.session_state.feedback[feedback_key]
                label = "👍 Helpful" if val == "up" else "👎 Not helpful"
                st.markdown(f'<span class="feedback-done">{label} — thanks for the feedback!</span>', unsafe_allow_html=True)
            else:
                fc1, fc2, _ = st.columns([1, 1, 6])
                with fc1:
                    if st.button("👍", key=f"up_{idx}", help="Helpful"):
                        st.session_state.feedback[feedback_key] = "up"
                        st.rerun()
                with fc2:
                    if st.button("👎", key=f"down_{idx}", help="Not helpful"):
                        st.session_state.feedback[feedback_key] = "down"
                        st.rerun()


# ---------------------------------------------------------------------------
# Chat input — handles both typed input and suggested question clicks
# ---------------------------------------------------------------------------
user_input = st.chat_input("Ask about GitLab's handbook, values, culture, or direction...")

pending = st.session_state.pop("pending_query", None)
active_query = user_input or pending

if active_query:
    guard_result = validate_input(active_query)

    if not guard_result.is_valid:
        with st.chat_message("assistant", avatar="🦊"):
            st.warning(guard_result.message)
    else:
        query = guard_result.sanitized_query or active_query
        is_relevant, relevance_hint = check_relevance(query)

        st.session_state.messages.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        with st.chat_message("assistant", avatar="🦊"):
            if relevance_hint:
                st.info(relevance_hint)

            with st.spinner("🔎 Searching handbook..."):
                results = retrieve(
                    query,
                    st.session_state.index,
                    st.session_state.chunks,
                    top_k=5,
                )
                context = format_context(results)
                citations = get_source_citations(results)

            chat_history_for_model = [
                m for m in st.session_state.messages[:-1]
                if "citations" not in m
            ][-6:]

            response_text = st.write_stream(
                generate_response_stream(
                    st.session_state.client,
                    chat_history_for_model,
                    query,
                    context,
                    relevance_hint,
                )
            )

            if citations:
                with st.expander("📚 Sources & Citations", expanded=True):
                    for cite in citations:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<a href="{cite["url"]}" target="_blank">{cite["title"]}</a>'
                            f'<span class="section-tag">{cite["section"]}</span><br>'
                            f'<span class="relevance">Relevance: {cite["relevance"]:.0%}</span>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "citations": citations,
        })
        st.session_state.query_count += 1
