# app_streamlit.py
import streamlit as st
import traceback
from typing import Any, Dict, List, Optional

from chatbot_app import create_client, init_context, chat_turn
from rag_utils import build_index_from_pdf_bytes, get_context_and_sources, format_context_with_citations


def merge_indexes(indexes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple indexes into one.
    Re-number chunk_id so there are no duplicates.
    """
    if not indexes:
        return {"filename": "KB", "embedding_model": "", "chunks": []}

    all_chunks: List[Dict[str, Any]] = []
    embedding_model = indexes[0].get("embedding_model", "")
    filenames: List[str] = []

    new_chunk_id = 0
    for idx in indexes:
        filenames.append(idx.get("filename", "document"))
        for ch in (idx.get("chunks") or []):
            ch2 = dict(ch)
            ch2["chunk_id"] = new_chunk_id
            new_chunk_id += 1
            all_chunks.append(ch2)

    return {
        "filename": " | ".join(filenames) if filenames else "KB",
        "embedding_model": embedding_model,
        "chunks": all_chunks,
    }


st.set_page_config(page_title="My Evaluated RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 My Evaluated RAG Chatbot")
st.caption("Streamlit UI with automatic evaluation + optional RAG over PDFs + citations.")

# ------------------------- SESSION STATE ------------------------- #
if "client" not in st.session_state:
    st.session_state.client = create_client()

if "conversation" not in st.session_state:
    st.session_state.conversation = init_context()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_eval" not in st.session_state:
    st.session_state.last_eval = None

if "rag_index" not in st.session_state:
    st.session_state.rag_index = None

if "use_rag" not in st.session_state:
    st.session_state.use_rag = False

if "show_sources" not in st.session_state:
    st.session_state.show_sources = True

if "show_scores" not in st.session_state:
    st.session_state.show_scores = False

if "top_k" not in st.session_state:
    st.session_state.top_k = 3

if "strict_rag" not in st.session_state:
    st.session_state.strict_rag = True

if "use_mmr" not in st.session_state:
    st.session_state.use_mmr = True


# ------------------------- SIDEBAR ------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")

    st.session_state.use_rag = st.checkbox("Use knowledge base (RAG mode)", value=st.session_state.use_rag)

    st.markdown("### 📄 Knowledge Base (PDF)")
    uploaded_pdfs = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    colA, colB = st.columns(2)
    with colA:
        build_btn = st.button("📚 Build KB", use_container_width=True)
    with colB:
        clear_btn = st.button("🧹 Clear KB", use_container_width=True)

    if clear_btn:
        st.session_state.rag_index = None
        st.success("Knowledge base cleared.")

    if uploaded_pdfs and build_btn:
        with st.spinner("Building knowledge base (embeddings)..."):
            try:
                indexes = []
                for up in uploaded_pdfs:
                    pdf_bytes = up.getvalue()
                    idx = build_index_from_pdf_bytes(
                        st.session_state.client,
                        pdf_bytes,
                        filename=up.name,
                    )
                    indexes.append(idx)
                st.session_state.rag_index = merge_indexes(indexes)
                st.success(f"✅ KB ready from {len(uploaded_pdfs)} PDF(s).")
            except Exception:
                st.error("Error while building the KB.")
                st.code(traceback.format_exc(), language="text")

    if st.session_state.rag_index is not None:
        st.caption("✅ Loaded KB:")
        st.caption(st.session_state.rag_index.get("filename", "KB"))
        st.caption(f"Chunks: {len(st.session_state.rag_index.get('chunks', []))}")
    else:
        st.caption("ℹ️ No KB loaded.")

    st.markdown("---")
    st.markdown("### 🔎 Retrieval settings")
    st.session_state.top_k = st.slider("Top-K chunks", 1, 8, int(st.session_state.top_k), 1)

    # toggle kept for the UI, but rag_utils does not apply MMR yet (stability first)
    st.session_state.use_mmr = st.checkbox("MMR diversity retrieval", value=bool(st.session_state.use_mmr))

    st.session_state.strict_rag = st.checkbox("Strict RAG (answer only from the PDF)", value=bool(st.session_state.strict_rag))
    st.session_state.show_sources = st.checkbox("Show sources under each answer", value=bool(st.session_state.show_sources))
    st.session_state.show_scores = st.checkbox("Show scores", value=bool(st.session_state.show_scores))

    st.markdown("---")
    st.markdown("### 🧪 Admin")
    show_eval = st.checkbox("Show last evaluation", value=False)

    if st.button("🔁 New conversation"):
        st.session_state.conversation = init_context()
        st.session_state.chat_history = []
        st.session_state.last_eval = None
        st.success("Started a new conversation.")


# ------------------------- CHAT HISTORY ------------------------- #
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(turn["user"])

    with st.chat_message("assistant"):
        st.markdown(turn["assistant"])

        sources = turn.get("sources") or []
        if sources:
            cites = ", ".join([f"p.{s.get('page', 0)} (chunk {s.get('chunk_id', -1)})" for s in sources])
            st.caption(f"Sources: {cites}")

        if st.session_state.show_sources and sources:
            with st.expander("📚 Sources (from PDF)", expanded=False):
                for i, s in enumerate(sources, start=1):
                    page = s.get("page", 0)
                    chunk_id = s.get("chunk_id", -1)
                    score = s.get("score", None)
                    text = (s.get("text") or "").strip()

                    header = f"**{i}. Page {page} | chunk {chunk_id}**"
                    if st.session_state.show_scores and isinstance(score, (int, float)):
                        header += f" — score={float(score):.3f}"
                    st.markdown(header)
                    st.code(text[:1500], language="text")
                    st.markdown("---")


# ------------------------- USER INPUT ------------------------- #
user_input = st.chat_input("Type your message...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # ALWAYS define these here (avoids NameError)
    sources: List[Dict[str, Any]] = []
    retrieved_context: Optional[str] = None

    rag_is_ready = (
        st.session_state.use_rag
        and st.session_state.rag_index is not None
        and len(st.session_state.rag_index.get("chunks", [])) > 0
    )

    if rag_is_ready:
        with st.spinner("🔎 Searching the PDF for relevant sources..."):
            try:
                result = get_context_and_sources(
                    st.session_state.client,
                    user_input,
                    st.session_state.rag_index,
                    top_k=int(st.session_state.top_k),
                    use_mmr=bool(st.session_state.use_mmr),
                )
                sources = result.get("sources") or []
                retrieved_context = result.get("context") or ""

                if st.session_state.show_scores and sources:
                    retrieved_context = format_context_with_citations(sources, include_scores=True)

            except Exception as e:
                st.warning(f"Could not retrieve context from the PDF: {e}")
                sources = []
                retrieved_context = None

    with st.chat_message("assistant"):
        placeholder = st.empty()

        # Strict RAG guardrail (UI-level)
        if rag_is_ready and st.session_state.strict_rag and (not sources):
            assistant_reply = (
                "I could not find relevant information in the PDF to answer reliably.\n\n"
                "Try:\n"
                "- rephrasing your question, or\n"
                "- uploading another/additional document."
            )
            new_conversation = st.session_state.conversation
            evaluation = {"label": "no_sources", "score": None, "explanation": "Strict RAG with no sources.", "raw": "{}"}
        else:
            with st.spinner("🤖 Thinking..."):
                assistant_reply, new_conversation, evaluation = chat_turn(
                    st.session_state.client,
                    user_input,
                    st.session_state.conversation,
                    retrieved_context=retrieved_context,
                    strict_rag=bool(st.session_state.strict_rag),
                    sources_used=len(sources),
                )

        placeholder.markdown(assistant_reply)

        if sources:
            cites = ", ".join([f"p.{s.get('page', 0)} (chunk {s.get('chunk_id', -1)})" for s in sources])
            st.caption(f"Sources: {cites}")

        if st.session_state.show_sources and sources:
            with st.expander("📚 Sources (from PDF)", expanded=False):
                for i, s in enumerate(sources, start=1):
                    page = s.get("page", 0)
                    chunk_id = s.get("chunk_id", -1)
                    score = s.get("score", None)
                    text = (s.get("text") or "").strip()

                    header = f"**{i}. Page {page} | chunk {chunk_id}**"
                    if st.session_state.show_scores and isinstance(score, (int, float)):
                        header += f" — score={float(score):.3f}"
                    st.markdown(header)
                    st.code(text[:1500], language="text")
                    st.markdown("---")

    st.session_state.conversation = new_conversation
    st.session_state.chat_history.append({"user": user_input, "assistant": assistant_reply, "sources": sources})
    st.session_state.last_eval = evaluation


# ------------------------- ADMIN: EVAL ------------------------- #
if show_eval:
    st.sidebar.subheader("Last evaluation")
    eval_info = st.session_state.last_eval
    if eval_info is None:
        st.sidebar.write("No evaluation available.")
    else:
        score = eval_info.get("score")
        label = eval_info.get("label", "unknown")
        explanation = eval_info.get("explanation", "")
        raw = eval_info.get("raw", "")

        st.sidebar.markdown(f"**Label:** {label}")
        st.sidebar.markdown(f"**Score:** {score:.3f}" if isinstance(score, (int, float)) else "**Score:** -")

        if explanation:
            st.sidebar.markdown("**Explanation:**")
            st.sidebar.markdown(explanation)

        with st.sidebar.expander("Raw JSON from the evaluator"):
            st.sidebar.code(raw, language="json")
