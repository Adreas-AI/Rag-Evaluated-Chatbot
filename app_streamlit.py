# app_streamlit.py
import streamlit as st
import traceback
from typing import Any, Dict, List, Optional

from chatbot_app import create_client, init_context, chat_turn
from rag_utils import build_index_from_pdf_bytes, get_context_and_sources, format_context_with_citations


def merge_indexes(indexes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Ενώνει πολλά indexes σε ένα.
    Reindex chunk_id ώστε να μην υπάρχουν duplicates.
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
st.caption("Streamlit UI με evaluation + προαιρετικό RAG πάνω σε PDF + citations.")

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
    st.header("⚙️ Ρυθμίσεις")

    st.session_state.use_rag = st.checkbox("Χρήση knowledge base (RAG mode)", value=st.session_state.use_rag)

    st.markdown("### 📄 Knowledge Base (PDF)")
    uploaded_pdfs = st.file_uploader("Ανέβασε PDF(s)", type=["pdf"], accept_multiple_files=True)

    colA, colB = st.columns(2)
    with colA:
        build_btn = st.button("📚 Χτίσε KB", use_container_width=True)
    with colB:
        clear_btn = st.button("🧹 Καθάρισε KB", use_container_width=True)

    if clear_btn:
        st.session_state.rag_index = None
        st.success("Καθαρίστηκε το knowledge base.")

    if uploaded_pdfs and build_btn:
        with st.spinner("Χτίζω knowledge base (embeddings)..."):
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
                st.success(f"✅ Έτοιμο KB από {len(uploaded_pdfs)} PDF(s).")
            except Exception:
                st.error("Σφάλμα κατά το χτίσιμο του KB.")
                st.code(traceback.format_exc(), language="text")

    if st.session_state.rag_index is not None:
        st.caption("✅ Φορτωμένο KB:")
        st.caption(st.session_state.rag_index.get("filename", "KB"))
        st.caption(f"Chunks: {len(st.session_state.rag_index.get('chunks', []))}")
    else:
        st.caption("ℹ️ Δεν έχει φορτωθεί KB.")

    st.markdown("---")
    st.markdown("### 🔎 Retrieval settings")
    st.session_state.top_k = st.slider("Top-K chunks", 1, 8, int(st.session_state.top_k), 1)

    # κρατάμε το toggle για UI, αλλά το rag_utils εδώ δεν εφαρμόζει MMR (σταθερότητα πρώτα)
    st.session_state.use_mmr = st.checkbox("MMR diversity retrieval", value=bool(st.session_state.use_mmr))

    st.session_state.strict_rag = st.checkbox("Strict RAG (απάντα μόνο από το PDF)", value=bool(st.session_state.strict_rag))
    st.session_state.show_sources = st.checkbox("Δείξε πηγές κάτω από κάθε απάντηση", value=bool(st.session_state.show_sources))
    st.session_state.show_scores = st.checkbox("Δείξε scores", value=bool(st.session_state.show_scores))

    st.markdown("---")
    st.markdown("### 🧪 Admin")
    show_eval = st.checkbox("Δείξε τελευταία αξιολόγηση", value=False)

    if st.button("🔁 Νέα συνομιλία"):
        st.session_state.conversation = init_context()
        st.session_state.chat_history = []
        st.session_state.last_eval = None
        st.success("Ξεκίνησε νέα συνομιλία.")


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
            with st.expander("📚 Πηγές (από PDF)", expanded=False):
                for i, s in enumerate(sources, start=1):
                    page = s.get("page", 0)
                    chunk_id = s.get("chunk_id", -1)
                    score = s.get("score", None)
                    text = (s.get("text") or "").strip()

                    header = f"**{i}. Σελ. {page} | chunk {chunk_id}**"
                    if st.session_state.show_scores and isinstance(score, (int, float)):
                        header += f" — score={float(score):.3f}"
                    st.markdown(header)
                    st.code(text[:1500], language="text")
                    st.markdown("---")


# ------------------------- USER INPUT ------------------------- #
user_input = st.chat_input("Γράψε το μήνυμά σου...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # ΠΑΝΤΑ define αυτά εδώ (για να μην υπάρξει NameError)
    sources: List[Dict[str, Any]] = []
    retrieved_context: Optional[str] = None

    rag_is_ready = (
        st.session_state.use_rag
        and st.session_state.rag_index is not None
        and len(st.session_state.rag_index.get("chunks", [])) > 0
    )

    if rag_is_ready:
        with st.spinner("🔎 Αναζητώ σχετικές πηγές από το PDF..."):
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
                st.warning(f"Δεν μπόρεσα να πάρω context από το PDF: {e}")
                sources = []
                retrieved_context = None

    with st.chat_message("assistant"):
        placeholder = st.empty()

        # Strict RAG guardrail (UI-level)
        if rag_is_ready and st.session_state.strict_rag and (not sources):
            assistant_reply = (
                "Δεν βρήκα σχετικές πληροφορίες μέσα στο PDF για να απαντήσω με ασφάλεια.\n\n"
                "Δοκίμασε:\n"
                "- να το ρωτήσεις με άλλα λόγια, ή\n"
                "- να ανεβάσεις άλλο/επιπλέον έγγραφο."
            )
            new_conversation = st.session_state.conversation
            evaluation = {"label": "no_sources", "score": None, "explanation": "Strict RAG χωρίς sources.", "raw": "{}"}
        else:
            with st.spinner("🤖 Σκέφτομαι..."):
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
            with st.expander("📚 Πηγές (από PDF)", expanded=False):
                for i, s in enumerate(sources, start=1):
                    page = s.get("page", 0)
                    chunk_id = s.get("chunk_id", -1)
                    score = s.get("score", None)
                    text = (s.get("text") or "").strip()

                    header = f"**{i}. Σελ. {page} | chunk {chunk_id}**"
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
    st.sidebar.subheader("Τελευταία αξιολόγηση")
    eval_info = st.session_state.last_eval
    if eval_info is None:
        st.sidebar.write("Δεν υπάρχει διαθέσιμη αξιολόγηση.")
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

        with st.sidebar.expander("Raw JSON από τον evaluator"):
            st.sidebar.code(raw, language="json")
