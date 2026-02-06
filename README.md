# Rag Evaluated Chatbot (Streamlit + PDF RAG + Citations)

A portfolio-ready **RAG chatbot** that builds a knowledge base from uploaded PDFs and answers with **citations**, plus **safety** (moderation + prompt-injection detection) and **automatic evaluation logging**.

## Demo
- Run locally with Streamlit UI
- Upload 1+ PDFs → Build KB → Ask questions → See sources (page + chunk) under each answer

---

## Key Features
### RAG (PDF Knowledge Base)
- PDF text extraction (per page)
- Chunking with overlap
- Embeddings: `text-embedding-3-small`
- Top-K retrieval with cosine similarity
- Multi-PDF KB support (merged index)

### Citations
- Shows retrieved sources (page + chunk)
- Optional Strict RAG mode (answer only from PDF sources)

### Safety & Robustness
- Moderation check (blocks unsafe content)
- Prompt injection detector (refuses override/leak attempts)
---

## Project Structure
```text
CHATBOT-RAG-AI/
  app_streamlit.py      # Streamlit UI
  chatbot_app.py        # Chat core: moderation + injection + evaluation + logging
  rag_utils.py          # PDF -> chunks -> embeddings -> retrieval
  README.md
  .gitignore
