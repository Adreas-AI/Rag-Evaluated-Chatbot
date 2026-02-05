# rag_utils.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import io
import math

import numpy as np
from pypdf import PdfReader
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"


# ------------------------- PDF EXTRACTION (WITH PAGES) ------------------------- #

def extract_text_by_page_from_pdf_bytes(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Επιστρέφει λίστα:
    [{"page": 1, "text": "..."}, ...]
    """
    pdf_file = io.BytesIO(file_bytes)
    reader = PdfReader(pdf_file)

    pages: List[Dict[str, Any]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        pages.append({"page": i, "text": (page_text or "").strip()})

    return pages


# ------------------------- CHUNKING ------------------------- #

def split_text_into_chunks(
    text: str,
    max_chars: int = 1000,
    overlap: int = 200,
    max_total_chars: int = 200_000,
) -> List[Dict[str, Any]]:
    """
    Επιστρέφει chunks με offsets:
    [{"text": "...", "char_start": 0, "char_end": 999}, ...]
    """
    text = (text or "").strip()
    if not text:
        return []

    if len(text) > max_total_chars:
        text = text[:max_total_chars]

    step = max_chars - overlap
    if step <= 0:
        step = max_chars

    out: List[Dict[str, Any]] = []
    n = len(text)
    for start in range(0, n, step):
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            out.append({"text": chunk, "char_start": start, "char_end": end})
    return out


def build_chunks_with_metadata_from_pages(
    pages: List[Dict[str, Any]],
    filename: str,
    max_chars: int = 1000,
    overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    [{"chunk_id": 0, "source": filename, "page": 1, "text": "...", "char_start": 0, "char_end": 1000}, ...]
    """
    chunks_meta: List[Dict[str, Any]] = []
    chunk_id = 0

    for p in pages:
        page_num = int(p.get("page", 0) or 0)
        page_text = (p.get("text") or "").strip()
        if not page_text:
            continue

        chunks = split_text_into_chunks(page_text, max_chars=max_chars, overlap=overlap)
        for ch in chunks:
            chunks_meta.append(
                {
                    "chunk_id": chunk_id,
                    "source": filename,
                    "page": page_num,
                    "text": ch["text"],
                    "char_start": int(ch["char_start"]),
                    "char_end": int(ch["char_end"]),
                }
            )
            chunk_id += 1

    return chunks_meta


# ------------------------- EMBEDDINGS ------------------------- #

def _batched(items: List[str], batch_size: int) -> List[List[str]]:
    if batch_size <= 0:
        return [items]
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def embed_texts(client: OpenAI, texts: List[str], batch_size: int = 64) -> List[List[float]]:
    if not texts:
        return []
    out: List[List[float]] = []
    for batch in _batched(texts, batch_size=batch_size):
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out


def build_index_from_pdf_bytes(
    client: OpenAI,
    file_bytes: bytes,
    filename: str = "uploaded.pdf",
    max_chars: int = 1000,
    overlap: int = 200,
    embed_batch_size: int = 64,
) -> Dict[str, Any]:
    pages = extract_text_by_page_from_pdf_bytes(file_bytes)
    chunks_meta = build_chunks_with_metadata_from_pages(pages, filename=filename, max_chars=max_chars, overlap=overlap)

    if not chunks_meta:
        return {"filename": filename, "embedding_model": EMBEDDING_MODEL, "chunks": []}

    texts = [c["text"] for c in chunks_meta]
    embeddings = embed_texts(client, texts, batch_size=embed_batch_size)

    indexed_chunks: List[Dict[str, Any]] = []
    for c, emb in zip(chunks_meta, embeddings):
        indexed_chunks.append({**c, "embedding": emb})

    return {"filename": filename, "embedding_model": EMBEDDING_MODEL, "chunks": indexed_chunks}


# ------------------------- RETRIEVAL ------------------------- #

def _safe_norm(x: np.ndarray) -> float:
    n = float(np.linalg.norm(x))
    return n if n > 0 else 1.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.array(a, dtype=float).reshape(-1)
    b = np.array(b, dtype=float).reshape(-1)
    a = a / _safe_norm(a)
    b = b / _safe_norm(b)
    return float(np.dot(a, b))


def get_relevant_sources(
    client: OpenAI,
    question: str,
    index: Dict[str, Any],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    chunks = index.get("chunks", [])
    if not chunks:
        return []

    q = client.embeddings.create(model=EMBEDDING_MODEL, input=[question])
    query_emb = np.array(q.data[0].embedding, dtype=float)

    scored: List[Dict[str, Any]] = []
    for ch in chunks:
        ch_emb = np.array(ch["embedding"], dtype=float)
        score = cosine_similarity(query_emb, ch_emb)
        scored.append(
            {
                "score": score,
                "source": ch.get("source", index.get("filename", "document")),
                "page": int(ch.get("page", 0) or 0),
                "chunk_id": int(ch.get("chunk_id", -1)),
                "text": ch.get("text", ""),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[: max(1, int(top_k))]


def format_context_with_citations(
    sources: List[Dict[str, Any]],
    include_scores: bool = False,
) -> str:
    """
    Φτιάχνει context string με markers.
    """
    blocks: List[str] = []
    for s in sources:
        source = s.get("source", "document")
        page = s.get("page", 0)
        chunk_id = s.get("chunk_id", -1)
        score = s.get("score", None)

        header = f"[SOURCE {source} | p.{page} | chunk {chunk_id}]"
        if include_scores and isinstance(score, (int, float)) and not math.isnan(float(score)):
            header += f" (score={float(score):.3f})"

        text = (s.get("text") or "").strip()
        blocks.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(blocks)


def get_context_and_sources(
    client: OpenAI,
    question: str,
    index: Dict[str, Any],
    top_k: int = 3,
    use_mmr: bool = True,  # υπάρχει για συμβατότητα με το UI (δεν το χρησιμοποιούμε εδώ)
) -> Dict[str, Any]:
    sources = get_relevant_sources(client, question, index, top_k=top_k)
    return {
        "context": format_context_with_citations(sources, include_scores=False) if sources else "",
        "sources": sources,
    }


def get_relevant_context(
    client: OpenAI,
    question: str,
    index: Dict[str, Any],
    top_k: int = 3,
) -> str:
    result = get_context_and_sources(client, question, index, top_k=top_k, use_mmr=False)
    return result["context"]




