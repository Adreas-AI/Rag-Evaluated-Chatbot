import math

import numpy as np
import pytest

import rag_utils as ru


def test_chunks_have_offsets_and_overlap():
    text = "a" * 2500
    chunks = ru.split_text_into_chunks(text, max_chars=1000, overlap=200)
    assert [(c["char_start"], c["char_end"]) for c in chunks] == [
        (0, 1000), (800, 1800), (1600, 2500), (2400, 2500),
    ]
    assert all(len(c["text"]) <= 1000 for c in chunks)


def test_chunking_edge_cases():
    assert ru.split_text_into_chunks("   ") == []
    assert ru.split_text_into_chunks(None) == []
    # overlap >= max_chars must not loop forever
    assert len(ru.split_text_into_chunks("x" * 30, max_chars=10, overlap=10)) == 3
    # very long input is capped
    chunks = ru.split_text_into_chunks("y" * 5000, max_chars=1000, overlap=0, max_total_chars=2000)
    assert chunks[-1]["char_end"] == 2000


def test_pdf_pages_are_extracted_with_page_numbers(sample_pdf_bytes):
    pages = ru.extract_text_by_page_from_pdf_bytes(sample_pdf_bytes)
    assert [p["page"] for p in pages] == [1, 2, 3]
    assert "annual leave" in pages[1]["text"].lower()
    assert "password" in pages[2]["text"].lower()


def test_chunk_metadata_skips_empty_pages_and_numbers_chunks():
    pages = [{"page": 1, "text": "hello world"}, {"page": 2, "text": ""}, {"page": 3, "text": "bye"}]
    meta = ru.build_chunks_with_metadata_from_pages(pages, filename="doc.pdf")
    assert [(m["chunk_id"], m["page"], m["source"]) for m in meta] == [
        (0, 1, "doc.pdf"), (1, 3, "doc.pdf"),
    ]


def test_cosine_similarity_is_safe_for_zero_vectors():
    assert ru.cosine_similarity(np.array([1, 0]), np.array([1, 0])) == pytest.approx(1.0)
    assert ru.cosine_similarity(np.array([1, 0]), np.array([0, 1])) == pytest.approx(0.0)
    assert ru.cosine_similarity(np.array([0, 0]), np.array([1, 1])) == 0.0


def test_index_embeds_in_batches(fake_client, sample_pdf_bytes):
    index = ru.build_index_from_pdf_bytes(
        fake_client, sample_pdf_bytes, filename="handbook.pdf", max_chars=300, overlap=50,
        embed_batch_size=2,
    )
    n = len(index["chunks"])
    assert n > 3 and index["filename"] == "handbook.pdf"
    assert all(len(c["embedding"]) == len(ru.np.array(c["embedding"])) for c in index["chunks"])
    batches = [kw["input"] for kind, kw in fake_client.calls if kind == "embeddings"]
    assert all(len(b) <= 2 for b in batches) and sum(map(len, batches)) == n


def test_empty_pdf_text_gives_empty_index(fake_client, monkeypatch):
    monkeypatch.setattr(ru, "extract_text_by_page_from_pdf_bytes", lambda b: [{"page": 1, "text": ""}])
    index = ru.build_index_from_pdf_bytes(fake_client, b"%PDF", filename="scan.pdf")
    assert index["chunks"] == [] and fake_client.calls == []


def test_retrieval_ranks_the_relevant_page_first(fake_client, sample_pdf_bytes):
    index = ru.build_index_from_pdf_bytes(fake_client, sample_pdf_bytes, filename="handbook.pdf")
    result = ru.get_context_and_sources(fake_client, "How many days of annual leave?", index, top_k=2)
    assert result["sources"][0]["page"] == 2
    assert len(result["sources"]) == 2
    assert result["context"].startswith("[SOURCE handbook.pdf | p.2 | chunk")
    scores = [s["score"] for s in result["sources"]]
    assert scores == sorted(scores, reverse=True)


def test_retrieval_on_empty_index_returns_nothing(fake_client):
    assert ru.get_context_and_sources(fake_client, "q", {"chunks": []}) == {
        "context": "", "sources": []
    }
    assert fake_client.calls == []  # no embedding call wasted


def test_citation_formatting():
    sources = [
        {"source": "a.pdf", "page": 3, "chunk_id": 7, "score": 0.91234, "text": " text A "},
        {"source": "b.pdf", "page": 1, "chunk_id": 0, "score": math.nan, "text": "text B"},
    ]
    plain = ru.format_context_with_citations(sources)
    assert plain == "[SOURCE a.pdf | p.3 | chunk 7]\ntext A\n\n---\n\n[SOURCE b.pdf | p.1 | chunk 0]\ntext B"
    scored = ru.format_context_with_citations(sources, include_scores=True)
    assert "(score=0.912)" in scored and "nan" not in scored
