# tests/test_hybrid.py
from __future__ import annotations

from pathlib import Path
from types import MethodType

import pytest

from src.bm25 import BM25Retriever
from src.hybrid import HybridRetriever
from src.semantic import SemanticRetriever


def make_fake_bm25(documents, returned_results):
    """
    Create a lightweight BM25Retriever test double.
    """
    retriever = BM25Retriever.__new__(BM25Retriever)
    retriever.documents = documents

    def fake_search(self, query: str, top_k: int = 5):
        return returned_results[:top_k]

    retriever.search = MethodType(fake_search, retriever)
    return retriever


def make_fake_semantic(documents, returned_results):
    """
    Create a lightweight SemanticRetriever test double.
    """
    retriever = SemanticRetriever.__new__(SemanticRetriever)
    retriever.documents = documents

    def fake_search(self, query: str, top_k: int = 5):
        return returned_results[:top_k]

    retriever.search = MethodType(fake_search, retriever)
    return retriever


@pytest.fixture
def sample_documents():
    """
    Provide a small shared document collection for testing.
    """
    return [
        {"doc_id": "d1", "text": "good moisturizer for dry skin"},
        {"doc_id": "d2", "text": "shampoo for curly hair"},
        {"doc_id": "d3", "text": "face wash for sensitive skin"},
        {"doc_id": "d4", "text": "lightweight sunscreen"},
    ]


def test_hybrid_search_merges_and_reranks(sample_documents):
    """
    Test that hybrid search merges duplicate documents and returns results
    ordered by descending fused score.
    """
    bm25_results = [
        {"doc_id": "d1", "text": "good moisturizer for dry skin", "score": 12.0},
        {"doc_id": "d2", "text": "shampoo for curly hair", "score": 8.0},
        {"doc_id": "d3", "text": "face wash for sensitive skin", "score": 5.0},
    ]

    semantic_results = [
        {"doc_id": "d2", "text": "shampoo for curly hair", "score": 0.91},
        {"doc_id": "d4", "text": "lightweight sunscreen", "score": 0.89},
        {"doc_id": "d1", "text": "good moisturizer for dry skin", "score": 0.82},
    ]

    bm25 = make_fake_bm25(sample_documents, bm25_results)
    semantic = make_fake_semantic(sample_documents, semantic_results)

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        semantic_retriever=semantic,
        bm25_weight=0.4,
        semantic_weight=0.6,
        rrf_k=60,
        fetch_k=3,
        key_field="doc_id",
    )

    results = hybrid.search("moisturizer", top_k=4)

    assert len(results) == 4

    scores = [doc["score"] for doc in results]
    assert scores == sorted(scores, reverse=True)

    doc_ids = [doc["doc_id"] for doc in results]
    assert len(doc_ids) == len(set(doc_ids))

    d2 = next(doc for doc in results if doc["doc_id"] == "d2")
    assert d2["bm25_score"] == 8.0
    assert d2["semantic_score"] == 0.91
    assert set(d2["retrieval_sources"]) == {"bm25", "semantic"}

    d4 = next(doc for doc in results if doc["doc_id"] == "d4")
    assert d4["bm25_score"] is None
    assert d4["semantic_score"] == 0.89
    assert d4["retrieval_sources"] == ["semantic"]


def test_hybrid_prefers_documents_returned_by_both_retrievers(sample_documents):
    """
    Test that a document returned by both retrievers ranks strongly after
    fusion.
    """
    bm25_results = [
        {"doc_id": "d1", "text": "good moisturizer for dry skin", "score": 10.0},
        {"doc_id": "d2", "text": "shampoo for curly hair", "score": 9.0},
    ]

    semantic_results = [
        {"doc_id": "d2", "text": "shampoo for curly hair", "score": 0.95},
        {"doc_id": "d3", "text": "face wash for sensitive skin", "score": 0.90},
    ]

    bm25 = make_fake_bm25(sample_documents, bm25_results)
    semantic = make_fake_semantic(sample_documents, semantic_results)

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        semantic_retriever=semantic,
        bm25_weight=0.5,
        semantic_weight=0.5,
        rrf_k=60,
        fetch_k=2,
        key_field="doc_id",
    )

    results = hybrid.search("hair care", top_k=3)

    assert results[0]["doc_id"] == "d2"


def test_search_rejects_invalid_query(sample_documents):
    """
    Test that invalid query values raise the expected exceptions.
    """
    bm25 = make_fake_bm25(sample_documents, [])
    semantic = make_fake_semantic(sample_documents, [])

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        semantic_retriever=semantic,
        key_field="doc_id",
    )

    with pytest.raises(TypeError):
        hybrid.search(123, top_k=5)

    with pytest.raises(ValueError):
        hybrid.search("", top_k=5)

    with pytest.raises(ValueError):
        hybrid.search("   ", top_k=5)

    with pytest.raises(ValueError):
        hybrid.search("valid query", top_k=0)


def test_constructor_rejects_incompatible_document_collections():
    """
    Test that the constructor raises an error when the two retrievers are built
    over different document collections.
    """
    bm25_docs = [
        {"doc_id": "d1", "text": "alpha"},
        {"doc_id": "d2", "text": "beta"},
    ]
    semantic_docs = [
        {"doc_id": "d1", "text": "alpha"},
        {"doc_id": "d3", "text": "gamma"},
    ]

    bm25 = make_fake_bm25(bm25_docs, [])
    semantic = make_fake_semantic(semantic_docs, [])

    with pytest.raises(ValueError, match="different document collections"):
        HybridRetriever(
            bm25_retriever=bm25,
            semantic_retriever=semantic,
            key_field="doc_id",
        )


def test_constructor_rejects_invalid_hyperparameters(sample_documents):
    """
    Test that invalid constructor hyperparameters raise the expected errors.
    """
    bm25 = make_fake_bm25(sample_documents, [])
    semantic = make_fake_semantic(sample_documents, [])

    with pytest.raises(ValueError):
        HybridRetriever(
            bm25_retriever=bm25,
            semantic_retriever=semantic,
            bm25_weight=-1.0,
            key_field="doc_id",
        )

    with pytest.raises(ValueError):
        HybridRetriever(
            bm25_retriever=bm25,
            semantic_retriever=semantic,
            semantic_weight=-1.0,
            key_field="doc_id",
        )

    with pytest.raises(ValueError):
        HybridRetriever(
            bm25_retriever=bm25,
            semantic_retriever=semantic,
            bm25_weight=0.0,
            semantic_weight=0.0,
            key_field="doc_id",
        )

    with pytest.raises(ValueError):
        HybridRetriever(
            bm25_retriever=bm25,
            semantic_retriever=semantic,
            rrf_k=0,
            key_field="doc_id",
        )

    with pytest.raises(ValueError):
        HybridRetriever(
            bm25_retriever=bm25,
            semantic_retriever=semantic,
            fetch_k=0,
            key_field="doc_id",
        )


def test_save_creates_expected_files(tmp_path: Path, sample_documents):
    """
    Test that saving the hybrid retriever creates the expected artifact files.
    """
    bm25 = make_fake_bm25(sample_documents, [])
    semantic = make_fake_semantic(sample_documents, [])

    bm25.save = MethodType(lambda self, out: Path(out).mkdir(parents=True, exist_ok=True), bm25)
    semantic.save = MethodType(
        lambda self, out: Path(out).mkdir(parents=True, exist_ok=True),
        semantic,
    )

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        semantic_retriever=semantic,
        key_field="doc_id",
    )

    hybrid.save(tmp_path)

    assert (tmp_path / "hybrid_config.json").exists()
    assert (tmp_path / "bm25").exists()
    assert (tmp_path / "semantic").exists()


def test_hybrid_with_real_retrievers(tmp_path: Path):
    """Smoke test for the full HybridRetriever with real BM25 and Semantic retrievers."""
    
    documents = [
        {"doc_id": "d1", "text": "good moisturizer for dry skin"},
        {"doc_id": "d2", "text": "shampoo for curly hair"},
        {"doc_id": "d3", "text": "face wash for sensitive skin"},
    ]

    bm25 = BM25Retriever(documents)
    semantic = SemanticRetriever(documents)

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        semantic_retriever=semantic,
        key_field="doc_id",
    )

    results = hybrid.search("best shampoo for curly hair", top_k=2)

    assert len(results) == 2
    assert all("doc_id" in doc for doc in results)
    assert all("score" in doc for doc in results)

    hybrid.save(tmp_path)
    loaded = HybridRetriever.load(tmp_path)

    loaded_results = loaded.search("best shampoo for curly hair", top_k=2)
    assert len(loaded_results) == 2