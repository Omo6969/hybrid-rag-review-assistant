from __future__ import annotations

from pathlib import Path

from src.bm25 import BM25Retriever
from src.hybrid import HybridRetriever
from src.semantic import SemanticRetriever


def load_saved_retrievers(
    project_root: str | Path,
    bm25_weight: float = 0.4,
    semantic_weight: float = 0.6,
    rrf_k: int = 60,
    fetch_k: int = 10,
    key_field: str = "doc_id",
) -> tuple[BM25Retriever, SemanticRetriever, HybridRetriever]:
    """Load saved BM25, semantic, and hybrid retrievers.

    Parameters
    ----------
    project_root : str or pathlib.Path
        Project root directory containing the `data/processed/` folder.
    bm25_weight : float, default=0.4
        Weight assigned to BM25 rankings in the hybrid retriever.
    semantic_weight : float, default=0.6
        Weight assigned to semantic rankings in the hybrid retriever.
    rrf_k : int, default=60
        Reciprocal Rank Fusion smoothing constant.
    fetch_k : int, default=10
        Number of documents fetched from each retriever before fusion.
    key_field : str, default="doc_id"
        Unique document identifier field used by the hybrid retriever.

    Returns
    -------
    tuple of BM25Retriever, SemanticRetriever, HybridRetriever
        Loaded BM25 retriever, semantic retriever, and hybrid retriever.

    Raises
    ------
    FileNotFoundError
        If any required BM25 or semantic retriever artifact is missing.
    """
    project_root = Path(project_root)

    bm25_dir = project_root / "data" / "processed" / "bm25_index"
    semantic_dir = project_root / "data" / "processed" / "semantic_index"

    bm25_required = [
        bm25_dir / "bm25_documents.pkl",
        bm25_dir / "bm25_tokenized_corpus.pkl",
        bm25_dir / "bm25_index.pkl",
    ]

    semantic_required = [
        semantic_dir / "semantic_documents.pkl",
        semantic_dir / "semantic_model_name.json",
        semantic_dir / "semantic_embeddings.npy",
        semantic_dir / "semantic_faiss.index",
    ]

    missing_bm25 = [str(path) for path in bm25_required if not path.exists()]
    missing_semantic = [str(path) for path in semantic_required if not path.exists()]

    if missing_bm25:
        raise FileNotFoundError(
            "Missing BM25 retriever artifact(s): " + ", ".join(missing_bm25)
        )

    if missing_semantic:
        raise FileNotFoundError(
            "Missing semantic retriever artifact(s): " + ", ".join(missing_semantic)
        )

    bm25_retriever = BM25Retriever.load(bm25_dir)
    semantic_retriever = SemanticRetriever.load(semantic_dir)

    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        semantic_retriever=semantic_retriever,
        bm25_weight=bm25_weight,
        semantic_weight=semantic_weight,
        rrf_k=rrf_k,
        fetch_k=fetch_k,
        key_field=key_field,
    )

    return bm25_retriever, semantic_retriever, hybrid_retriever