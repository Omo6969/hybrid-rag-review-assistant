# src/hybrid.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.bm25 import BM25Retriever
from src.semantic import SemanticRetriever


class HybridRetriever:
    """
    A hybrid retriever that combines BM25 and semantic retrieval using
    weighted Reciprocal Rank Fusion (RRF).

    This class assumes that both the BM25 retriever and the semantic retriever
    are built over the same document collection, in the same order or with
    equivalent document identifiers. The hybrid search process retrieves the
    top documents from each retriever independently, fuses the ranked lists
    using weighted RRF, removes duplicates, and returns the final top-ranked
    results.

    Parameters
    ----------
    bm25_retriever : BM25Retriever
        A fitted BM25 retriever instance.
    semantic_retriever : SemanticRetriever
        A fitted semantic retriever instance.
    bm25_weight : float, default=0.5
        Weight assigned to BM25 rankings in the RRF fusion step.
    semantic_weight : float, default=0.5
        Weight assigned to semantic rankings in the RRF fusion step.
    rrf_k : int, default=60
        The RRF smoothing constant. Larger values reduce the influence of
        top-rank differences.
    fetch_k : int, default=10
        Number of documents to fetch from each underlying retriever before
        fusion. The final returned results may be fewer, depending on
        ``top_k`` and duplicate removal.
    key_field : str or None, default=None
        Optional document field used as the primary unique identifier during
        deduplication. If ``None``, a heuristic key selection strategy is used.

    Attributes
    ----------
    bm25_retriever : BM25Retriever
        The BM25 retriever used in the hybrid system.
    semantic_retriever : SemanticRetriever
        The semantic retriever used in the hybrid system.
    bm25_weight : float
        Weight assigned to BM25 rankings in the RRF fusion step.
    semantic_weight : float
        Weight assigned to semantic rankings in the RRF fusion step.
    rrf_k : int
        RRF smoothing constant.
    fetch_k : int
        Number of documents fetched from each retriever before fusion.
    key_field : str or None
        Optional field used as a unique document identifier.
    documents : list of dict of str to Any
        The shared document collection used by both retrievers.

    Raises
    ------
    TypeError
        If any argument has an invalid type.
    ValueError
        If any numeric hyperparameter is invalid or if the underlying
        retrievers are built over incompatible document collections.

    Notes
    -----
    Reciprocal Rank Fusion computes a fused score as:

    ``score(d) = sum_i w_i / (rrf_k + rank_i(d))``

    where ``w_i`` is the retriever weight and ``rank_i(d)`` is the
    1-based rank of document ``d`` in retriever ``i``.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        semantic_retriever: SemanticRetriever,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        rrf_k: int = 60,
        fetch_k: int = 10,
        key_field: str | None = None,
    ) -> None:
        """
        Initialize the hybrid retriever.

        Parameters
        ----------
        bm25_retriever : BM25Retriever
            A fitted BM25 retriever instance.
        semantic_retriever : SemanticRetriever
            A fitted semantic retriever instance.
        bm25_weight : float, default=0.5
            Weight assigned to BM25 rankings in the RRF fusion step.
        semantic_weight : float, default=0.5
            Weight assigned to semantic rankings in the RRF fusion step.
        rrf_k : int, default=60
            The RRF smoothing constant.
        fetch_k : int, default=10
            Number of documents to fetch from each underlying retriever before
            fusion.
        key_field : str or None, default=None
            Optional field used to identify and deduplicate documents.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If any argument has an invalid type.
        ValueError
            If any numeric hyperparameter is invalid or if the retrievers are
            incompatible.
        """
        if not isinstance(bm25_retriever, BM25Retriever):
            raise TypeError(
                "`bm25_retriever` must be an instance of BM25Retriever, "
                f"got {type(bm25_retriever).__name__}."
            )

        if not isinstance(semantic_retriever, SemanticRetriever):
            raise TypeError(
                "`semantic_retriever` must be an instance of SemanticRetriever, "
                f"got {type(semantic_retriever).__name__}."
            )

        if not isinstance(bm25_weight, (int, float)):
            raise TypeError(
                f"`bm25_weight` must be numeric, got {type(bm25_weight).__name__}."
            )

        if not isinstance(semantic_weight, (int, float)):
            raise TypeError(
                "`semantic_weight` must be numeric, "
                f"got {type(semantic_weight).__name__}."
            )

        if not isinstance(rrf_k, int):
            raise TypeError(f"`rrf_k` must be an integer, got {type(rrf_k).__name__}.")

        if not isinstance(fetch_k, int):
            raise TypeError(
                f"`fetch_k` must be an integer, got {type(fetch_k).__name__}."
            )

        if key_field is not None and not isinstance(key_field, str):
            raise TypeError(
                f"`key_field` must be a string or None, got {type(key_field).__name__}."
            )

        if bm25_weight < 0:
            raise ValueError("`bm25_weight` must be non-negative.")

        if semantic_weight < 0:
            raise ValueError("`semantic_weight` must be non-negative.")

        if bm25_weight == 0 and semantic_weight == 0:
            raise ValueError(
                "At least one of `bm25_weight` or `semantic_weight` must be positive."
            )

        if rrf_k < 1:
            raise ValueError("`rrf_k` must be at least 1.")

        if fetch_k < 1:
            raise ValueError("`fetch_k` must be at least 1.")

        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.bm25_weight = float(bm25_weight)
        self.semantic_weight = float(semantic_weight)
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k
        self.key_field = key_field

        self._validate_retriever_compatibility()
        self.documents = self.bm25_retriever.documents

    def _validate_retriever_compatibility(self) -> None:
        """
        Validate that the two underlying retrievers are built on the same
        document collection.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the underlying retrievers do not appear to share the same
            document collection.
        """
        bm25_docs = self.bm25_retriever.documents
        semantic_docs = self.semantic_retriever.documents

        if len(bm25_docs) != len(semantic_docs):
            raise ValueError(
                "BM25 and semantic retrievers must contain the same number of documents."
            )

        bm25_keys = [self._get_document_key(doc) for doc in bm25_docs]
        semantic_keys = [self._get_document_key(doc) for doc in semantic_docs]

        if bm25_keys != semantic_keys:
            raise ValueError(
                "BM25 and semantic retrievers appear to be built on different "
                "document collections or in different orders."
            )

    def _get_document_key(self, doc: dict[str, Any]) -> str:
        """
        Build a stable deduplication key for a document.

        Parameters
        ----------
        doc : dict of str to Any
            A document dictionary.

        Returns
        -------
        str
            A stable string key used to identify the document.

        Notes
        -----
        The method first tries ``key_field`` if provided. Otherwise, it checks a
        set of common identifier fields. If none are present, it falls back to
        the document text.
        """
        if self.key_field is not None:
            if self.key_field not in doc:
                raise ValueError(
                    f"Configured key field '{self.key_field}' is missing from a document."
                )
            return str(doc[self.key_field])

        candidate_fields = [
            "doc_id",
            "review_id",
            "id",
            "parent_asin",
            "asin",
            "product_id",
        ]

        for field in candidate_fields:
            if field in doc:
                return f"{field}:{doc[field]}"

        if "text" in doc:
            return f"text:{str(doc['text'])}"

        raise ValueError(
            "Unable to infer a unique key for a document. Provide `key_field` "
            "or ensure documents contain a stable identifier or `text`."
        )

    def _fuse_ranked_results(
        self,
        bm25_results: list[dict[str, Any]],
        semantic_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Fuse BM25 and semantic result lists using weighted Reciprocal Rank
        Fusion.

        Parameters
        ----------
        bm25_results : list of dict of str to Any
            Ranked BM25 results. Each document is expected to include a
            ``"score"`` field representing the original BM25 score.
        semantic_results : list of dict of str to Any
            Ranked semantic results. Each document is expected to include a
            ``"score"`` field representing the original semantic similarity
            score.

        Returns
        -------
        list of dict of str to Any
            Fused result documents sorted by descending hybrid score. Each
            returned document includes:
            - ``"score"``: fused hybrid score
            - ``"hybrid_score"``: same as ``"score"``
            - ``"bm25_score"``: original BM25 score if present, else ``None``
            - ``"semantic_score"``: original semantic score if present, else
              ``None``
            - ``"retrieval_sources"``: list of retrievers that returned the
              document
        """
        fused: dict[str, dict[str, Any]] = {}

        for rank, doc in enumerate(bm25_results, start=1):
            key = self._get_document_key(doc)
            contribution = self.bm25_weight / (self.rrf_k + rank)

            if key not in fused:
                fused[key] = doc.copy()
                fused[key]["bm25_score"] = doc.get("score")
                fused[key]["semantic_score"] = None
                fused[key]["retrieval_sources"] = ["bm25"]
                fused[key]["hybrid_score"] = 0.0
            else:
                if "bm25" not in fused[key]["retrieval_sources"]:
                    fused[key]["retrieval_sources"].append("bm25")
                fused[key]["bm25_score"] = doc.get("score")

            fused[key]["hybrid_score"] += contribution

        for rank, doc in enumerate(semantic_results, start=1):
            key = self._get_document_key(doc)
            contribution = self.semantic_weight / (self.rrf_k + rank)

            if key not in fused:
                fused[key] = doc.copy()
                fused[key]["bm25_score"] = None
                fused[key]["semantic_score"] = doc.get("score")
                fused[key]["retrieval_sources"] = ["semantic"]
                fused[key]["hybrid_score"] = 0.0
            else:
                if "semantic" not in fused[key]["retrieval_sources"]:
                    fused[key]["retrieval_sources"].append("semantic")
                fused[key]["semantic_score"] = doc.get("score")

            fused[key]["hybrid_score"] += contribution

        fused_results = list(fused.values())
        fused_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        for doc in fused_results:
            doc["score"] = float(doc["hybrid_score"])

        return fused_results

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Search the hybrid retriever and return the top-ranked fused results.

        Parameters
        ----------
        query : str
            The user query to search for.
        top_k : int, default=5
            The number of final fused results to return.

        Returns
        -------
        list of dict of str to Any
            A list of the top ``top_k`` fused documents sorted by descending
            hybrid score.

        Raises
        ------
        TypeError
            If ``query`` is not a string or if ``top_k`` is not an integer.
        ValueError
            If ``query`` is empty after stripping or if ``top_k`` is less than 1.
        """
        if not isinstance(query, str):
            raise TypeError(f"`query` must be a string, got {type(query).__name__}.")

        if not isinstance(top_k, int):
            raise TypeError(f"`top_k` must be an integer, got {type(top_k).__name__}.")

        if top_k < 1:
            raise ValueError("`top_k` must be at least 1.")

        query = query.strip()
        if not query:
            raise ValueError("`query` must not be empty or whitespace only.")

        n_fetch = min(max(top_k, self.fetch_k), len(self.documents))

        bm25_results = self.bm25_retriever.search(query=query, top_k=n_fetch)
        semantic_results = self.semantic_retriever.search(query=query, top_k=n_fetch)

        fused_results = self._fuse_ranked_results(
            bm25_results=bm25_results,
            semantic_results=semantic_results,
        )

        return fused_results[:top_k]

    def save(self, output_dir: str | Path) -> None:
        """
        Save the hybrid retriever configuration and underlying retrievers.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            Directory where the hybrid retriever artifacts will be saved.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If ``output_dir`` is not a string or ``Path``.
        """
        if not isinstance(output_dir, (str, Path)):
            raise TypeError(
                f"`output_dir` must be a string or Path, got {type(output_dir).__name__}."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "bm25_weight": self.bm25_weight,
            "semantic_weight": self.semantic_weight,
            "rrf_k": self.rrf_k,
            "fetch_k": self.fetch_k,
            "key_field": self.key_field,
        }

        with open(output_dir / "hybrid_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        self.bm25_retriever.save(output_dir / "bm25")
        self.semantic_retriever.save(output_dir / "semantic")

    @classmethod
    def load(cls, output_dir: str | Path) -> "HybridRetriever":
        """
        Load a previously saved hybrid retriever from disk.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            Directory containing the saved hybrid retriever artifacts.

        Returns
        -------
        HybridRetriever
            A reconstructed hybrid retriever instance.

        Raises
        ------
        TypeError
            If ``output_dir`` is not a string or ``Path``.
        FileNotFoundError
            If required artifact files are missing.
        """
        if not isinstance(output_dir, (str, Path)):
            raise TypeError(
                f"`output_dir` must be a string or Path, got {type(output_dir).__name__}."
            )

        output_dir = Path(output_dir)

        config_path = output_dir / "hybrid_config.json"
        bm25_dir = output_dir / "bm25"
        semantic_dir = output_dir / "semantic"

        missing_paths = [
            str(path)
            for path in [config_path, bm25_dir, semantic_dir]
            if not path.exists()
        ]
        if missing_paths:
            raise FileNotFoundError(
                "Missing hybrid artifact file(s)/directory(ies): "
                + ", ".join(missing_paths)
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        bm25_retriever = BM25Retriever.load(bm25_dir)
        semantic_retriever = SemanticRetriever.load(semantic_dir)

        return cls(
            bm25_retriever=bm25_retriever,
            semantic_retriever=semantic_retriever,
            bm25_weight=config["bm25_weight"],
            semantic_weight=config["semantic_weight"],
            rrf_k=config["rrf_k"],
            fetch_k=config["fetch_k"],
            key_field=config["key_field"],
        )