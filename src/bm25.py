# src/bm25.py

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.utils.preprocessing import tokenize_text


class BM25Retriever:
    """
    A BM25-based retriever for keyword search over product-review documents.

    This class builds a BM25 index from a collection of documents, where each
    document is represented as a dictionary containing at least a ``"text"``
    field. Queries are tokenized using the same preprocessing pipeline as the
    documents to ensure consistent retrieval behavior.

    Parameters
    ----------
    documents : list of dict of str to Any
        A list of document dictionaries. Each document must contain a ``"text"``
        key whose value is the text used for indexing. Additional keys such as
        ``"doc_id"``, ``"title"``, ``"rating"``, or ``"review_text"`` may also
        be included and will be returned in search results.

    Attributes
    ----------
    documents : list of dict of str to Any
        The original list of indexed documents.
    tokenized_corpus : list of list of str
        The tokenized representation of the document corpus.
    bm25 : BM25Okapi
        The fitted BM25 index built from the tokenized corpus.

    Raises
    ------
    TypeError
        If ``documents`` is not a list, or if any document is not a dictionary.
    ValueError
        If ``documents`` is empty, if a document is missing the ``"text"`` key,
        or if a document's ``"text"`` value is empty after normalization.
    """

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        """
        Initialize the BM25 retriever and build the index.

        Parameters
        ----------
        documents : list of dict of str to Any
            A list of documents to index. Each document must contain a ``"text"``
            key used for tokenization and indexing.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If ``documents`` is not a list, or if any document is not a
            dictionary.
        ValueError
            If ``documents`` is empty, if a document is missing the ``"text"``
            key, or if a document's ``"text"`` value is empty after
            tokenization.
        """
        self._validate_documents(documents)
        self.documents = documents
        self.tokenized_corpus = [tokenize_text(doc["text"]) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @staticmethod
    def _validate_documents(documents: list[dict[str, Any]]) -> None:
        """
        Validate the input documents before indexing.

        Parameters
        ----------
        documents : list of dict of str to Any
            Candidate documents for BM25 indexing.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If ``documents`` is not a list, or if any document is not a
            dictionary.
        ValueError
            If ``documents`` is empty, if a document is missing the ``"text"``
            key, or if a document's ``"text"`` value is empty after
            tokenization.
        """
        if not isinstance(documents, list):
            raise TypeError(
                f"`documents` must be a list of dictionaries, got {type(documents).__name__}."
            )

        if not documents:
            raise ValueError("`documents` must not be empty.")

        for i, doc in enumerate(documents):
            if not isinstance(doc, dict):
                raise TypeError(
                    f"Each document must be a dictionary. "
                    f"Document at index {i} has type {type(doc).__name__}."
                )

            if "text" not in doc:
                raise ValueError(
                    f"Document at index {i} is missing the required 'text' key."
                )

            if not tokenize_text(doc["text"]):
                raise ValueError(
                    f"Document at index {i} has empty or non-tokenizable text."
                )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Search the BM25 index and return the top-ranked results.

        Parameters
        ----------
        query : str
            The user query to search for.
        top_k : int, default=5
            The number of top-ranked results to return.

        Returns
        -------
        list of dict of str to Any
            A list of the top ``top_k`` documents sorted by descending BM25
            score. Each returned document includes an added ``"score"`` field
            containing the BM25 relevance score as a float.

        Raises
        ------
        TypeError
            If ``query`` is not a string or if ``top_k`` is not an integer.
        ValueError
            If ``query`` is empty after tokenization or if ``top_k`` is less
            than 1.

        Notes
        -----
        The query is tokenized using the same preprocessing pipeline used for
        the document corpus.
        """
        if not isinstance(query, str):
            raise TypeError(f"`query` must be a string, got {type(query).__name__}.")

        if not isinstance(top_k, int):
            raise TypeError(f"`top_k` must be an integer, got {type(top_k).__name__}.")

        if top_k < 1:
            raise ValueError("`top_k` must be at least 1.")

        tokenized_query = tokenize_text(query)
        if not tokenized_query:
            raise ValueError("`query` is empty after preprocessing/tokenization.")

        scores = self.bm25.get_scores(tokenized_query)
        n_results = min(top_k, len(self.documents))
        top_indices = np.argsort(scores)[::-1][:n_results]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc["score"] = float(scores[idx])
            results.append(doc)

        return results

    def save(self, output_dir: str | Path) -> None:
        """
        Save the retriever artifacts to disk.

        This method stores the original documents, tokenized corpus, and BM25
        index as pickle files in the specified directory.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            The directory where the retriever artifacts will be saved.

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

        with open(output_dir / "bm25_documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        with open(output_dir / "bm25_tokenized_corpus.pkl", "wb") as f:
            pickle.dump(self.tokenized_corpus, f)

        with open(output_dir / "bm25_index.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

    @classmethod
    def load(cls, output_dir: str | Path) -> "BM25Retriever":
        """
        Load a previously saved BM25 retriever from disk.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            The directory containing the saved BM25 retriever artifacts.

        Returns
        -------
        BM25Retriever
            A reconstructed ``BM25Retriever`` instance with loaded documents,
            tokenized corpus, and BM25 index.

        Raises
        ------
        TypeError
            If ``output_dir`` is not a string or ``Path``.
        FileNotFoundError
            If one or more required artifact files are missing.
        """
        if not isinstance(output_dir, (str, Path)):
            raise TypeError(
                f"`output_dir` must be a string or Path, got {type(output_dir).__name__}."
            )

        output_dir = Path(output_dir)

        required_files = [
            output_dir / "bm25_documents.pkl",
            output_dir / "bm25_tokenized_corpus.pkl",
            output_dir / "bm25_index.pkl",
        ]

        missing_files = [str(path) for path in required_files if not path.exists()]
        if missing_files:
            raise FileNotFoundError(
                "Missing BM25 artifact file(s): " + ", ".join(missing_files)
            )

        with open(output_dir / "bm25_documents.pkl", "rb") as f:
            documents = pickle.load(f)

        with open(output_dir / "bm25_tokenized_corpus.pkl", "rb") as f:
            tokenized_corpus = pickle.load(f)

        with open(output_dir / "bm25_index.pkl", "rb") as f:
            bm25 = pickle.load(f)

        instance = cls.__new__(cls)
        instance.documents = documents
        instance.tokenized_corpus = tokenized_corpus
        instance.bm25 = bm25
        return instance
