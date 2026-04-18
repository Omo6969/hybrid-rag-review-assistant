# src/semantic.py

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    """
    A semantic retriever based on sentence embeddings and FAISS.

    This class builds a semantic search index over a collection of documents,
    where each document is represented as a dictionary containing at least a
    ``"text"`` field. Document texts are embedded with a sentence-transformer
    model, normalized, and indexed in FAISS for efficient similarity search.

    Parameters
    ----------
    documents : list of dict of str to Any
        A list of document dictionaries. Each document must contain a ``"text"``
        key whose value is the text used for embedding and indexing.
    model_name : str, default="sentence-transformers/all-MiniLM-L6-v2"
        Name of the sentence-transformers model used to generate embeddings.

    Attributes
    ----------
    documents : list of dict of str to Any
        The original list of indexed documents.
    model_name : str
        Name of the sentence-transformers model used by the retriever.
    model : SentenceTransformer
        Loaded sentence-transformer model.
    embeddings : numpy.ndarray
        Normalized document embedding matrix of shape ``(n_documents, dim)``.
    index : faiss.Index
        FAISS index built over the normalized document embeddings.

    Raises
    ------
    TypeError
        If ``documents`` is not a list, if any document is not a dictionary,
        or if ``model_name`` is not a string.
    ValueError
        If ``documents`` is empty, if a document is missing the ``"text"``
        key, or if a document's ``"text"`` value is empty after stripping.
    """

    def __init__(
        self,
        documents: list[dict[str, Any]],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """
        Initialize the semantic retriever and build the FAISS index.

        Parameters
        ----------
        documents : list of dict of str to Any
            A list of documents to index. Each document must contain a
            ``"text"`` key used for embedding and search.
        model_name : str, default="sentence-transformers/all-MiniLM-L6-v2"
            Name of the sentence-transformers model used to generate document
            and query embeddings.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If ``documents`` is not a list, if any document is not a
            dictionary, or if ``model_name`` is not a string.
        ValueError
            If ``documents`` is empty, if a document is missing the ``"text"``
            key, or if a document's ``"text"`` value is empty after stripping.
        """
        self._validate_documents(documents)

        if not isinstance(model_name, str):
            raise TypeError(
                f"`model_name` must be a string, got {type(model_name).__name__}."
            )

        self.documents = documents
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        texts = [str(doc["text"]).strip() for doc in documents]
        self.embeddings = self._encode_texts(texts)
        self.index = self._build_index(self.embeddings)

    @staticmethod
    def _validate_documents(documents: list[dict[str, Any]]) -> None:
        """
        Validate the input documents before indexing.

        Parameters
        ----------
        documents : list of dict of str to Any
            Candidate documents for semantic indexing.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If ``documents`` is not a list or if any document is not a
            dictionary.
        ValueError
            If ``documents`` is empty, if a document is missing the ``"text"``
            key, or if a document's ``"text"`` value is empty after stripping.
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

            text = str(doc["text"]).strip()
            if not text:
                raise ValueError(
                    f"Document at index {i} has empty or whitespace-only text."
                )

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """
        Encode and normalize a list of texts into dense embeddings.

        Parameters
        ----------
        texts : list of str
            Texts to embed.

        Returns
        -------
        numpy.ndarray
            A float32 array of normalized embeddings with shape
            ``(n_texts, embedding_dim)``.
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")

        faiss.normalize_L2(embeddings)
        return embeddings

    @staticmethod
    def _build_index(embeddings: np.ndarray) -> faiss.Index:
        """
        Build a FAISS index from normalized embeddings.

        Parameters
        ----------
        embeddings : numpy.ndarray
            A float32 array of normalized embeddings with shape
            ``(n_documents, embedding_dim)``.

        Returns
        -------
        faiss.Index
            A FAISS inner-product index over the embeddings.

        Raises
        ------
        TypeError
            If ``embeddings`` is not a NumPy array.
        ValueError
            If ``embeddings`` is empty or not two-dimensional.
        """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError(
                f"`embeddings` must be a NumPy array, got {type(embeddings).__name__}."
            )

        if embeddings.ndim != 2:
            raise ValueError(
                f"`embeddings` must be 2-dimensional, got shape {embeddings.shape}."
            )

        if embeddings.shape[0] == 0:
            raise ValueError("`embeddings` must contain at least one row.")

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Search the semantic index and return the top-ranked results.

        Parameters
        ----------
        query : str
            The user query to search for.
        top_k : int, default=5
            The number of top-ranked results to return.

        Returns
        -------
        list of dict of str to Any
            A list of the top ``top_k`` documents sorted by descending semantic
            similarity score. Each returned document includes an added
            ``"score"`` field containing the similarity score as a float.

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

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        faiss.normalize_L2(query_embedding)

        n_results = min(top_k, len(self.documents))
        scores, indices = self.index.search(query_embedding, n_results)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            doc = self.documents[idx].copy()
            doc["score"] = float(score)
            results.append(doc)

        return results

    def save(self, output_dir: str | Path) -> None:
        """
        Save the retriever artifacts to disk.

        This method stores the original documents, embedding matrix, FAISS
        index, and model metadata in the specified directory.

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

        with open(output_dir / "semantic_documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        with open(output_dir / "semantic_model_name.json", "w", encoding="utf-8") as f:
            json.dump({"model_name": self.model_name}, f)

        np.save(output_dir / "semantic_embeddings.npy", self.embeddings)
        faiss.write_index(self.index, str(output_dir / "semantic_faiss.index"))

    @classmethod
    def load(cls, output_dir: str | Path) -> "SemanticRetriever":
        """
        Load a previously saved semantic retriever from disk.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            The directory containing the saved semantic retriever artifacts.

        Returns
        -------
        SemanticRetriever
            A reconstructed ``SemanticRetriever`` instance with loaded
            documents, embeddings, model, and FAISS index.

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
            output_dir / "semantic_documents.pkl",
            output_dir / "semantic_model_name.json",
            output_dir / "semantic_embeddings.npy",
            output_dir / "semantic_faiss.index",
        ]

        missing_files = [str(path) for path in required_files if not path.exists()]
        if missing_files:
            raise FileNotFoundError(
                "Missing semantic artifact file(s): " + ", ".join(missing_files)
            )

        with open(output_dir / "semantic_documents.pkl", "rb") as f:
            documents = pickle.load(f)

        with open(output_dir / "semantic_model_name.json", "r", encoding="utf-8") as f:
            model_name = json.load(f)["model_name"]

        embeddings = np.load(output_dir / "semantic_embeddings.npy")
        index = faiss.read_index(str(output_dir / "semantic_faiss.index"))

        instance = cls.__new__(cls)
        instance.documents = documents
        instance.model_name = model_name
        instance.model = SentenceTransformer(model_name)
        instance.embeddings = embeddings
        instance.index = index
        return instance
