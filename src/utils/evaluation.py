from __future__ import annotations

from typing import Any

import pandas as pd


def collect_results(
    retriever: Any,
    queries_df: pd.DataFrame,
    method_name: str,
    top_k: int = 5,
) -> pd.DataFrame:
    """Collect top-k retrieval results for a set of queries.

    Parameters
    ----------
    retriever : Any
        Retriever instance with a ``search(query, top_k=...)`` method.
    queries_df : pandas.DataFrame
        DataFrame containing ``query_id``, ``query``, and ``difficulty``.
    method_name : str
        Name of retrieval method, such as ``"BM25"`` or ``"Semantic"``.
    top_k : int, default=5
        Number of results to collect per query.

    Returns
    -------
    pandas.DataFrame
        Long-format dataframe containing one row per retrieved result.
    """
    rows: list[dict[str, Any]] = []

    for _, row in queries_df.iterrows():
        query_id = row["query_id"]
        query = row["query"]
        difficulty = row["difficulty"]

        results = retriever.search(query, top_k=top_k)

        for rank, result in enumerate(results, start=1):
            rows.append(
                {
                    "method": method_name,
                    "query_id": query_id,
                    "query": query,
                    "difficulty": difficulty,
                    "rank": rank,
                    "doc_id": result.get("doc_id"),
                    "title": result.get("title"),
                    "rating": result.get("rating"),
                    "score": result.get("score"),
                    "text": result.get("text"),
                }
            )

    return pd.DataFrame(rows)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute precision@k.

    Parameters
    ----------
    retrieved_ids : list of str
        Ranked retrieved document IDs.
    relevant_ids : set of str
        Ground-truth relevant document IDs.
    k : int
        Evaluation cutoff.

    Returns
    -------
    float
        Precision at rank k.
    """
    top_k = retrieved_ids[:k]
    if k == 0:
        return 0.0

    hits = sum(doc_id in relevant_ids for doc_id in top_k)
    return hits / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute recall@k.

    Parameters
    ----------
    retrieved_ids : list of str
        Ranked retrieved document IDs.
    relevant_ids : set of str
        Ground-truth relevant document IDs.
    k : int
        Evaluation cutoff.

    Returns
    -------
    float
        Recall at rank k.
    """
    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(doc_id in relevant_ids for doc_id in top_k)
    return hits / len(relevant_ids)


def evaluate_results(
    results_df: pd.DataFrame,
    relevance_judgments: dict[str, set[str]],
    k: int = 5,
) -> pd.DataFrame:
    """Evaluate collected retrieval results using precision@k and recall@k.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Long-format dataframe produced by ``collect_results``.
    relevance_judgments : dict of str to set of str
        Mapping from query text to relevant document IDs.
    k : int, default=5
        Evaluation cutoff.

    Returns
    -------
    pandas.DataFrame
        Per-query evaluation results grouped by retrieval method.
    """
    rows: list[dict[str, Any]] = []

    grouped = results_df.groupby(["method", "query_id", "query", "difficulty"])

    for (method, query_id, query, difficulty), group in grouped:
        group_sorted = group.sort_values("rank")
        retrieved_ids = [
            str(doc_id)
            for doc_id in group_sorted["doc_id"].tolist()
            if pd.notna(doc_id)
        ]
        relevant_ids = relevance_judgments.get(query, set())

        rows.append(
            {
                "retriever": method,
                "query_id": query_id,
                "query": query,
                "difficulty": difficulty,
                "k": k,
                "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k),
                "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k),
            }
        )

    return pd.DataFrame(rows)


def summarize_evaluation(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize evaluation metrics by retriever.

    Parameters
    ----------
    evaluation_df : pandas.DataFrame
        Per-query evaluation dataframe.

    Returns
    -------
    pandas.DataFrame
        Mean precision@k and recall@k per retriever.
    """
    return (
        evaluation_df.groupby("retriever")[["precision_at_k", "recall_at_k"]]
        .mean()
        .reset_index()
        .sort_values(by="precision_at_k", ascending=False)
    )