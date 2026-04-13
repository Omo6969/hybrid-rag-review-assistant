from typing import Iterable, Sequence, Set, List


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Compute Precision@k.

    Precision@k is the fraction of the top-``k`` retrieved documents that are
    relevant.

    Parameters
    ----------
    retrieved : Sequence[str]
        Ordered list of retrieved document identifiers (most relevant first).
    relevant : set of str
        Set of relevant document identifiers for the query.
    k : int
        Cut-off rank. Must be a positive integer.

    Returns
    -------
    float
        Precision at rank ``k`` (in range ``[0.0, 1.0]``).

    Raises
    ------
    ValueError
        If ``k`` is less than 1.

    Examples
    --------
    >>> precision_at_k(['d1','d2','d3'], {'d2','d4'}, 2)
    0.5
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    topk = retrieved[:k]
    if not topk:
        return 0.0
    return sum(1 for d in topk if d in relevant) / len(topk)


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Compute Recall@k.

    Recall@k is the fraction of all relevant documents that appear in the
    top-``k`` retrieved list.

    Parameters
    ----------
    retrieved : Sequence[str]
        Ordered list of retrieved document identifiers (most relevant first).
    relevant : set of str
        Set of relevant document identifiers for the query.
    k : int
        Cut-off rank. Only the top-``k`` retrieved documents are considered.

    Returns
    -------
    float
        Recall at rank ``k`` (in range ``[0.0, 1.0]``). Returns ``0.0`` when
        ``relevant`` is empty.

    Examples
    --------
    >>> recall_at_k(['d1','d2','d3'], {'d2','d4'}, 3)
    0.5
    """
    if not relevant:
        return 0.0
    topk = set(retrieved[:k])
    return len(topk & relevant) / len(relevant)


def average_precision(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Compute Average Precision (AP) up to rank ``k`` for a single query.

    AP is the average of precision values computed at the ranks where relevant
    documents are found, truncated at ``k``.

    Parameters
    ----------
    retrieved : Sequence[str]
        Ordered list of retrieved document identifiers (most relevant first).
    relevant : set of str
        Set of relevant document identifiers for the query.
    k : int
        Cut-off rank. Must be a positive integer.

    Returns
    -------
    float
        Average precision in range ``[0.0, 1.0]``. Returns ``0.0`` when no
        relevant documents are found in the top-``k`` results.

    Raises
    ------
    ValueError
        If ``k`` is less than 1.

    Examples
    --------
    >>> average_precision(['d1','d2','d3'], {'d2','d3'}, 3)
    0.75
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    num_rel = 0
    score = 0.0
    for i, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            num_rel += 1
            score += num_rel / i
    if num_rel == 0:
        return 0.0
    return score / num_rel


def mean_reciprocal_rank(retrieved: Sequence[str], relevant: Set[str]) -> float:
    """Compute Mean Reciprocal Rank (MRR) for a single query.

    For a single query this is the reciprocal of the rank position of the
    first relevant document in ``retrieved``. If no relevant document is found,
    returns ``0.0``.

    Parameters
    ----------
    retrieved : Sequence[str]
        Ordered list of retrieved document identifiers (most relevant first).
    relevant : set of str
        Set of relevant document identifiers for the query.

    Returns
    -------
    float
        Reciprocal rank of the first relevant document, or ``0.0`` if none are
        present.

    Examples
    --------
    >>> mean_reciprocal_rank(['d1','d2','d3'], {'d2'})
    0.5
    """
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def evaluate_query(retrieved: Sequence[str], relevant: Set[str], k: int = 5) -> dict:
    """Compute a set of standard retrieval metrics for a single query.

    This convenience function computes Precision@k, Recall@k, Average
    Precision (AP) up to ``k``, and Mean Reciprocal Rank (MRR) based on the
    provided retrieved ranking and the set of relevant documents.

    Parameters
    ----------
    retrieved : Sequence[str]
        Ordered list of retrieved document identifiers (most relevant first).
    relevant : set of str
        Set of relevant document identifiers for the query.
    k : int, optional
        Cut-off rank for metrics that use ``k`` (default is 5).

    Returns
    -------
    dict
        A dictionary with keys:

        - ``precision@k``: float
        - ``recall@k``: float
        - ``avg_precision``: float
        - ``mrr``: float

    Examples
    --------
    >>> evaluate_query(['d1','d2','d3'], {'d2','d4'}, k=3)
    {'precision@k': 0.3333333333333333, 'recall@k': 0.5, 'avg_precision': 0.5, 'mrr': 0.5}
    """
    return {
        "precision@k": precision_at_k(retrieved, relevant, k),
        "recall@k": recall_at_k(retrieved, relevant, k),
        "avg_precision": average_precision(retrieved, relevant, k),
        "mrr": mean_reciprocal_rank(retrieved[:k], relevant),
    }
