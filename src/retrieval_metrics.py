from typing import Iterable, Sequence, Set, List


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Precision@k: fraction of top-k retrieved that are relevant."""
    if k <= 0:
        raise ValueError("k must be >= 1")
    topk = retrieved[:k]
    if not topk:
        return 0.0
    return sum(1 for d in topk if d in relevant) / len(topk)


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Recall@k: fraction of relevant documents that appear in top-k."""
    if not relevant:
        return 0.0
    topk = set(retrieved[:k])
    return len(topk & relevant) / len(relevant)


def average_precision(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Average precision up to k for a single query."""
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
    """MRR for a single query: reciprocal of rank of first relevant item."""
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def evaluate_query(retrieved: Sequence[str], relevant: Set[str], k: int = 5) -> dict:
    """Return common metrics for a single query."""
    return {
        "precision@k": precision_at_k(retrieved, relevant, k),
        "recall@k": recall_at_k(retrieved, relevant, k),
        "avg_precision": average_precision(retrieved, relevant, k),
        "mrr": mean_reciprocal_rank(retrieved[:k], relevant),
    }
