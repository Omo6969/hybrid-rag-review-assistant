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


