import re
from typing import List


def normalize_text(text: str) -> str:
    """Lowercase and remove extra whitespace."""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> List[str]:
    """Simple tokenizer for BM25."""
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return tokens
