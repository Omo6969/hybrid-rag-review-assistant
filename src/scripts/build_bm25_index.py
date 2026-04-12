from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.bm25 import BM25Retriever
from src.preprocessing import find_repo_root


def main() -> None:
    """
    Build and save the BM25 retrieval artifacts.

    The script loads the processed retrieval dataset, constructs the BM25
    retriever, and saves the retriever artifacts for later reuse.
    """
    repo_root = find_repo_root()
    
    data_path = repo_root / Path("data/processed/All_Beauty_clean.parquet")
    output_dir = repo_root / Path("data/processed/bm25_index")
    output_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        output_dir / "bm25_documents.pkl",
        output_dir / "bm25_tokenized_corpus.pkl",
        output_dir / "bm25_index.pkl",
    ]

    if all(path.exists() for path in required_files):
        retriever = BM25Retriever.load(output_dir)
        print(f"Loaded existing BM25 artifacts from: {output_dir}")
        print(f"Indexed {len(retriever.documents)} documents.")
        return

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {data_path}. "
            "Run scripts/make_datasets.py first."
        )

    df = pd.read_parquet(data_path)
    if df.empty:
        raise ValueError("Processed dataset is empty.")

    required_cols = {"doc_id", "parent_asin", "asin", "title", "rating", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Processed dataset is missing required columns: {sorted(missing)}"
        )

    documents = df.to_dict(orient="records")

    retriever = BM25Retriever(documents)
    retriever.save(output_dir)

    print(f"Built and saved BM25 artifacts to: {output_dir}")
    print(f"Indexed {len(documents)} documents.")

    print(f"Saved BM25 artifacts to: {output_dir}")
    print(f"Indexed {len(documents)} documents.")


if __name__ == "__main__":
    main()
