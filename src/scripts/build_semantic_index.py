from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing import find_repo_root
from src.semantic import SemanticRetriever


def main() -> None:
    """
    Build and save the semantic retrieval artifacts.

    The script loads the processed retrieval dataset, constructs the semantic
    retriever, and saves the model metadata, embeddings, and FAISS index for
    later reuse.
    """
    repo_root = find_repo_root()

    data_path = repo_root / Path("data/processed/All_Beauty_clean.parquet")
    output_dir = repo_root / Path("data/processed/semantic_index")
    output_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        output_dir / "semantic_documents.pkl",
        output_dir / "semantic_model_name.json",
        output_dir / "semantic_embeddings.npy",
        output_dir / "semantic_faiss.index",
    ]

    if all(path.exists() for path in required_files):
        retriever = SemanticRetriever.load(output_dir)
        print(f"Loaded existing semantic artifacts from: {output_dir}")
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

    retriever = SemanticRetriever(documents)
    retriever.save(output_dir)

    print(f"Built and saved semantic artifacts to: {output_dir}")
    print(f"Indexed {len(documents)} documents.")


if __name__ == "__main__":
    main()
