from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_reviews(path: str | Path) -> pd.DataFrame:
    """Load a review dataset from CSV, parquet, or JSONL.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the dataset file.

    Returns
    -------
    pandas.DataFrame
        Loaded review dataframe.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is unsupported.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=True)

    raise ValueError(
        "Unsupported dataset format. Please provide a CSV, parquet, or JSONL file."
    )