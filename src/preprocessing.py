from __future__ import annotations

from pathlib import Path
import json
import re

import pandas as pd

from src.utils.preprocessing import tokenize_text, normalize_text


def find_repo_root(max_up: int = 6) -> Path:
    """
    Locate the repository root by searching parent directories.

    A directory is considered the repository root if it contains either a
    `.git` directory or a `README.md` file.

    Parameters
    ----------
    max_up : int, default=6
        Maximum number of parent directories to search upward from the
        current working directory.

    Returns
    -------
    Path
        Path to the detected repository root. If no root is found within the
        search limit, the current working directory is returned.

    Raises
    ------
    ValueError
        If `max_up` is negative.
    """
    if max_up < 0:
        raise ValueError("max_up must be non-negative.")

    current = Path.cwd()

    for _ in range(max_up + 1):
        if (current / ".git").exists() or (current / "README.md").exists():
            return current
        if current.parent == current:
            break
        current = current.parent

    return Path.cwd()


def load_jsonl(path: Path) -> pd.DataFrame:
    """
    Load a JSONL file into a pandas DataFrame.

    The function first attempts to read the file with `pandas.read_json`.
    If that fails due to formatting issues, it falls back to line-by-line
    JSON parsing.

    Parameters
    ----------
    path : Path
        Path to the JSONL file.

    Returns
    -------
    pd.DataFrame
        Parsed records as a DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file exists but contains no valid JSON records.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    try:
        df = pd.read_json(path, lines=True)
    except ValueError:
        records = []
        with open(path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} in {path}"
                    ) from exc
        df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(f"No valid records found in {path}")

    return df


def summarize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize completeness and data types for each DataFrame column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        Column summary containing column name, non-null count, null count,
        dtype, and non-null percentage.

    Raises
    ------
    ValueError
        If the input DataFrame is empty.
    """
    if df.empty:
        raise ValueError("Cannot summarize columns of an empty DataFrame.")

    summary = pd.DataFrame(
        {
            "column": df.columns,
            "non_null_count": [df[col].notna().sum() for col in df.columns],
            "null_count": [df[col].isna().sum() for col in df.columns],
            "dtype": [str(df[col].dtype) for col in df.columns],
        }
    )
    summary["non_null_pct"] = (
        summary["non_null_count"] / len(df)
    ).round(3)

    return summary.sort_values(
        by="non_null_count", ascending=False
    ).reset_index(drop=True)


def simple_clean(text: object) -> str:
    """
    Apply lightweight text cleaning for retrieval.

    Cleaning steps:
    - convert to lowercase
    - remove HTML tags
    - remove URLs
    - normalize whitespace

    Parameters
    ----------
    text : object
        Input text-like value.

    Returns
    -------
    str
        Cleaned text string.
    """
    if pd.isna(text):
        return ""

    cleaned = str(text).lower()
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"http\\S+|www\\.\\S+", " ", cleaned)
    cleaned = re.sub(r"\\s+", " ", cleaned)

    return cleaned.strip()


def build_retrieval_dataframe(
    reviews_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    join_col: str = "parent_asin",
    clean_text: bool = True,
) -> pd.DataFrame:
    """
    Build a compact retrieval dataset by merging reviews with metadata.

    The resulting dataset uses review text as the primary retrieval signal,
    with metadata retained only when available and useful.

    Parameters
    ----------
    reviews_df : pd.DataFrame
        Review-level DataFrame. Must contain `asin`, `parent_asin`, `title`,
        `text`, and `rating`.
    meta_df : pd.DataFrame
        Metadata DataFrame used for optional enrichment.
    join_col : str, default="parent_asin"
        Column used to merge reviews and metadata.
    clean_text : bool, default=True
        Whether to normalize the combined text field.

    Returns
    -------
    pd.DataFrame
        Retrieval-ready DataFrame containing:
        `doc_id`, `parent_asin`, `asin`, `title` (the review's own headline),
        `product_title` (the product's actual name, from `meta_df`, empty
        string if unavailable), `rating`, and `text`.

    Raises
    ------
    KeyError
        If required columns are missing from `reviews_df`.
    ValueError
        If no valid retrieval records remain after filtering.
    """
    required_review_cols = {"asin", "parent_asin", "title", "text", "rating"}
    missing = required_review_cols - set(reviews_df.columns)
    if missing:
        raise KeyError(
            f"reviews_df is missing required columns: {sorted(missing)}"
        )

    merged_df = reviews_df.merge(
        meta_df,
        on=join_col,
        how="left",
        suffixes=("_review", "_meta"),
    )

    # The product's real name comes from metadata ("title_meta" after the
    # merge suffix), not from the review's own headline ("title_review") --
    # the two are easy to conflate but serve very different purposes: the
    # review headline is user-written ("This stuff is your friend!") while
    # product_title is what a shopper would recognize as the item name. Both
    # are kept: `title` for the review headline, `product_title` for the
    # actual product, so RAG generation and the UI can refer to products by
    # name instead of an opaque ASIN or a review number.
    product_title_col = (
        merged_df["title_meta"] if "title_meta" in merged_df.columns else ""
    )

    retrieval_df = pd.DataFrame(
        {
            "parent_asin": merged_df["parent_asin"],
            "asin": merged_df["asin"],
            "title": merged_df["title_review"].fillna("").astype(str),
            "product_title": pd.Series(product_title_col, index=merged_df.index)
            .fillna("")
            .astype(str),
            "review_text": merged_df["text"].fillna("").astype(str),
            "rating": pd.to_numeric(merged_df["rating"], errors="coerce"),
        }
    )

    retrieval_df["doc_id"] = (
        retrieval_df["parent_asin"].astype(str) + "_" + retrieval_df.index.astype(str)
    )

    retrieval_df["text"] = (
        retrieval_df["title"].str.cat(retrieval_df["review_text"], sep=" ").str.strip()
    )

    if clean_text:
        retrieval_df["text"] = retrieval_df["text"].apply(normalize_text)

    final_df = retrieval_df[
        ["doc_id", "parent_asin", "asin", "title", "product_title", "rating", "text"]
    ].copy()

    # Remove rows with empty or non-tokenizable text.
    before_count = len(final_df)

    final_df["text"] = final_df["text"].fillna("").astype(str)
    final_df = final_df[final_df["text"].str.strip() != ""].copy()
    final_df = final_df[
        final_df["text"].apply(lambda x: len(tokenize_text(x)) > 0)
    ].copy()

    removed_count = before_count - len(final_df)
    if removed_count > 0:
        print(f"Removed {removed_count} rows with empty or non-tokenizable text.")

    validate_retrieval_dataframe(final_df)

    return final_df


def validate_retrieval_dataframe(df: pd.DataFrame) -> None:
    """
    Validate a retrieval-ready DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Retrieval dataset to validate.

    Raises
    ------
    ValueError
        If required columns are missing, `doc_id` is not unique, or all text
        values are empty.
    """
    required_cols = {"doc_id", "parent_asin", "asin", "title", "rating", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Retrieval DataFrame is missing required columns: {sorted(missing)}"
        )

    if not df["doc_id"].is_unique:
        raise ValueError("`doc_id` values must be unique.")

    if df["text"].fillna("").str.strip().eq("").all():
        raise ValueError("All values in `text` are empty.")

    if len(df) == 0:
        raise ValueError("Retrieval DataFrame is empty.")