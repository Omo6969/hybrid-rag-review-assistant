"""Data utilities: loading and sampling .jsonl.gz Amazon review files.

Functions:
- read_jsonl_gz(path, n=None) -> yields dicts or returns pandas.DataFrame
- save_sample(input_path, out_path, n=200) -> writes first n records to out_path

Usage (example):
python -m src.data_utils --input data/raw/Wireless.jsonl.gz --out data/processed/sample_wireless.jsonl --n 200
"""
from pathlib import Path
import gzip
import json
import argparse
from typing import Iterator, Dict, Optional
import pandas as pd


def read_jsonl_gz(path: str, n: Optional[int] = None) -> Iterator[Dict]:
    """Yield records from a .jsonl.gz file. If n is set, stop after n records."""
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if n is not None and i >= n:
                break
            if not line.strip():
                continue
            yield json.loads(line)


def save_sample(input_path: str, out_path: str, n: int = 200) -> None:
    """Save first `n` records from input_path to out_path (jsonl).

    out_path will be created (parent dirs too). Does not compress output.
    """
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as out_f:
        for rec in read_jsonl_gz(input_path, n=n):
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def to_dataframe(input_path: str, n: Optional[int] = None) -> pd.DataFrame:
    """Load up to n records into a pandas DataFrame."""
    rows = list(read_jsonl_gz(input_path, n=n))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Sample Amazon Reviews jsonl.gz files")
    parser.add_argument("--input", required=True, help="Path to input .jsonl.gz file")
    parser.add_argument("--out", required=True, help="Path to output sample .jsonl file")
    parser.add_argument("--n", type=int, default=200, help="Number of records to sample")
    args = parser.parse_args()
    save_sample(args.input, args.out, args.n)


if __name__ == "__main__":
    main()
