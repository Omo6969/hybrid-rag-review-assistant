from src.preprocessing import (
    load_jsonl,
    build_retrieval_dataframe,
    find_repo_root,
)


def main() -> None:
    repo_root = find_repo_root()

    raw_dir = repo_root / "data" / "raw"
    output_dir = repo_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    reviews_path = raw_dir / "All_Beauty.jsonl"
    meta_path = raw_dir / "meta_All_Beauty.jsonl"

    reviews_df = load_jsonl(reviews_path)
    meta_df = load_jsonl(meta_path)

    final_df = build_retrieval_dataframe(
        reviews_df=reviews_df,
        meta_df=meta_df,
        join_col="parent_asin",
        clean_text=True,
    )

    final_df.to_parquet(output_dir / "All_Beauty_clean.parquet", index=False)
    final_df.to_json(
        output_dir / "All_Beauty_clean.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    print(f"Saved {len(final_df)} records to {output_dir}")


if __name__ == "__main__":
    main()
