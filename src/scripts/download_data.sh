#!/usr/bin/env bash
# Download selected Amazon Reviews 2023 category files from Hugging Face into data/raw/
# Usage: bash src/scripts/download_data.sh All_Beauty Health_and_Personal_Care

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <Category1> [Category2 ...]"
  exit 1
fi

mkdir -p data/raw

for cat in "$@"; do
  review_url="https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/${cat}.jsonl?download=true"
  meta_url="https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_${cat}.jsonl?download=true"

  out_review="data/raw/${cat}.jsonl"
  out_meta="data/raw/meta_${cat}.jsonl"

  echo "Downloading ${cat} reviews to ${out_review}..."
  curl -L --fail -o "${out_review}" "${review_url}"

  echo "Downloading ${cat} metadata to ${out_meta}..."
  curl -L --fail -o "${out_meta}" "${meta_url}"
done

echo "Download finished. Remember: do not commit files in data/raw/ to Git."