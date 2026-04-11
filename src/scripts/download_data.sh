#!/usr/bin/env bash
# Download full gz files for selected categories into data/raw/
# Usage: bash scripts/download_data.sh All_Beauty Beauty_and_Personal_Care

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <Category1> [Category2 ...]"
  exit 1
fi

mkdir -p data/raw

for cat in "$@"; do
  review_url="https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/${cat}.jsonl.gz"
  meta_url="https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/meta_${cat}.jsonl.gz"
  out_review="data/raw/${cat}.jsonl.gz"
  out_meta="data/raw/meta_${cat}.jsonl.gz"

  echo "Downloading ${cat} reviews to ${out_review}..."
  curl -L -o "${out_review}" "${review_url}"

  echo "Downloading ${cat} metadata to ${out_meta}..."
  curl -L -o "${out_meta}" "${meta_url}"
done

echo "Download finished. Remember: do not commit files in data/raw/ to Git." 
