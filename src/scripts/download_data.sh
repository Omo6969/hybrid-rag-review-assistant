#!/usr/bin/env bash
# Download selected Amazon Reviews 2023 category files from Hugging Face into data/raw/
#
# Usage:
#   bash src/scripts/download_data.sh All_Beauty Health_and_Personal_Care
#   bash src/scripts/download_data.sh --force All_Beauty

set -euo pipefail

force=false

if [ "$#" -lt 1 ]; then
  existing_count=$(find data/raw -maxdepth 1 -type f -name "*.jsonl" 2>/dev/null | wc -l | tr -d ' ')

  if [ "$existing_count" -gt 0 ]; then
    echo "Found ${existing_count} existing data file(s) in data/raw/."
    echo "Nothing to download unless you provide category names."
    echo "Use --force with category names to re-download existing files."
  else
    echo "No category names provided and no existing downloaded data was found."
  fi

  echo "Usage: $0 [--force] <Category1> [Category2 ...]"
  exit 1
fi

mkdir -p data/raw

for cat in "$@"; do
  review_url="https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/${cat}.jsonl?download=true"
  meta_url="https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_${cat}.jsonl?download=true"

  out_review="data/raw/${cat}.jsonl"
  out_meta="data/raw/meta_${cat}.jsonl"

  if [ -f "${out_review}" ] && [ "$force" = false ]; then
    echo "Skipping ${out_review} because it already exists."
  else
    echo "Downloading ${cat} reviews to ${out_review}..."
    curl -L --fail -o "${out_review}" "${review_url}"
  fi

  if [ -f "${out_meta}" ] && [ "$force" = false ]; then
    echo "Skipping ${out_meta} because it already exists."
  else
    echo "Downloading ${cat} metadata to ${out_meta}..."
    curl -L --fail -o "${out_meta}" "${meta_url}"
  fi
done

echo "Download finished. Remember: do not commit files in data/raw/ to Git."