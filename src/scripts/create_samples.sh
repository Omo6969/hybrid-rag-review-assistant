#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <DatasetName1> [DatasetName2 ...]"
  echo "Example: $0 All_Beauty meta_All_Beauty Health_and_Personal_Care meta_Health_and_Personal_Care"
  exit 1
fi

mkdir -p data/processed

for dataset_name in "$@"; do
  inpath="data/raw/${dataset_name}.jsonl"
  outpath="data/processed/sample_${dataset_name}.jsonl"

  if [ ! -f "$inpath" ]; then
    echo "Input file not found: $inpath"
    continue
  fi

  echo "Creating sample for ${dataset_name} -> ${outpath} (first 200 records)"
  head -n 200 "$inpath" > "$outpath"
done

echo "Samples created under data/processed/"
