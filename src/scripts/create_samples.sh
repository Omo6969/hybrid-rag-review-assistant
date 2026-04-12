#!/usr/bin/env bash
# Create small sample JSONL files from downloaded Amazon Reviews 2023 raw files.
#
# This script copies the first N records from each input file in `data/raw/`
# into `data/processed/` using the naming pattern `sample_<DatasetName>.jsonl`.
#
# Usage:
#   bash src/scripts/create_samples.sh All_Beauty meta_All_Beauty
#   bash src/scripts/create_samples.sh --lines 300 All_Beauty
#   bash src/scripts/create_samples.sh --force All_Beauty meta_All_Beauty

set -euo pipefail

force=false
num_lines=200

# Parse optional flags.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      force=true
      shift
      ;;
    --lines)
      if [[ $# -lt 2 ]]; then
        echo "Error: --lines requires a positive integer."
        exit 1
      fi
      num_lines="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--force] [--lines N] <DatasetName1> [DatasetName2 ...]"
      echo "Example:"
      echo "  $0 All_Beauty meta_All_Beauty"
      echo "  $0 --lines 300 --force All_Beauty"
      exit 0
      ;;
    --*)
      echo "Unknown option: $1"
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

# If no dataset names are provided, report existing samples first.
if [ "$#" -lt 1 ]; then
  existing_count=$(find data/processed -maxdepth 1 -type f -name "sample_*.jsonl" 2>/dev/null | wc -l | tr -d ' ')

  if [ "$existing_count" -gt 0 ]; then
    echo "Found ${existing_count} existing sample file(s) in data/processed/."
    echo "Nothing to create unless you provide dataset names."
    echo "Use --force with dataset names to recreate existing samples."
  else
    echo "No dataset names provided and no existing sample files were found."
  fi

  echo "Usage: $0 [--force] [--lines N] <DatasetName1> [DatasetName2 ...]"
  echo "Example: $0 All_Beauty meta_All_Beauty Health_and_Personal_Care meta_Health_and_Personal_Care"
  exit 1
fi

# Validate the requested sample size.
if ! [[ "$num_lines" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --lines must be a positive integer."
  exit 1
fi

# Ensure the processed data directory exists.
mkdir -p data/processed

# Create samples for each requested dataset.
for dataset_name in "$@"; do
  inpath="data/raw/${dataset_name}.jsonl"
  outpath="data/processed/sample_${dataset_name}.jsonl"

  # Skip missing raw input files.
  if [ ! -f "$inpath" ]; then
    echo "Skipping ${dataset_name}: input file not found at ${inpath}"
    continue
  fi

  # Skip existing output unless forced.
  if [ -f "$outpath" ] && [ "$force" = false ]; then
    echo "Skipping ${outpath}: sample already exists."
    continue
  fi

  echo "Creating sample for ${dataset_name} -> ${outpath} (first ${num_lines} records)"
  head -n "$num_lines" "$inpath" > "$outpath"
done

echo "Samples created under data/processed/"