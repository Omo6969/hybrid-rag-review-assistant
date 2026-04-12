# DSCI_575_project_omo001_deepray

## Overview

**Smart Amazon Product Query Assistant** is a retrieval-focused product search system built on the **Amazon Reviews 2023** dataset. The goal of the project is to retrieve relevant Amazon product-review documents from natural-language queries and compare how well different retrieval methods capture user intent.

For **Milestone 1**, the project focuses on **retrieval only**. The system implements:

- **BM25** keyword-based retrieval
- **Semantic search** using sentence embeddings and vector similarity
- qualitative comparison of both methods on a shared query set

The project initially explored two categories:

- **All_Beauty**
- **Health_and_Personal_Care**

After exploratory analysis, **All_Beauty** was selected as the primary category for Milestone 1 retrieval experiments.

Example query types include:

> "fragrance-free moisturizer for sensitive skin"  
> "gentle makeup remover"  
> "skin care product for very dry lips in winter"

By comparing BM25 and semantic retrieval on the same query set, the project highlights the strengths and weaknesses of lexical and embedding-based search for product discovery.

## Project Goals

This project aims to:

- build a reproducible retrieval pipeline on Amazon review and metadata files
- compare **keyword-based** and **semantic** retrieval approaches
- evaluate retrieval quality qualitatively across different query types
- prepare retrieval outputs for later app integration

## Repository Structure

```text
DSCI_575_project_omo001_deepray/
│
├── README.md
├── environment.yml
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── raw/                   # downloaded Amazon .jsonl/.jsonl.gz files (gitignored)
│   └── processed/             # cleaned datasets and saved retrieval artifacts
│
├── notebooks/
│   ├── milestone1_exploration.ipynb
│   └── evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── bm25.py
│   ├── semantic.py
│   ├── preprocessing.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       └── preprocessing.py
│
├── results/
│   └── milestone1_discussion.md
│
└── scripts/
    ├── make_datasets.py
    ├── build_bm25_index.py
    └── build_semantic_index.py
```

## Dataset

This project uses the **Amazon Reviews 2023** dataset from McAuley Lab.

For Milestone 1, the project explores:

- `All_Beauty`
- `Health_and_Personal_Care`

The final retrieval pipeline for Milestone 1 uses the **All_Beauty** category:

- `All_Beauty.jsonl` or `All_Beauty.jsonl.gz`
- `meta_All_Beauty.jsonl` or `meta_All_Beauty.jsonl.gz`

These files should be stored in `data/raw/` and should not be committed to Git.

## Retrieval Workflow

### 1. Data exploration and preprocessing

In `notebooks/milestone1_exploration.ipynb`, the project:

- inspects review and metadata records
- examines available fields and missingness
- compares candidate categories
- justifies the selected retrieval fields
- explains preprocessing decisions

The final processed retrieval dataset is built separately using `scripts/make_datasets.py`, rather than depending on notebook execution.

### 2. BM25 retrieval

The BM25 pipeline:

- tokenizes the document corpus
- preprocesses queries consistently with documents
- ranks documents by BM25 relevance score
- saves reusable BM25 artifacts to disk

### 3. Semantic retrieval

The semantic retrieval pipeline:

- generates dense sentence embeddings for retrieval documents
- indexes them with FAISS
- retrieves semantically similar results for natural-language queries
- saves reusable semantic retrieval artifacts to disk

### 4. Qualitative evaluation

The project creates a diverse set of queries spanning easy, medium, and complex search intent, then compares BM25 and semantic retrieval on the same query set.

The evaluation workflow is documented in:

- `notebooks/evaluation.ipynb`

The final written discussion and observations are recorded in:

- `results/milestone1_discussion.md`

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/UBC-MDS/DSCI_575_project_omo001_deepray
cd DSCI_575_project_omo001_deepray
```

### 2. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate amazon-retrieval
```

### 3. Install the project in editable mode

```bash
pip install -e .
```

## Data Setup

Place the selected raw data files in:

```text
data/raw/
```

At minimum, for Milestone 1, the following files should be available:

```text
data/raw/All_Beauty.jsonl
or
 data/raw/All_Beauty.jsonl.gz

data/raw/meta_All_Beauty.jsonl
or
 data/raw/meta_All_Beauty.jsonl.gz
```

## Build the Processed Dataset

Run the dataset-building script to generate the cleaned retrieval dataset:

```bash
python scripts/make_datasets.py
```

This creates processed outputs in:

```text
data/processed/
```

including files such as:

```text
All_Beauty_clean.parquet
All_Beauty_clean.jsonl
```

## Build Retrieval Artifacts

### Build BM25 artifacts

```bash
python scripts/build_bm25_index.py
```

This saves BM25 artifacts under:

```text
data/processed/bm25_index/
```

### Build semantic retrieval artifacts

```bash
python scripts/build_semantic_index.py
```

This saves semantic retrieval artifacts under:

```text
data/processed/semantic_index/
```

These saved artifacts make later runs faster because the retrievers can be loaded instead of rebuilt.

## Running the Notebooks

Start Jupyter:

```bash
jupyter lab
```

Then open:

```text
notebooks/milestone1_exploration.ipynb
```

for exploratory analysis and preprocessing decisions, and:

```text
notebooks/evaluation.ipynb
```

for retrieval comparison and qualitative evaluation.

## Reproducibility Notes

To reproduce this project successfully:

- create the environment from `environment.yml`
- install the project with `pip install -e .`
- place the raw dataset files in `data/raw/`
- run `scripts/make_datasets.py` to build the cleaned dataset
- run the BM25 and semantic indexing scripts to save retrieval artifacts
- open the notebooks as needed for EDA and evaluation
- keep raw data and secrets out of version control

## Current Milestone Scope

This repository currently targets **Milestone 1**, which focuses on:

- retrieval foundations
- qualitative evaluation
- reproducible indexing workflows

This milestone does **not** use LLMs yet.

## Contributors

- **omo001**
- **deepray**

## License

This repository was developed for the DSCI 575 course project at UBC.
It is intended for course-related use within DSCI 575 and should not be
copied, redistributed, or reused outside the course context without permission.
