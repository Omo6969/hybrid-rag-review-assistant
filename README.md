# DSCI_575_project_omo001_deepray

## Overview

**Smart Amazon Product Query Assistant** is a retrieval-focused product search system built on the **Amazon Reviews 2023** dataset. The goal of this project is to help users search for relevant Amazon products using natural-language queries and compare how well different retrieval methods capture user intent.

For **Milestone 1**, the project focuses on **retrieval only**. The system implements:

- **BM25** keyword-based retrieval
- **Semantic search** using sentence embeddings and vector similarity
- A simple **interactive web app** for searching the corpus

Our project uses two Amazon product categories to strike a balance between dataset richness and computational manageability:

- **All_Beauty**
- **Health_and_Personal_Care**

The system is designed to support queries such as:

> "fragrance-free moisturizer for sensitive skin"  
> "gentle cleanser for acne-prone skin"  
> "travel-size skincare products"

By comparing BM25 and semantic retrieval on the same query set, the project highlights the strengths and weaknesses of classical and embedding-based search for product discovery.

## Project Goals

This project aims to:

- Build a reproducible retrieval pipeline on Amazon review and metadata files
- Compare **keyword-based** and **semantic** retrieval approaches
- Evaluate retrieval quality qualitatively across different query types
- Provide a simple interface for interactive search

## Major Repository Structure

```text
DSCI_575_project_omo001_deepray/
│
├── README.md
├── environment.yml
├── requirements.txt
├── .env                       # never commit secrets
│
├── data/
│   ├── raw/                   # downloaded .jsonl.gz files (gitignored)
│   └── processed/             # cleaned data, indexes, embeddings, cached artifacts
│
├── notebooks/
│   └── milestone1_exploration.ipynb
│
├── src/
│   ├── bm25.py
│   ├── semantic.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── preprocessing.py
│       └── corpus.py
│
├── results/
│   └── milestone1_discussion.md
│
├── app/
│   └── app.py
│
└── scripts/
    └── download_data.sh
```

## Dataset

This project uses the **Amazon Reviews 2023** dataset from McAuley Lab.

Selected categories:

- `All_Beauty`
- `Health_and_Personal_Care`

For each category, we use:

- the **review file**: `<Category>.jsonl.gz`
- the **metadata file**: `meta_<Category>.jsonl.gz`

These files are stored in `data/raw/` and are excluded from Git.

## Retrieval Workflow

### 1. Data exploration and preprocessing

In `notebooks/milestone1_exploration.ipynb`, we:

- inspect review and metadata records
- examine available fields
- choose the fields used for retrieval
- justify preprocessing decisions

A retrieval document is constructed by combining review text with relevant product metadata such as title, description, and product features where available.

### 2. BM25 retrieval

The BM25 pipeline:

- tokenizes the corpus
- preprocesses queries consistently with documents
- ranks documents by BM25 relevance score

### 3. Semantic retrieval

The semantic retrieval pipeline:

- generates sentence embeddings for retrieval documents
- indexes them using a vector similarity backend
- retrieves semantically similar results for natural-language queries

### 4. Qualitative evaluation

We create a diverse set of queries and compare BM25 and semantic retrieval on:

- direct keyword queries
- intent-driven semantic queries
- more complex product-search queries

The discussion and observations are recorded in:

- `results/milestone1_discussion.md`

### 5. Web app

The Shiny app provides:

- a query input box
- a retrieval mode selector
- top search results with product title, text snippet, rating, and retrieval score

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/UBC-MDS/DSCI_575_project_omo001_deepray
cd DSCI_575_project_omo001_deepray
```

### 2. Create and activate the environment

```bash
conda env create -f environment.yml
conda activate amazon-retrieval
```

### 3. Install any runtime packages if needed

```bash
python -m pip install -r requirements.txt
```

### 4. Install the project in editable mode

```bash
pip install -e .
```

## Data Download

Download the selected category files into `data/raw/` using:

```bash
bash scripts/download_data.sh All_Beauty Health_and_Personal_Care
```

After downloading, confirm the files exist in:

```text
data/raw/
```

## Running the notebook

Start Jupyter and open the EDA notebook:

```bash
jupyter lab
```

Then open:

```text
notebooks/milestone1_exploration.ipynb
```

---

## Running the app locally

To run the Shiny app locally:

```bash
shiny run --reload app/app.py
```

If needed, you can also run:

```bash
python -m shiny run --reload app/app.py
```

---

## Reproducibility Notes

To reproduce this project successfully:

- create the environment from `environment.yml`
- install runtime packages from `requirements.txt` if necessary
- download the raw dataset files into `data/raw/`
- keep `.env` out of version control
- run the notebook for exploration and preprocessing
- run the app from `app/app.py`

---

## Current Milestone Scope

This repository currently targets **Milestone 1**, which focuses on:

- retrieval foundations
- qualitative evaluation
- a basic retrieval app

This milestone does **not** use LLMs yet.

---

## Contributors

- **omo001**
- **deepray**

---

## Contribution Notes

This project is being developed collaboratively through GitHub commits and incremental milestone submissions. Both contributors should make regular, meaningful commits with descriptive commit messages.

---

## License

This project is for academic use within **DSCI 575**.
