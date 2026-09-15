# Deploying to Posit Connect Cloud

This app is deployed to **[Posit Connect Cloud](https://connect.posit.cloud)**, Posit's
git-connected hosting service for Python/R apps (distinct from
[posit.cloud](https://posit.cloud), which is an interactive RStudio/Jupyter workspace and
isn't meant for always-on public app hosting).

## Why `deploy/sample_data/`

The full retrieval pipeline is built from **701,092 cleaned All Beauty records**
(see `results/final_discussion.md`), and its BM25/semantic artifacts under
`data/processed/` are intentionally gitignored because they are large and
reproducible from raw data (`data/raw/`, which is also gitignored).

Posit Connect Cloud deploys straight from this GitHub repo with no separate
data volume, so the app needs *something* to load at startup. `deploy/sample_data/`
holds a small, prebuilt BM25 + semantic (FAISS) index built from a **4,000-review
random sample** of the same All Beauty dataset, kept small enough (~18 MB) to
commit directly to git and to load comfortably on a free-tier instance.

`app/app.py` prefers the full indices at `data/processed/` when present (e.g. for
local development after running the full pipeline per the main README) and
automatically falls back to `deploy/sample_data/` when they aren't -- which is
what happens on a fresh Connect Cloud deployment. No code changes are needed to
switch between the two; it's purely based on which directories exist.

### Rebuilding the sample

The sample was generated with a one-off script (not part of the normal
pipeline) that:

1. Downloads a partial slice of `All_Beauty.jsonl` / `meta_All_Beauty.jsonl`
   from the [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
   dataset (or reuses full files already in `data/raw/`).
2. Runs them through the existing `src.preprocessing.build_retrieval_dataframe`.
3. Randomly samples 4,000 rows (`random_state=42`).
4. Builds and saves `BM25Retriever` and `SemanticRetriever` artifacts to
   `deploy/sample_data/`.

To regenerate it with a different sample size, adjust `MAX_DOCS` and rerun the
equivalent of `src/scripts/build_bm25_index.py` / `build_semantic_index.py`
pointed at `deploy/sample_data/` instead of `data/processed/`.

## Publishing (one-time setup)

1. Go to **[connect.posit.cloud](https://connect.posit.cloud)** and sign in
   with GitHub.
2. Click **Publish** → **Python Shiny** (or "Publish from GitHub") and select
   this repository and branch.
3. Set the **primary file** to `app/app.py` (the app lives in `app/`, not the
   repo root).
4. Connect Cloud auto-detects `requirements.txt` at the repo root for
   dependencies -- no `environment.yml`/conda support, hence the separate
   pip-only `requirements.txt`.
5. Under the app's **Environment Variables / Secrets** settings, add:
   - `GROQ_API_KEY` -- your key from [console.groq.com](https://console.groq.com)
   - `GROQ_MODEL` -- e.g. `llama-3.3-70b-versatile` (optional; defaults to this
     if unset)
   Do **not** commit a real `.env` file -- secrets are set through the Connect
   Cloud UI only.
6. Deploy. Subsequent pushes to the connected branch redeploy automatically.

## Known limitations of this deployment

- **Sample data only.** Retrieval quality reflects a 4,000-review sample, not
  the full 701k-row corpus used for the evaluation in
  `results/final_discussion.md`. This is a demo/showcase deployment, not a
  reproduction of the full-scale evaluation.
- **Ephemeral feedback log.** `data/processed/feedback.csv` (👍/👎 logging in
  `app/app.py`) is written to local disk, which is not guaranteed to persist
  across redeploys/restarts on Connect Cloud.
- **Free-tier sleep.** Free Connect Cloud apps may sleep after inactivity and
  take a few seconds to wake on the next request.
