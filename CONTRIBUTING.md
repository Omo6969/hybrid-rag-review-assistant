# Contributing

Thank you for contributing to this DSCI_575 project. This document explains the preferred workflow, coding norms, and practical steps to get started.

## Quickstart
- Fork (if external) and clone the repo, or clone the team repo directly.
- Create a branch for each task (see Branch Naming Convention).
- Work in small commits, open a focused PR, and request at least one reviewer.

## Issues & PRs
- Use GitHub Issues for bugs, feature requests, and tasks. One issue = one task.
- Link issues from PRs and describe the change and rationale in the PR body.
- Do not push directly to `main`, use PRs for all changes.

## Collaboration Workflow
- Follow GitHub Flow: branch from `main`, open PRs, require reviews before merging.
- Assign each issue to a single team member so responsibilities are clear.
- Keep discussions on GitHub Issues or PR comments for traceability.

## Branch Naming Convention
- `feature/<short-description>` - new features
- `fix/<short-description>` - bug fixes
- `docs/<short-description>` - documentation updates
- `test/<short-description>` - tests
- `chore/<short-description>` - maintenance

Examples: `feature/eda-notebook`, `fix/tokenization-bug`, `docs/readme-update`.

## Development Setup
Clone the repository and set up a reproducible environment. Examples:

Conda:
```bash
git clone git@github.com:UBC-MDS/DSCI_575_project_omo001_deepray.git
cd DSCI_575_project_omo001_deepray
conda env create -f environment.yml
conda activate amazon-retrieval
```

Pip + venv:
```bash
git clone git@github.com:UBC-MDS/DSCI_575_project_omo001_deepray.git
cd DSCI_575_project_omo001_deepray
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Run the app or notebooks as described in `README.md` (example: `streamlit run app/app.py`).

## Reproducibility
- Keep dependencies up to date in `requirements.txt` or `environment.yml`.
- Update `README.md` when setup or run steps change.

## Code Style & Tests
- Follow PEP8; prefer clear, modular functions over monolithic scripts.
- Separate visualization code from data processing and modeling.
- Add small verification scripts or tests for non-trivial changes.

## Data, Secrets & Licensing
- Never commit raw data or secrets. Keep raw files under `data/raw/` locally and add them to `.gitignore`.
- Store environment values in `.env` and commit a `.env.template` with placeholders.
- Respect third-party licenses and cite models/packages in `README.md`.

## Pull Request Checklist
- The change runs locally and is tested.
- The PR links the related issue (when applicable).
- Changes are focused and scoped to a single concern.
- At least one teammate is requested to review.
- No direct commits were made to `main`.

## Collaboration Norms
- Keep PRs small, request reviews early.
- Scope branches to a single concern and avoid unrelated commits.
- If collaboration problems recur, open an issue to adjust the process.

---

Thank you, your contributions keep the project reproducible, reviewable, and useful.
