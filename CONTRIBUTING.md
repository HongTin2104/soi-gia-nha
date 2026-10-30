# Contributing

## Setup

```sh
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]" pre-commit
pre-commit install               # ruff and basic checks on every commit
```

The app tests need the trained model and the app data (about 140 MB, from GitHub releases).
Fetch them once:

```sh
python -c "from soi_gia_nha.assets import ensure_data, ensure_model; ensure_model(); ensure_data()"
```

Without them those tests are skipped, so `pytest` still runs on a fresh clone, but coverage
falls below the threshold.

## Before opening a pull request

```sh
pre-commit run --all-files   # ruff lint + format, YAML/TOML checks, private keys, large files
pytest --cov
```

CI runs the same hooks, then the tests on Python 3.11, 3.12 and 3.13 with the model and app data
downloaded, builds the Docker image, and runs CodeQL. Coverage must stay above the `fail_under`
value in `pyproject.toml` (currently 95%); the report shows which lines your change left
untested.

## What the tests cover

- `tests/test_*.py` next to each module: unit tests on small hand-written inputs.
- `tests/test_pipeline.py`: runs every script in `scripts/` in order on a synthetic snapshot in a
  temporary directory, so a change that breaks the hand-off between steps fails here without
  the 1.7 GB dataset.
- `tests/test_app.py`: drives the Streamlit app with `streamlit.testing.v1.AppTest`.

## Changes that move a number in the README

Every figure in the README comes from a script in `scripts/`. If your change touches text
extraction (`text.py`, `listing.py`) or the model, rerun the matching script, for example
`python scripts/eval_text.py`, and commit the updated report in `reports/`. If you cannot run
the pipeline (it downloads 1.7 GB), say so in the PR and the maintainer will run it.

## Pull requests

- One topic per PR, with a test for the behaviour it adds or fixes.
- Link the issue it closes (`Closes #12`).
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).
