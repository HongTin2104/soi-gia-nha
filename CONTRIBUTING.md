# Contributing

## Setup

```sh
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before opening a pull request

```sh
ruff check .
ruff format --check .
pytest --cov
```

CI runs the same commands on Python 3.11, 3.12 and 3.13, builds the Docker image, and runs
CodeQL. Coverage must stay above the `fail_under` value in `pyproject.toml`; the report shows
which lines your change left untested.

Tests that need the trained model or the app data are skipped when those files are absent, so
the suite runs on a fresh clone.

## Changes that move a number in the README

Every figure in the README comes from a script in `scripts/`. If your change touches text
extraction (`text.py`, `listing.py`) or the model, rerun the matching script, for example
`python scripts/eval_text.py`, and commit the updated report in `reports/`. If you cannot run
the pipeline (it downloads 1.7 GB), say so in the PR and the maintainer will run it.

## Pull requests

- One topic per PR, with a test for the behaviour it adds or fixes.
- Link the issue it closes (`Closes #12`).
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).
