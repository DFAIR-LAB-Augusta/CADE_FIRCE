# Contributing

Thanks for your interest in contributing!

This repository is a modernization and integration-focused fork of the original CADE implementation:

- Upstream: https://github.com/whyisyoung/CADE
- This fork: https://github.com/DFAIR-LAB-Augusta/CADE_FIRCE

We aim to preserve the original research implementation while improving:
- Packaging (pyproject.toml)
- Tooling (uv, ruff, pytest)
- Reproducibility and CI
- Integration readiness for FIRCE

## Development Setup (uv)

Prereqs:
- Python 3.11+
- uv installed: https://docs.astral.sh/uv/

Install dependencies:
```bash
uv sync
````

Run style checks:

```bash
uv run ruff format --check .
uv run ruff check .
```

Run tests:

```bash
uv run pytest
```

(If you use the Makefile targets:)

```bash
make sync
make style
make test
```

## Branch / PR Workflow

1. Create a feature branch off `main`:

   * `feat/...`, `fix/...`, `chore/...`, `docs/...`
2. Keep PRs focused and small when possible.
3. Ensure CI passes (style + tests).

## Code Style

We use:

* `ruff` for formatting and linting
* `pytest` for tests

Please do not introduce new formatting tools unless discussed.

## Testing

* Add tests for new functionality when reasonable.
* Keep tests deterministic (seed RNGs when needed).

## Licensing and Attribution

This repo includes material derived from the upstream CADE repository. Please:

* Keep upstream attribution intact
* Preserve license headers and licensing terms
* Avoid removing citations or acknowledgements related to the original research

## Questions

Open an issue for questions, or start a draft PR early if you want feedback.

## `.github/workflows/tests.yml`

```yaml
n
```

---

## Quick “add all files” helper (optional)

From repo root:

```bash
mkdir -p .github/ISSUE_TEMPLATE .github/workflows
# then paste files
git add .github
git commit -m "chore: add GitHub meta files and CI (uv/ruff/pytest)"
```

---

If you want one extra nicety next: I can give you a **minimal `tests/` scaffold** (even a single sanity test) so the `tests.yml` pipeline is immediately meaningful and doesn’t pass “empty.”
