SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

UV   ?= uv
PY   ?= python
RUFF ?= ruff

TREE_IGNORE := .venv|.ruff_cache|.pytest_cache|.mypy_cache|__pycache__|*.pyc|.git|dist|build|*.egg-info|logs|datasets|reports|models|assets|.vscode|.idea|pure_ae_fig|fig|pure_ae_reports|data|.github

.PHONY: help lock sync sync-prod sync-tf sync-all \
        run test test-cov lint fmt style \
        clean tree

help: ## Show available targets
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make lock"
	@echo "  make sync"
	@echo "  make sync-tf"
	@echo "  make test"
	@echo "  make run"

lock: ## Resolve + write uv.lock
	$(UV) lock

sync: ## Sync env (installs dev dependency group by default)
	$(UV) sync

sync-prod: ## Sync env without dev dependencies (production-ish)
	$(UV) sync --no-dev

sync-tf: ## Sync env with TensorFlow extra (platform-specific via markers)
	$(UV) sync --extra tf

sync-all: ## Sync env with all optional extras enabled
	$(UV) sync --all-extras

run: ## Run main entrypoint (adjust if you add a proper console script)
	$(UV) run $(PY) main.py

test: ## Run tests
	$(UV) run pytest -q
# 	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
# 	NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \

test-cov: ## Run tests with coverage
	$(UV) run pytest --cov=cade --cov-report=term-missing --cov-report=xml
# 	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
# 	NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \

lint: ## Lint with ruff and apply safe auto-fixes
	$(UV) run $(RUFF) check .
	uv run $(RUFF) check . --fix

clean: ## Remove common caches/artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache coverage.xml htmlcov **/__pycache__ **/*.pyc dist build *.egg-info

tree: ## Print repo tree (ignoring common dirs)
	tree -a --dirsfirst -I "$(TREE_IGNORE)" .

build: ## Build sdist/wheel
	$(UV) build
	
preflight: ## Build + run twine metadata checks
	$(UV) build
	uvx twine check dist/*