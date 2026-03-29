.PHONY: setup setup-cpu setup-gpu setup-kaggle update clean \
        lint typecheck test test-pbt pre-commit \
        train eval debug validate-config sweep validate-data

ENV ?= dev

# ── Environment ──────────────────────────────────────────────────────────────

setup:
	conda env create -f conda-recipes/$(ENV).yaml
	conda run -n ugtsdti pre-commit install

setup-cpu:
	conda env create -f conda-recipes/cpu.yaml

setup-gpu:
	conda env create -f conda-recipes/gpu.yaml

setup-kaggle:
	conda env create -f conda-recipes/kaggle.yaml

update:
	conda env update -f conda-recipes/$(ENV).yaml --prune

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf build/ dist/ *.egg-info

# ── Code quality ─────────────────────────────────────────────────────────────

lint:
	ruff check ugtsdti/ tests/

typecheck:
	mypy ugtsdti/ --strict

pre-commit:
	pre-commit run --all-files

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-pbt:
	pytest tests/pbt/ -v --hypothesis-seed=42

# ── Pipeline commands ─────────────────────────────────────────────────────────

train:
	python -m ugtsdti train --config configs/baseline.yaml

eval:
	python -m ugtsdti eval --config configs/baseline.yaml --checkpoint artifacts/model.pt

debug:
	python -m ugtsdti train --config configs/minimal.yaml --dry-run

validate-config:
	python -m ugtsdti validate --config configs/baseline.yaml

sweep:
	python -m ugtsdti sweep --config configs/sweep.yaml

validate-data:
	python -m ugtsdti validate --data --config configs/baseline.yaml
