.PHONY: setup update clean format lint test

setup:
	conda env create -f conda-recipes/full.yaml
	conda run -n ugtsdti pre-commit install

update:
	conda env update -f conda-recipes/full.yaml --prune

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info

format:
	black ugtsdti/ tests/ scripts/
	isort ugtsdti/ tests/ scripts/

lint:
	flake8 ugtsdti/ tests/
	mypy ugtsdti/
	ruff check ugtsdti/ tests/

test:
	pytest tests/ -v
