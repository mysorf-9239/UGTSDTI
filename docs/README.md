# UGTSDTI Documentation

Sphinx documentation for `UGTSDTI` — Uncertainty-Gated Teacher–Student Learning for Drug–Target Interaction Prediction.

## Local build

```bash
cd docs
pip install -r requirements.txt
sphinx-autobuild --watch ../ugtsdti --open-browser source build
```

## Docker build

From the project root:

```bash
docker build --file docs/Dockerfile --tag ugtsdti-docs .
docker run -it --rm -p 8000:8000 ugtsdti-docs
```

Browse at http://localhost:8000.

Mount the working directory to rebuild on file changes:

```bash
docker run -it --rm -p 8000:8000 -v "${PWD}:/ugtsdti" ugtsdti-docs
```

## One-shot HTML build

```bash
cd docs
make html
# output: docs/build/html/index.html
```
