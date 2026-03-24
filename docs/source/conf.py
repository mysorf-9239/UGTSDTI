# Configuration file for the Sphinx documentation builder.

import os
import sys

# Allow autodoc to import the package from the repo checkout.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

project = "UGTSDTI"
copyright = "2026, Mysorf"
author = "Mysorf"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "build", "Thumbs.db", ".DS_Store"]

# autodoc
autodoc_member_order = "bysource"
autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "rdkit",
    "tdc",
    "transformers",
    "loguru",
    "wandb",
    "hydra",
    "omegaconf",
    "tqdm",
    "sklearn",
    "numpy",
]

# napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# todo
todo_include_todos = True

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["style.css"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}
