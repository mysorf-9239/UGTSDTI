# Minimal setup.py for legacy compatibility.
# Primary project configuration is in pyproject.toml.
from setuptools import find_packages, setup

setup(
    name="ugtsdti",
    version="0.1.0",
    description="Uncertainty-Gated Teacher-Student for DTI",
    author="mysorf",
    packages=find_packages(include=["ugtsdti", "ugtsdti.*"]),
    python_requires=">=3.10",
)
