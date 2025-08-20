"""Setup script for doc-indexer."""

from setuptools import setup, find_packages

setup(
    name="doc-indexer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "doc-indexer=doc_indexer.cli:main",
        ],
    },
)