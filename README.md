# Document Indexer

A CLI tool for indexing and searching documents using ChromaDB vector search.

## Features

- Index PDF, Word (docx), and PowerPoint (pptx) documents
- Vector-based semantic search using ChromaDB
- Simple CLI interface
- Batch indexing of entire directories

## Installation

```bash
poetry install
```

## Usage

### Index documents
```bash
doc-indexer index /path/to/documents
```

### Search documents
```bash
doc-indexer search "your search query"
```

### View statistics
```bash
doc-indexer stats
```