# Document Indexer

A powerful, secure document indexing and search system with advanced LLM-based content extraction capabilities. Index and search through PDF, Word, and PowerPoint documents using semantic search powered by ChromaDB and optional AI enhancement.

## Features

- **Multi-format Support**: Index PDF, Word (.docx), and PowerPoint (.pptx) documents
- **Semantic Search**: Find relevant content using vector similarity search with ChromaDB
- **LLM Enhancement**: Optional AI-powered content extraction for better understanding of:
  - Images and diagrams
  - Complex tables
  - Charts and graphs
  - Handwritten text
- **Multiple LLM Providers**: Support for OpenAI GPT-4 Vision and Ollama (local models)
- **Batch Processing**: Efficient batch indexing with memory management
- **Security First**: Built with security best practices including input sanitization and secure API handling
- **CLI Interface**: Easy-to-use command-line interface for all operations

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from PyPI (when published)

```bash
pip install doc-indexer
```

### Install from Source

```bash
git clone https://github.com/yourusername/doc_indexer.git
cd doc_indexer
pip install -e .
```

### Install with Development Dependencies

```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

1. **Index documents in a directory:**

```bash
doc-indexer index /path/to/documents
```

2. **Search indexed documents:**

```bash
doc-indexer search "your search query"
```

3. **View index statistics:**

```bash
doc-indexer stats
```

### Advanced Usage with LLM Enhancement

#### Using OpenAI GPT-4 Vision

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Index with OpenAI enhancement
doc-indexer index /path/to/documents \
  --llm-provider openai \
  --parsing-mode hybrid \
  --extract-images
```

#### Using Ollama (Local Models)

```bash
# Start Ollama service first
ollama serve

# Pull required models
ollama pull llava  # For image analysis
ollama pull llama2  # For text enhancement

# Index with Ollama enhancement (using default models)
doc-indexer index /path/to/documents \
  --llm-provider ollama \
  --parsing-mode hybrid \
  --ollama-url http://localhost:11434

# Use specific models for different tasks
doc-indexer index /path/to/documents \
  --llm-provider ollama \
  --parsing-mode hybrid \
  --ollama-image-model llava:13b \
  --ollama-text-model mistral:latest

# Use advanced vision model for better accuracy
doc-indexer index /path/to/documents \
  --llm-provider ollama \
  --parsing-mode llm_only \
  --ollama-image-model bakllava:latest \
  --ollama-text-model mixtral:8x7b
```

## CLI Commands

### `index` - Index Documents

Index documents from a directory into the vector database.

```bash
doc-indexer index [OPTIONS] DIRECTORY
```

**Options:**
- `--persist-dir PATH`: Directory to store the vector database (default: ./chroma_db)
- `--clear`: Clear existing index before indexing
- `--llm-provider [ollama|openai|none]`: LLM provider for enhanced extraction (default: none)
- `--parsing-mode [text_only|hybrid|llm_only]`: Document parsing strategy (default: text_only)
- `--extract-images`: Extract and analyze images from documents
- `--llm-model TEXT`: Override default model for LLM provider
- `--ollama-url TEXT`: Ollama API URL (default: http://localhost:11434)
- `--ollama-image-model TEXT`: Ollama model for image/vision tasks (e.g., llava:13b, bakllava)
- `--ollama-text-model TEXT`: Ollama model for text processing (e.g., llama2:70b, mistral, mixtral)

### `search` - Search Documents

Search through indexed documents using semantic search.

```bash
doc-indexer search [OPTIONS] QUERY
```

**Options:**
- `--limit INTEGER`: Number of results to return (default: 5)
- `--persist-dir PATH`: Directory containing the vector database (default: ./chroma_db)

### `stats` - View Statistics

Display statistics about the indexed documents.

```bash
doc-indexer stats [OPTIONS]
```

**Options:**
- `--persist-dir PATH`: Directory containing the vector database (default: ./chroma_db)

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required for OpenAI provider)
- `OLLAMA_HOST`: Ollama server URL (optional, defaults to http://localhost:11434)

### Parsing Modes

1. **text_only**: Extract text content only (fastest, no LLM required)
2. **hybrid**: Combine text extraction with LLM enhancement for images and complex content
3. **llm_only**: Use LLM for all content extraction (most accurate but slower)

## API Usage

You can also use the document indexer programmatically:

```python
from pathlib import Path
from doc_indexer import DocumentIndexer

# Initialize indexer
indexer = DocumentIndexer(
    persist_directory="./my_index",
    parser_config={
        "llm_provider": "openai",
        "parsing_mode": "hybrid",
        "extract_images": True
    }
)

# Index documents
count = indexer.index_directory(Path("/path/to/documents"))
print(f"Indexed {count} documents")

# Search documents
results = indexer.search("machine learning", n_results=10)
for result in results:
    print(f"Score: {result.score:.2f}")
    print(f"File: {result.metadata.filename}")
    print(f"Content: {result.content[:200]}...")
    print("---")

# Get statistics
stats = indexer.get_stats()
print(f"Total documents: {stats['total_documents']}")
```

## Security Features

The document indexer includes several security features:

- **Input Sanitization**: All user inputs and file paths are sanitized to prevent injection attacks
- **Secure API Key Handling**: API keys are never stored in memory and support environment variables
- **Path Traversal Protection**: File paths are validated to prevent directory traversal attacks
- **Prompt Injection Prevention**: LLM prompts are sanitized to prevent prompt injection
- **Secure HTTP Clients**: All HTTP connections use proper SSL verification and timeouts
- **Resource Management**: Automatic garbage collection and memory management to prevent leaks

## Project Structure

```
doc_indexer/
├── src/
│   └── doc_indexer/
│       ├── cli.py                 # CLI interface
│       ├── indexer.py             # Main indexer logic
│       ├── parser_factory.py      # Document parser factory
│       ├── vector_store.py        # ChromaDB integration
│       ├── models.py              # Data models
│       ├── parsers/               # Document parsers
│       │   ├── pdf_parser.py
│       │   ├── word_parser.py
│       │   ├── powerpoint_parser.py
│       │   ├── llm_providers/     # LLM provider implementations
│       │   └── strategies/        # Parsing strategies
│       └── utils/                 # Utility functions
│           └── security.py        # Security utilities
├── tests/                         # Test suite
├── pyproject.toml                 # Project configuration
└── README.md                      # This file
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/doc_indexer --cov-report=term-missing

# Run specific test file
pytest tests/test_indexer.py -v
```

### Code Quality

```bash
# Format code with Black
black src/ tests/

# Run linting with Ruff
ruff check src/ tests/

# Type checking with mypy
mypy src/doc_indexer
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Requirements

### Core Dependencies

- `chromadb>=0.4.0`: Vector database for semantic search
- `pypdf2`: PDF document parsing
- `python-docx`: Word document parsing
- `python-pptx`: PowerPoint document parsing
- `pillow`: Image processing
- `click`: CLI framework
- `tqdm`: Progress bars
- `pydantic`: Data validation

### Optional Dependencies

- `langchain-openai`: OpenAI GPT integration
- `aiohttp`: Async HTTP client for Ollama
- `psutil`: System resource monitoring

## Troubleshooting

### Common Issues

1. **ChromaDB Connection Error**
   - Ensure you have write permissions to the persist directory
   - Try clearing the index with `--clear` flag

2. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check the Ollama URL is correct
   - Verify required models are installed: `ollama list`

3. **OpenAI API Error**
   - Verify your API key is set correctly
   - Check your OpenAI account has sufficient credits
   - Ensure you're using a vision-capable model for image extraction

4. **Memory Issues with Large Documents**
   - The indexer automatically manages memory with batch processing
   - For very large document sets, consider indexing in smaller batches
   - Adjust batch size if needed (default is optimized for most systems)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ChromaDB for the excellent vector database
- OpenAI for GPT-4 Vision API
- Ollama for local LLM support
- The Python community for the amazing document parsing libraries

## Support

For issues, questions, or contributions, please visit:
- GitHub Issues: [https://github.com/yourusername/doc_indexer/issues](https://github.com/yourusername/doc_indexer/issues)
- Documentation: [https://doc-indexer.readthedocs.io](https://doc-indexer.readthedocs.io)

---

Made with ❤️ for better document understanding