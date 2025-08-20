"""
CLI interface for document indexer.
"""

import os
from pathlib import Path

import click

from .indexer import DocumentIndexer


def validate_safe_path(
    path_str: str, base_dir: str = None, allow_absolute: bool = False
) -> Path:
    """Validate that a path is safe and doesn't contain directory traversal.

    Args:
        path_str: Path string to validate
        base_dir: Base directory to restrict access to (default: current working directory)
        allow_absolute: If True, allow absolute paths that exist

    Returns:
        Validated Path object

    Raises:
        click.ClickException: If path is unsafe
    """
    if base_dir is None:
        base_dir = os.getcwd()

    # Resolve to absolute path
    target_path = Path(path_str).resolve()
    base_path = Path(base_dir).resolve()

    # Allow absolute paths if they exist and flag is set
    if allow_absolute and target_path.is_absolute():
        # Still check for dangerous patterns
        if ".." in str(path_str):
            raise click.ClickException(
                f"Path '{path_str}' contains directory traversal"
            )
        return target_path

    # Check if target is within base directory
    try:
        target_path.relative_to(base_path)
    except ValueError:
        raise click.ClickException(
            f"Path '{path_str}' is outside the allowed directory. "
            f"Please use a path within {base_path}"
        )

    return target_path


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """Document indexer CLI for indexing and searching documents."""
    pass


@main.command()
@click.argument("directory", type=click.Path(exists=False))
@click.option(
    "--persist-dir",
    default="./chroma_db",
    help="Directory to persist the vector database",
)
@click.option("--clear", is_flag=True, help="Clear existing index before indexing")
@click.option(
    "--llm-provider",
    type=click.Choice(["ollama", "openai", "none"], case_sensitive=False),
    default="none",
    help="LLM provider for enhanced parsing (default: none)",
)
@click.option(
    "--parsing-mode",
    type=click.Choice(["text_only", "hybrid", "llm_only"], case_sensitive=False),
    default="text_only",
    help="Parsing mode for document extraction (default: text_only)",
)
@click.option(
    "--llm-model",
    default=None,
    help="Override default model for LLM provider (e.g., llava:latest for Ollama, gpt-4-vision-preview for OpenAI)",
)
@click.option(
    "--ollama-url",
    default="http://localhost:11434",
    help="Ollama API URL (default: http://localhost:11434)",
)
@click.option(
    "--extract-images",
    is_flag=True,
    default=True,
    help="Extract images from documents for LLM analysis",
)
def index(
    directory: str,
    persist_dir: str,
    clear: bool,
    llm_provider: str,
    parsing_mode: str,
    llm_model: str,
    ollama_url: str,
    extract_images: bool,
) -> None:
    """Index documents in a directory with optional LLM enhancement."""
    # Validate and sanitize paths
    try:
        # Allow test directories to be created temporarily
        dir_path = Path(directory).resolve()
        persist_path = Path(persist_dir).resolve()

        # Check for dangerous patterns
        if ".." in directory or ".." in persist_dir:
            raise click.ClickException("Directory traversal not allowed")
    except click.ClickException as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()

    if not dir_path.exists():
        click.echo(f"Error: Directory '{directory}' does not exist.", err=True)
        raise click.Abort()

    if not dir_path.is_dir():
        click.echo(f"Error: '{directory}' is not a directory.", err=True)
        raise click.Abort()

    # Prepare parser configuration
    parser_config = {
        "llm_provider": llm_provider,
        "parsing_mode": parsing_mode,
        "extract_images": extract_images,
    }

    if llm_model:
        parser_config["llm_model"] = llm_model

    if llm_provider == "ollama":
        parser_config["ollama_url"] = ollama_url

    indexer = DocumentIndexer(
        persist_directory=str(persist_path), parser_config=parser_config
    )

    if clear:
        if click.confirm("Are you sure you want to clear the existing index?"):
            indexer.clear_index()
            click.echo("Existing index cleared.")
        else:
            click.echo("Operation cancelled.")
            return

    click.echo(f"Indexing documents in: {dir_path}")
    click.echo(f"Using persist directory: {persist_path}")

    if llm_provider != "none":
        click.echo(f"LLM Provider: {llm_provider}")
        click.echo(f"Parsing Mode: {parsing_mode}")
        if llm_model:
            click.echo(f"Model: {llm_model}")

    try:
        count = indexer.index_directory(dir_path)
        click.echo(f"\nSuccessfully indexed {count} documents.")

        stats = indexer.get_stats()
        click.echo(f"Total documents in index: {stats['total_documents']}")

    except Exception as e:
        click.echo(f"Error during indexing: {e}", err=True)
        raise click.Abort()


@main.command()
@click.argument("query")
@click.option("--limit", default=5, help="Number of results to return")
@click.option(
    "--persist-dir",
    default="./chroma_db",
    help="Directory where vector database is persisted",
)
def search(query: str, limit: int, persist_dir: str) -> None:
    """Search indexed documents."""
    indexer = DocumentIndexer(persist_directory=persist_dir)

    try:
        results = indexer.search(query, n_results=limit)

        if not results:
            click.echo("No results found.")
            return

        click.echo(f"Found {len(results)} results for: '{query}'\n")
        click.echo("-" * 80)

        for i, result in enumerate(results, 1):
            click.echo(f"\n{i}. {result.metadata.filename}")
            click.echo(f"   Path: {result.metadata.file_path}")
            click.echo(f"   Score: {result.score:.4f}")
            click.echo(f"   Preview: {result.content[:200]}...")
            click.echo("-" * 80)

    except Exception as e:
        click.echo(f"Error during search: {e}", err=True)
        raise click.Abort()


@main.command()
@click.option(
    "--persist-dir",
    default="./chroma_db",
    help="Directory where vector database is persisted",
)
def stats(persist_dir: str) -> None:
    """Show index statistics."""
    indexer = DocumentIndexer(persist_directory=persist_dir)

    try:
        stats = indexer.get_stats()

        click.echo("Index Statistics")
        click.echo("=" * 40)
        click.echo(f"Total documents: {stats['total_documents']}")
        click.echo(f"Collection: {stats['collection_name']}")
        click.echo(f"Storage: {stats['persist_directory']}")

        if stats["total_documents"] == 0:
            click.echo("\nIndex is empty. Use 'index' command to add documents.")

    except Exception as e:
        click.echo(f"Error getting stats: {e}", err=True)
        raise click.Abort()


# Add these individual command functions for testing
index_command = index
search_command = search
stats_command = stats


if __name__ == "__main__":
    main()
