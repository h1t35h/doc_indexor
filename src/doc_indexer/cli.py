"""
CLI interface for document indexer.
"""

from pathlib import Path

import click

from .indexer import DocumentIndexer


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
def index(directory: str, persist_dir: str, clear: bool) -> None:
    """Index documents in a directory."""
    dir_path = Path(directory)

    if not dir_path.exists():
        click.echo(f"Error: Directory '{directory}' does not exist.", err=True)
        raise click.Abort()

    if not dir_path.is_dir():
        click.echo(f"Error: '{directory}' is not a directory.", err=True)
        raise click.Abort()

    indexer = DocumentIndexer(persist_directory=persist_dir)

    if clear:
        if click.confirm("Are you sure you want to clear the existing index?"):
            indexer.clear_index()
            click.echo("Existing index cleared.")
        else:
            click.echo("Operation cancelled.")
            return

    click.echo(f"Indexing documents in: {dir_path}")
    click.echo(f"Using persist directory: {persist_dir}")

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
