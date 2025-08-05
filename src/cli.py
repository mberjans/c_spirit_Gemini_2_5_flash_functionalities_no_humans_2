"""
Command-Line Interface for AIM2 Project

This module provides a comprehensive CLI for ontology management and corpus
development operations in the AIM2 project.

Features:
- Load ontologies from various formats
- Trim/filter ontologies based on keywords
- Export ontologies to different formats
- Download papers from PubMed
- Extract content from PDF files
- Scrape content from journal websites
- Comprehensive error handling and user feedback

Dependencies:
- Typer for CLI framework
- Rich for enhanced output formatting
"""

import typer
import sys
import os
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# Import ontology modules
try:
    from src.ontology.loader import load_ontology, OntologyLoadError
    from src.ontology.trimmer import trim_ontology, OntologyTrimmerError
    from src.ontology.exporter import export_ontology, OntologyExportError
except ImportError as e:
    print(f"Error importing ontology modules: {e}")
    sys.exit(1)

# Initialize Typer app and Rich console
app = typer.Typer(
    name="aim2-odie",
    help="AIM2 Ontology Development and Information Extraction CLI",
    add_completion=False
)
console = Console()

# Create ontology subcommand group
ontology_app = typer.Typer(
    name="ontology",
    help="Ontology management commands (load, trim, export)"
)
app.add_typer(ontology_app, name="ontology")

# Create corpus subcommand group
corpus_app = typer.Typer(
    name="corpus",
    help="Corpus management commands (pubmed-download, pdf-extract, journal-scrape)"
)
app.add_typer(corpus_app, name="corpus")


@ontology_app.command("load")
def load_ontology_command(
    file_path: str = typer.Argument(..., help="Path to the ontology file to load"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Validate ontology after loading")
):
    """
    Load an ontology from a file.
    
    Supports various ontology formats including OWL, RDF, and Turtle.
    """
    try:
        if verbose:
            console.print(f"[blue]Loading ontology from: {file_path}[/blue]")
        
        # Check if file exists
        if not os.path.exists(file_path):
            console.print(f"[red]Error: File not found: {file_path}[/red]")
            raise typer.Exit(1)
        
        # Load the ontology
        ontology = load_ontology(file_path)
        
        if verbose:
            console.print(f"[green]Successfully loaded ontology[/green]")
            console.print(f"Base IRI: {getattr(ontology, 'base_iri', 'Unknown')}")
            
            # Display basic statistics
            try:
                num_classes = len(list(ontology.classes()))
                num_individuals = len(list(ontology.individuals()))
                num_properties = len(list(ontology.properties()))
                
                table = Table(title="Ontology Statistics")
                table.add_column("Component", style="cyan")
                table.add_column("Count", style="magenta")
                
                table.add_row("Classes", str(num_classes))
                table.add_row("Individuals", str(num_individuals))
                table.add_row("Properties", str(num_properties))
                
                console.print(table)
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]Warning: Could not gather statistics: {e}[/yellow]")
        
        console.print(f"[green]✓ Ontology loaded successfully from {file_path}[/green]")
        
    except OntologyLoadError as e:
        console.print(f"[red]Error loading ontology: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@ontology_app.command("trim")
def trim_ontology_command(
    file_path: str = typer.Argument(..., help="Path to the ontology file to trim"),
    keyword: List[str] = typer.Option([], "--keyword", "-k", help="Keywords to filter by (can be used multiple times)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path (default: adds '_trimmed' suffix)"),
    min_relevance: float = typer.Option(0.5, "--min-relevance", help="Minimum relevance score for filtering"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """
    Trim/filter an ontology based on keywords and relevance criteria.
    
    Filters ontology classes, properties, and individuals based on specified keywords
    and relevance scores to create a more focused, manageable ontology.
    """
    try:
        if not keyword:
            console.print("[red]Error: At least one keyword must be specified using --keyword[/red]")
            console.print("Example: ontology trim myfile.owl --keyword plant --keyword metabolite")
            raise typer.Exit(1)
        
        if verbose:
            console.print(f"[blue]Trimming ontology from: {file_path}[/blue]")
            console.print(f"Keywords: {', '.join(keyword)}")
            console.print(f"Minimum relevance: {min_relevance}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            console.print(f"[red]Error: File not found: {file_path}[/red]")
            raise typer.Exit(1)
        
        # Load the ontology first
        ontology = load_ontology(file_path)
        
        if verbose:
            console.print("[blue]Ontology loaded, starting trimming process...[/blue]")
        
        # Trim the ontology
        trimmed_ontology = trim_ontology(
            ontology, 
            keywords=keyword,
            min_relevance_score=min_relevance
        )
        
        # Determine output file path
        if output is None:
            input_path = Path(file_path)
            output = str(input_path.parent / f"{input_path.stem}_trimmed{input_path.suffix}")
        
        # Export the trimmed ontology
        export_ontology(trimmed_ontology, output)
        
        if verbose:
            console.print(f"[green]Trimmed ontology saved to: {output}[/green]")
            
            # Show trimming statistics if possible
            try:
                original_classes = len(list(ontology.classes()))
                trimmed_classes = len(list(trimmed_ontology.classes()))
                reduction_percent = ((original_classes - trimmed_classes) / original_classes) * 100
                
                table = Table(title="Trimming Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")
                
                table.add_row("Original Classes", str(original_classes))
                table.add_row("Trimmed Classes", str(trimmed_classes))
                table.add_row("Reduction", f"{reduction_percent:.1f}%")
                
                console.print(table)
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]Warning: Could not calculate statistics: {e}[/yellow]")
        
        console.print(f"[green]✓ Ontology trimmed and saved to {output}[/green]")
        
    except (OntologyLoadError, OntologyTrimmerError) as e:
        console.print(f"[red]Error processing ontology: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@ontology_app.command("export")
def export_ontology_command(
    input_file: str = typer.Argument(..., help="Path to the input ontology file"),
    output_file: str = typer.Argument(..., help="Path to the output file"),
    format: Optional[str] = typer.Option(None, "--format", "-f", help="Output format (owl, rdf, ttl, json-ld)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    compress: bool = typer.Option(False, "--compress", help="Compress the output file")
):
    """
    Export an ontology to a different format or location.
    
    Supports exporting to various formats including OWL, RDF, Turtle, and JSON-LD.
    """
    try:
        if verbose:
            console.print(f"[blue]Exporting ontology from: {input_file}[/blue]")
            console.print(f"Output file: {output_file}")
            if format:
                console.print(f"Format: {format}")
        
        # Check if input file exists
        if not os.path.exists(input_file):
            console.print(f"[red]Error: Input file not found: {input_file}[/red]")
            raise typer.Exit(1)
        
        # Load the ontology
        ontology = load_ontology(input_file)
        
        if verbose:
            console.print("[blue]Ontology loaded, starting export...[/blue]")
        
        # Determine format from file extension if not specified
        if format is None:
            output_path = Path(output_file)
            extension = output_path.suffix.lower()
            format_map = {
                '.owl': 'owl',
                '.rdf': 'rdf',
                '.ttl': 'turtle',
                '.jsonld': 'json-ld',
                '.json': 'json-ld'
            }
            format = format_map.get(extension, 'owl')
            
            if verbose:
                console.print(f"[blue]Detected format from extension: {format}[/blue]")
        
        # Export the ontology
        success = export_ontology(
            ontology, 
            output_file, 
            format=format,
            compress=compress
        )
        
        if success:
            console.print(f"[green]✓ Ontology exported successfully to {output_file}[/green]")
            
            if verbose:
                # Show file size information
                try:
                    file_size = os.path.getsize(output_file)
                    size_mb = file_size / (1024 * 1024)
                    console.print(f"Output file size: {size_mb:.2f} MB")
                except Exception:
                    pass
        else:
            console.print("[red]Export failed[/red]")
            raise typer.Exit(1)
        
    except (OntologyLoadError, OntologyExportError) as e:
        console.print(f"[red]Error exporting ontology: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@corpus_app.command("pubmed-download")
def pubmed_download_command(
    query: str = typer.Argument(..., help="PubMed search query"),
    output: str = typer.Option("./pubmed_data", "--output", "-o", help="Output directory for downloaded papers"),
    max_results: int = typer.Option(100, "--max-results", "-m", help="Maximum number of results to download"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    format: str = typer.Option("xml", "--format", "-f", help="Download format (xml, json, txt)")
):
    """
    Download papers from PubMed based on search query.
    
    Downloads academic papers and metadata from PubMed database using the specified
    search query and saves them to the output directory.
    """
    try:
        # Import PubMed functions
        from src.data_acquisition.pubmed import (
            search_pubmed, fetch_pubmed_xml, search_and_fetch,
            set_entrez_email, configure_api_key, PubMedError
        )
        
        if verbose:
            console.print(f"[blue]Starting PubMed download with query: '{query}'[/blue]")
            console.print(f"Output directory: {output}")
            console.print(f"Maximum results: {max_results}")
            console.print(f"Format: {format}")
        
        # Create output directory if it doesn't exist
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            console.print(f"[blue]Created output directory: {output_path.absolute()}[/blue]")
        
        # Set up Entrez email (required by NCBI)
        # Try to get email from environment variable, otherwise use default
        email = os.environ.get('NCBI_EMAIL', 'user@example.com')
        try:
            set_entrez_email(email)
            if verbose:
                console.print(f"[blue]Configured NCBI email: {email}[/blue]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not set email ({e}), using default[/yellow]")
        
        # Configure API key if provided
        api_key = os.environ.get('NCBI_API_KEY')
        if api_key:
            try:
                configure_api_key(api_key)
                if verbose:
                    console.print("[blue]NCBI API key configured for higher rate limits[/blue]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not configure API key ({e})[/yellow]")
        
        # Validate format (currently only XML is fully supported)
        if format.lower() != "xml":
            console.print(f"[yellow]Warning: Format '{format}' requested, but only XML is currently supported. Using XML.[/yellow]")
        
        # Search and fetch data
        console.print(f"[blue]Searching PubMed for: '{query}'[/blue]")
        
        if verbose:
            console.print("[blue]Step 1: Searching for article IDs...[/blue]")
        
        # Search for PubMed IDs
        id_list = search_pubmed(query, max_results)
        
        if not id_list:
            console.print(f"[yellow]No articles found for query: '{query}'[/yellow]")
            return
        
        console.print(f"[green]Found {len(id_list)} articles[/green]")
        
        if verbose:
            console.print(f"[blue]Step 2: Fetching XML content for {len(id_list)} articles...[/blue]")
        
        # Fetch XML content
        xml_content = fetch_pubmed_xml(id_list)
        
        if not xml_content:
            console.print("[yellow]No content retrieved[/yellow]")
            return
        
        # Save XML content to file
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"pubmed_results_{timestamp}_{len(id_list)}_articles.xml"
        output_file_path = output_path / output_filename
        
        if verbose:
            console.print(f"[blue]Step 3: Saving results to {output_file_path}[/blue]")
        
        # Handle both string and bytes content
        if isinstance(xml_content, bytes):
            with open(output_file_path, 'wb') as f:
                f.write(xml_content)
        else:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
        
        # Create a metadata file with query information
        metadata_filename = f"pubmed_metadata_{timestamp}_{len(id_list)}_articles.txt"
        metadata_file_path = output_path / metadata_filename
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_content = f"""PubMed Download Metadata
========================
Query: {query}
Date: {current_time}
Results: {len(id_list)} articles
IDs: {', '.join(id_list[:10])}{'...' if len(id_list) > 10 else ''}
Output File: {output_filename}
XML Content Size: {len(xml_content)} characters

PubMed IDs (complete list):
{chr(10).join(id_list)}
"""
        
        with open(metadata_file_path, 'w', encoding='utf-8') as f:
            f.write(metadata_content)
        
        # Summary
        console.print(f"[green]✓ PubMed download completed successfully![/green]")
        console.print(f"[green]  - Downloaded {len(id_list)} articles[/green]")
        console.print(f"[green]  - XML content: {len(xml_content):,} characters[/green]")
        console.print(f"[green]  - Results saved to: {output_file_path}[/green]")
        console.print(f"[green]  - Metadata saved to: {metadata_file_path}[/green]")
        
    except PubMedError as e:
        console.print(f"[red]PubMed API error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)
    except ImportError as e:
        console.print(f"[red]Missing required dependencies: {e}[/red]")
        console.print("[yellow]Please install Biopython with: pip install biopython[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@corpus_app.command("pdf-extract")
def pdf_extract_command(
    input_file: str = typer.Argument(..., help="Path to the PDF file to extract"),
    output: str = typer.Option("./extracted_text", "--output", "-o", help="Output directory for extracted content"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    extract_images: bool = typer.Option(False, "--extract-images", help="Also extract images from PDF"),
    extract_tables: bool = typer.Option(False, "--extract-tables", help="Also extract tables from PDF")
):
    """
    Extract text and content from PDF files.
    
    Processes PDF files to extract text, images, and tables for further analysis
    and corpus development.
    """
    try:
        if verbose:
            console.print(f"[blue]Starting PDF extraction from: {input_file}[/blue]")
            console.print(f"Output directory: {output}")
            console.print(f"Extract images: {extract_images}")
            console.print(f"Extract tables: {extract_tables}")
        
        # Check if input file exists
        if not os.path.exists(input_file):
            console.print(f"[red]Error: PDF file not found: {input_file}[/red]")
            raise typer.Exit(1)
        
        # Create output directory if it doesn't exist
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            console.print(f"[blue]Created output directory: {output_path.absolute()}[/blue]")
        
        # Placeholder implementation
        console.print("[yellow]Note: This is a placeholder implementation[/yellow]")
        console.print(f"[green]Would extract content from: {input_file}[/green]")
        console.print(f"[green]Would save extracted content to: {output}[/green]")
        
        if extract_images:
            console.print("[green]Would extract images from PDF[/green]")
        if extract_tables:
            console.print("[green]Would extract tables from PDF[/green]")
        
        # TODO: Call actual PDF extraction function from src.data_acquisition.pdf_extractor
        # from src.data_acquisition.pdf_extractor import extract_pdf_content
        # results = extract_pdf_content(input_file, output_dir=output, extract_images=extract_images, extract_tables=extract_tables)
        
        console.print(f"[green]✓ PDF extraction completed (placeholder)[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during PDF extraction: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@corpus_app.command("journal-scrape")
def journal_scrape_command(
    url: str = typer.Argument(..., help="URL of the journal or article to scrape"),
    output: str = typer.Option("./scraped_content", "--output", "-o", help="Output directory for scraped content"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    max_depth: int = typer.Option(1, "--max-depth", help="Maximum depth for recursive scraping"),
    delay: float = typer.Option(1.0, "--delay", help="Delay between requests in seconds"),
    include_metadata: bool = typer.Option(True, "--include-metadata/--no-metadata", help="Include article metadata")
):
    """
    Scrape content from journal websites and articles.
    
    Extracts article content, metadata, and related information from academic
    journal websites for corpus development.
    """
    try:
        if verbose:
            console.print(f"[blue]Starting journal scraping from: {url}[/blue]")
            console.print(f"Output directory: {output}")
            console.print(f"Maximum depth: {max_depth}")
            console.print(f"Request delay: {delay}s")
            console.print(f"Include metadata: {include_metadata}")
        
        # Basic URL validation
        if not url.startswith(('http://', 'https://')):
            console.print(f"[red]Error: Invalid URL format: {url}[/red]")
            console.print("URL must start with http:// or https://")
            raise typer.Exit(1)
        
        # Create output directory if it doesn't exist
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            console.print(f"[blue]Created output directory: {output_path.absolute()}[/blue]")
        
        # Placeholder implementation
        console.print("[yellow]Note: This is a placeholder implementation[/yellow]")
        console.print(f"[green]Would scrape content from: {url}[/green]")
        console.print(f"[green]Would save scraped content to: {output}[/green]")
        console.print(f"[green]Would use max depth: {max_depth}[/green]")
        console.print(f"[green]Would use delay: {delay}s between requests[/green]")
        
        if include_metadata:
            console.print("[green]Would include article metadata[/green]")
        
        # TODO: Call actual journal scraping function from src.data_acquisition.journal_scraper
        # from src.data_acquisition.journal_scraper import scrape_journal
        # results = scrape_journal(url=url, output_dir=output, max_depth=max_depth, delay=delay, include_metadata=include_metadata)
        
        console.print(f"[green]✓ Journal scraping completed (placeholder)[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during journal scraping: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command("version")
def version():
    """Show version information."""
    console.print("[bold blue]AIM2 Ontology Development and Information Extraction CLI[/bold blue]")
    console.print("Version: 0.1.0")
    console.print("Python package for automated ontology development and information extraction")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode")
):
    """
    AIM2 Ontology Development and Information Extraction CLI
    
    A comprehensive command-line tool for ontology management, corpus development,
    and information extraction tasks in the AIM2 project.
    """
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    elif verbose:
        import logging
        logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    app()