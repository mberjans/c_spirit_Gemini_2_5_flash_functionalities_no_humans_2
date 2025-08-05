"""
Command-Line Interface for Ontology Management

This module provides a comprehensive CLI for ontology operations including
loading, trimming/filtering, and exporting ontologies.

Features:
- Load ontologies from various formats
- Trim/filter ontologies based on keywords
- Export ontologies to different formats
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
    
    A comprehensive command-line tool for ontology management and information extraction
    tasks in the AIM2 project.
    """
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    elif verbose:
        import logging
        logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    app()