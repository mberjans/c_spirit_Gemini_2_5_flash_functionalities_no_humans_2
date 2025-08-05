"""
Command-Line Interface for AIM2 Project

This module provides a comprehensive CLI for ontology management, corpus
development, text processing, and information extraction operations in the AIM2 project.

Features:
- Load ontologies from various formats
- Trim/filter ontologies based on keywords
- Export ontologies to different formats
- Download papers from PubMed
- Extract content from PDF files
- Scrape content from journal websites
- Clean and preprocess text data
- Chunk text into manageable segments
- Extract entities using named entity recognition
- Extract relationships between entities
- Comprehensive error handling and user feedback

Dependencies:
- Typer for CLI framework
- Rich for enhanced output formatting
- Text processing libraries for cleaning and chunking
- LLM libraries for information extraction
"""

import typer
import sys
import os
import json
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

# Import PDF extraction modules
try:
    from src.data_acquisition.pdf_extractor import (
        extract_text_from_pdf, 
        extract_tables_from_pdf, 
        get_pdf_metadata, 
        PDFExtractionError
    )
except ImportError as e:
    print(f"Error importing PDF extraction modules: {e}")
    sys.exit(1)

# Import text processing modules
try:
    from src.text_processing.cleaner import (
        normalize_text, tokenize_text, remove_duplicates, 
        filter_stopwords, standardize_encoding, TextCleaningError
    )
    from src.text_processing.chunker import (
        chunk_fixed_size, chunk_by_sentences, chunk_recursive_char, ChunkingError
    )
except ImportError as e:
    print(f"Error importing text processing modules: {e}")
    sys.exit(1)

# Import LLM extraction modules
try:
    from src.llm_extraction.ner import (
        extract_entities, extract_entities_few_shot, NERError
    )
    from src.llm_extraction.relations import (
        extract_relationships, extract_domain_specific_relationships, RelationsError
    )
except ImportError as e:
    print(f"Error importing LLM extraction modules: {e}")
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
    help="""Academic corpus development and content acquisition tools.

    Commands for downloading, extracting, and processing academic content from
    various sources including PubMed database, PDF documents, and journal websites.
    
    Available commands:
    • pubmed-download - Download papers and metadata from PubMed database
    • pdf-extract - Extract text, tables, and metadata from PDF files  
    • journal-scrape - Scrape content from academic journal websites
    
    Use 'corpus [command] --help' for detailed information about each command."""
)
app.add_typer(corpus_app, name="corpus")

# Create text processing subcommand group
process_app = typer.Typer(
    name="process",
    help="""Text processing and preprocessing tools for corpus preparation.

    Commands for cleaning, normalizing, and chunking text data to prepare
    it for analysis, machine learning, and information extraction tasks.
    
    Available commands:
    • clean - Clean and normalize raw text data removing noise and artifacts
    • chunk - Split text into manageable segments for processing
    
    Use 'process [command] --help' for detailed information about each command."""
)
app.add_typer(process_app, name="process")

# Create LLM extraction subcommand group  
extract_app = typer.Typer(
    name="extract",
    help="""LLM-powered information extraction and analysis tools.

    Commands for extracting structured information from text using large language
    models including named entity recognition and relationship extraction.
    
    Available commands:
    • ner - Named Entity Recognition to identify entities in text
    • relations - Extract relationships and connections between entities
    
    Use 'extract [command] --help' for detailed information about each command."""
)
app.add_typer(extract_app, name="extract")


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
    query: str = typer.Argument(
        ..., 
        help="PubMed search query using standard PubMed syntax. Examples: 'covid vaccine', 'diabetes[MeSH Terms]', 'smith[Author] AND cancer', 'journal nature[Journal]'"
    ),
    output: str = typer.Option(
        "./pubmed_data", 
        "--output", "-o", 
        help="Output directory path where downloaded papers and metadata will be saved. Creates directory if it doesn't exist."
    ),
    max_results: int = typer.Option(
        100, 
        "--max-results", "-m", 
        help="Maximum number of articles to download (1-10000). Higher numbers may take longer and use more storage."
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", "-v", 
        help="Enable detailed progress information including search steps, API responses, and file operations."
    ),
    format: str = typer.Option(
        "xml", 
        "--format", "-f", 
        help="Output format for downloaded data (currently only 'xml' is fully supported). XML contains complete article metadata and abstracts."
    )
):
    """
    Download academic papers and metadata from PubMed database.
    
    This command searches the PubMed database using your query and downloads article
    metadata, abstracts, and bibliographic information. The results are saved as
    XML files along with metadata summaries for further processing.
    
    \b
    SEARCH QUERY EXAMPLES:
    • Basic keyword search: 'machine learning'
    • MeSH terms: 'diabetes[MeSH Terms]'
    • Author search: 'smith[Author]'
    • Journal search: 'nature[Journal]'
    • Date range: 'cancer AND 2020:2023[PDAT]'
    • Complex query: '(covid OR coronavirus) AND vaccine AND clinical trial[Publication Type]'
    
    \b
    OUTPUT FILES:
    • pubmed_results_[timestamp]_[count]_articles.xml - Main XML data with articles
    • pubmed_metadata_[timestamp]_[count]_articles.txt - Summary metadata file
    
    \b
    REQUIREMENTS:
    • Internet connection for PubMed API access
    • Biopython library (installed automatically)
    • Optional: NCBI_EMAIL environment variable for better API access
    • Optional: NCBI_API_KEY environment variable for higher rate limits
    
    \b
    RATE LIMITS:
    • Without API key: 3 requests/second
    • With API key: 10 requests/second
    • Large queries may take several minutes
    
    \b
    USAGE EXAMPLES:
    # Download 50 COVID-19 vaccine papers
    corpus pubmed-download "covid vaccine" --max-results 50 --output ./covid_papers
    
    # Search with MeSH terms and save to specific directory
    corpus pubmed-download "diabetes[MeSH Terms]" --output ~/research/diabetes --verbose
    
    # Complex search with author and date filters
    corpus pubmed-download "smith[Author] AND cancer AND 2020:2023[PDAT]" --max-results 200
    
    \b
    TROUBLESHOOTING:
    • If download fails, check internet connection and query syntax
    • Large queries may timeout - try reducing max-results
    • Set NCBI_EMAIL environment variable to avoid warnings
    • Use --verbose flag to see detailed progress and debug issues
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
    input_file: str = typer.Argument(
        ..., 
        help="Path to the PDF file to process. Supports both scientific papers and general documents. File must be readable and not password-protected."
    ),
    output: str = typer.Option(
        "./extracted_text", 
        "--output", "-o", 
        help="Output directory where extracted content will be saved. Creates directory structure if it doesn't exist."
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", "-v", 
        help="Enable detailed output showing extraction progress, file sizes, metadata fields, and table statistics."
    ),
    extract_images: bool = typer.Option(
        False, 
        "--extract-images", 
        help="Extract embedded images from PDF (feature planned for future release). Currently shows notification only."
    ),
    extract_tables: bool = typer.Option(
        False, 
        "--extract-tables", 
        help="Extract tables from PDF and save as structured JSON data with row/column information and cell contents."
    )
):
    """
    Extract text, metadata, and structured content from PDF files.
    
    This command processes PDF files to extract readable text content, document
    metadata, and optionally tables for corpus development and text analysis.
    Uses multiple extraction methods with automatic fallback for maximum reliability.
    
    \b
    EXTRACTION CAPABILITIES:
    • Text content - Full document text with layout preservation
    • Document metadata - Title, author, creation date, page count, etc.  
    • Table extraction - Structured tables as JSON with row/column data
    • Multiple PDF formats - Academic papers, reports, books, articles
    • Fallback methods - PyMuPDF primary, with automatic fallback options
    
    \b
    OUTPUT FILES:
    • [filename]_text.txt - Extracted plain text content
    • [filename]_metadata.json - PDF metadata (title, author, dates, etc.)
    • [filename]_tables.json - Structured table data (if --extract-tables used)
    
    \b
    SUPPORTED PDF TYPES:
    • Research papers and journal articles
    • Technical reports and documentation  
    • Books and e-books with text content
    • Multi-column layouts (newspapers, magazines)
    • Mixed content with text and tables
    
    \b
    REQUIREMENTS:
    • PyMuPDF (fitz) library for PDF processing
    • Readable PDF files (not scanned images or password-protected)
    • Sufficient disk space for output files
    • For table extraction: pandas and tabula-py libraries
    
    \b
    USAGE EXAMPLES:
    # Basic text extraction from research paper
    corpus pdf-extract research_paper.pdf --output ./text_output --verbose
    
    # Extract text and tables from technical report
    corpus pdf-extract report.pdf --extract-tables --output ./structured_data
    
    # Process multiple files with detailed output
    corpus pdf-extract document.pdf --extract-tables --verbose --output ~/extracts
    
    # Extract from PDF with custom output location
    corpus pdf-extract "/path/to/document.pdf" --output "./results/pdf_content"
    
    \b
    TEXT EXTRACTION FEATURES:
    • Preserves paragraph structure and line breaks
    • Handles multiple languages and character encodings
    • Processes multi-column layouts intelligently
    • Extracts footnotes and headers when possible
    • Automatic text cleaning and formatting
    
    \b
    TABLE EXTRACTION DETAILS:
    • Detects table boundaries automatically
    • Preserves cell relationships and structure  
    • Outputs JSON with table metadata (rows, columns, position)
    • Handles merged cells and complex table layouts
    • Provides statistics on extracted tables
    
    \b
    TROUBLESHOOTING:
    • If extraction fails, PDF may be corrupted or password-protected
    • Poor quality scanned PDFs may have limited text extraction
    • Large files may take longer to process - use --verbose to monitor progress
    • For complex tables, manual review of JSON output may be needed
    • Some PDF protection methods may prevent content extraction
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
        
        # Extract text content
        console.print("[blue]Extracting text content from PDF...[/blue]")
        try:
            extracted_text = extract_text_from_pdf(input_file, method="pymupdf", use_fallback=True)
            
            # Create base filename from input file
            input_path = Path(input_file)
            base_filename = input_path.stem
            
            # Save extracted text
            text_file = output_path / f"{base_filename}_text.txt"
            text_file.write_text(extracted_text, encoding='utf-8')
            
            if verbose:
                console.print(f"[green]✓ Text extracted ({len(extracted_text)} characters) and saved to: {text_file}[/green]")
            else:
                console.print(f"[green]✓ Text extracted and saved to: {text_file.name}[/green]")
            
        except PDFExtractionError as e:
            console.print(f"[red]Failed to extract text: {e}[/red]")
            raise typer.Exit(1)
        
        # Extract metadata
        console.print("[blue]Extracting PDF metadata...[/blue]")
        try:
            metadata = get_pdf_metadata(input_file)
            
            # Save metadata as JSON
            metadata_file = output_path / f"{base_filename}_metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
            
            if verbose:
                console.print(f"[green]✓ Metadata extracted ({len(metadata)} fields) and saved to: {metadata_file}[/green]")
                # Display key metadata fields
                if metadata:
                    console.print("[dim]Key metadata:[/dim]")
                    for key, value in list(metadata.items())[:5]:  # Show first 5 fields
                        console.print(f"[dim]  {key}: {value}[/dim]")
            else:
                console.print(f"[green]✓ Metadata extracted and saved to: {metadata_file.name}[/green]")
                
        except PDFExtractionError as e:
            console.print(f"[yellow]Warning: Failed to extract metadata: {e}[/yellow]")
        
        # Extract tables if requested
        if extract_tables:
            console.print("[blue]Extracting tables from PDF...[/blue]")
            try:
                tables = extract_tables_from_pdf(input_file)
                
                if tables:
                    # Save tables as JSON
                    tables_file = output_path / f"{base_filename}_tables.json"
                    
                    # Convert tables to serializable format
                    serializable_tables = []
                    for i, table in enumerate(tables):
                        table_data = {
                            "table_id": i + 1,
                            "rows": len(table) if table else 0,
                            "columns": len(table[0]) if table and table[0] else 0,
                            "data": table
                        }
                        serializable_tables.append(table_data)
                    
                    tables_file.write_text(json.dumps(serializable_tables, indent=2), encoding='utf-8')
                    
                    if verbose:
                        console.print(f"[green]✓ {len(tables)} tables extracted and saved to: {tables_file}[/green]")
                        # Show table statistics
                        for i, table_info in enumerate(serializable_tables):
                            console.print(f"[dim]  Table {i+1}: {table_info['rows']} rows × {table_info['columns']} columns[/dim]")
                    else:
                        console.print(f"[green]✓ {len(tables)} tables extracted and saved to: {tables_file.name}[/green]")
                else:
                    console.print("[yellow]No tables found in PDF[/yellow]")
                    
            except PDFExtractionError as e:
                console.print(f"[yellow]Warning: Failed to extract tables: {e}[/yellow]")
        
        # Handle image extraction request
        if extract_images:
            console.print("[yellow]Note: Image extraction is not yet implemented[/yellow]")
            console.print("[dim]Future enhancement: Will extract embedded images from PDF[/dim]")
        
        # Summary
        console.print(f"[green]✓ PDF extraction completed successfully![/green]")
        console.print(f"[blue]Output directory: {output_path.absolute()}[/blue]")
        
    except PDFExtractionError as e:
        console.print(f"[red]PDF extraction error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error during PDF extraction: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@corpus_app.command("journal-scrape")
def journal_scrape_command(
    url: str = typer.Argument(
        ..., 
        help="URL of the journal article or publisher page to scrape. Must be a valid HTTP/HTTPS URL. Examples: 'https://www.nature.com/articles/article-id', 'https://doi.org/10.1000/journal'"
    ),
    output: str = typer.Option(
        "./scraped_content", 
        "--output", "-o", 
        help="Output directory where scraped content, metadata, and summary files will be saved. Creates directory if it doesn't exist."
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", "-v", 
        help="Enable detailed logging of scraping progress, HTTP requests, file operations, and metadata extraction steps."
    ),
    max_depth: int = typer.Option(
        1, 
        "--max-depth", 
        help="Maximum depth for recursive link following (1-5). Higher values scrape linked articles but increase time and data usage."
    ),
    delay: float = typer.Option(
        1.0, 
        "--delay", 
        help="Delay between HTTP requests in seconds (0.5-10.0). Longer delays are more respectful to servers but slower. Recommended: 1-2 seconds."
    ),
    include_metadata: bool = typer.Option(
        True, 
        "--include-metadata/--no-metadata", 
        help="Whether to extract and save article metadata (title, authors, DOI, publication date, etc.) in addition to full text."
    ),
    journal_name: Optional[str] = typer.Option(
        None, 
        "--journal", "-j", 
        help="Specific journal name for targeted metadata scraping. Examples: 'Nature', 'Science', 'PLOS ONE'. Used with --query for journal-specific searches."
    ),
    query: Optional[str] = typer.Option(
        None, 
        "--query", "-q", 
        help="Search query for finding articles within the specified journal. Used together with --journal for targeted content discovery."
    ),
    max_results: int = typer.Option(
        10, 
        "--max-results", "-m", 
        help="Maximum number of search results to process when using --journal and --query options (1-100)."
    )
):
    """
    Scrape academic content from journal websites and publisher platforms.
    
    This command extracts full-text articles, metadata, and bibliographic information
    from academic journal websites. It supports both direct article URL scraping and
    journal-specific search-based content discovery with respectful rate limiting.
    
    \b
    SCRAPING CAPABILITIES:
    • Full-text article content in PDF/HTML formats
    • Article metadata (title, authors, DOI, dates, keywords)
    • Bibliographic information and citation data
    • Journal-specific search and discovery
    • Respectful crawling with configurable delays
    • Robots.txt compliance checking
    
    \b
    SUPPORTED PUBLISHERS:
    • Nature Publishing Group (nature.com)
    • Science/AAAS (science.org) 
    • PLOS journals (plos.org)
    • Springer journals (springer.com)
    • Elsevier ScienceDirect (sciencedirect.com)
    • Many others through general scraping methods
    
    \b
    OUTPUT FILES:
    • [article_filename].pdf/html - Downloaded full-text content
    • metadata_[journal]_[timestamp].json - Article metadata and search results
    • scraping_summary_[timestamp].json - Complete session summary with parameters
    
    \b
    USAGE MODES:
    
    1. Direct Article Scraping:
       Provide a specific article URL to download that article's content
       
    2. Journal Search Mode:
       Use --journal and --query to search within a specific journal
       and download multiple matching articles
    
    \b
    REQUIREMENTS:
    • Internet connection for web access
    • paperscraper library for academic content extraction
    • requests library for HTTP operations
    • Compliance with website terms of service and robots.txt
    
    \b
    RATE LIMITING & ETHICS:
    • Default 1-second delay between requests (adjustable)
    • Automatic robots.txt checking and compliance
    • User-agent identification for transparency
    • Respectful crawling practices to avoid server overload
    
    \b
    USAGE EXAMPLES:
    # Download specific article by URL
    corpus journal-scrape "https://www.nature.com/articles/nature12373" --output ./nature_articles --verbose
    
    # Search Nature journal for machine learning articles
    corpus journal-scrape "https://nature.com" --journal "Nature" --query "machine learning" --max-results 20 --output ./ml_papers
    
    # Scrape with custom delay and no metadata
    corpus journal-scrape "https://doi.org/10.1126/science.123456" --delay 2.0 --no-metadata --output ./science_papers
    
    # Comprehensive scraping with full options
    corpus journal-scrape "https://journals.plos.org/plosone" --journal "PLOS ONE" --query "covid vaccine" --max-results 50 --delay 1.5 --verbose --output ./covid_research
    
    \b
    METADATA EXTRACTION:
    • Article title and subtitle
    • Author names and affiliations  
    • Publication date and DOI
    • Abstract and keywords
    • Journal name and volume/issue
    • Citation information
    
    \b
    TROUBLESHOOTING:
    • If scraping fails, check URL validity and internet connection
    • Some publishers block automated access - try different delay settings
    • Large max-results values may take very long - start with smaller numbers
    • Use --verbose to see detailed progress and identify issues
    • Respect rate limits - if blocked, increase --delay parameter
    • Check robots.txt compliance for specific publishers
    
    \b
    LEGAL & ETHICAL NOTES:
    • Always respect website terms of service
    • Use reasonable delays to avoid overloading servers  
    • Check copyright restrictions for downloaded content
    • Some content may require institutional access
    • Consider contacting publishers for bulk access needs
    """
    try:
        # Import journal scraper functions
        from src.data_acquisition.journal_scraper import (
            download_journal_fulltext, scrape_journal_metadata, 
            configure_rate_limiter, JournalScraperError
        )
        
        if verbose:
            console.print(f"[blue]Starting journal scraping from: {url}[/blue]")
            console.print(f"Output directory: {output}")
            console.print(f"Maximum depth: {max_depth}")
            console.print(f"Request delay: {delay}s")
            console.print(f"Include metadata: {include_metadata}")
            if journal_name:
                console.print(f"Journal name: {journal_name}")
            if query:
                console.print(f"Search query: {query}")
                console.print(f"Max results: {max_results}")
        
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
        
        # Configure rate limiter based on delay parameter
        requests_per_second = 1.0 / delay if delay > 0 else 1.0
        configure_rate_limiter(requests_per_second=requests_per_second)
        
        if verbose:
            console.print(f"[blue]Configured rate limiter: {requests_per_second:.2f} requests/second[/blue]")
        
        results = {}
        
        # If journal name and query are provided, scrape metadata first
        if journal_name and query:
            if verbose:
                console.print(f"[blue]Step 1: Scraping metadata for journal '{journal_name}' with query '{query}'...[/blue]")
            
            try:
                metadata_results = scrape_journal_metadata(
                    journal_name=journal_name,
                    query=query,
                    max_results=max_results,
                    return_detailed=True
                )
                
                if metadata_results and isinstance(metadata_results, dict):
                    articles = metadata_results.get('articles', [])
                    results['metadata'] = metadata_results
                    
                    console.print(f"[green]✓ Found {len(articles)} articles[/green]")
                    
                    # Save metadata results
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    metadata_filename = f"metadata_{journal_name.replace(' ', '_')}_{timestamp}.json"
                    metadata_file_path = output_path / metadata_filename
                    
                    with open(metadata_file_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata_results, f, indent=2, default=str)
                    
                    console.print(f"[green]✓ Metadata saved to: {metadata_filename}[/green]")
                    
                    if verbose and articles:
                        console.print("[dim]Sample articles found:[/dim]")
                        for i, article in enumerate(articles[:3]):  # Show first 3
                            title = article.get('title', 'No title')[:60]
                            console.print(f"[dim]  {i+1}. {title}...[/dim]")
                
                else:
                    console.print("[yellow]No metadata results found[/yellow]")
                    
            except JournalScraperError as e:
                console.print(f"[yellow]Warning: Metadata scraping failed: {e}[/yellow]")
                if verbose:
                    import traceback
                    console.print(traceback.format_exc())
        
        # Download full-text content from the provided URL
        if verbose:
            console.print(f"[blue]Step 2: Downloading full-text content from: {url}[/blue]")
        
        try:
            # Generate filename from URL
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            filename = parsed_url.path.split('/')[-1] if parsed_url.path else 'article'
            if not filename or filename == '/':
                filename = 'article'
            
            # Ensure proper file extension
            if not filename.endswith(('.pdf', '.xml', '.html')):
                filename += '.pdf'  # Default to PDF
            
            article_file_path = output_path / filename
            
            success = download_journal_fulltext(
                article_url=url,
                output_path=str(article_file_path),
                check_robots=True,
                use_paperscraper=True
            )
            
            if success:
                console.print(f"[green]✓ Full-text content downloaded to: {filename}[/green]")
                results['fulltext_file'] = filename
                
                # Get file size information
                if article_file_path.exists():
                    file_size = article_file_path.stat().st_size
                    size_mb = file_size / (1024 * 1024)
                    console.print(f"[blue]Downloaded file size: {size_mb:.2f} MB[/blue]")
                    results['file_size_mb'] = round(size_mb, 2)
            else:
                console.print("[yellow]Full-text download failed or no content available[/yellow]")
                results['fulltext_error'] = "Download failed"
                
        except JournalScraperError as e:
            console.print(f"[yellow]Warning: Full-text download failed: {e}[/yellow]")
            results['fulltext_error'] = str(e)
            if verbose:
                import traceback
                console.print(traceback.format_exc())
        
        # Save summary results
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"scraping_summary_{timestamp}.json"
        summary_file_path = output_path / summary_filename
        
        summary_data = {
            "timestamp": timestamp,
            "url": url,
            "output_directory": str(output_path.absolute()),
            "parameters": {
                "max_depth": max_depth,
                "delay": delay,
                "include_metadata": include_metadata,
                "journal_name": journal_name,
                "query": query,
                "max_results": max_results
            },
            "results": results
        }
        
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        console.print(f"[green]✓ Scraping summary saved to: {summary_filename}[/green]")
        
        # Final summary
        console.print(f"[green]✓ Journal scraping completed successfully![/green]")
        console.print(f"[blue]Output directory: {output_path.absolute()}[/blue]")
        
        total_files = len([f for f in output_path.iterdir() if f.is_file()])
        console.print(f"[blue]Total files created: {total_files}[/blue]")
        
    except ImportError as e:
        console.print(f"[red]Missing required dependencies for journal scraping: {e}[/red]")
        console.print("[yellow]Please install required packages: pip install paperscraper requests[/yellow]")
        raise typer.Exit(1)
    except JournalScraperError as e:
        console.print(f"[red]Journal scraping error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error during journal scraping: {e}[/red]")
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
    text processing, and information extraction tasks in the AIM2 project.
    
    Available command groups:
    • ontology - Load, trim, and export ontology files
    • corpus - Download papers, extract PDF content, scrape journals
    • process - Clean and chunk text data for analysis
    • extract - Extract entities and relationships using LLMs
    """
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    elif verbose:
        import logging
        logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    app()