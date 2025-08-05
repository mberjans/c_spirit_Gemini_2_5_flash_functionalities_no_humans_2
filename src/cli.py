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
    • chunk - Split text into manageable segments for processing and analysis
    
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


@process_app.command("chunk")
def process_chunk_command(
    input_file: str = typer.Argument(
        ..., 
        help="Path to the input text file to chunk and segment. File must be readable and contain text content suitable for processing."
    ),
    output: str = typer.Option(
        "./chunked_text", 
        "--output", "-o", 
        help="Output directory where chunk files and metadata will be saved. Creates directory structure if it doesn't exist."
    ),
    method: str = typer.Option(
        "fixed", 
        "--method", "-m", 
        help="Chunking method to use: 'fixed' (fixed-size chunks), 'sentences' (sentence-based), or 'recursive' (semantic chunking)."
    ),
    chunk_size: int = typer.Option(
        1000, 
        "--chunk-size", "-s", 
        help="Maximum size of each chunk in characters (for 'fixed' and 'recursive' methods). Recommended: 500-2000 for most applications."
    ),
    chunk_overlap: int = typer.Option(
        100, 
        "--chunk-overlap", 
        help="Number of characters to overlap between consecutive chunks (for 'fixed' and 'recursive' methods). Helps maintain context."
    ),
    tokenizer: str = typer.Option(
        "nltk", 
        "--tokenizer", 
        help="Tokenizer for sentence-based chunking: 'nltk' (default) or 'spacy'. Only applies to 'sentences' method."
    ),
    separators: Optional[str] = typer.Option(
        None, 
        "--separators", 
        help="Custom separators for recursive chunking (comma-separated). Example: '\\n\\n,\\n,.,!,?'. Only applies to 'recursive' method."
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", "-v", 
        help="Enable detailed progress information including chunk statistics, processing steps, and file operations."
    )
):
    """
    Split text into manageable chunks for processing and analysis.
    
    This command segments large text files into smaller, manageable chunks suitable
    for LLM processing, analysis, and information extraction. Multiple chunking
    strategies are available to handle different text types and use cases.
    
    \b
    CHUNKING METHODS:
    • fixed - Fixed-size character chunks with optional overlap for consistent processing
    • sentences - Sentence-based chunks preserving natural language boundaries  
    • recursive - Semantic chunking using hierarchical separators for context preservation
    
    \b
    METHOD DETAILS:
    
    Fixed-Size Chunking:
    • Creates chunks of exactly specified character size with optional overlap
    • Attempts to avoid splitting words when possible by finding word boundaries
    • Best for: Consistent processing requirements, memory-constrained applications
    • Parameters: --chunk-size, --chunk-overlap
    
    Sentence-Based Chunking:
    • Splits text at sentence boundaries using NLTK or spaCy tokenizers
    • Preserves complete sentences and handles scientific abbreviations
    • Best for: Natural language processing, maintaining linguistic coherence
    • Parameters: --tokenizer (nltk/spacy)
    
    Recursive Character Chunking:
    • Uses hierarchical separators to find optimal split points
    • Maintains semantic coherence by respecting document structure
    • Best for: Complex documents, maintaining context and meaning
    • Parameters: --chunk-size, --chunk-overlap, --separators
    
    \b
    OUTPUT FILES:
    • chunk_001.txt, chunk_002.txt, ... - Individual chunk files numbered sequentially
    • chunking_metadata.json - Complete chunking session metadata and statistics
    • chunk_summary.txt - Human-readable summary of chunking results
    
    \b
    CHUNK OVERLAP BENEFITS:
    • Maintains context across chunk boundaries
    • Helps with entity recognition spanning chunks
    • Reduces information loss at chunk edges
    • Recommended: 10-20% of chunk size
    
    \b
    CHUNKING PARAMETERS:
    • Small chunks (200-500 chars): Better for fine-grained analysis, more files
    • Medium chunks (500-1500 chars): Balanced approach for most applications
    • Large chunks (1500-3000 chars): Better context retention, fewer files
    • Overlap: Typically 10-20% of chunk size for good context preservation
    
    \b
    REQUIREMENTS:
    • Input file must be readable text format
    • NLTK library for sentence tokenization (auto-downloaded if needed)
    • spaCy library for advanced sentence tokenization (optional)
    • LangChain library for recursive chunking (optional, fallback available)
    • Sufficient disk space for output chunks
    
    \b
    USAGE EXAMPLES:
    # Basic fixed-size chunking with default settings
    process chunk research_paper.txt --output ./chunks --verbose
    
    # Sentence-based chunking for natural language processing
    process chunk article.txt --method sentences --tokenizer spacy --output ./sentences
    
    # Recursive chunking with custom parameters
    process chunk document.txt --method recursive --chunk-size 1500 --chunk-overlap 200 --output ./semantic_chunks
    
    # Fixed chunking with custom size and no overlap
    process chunk large_text.txt --method fixed --chunk-size 800 --chunk-overlap 0 --output ./fixed_chunks
    
    # Recursive chunking with custom separators
    process chunk structured_doc.txt --method recursive --separators "\\n\\n,\\n,.,!,?" --output ./custom_chunks
    
    \b
    PERFORMANCE CONSIDERATIONS:
    • Large files may take time to process - use --verbose to monitor progress
    • Many small chunks create more files but allow parallel processing
    • Fewer large chunks reduce I/O overhead but may exceed processing limits
    • Consider downstream processing requirements when choosing chunk size
    
    \b
    TEXT TYPE RECOMMENDATIONS:
    • Scientific papers: sentence or recursive method for preserving structure
    • News articles: sentence method for maintaining readability
    • Technical documentation: recursive method with custom separators
    • General text: fixed method for consistent processing requirements
    • Multi-language content: sentence method with appropriate tokenizer
    
    \b
    TROUBLESHOOTING:
    • If chunking fails, check input file encoding and readability
    • For sentence chunking errors, try switching tokenizer (nltk/spacy)
    • Large overlap values may cause processing slowdown
    • Use --verbose to identify specific chunking issues
    • Ensure sufficient disk space for output chunks
    • Some methods require additional libraries - install as prompted
    """
    try:
        if verbose:
            console.print(f"[blue]Starting text chunking process for: {input_file}[/blue]")
            console.print("Chunking parameters:")
            console.print(f"  - Method: {method}")
            console.print(f"  - Output directory: {output}")
            if method in ["fixed", "recursive"]:
                console.print(f"  - Chunk size: {chunk_size} characters")
                console.print(f"  - Chunk overlap: {chunk_overlap} characters")
            if method == "sentences":
                console.print(f"  - Tokenizer: {tokenizer}")
            if method == "recursive" and separators:
                console.print(f"  - Custom separators: {separators}")
        
        # Validate method
        if method not in ["fixed", "sentences", "recursive"]:
            console.print(f"[red]Error: Invalid chunking method '{method}'. Must be 'fixed', 'sentences', or 'recursive'.[/red]")
            console.print("Use --help to see available methods and their descriptions.")
            raise typer.Exit(1)
        
        # Check if input file exists
        if not os.path.exists(input_file):
            console.print(f"[red]Error: Input file not found: {input_file}[/red]")
            raise typer.Exit(1)
        
        # Validate parameters for specific methods
        if method in ["fixed", "recursive"]:
            if chunk_size <= 0:
                console.print(f"[red]Error: Chunk size must be positive (got {chunk_size})[/red]")
                raise typer.Exit(1)
            
            if chunk_overlap < 0:
                console.print(f"[red]Error: Chunk overlap cannot be negative (got {chunk_overlap})[/red]")
                raise typer.Exit(1)
            
            if chunk_overlap >= chunk_size:
                console.print(f"[red]Error: Chunk overlap ({chunk_overlap}) cannot be larger than chunk size ({chunk_size})[/red]")
                raise typer.Exit(1)
        
        if method == "sentences" and tokenizer not in ["nltk", "spacy"]:
            console.print(f"[red]Error: Invalid tokenizer '{tokenizer}'. Must be 'nltk' or 'spacy'.[/red]")
            raise typer.Exit(1)
        
        # Create output directory
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            console.print(f"[blue]Created output directory: {output_path.absolute()}[/blue]")
        
        # Read input file
        console.print("[blue]Reading input file...[/blue]")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                text_content = f.read()
        except UnicodeDecodeError:
            # Try alternative encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(input_file, 'r', encoding=encoding) as f:
                        text_content = f.read()
                    if verbose:
                        console.print(f"[yellow]Successfully read file using {encoding} encoding[/yellow]")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                console.print("[red]Error: Could not decode input file. Please ensure it's a valid text file.[/red]")
                raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Error reading input file: {e}[/red]")
            raise typer.Exit(1)
        
        if not text_content.strip():
            console.print("[yellow]Warning: Input file is empty or contains only whitespace[/yellow]")
            return
        
        original_length = len(text_content)
        if verbose:
            console.print(f"[green]✓ Read {original_length:,} characters from input file[/green]")
        
        # Perform chunking based on selected method
        console.print(f"[blue]Chunking text using '{method}' method...[/blue]")
        
        try:
            if method == "fixed":
                chunks = chunk_fixed_size(text_content, chunk_size, chunk_overlap)
            elif method == "sentences":
                chunks = chunk_by_sentences(text_content, tokenizer)
            elif method == "recursive":
                # Parse custom separators if provided
                custom_separators = None
                if separators:
                    # Split by comma and replace escape sequences
                    custom_separators = [sep.replace('\\n', '\n').replace('\\t', '\t') 
                                       for sep in separators.split(',')]
                    if verbose:
                        console.print(f"[blue]Using custom separators: {custom_separators}[/blue]")
                
                chunks = chunk_recursive_char(text_content, chunk_size, chunk_overlap, custom_separators)
                
        except ChunkingError as e:
            console.print(f"[red]Chunking error: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Unexpected error during chunking: {e}[/red]")
            if verbose:
                import traceback
                console.print(traceback.format_exc())
            raise typer.Exit(1)
        
        if not chunks:
            console.print("[yellow]No chunks were created (empty result)[/yellow]")
            return
        
        console.print(f"[green]✓ Successfully created {len(chunks)} chunks[/green]")
        
        # Save chunks to individual files
        console.print("[blue]Saving chunks to files...[/blue]")
        
        chunk_files = []
        total_chunk_chars = 0
        
        for i, chunk in enumerate(chunks, 1):
            chunk_filename = f"chunk_{i:03d}.txt"
            chunk_file_path = output_path / chunk_filename
            
            try:
                with open(chunk_file_path, 'w', encoding='utf-8') as f:
                    f.write(chunk)
                
                chunk_files.append(chunk_filename)
                total_chunk_chars += len(chunk)
                
                if verbose and i <= 5:  # Show first 5 files being created
                    console.print(f"[dim]  Created {chunk_filename} ({len(chunk)} characters)[/dim]")
                elif verbose and i == 6 and len(chunks) > 5:
                    console.print(f"[dim]  ... and {len(chunks) - 5} more files[/dim]")
                
            except Exception as e:
                console.print(f"[red]Error writing chunk {i}: {e}[/red]")
                raise typer.Exit(1)
        
        if verbose:
            console.print(f"[green]✓ Saved {len(chunk_files)} chunk files[/green]")
        
        # Create metadata file
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate chunk statistics
        chunk_lengths = [len(chunk) for chunk in chunks]
        avg_chunk_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
        min_chunk_length = min(chunk_lengths) if chunk_lengths else 0
        max_chunk_length = max(chunk_lengths) if chunk_lengths else 0
        
        metadata = {
            "timestamp": timestamp,
            "input_file": str(Path(input_file).absolute()),
            "output_directory": str(output_path.absolute()),
            "chunking_method": method,
            "parameters": {
                "chunk_size": chunk_size if method in ["fixed", "recursive"] else None,
                "chunk_overlap": chunk_overlap if method in ["fixed", "recursive"] else None,
                "tokenizer": tokenizer if method == "sentences" else None,
                "separators": separators if method == "recursive" else None
            },
            "statistics": {
                "original_text_length": original_length,
                "total_chunks": len(chunks),
                "total_chunk_characters": total_chunk_chars,
                "average_chunk_length": round(avg_chunk_length, 2),
                "min_chunk_length": min_chunk_length,
                "max_chunk_length": max_chunk_length,
                "compression_ratio": round(total_chunk_chars / original_length, 4) if original_length > 0 else 0
            },
            "chunk_files": chunk_files
        }
        
        # Save metadata as JSON
        metadata_file = output_path / "chunking_metadata.json"
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            if verbose:
                console.print(f"[green]✓ Metadata saved to: {metadata_file.name}[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save metadata: {e}[/yellow]")
        
        # Create human-readable summary
        summary_content = f"""Text Chunking Summary
====================
Date: {timestamp}
Input: {Path(input_file).name}
Method: {method.title()} Chunking

Parameters:
"""
        
        if method in ["fixed", "recursive"]:
            summary_content += f"- Chunk Size: {chunk_size:,} characters\n"
            summary_content += f"- Chunk Overlap: {chunk_overlap:,} characters\n"
        if method == "sentences":
            summary_content += f"- Tokenizer: {tokenizer}\n"
        if method == "recursive" and separators:
            summary_content += f"- Custom Separators: {separators}\n"
        
        summary_content += f"""
Results:
- Original Text: {original_length:,} characters
- Total Chunks: {len(chunks):,}
- Average Chunk Size: {avg_chunk_length:.0f} characters
- Size Range: {min_chunk_length:,} - {max_chunk_length:,} characters
- Output Files: {len(chunk_files)} chunk files + metadata

Files Created:
"""
        
        for filename in chunk_files[:10]:  # Show first 10 files
            summary_content += f"- {filename}\n"
        
        if len(chunk_files) > 10:
            summary_content += f"- ... and {len(chunk_files) - 10} more files\n"
        
        summary_content += f"- chunking_metadata.json\n"
        
        # Save summary
        summary_file = output_path / "chunk_summary.txt"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            if verbose:
                console.print(f"[green]✓ Summary saved to: {summary_file.name}[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save summary: {e}[/yellow]")
        
        # Display results table
        if verbose:
            table = Table(title="Chunking Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Original text size", f"{original_length:,} characters")
            table.add_row("Total chunks", f"{len(chunks):,}")
            table.add_row("Average chunk size", f"{avg_chunk_length:.0f} characters")
            table.add_row("Size range", f"{min_chunk_length:,} - {max_chunk_length:,} characters")
            table.add_row("Files created", f"{len(chunk_files) + 2}")  # +2 for metadata and summary
            table.add_row("Method", method.title())
            
            if method in ["fixed", "recursive"]:
                overlap_percent = (chunk_overlap / chunk_size * 100) if chunk_size > 0 else 0
                table.add_row("Overlap", f"{chunk_overlap} chars ({overlap_percent:.1f}%)")
            
            console.print(table)
        
        # Final summary
        console.print(f"[green]✓ Text chunking completed successfully![/green]")
        console.print(f"[green]  Input: {Path(input_file).name} ({original_length:,} characters)[/green]")
        console.print(f"[green]  Output: {len(chunks)} chunks in {output_path.name}/[/green]")
        console.print(f"[blue]  Average chunk size: {avg_chunk_length:.0f} characters[/blue]")
        console.print(f"[blue]  Total files created: {len(chunk_files) + 2}[/blue]")
        
    except ChunkingError as e:
        console.print(f"[red]Chunking error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error during text chunking: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@process_app.command("clean")
def process_clean_command(
    input_file: str = typer.Argument(
        ..., 
        help="Path to the input text file to clean and process. File must be readable and contain text content."
    ),
    output: Optional[str] = typer.Option(
        None, 
        "--output", "-o", 
        help="Output file path for cleaned text. If not specified, adds '_cleaned' suffix to input filename."
    ),
    normalize: bool = typer.Option(
        False, 
        "--normalize", 
        help="Apply text normalization: convert to lowercase, remove HTML tags, clean whitespace."
    ),
    tokenize: str = typer.Option(
        None, 
        "--tokenize", 
        help="Tokenize text into 'words' or 'sentences'. Output will be one token per line."
    ),
    remove_dupes: bool = typer.Option(
        False, 
        "--remove-duplicates", 
        help="Remove exact and fuzzy duplicate lines from the text."
    ),
    filter_stops: bool = typer.Option(
        False, 
        "--filter-stopwords", 
        help="Remove common English stopwords from tokenized text."
    ),
    standardize_encoding: bool = typer.Option(
        False, 
        "--standardize-encoding", 
        help="Standardize text encoding to UTF-8 with automatic encoding detection."
    ),
    fuzzy_threshold: int = typer.Option(
        90, 
        "--fuzzy-threshold", 
        help="Similarity threshold (0-100) for fuzzy duplicate detection. Higher values are more strict."
    ),
    custom_stopwords: Optional[str] = typer.Option(
        None, 
        "--custom-stopwords", 
        help="Path to file containing custom stopwords (one per line) to use instead of default English stopwords."
    ),
    filter_punct: bool = typer.Option(
        False, 
        "--filter-punct", 
        help="Filter out punctuation tokens when tokenizing (only applies to word tokenization)."
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", "-v", 
        help="Enable detailed progress information including processing steps, statistics, and file operations."
    )
):
    """
    Clean and preprocess text data using various normalization and filtering techniques.
    
    This command provides comprehensive text cleaning capabilities for preparing raw text
    data for analysis, machine learning, and information extraction tasks. Multiple
    cleaning operations can be combined in a single processing pipeline.
    
    \b
    CLEANING OPERATIONS:
    • Text normalization - Convert to lowercase, remove HTML, clean whitespace
    • Tokenization - Split text into words or sentences with punctuation filtering
    • Duplicate removal - Remove exact and fuzzy duplicates with configurable similarity
    • Stopword filtering - Remove common English words using NLTK or custom lists
    • Encoding standardization - Convert to UTF-8 with automatic detection
    
    \b
    PROCESSING PIPELINE:
    Operations are applied in this order when multiple options are selected:
    1. Encoding standardization (if --standardize-encoding)
    2. Text normalization (if --normalize)
    3. Tokenization (if --tokenize specified)
    4. Stopword filtering (if --filter-stopwords and tokenized)
    5. Duplicate removal (if --remove-duplicates)
    
    \b
    OUTPUT FORMATS:
    • Default: Cleaned text preserving original structure
    • Tokenized: One token per line when using --tokenize
    • Deduplicated: Unique lines only when using --remove-duplicates
    
    \b
    TOKENIZATION MODES:
    • words - Split into individual words and punctuation
    • sentences - Split into complete sentences
    • Use --filter-punct to remove punctuation from word tokens
    
    \b
    DUPLICATE REMOVAL:
    • Exact duplicates: Removed based on string equality
    • Fuzzy duplicates: Removed using configurable similarity threshold
    • Case sensitivity: Configurable for comparison operations
    
    \b
    REQUIREMENTS:
    • Input file must be readable text format
    • NLTK library for tokenization and stopwords (auto-downloaded)
    • spaCy library for advanced tokenization (optional, NLTK fallback)
    • BeautifulSoup for HTML tag removal
    • chardet for encoding detection
    
    \b
    USAGE EXAMPLES:
    # Basic normalization and cleanup
    process clean raw_text.txt --normalize --output clean_text.txt --verbose
    
    # Tokenize into words and remove stopwords
    process clean document.txt --tokenize words --filter-stopwords --filter-punct --output tokens.txt
    
    # Full cleaning pipeline with duplicate removal
    process clean corpus.txt --normalize --tokenize sentences --remove-duplicates --fuzzy-threshold 85 --output processed.txt
    
    # Custom stopwords and encoding standardization
    process clean multilingual.txt --standardize-encoding --filter-stopwords --custom-stopwords my_stopwords.txt
    
    # Sentence segmentation for analysis
    process clean research_paper.txt --normalize --tokenize sentences --output sentences.txt --verbose
    
    \b
    ADVANCED OPTIONS:
    • --fuzzy-threshold: Control similarity for duplicate detection (default: 90)
    • --custom-stopwords: Use domain-specific stopword lists
    • --filter-punct: Clean up tokenized output by removing punctuation
    • Multiple operations can be combined for comprehensive cleaning
    
    \b
    FILE HANDLING:
    • Input: Any readable text file in various encodings
    • Output: UTF-8 encoded text file with cleaned content
    • Automatic output naming with '_cleaned' suffix if not specified
    • Preserves directory structure when using relative paths
    
    \b
    TROUBLESHOOTING:
    • For encoding issues, try --standardize-encoding first
    • Large files may take time - use --verbose to monitor progress
    • If tokenization fails, NLTK fallback will be used automatically
    • Custom stopwords file should contain one word per line
    • Check input file permissions if processing fails
    """
    try:
        if verbose:
            console.print(f"[blue]Starting text cleaning process for: {input_file}[/blue]")
            console.print("Processing options:")
            console.print(f"  - Normalize text: {normalize}")
            console.print(f"  - Tokenize: {tokenize if tokenize else 'No'}")
            console.print(f"  - Remove duplicates: {remove_dupes}")
            console.print(f"  - Filter stopwords: {filter_stops}")
            console.print(f"  - Standardize encoding: {standardize_encoding}")
            if remove_dupes:
                console.print(f"  - Fuzzy threshold: {fuzzy_threshold}")
        
        # Check if input file exists
        if not os.path.exists(input_file):
            console.print(f"[red]Error: Input file not found: {input_file}[/red]")
            raise typer.Exit(1)
        
        # Determine output file path
        if output is None:
            input_path = Path(input_file)
            output = str(input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}")
        
        if verbose:
            console.print(f"[blue]Output file: {output}[/blue]")
        
        # Read input file
        console.print("[blue]Reading input file...[/blue]")
        try:
            # Try reading as UTF-8 first
            with open(input_file, 'r', encoding='utf-8') as f:
                text_content = f.read()
        except UnicodeDecodeError:
            # If UTF-8 fails, read as bytes for encoding standardization
            with open(input_file, 'rb') as f:
                raw_bytes = f.read()
            
            if standardize_encoding:
                if verbose:
                    console.print("[blue]Detecting and standardizing encoding...[/blue]")
                text_content = standardize_encoding(raw_bytes, auto_detect=True)
            else:
                # Try common encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        text_content = raw_bytes.decode(encoding)
                        if verbose:
                            console.print(f"[yellow]Successfully decoded using {encoding} encoding[/yellow]")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    console.print("[red]Error: Could not decode file. Try using --standardize-encoding option.[/red]")
                    raise typer.Exit(1)
        
        original_length = len(text_content)
        if verbose:
            console.print(f"[green]✓ Read {original_length:,} characters from input file[/green]")
        
        # Load custom stopwords if provided
        custom_stopwords_list = None
        if custom_stopwords:
            if not os.path.exists(custom_stopwords):
                console.print(f"[red]Error: Custom stopwords file not found: {custom_stopwords}[/red]")
                raise typer.Exit(1)
            
            try:
                with open(custom_stopwords, 'r', encoding='utf-8') as f:
                    custom_stopwords_list = [line.strip() for line in f if line.strip()]
                if verbose:
                    console.print(f"[green]✓ Loaded {len(custom_stopwords_list)} custom stopwords[/green]")
            except Exception as e:
                console.print(f"[red]Error reading custom stopwords file: {e}[/red]")
                raise typer.Exit(1)
        
        # Apply processing pipeline in order
        processed_content = text_content
        
        # Step 1: Encoding standardization (already done during file reading if requested)
        
        # Step 2: Text normalization
        if normalize:
            console.print("[blue]Normalizing text...[/blue]")
            try:
                processed_content = normalize_text(processed_content)
                if verbose:
                    console.print(f"[green]✓ Text normalized ({len(processed_content):,} characters)[/green]")
            except TextCleaningError as e:
                console.print(f"[red]Error during normalization: {e}[/red]")
                raise typer.Exit(1)
        
        # Step 3: Tokenization
        tokens = None
        if tokenize:
            if tokenize not in ["words", "sentences"]:
                console.print(f"[red]Error: Invalid tokenization mode '{tokenize}'. Must be 'words' or 'sentences'.[/red]")
                raise typer.Exit(1)
            
            console.print(f"[blue]Tokenizing text into {tokenize}...[/blue]")
            try:
                tokens = tokenize_text(processed_content, mode=tokenize, filter_punct=filter_punct)
                if verbose:
                    console.print(f"[green]✓ Tokenized into {len(tokens):,} {tokenize}[/green]")
                    if tokenize == "words" and filter_punct:
                        console.print("[dim]  Punctuation tokens filtered out[/dim]")
            except TextCleaningError as e:
                console.print(f"[red]Error during tokenization: {e}[/red]")
                raise typer.Exit(1)
        
        # Step 4: Stopword filtering (only applies to tokenized content)
        if filter_stops and tokens:
            console.print("[blue]Filtering stopwords...[/blue]")
            try:
                original_token_count = len(tokens)
                tokens = filter_stopwords(tokens, custom_stopwords_list=custom_stopwords_list)
                filtered_count = original_token_count - len(tokens)
                if verbose:
                    console.print(f"[green]✓ Filtered {filtered_count:,} stopwords ({len(tokens):,} tokens remaining)[/green]")
            except TextCleaningError as e:
                console.print(f"[red]Error during stopword filtering: {e}[/red]")
                raise typer.Exit(1)
        elif filter_stops and not tokens:
            console.print("[yellow]Warning: --filter-stopwords requires tokenization. Use --tokenize option.[/yellow]")
        
        # Step 5: Duplicate removal
        if remove_dupes:
            console.print("[blue]Removing duplicates...[/blue]")
            try:
                if tokens:
                    # Remove duplicates from tokens
                    original_count = len(tokens)
                    tokens = remove_duplicates(tokens, fuzzy_threshold=fuzzy_threshold, case_sensitive=not normalize)
                    removed_count = original_count - len(tokens)
                    if verbose:
                        console.print(f"[green]✓ Removed {removed_count:,} duplicates ({len(tokens):,} unique tokens remaining)[/green]")
                else:
                    # Remove duplicates from lines
                    lines = processed_content.split('\n')
                    original_count = len(lines)
                    lines = remove_duplicates(lines, fuzzy_threshold=fuzzy_threshold, case_sensitive=not normalize)
                    processed_content = '\n'.join(lines)
                    removed_count = original_count - len(lines)
                    if verbose:
                        console.print(f"[green]✓ Removed {removed_count:,} duplicate lines ({len(lines):,} unique lines remaining)[/green]")
            except TextCleaningError as e:
                console.print(f"[red]Error during duplicate removal: {e}[/red]")
                raise typer.Exit(1)
        
        # Prepare final output content
        if tokens:
            # If we have tokens, output one per line
            final_content = '\n'.join(tokens)
        else:
            final_content = processed_content
        
        # Write output file
        console.print(f"[blue]Writing cleaned content to: {output}[/blue]")
        try:
            # Ensure output directory exists
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write UTF-8 encoded output
            with open(output, 'w', encoding='utf-8') as f:
                f.write(final_content)
            
            final_length = len(final_content)
            if verbose:
                console.print(f"[green]✓ Wrote {final_length:,} characters to output file[/green]")
                
                # Show processing statistics
                table = Table(title="Text Cleaning Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")
                
                table.add_row("Original size", f"{original_length:,} characters")
                table.add_row("Final size", f"{final_length:,} characters")
                
                if tokens:
                    table.add_row("Tokens", f"{len(tokens):,}")
                    table.add_row("Output format", f"One {tokenize[:-1]} per line")
                
                size_change = ((final_length - original_length) / original_length * 100) if original_length > 0 else 0
                change_color = "green" if size_change < 0 else "yellow" if size_change > 0 else "white"
                table.add_row("Size change", f"[{change_color}]{size_change:+.1f}%[/{change_color}]")
                
                console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error writing output file: {e}[/red]")
            raise typer.Exit(1)
        
        # Summary
        console.print(f"[green]✓ Text cleaning completed successfully![/green]")
        console.print(f"[green]  Input: {input_file}[/green]")
        console.print(f"[green]  Output: {output}[/green]")
        
        # Show what operations were applied
        applied_operations = []
        if standardize_encoding:
            applied_operations.append("encoding standardization")
        if normalize:
            applied_operations.append("text normalization")
        if tokenize:
            applied_operations.append(f"{tokenize} tokenization")
        if filter_stops and tokens:
            applied_operations.append("stopword filtering")
        if remove_dupes:
            applied_operations.append("duplicate removal")
        
        if applied_operations:
            console.print(f"[blue]Applied operations: {', '.join(applied_operations)}[/blue]")
        
    except TextCleaningError as e:
        console.print(f"[red]Text cleaning error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error during text cleaning: {e}[/red]")
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