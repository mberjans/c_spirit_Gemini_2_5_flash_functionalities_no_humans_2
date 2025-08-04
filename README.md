# C-Spirit: AI-Driven Ontology Development and Information Extraction in Plant Metabolomics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-red.svg)](https://github.com/astral-sh/ruff)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-green.svg)](https://github.com/pytest-dev/pytest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The C-Spirit project (AIM2: AI-Driven Ontology Development and Information Extraction in Plant Metabolomics) is a revolutionary system designed to transform metabolomics data enrichment through the construction of a comprehensive, annotated metabolites network. This project focuses on understanding plant functional roles and resilience by leveraging fully automated Python-based solutions that eliminate manual intervention entirely.

## Key Objectives

- **Automated Ontology Development**: Create and manage robust biological ontologies using AI-driven techniques
- **Literature Information Extraction**: Extract structured biological information from scientific literature using Large Language Models (LLMs)
- **Metabolite Network Construction**: Build an annotated network of plant metabolites with functional and structural annotations
- **Database Integration**: Seamlessly integrate with existing biological databases and tools
- **Zero Manual Intervention**: Eliminate human effort through comprehensive automation and AI-powered processes

## Features

### 🧬 Automated Ontology Management
- **Multi-source Integration**: Seamlessly integrate ontologies from ChEBI, NCBI Taxonomy, Plant Ontology (PO), Gene Ontology (GO), and more
- **Intelligent Trimming**: AI-driven semantic filtering to reduce ontology complexity while maintaining relevance
- **Automated Alignment**: Resolve semantic conflicts and redundancies across different ontology sources
- **Dynamic Schema Development**: Create custom ontological relationships using OWL/SWRL rules

### 🤖 LLM-Powered Information Extraction
- **Named Entity Recognition (NER)**: Extract chemicals, metabolites, genes, species, and traits from literature
- **Relationship Extraction (RE)**: Identify complex biological relationships with fine-grained specificity
- **Synthetic Data Generation**: Create training data automatically to improve extraction accuracy
- **Self-Correction Mechanisms**: Automated validation and error correction without human oversight

### 🔬 Biological Database Integration
- **PubMed/PMC Access**: Automated literature corpus building from authoritative sources
- **Species Normalization**: Robust taxonomic identification using NCBI Taxonomy
- **Chemical Structure Handling**: Integration with chemical databases and structural similarity analysis
- **Pathway Mapping**: Connection to metabolic pathway databases like Plant Metabolic Network (PMN)

### 📊 Advanced Analytics
- **Compound Prioritization**: Automated ranking of metabolites for experimental testing
- **Visualization Support**: Generate data for eFP browser and pathway visualization tools
- **Quality Assurance**: Comprehensive validation and benchmarking frameworks
- **Version Control**: Automated ontology versioning and documentation

## Technical Architecture

### Core Components

```
C-Spirit/
├── src/                           # Source code modules
│   ├── ontology/                  # Ontology management and development
│   ├── llm_extraction/            # LLM-based information extraction
│   ├── data_acquisition/          # Literature and database access
│   ├── text_processing/           # NLP and text preprocessing
│   ├── ontology_mapping/          # Entity mapping and alignment
│   ├── data_quality/              # Validation and quality assurance
│   ├── evaluation/                # Benchmarking and metrics
│   └── cli/                       # Command-line interfaces
├── data/                          # Input and output data
├── tests/                         # Unit and integration tests
└── docs/                         # Documentation
```

### Key Technologies

- **Ontology Management**: Owlready2 for OWL 2.0 ontology manipulation
- **LLM Integration**: Support for Llama, Gemma, GPT-4o, and other state-of-the-art models
- **Database Access**: Biopython, libChEBIpy, NCBI-taxonomist for biological data
- **Text Processing**: spaCy, NLTK for NLP and langchain for intelligent chunking
- **Ontology Alignment**: OntoAligner for automated semantic alignment
- **Chemical Informatics**: RDKit for structural analysis and similarity calculation
- **Quality Tools**: pytest for testing, ruff for linting, black for code formatting

## Installation

### Prerequisites

- Python 3.8 or higher
- Poetry for dependency management

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/c-spirit.git
   cd c-spirit
   ```

2. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies**:
   ```bash
   poetry install
   ```

4. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```

5. **Verify installation**:
   ```bash
   pytest tests/
   ```

## Quick Start

### Basic Ontology Development

```python
from src.ontology import OntologyManager
from src.data_acquisition import PubMedClient

# Initialize ontology manager
onto_manager = OntologyManager()

# Load and integrate multiple ontologies
onto_manager.load_chebi_ontology()
onto_manager.load_plant_ontology()
onto_manager.load_gene_ontology()

# Apply AI-driven trimming
trimmed_ontology = onto_manager.trim_with_llm(
    relevance_criteria="plant metabolomics and resilience"
)

# Save refined ontology
onto_manager.save_ontology("refined_plant_metabolomics.owl")
```

### Literature Information Extraction

```python
from src.llm_extraction import LLMExtractor
from src.data_acquisition import CorpusBuilder

# Build literature corpus
corpus = CorpusBuilder()
papers = corpus.search_pubmed("plant metabolites AND stress response", limit=100)

# Extract entities and relationships
extractor = LLMExtractor(model="llama-70b")
entities = extractor.extract_entities(papers, ontology=trimmed_ontology)
relationships = extractor.extract_relationships(papers, entities)

# Map to canonical ontology terms
mapped_data = extractor.map_to_ontology(entities, relationships)
```

## Usage Examples

### 1. Automated Ontology Integration

```python
# Integrate multiple biological ontologies
from src.ontology import MultiOntologyIntegrator

integrator = MultiOntologyIntegrator()
integrator.add_source("chebi", source_type="api")
integrator.add_source("ncbi_taxonomy", source_type="api")  
integrator.add_source("plant_ontology", source_type="ols")

# Perform automated alignment and conflict resolution
unified_ontology = integrator.integrate_with_llm_alignment()
```

### 2. Synthetic Training Data Generation

```python
# Generate training data for relationship extraction
from src.evaluation import SyntheticDataGenerator

generator = SyntheticDataGenerator()
training_data = generator.generate_relation_examples(
    relations=["affects", "made_via", "accumulates_in"],
    num_examples=1000
)
```

### 3. Compound Prioritization

```python
# Prioritize metabolites for experimental testing
from src.data_quality import CompoundPrioritizer

prioritizer = CompoundPrioritizer()
prioritized_compounds = prioritizer.rank_by_novelty_and_relevance(
    compounds=extracted_metabolites,
    criteria=["structural_uniqueness", "pathway_centrality"]
)
```

## Configuration

The system supports flexible configuration through environment variables and configuration files:

```bash
# Set LLM provider and model
export C_SPIRIT_LLM_PROVIDER="openai"
export C_SPIRIT_LLM_MODEL="gpt-4o"

# Configure data sources
export NCBI_API_KEY="your_ncbi_api_key"
export PUBMED_EMAIL="your_email@domain.com"
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/ontology/
pytest tests/llm_extraction/
pytest tests/data_quality/

# Run with coverage
pytest --cov=src --cov-report=html
```

## Project Structure

```
├── README.md                     # This file
├── pyproject.toml                # Poetry configuration and dependencies
├── src/                          # Main source code
│   ├── __init__.py
│   ├── cli/                      # Command-line interfaces
│   ├── ontology/                 # Ontology development and management
│   │   ├── acquisition.py        # Multi-source ontology loading
│   │   ├── alignment.py          # Automated ontology alignment
│   │   ├── schema.py             # Custom schema development
│   │   └── trimming.py           # AI-driven ontology filtering
│   ├── llm_extraction/           # LLM-based information extraction
│   │   ├── ner.py                # Named entity recognition
│   │   ├── relation_extraction.py # Relationship extraction
│   │   └── synthetic_data.py     # Training data generation
│   ├── data_acquisition/         # Data source interfaces
│   │   ├── pubmed.py             # PubMed/PMC access
│   │   ├── web_scraping.py       # Journal web scraping
│   │   └── pdf_processing.py     # PDF text extraction
│   ├── text_processing/          # NLP and preprocessing
│   │   ├── chunking.py           # Intelligent text chunking
│   │   ├── cleaning.py           # Text preprocessing
│   │   └── normalization.py      # Entity normalization
│   ├── ontology_mapping/         # Entity and relationship mapping
│   │   ├── text2term.py          # Text-to-term mapping
│   │   └── post_processing.py    # Data normalization
│   ├── data_quality/             # Quality assurance and validation
│   │   ├── validation.py         # Automated validation
│   │   ├── deduplication.py      # Data deduplication
│   │   └── prioritization.py     # Compound prioritization
│   └── evaluation/               # Benchmarking and metrics
│       ├── benchmarking.py       # Model performance evaluation
│       └── gold_standard.py      # Synthetic gold standard generation
├── data/                         # Data directories
│   ├── ontologies/               # Stored ontology files
│   ├── literature/               # Literature corpus
│   ├── extracted/                # Extracted entities and relationships
│   └── outputs/                  # Final processed outputs
├── tests/                        # Test suite
│   ├── conftest.py               # Pytest configuration
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
└── docs/                         # Documentation
    ├── plan.md                   # Detailed project plan
    ├── tickets.md                # Development tickets
    └── api/                      # API documentation
```

## Contributing

We welcome contributions to the C-Spirit project! Please follow these guidelines:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Install development dependencies: `poetry install --with dev`
4. Make your changes following the coding standards
5. Run tests: `pytest`
6. Run code quality checks:
   ```bash
   ruff check src/
   black --check src/
   ```
7. Commit your changes: `git commit -m "Add your feature"`
8. Push to your fork: `git push origin feature/your-feature-name`
9. Submit a pull request

### Coding Standards

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write comprehensive docstrings for all public functions and classes
- Maintain test coverage above 80%
- Use meaningful variable and function names
- Add appropriate error handling and logging

### Commit Message Format

```
type(scope): brief description

Detailed explanation of changes if needed.

- Specific change 1
- Specific change 2
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Roadmap

### Phase 1: Core Infrastructure (Completed)
- ✅ Project structure and development environment
- ✅ Basic ontology management framework
- ✅ Literature acquisition capabilities

### Phase 2: Ontology Development (In Progress)
- 🔄 Multi-source ontology integration
- 🔄 AI-driven ontology trimming and filtering
- 🔄 Automated ontology alignment

### Phase 3: Information Extraction (Planned)
- 📋 LLM-based NER and relationship extraction
- 📋 Synthetic training data generation
- 📋 Automated validation and quality assurance

### Phase 4: Integration and Optimization (Planned)
- 📋 Database integration and API development
- 📋 Performance optimization and scalability
- 📋 Comprehensive evaluation framework

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use C-Spirit in your research, please cite:

```bibtex
@software{c_spirit_2025,
  title={C-Spirit: AI-Driven Ontology Development and Information Extraction in Plant Metabolomics},
  author={AIM2 Project Team},
  year={2025},
  url={https://github.com/your-org/c-spirit}
}
```

## Acknowledgments

- The Plant Metabolic Network (PMN) for metabolic pathway data
- NCBI for taxonomic and literature databases  
- ChEBI for chemical entity classification
- The Gene Ontology Consortium for GO terms
- All open-source contributors to the libraries used in this project

## Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/your-org/c-spirit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/c-spirit/discussions)
- **Documentation**: [Project Documentation](https://c-spirit.readthedocs.io)

---

**Note**: This project emphasizes complete automation and eliminates manual intervention. All processes are designed to be fully automated using AI and machine learning techniques, making it suitable for large-scale, reproducible research in plant metabolomics.