"""
Test package for AIM2-ODIE project.

This package contains comprehensive tests for the AIM2-ODIE ontology development
and information extraction system, including unit tests and integration tests
for all major components.

Test Organization:
- ontology/: Tests for ontology loading, trimming, editing, and export
- data_acquisition/: Tests for PubMed, PDF extraction, and journal scraping
- text_processing/: Tests for text cleaning and chunking
- llm_extraction/: Tests for NER and relationship extraction
- ontology_mapping/: Tests for entity and relation mapping
- data_quality/: Tests for normalization, deduplication, and taxonomy
- evaluation/: Tests for benchmarking and curation
- cli/: Tests for command-line interface

Usage:
    Run all tests: pytest tests/
    Run specific module: pytest tests/ontology/
    Run with markers: pytest -m unit tests/
"""

__version__ = "1.0.0"