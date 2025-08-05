"""
Ontology module for the C-Spirit project.

This module provides functionality for loading, manipulating, and working
with OWL 2.0 ontologies using Owlready2. It includes utilities for loading
ontologies from both local files and remote URLs with comprehensive error
handling, as well as trimming and filtering capabilities.
"""

from .loader import (
    OntologyLoadError,
    load_ontology_from_file,
    load_ontology_from_url,
)

from .trimmer import (
    OntologyTrimmerError,
    filter_classes_by_keyword,
    filter_individuals_by_property,
    get_subclasses,
    apply_filters,
)

__all__ = [
    "OntologyLoadError",
    "load_ontology_from_file", 
    "load_ontology_from_url",
    "OntologyTrimmerError",
    "filter_classes_by_keyword",
    "filter_individuals_by_property",
    "get_subclasses",
    "apply_filters",
]