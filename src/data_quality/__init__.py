"""
Data quality module for the AIM2-ODIE ontology development system.

This module provides functionality for data quality assessment, cleaning,
and normalization in the context of plant metabolomics research and
ontology development.

Modules:
    normalizer: Data normalization and fuzzy string matching functionality
"""

# Import main functions and classes for easy access
from .normalizer import normalize_name, find_fuzzy_matches, NormalizationError

__all__ = [
    'normalize_name',
    'find_fuzzy_matches', 
    'NormalizationError'
]