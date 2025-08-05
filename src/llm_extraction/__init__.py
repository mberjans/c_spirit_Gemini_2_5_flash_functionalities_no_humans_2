"""
LLM-based information extraction module for AIM2-ODIE ontology development.

This module provides functionality for extracting structured information from scientific
text using Large Language Models (LLMs). It includes named entity recognition (NER),
relation extraction, and other information extraction tasks specific to plant metabolomics
and related biological domains.

Modules:
    ner: Named Entity Recognition for extracting domain-specific entities
    relations: Relation extraction between identified entities
    schema: Entity and relation schema definitions for biological domains
"""

from .ner import (
    extract_entities,
    NERError,
    LLMAPIError,
    InvalidSchemaError,
    RateLimitError
)

__all__ = [
    "extract_entities",
    "NERError", 
    "LLMAPIError",
    "InvalidSchemaError",
    "RateLimitError"
]