"""
Text Processing Module for AIM2-ODIE.

This module provides functionality for cleaning, normalizing, preprocessing,
and chunking text data for use in ontology development and information 
extraction systems.

The module includes functions for:
- Text normalization (case conversion, whitespace handling, HTML removal)
- Text tokenization (word and sentence segmentation)
- Duplicate removal (exact and fuzzy matching)
- Stopword filtering (English and custom stopwords)
- Encoding standardization (handling various input encodings)
- Text chunking (fixed-size, sentence-based, and recursive chunking)

Functions:
    normalize_text: Normalize text by converting case, removing HTML, and cleaning whitespace
    tokenize_text: Tokenize text into words or sentences using spaCy/NLTK
    remove_duplicates: Remove exact and fuzzy duplicates from text lists
    filter_stopwords: Filter stopwords from token lists
    standardize_encoding: Standardize text encoding from bytes to UTF-8 strings
    chunk_fixed_size: Split text into fixed-size chunks with optional overlap
    chunk_by_sentences: Split text into sentence-based chunks using NLTK/spaCy
    chunk_recursive_char: Use LangChain's RecursiveCharacterTextSplitter for semantic chunking

Exceptions:
    TextCleaningError: Custom exception for text processing failures
    ChunkingError: Custom exception for text chunking failures
"""

from .cleaner import (
    TextCleaningError,
    filter_stopwords,
    normalize_text,
    remove_duplicates,
    standardize_encoding,
    tokenize_text,
)

from .chunker import (
    ChunkingError,
    chunk_by_sentences,
    chunk_fixed_size,
    chunk_recursive_char,
)

__all__ = [
    "TextCleaningError",
    "ChunkingError",
    "filter_stopwords",
    "normalize_text", 
    "remove_duplicates",
    "standardize_encoding",
    "tokenize_text",
    "chunk_by_sentences",
    "chunk_fixed_size", 
    "chunk_recursive_char",
]