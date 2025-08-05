"""
Text Processing Chunker Module for AIM2-ODIE.

This module provides comprehensive text chunking functionality for preparing 
literature text for LLM processing in the AIM2-ODIE ontology development 
and information extraction system.

The module handles various text chunking strategies including:
- Fixed-size chunking with optional overlap support for both character and word-based splitting
- Sentence-based chunking using NLTK or spaCy tokenizers with scientific text support
- Recursive character chunking using LangChain's RecursiveCharacterTextSplitter
- Comprehensive error handling and dependency management

Functions:
    chunk_fixed_size: Split text into fixed-size chunks with optional overlap
    chunk_by_sentences: Split text into sentence-based chunks using NLTK/spaCy
    chunk_recursive_char: Use LangChain's RecursiveCharacterTextSplitter for semantic chunking

Exceptions:
    ChunkingError: Custom exception for text chunking failures
"""

import re
from typing import List, Optional

import nltk


class ChunkingError(Exception):
    """
    Custom exception for text chunking failures.
    
    This exception is raised when text chunking operations fail due to
    invalid inputs, missing dependencies, or other processing errors.
    """
    pass


def chunk_fixed_size(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split text into fixed-size chunks with optional overlap.
    
    This function creates chunks of specified size with optional overlap between chunks.
    It uses character-based chunking and avoids splitting words inappropriately when possible.
    
    Args:
        text: Input text string to chunk
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
        
    Raises:
        ChunkingError: If input is invalid or parameters are incorrect
        
    Examples:
        >>> chunk_fixed_size("Plant metabolomics research", chunk_size=10, chunk_overlap=0)
        ['Plant meta', 'bolomics r', 'esearch']
    """
    if text is None:
        raise ChunkingError("Input text cannot be None")
    
    if not isinstance(text, str):
        raise ChunkingError("Input must be a string")
    
    if chunk_size <= 0:
        raise ChunkingError("Chunk size must be positive")
    
    if chunk_overlap < 0:
        raise ChunkingError("Chunk overlap cannot be negative")
    
    if chunk_overlap >= chunk_size:
        raise ChunkingError("Chunk overlap cannot be larger than chunk size")
    
    if not text.strip():
        return []
    
    # Always use character-based chunking as per task specification
    return _chunk_by_characters(text, chunk_size, chunk_overlap)


def _chunk_by_characters(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into character-based chunks avoiding word splitting."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            # Last chunk - take remaining text
            remaining = text[start:]
            if remaining.strip():
                chunks.append(remaining)
            break
        
        # Get initial chunk
        chunk = text[start:end]
        
        # Try to avoid splitting words - look for word boundary
        if end < len(text) and not text[end].isspace():
            # Look backwards for a space within the chunk
            space_pos = chunk.rfind(' ')
            if space_pos > 0:  # Found a space, use it as split point
                end = start + space_pos
                chunk = text[start:end]
            # If no space found, we'll have to split the word
        
        if chunk.strip():
            chunks.append(chunk)
        
        # Calculate next start position with overlap
        if chunk_overlap > 0:
            start = end - chunk_overlap
        else:
            start = end
    
    return chunks


def _chunk_by_words(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into word-based chunks."""
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(' '.join(chunk_words))
        
        # Calculate next start position with overlap
        step = chunk_size - chunk_overlap
        if step <= 0:
            step = 1  # Ensure we make progress
        start += step
        
        if start >= len(words):
            break
    
    return chunks


def chunk_by_sentences(text: str, tokenizer: str = 'nltk') -> List[str]:
    """
    Split text into sentence-based chunks using NLTK or spaCy.
    
    This function segments text into sentences using either NLTK or spaCy tokenizers.
    It handles complex punctuation and scientific abbreviations appropriately.
    
    Args:
        text: Input text string to chunk into sentences
        tokenizer: Tokenizer to use - 'nltk' or 'spacy'
        
    Returns:
        List[str]: List of sentences
        
    Raises:
        ChunkingError: If input is invalid or tokenizer is unsupported
        
    Examples:
        >>> chunk_by_sentences("Plant research is important. It studies metabolites.")
        ['Plant research is important.', 'It studies metabolites.']
        >>> chunk_by_sentences("Dr. Smith's research shows results.", tokenizer='spacy')
        ["Dr. Smith's research shows results."]
    """
    if text is None:
        raise ChunkingError("Input text cannot be None")
    
    if not isinstance(text, str):
        raise ChunkingError("Input must be a string")
    
    if tokenizer not in ['nltk', 'spacy']:
        raise ChunkingError(f"Unsupported tokenizer: {tokenizer}")
    
    if not text.strip():
        return []
    
    # Try spaCy first if requested
    if tokenizer == 'spacy':
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
            return [sent for sent in sentences if sent]
        except (ImportError, OSError):
            # Fall back to NLTK if spaCy is not available
            pass
    
    # Use NLTK tokenizer
    try:
        from nltk.tokenize import sent_tokenize
        
        # Download required NLTK data if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        sentences = sent_tokenize(text)
        
        # Clean up whitespace
        cleaned_sentences = []
        for sent in sentences:
            cleaned = sent.strip()
            if cleaned:
                cleaned_sentences.append(cleaned)
        
        return cleaned_sentences
        
    except ImportError:
        raise ChunkingError("Neither spaCy nor NLTK is available for sentence tokenization")


def chunk_recursive_char(text: str, chunk_size: int, chunk_overlap: int, 
                        separators: Optional[List[str]] = None) -> List[str]:
    """
    Use LangChain's RecursiveCharacterTextSplitter for semantic chunking.
    
    This function uses LangChain's recursive text splitter to maintain semantic
    coherence while chunking text. It tries different separators in order to
    find the best split points.
    
    Args:
        text: Input text string to chunk
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        separators: List of separators to try in order (default: ["\n\n", "\n", " ", ""])
        
    Returns:
        List[str]: List of text chunks
        
    Raises:
        ChunkingError: If input is invalid, parameters are incorrect, or LangChain is unavailable
        
    Examples:
        >>> chunk_recursive_char("Section 1\\n\\nContent here\\n\\nSection 2", chunk_size=20, chunk_overlap=0)
        ['Section 1', 'Content here', 'Section 2']
    """
    if text is None:
        raise ChunkingError("Input text cannot be None")
    
    if not isinstance(text, str):
        raise ChunkingError("Input must be a string")
    
    if chunk_size <= 0:
        raise ChunkingError("Chunk size must be positive")
    
    if chunk_overlap < 0:
        raise ChunkingError("Chunk overlap cannot be negative")
    
    if chunk_overlap >= chunk_size:
        raise ChunkingError("Chunk overlap cannot be larger than chunk size")
    
    if separators is not None and not isinstance(separators, list):
        raise ChunkingError("Separators must be a list")
    
    if not text.strip():
        return []
    
    # Set default separators if none provided
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
        
        chunks = text_splitter.split_text(text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]
        
    except ImportError:
        raise ChunkingError("LangChain library is required for recursive chunking")