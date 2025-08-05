"""
Text Processing Cleaner Module for AIM2-ODIE.

This module provides comprehensive text cleaning and preprocessing functionality
for normalizing, tokenizing, deduplicating, filtering, and encoding text data
in the AIM2-ODIE ontology development and information extraction system.

The module handles various text processing tasks including:
- Text normalization with HTML tag removal and whitespace handling
- Tokenization using spaCy with NLTK fallback support
- Duplicate removal with exact and fuzzy matching capabilities
- Stopword filtering with custom and biomedical stopword support
- Encoding standardization with automatic detection capabilities

Functions:
    normalize_text: Normalize text by converting case, removing HTML, and cleaning whitespace
    tokenize_text: Tokenize text into words or sentences using spaCy/NLTK
    remove_duplicates: Remove exact and fuzzy duplicates from text lists
    filter_stopwords: Filter stopwords from token lists
    standardize_encoding: Standardize text encoding from bytes to UTF-8 strings

Exceptions:
    TextCleaningError: Custom exception for text processing failures
"""

import html
import re
from typing import List

import chardet
import nltk
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz


class TextCleaningError(Exception):
    """
    Custom exception for text cleaning and processing failures.
    
    This exception is raised when text processing operations fail due to
    invalid inputs, encoding issues, or other processing errors.
    """
    pass


def normalize_text(text: str) -> str:
    """
    Normalize text by converting to lowercase, removing HTML tags, and cleaning whitespace.
    
    This function performs comprehensive text normalization including:
    - Case conversion to lowercase
    - HTML tag and structure removal
    - HTML entity decoding
    - Whitespace normalization (multiple spaces to single space)
    - Leading/trailing whitespace removal
    
    Args:
        text: Input text string to normalize
        
    Returns:
        str: Normalized text string
        
    Raises:
        TextCleaningError: If input is None or not a string
        
    Examples:
        >>> normalize_text("  PLANT <strong>Metabolomics</strong>   Research  ")
        'plant metabolomics research'
        >>> normalize_text("<p>Plant &amp; metabolomics</p>")
        'plant & metabolomics'
    """
    if text is None:
        raise TextCleaningError("Input text cannot be None")
    
    if not isinstance(text, str):
        raise TextCleaningError("Input must be a string")
    
    # Remove HTML tags using BeautifulSoup for robust parsing
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Normalize whitespace: replace multiple whitespace characters with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text


def tokenize_text(text: str, mode: str = "words", use_nltk: bool = False, filter_punct: bool = False) -> List[str]:
    """
    Tokenize text into words or sentences using spaCy with NLTK fallback.
    
    This function provides flexible tokenization with support for both word and
    sentence segmentation. It primarily uses spaCy for accurate tokenization with
    NLTK as a fallback option when spaCy is unavailable.
    
    Args:
        text: Input text string to tokenize
        mode: Tokenization mode - "words" for word tokenization, "sentences" for sentence segmentation
        use_nltk: If True, use NLTK directly instead of trying spaCy first
        filter_punct: If True, filter out punctuation tokens (only applies to word mode)
        
    Returns:
        List[str]: List of tokens (words or sentences)
        
    Raises:
        TextCleaningError: If input is None, not a string, or mode is invalid
        
    Examples:
        >>> tokenize_text("Plant metabolomics research.")
        ['Plant', 'metabolomics', 'research', '.']
        >>> tokenize_text("First sentence. Second sentence.", mode="sentences")
        ['First sentence.', 'Second sentence.']
        >>> tokenize_text("Plant, metabolomics!", filter_punct=True)
        ['Plant', 'metabolomics']
    """
    if text is None:
        raise TextCleaningError("Input text cannot be None")
    
    if not isinstance(text, str):
        raise TextCleaningError("Input must be a string")
    
    if mode not in ["words", "sentences"]:
        raise TextCleaningError("Mode must be 'words' or 'sentences'")
    
    if not text.strip():
        return []
    
    # Try spaCy first unless explicitly using NLTK
    if not use_nltk:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            
            if mode == "sentences":
                return [sent.text for sent in doc.sents]
            else:  # words mode
                tokens = []
                for token in doc:
                    # Skip whitespace tokens
                    if token.is_space:
                        continue
                    # Filter punctuation if requested
                    if filter_punct and token.is_punct:
                        continue
                    tokens.append(token.text)
                return tokens
                
        except (ImportError, OSError):
            # Fall back to NLTK if spaCy is not available
            pass
    
    # Use NLTK fallback
    try:
        # Download required NLTK data if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        if mode == "sentences":
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        else:  # words mode
            from nltk.tokenize import word_tokenize
            tokens = word_tokenize(text)
            
            if filter_punct:
                # Filter punctuation using basic string methods - keep only alphabetic tokens
                tokens = [token for token in tokens if token.isalpha()]
            
            return tokens
            
    except ImportError:
        raise TextCleaningError("Neither spaCy nor NLTK is available for tokenization")


def remove_duplicates(text_list: List[str], fuzzy_threshold: int = 90, case_sensitive: bool = True) -> List[str]:
    """
    Remove exact and fuzzy duplicates from a list of text strings.
    
    This function removes duplicates in two stages:
    1. Exact duplicate removal (preserving order)
    2. Fuzzy duplicate removal using configurable similarity threshold
    
    Args:
        text_list: List of text strings to deduplicate
        fuzzy_threshold: Similarity threshold (0-100) for fuzzy matching
        case_sensitive: If False, perform case-insensitive comparison
        
    Returns:
        List[str]: List with duplicates removed, preserving original order
        
    Raises:
        TextCleaningError: If input is None, not a list, or threshold is invalid
        
    Examples:
        >>> remove_duplicates(["plant", "Plant", "plant"])
        ['plant', 'Plant']
        >>> remove_duplicates(["plant", "Plant", "plant"], case_sensitive=False)
        ['plant']
        >>> remove_duplicates(["plant metabolomics", "plant metabolomic"], fuzzy_threshold=90)
        ['plant metabolomics']
    """
    if text_list is None:
        raise TextCleaningError("Input text_list cannot be None")
    
    if not isinstance(text_list, list):
        raise TextCleaningError("Input must be a list")
    
    if not 0 <= fuzzy_threshold <= 100:
        raise TextCleaningError("Fuzzy threshold must be between 0 and 100")
    
    if not text_list:
        return []
    
    # Stage 1: Remove exact duplicates while preserving order
    seen = set()
    exact_deduped = []
    
    for text in text_list:
        comparison_text = text.lower() if not case_sensitive else text
        if comparison_text not in seen:
            seen.add(comparison_text)
            exact_deduped.append(text)
    
    # Stage 2: Remove fuzzy duplicates
    fuzzy_deduped = []
    
    for text in exact_deduped:
        is_duplicate = False
        comparison_text = text.lower() if not case_sensitive else text
        
        for existing_text in fuzzy_deduped:
            existing_comparison = existing_text.lower() if not case_sensitive else existing_text
            
            # Calculate fuzzy similarity
            similarity = fuzz.ratio(comparison_text, existing_comparison)
            if similarity >= fuzzy_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            fuzzy_deduped.append(text)
    
    return fuzzy_deduped


def filter_stopwords(tokens: list[str], custom_stopwords_list: list[str] = None) -> list[str]:
    """
    Filter stopwords from a list of tokens using NLTK's English stopwords and custom lists.
    
    This function removes common English stopwords and optionally custom stopwords
    from a token list. Filtering is performed in case-insensitive mode while preserving
    the original case of non-stopword tokens in the output.
    
    Args:
        tokens: List of token strings to filter
        custom_stopwords_list: Optional list of custom stopwords to use instead of default NLTK stopwords
        
    Returns:
        list[str]: List of tokens with stopwords removed, preserving original case
        
    Raises:
        TextCleaningError: If input is None or not a list
        
    Examples:
        >>> filter_stopwords(["the", "plant", "is", "metabolomics"])
        ['plant', 'metabolomics']
        >>> filter_stopwords(["Plant", "study"], custom_stopwords_list=["study"])
        ['Plant']
        >>> filter_stopwords(["The", "Plant", "AND", "research"])
        ['Plant', 'research']
    """
    if tokens is None:
        raise TextCleaningError("Input tokens cannot be None")
    
    if not isinstance(tokens, list):  
        raise TextCleaningError("Input must be a list")
    
    if not tokens:
        return []
    
    # Determine which stopwords to use
    if custom_stopwords_list is None:
        # Use NLTK English stopwords only
        try:
            from nltk.corpus import stopwords
            
            # Download stopwords if not already present
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            all_stopwords = set(stopwords.words('english'))
        except ImportError:
            # Fallback to basic English stopwords if NLTK is not available
            all_stopwords = {
                'the', 'is', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                'by', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had',
                'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might',
                'must', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
                'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its',
                'our', 'their'
            }
    else:
        # Use only custom stopwords (replace default ones)
        all_stopwords = set(custom_stopwords_list)
    
    # Convert stopwords to lowercase for case-insensitive comparison
    lowercase_stopwords = {sw.lower() for sw in all_stopwords}
    
    # Filter tokens (case-insensitive comparison, preserving original case)
    filtered_tokens = []
    for token in tokens:
        if token.lower() not in lowercase_stopwords:
            filtered_tokens.append(token)
    
    return filtered_tokens


def standardize_encoding(text_bytes: bytes, source_encoding: str = 'utf-8', target_encoding: str = 'utf-8', 
                        auto_detect: bool = False, fallback_encoding: str = 'utf-8', errors: str = 'strict') -> str:
    """
    Standardize text encoding by decoding bytes to UTF-8 strings.
    
    This function handles various input encodings and converts them to standardized
    UTF-8 strings. It supports automatic encoding detection and configurable error
    handling strategies.
    
    Args:
        text_bytes: Input bytes to decode
        source_encoding: Source encoding to use for decoding (if not auto-detecting)
        target_encoding: Target encoding for the output string (typically UTF-8)
        auto_detect: If True, automatically detect source encoding using chardet
        fallback_encoding: Encoding to use if auto-detection fails
        errors: Error handling strategy ('strict', 'ignore', 'replace', etc.)
        
    Returns:
        str: Decoded text string in target encoding
        
    Raises:
        TextCleaningError: If input is None, not bytes, or decoding fails
        
    Examples:
        >>> text_bytes = "Plant metabolomics".encode('utf-8')
        >>> standardize_encoding(text_bytes)
        'Plant metabolomics'
        >>> text_bytes = "Café research".encode('latin-1')
        >>> standardize_encoding(text_bytes, source_encoding='latin-1')
        'Café research'
    """
    if text_bytes is None:
        raise TextCleaningError("Input bytes cannot be None")
    
    if not isinstance(text_bytes, bytes):
        raise TextCleaningError("Input must be bytes")
    
    if not text_bytes:
        return ""
    
    # Auto-detect encoding if requested
    if auto_detect:
        try:
            detection_result = chardet.detect(text_bytes)
            detected_encoding = detection_result.get('encoding')
            
            if detected_encoding:
                source_encoding = detected_encoding
            else:
                source_encoding = fallback_encoding
        except Exception:
            source_encoding = fallback_encoding
    
    # Decode bytes to string
    try:
        decoded_text = text_bytes.decode(source_encoding, errors=errors)
        
        # If target encoding is different from UTF-8, encode and decode again
        if target_encoding.lower() != 'utf-8':
            try:
                # Re-encode to target encoding and decode back to string
                encoded_bytes = decoded_text.encode(target_encoding, errors=errors)
                decoded_text = encoded_bytes.decode(target_encoding)
            except (UnicodeEncodeError, UnicodeDecodeError) as e:
                raise TextCleaningError(f"Failed to convert to target encoding {target_encoding}: {e}")
        
        return decoded_text
        
    except (UnicodeDecodeError, LookupError) as e:
        raise TextCleaningError(f"Failed to decode bytes with encoding {source_encoding}: {e}")