"""
Data quality normalization module for the AIM2-ODIE ontology development system.

This module provides functionality for cleaning and standardizing entity names,
and performing fuzzy string matching to identify similar entities in ontologies.
It's designed for use in plant metabolomics research where entity names from
literature extraction need to be normalized and matched against existing ontologies.

Key Features:
- Name normalization: case conversion, whitespace handling, specific word processing
- Fuzzy matching: FuzzyWuzzy integration with configurable thresholds
- Unicode support: handles accented characters and special scientific notation
- Error handling: comprehensive input validation with descriptive error messages

Functions:
    normalize_name(name: str) -> str: Normalizes entity names for consistency
    find_fuzzy_matches(query: str, candidates: List[str], threshold: int = 80) -> List[Tuple[str, int]]: 
        Finds fuzzy string matches using FuzzyWuzzy

Classes:
    NormalizationError: Custom exception for input validation errors
"""

import re
from typing import List, Tuple, Union
from fuzzywuzzy import process


class NormalizationError(Exception):
    """
    Custom exception raised when input validation fails in normalization functions.
    
    This exception is used to provide clear, descriptive error messages for
    invalid inputs to the normalization and fuzzy matching functions.
    """
    pass


def normalize_name(name: Union[str, None]) -> str:
    """
    Normalize entity names for case, spacing, and specific word handling.
    
    This function standardizes entity names by applying consistent formatting rules:
    - Converts to title case with special handling for articles and prepositions
    - Removes extra whitespace and normalizes whitespace characters
    - Handles scientific names, chemical compounds, and special characters
    - Preserves hyphens, apostrophes, and parentheses appropriately
    
    Args:
        name (str): The entity name to normalize
        
    Returns:
        str: The normalized entity name
        
    Raises:
        NormalizationError: If input is None, not a string, or otherwise invalid
        
    Examples:
        >>> normalize_name("KING ARTHUR")
        'King Arthur'
        >>> normalize_name("arabidopsis thaliana")
        'Arabidopsis Thaliana'
        >>> normalize_name("alpha-D-glucose")
        'Alpha-D-Glucose'
        >>> normalize_name("THE LORD OF THE RINGS")
        'The Lord of the Rings'
    """
    # Input validation
    if name is None:
        raise NormalizationError("Input name cannot be None")
    
    if not isinstance(name, str):
        raise NormalizationError("Input must be a string")
    
    # Handle empty string or whitespace-only string
    if not name or not name.strip():
        return ""
    
    # Normalize whitespace: replace all whitespace characters with single spaces
    # and strip leading/trailing whitespace
    normalized = re.sub(r'\s+', ' ', name.strip())
    
    # Split into words for processing
    words = normalized.split()
    
    # Define articles, prepositions, and conjunctions that should be lowercase
    # (except when they are the first word)
    lowercase_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'of', 'in', 'on', 'at', 'to', 
        'for', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'under', 'over'
    }
    
    result_words = []
    
    for i, word in enumerate(words):
        # Convert to lowercase for comparison, but preserve original for processing
        word_lower = word.lower()
        
        # Handle special cases for first word or words not in lowercase_words set
        if i == 0 or word_lower not in lowercase_words:
            # Apply title case, but handle special characters carefully
            if '-' in word:
                # Handle hyphenated words (e.g., "alpha-D-glucose" -> "Alpha-D-Glucose")
                hyphen_parts = word.split('-')
                title_parts = []
                for part in hyphen_parts:
                    if part:  # Skip empty parts
                        title_parts.append(part[0].upper() + part[1:].lower() if len(part) > 1 else part.upper())
                    else:
                        title_parts.append('')
                result_word = '-'.join(title_parts)
            elif "'" in word:
                # Handle apostrophes - different handling for possessives vs names
                if word.lower().endswith("'s"):
                    # Handle possessives (e.g., "mcdonald's" -> "Mcdonald's")
                    base_word = word[:-2]  # Remove 's
                    result_word = (base_word[0].upper() + base_word[1:].lower() if len(base_word) > 1 else base_word.upper()) + "'s"
                else:
                    # Handle names with apostrophes (e.g., "o'malley" -> "O'Malley")
                    apostrophe_parts = word.split("'")
                    title_parts = []
                    for j, part in enumerate(apostrophe_parts):
                        if part:  # Skip empty parts
                            # All parts get title case for names like O'Malley
                            title_parts.append(part[0].upper() + part[1:].lower() if len(part) > 1 else part.upper())
                        else:
                            title_parts.append('')
                    result_word = "'".join(title_parts)
            elif '(' in word and ')' in word:
                # Handle parentheses (e.g., "calcium (ca2+)" -> "Calcium (Ca2+)")
                # Find the content inside parentheses and apply title case to it
                def title_case_parentheses(match):
                    content = match.group(1)  # Content inside parentheses
                    return f"({content[0].upper() + content[1:].lower() if len(content) > 1 else content.upper()})"
                
                # Apply title case to the word first, then fix parentheses content
                basic_title = word[0].upper() + word[1:].lower() if len(word) > 1 else word.upper()
                result_word = re.sub(r'\(([^)]+)\)', title_case_parentheses, basic_title)
            else:
                # Standard title case
                result_word = word[0].upper() + word[1:].lower() if len(word) > 1 else word.upper()
        else:
            # Keep lowercase for articles, prepositions, conjunctions (not first word)
            result_word = word_lower
        
        result_words.append(result_word)
    
    return ' '.join(result_words)


def find_fuzzy_matches(query: Union[str, None], candidates: Union[List[str], None], 
                      threshold: Union[int, None] = 80) -> List[Tuple[str, int]]:
    """
    Find fuzzy string matches using FuzzyWuzzy with configurable threshold.
    
    This function uses the FuzzyWuzzy library to find similar strings in a list
    of candidates based on various string similarity algorithms. Results are
    filtered by a configurable similarity threshold.
    
    Args:
        query (str): The query string to match against candidates
        candidates (List[str]): List of candidate strings to search through
        threshold (int, optional): Minimum similarity score (0-100). Defaults to 80.
        
    Returns:
        List[Tuple[str, int]]: List of tuples containing (match_string, score)
                              for matches above the threshold, sorted by score descending
        
    Raises:
        NormalizationError: If inputs are invalid (None values, wrong types, 
                           invalid threshold range, non-string candidates)
        
    Examples:
        >>> find_fuzzy_matches("glucose", ["glucose", "fructose", "sucrose"])
        [('glucose', 100)]
        >>> find_fuzzy_matches("arabidopsis", ["Arabidopsis thaliana", "Brassica napus"], 70)
        [('Arabidopsis thaliana', 85)]
    """
    # Input validation for query
    if query is None:
        raise NormalizationError("Query string cannot be None")
    
    if not isinstance(query, str):
        raise NormalizationError("Query must be a string")
    
    # Input validation for candidates
    if candidates is None:
        raise NormalizationError("Candidates list cannot be None")
    
    if not isinstance(candidates, list):
        raise NormalizationError("Candidates must be a list")
    
    # Validate all candidates are strings
    for i, candidate in enumerate(candidates):
        if not isinstance(candidate, str):
            raise NormalizationError("All candidates must be strings")
    
    # Input validation for threshold
    if threshold is None:
        raise NormalizationError("Threshold must be an integer")
    
    if not isinstance(threshold, int):
        raise NormalizationError("Threshold must be an integer")
    
    if threshold < 0 or threshold > 100:
        raise NormalizationError("Threshold must be between 0 and 100")
    
    # Handle empty candidates list
    if not candidates:
        return []
    
    # Use FuzzyWuzzy to find matches
    # process.extract returns a list of tuples: (match, score)
    # We set limit to len(candidates) to get all results, then filter by threshold
    fuzzy_results = process.extract(query, candidates, limit=len(candidates))
    
    # Filter results by threshold and return as list of tuples
    filtered_results = [
        (match, score) for match, score in fuzzy_results 
        if score >= threshold
    ]
    
    return filtered_results