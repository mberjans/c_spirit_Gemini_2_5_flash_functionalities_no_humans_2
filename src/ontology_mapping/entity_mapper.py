"""
Entity-to-ontology mapping functionality using text2term.

This module provides functionality for mapping extracted entities to ontology terms using
the text2term library. It supports various mapping methods, minimum similarity scores,
and different term types for flexible ontology integration.

Key Features:
- Multiple mapping methods (TFIDF, Levenshtein, Jaro-Winkler, Jaccard, Fuzzy)
- Minimum score filtering for high-confidence mappings
- Support for different term types (class, property, individual)
- Comprehensive input validation and error handling
- Integration with standard ontologies (ChEBI, GO, NCBI Taxonomy, etc.)

Usage:
    from src.ontology_mapping.entity_mapper import map_entities_to_ontology
    
    entities = ["glucose", "arabidopsis", "photosynthesis"]
    ontology_iri = "http://purl.obolibrary.org/obo/chebi.owl"
    
    results = map_entities_to_ontology(
        entities=entities,
        ontology_iri=ontology_iri,
        mapping_method='tfidf',
        min_score=0.8
    )
"""

import pandas as pd
import re
from typing import List, Optional
from urllib.parse import urlparse

try:
    import text2term
except ImportError:
    # For testing purposes, we'll define a mock text2term module structure
    # The actual import error will be raised at runtime if text2term is needed
    class MockText2Term:
        class Mapper:
            TFIDF = "TFIDF"
            LEVENSHTEIN = "LEVENSHTEIN" 
            JARO_WINKLER = "JARO_WINKLER"
            JACCARD = "JACCARD"
            FUZZY = "FUZZY"
        
        @staticmethod
        def map_terms(**kwargs):
            raise ImportError(
                "text2term is required for entity mapping functionality. "
                "Install it with: pip install text2term"
            )
    
    text2term = MockText2Term()


# Custom Exception Classes
class EntityMapperError(Exception):
    """Base exception for entity mapper errors."""
    pass


class OntologyNotFoundError(EntityMapperError):
    """Exception raised when specified ontology cannot be found or accessed."""
    pass


class MappingError(EntityMapperError):
    """Exception raised when the mapping process fails."""
    pass


# Validation Functions
def _validate_entities(entities: List[str]) -> None:
    """
    Validate entities list input.
    
    Args:
        entities: List of entity strings to validate
        
    Raises:
        ValueError: If entities list is invalid
    """
    if entities is None:
        raise ValueError("Entities list cannot be None")
    
    if not isinstance(entities, list):
        raise ValueError("Entities must be a list")
    
    if len(entities) == 0:
        raise ValueError("Entities list cannot be empty")
    
    for i, entity in enumerate(entities):
        if not isinstance(entity, str):
            raise ValueError(f"Entity at index {i} must be a string, got {type(entity)}")
        
        if entity.strip() == "":
            raise ValueError(f"Entity at index {i} cannot be empty or whitespace only")


def _validate_mapping_method(method: str) -> None:
    """
    Validate mapping method parameter.
    
    Args:
        method: Mapping method string to validate
        
    Raises:
        ValueError: If mapping method is invalid
    """
    valid_methods = {'tfidf', 'levenshtein', 'jaro_winkler', 'jaccard', 'fuzzy'}
    
    if not isinstance(method, str):
        raise ValueError("Invalid mapping method: must be a string")
    
    if method not in valid_methods:
        raise ValueError(
            f"Invalid mapping method: '{method}'. "
            f"Valid methods are: {', '.join(sorted(valid_methods))}"
        )


def _validate_ontology_iri(ontology_iri: str) -> None:
    """
    Validate ontology IRI format.
    
    Args:
        ontology_iri: Ontology IRI string to validate
        
    Raises:
        ValueError: If ontology IRI is invalid
    """
    if not isinstance(ontology_iri, str):
        raise ValueError("Invalid ontology IRI: must be a string")
    
    if not ontology_iri.strip():
        raise ValueError("Invalid ontology IRI: cannot be empty")
    
    # Basic URL validation
    try:
        parsed = urlparse(ontology_iri)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid ontology IRI: must be a valid URL")
        
        # Check for supported protocols
        if parsed.scheme.lower() not in ['http', 'https', 'file']:
            raise ValueError(
                f"Invalid ontology IRI: unsupported protocol '{parsed.scheme}'. "
                "Supported protocols: http, https, file"
            )
    except Exception as e:
        raise ValueError(f"Invalid ontology IRI format: {str(e)}")


def _validate_min_score(min_score: float) -> None:
    """
    Validate minimum score parameter.
    
    Args:
        min_score: Minimum score value to validate
        
    Raises:
        ValueError: If minimum score is invalid
    """
    if not isinstance(min_score, (int, float)):
        raise ValueError("Minimum score must be between 0.0 and 1.0")
    
    if not (0.0 <= min_score <= 1.0):
        raise ValueError("Minimum score must be between 0.0 and 1.0")


def _validate_term_type(term_type: str) -> None:
    """
    Validate term type parameter.
    
    Args:
        term_type: Term type string to validate
        
    Raises:
        ValueError: If term type is invalid
    """
    valid_term_types = {'class', 'property', 'individual'}
    
    if not isinstance(term_type, str):
        raise ValueError("Invalid term type: must be a string")
    
    if term_type not in valid_term_types:
        raise ValueError(
            f"Invalid term type: '{term_type}'. "
            f"Valid term types are: {', '.join(sorted(valid_term_types))}"
        )


# Utility Functions
def _process_mapping_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and clean mapping results DataFrame.
    
    Args:
        df: Raw mapping results DataFrame from text2term
        
    Returns:
        Processed DataFrame with cleaned data
    """
    if df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Remove rows with null values in critical columns
    critical_columns = ['Source Term', 'Mapped Term IRI', 'Mapping Score']
    for col in critical_columns:
        if col in processed_df.columns:
            processed_df = processed_df.dropna(subset=[col])
    
    # Remove rows with empty string IRIs
    if 'Mapped Term IRI' in processed_df.columns:
        processed_df = processed_df[processed_df['Mapped Term IRI'].str.strip() != '']
    
    # Reset index after filtering
    processed_df = processed_df.reset_index(drop=True)
    
    return processed_df


def _filter_by_score(df: pd.DataFrame, min_score: float) -> pd.DataFrame:
    """
    Filter mapping results by minimum score threshold.
    
    Args:
        df: DataFrame with mapping results
        min_score: Minimum score threshold
        
    Returns:
        Filtered DataFrame with mappings above threshold
    """
    if df.empty or 'Mapping Score' not in df.columns:
        return df
    
    # Filter by minimum score
    filtered_df = df[df['Mapping Score'] >= min_score]
    
    # Reset index after filtering
    filtered_df = filtered_df.reset_index(drop=True)
    
    return filtered_df


def _clean_entities(entities: List[str]) -> List[str]:
    """
    Clean entity strings by removing leading/trailing whitespace.
    
    Args:
        entities: List of entity strings to clean
        
    Returns:
        List of cleaned entity strings
    """
    return [entity.strip() for entity in entities]


def _get_text2term_mapper(mapping_method: str):
    """
    Get the appropriate text2term Mapper enum value.
    
    Args:
        mapping_method: String name of the mapping method
        
    Returns:
        text2term.Mapper enum value
    """
    mapper_mapping = {
        'tfidf': text2term.Mapper.TFIDF,
        'levenshtein': text2term.Mapper.LEVENSHTEIN,
        'jaro_winkler': text2term.Mapper.JARO_WINKLER,
        'jaccard': text2term.Mapper.JACCARD,
        'fuzzy': text2term.Mapper.FUZZY
    }
    
    return mapper_mapping[mapping_method]


# Main Function
def map_entities_to_ontology(
    entities: List[str],
    ontology_iri: str,
    mapping_method: str = 'tfidf',
    min_score: float = 0.3,
    term_type: str = 'class'
) -> pd.DataFrame:
    """
    Map entities to ontology terms using text2term.
    
    This function takes a list of entity strings and maps them to terms in a specified
    ontology using the text2term library. It supports various mapping methods and
    filtering options for high-quality results.
    
    Args:
        entities: List of entity strings to map to ontology terms
        ontology_iri: IRI/URL of the target ontology
        mapping_method: Method to use for mapping ('tfidf', 'levenshtein', 
                       'jaro_winkler', 'jaccard', 'fuzzy'). Defaults to 'tfidf'.
        min_score: Minimum similarity score threshold (0.0-1.0). Defaults to 0.3.
        term_type: Type of ontology terms to map to ('class', 'property', 
                  'individual'). Defaults to 'class'.
    
    Returns:
        pandas.DataFrame: DataFrame with columns:
            - 'Source Term': Original entity string
            - 'Mapped Term Label': Label of the mapped ontology term
            - 'Mapped Term IRI': IRI of the mapped ontology term
            - 'Mapping Score': Similarity score (0.0-1.0)
            - 'Term Type': Type of the ontology term
    
    Raises:
        ValueError: If input parameters are invalid
        OntologyNotFoundError: If the specified ontology cannot be found
        MappingError: If the mapping process fails
        
    Example:
        >>> entities = ["glucose", "arabidopsis", "photosynthesis"]
        >>> ontology_iri = "http://purl.obolibrary.org/obo/chebi.owl"
        >>> results = map_entities_to_ontology(
        ...     entities=entities,
        ...     ontology_iri=ontology_iri,
        ...     mapping_method='tfidf',
        ...     min_score=0.8
        ... )
        >>> print(results)
    """
    # Input validation
    _validate_entities(entities)
    _validate_ontology_iri(ontology_iri)
    _validate_mapping_method(mapping_method)
    _validate_min_score(min_score)
    _validate_term_type(term_type)
    
    # Clean entities
    cleaned_entities = _clean_entities(entities)
    
    # Get text2term mapper
    mapper = _get_text2term_mapper(mapping_method)
    
    try:
        # Call text2term mapping function
        mapping_results = text2term.map_terms(
            source_terms=cleaned_entities,
            target_ontology=ontology_iri,
            mapper=mapper,
            min_score=min_score,
            term_type=term_type,
            incl_unmapped=False
        )
        
    except FileNotFoundError as e:
        raise OntologyNotFoundError(f"Ontology not found: {str(e)}")
    except Exception as e:
        raise MappingError(f"Failed to map entities: {str(e)}")
    
    # Process and filter results
    try:
        processed_results = _process_mapping_results(mapping_results)
        filtered_results = _filter_by_score(processed_results, min_score)
        
        return filtered_results
        
    except Exception as e:
        raise MappingError(f"Failed to process mapping results: {str(e)}")