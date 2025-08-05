"""
Entity-to-ontology mapping functionality using text2term.

This module provides functionality for mapping extracted entities to ontology terms using
the text2term library. It supports various mapping methods, minimum similarity scores,
and different term types for flexible ontology integration.

Key Features:
- Multiple mapping methods (TFIDF, Levenshtein, Jaro-Winkler, Jaccard, Fuzzy)
- Minimum score filtering for high-confidence mappings
- Support for different term types (class, property, individual)
- Dual input support: string IRIs and Owlready2 ontology objects
- Comprehensive input validation and error handling
- Integration with standard ontologies (ChEBI, GO, NCBI Taxonomy, etc.)
- Automatic IRI extraction from Owlready2 objects

Usage:
    from src.ontology_mapping.entity_mapper import map_entities_to_ontology
    
    entities = ["glucose", "arabidopsis", "photosynthesis"]
    
    # Using string IRI (backward compatible)
    target_ontology = "http://purl.obolibrary.org/obo/chebi.owl"
    results = map_entities_to_ontology(
        entities=entities,
        target_ontology=target_ontology,
        mapping_method='tfidf',
        min_score=0.8
    )
    
    # Using Owlready2 ontology object (new functionality)
    import owlready2
    onto = owlready2.get_ontology("http://purl.obolibrary.org/obo/chebi.owl").load()
    results = map_entities_to_ontology(
        entities=entities,
        target_ontology=onto,
        mapping_method='tfidf',
        min_score=0.8
    )
"""

import pandas as pd
import re
from typing import List, Optional, Union, Any
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

# Conditional import of owlready2 to avoid hard dependency
try:
    import owlready2
    OWLREADY2_AVAILABLE = True
except ImportError:
    owlready2 = None
    OWLREADY2_AVAILABLE = False


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


class InvalidOwlready2ObjectError(EntityMapperError):
    """Exception raised when an invalid Owlready2 object is provided."""
    pass


# Helper Functions for Owlready2 Integration
def _is_owlready2_ontology(obj: Any) -> bool:
    """
    Check if an object is an Owlready2 ontology.
    
    Args:
        obj: Object to check
        
    Returns:
        bool: True if object is an Owlready2 ontology, False otherwise
    """
    if not OWLREADY2_AVAILABLE:
        return False
    
    # Check if object is an instance of owlready2.Ontology
    try:
        return isinstance(obj, owlready2.Ontology)
    except Exception:
        return False


def _extract_iri_from_owlready2_ontology(ontology: Any) -> str:
    """
    Extract IRI from an Owlready2 ontology object.
    
    Args:
        ontology: Owlready2 ontology object
        
    Returns:
        str: IRI of the ontology
        
    Raises:
        InvalidOwlready2ObjectError: If ontology object is invalid or has no IRI
    """
    if not OWLREADY2_AVAILABLE:
        raise InvalidOwlready2ObjectError(
            "Owlready2 is not available. Install it with: pip install owlready2"
        )
    
    if not _is_owlready2_ontology(ontology):
        raise InvalidOwlready2ObjectError(
            "Object is not a valid Owlready2 ontology. Expected owlready2.Ontology instance."
        )
    
    try:
        # Get the ontology IRI
        iri = ontology.base_iri
        
        if not iri:
            raise InvalidOwlready2ObjectError(
                "Owlready2 ontology does not have a valid base IRI. "
                "Ensure the ontology is properly loaded and has an IRI."
            )
        
        # Remove trailing slash if present for consistency
        if iri.endswith('/'):
            iri = iri[:-1]
        
        return iri
        
    except AttributeError:
        raise InvalidOwlready2ObjectError(
            "Unable to extract IRI from Owlready2 ontology. "
            "The ontology object may be corrupted or improperly loaded."
        )
    except Exception as e:
        raise InvalidOwlready2ObjectError(
            f"Error extracting IRI from Owlready2 ontology: {str(e)}"
        )


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


def _validate_target_ontology(target_ontology: Union[str, Any]) -> str:
    """
    Validate and process target ontology parameter.
    
    This function accepts both string IRIs and Owlready2 ontology objects,
    and returns a validated IRI string for use with text2term.
    
    Args:
        target_ontology: Either a string IRI or an Owlready2 ontology object
        
    Returns:
        str: Validated ontology IRI
        
    Raises:
        ValueError: If the target ontology parameter is invalid
        InvalidOwlready2ObjectError: If Owlready2 object is invalid
    """
    if target_ontology is None:
        raise ValueError("Invalid ontology IRI: cannot be None")
    
    # Handle string IRI input (backward compatibility)
    if isinstance(target_ontology, str):
        _validate_ontology_iri(target_ontology)
        return target_ontology
    
    # Handle Owlready2 ontology object input
    elif _is_owlready2_ontology(target_ontology):
        return _extract_iri_from_owlready2_ontology(target_ontology)
    
    # Invalid input type
    else:
        raise ValueError(
            f"Invalid ontology IRI: must be a string IRI or Owlready2 ontology object, got {type(target_ontology)}. "
            "If using Owlready2, ensure it's installed with: pip install owlready2"
        )


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
    target_ontology: Union[str, Any],
    mapping_method: str = 'tfidf',
    min_score: float = 0.3,
    term_type: str = 'class'
) -> pd.DataFrame:
    """
    Map entities to ontology terms using text2term.
    
    This function takes a list of entity strings and maps them to terms in a specified
    ontology using the text2term library. It supports various mapping methods and
    filtering options for high-quality results.
    
    The target ontology can be specified in two ways:
    1. As a string IRI/URL (backward compatible)
    2. As an Owlready2 ontology object (new functionality for better integration)
    
    Args:
        entities: List of entity strings to map to ontology terms
        target_ontology: Either a string IRI/URL of the target ontology or an 
                        Owlready2 ontology object. If using an Owlready2 object,
                        the IRI will be automatically extracted.
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
        InvalidOwlready2ObjectError: If Owlready2 object is invalid
        OntologyNotFoundError: If the specified ontology cannot be found
        MappingError: If the mapping process fails
        
    Examples:
        Using string IRI (backward compatible):
        >>> entities = ["glucose", "arabidopsis", "photosynthesis"]
        >>> target_ontology = "http://purl.obolibrary.org/obo/chebi.owl"
        >>> results = map_entities_to_ontology(
        ...     entities=entities,
        ...     target_ontology=target_ontology,
        ...     mapping_method='tfidf',
        ...     min_score=0.8
        ... )
        >>> print(results)
        
        Using Owlready2 ontology object:
        >>> import owlready2
        >>> onto = owlready2.get_ontology("http://purl.obolibrary.org/obo/chebi.owl").load()
        >>> results = map_entities_to_ontology(
        ...     entities=entities,
        ...     target_ontology=onto,
        ...     mapping_method='tfidf',
        ...     min_score=0.8
        ... )
        >>> print(results)
    """
    # Input validation
    _validate_entities(entities)
    ontology_iri = _validate_target_ontology(target_ontology)
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