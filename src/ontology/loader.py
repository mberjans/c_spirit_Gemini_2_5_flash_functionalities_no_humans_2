"""
Ontology Loading Module for AIM2-ODIE-005.

This module provides functionality to load OWL 2.0 ontologies using Owlready2
from URLs or local files. It implements comprehensive error handling for
loading failures and provides informative custom exceptions.

Functions:
    load_ontology_from_file: Load ontology from local file path
    load_ontology_from_url: Load ontology from URL
    
Exceptions:
    OntologyLoadError: Custom exception for ontology loading failures
"""

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import owlready2
from owlready2 import OwlReadyError, OwlReadyOntologyParsingError
import requests


logger = logging.getLogger(__name__)


class OntologyLoadError(Exception):
    """
    Custom exception for ontology loading failures.
    
    This exception is raised when ontology loading fails for any reason,
    providing more informative error messages than the underlying library
    exceptions.
    """
    pass


def _validate_file_path(file_path: str) -> Path:
    """
    Validate and normalize file path for ontology loading.
    
    Args:
        file_path: Path to the ontology file
        
    Returns:
        Path: Validated and resolved absolute path
        
    Raises:
        OntologyLoadError: If file path is invalid or empty
    """
    if not file_path or not file_path.strip():
        raise OntologyLoadError("Invalid file path: path cannot be empty")
    
    path_obj = Path(file_path.strip())
    
    try:
        # Resolve to absolute path
        absolute_path = path_obj.resolve()
        return absolute_path
    except (OSError, RuntimeError) as e:
        raise OntologyLoadError(f"Invalid file path: {e}") from e


def _validate_url(url: str) -> str:
    """
    Validate URL for ontology loading.
    
    Args:
        url: URL to validate
        
    Returns:
        str: Validated URL
        
    Raises:
        OntologyLoadError: If URL is invalid
    """
    if not url or not url.strip():
        raise OntologyLoadError("Invalid URL: URL cannot be empty")
    
    url = url.strip()
    
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise OntologyLoadError("Invalid URL: missing scheme or netloc")
        
        # Only support HTTP and HTTPS
        if parsed.scheme not in ('http', 'https'):
            raise OntologyLoadError(
                f"Invalid URL: unsupported protocol '{parsed.scheme}'. "
                "Only HTTP and HTTPS are supported."
            )
        
        return url
    except Exception as e:
        if isinstance(e, OntologyLoadError):
            raise
        raise OntologyLoadError(f"Invalid URL: {e}") from e


def load_ontology_from_file(file_path: str) -> Any:
    """
    Load an OWL 2.0 ontology from a local file using Owlready2.
    
    This function loads an ontology from a local file path, handling various
    error conditions and providing informative error messages.
    
    Args:
        file_path: Path to the local OWL file
        
    Returns:
        The loaded ontology object from Owlready2
        
    Raises:
        OntologyLoadError: If loading fails for any reason:
            - File not found
            - Permission denied  
            - Invalid OWL format
            - Other Owlready2 errors
            
    Example:
        >>> ontology = load_ontology_from_file("/path/to/ontology.owl")
        >>> print(f"Loaded ontology: {ontology.name}")
    """
    logger.info(f"Loading ontology from file: {file_path}")
    
    # Validate and normalize file path
    absolute_path = _validate_file_path(file_path)
    
    # Check if file exists
    if not absolute_path.exists():
        raise OntologyLoadError(f"File not found: {absolute_path}")
    
    # Check if file is readable
    if not absolute_path.is_file():
        raise OntologyLoadError(f"Path is not a file: {absolute_path}")
    
    try:
        # Create file URI for Owlready2
        file_uri = f"file://{absolute_path}"
        
        # Load ontology using Owlready2
        ontology = owlready2.get_ontology(file_uri)
        loaded_ontology = ontology.load()
        
        logger.info(f"Successfully loaded ontology from file: {file_path}")
        return loaded_ontology
        
    except FileNotFoundError as e:
        raise OntologyLoadError(f"File not found: {e}") from e
    except PermissionError as e:
        raise OntologyLoadError(f"Permission denied: {e}") from e
    except OwlReadyOntologyParsingError as e:
        raise OntologyLoadError(f"Invalid OWL format: {e}") from e
    except OwlReadyError as e:
        raise OntologyLoadError(f"Owlready2 error: {e}") from e
    except Exception as e:
        # Catch any other unexpected errors
        raise OntologyLoadError(f"Unexpected error loading ontology from file: {e}") from e


def load_ontology_from_url(url: str) -> Any:
    """
    Load an OWL 2.0 ontology from a URL using Owlready2.
    
    This function loads an ontology from a remote URL, handling various
    network and parsing error conditions.
    
    Args:
        url: URL of the remote OWL file
        
    Returns:
        The loaded ontology object from Owlready2
        
    Raises:
        OntologyLoadError: If loading fails for any reason:
            - Network connection errors
            - HTTP errors (404, 500, etc.)
            - Request timeouts
            - Invalid OWL format
            - Other Owlready2 errors
            
    Example:
        >>> ontology = load_ontology_from_url("http://example.com/ontology.owl")
        >>> print(f"Loaded ontology: {ontology.name}")
    """
    logger.info(f"Loading ontology from URL: {url}")
    
    # Validate URL
    validated_url = _validate_url(url)
    
    try:
        # Load ontology using Owlready2
        ontology = owlready2.get_ontology(validated_url)
        loaded_ontology = ontology.load()
        
        logger.info(f"Successfully loaded ontology from URL: {url}")
        return loaded_ontology
        
    except requests.exceptions.ConnectionError as e:
        raise OntologyLoadError(f"Network error: Failed to connect to {url}. {e}") from e
    except requests.exceptions.Timeout as e:
        raise OntologyLoadError(f"Network error: Request timeout for {url}. {e}") from e
    except requests.exceptions.HTTPError as e:
        raise OntologyLoadError(f"Network error: HTTP error for {url}. {e}") from e
    except requests.exceptions.RequestException as e:
        raise OntologyLoadError(f"Network error: Request failed for {url}. {e}") from e
    except OwlReadyOntologyParsingError as e:
        raise OntologyLoadError(f"Invalid OWL format: {e}") from e
    except OwlReadyError as e:
        raise OntologyLoadError(f"Owlready2 error: {e}") from e
    except Exception as e:
        # Catch any other unexpected errors
        raise OntologyLoadError(f"Unexpected error loading ontology from URL: {e}") from e


# Export public interface
__all__ = [
    "OntologyLoadError",
    "load_ontology_from_file",
    "load_ontology_from_url",
]