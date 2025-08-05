"""
Ontology Export Module for AIM2-ODIE-008-T2.

This module provides functionality to export OWL 2.0 ontologies using Owlready2
to various formats including RDF/XML, OWL/XML, N-Triples, and Turtle. It implements
comprehensive error handling for export failures and provides informative custom exceptions.

Functions:
    export_ontology: Export ontology to file in specified format
    
Exceptions:
    OntologyExportError: Custom exception for ontology export failures
"""

import logging
from pathlib import Path
from typing import Any, Optional
import threading

import owlready2
from owlready2 import OwlReadyError


logger = logging.getLogger(__name__)
_export_lock = threading.Lock()


class OntologyExportError(Exception):
    """
    Custom exception for ontology export failures.
    
    This exception is raised when ontology export fails for any reason,
    providing more informative error messages than the underlying library
    exceptions.
    """
    pass


def _validate_ontology(ontology: Any) -> None:
    """
    Validate that the provided ontology object is valid for export operations.
    
    Args:
        ontology: The ontology object to validate
        
    Raises:
        OntologyExportError: If ontology is invalid or None
    """
    if ontology is None:
        raise OntologyExportError("Invalid ontology: ontology cannot be None")
    
    # Check if the ontology has the expected save method
    if not hasattr(ontology, 'save'):
        raise OntologyExportError("Invalid ontology: ontology object must have a 'save' method")


def _validate_file_path(file_path: str) -> str:
    """
    Validate and normalize file path for ontology export.
    
    Args:
        file_path: Path to the export file
        
    Returns:
        str: Validated file path
        
    Raises:
        OntologyExportError: If file path is invalid or empty
    """
    if not file_path or not isinstance(file_path, str) or not file_path.strip():
        raise OntologyExportError("Invalid file path: path cannot be empty or None")
    
    # Return the stripped path
    return file_path.strip()


def _validate_format(format_str: str) -> str:
    """
    Validate export format.
    
    Args:
        format_str: Format string to validate
        
    Returns:
        str: Validated format string
        
    Raises:
        OntologyExportError: If format is invalid or unsupported
    """
    if not format_str or not isinstance(format_str, str):
        raise OntologyExportError("Invalid format: format cannot be None or empty")
    
    # Supported formats based on test requirements
    supported_formats = {'rdfxml', 'owlxml', 'ntriples', 'turtle'}
    format_lower = format_str.lower().strip()
    
    if format_lower not in supported_formats:
        raise OntologyExportError(
            f"Invalid format: '{format_str}'. Supported formats are: {', '.join(sorted(supported_formats))}"
        )
    
    return format_lower


def _create_parent_directories(file_path: str) -> None:
    """
    Create parent directories for the export file if they don't exist.
    
    Args:
        file_path: Path to the export file
        
    Raises:
        OntologyExportError: If directory creation fails
    """
    try:
        path_obj = Path(file_path)
        parent_dir = path_obj.parent
        
        # Create parent directories if they don't exist
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created parent directories for: {file_path}")
            
    except PermissionError as e:
        # Specifically handle permission errors for directory creation
        raise OntologyExportError(f"Permission denied: Unable to create directories for {file_path}. Check write permissions. {e}") from e
    except OSError as e:
        # Handle other OS errors during directory creation
        error_msg = str(e).lower()
        if "read-only" in error_msg:
            raise OntologyExportError(f"Permission denied: Cannot write to read-only file system. {e}") from e
        else:
            raise OntologyExportError(f"Failed to create parent directories: {e}") from e


def export_ontology(ontology: Any, file_path: str, format: str = 'rdfxml') -> None:
    """
    Export an OWL 2.0 ontology to a file using Owlready2.
    
    This function exports an ontology to a specified file path in the given format,
    handling various error conditions and providing informative error messages.
    The function is thread-safe and supports file overwriting.
    
    Args:
        ontology: The ontology object to export (must have a 'save' method)
        file_path: Path where the ontology should be exported
        format: Export format - one of 'rdfxml', 'owlxml', 'ntriples', 'turtle'
               (default: 'rdfxml')
        
    Raises:
        OntologyExportError: If export fails for any reason:
            - Invalid ontology object (None or missing save method)
            - Invalid file path (None, empty, or invalid)
            - Invalid format (unsupported format)
            - Permission denied (write access issues)
            - Disk space issues
            - Other Owlready2 errors during save
            
    Example:
        >>> export_ontology(my_ontology, "/path/to/export.owl")
        >>> export_ontology(my_ontology, "/path/to/export.xml", format="owlxml")
    """
    # Use lock for thread safety
    with _export_lock:
        logger.info(f"Exporting ontology to file: {file_path} in format: {format}")
        
        # Validate inputs
        _validate_ontology(ontology)
        validated_path = _validate_file_path(file_path)
        validated_format = _validate_format(format)
        
        try:
            # Create parent directories if needed
            _create_parent_directories(validated_path)
            
            # Export ontology using Owlready2
            logger.debug(f"Calling ontology.save with file={validated_path}, format={validated_format}")
            ontology.save(file=validated_path, format=validated_format)
            
            # Verify file was created
            exported_file = Path(validated_path)
            if not exported_file.exists():
                raise OntologyExportError(f"Export appeared to succeed but file was not created: {validated_path}")
            
            logger.info(f"Successfully exported ontology to: {validated_path}")
            
        except PermissionError as e:
            raise OntologyExportError(f"Permission denied: Unable to write to {validated_path}. Check write permissions. {e}") from e
        except FileNotFoundError as e:
            raise OntologyExportError(f"File path error: {e}") from e
        except OSError as e:
            # Handle disk space and other OS-level errors
            error_msg = str(e).lower()
            if "no space left" in error_msg or "disk full" in error_msg:
                raise OntologyExportError(f"Insufficient disk space: {e}") from e
            else:
                raise OntologyExportError(f"File system error: {e}") from e
        except ValueError as e:
            # Handle format validation errors from Owlready2
            if "format" in str(e).lower():
                raise OntologyExportError(f"Unsupported format: {e}") from e
            else:
                raise OntologyExportError(f"Invalid parameter: {e}") from e
        except OwlReadyError as e:
            raise OntologyExportError(f"Owlready2 error during export: {e}") from e
        except Exception as e:
            # Catch any other unexpected errors
            raise OntologyExportError(f"Unexpected error during ontology export: {e}") from e


# Export public interface
__all__ = [
    "OntologyExportError",
    "export_ontology",
]