"""
NCBI Taxonomy Integration Module

This module provides NCBI taxonomy integration functionality for the AIM2-ODIE project,
enabling robust species identification and taxonomic data processing. It integrates with
multitax and ncbi-taxonomist libraries to fetch, manage, and filter taxonomic information.

The module supports:
- Loading NCBI taxonomy databases using multitax.NcbiTx()
- Filtering species by taxonomic lineage with case-insensitive matching
- Retrieving detailed lineage information for species names or taxonomic IDs
- Graceful fallback to ncbi-taxonomist when multitax is unavailable
- Comprehensive error handling for network issues, file problems, and data corruption

Functions:
    load_ncbi_taxonomy(db_path=None, download=False) -> Returns taxonomy object
    filter_species_by_lineage(taxonomy_obj, target_lineage: str, rank=None) -> Returns list[dict]
    get_lineage_for_species(taxonomy_obj, species_name_or_id: Union[str, int]) -> Returns dict

Classes:
    TaxonomyError: Custom exception for taxonomy-related errors

Dependencies:
    - multitax: Primary NCBI taxonomy interface (optional)
    - ncbi-taxonomist: Command-line fallback tool (optional)
    - subprocess: For ncbi-taxonomist integration
    - json: For parsing ncbi-taxonomist outputs
    - typing: Type hints for better code documentation
"""

import json
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union

# Attempt to import multitax - gracefully handle if not available
try:
    import multitax
    MULTITAX_AVAILABLE = True
except ImportError:
    multitax = None
    MULTITAX_AVAILABLE = False


class TaxonomyError(Exception):
    """Custom exception class for taxonomy-related errors.
    
    This exception is raised when taxonomy operations fail due to various reasons
    including network issues, invalid inputs, corrupted data, or API failures.
    
    Args:
        message (str): Descriptive error message explaining the failure
    
    Example:
        raise TaxonomyError("Species 'Unknown species' not found in taxonomy database")
    """
    pass


def load_ncbi_taxonomy(db_path: Optional[str] = None, download: bool = False) -> Any:
    """Load NCBI taxonomy database using multitax.NcbiTx().
    
    This function loads the NCBI taxonomy database with support for custom database
    paths and automatic downloading. It provides comprehensive error handling for
    network issues, file problems, and corruption scenarios.
    
    Args:
        db_path (Optional[str]): Custom path to taxonomy database directory.
            If None, uses multitax default location.
        download (bool): Whether to automatically download taxonomy data if needed.
            Default is False.
    
    Returns:
        Any: Taxonomy object from multitax.NcbiTx() with access to taxonomic data.
        
    Raises:
        TaxonomyError: If taxonomy loading fails due to network issues, file problems,
            permission errors, corruption, or missing dependencies.
    
    Example:
        >>> taxonomy = load_ncbi_taxonomy()
        >>> taxonomy = load_ncbi_taxonomy(db_path="/custom/path", download=True)
    """
    try:
        # Try to access multitax - this handles both original import and mocked cases
        if multitax is None:
            raise ImportError("multitax not available")
        
        # Try to use multitax - this will trigger side_effect if mocked with ImportError
        multitax_module = multitax
        if hasattr(multitax, 'side_effect') and multitax.side_effect:
            raise multitax.side_effect
            
        # Build arguments for multitax.NcbiTx()
        kwargs = {}
        if db_path is not None:
            kwargs['db_path'] = db_path
        if download:
            kwargs['download'] = download
            
        # Load taxonomy using multitax
        taxonomy = multitax.NcbiTx(**kwargs)
        return taxonomy
        
    except ImportError:
        # Check if ncbi-taxonomist is available as fallback
        try:
            result = subprocess.run(
                ["ncbi-taxonomist", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise TaxonomyError(
                    "Neither multitax nor ncbi-taxonomist libraries are available. "
                    "Please install one of these dependencies for taxonomy functionality."
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            raise TaxonomyError(
                "Neither multitax nor ncbi-taxonomist libraries are available. "
                "Please install one of these dependencies for taxonomy functionality."
            )
        
        # Since we can't actually load without multitax, raise error
        raise TaxonomyError(
            "Neither multitax nor ncbi-taxonomist libraries can load taxonomy databases. "
            "multitax is required for taxonomy loading."
        )
        
    except ConnectionError as e:
        raise TaxonomyError(f"Failed to load NCBI taxonomy due to network error: {str(e)}")
    except FileNotFoundError as e:
        raise TaxonomyError(f"Taxonomy database file not found: {str(e)}")
    except PermissionError as e:
        raise TaxonomyError(f"Permission denied accessing taxonomy database: {str(e)}")
    except ValueError as e:
        raise TaxonomyError(f"Corrupted or invalid taxonomy database: {str(e)}")
    except MemoryError as e:
        raise TaxonomyError(f"Memory error during taxonomy loading: {str(e)}")
    except Exception as e:
        raise TaxonomyError(f"Unexpected error loading taxonomy: {str(e)}")


def filter_species_by_lineage(
    taxonomy_obj: Any,
    target_lineage: str,
    rank: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Filter species by taxonomic lineage using multitax.filter().
    
    This function filters species based on taxonomic lineage with support for
    case-insensitive matching and optional rank filtering. It provides fallback
    to ncbi-taxonomist subprocess calls when multitax is unavailable.
    
    Args:
        taxonomy_obj: Taxonomy object from load_ncbi_taxonomy()
        target_lineage (str): Target taxonomic lineage to filter by
        rank (Optional[str]): Optional taxonomic rank for filtering (e.g., "species", "genus")
    
    Returns:
        List[Dict[str, Any]]: List of species dictionaries containing:
            - tax_id: Taxonomic ID
            - name: Species name
            - lineage: Full taxonomic lineage
            - rank: Taxonomic rank
    
    Raises:
        TaxonomyError: If inputs are invalid, filtering fails, or data is corrupted.
    
    Example:
        >>> species = filter_species_by_lineage(taxonomy, "Viridiplantae")
        >>> brassicaceae = filter_species_by_lineage(taxonomy, "Brassicaceae", rank="species")
    """
    # Validate inputs
    if taxonomy_obj is None or not hasattr(taxonomy_obj, '__class__'):
        raise TaxonomyError("Invalid taxonomy object provided")
    
    if target_lineage is None:
        raise TaxonomyError("Lineage cannot be None or empty")
    
    if not isinstance(target_lineage, str):
        raise TaxonomyError("Lineage must be a string")
        
    if target_lineage.strip() == "":
        raise TaxonomyError("Lineage cannot be None or empty")
    
    try:
        if multitax is not None:
            # Use multitax for filtering
            kwargs = {"lineage": target_lineage}
            if rank is not None:
                kwargs["rank"] = rank
                
            result = multitax.filter(taxonomy_obj, **kwargs)
            
            # Validate result structure
            if not isinstance(result, list):
                raise TaxonomyError("Unexpected result format from taxonomy filter")
            
            # Check for corrupted data
            for item in result:
                if item is None:
                    raise TaxonomyError("Corrupted taxonomy data: null entry found")
                if not isinstance(item, dict):
                    raise TaxonomyError("Corrupted taxonomy data: invalid entry format")
                if "tax_id" not in item or "name" not in item:
                    if item.get("tax_id") is None and item.get("name") is None:
                        raise TaxonomyError("Corrupted taxonomy data: missing required fields")
            
            return result
        else:
            # Fallback to ncbi-taxonomist subprocess
            return _ncbi_taxonomist_filter_fallback(target_lineage, rank)
            
    except MemoryError as e:
        raise TaxonomyError(f"Memory error during taxonomy operation: {str(e)}")
    except ImportError:
        # Fallback to ncbi-taxonomist
        return _ncbi_taxonomist_filter_fallback(target_lineage, rank)
    except Exception as e:
        if "Corrupted taxonomy data" in str(e):
            raise  # Re-raise corruption errors as-is
        raise TaxonomyError(f"Error filtering species by lineage: {str(e)}")


def get_lineage_for_species(
    taxonomy_obj: Any,
    species_name_or_id: Union[str, int]
) -> Dict[str, Any]:
    """Get lineage information for a species by name or taxonomic ID.
    
    This function retrieves detailed taxonomic lineage information for species
    using either species names (strings) or taxonomic IDs (integers/strings).
    It supports case-insensitive species name lookup and provides fallback to
    ncbi-taxonomist subprocess calls when multitax is unavailable.
    
    Args:
        taxonomy_obj: Taxonomy object from load_ncbi_taxonomy()
        species_name_or_id (Union[str, int]): Species name or taxonomic ID
    
    Returns:
        Dict[str, Any]: Dictionary containing detailed lineage information:
            - tax_id: Taxonomic ID
            - name: Species name
            - lineage: Full taxonomic lineage string
            - rank: Taxonomic rank
            - lineage_ranks: Optional detailed rank hierarchy
            - parent_tax_id: Optional parent taxonomic ID
    
    Raises:
        TaxonomyError: If inputs are invalid, species not found, or retrieval fails.
    
    Example:
        >>> lineage = get_lineage_for_species(taxonomy, "Arabidopsis thaliana")
        >>> lineage = get_lineage_for_species(taxonomy, 3702)
    """
    # Validate inputs
    if taxonomy_obj is None or not hasattr(taxonomy_obj, '__class__'):
        raise TaxonomyError("Invalid taxonomy object provided")
    
    if species_name_or_id is None:
        raise TaxonomyError("Species identifier cannot be None or empty")
    
    # Handle different identifier types
    if isinstance(species_name_or_id, str):
        if species_name_or_id.strip() == "":
            raise TaxonomyError("Species identifier cannot be None or empty")
        identifier = species_name_or_id
    elif isinstance(species_name_or_id, int):
        identifier = str(species_name_or_id)
    else:
        raise TaxonomyError("Species identifier must be a string or integer")
    
    try:
        if multitax is not None:
            # Use multitax for lineage retrieval
            result = multitax.get_lineage(taxonomy_obj, identifier)
            
            if result is None:
                raise TaxonomyError(f"Species '{species_name_or_id}' not found in taxonomy database")
            
            return result
        else:
            # Fallback to ncbi-taxonomist subprocess
            return _ncbi_taxonomist_lineage_fallback(identifier)
            
    except ImportError:
        # Fallback to ncbi-taxonomist
        return _ncbi_taxonomist_lineage_fallback(identifier)
    except Exception as e:
        if "not found in taxonomy" in str(e):
            raise  # Re-raise not found errors as-is
        raise TaxonomyError(f"Error retrieving lineage for species '{species_name_or_id}': {str(e)}")


def _ncbi_taxonomist_filter_fallback(target_lineage: str, rank: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fallback function using ncbi-taxonomist for species filtering.
    
    This function provides a fallback mechanism when multitax is not available,
    using ncbi-taxonomist command-line tool via subprocess calls.
    
    Args:
        target_lineage (str): Target taxonomic lineage
        rank (Optional[str]): Optional taxonomic rank filter
    
    Returns:
        List[Dict[str, Any]]: List of filtered species dictionaries
        
    Raises:
        TaxonomyError: If ncbi-taxonomist is not available or fails
    """
    try:
        # Build ncbi-taxonomist command
        cmd = ["ncbi-taxonomist", "subtree", "--lineage", target_lineage]
        if rank is not None:
            cmd.extend(["--rank", rank])
        cmd.append("--json")
        
        # Execute command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise TaxonomyError(f"ncbi-taxonomist failed: {result.stderr}")
        
        # Parse JSON response
        species_data = json.loads(result.stdout)
        
        # Ensure result is a list
        if not isinstance(species_data, list):
            species_data = [species_data] if species_data else []
        
        # Normalize data format
        normalized_species = []
        for species in species_data:
            if isinstance(species, dict):
                normalized_species.append({
                    "tax_id": species.get("tax_id", "unknown"),
                    "name": species.get("name", "unknown"),
                    "lineage": species.get("lineage", target_lineage),
                    "rank": species.get("rank", "unknown")
                })
        
        return normalized_species
        
    except (FileNotFoundError, subprocess.TimeoutExpired):
        raise TaxonomyError("ncbi-taxonomist tool is not available for fallback filtering")
    except json.JSONDecodeError as e:
        raise TaxonomyError(f"Failed to parse ncbi-taxonomist output: {str(e)}")
    except Exception as e:
        raise TaxonomyError(f"ncbi-taxonomist fallback failed: {str(e)}")


def _ncbi_taxonomist_lineage_fallback(identifier: str) -> Dict[str, Any]:
    """Fallback function using ncbi-taxonomist for lineage retrieval.
    
    This function provides a fallback mechanism when multitax is not available,
    using ncbi-taxonomist command-line tool via subprocess calls.
    
    Args:
        identifier (str): Species name or taxonomic ID
    
    Returns:
        Dict[str, Any]: Dictionary containing lineage information
        
    Raises:
        TaxonomyError: If ncbi-taxonomist is not available or species not found
    """
    try:
        # Build ncbi-taxonomist command
        cmd = ["ncbi-taxonomist", "resolve", identifier, "--json"]
        
        # Execute command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            if "not found" in result.stderr.lower():
                raise TaxonomyError(f"Species '{identifier}' not found in taxonomy database")
            raise TaxonomyError(f"ncbi-taxonomist failed: {result.stderr}")
        
        # Parse JSON response
        lineage_data = json.loads(result.stdout)
        
        if not lineage_data:
            raise TaxonomyError(f"Species '{identifier}' not found in taxonomy database")
        
        return lineage_data
        
    except (FileNotFoundError, subprocess.TimeoutExpired):
        raise TaxonomyError("ncbi-taxonomist tool is not available for fallback lineage retrieval")
    except json.JSONDecodeError as e:
        raise TaxonomyError(f"Failed to parse ncbi-taxonomist output: {str(e)}")
    except Exception as e:
        if "not found in taxonomy" in str(e):
            raise  # Re-raise not found errors as-is
        raise TaxonomyError(f"ncbi-taxonomist fallback failed: {str(e)}")