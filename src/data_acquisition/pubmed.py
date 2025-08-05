"""
PubMed/PMC Data Acquisition Module

This module provides functionality to search and retrieve abstracts/full texts
from PubMed/PMC using Biopython.Entrez. It implements rate limiting and
comprehensive error handling for robust data acquisition.

Key Features:
- Search PubMed using keywords with configurable result limits
- Fetch XML content for retrieved PubMed IDs
- Rate limiting to comply with NCBI E-utilities guidelines
- Comprehensive error handling for network and API issues
- Email configuration for NCBI Entrez access

Dependencies:
- Biopython for Entrez API access
- Threading for rate limiting implementation
"""

import logging
import time
import threading
from typing import List, Optional
from urllib.error import URLError, HTTPError
import re

try:
    from Bio import Entrez
    from Bio.Entrez import Parser
except ImportError as e:
    raise ImportError(f"Biopython is required for PubMed access: {e}")

# Set up logging
logger = logging.getLogger(__name__)

# Global rate limiter
_rate_limiter = None
_rate_limiter_lock = threading.Lock()

# Default rate limiting settings (NCBI guidelines)
DEFAULT_REQUESTS_PER_SECOND = 3  # Without API key
API_KEY_REQUESTS_PER_SECOND = 10  # With API key


class PubMedError(Exception):
    """Custom exception for PubMed-related errors."""
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        """
        Initialize PubMedError.
        
        Args:
            message: Error message
            cause: Optional underlying exception that caused this error
        """
        super().__init__(message)
        self.cause = cause
        if cause:
            self.message = f"{message}. Caused by: {str(cause)}"
        else:
            self.message = message
    
    def __str__(self):
        return self.message


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second allowed
        """
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self.lock = threading.Lock()
    
    def acquire(self):
        """Acquire permission to make a request, blocking if necessary."""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.3f} seconds")
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()


def get_rate_limiter() -> RateLimiter:
    """
    Get the global rate limiter instance.
    
    Returns:
        RateLimiter: Global rate limiter instance
    """
    global _rate_limiter
    
    if _rate_limiter is None:
        with _rate_limiter_lock:
            if _rate_limiter is None:
                _rate_limiter = RateLimiter()
    
    return _rate_limiter


def set_entrez_email(email: str) -> None:
    """
    Set the email address for NCBI Entrez access.
    
    NCBI requires an email address to be set for API access.
    
    Args:
        email: Valid email address
        
    Raises:
        ValueError: If email format is invalid
        PubMedError: If email setting fails
    """
    if not email or not isinstance(email, str):
        raise ValueError("Email must be a non-empty string")
    
    email = email.strip()
    if not email:
        raise ValueError("Email cannot be empty or whitespace")
    
    # Basic email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        raise ValueError(f"Invalid email format: {email}")
    
    try:
        Entrez.email = email
        logger.info(f"Entrez email set to: {email}")
    except Exception as e:
        raise PubMedError(f"Failed to set Entrez email: {e}", e)


def search_pubmed(query: str, max_results: int = 100) -> List[str]:
    """
    Search PubMed using keywords and return list of PubMed IDs.
    
    This function searches PubMed using the provided query string and returns
    a list of PubMed IDs for matching articles. Rate limiting is automatically
    applied to comply with NCBI guidelines.
    
    Args:
        query: Search query string (supports PubMed search syntax)
        max_results: Maximum number of results to return (default: 100)
        
    Returns:
        List[str]: List of PubMed IDs as strings
        
    Raises:
        PubMedError: If search fails for any reason:
            - Network connection errors
            - Invalid query format
            - NCBI API errors
            - Rate limiting failures
            
    Example:
        >>> ids = search_pubmed("plant metabolites", max_results=50)
        >>> print(f"Found {len(ids)} articles")
    """
    logger.info(f"Searching PubMed for: '{query}' (max_results: {max_results})")
    
    # Validate inputs
    if not query or not isinstance(query, str):
        raise PubMedError("Query must be a non-empty string")
    
    query = query.strip()
    if not query:
        raise PubMedError("Query cannot be empty or whitespace")
    
    if max_results <= 0:
        raise PubMedError("max_results must be positive")
    
    # Check if email is set
    if not hasattr(Entrez, 'email') or not Entrez.email:
        logger.warning("Entrez.email not set - this may cause issues with NCBI API")
    
    # Apply rate limiting
    rate_limiter = get_rate_limiter()
    rate_limiter.acquire()
    
    try:
        # Perform PubMed search
        logger.debug(f"Executing Entrez.esearch with query: {query}")
        
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            retmode="xml"
        )
        
        # Parse the results
        search_results = Entrez.read(handle)
        handle.close()
        
        # Extract ID list
        id_list = search_results.get("IdList", [])
        
        logger.info(f"Found {len(id_list)} PubMed IDs for query: '{query}'")
        return id_list
        
    except URLError as e:
        error_msg = f"Network error during PubMed search: {e}"
        logger.error(error_msg)
        raise PubMedError(error_msg, e)
    
    except HTTPError as e:
        error_msg = f"HTTP error during PubMed search: {e}"
        logger.error(error_msg)
        raise PubMedError(error_msg, e)
    
    except Parser.ValidationError as e:
        error_msg = f"Invalid query or response format: {e}"
        logger.error(error_msg)
        raise PubMedError(error_msg, e)
    
    except Exception as e:
        error_msg = f"Unexpected error during PubMed search: {e}"
        logger.error(error_msg)
        raise PubMedError(error_msg, e)


def fetch_pubmed_xml(id_list: List[str]) -> str:
    """
    Fetch XML content for a list of PubMed IDs.
    
    This function retrieves the full XML content for the specified PubMed IDs
    using NCBI's efetch service. Rate limiting is automatically applied.
    
    Args:
        id_list: List of PubMed IDs as strings
        
    Returns:
        str: XML content containing article information
        
    Raises:
        PubMedError: If fetching fails for any reason:
            - Network connection errors
            - Invalid PubMed IDs
            - NCBI API errors
            - Rate limiting failures
            
    Example:
        >>> xml_content = fetch_pubmed_xml(["12345678", "87654321"])
        >>> print(f"Retrieved XML content: {len(xml_content)} characters")
    """
    logger.info(f"Fetching XML for {len(id_list)} PubMed IDs")
    
    # Validate inputs
    if not id_list or not isinstance(id_list, list):
        raise PubMedError("id_list must be a non-empty list")
    
    if not all(isinstance(id_str, str) for id_str in id_list):
        raise PubMedError("All IDs must be strings")
    
    # Filter out empty IDs
    valid_ids = [id_str.strip() for id_str in id_list if id_str.strip()]
    if not valid_ids:
        raise PubMedError("No valid IDs provided")
    
    # Check if email is set
    if not hasattr(Entrez, 'email') or not Entrez.email:
        logger.warning("Entrez.email not set - this may cause issues with NCBI API")
    
    # Apply rate limiting
    rate_limiter = get_rate_limiter()
    rate_limiter.acquire()
    
    try:
        # Join IDs for batch fetching
        id_string = ",".join(valid_ids)
        logger.debug(f"Executing Entrez.efetch for IDs: {id_string}")
        
        # Fetch XML content
        handle = Entrez.efetch(
            db="pubmed",
            id=id_string,
            rettype="xml",
            retmode="xml"
        )
        
        # Read the XML content
        xml_content = handle.read()
        handle.close()
        
        logger.info(f"Successfully fetched XML content: {len(xml_content)} characters")
        return xml_content
        
    except URLError as e:
        error_msg = f"Network error during PubMed fetch: {e}"
        logger.error(error_msg)
        raise PubMedError(error_msg, e)
    
    except HTTPError as e:
        error_msg = f"HTTP error during PubMed fetch: {e}"
        logger.error(error_msg)
        raise PubMedError(error_msg, e)
    
    except Exception as e:
        error_msg = f"Unexpected error during PubMed fetch: {e}"
        logger.error(error_msg)
        raise PubMedError(error_msg, e)


def configure_api_key(api_key: str) -> None:
    """
    Configure NCBI API key for higher rate limits.
    
    With an API key, the rate limit increases from 3 to 10 requests per second.
    
    Args:
        api_key: NCBI API key
        
    Raises:
        ValueError: If API key format is invalid
        PubMedError: If API key configuration fails
    """
    if not api_key or not isinstance(api_key, str):
        raise ValueError("API key must be a non-empty string")
    
    api_key = api_key.strip()
    if not api_key:
        raise ValueError("API key cannot be empty or whitespace")
    
    try:
        Entrez.api_key = api_key
        
        # Update rate limiter for higher limits with API key
        global _rate_limiter
        with _rate_limiter_lock:
            _rate_limiter = RateLimiter(API_KEY_REQUESTS_PER_SECOND)
        
        logger.info("NCBI API key configured - rate limit increased to 10 requests/second")
        
    except Exception as e:
        raise PubMedError(f"Failed to configure API key: {e}", e)


def search_and_fetch(query: str, max_results: int = 100) -> str:
    """
    Convenience function to search PubMed and fetch XML in one operation.
    
    This function combines search_pubmed and fetch_pubmed_xml for common
    use cases where you want to search and immediately fetch the results.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to search for
        
    Returns:
        str: XML content for all found articles
        
    Raises:
        PubMedError: If search or fetch operations fail
        
    Example:
        >>> xml_data = search_and_fetch("plant metabolites", max_results=20)
        >>> print(f"Retrieved data for query: {len(xml_data)} characters")
    """
    logger.info(f"Search and fetch for: '{query}' (max_results: {max_results})")
    
    # Search for IDs
    id_list = search_pubmed(query, max_results)
    
    if not id_list:
        logger.warning(f"No results found for query: '{query}'")
        return ""
    
    # Fetch XML content
    xml_content = fetch_pubmed_xml(id_list)
    
    logger.info(f"Search and fetch completed: {len(id_list)} articles, {len(xml_content)} characters")
    return xml_content


# Module initialization
logger.info("PubMed data acquisition module loaded successfully")