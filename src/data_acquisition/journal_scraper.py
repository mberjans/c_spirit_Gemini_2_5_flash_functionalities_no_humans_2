"""
Scientific Journal Web Scraper Module

This module provides functionality to scrape metadata and full-text content
from scientific journals using paperscraper and web scraping techniques.
It implements comprehensive bot protection handling, rate limiting, and
robots.txt compliance for robust and ethical data acquisition.

Key Features:
- Scientific journal metadata scraping using paperscraper
- Full-text PDF/XML download with proper error handling
- Bot protection mechanisms (User-Agent rotation, request throttling)
- Robots.txt compliance checking and enforcement
- Rate limiting to respect server resources
- Comprehensive error handling for network and parsing issues

Dependencies:
- paperscraper for journal-specific scraping functionality
- requests for HTTP operations and header management
- urllib.robotparser for robots.txt compliance
- time for rate limiting and throttling
"""

import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from urllib.error import URLError, HTTPError
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import random
import os

try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
except ImportError as e:
    raise ImportError(f"requests is required for journal scraping: {e}")

try:
    import paperscraper
except ImportError as e:
    raise ImportError(f"paperscraper is required for journal scraping: {e}")

# Set up logging
logger = logging.getLogger(__name__)

# Global rate limiter
_rate_limiter = None
_rate_limiter_lock = threading.Lock()

# Default rate limiting settings (conservative for web scraping)
DEFAULT_REQUESTS_PER_SECOND = 1.0  # Conservative rate for web scraping
DEFAULT_MIN_DELAY_SECONDS = 2.0    # Minimum delay between requests

# User agent strings for rotation (academic/research purposes)
DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
]

# Request timeout settings
REQUEST_TIMEOUT = 30  # seconds
CONNECT_TIMEOUT = 10  # seconds


class JournalScraperError(Exception):
    """Custom exception for journal scraping-related errors."""
    
    def __init__(self, message: str, cause: Optional[Exception] = None, url: Optional[str] = None):
        """
        Initialize JournalScraperError.
        
        Args:
            message: Error message
            cause: Optional underlying exception that caused this error
            url: Optional URL that caused the error
        """
        super().__init__(message)
        self.cause = cause
        self.url = url
        if cause and url:
            self.message = f"{message} [URL: {url}]. Caused by: {str(cause)}"
        elif cause:
            self.message = f"{message}. Caused by: {str(cause)}"
        elif url:
            self.message = f"{message} [URL: {url}]"
        else:
            self.message = message
    
    def __str__(self):
        return self.message


class RateLimiter:
    """Rate limiter for web scraping requests."""
    
    def __init__(self, requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
                 min_delay_seconds: float = DEFAULT_MIN_DELAY_SECONDS):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second allowed
            min_delay_seconds: Minimum delay between requests
        """
        self.requests_per_second = requests_per_second
        self.min_interval = max(1.0 / requests_per_second, min_delay_seconds)
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


class UserAgentRotator:
    """User agent rotation for bot protection."""
    
    def __init__(self, user_agents: Optional[List[str]] = None):
        """
        Initialize user agent rotator.
        
        Args:
            user_agents: List of user agent strings to rotate between
        """
        self.user_agents = user_agents or DEFAULT_USER_AGENTS.copy()
        self.current_index = 0
        self.lock = threading.Lock()
    
    def get_next_user_agent(self) -> str:
        """
        Get the next user agent in rotation.
        
        Returns:
            str: User agent string
        """
        with self.lock:
            user_agent = self.user_agents[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.user_agents)
            return user_agent
    
    def get_random_user_agent(self) -> str:
        """
        Get a random user agent.
        
        Returns:
            str: Random user agent string
        """
        return random.choice(self.user_agents)


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


def create_session_with_retries(
    retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: Optional[List[int]] = None
) -> requests.Session:
    """
    Create a requests session with retry strategy.
    
    Args:
        retries: Number of retry attempts
        backoff_factor: Backoff factor for retries
        status_forcelist: HTTP status codes to retry on
        
    Returns:
        requests.Session: Configured session with retry strategy
    """
    if status_forcelist is None:
        status_forcelist = [500, 502, 503, 504, 429]
    
    session = requests.Session()
    
    retry_strategy = Retry(
        total=retries,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "OPTIONS"],  # Updated parameter name
        backoff_factor=backoff_factor
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


def check_robots_txt(url: str, user_agent: str = "*") -> bool:
    """
    Check if URL is allowed by robots.txt.
    
    This function checks the robots.txt file for the given URL's domain
    to determine if scraping is allowed for the specified user agent.
    
    Args:
        url: URL to check
        user_agent: User agent string to check against robots.txt
        
    Returns:
        bool: True if URL is allowed, False if disallowed
        
    Raises:
        JournalScraperError: If robots.txt checking fails
        
    Example:
        >>> is_allowed = check_robots_txt("https://example.com/article")
        >>> if not is_allowed:
        ...     print("Scraping not allowed by robots.txt")
    """
    logger.debug(f"Checking robots.txt for URL: {url}")
    
    # Validate inputs
    if not url or not isinstance(url, str):
        raise JournalScraperError("URL must be a non-empty string")
    
    url = url.strip()
    if not url:
        raise JournalScraperError("URL cannot be empty or whitespace")
    
    try:
        # Parse URL to get base domain
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise JournalScraperError(f"Invalid URL format: {url}")
        
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        robots_url = urljoin(base_url, "/robots.txt")
        
        logger.debug(f"Checking robots.txt at: {robots_url}")
        
        # Create and configure robot parser
        rp = RobotFileParser()
        rp.set_url(robots_url)
        
        # Read robots.txt with timeout
        try:
            rp.read()
        except URLError as e:
            # If robots.txt doesn't exist or can't be read, assume allowed
            logger.warning(f"Could not read robots.txt from {robots_url}: {e}")
            return True
        
        # Check if URL is allowed
        is_allowed = rp.can_fetch(user_agent, url)
        
        logger.debug(f"Robots.txt check result: {'allowed' if is_allowed else 'disallowed'}")
        return is_allowed
        
    except Exception as e:
        error_msg = f"Error checking robots.txt for URL: {url}"
        logger.error(error_msg)
        raise JournalScraperError(error_msg, e, url)


def scrape_journal_metadata(journal_name: str, query: str, max_results: int = 100) -> Dict[str, Any]:
    """
    Scrape metadata from scientific journals using paperscraper and PubMed API.
    
    This function searches for articles in specified journals and returns
    comprehensive metadata including titles, authors, abstracts, DOIs, and 
    publication dates. Utilizes paperscraper's PubMed integration with automatic
    rate limiting and robust error handling.
    
    Args:
        journal_name: Name of the journal to search (e.g., "Nature", "Science", 
                     "Plant Physiology"). Both single-word and multi-word journal 
                     names are supported.
        query: Search query string (e.g., "plant metabolites", "protein folding")
        max_results: Maximum number of results to return (default: 100, max: 9998)
        
    Returns:
        Dict[str, Any]: Dictionary containing scraped metadata with structure:
            {
                "journal": str,           # Original journal name
                "query": str,            # Original search query
                "total_results": int,    # Number of articles found
                "articles": List[Dict[str, Any]]  # Article metadata
            }
            
        Each article contains:
            - title: Article title
            - authors: List of author names
            - abstract: Article abstract (if available)
            - journal: Journal name from PubMed
            - publication_date: Publication date
            - doi: Digital Object Identifier
            - url: DOI URL for accessing the article
            - source: "PubMed" (data source identifier)
            - query_matched: Original query that matched this article
        
    Raises:
        JournalScraperError: If metadata scraping fails for any reason:
            - Network connection errors or PubMed API failures
            - Invalid journal name or query parameters
            - Rate limiting from PubMed API
            - paperscraper library not available
            
    Example:
        >>> # Search for plant metabolomics papers in Nature
        >>> metadata = scrape_journal_metadata("Nature", "plant metabolites", 50)
        >>> print(f"Found {metadata['total_results']} articles")
        >>> for article in metadata['articles'][:3]:
        ...     print(f"- {article['title']} ({article['publication_date']})")
        
        >>> # Search in multi-word journal name
        >>> metadata = scrape_journal_metadata("Plant Physiology", "auxin transport")
        >>> print(f"Found {len(metadata['articles'])} articles in Plant Physiology")
    """
    logger.info(f"Scraping metadata from journal '{journal_name}' with query: '{query}' (max_results: {max_results})")
    
    # Validate inputs
    if not journal_name or not isinstance(journal_name, str):
        raise JournalScraperError("Journal name must be a non-empty string")
    
    if not query or not isinstance(query, str):
        raise JournalScraperError("Query must be a non-empty string")
    
    journal_name = journal_name.strip()
    query = query.strip()
    
    if not journal_name:
        raise JournalScraperError("Journal name cannot be empty or whitespace")
    
    if not query:
        raise JournalScraperError("Query cannot be empty or whitespace")
    
    if max_results <= 0:
        raise JournalScraperError("max_results must be positive")
    
    # Apply rate limiting
    rate_limiter = get_rate_limiter()
    rate_limiter.acquire()
    
    try:
        # Import paperscraper at runtime to handle potential import issues
        # paperscraper provides access to PubMed, arXiv, bioRxiv, medRxiv, and chemRxiv
        # We use the PubMed backend for journal-specific searches
        try:
            import paperscraper.pubmed as pubmed
            import pandas as pd
        except ImportError as import_error:
            raise JournalScraperError(
                "paperscraper is not available. Please install it with: pip install paperscraper",
                import_error
            )
        
        logger.debug(f"Searching paperscraper/PubMed for journal: {journal_name}")
        
        # Construct PubMed query with journal filter
        # Format: 'query terms AND "Journal Name"[Journal]'
        if journal_name.lower() in ['nature', 'science', 'cell']:
            # Handle common single-word journals
            pubmed_query = f'{query} AND {journal_name}[Journal]'
        else:
            # Handle multi-word journal names (need quotes)
            pubmed_query = f'{query} AND "{journal_name}"[Journal]'
        
        logger.debug(f"PubMed query: {pubmed_query}")
        
        # Define fields to retrieve
        fields = ['title', 'authors', 'abstract', 'journal', 'date', 'doi']
        
        # Perform the search using paperscraper with error handling
        try:
            df = pubmed.get_pubmed_papers(
                query=pubmed_query,
                fields=fields,
                max_results=max_results
            )
        except Exception as pubmed_error:
            # Handle specific paperscraper/PubMed API errors
            error_types = [
                "network", "connection", "timeout", "ssl", "certificate"
            ]
            error_msg_lower = str(pubmed_error).lower()
            
            if any(error_type in error_msg_lower for error_type in error_types):
                raise JournalScraperError(
                    f"Network error while accessing PubMed API: {str(pubmed_error)}",
                    pubmed_error
                )
            elif "query" in error_msg_lower or "search" in error_msg_lower:
                raise JournalScraperError(
                    f"Invalid search query for PubMed: '{pubmed_query}'. Error: {str(pubmed_error)}",
                    pubmed_error
                )
            elif "rate" in error_msg_lower or "limit" in error_msg_lower or "throttle" in error_msg_lower:
                logger.warning(f"PubMed API rate limiting detected, applying additional delay")
                time.sleep(5)  # Additional delay for rate limiting
                raise JournalScraperError(
                    f"PubMed API rate limiting detected: {str(pubmed_error)}",
                    pubmed_error
                )
            else:
                raise JournalScraperError(
                    f"PubMed API error: {str(pubmed_error)}",
                    pubmed_error
                )
        
        # Convert DataFrame to list of dictionaries
        articles = []
        if not df.empty:
            logger.debug(f"Processing {len(df)} articles from PubMed response")
            for idx, row in df.iterrows():
                # Handle potential missing or malformed data
                authors = row.get("authors", [])
                if isinstance(authors, str):
                    # Sometimes authors come as a single string, convert to list
                    authors = [authors] if authors else []
                elif not isinstance(authors, list):
                    authors = []
                
                # Clean and validate DOI
                doi = row.get("doi", "").strip()
                doi_url = ""
                if doi:
                    # Remove common DOI prefixes if they exist
                    if doi.startswith("doi:"):
                        doi = doi[4:]
                    if doi.startswith("https://doi.org/"):
                        doi_url = doi
                        doi = doi.replace("https://doi.org/", "")
                    else:
                        doi_url = f"https://doi.org/{doi}"
                
                # Clean title and abstract
                title = str(row.get("title", "")).strip()
                abstract = str(row.get("abstract", "")).strip()
                
                article = {
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "journal": str(row.get("journal", "")).strip(),
                    "publication_date": str(row.get("date", "")).strip(),
                    "doi": doi,
                    "url": doi_url,
                    "source": "PubMed",  # Add source identifier
                    "query_matched": query  # Track which query this result matched
                }
                articles.append(article)
        else:
            logger.info(f"No articles found for query: '{pubmed_query}'")
        
        results = {
            "journal": journal_name,
            "query": query,
            "total_results": len(articles),
            "articles": articles
        }
        
        logger.info(f"Successfully scraped metadata: {results['total_results']} articles found from {journal_name}")
        return results
        
    except Exception as e:
        error_msg = f"Error scraping metadata from journal '{journal_name}' with query '{query}'"
        logger.error(error_msg)
        raise JournalScraperError(error_msg, e)


def download_journal_fulltext(article_url: str, output_path: str, 
                            check_robots: bool = True,
                            use_paperscraper: bool = True) -> bool:
    """
    Download full-text content from journal articles.
    
    This function downloads PDF or XML content from journal article URLs,
    with automatic robots.txt checking, rate limiting, and bot protection.
    
    Args:
        article_url: URL of the article to download
        output_path: Local path where the content should be saved
        check_robots: Whether to check robots.txt before downloading
        use_paperscraper: Whether to use paperscraper for download (fallback to requests)
        
    Returns:
        bool: True if download successful, False otherwise
        
    Raises:
        JournalScraperError: If download fails for any reason:
            - Network connection errors
            - Invalid URL or file path
            - Robots.txt violations
            - File system errors
            
    Example:
        >>> success = download_journal_fulltext(
        ...     "https://example.com/article.pdf",
        ...     "/path/to/output.pdf"
        ... )
        >>> if success:
        ...     print("Download completed successfully")
    """
    logger.info(f"Downloading full-text from: {article_url}")
    
    # Validate inputs
    if not article_url or not isinstance(article_url, str):
        raise JournalScraperError("Article URL must be a non-empty string")
    
    if not output_path or not isinstance(output_path, str):
        raise JournalScraperError("Output path must be a non-empty string")
    
    article_url = article_url.strip()
    output_path = output_path.strip()
    
    if not article_url:
        raise JournalScraperError("Article URL cannot be empty or whitespace")
    
    if not output_path:
        raise JournalScraperError("Output path cannot be empty or whitespace")
    
    # Validate URL format
    try:
        parsed_url = urlparse(article_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise JournalScraperError(f"Invalid URL format: {article_url}")
        
        # Ensure we have http/https scheme
        if parsed_url.scheme.lower() not in ['http', 'https']:
            raise JournalScraperError(f"Unsupported URL scheme: {parsed_url.scheme}")
            
    except Exception as e:
        raise JournalScraperError(f"Error parsing URL: {article_url}", e, article_url)
    
    # Check robots.txt if requested
    if check_robots:
        user_agent_rotator = UserAgentRotator()
        user_agent = user_agent_rotator.get_random_user_agent()
        
        if not check_robots_txt(article_url, user_agent):
            raise JournalScraperError(f"Download not allowed by robots.txt: {article_url}", url=article_url)
    
    # Create output directory if it doesn't exist
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    except PermissionError as e:
        raise JournalScraperError(f"Permission denied creating output directory: {output_dir}", e)
    except OSError as e:
        raise JournalScraperError(f"Error creating output directory: {output_dir}", e)
    
    # Apply rate limiting
    rate_limiter = get_rate_limiter()
    rate_limiter.acquire()
    
    try:
        if use_paperscraper:
            # Try paperscraper first
            logger.debug(f"Attempting download with paperscraper: {article_url}")
            success = _download_with_paperscraper(article_url, output_path)
        else:
            success = False
        
        if not success:
            # Fallback to direct requests
            logger.debug(f"Attempting download with requests: {article_url}")
            success = _download_with_requests(article_url, output_path)
        
        if success:
            logger.info(f"Successfully downloaded full-text to: {output_path}")
            # Verify the downloaded file exists and has reasonable size
            try:
                file_size = os.path.getsize(output_path)
                if file_size == 0:
                    logger.warning(f"Downloaded file is empty: {output_path}")
                    return False
                elif file_size < 1024:  # Less than 1KB might indicate an error page
                    logger.warning(f"Downloaded file is suspiciously small ({file_size} bytes): {output_path}")
                else:
                    logger.debug(f"Downloaded file size: {file_size} bytes")
            except OSError:
                # File might not exist or be accessible
                logger.warning(f"Could not verify downloaded file: {output_path}")
                return False
        else:
            logger.warning(f"Failed to download full-text from: {article_url}")
        
        return success
        
    except JournalScraperError:
        # Re-raise our custom exceptions without wrapping
        raise
    except Exception as e:
        error_msg = f"Unexpected error downloading full-text from: {article_url}"
        logger.error(f"{error_msg}: {e}")
        raise JournalScraperError(error_msg, e, article_url)


def _download_with_paperscraper(url: str, output_path: str) -> bool:
    """
    Download content using paperscraper with DOI extraction and fallback mechanisms.
    
    Args:
        url: URL to download from (can be DOI URL or article URL)
        output_path: Path to save the downloaded content
        
    Returns:
        bool: True if download successful, False otherwise
        
    Raises:
        JournalScraperError: If critical errors occur during download
    """
    try:
        # Import paperscraper components
        from paperscraper.pdf import save_pdf
        import re
        from pathlib import Path
        
        logger.debug(f"Processing URL for paperscraper: {url}")
        
        # Extract DOI from URL or validate if it's already a DOI
        doi = _extract_doi_from_url(url)
        if not doi:
            logger.debug(f"Could not extract DOI from URL: {url}")
            return False
        
        logger.debug(f"Extracted DOI: {doi}")
        
        # Prepare metadata dictionary for paperscraper
        paper_metadata = {"doi": doi}
        
        # Ensure output directory exists
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Use paperscraper to download the paper
        try:
            save_pdf(
                paper_metadata=paper_metadata,
                filepath=str(output_path_obj.with_suffix('.pdf')),
                save_metadata=False,  # We don't need metadata for this use case
                api_keys=None  # Could be enhanced to load API keys from config
            )
            
            # Check if the file was actually created and has content
            # paperscraper might save as PDF, XML, or other formats
            possible_files = [
                output_path_obj.with_suffix('.pdf'),
                output_path_obj.with_suffix('.xml'),
                output_path_obj  # Original path
            ]
            
            for output_file in possible_files:
                if output_file.exists() and output_file.stat().st_size > 0:
                    logger.debug(f"Successfully downloaded content via paperscraper: {output_file}")
                    # If paperscraper saved with a different extension, rename to match expected output
                    if output_file != Path(output_path):
                        import shutil
                        shutil.move(str(output_file), output_path)
                        logger.debug(f"Moved {output_file} to {output_path}")
                    return True
            
            logger.debug(f"Paperscraper download failed - no valid file created")
            return False
                
        except Exception as e:
            # Log paperscraper-specific errors but don't raise - let fallback handle it
            error_msg = str(e).lower()
            if "doi" in error_msg:
                logger.debug(f"Paperscraper failed - DOI issue: {e}")
            elif "network" in error_msg or "connection" in error_msg:
                logger.debug(f"Paperscraper failed - network issue: {e}")
            elif "access" in error_msg or "permission" in error_msg:
                logger.debug(f"Paperscraper failed - access denied: {e}")
            else:
                logger.debug(f"Paperscraper failed - general error: {e}")
            return False
            
    except ImportError as e:
        logger.debug(f"Paperscraper not available: {e}")
        return False
    except Exception as e:
        logger.debug(f"Unexpected error in paperscraper download: {e}")
        return False


def _extract_doi_from_url(url: str) -> Optional[str]:
    """
    Extract DOI from various URL formats.
    
    Args:
        url: URL that may contain a DOI
        
    Returns:
        str or None: Extracted DOI if found, None otherwise
    """
    import re
    
    # Common DOI patterns
    doi_patterns = [
        # Standard DOI URL: https://doi.org/10.1000/182
        r'https?://(?:dx\.)?doi\.org/(.+)',
        # DOI in URL path: https://example.com/doi/10.1000/182
        r'/doi/(.+?)(?:\?|#|$)',
        # Direct DOI format: doi:10.1000/182
        r'doi:(.+?)(?:\?|#|$)',
        # DOI in query parameters: ?doi=10.1000/182
        r'[?&]doi=([^&]+)',
        # Nature-style URLs: Convert Nature article IDs to DOIs
        # Pattern for s41586-XXX-XXXXX-X format -> 10.1038/s41586-XXX-XXXXX-X
        r'nature\.com/articles/(s\d+-\d+-\d+-\d+)',
        # Science-style URLs: https://www.science.org/doi/10.1126/science.abc1234
        r'science\.org/doi/(.+?)(?:\?|#|$)',
        # PLoS ONE style: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0123456
        r'plos\.org/.+?id=(.+?)(?:&|#|$)',
        # Generic DOI pattern in URL: 10.1000/182
        r'(10\.\d{4,}/[^\s\?#]+)'
    ]
    
    for pattern in doi_patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            doi = match.group(1)
            # Clean up the DOI
            doi = doi.strip()
            # Remove trailing punctuation and URL fragments
            doi = re.sub(r'[.,;:)\]}>]*$', '', doi)
            
            # Handle special cases for Nature article IDs
            if doi.startswith('s'):
                # Convert Nature article ID to DOI format
                # s41586-XXX-XXXXX-X -> 10.1038/s41586-XXX-XXXXX-X
                if re.match(r'^s\d+-\d+-\d+-\d+$', doi):
                    doi = f'10.1038/{doi}'
            
            # Validate DOI format (basic check)
            if re.match(r'^10\.\d{4,}/[^\s]+', doi):
                return doi
    
    return None


def _download_with_requests(url: str, output_path: str) -> bool:
    """
    Download content using requests with proper headers and error handling.
    
    Args:
        url: URL to download from
        output_path: Path to save the downloaded content
        
    Returns:
        bool: True if download successful, False otherwise
        
    Raises:
        JournalScraperError: If download fails
    """
    user_agent_rotator = UserAgentRotator()
    session = create_session_with_retries()
    
    headers = {
        'User-Agent': user_agent_rotator.get_random_user_agent(),
        'Accept': 'application/pdf,application/xml,text/xml,text/html,application/xhtml+xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Referer': url,  # Some sites require a valid referer
    }
    
    try:
        response = session.get(
            url,
            headers=headers,
            timeout=(CONNECT_TIMEOUT, REQUEST_TIMEOUT),
            stream=True
        )
        response.raise_for_status()
        
        # Check content type and adjust output path accordingly
        content_type = response.headers.get('content-type', '').lower()
        logger.debug(f"Response content type: {content_type}")
        
        # Determine appropriate file extension based on content type
        if 'application/pdf' in content_type:
            if not output_path.lower().endswith('.pdf'):
                output_path = output_path + '.pdf'
        elif 'application/xml' in content_type or 'text/xml' in content_type:
            if not output_path.lower().endswith('.xml'):
                output_path = output_path + '.xml'
        elif 'text/html' in content_type:
            if not output_path.lower().endswith('.html'):
                output_path = output_path + '.html'
        
        # Validate that we actually got content (not just an error page)
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) < 1024:
            logger.warning(f"Response content is very small ({content_length} bytes), might be an error page")
        
        # Write content to file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Verify the downloaded file
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            logger.error(f"Downloaded file is empty: {output_path}")
            return False
        
        logger.debug(f"Successfully downloaded {file_size} bytes to {output_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for URL {url}: {e}")
        return False
    except IOError as e:
        logger.error(f"File I/O error while saving to {output_path}: {e}")
        return False


def configure_rate_limiter(requests_per_second: float, 
                         min_delay_seconds: float = DEFAULT_MIN_DELAY_SECONDS) -> None:
    """
    Configure the global rate limiter settings.
    
    Args:
        requests_per_second: Maximum requests per second allowed
        min_delay_seconds: Minimum delay between requests
        
    Raises:
        ValueError: If parameters are invalid
        JournalScraperError: If configuration fails
    """
    if requests_per_second <= 0:
        raise ValueError("requests_per_second must be positive")
    
    if min_delay_seconds < 0:
        raise ValueError("min_delay_seconds cannot be negative")
    
    try:
        global _rate_limiter
        with _rate_limiter_lock:
            _rate_limiter = RateLimiter(requests_per_second, min_delay_seconds)
        
        logger.info(f"Rate limiter configured: {requests_per_second} req/sec, {min_delay_seconds}s min delay")
        
    except Exception as e:
        raise JournalScraperError(f"Failed to configure rate limiter: {e}", e)


def get_journal_base_urls() -> Dict[str, str]:
    """
    Get a mapping of common journal names to their base URLs.
    
    Returns:
        Dict[str, str]: Dictionary mapping journal names to base URLs
    """
    return {
        "nature": "https://www.nature.com",
        "science": "https://www.science.org",
        "cell": "https://www.cell.com",
        "plos_one": "https://journals.plos.org/plosone",
        "bmc_genomics": "https://bmcgenomics.biomedcentral.com",
        "frontiers": "https://www.frontiersin.org",
        "springer": "https://link.springer.com",
        "wiley": "https://onlinelibrary.wiley.com",
        "elsevier": "https://www.sciencedirect.com",
        "oxford": "https://academic.oup.com"
    }


def validate_journal_access(journal_name: str) -> Dict[str, Any]:
    """
    Validate access to a journal and check its scraping policies.
    
    Args:
        journal_name: Name of the journal to validate
        
    Returns:
        Dict[str, Any]: Validation results including robots.txt status and accessibility
        
    Raises:
        JournalScraperError: If validation fails
    """
    logger.info(f"Validating access to journal: {journal_name}")
    
    if not journal_name or not isinstance(journal_name, str):
        raise JournalScraperError("Journal name must be a non-empty string")
    
    journal_name = journal_name.strip().lower()
    if not journal_name:
        raise JournalScraperError("Journal name cannot be empty or whitespace")
    
    journal_urls = get_journal_base_urls()
    
    if journal_name not in journal_urls:
        logger.warning(f"Unknown journal: {journal_name}")
        return {
            "journal": journal_name,
            "known": False,
            "accessible": False,
            "robots_allowed": False,
            "base_url": None
        }
    
    base_url = journal_urls[journal_name]
    
    try:
        # Check if base URL is accessible
        session = create_session_with_retries()
        user_agent_rotator = UserAgentRotator()
        
        headers = {'User-Agent': user_agent_rotator.get_random_user_agent()}
        response = session.head(base_url, headers=headers, timeout=REQUEST_TIMEOUT)
        accessible = response.status_code == 200
        
        # Check robots.txt
        robots_allowed = check_robots_txt(base_url)
        
        return {
            "journal": journal_name,
            "known": True,
            "accessible": accessible,
            "robots_allowed": robots_allowed,
            "base_url": base_url,
            "status_code": response.status_code
        }
        
    except Exception as e:
        error_msg = f"Error validating journal access: {journal_name}"
        logger.error(error_msg)
        raise JournalScraperError(error_msg, e)


# Module initialization
logger.info("Journal scraper module loaded successfully")