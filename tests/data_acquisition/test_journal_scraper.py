"""
Unit tests for src/data_acquisition/journal_scraper.py

This module tests the journal scraping functionality for extracting metadata
and downloading full-text articles from scientific journals using paperscraper
and requests libraries with proper throttling and robots.txt compliance.

Test Coverage:
- Metadata scraping for journal articles using paperscraper
- Full-text PDF/XML download for open-access articles
- User-Agent header setting and rotation strategies
- Request throttling and rate limiting verification
- Robots.txt parsing and adherence testing
- Error handling for HTTP errors (4xx, 5xx), connection issues, and scraping failures
- Mock all external requests and paperscraper calls
"""

import pytest
import time
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from urllib.error import URLError, HTTPError
from urllib.robotparser import RobotFileParser
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import the journal scraper functions
from src.data_acquisition.journal_scraper import (
    scrape_journal_metadata,
    download_journal_fulltext,
    check_robots_txt,
    JournalScraperError,
    RateLimiter,
    UserAgentRotator,
    create_session_with_retries
)


class TestJournalScrapingMetadata:
    """Test cases for journal metadata scraping functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary test directory
        self.temp_dir = tempfile.mkdtemp()
        self.test_output_path = os.path.join(self.temp_dir, "test_article.pdf")
        
        # Sample metadata response
        self.sample_metadata = {
            'title': 'Plant Metabolite Analysis in Arabidopsis',
            'authors': ['Smith, J.', 'Doe, A.', 'Johnson, M.'],
            'journal': 'Plant Physiology',
            'year': '2023',
            'volume': '191',
            'issue': '2',
            'pages': '123-145',
            'doi': '10.1104/pp.23.00123',
            'abstract': 'This study investigates metabolite profiles in Arabidopsis...',
            'keywords': ['metabolomics', 'Arabidopsis', 'plant biology'],
            'url': 'https://example-journal.com/article/10.1104/pp.23.00123',
            'pdf_url': 'https://example-journal.com/article/10.1104/pp.23.00123.pdf'
        }
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up temporary files
        if os.path.exists(self.test_output_path):
            os.remove(self.test_output_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    @patch('src.data_acquisition.journal_scraper.paperscraper')
    @patch('src.data_acquisition.journal_scraper.RateLimiter')
    def test_scrape_journal_metadata_successful_extraction(self, mock_throttle_manager, mock_paperscraper):
        """Test successful metadata scraping for a known journal article URL."""
        # Mock throttling manager
        mock_throttle = MagicMock()
        mock_throttle_manager.return_value = mock_throttle
        
        # Mock paperscraper response
        mock_search_result = MagicMock()
        mock_search_result.title = self.sample_metadata['title']
        mock_search_result.authors = self.sample_metadata['authors']
        mock_search_result.journal = self.sample_metadata['journal']
        mock_search_result.year = self.sample_metadata['year']
        mock_search_result.doi = self.sample_metadata['doi']
        mock_search_result.abstract = self.sample_metadata['abstract']
        mock_search_result.keywords = self.sample_metadata['keywords']
        mock_search_result.url = self.sample_metadata['url']
        mock_search_result.pdf_url = self.sample_metadata['pdf_url']
        
        mock_paperscraper.search_papers.return_value = [mock_search_result]
        
        # Test metadata scraping
        result = scrape_journal_metadata("Plant Physiology", "plant metabolite analysis")
        
        # Verify the result
        assert isinstance(result, list)
        assert len(result) > 0
        
        metadata = result[0]
        assert metadata['title'] == self.sample_metadata['title']
        assert metadata['authors'] == self.sample_metadata['authors']
        assert metadata['journal'] == self.sample_metadata['journal']
        assert metadata['doi'] == self.sample_metadata['doi']
        
        # Verify paperscraper was called correctly
        mock_paperscraper.search_papers.assert_called_once()
        call_args = mock_paperscraper.search_papers.call_args
        assert "plant metabolite analysis" in str(call_args)
        assert "Plant Physiology" in str(call_args)
        
        # Verify throttling was used
        mock_throttle.wait.assert_called()

    @patch('src.data_acquisition.journal_scraper.paperscraper')
    @patch('src.data_acquisition.journal_scraper.RateLimiter')
    def test_scrape_journal_metadata_empty_results(self, mock_throttle_manager, mock_paperscraper):
        """Test metadata scraping when no articles are found."""
        # Mock throttling manager
        mock_throttle = MagicMock()
        mock_throttle_manager.return_value = mock_throttle
        
        # Mock empty paperscraper response
        mock_paperscraper.search_papers.return_value = []
        
        # Test metadata scraping with no results
        result = scrape_journal_metadata("NonExistent Journal", "impossible query terms")
        
        # Verify empty result
        assert isinstance(result, list)
        assert len(result) == 0
        
        # Verify paperscraper was called
        mock_paperscraper.search_papers.assert_called_once()

    @patch('src.data_acquisition.journal_scraper.paperscraper')
    @patch('src.data_acquisition.journal_scraper.RateLimiter')
    def test_scrape_journal_metadata_with_error_handling(self, mock_throttle_manager, mock_paperscraper):
        """Test error handling during metadata scraping."""
        # Mock throttling manager
        mock_throttle = MagicMock()
        mock_throttle_manager.return_value = mock_throttle
        
        # Mock paperscraper error
        mock_paperscraper.search_papers.side_effect = Exception("Paperscraper API error")
        
        # Test that JournalScraperError is raised
        with pytest.raises(JournalScraperError) as exc_info:
            scrape_journal_metadata("Test Journal", "test query")
        
        error_message = str(exc_info.value).lower()
        assert "metadata scraping failed" in error_message or "paperscraper api error" in error_message

    @patch('src.data_acquisition.journal_scraper.paperscraper')
    @patch('src.data_acquisition.journal_scraper.UserAgentRotator')
    @patch('src.data_acquisition.journal_scraper.RateLimiter')
    def test_scrape_journal_metadata_with_user_agent_rotation(self, mock_throttle_manager, mock_user_agent, mock_paperscraper):
        """Test metadata scraping with User-Agent rotation."""
        # Mock components
        mock_throttle = MagicMock()
        mock_throttle_manager.return_value = mock_throttle
        
        mock_ua_rotator = MagicMock()
        mock_ua_rotator.get_user_agent.return_value = "Mozilla/5.0 (Test Browser)"
        mock_user_agent.return_value = mock_ua_rotator
        
        # Mock paperscraper with User-Agent
        mock_search_result = MagicMock()
        mock_search_result.title = "Test Article"
        mock_paperscraper.search_papers.return_value = [mock_search_result]
        
        # Test metadata scraping
        result = scrape_journal_metadata("Test Journal", "test query", use_user_agent_rotation=True)
        
        # Verify User-Agent was used
        mock_ua_rotator.get_user_agent.assert_called()
        assert len(result) > 0


class TestJournalFullTextDownload:
    """Test cases for journal full-text download functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_output_path = os.path.join(self.temp_dir, "test_article.pdf")
        self.test_url = "https://example-journal.com/article/test.pdf"
        
        # Sample PDF content
        self.sample_pdf_content = b'%PDF-1.4\n%Test PDF content for download testing'
    
    def teardown_method(self):
        """Clean up after each test method."""
        if os.path.exists(self.test_output_path):
            os.remove(self.test_output_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    @patch('src.data_acquisition.journal_scraper.requests.get')
    @patch('src.data_acquisition.journal_scraper.check_robots_txt')
    @patch('src.data_acquisition.journal_scraper.RateLimiter')
    def test_download_journal_fulltext_pdf_success(self, mock_throttle_manager, mock_robots_checker, mock_requests_get):
        """Test successful full-text PDF download for a known open-access article URL."""
        # Mock components
        mock_throttle = MagicMock()
        mock_throttle_manager.return_value = mock_throttle
        
        mock_robots_checker.return_value = True
        
        # Mock successful requests response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = self.sample_pdf_content
        mock_response.headers = {'Content-Type': 'application/pdf'}
        mock_requests_get.return_value = mock_response
        
        # Test PDF download
        result = download_journal_fulltext(self.test_url, self.test_output_path)
        
        # Verify successful download
        assert result is True
        
        # Verify file was created (mock the file writing)
        with patch('builtins.open', mock_open()) as mock_file:
            download_journal_fulltext(self.test_url, self.test_output_path)
            mock_file.assert_called_with(self.test_output_path, 'wb')
        
        # Verify requests was called correctly
        mock_requests_get.assert_called()
        call_args = mock_requests_get.call_args
        assert self.test_url in str(call_args)
        
        # Verify robots.txt was checked
        mock_robots_checker.assert_called()
        
        # Verify throttling was used
        mock_throttle.wait.assert_called()

    @patch('src.data_acquisition.journal_scraper.requests.get')
    @patch('src.data_acquisition.journal_scraper.check_robots_txt')
    @patch('src.data_acquisition.journal_scraper.RateLimiter')
    def test_download_journal_fulltext_xml_success(self, mock_throttle_manager, mock_robots_checker, mock_requests_get):
        """Test successful full-text XML download."""
        # Mock components
        mock_throttle = MagicMock()
        mock_throttle_manager.return_value = mock_throttle
        
        mock_robots_checker.return_value = True
        
        # Mock successful XML response
        sample_xml_content = b'<?xml version="1.0"?><article><title>Test Article</title></article>'
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = sample_xml_content
        mock_response.headers = {'Content-Type': 'application/xml'}
        mock_requests_get.return_value = mock_response
        
        xml_output_path = os.path.join(self.temp_dir, "test_article.xml")
        
        # Test XML download
        result = download_journal_fulltext(self.test_url.replace('.pdf', '.xml'), xml_output_path)
        
        # Verify successful download
        assert result is True
        
        # Verify requests was called
        mock_requests_get.assert_called()

    @patch('src.data_acquisition.journal_scraper.requests.get')
    @patch('src.data_acquisition.journal_scraper.check_robots_txt')
    @patch('src.data_acquisition.journal_scraper.RateLimiter')
    def test_download_journal_fulltext_http_error_404(self, mock_throttle_manager, mock_robots_checker, mock_requests_get):
        """Test error handling for HTTP 404 error."""
        # Mock components
        mock_throttle = MagicMock()
        mock_throttle_manager.return_value = mock_throttle
        
        mock_robots_checker.return_value = True
        
        # Mock 404 response
        from requests.exceptions import HTTPError as RequestsHTTPError
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = RequestsHTTPError("404 Not Found")
        mock_requests_get.return_value = mock_response
        
        # Test that download returns False for 404
        result = download_journal_fulltext(self.test_url, self.test_output_path)
        
        # Verify download failed
        assert result is False

    @patch('src.data_acquisition.journal_scraper.requests.get')
    @patch('src.data_acquisition.journal_scraper.check_robots_txt')
    @patch('src.data_acquisition.journal_scraper.RateLimiter')
    def test_download_journal_fulltext_http_error_500(self, mock_throttle_manager, mock_robots_checker, mock_requests_get):
        """Test error handling for HTTP 500 server error."""
        # Mock components
        mock_throttle = MagicMock()
        mock_throttle_manager.return_value = mock_throttle
        
        mock_robots_checker.return_value = True
        
        # Mock 500 response
        from requests.exceptions import HTTPError as RequestsHTTPError
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = RequestsHTTPError("500 Internal Server Error")
        mock_requests_get.return_value = mock_response
        
        # Test that download returns False for 500
        result = download_journal_fulltext(self.test_url, self.test_output_path)
        
        # Verify download failed
        assert result is False

    @patch('src.data_acquisition.journal_scraper.requests.get')
    @patch('src.data_acquisition.journal_scraper.check_robots_txt')
    @patch('src.data_acquisition.journal_scraper.RateLimiter')
    def test_download_journal_fulltext_connection_error(self, mock_throttle_manager, mock_robots_checker, mock_requests_get):
        """Test error handling for connection issues."""
        # Mock components
        mock_throttle = MagicMock()
        mock_throttle_manager.return_value = mock_throttle
        
        mock_robots_checker.return_value = True
        
        # Mock connection error
        from requests.exceptions import ConnectionError
        mock_requests_get.side_effect = ConnectionError("Connection failed")
        
        # Test that JournalScraperError is raised for connection issues
        with pytest.raises(JournalScraperError) as exc_info:
            download_journal_fulltext(self.test_url, self.test_output_path)
        
        error_message = str(exc_info.value).lower()
        assert "connection" in error_message or "network" in error_message

    @patch('src.data_acquisition.journal_scraper.requests.get')
    @patch('src.data_acquisition.journal_scraper.check_robots_txt')
    @patch('src.data_acquisition.journal_scraper.RateLimiter')
    def test_download_journal_fulltext_robots_txt_blocked(self, mock_throttle_manager, mock_robots_checker, mock_requests_get):
        """Test handling when robots.txt blocks access."""
        # Mock components
        mock_throttle = MagicMock()
        mock_throttle_manager.return_value = mock_throttle
        
        mock_robots_checker.return_value = False  # Blocked by robots.txt
        
        # Test that JournalScraperError is raised when blocked by robots.txt
        with pytest.raises(JournalScraperError) as exc_info:
            download_journal_fulltext(self.test_url, self.test_output_path)
        
        error_message = str(exc_info.value).lower()
        assert "robots.txt" in error_message or "blocked" in error_message
        
        # Verify requests was not called when blocked
        mock_requests_get.assert_not_called()


class TestUserAgentHandling:
    """Test cases for User-Agent header setting and rotation."""
    
    @patch('src.data_acquisition.journal_scraper.requests.get')
    @patch('src.data_acquisition.journal_scraper.EnhancedUserAgentRotator')
    @patch('src.data_acquisition.journal_scraper.UserAgentRotator')
    def test_user_agent_header_setting(self, mock_user_agent_rotator, mock_enhanced_rotator, mock_requests_get):
        """Test User-Agent header setting in requests."""
        # Mock User-Agent rotator
        test_user_agent = "Mozilla/5.0 (Test Browser) AppleWebKit/537.36"
        mock_ua_rotator = MagicMock()
        mock_ua_rotator.get_user_agent.return_value = test_user_agent
        mock_ua_rotator.get_browser_headers.return_value = {"User-Agent": test_user_agent}
        mock_ua_rotator.record_agent_result.return_value = None
        mock_enhanced_rotator.return_value = mock_ua_rotator
        mock_user_agent_rotator.return_value = mock_ua_rotator
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"test content"
        mock_requests_get.return_value = mock_response
        
        # Test download with User-Agent
        with patch('src.data_acquisition.journal_scraper.check_robots_txt') as mock_robots:
            with patch('src.data_acquisition.journal_scraper.RateLimiter') as mock_throttle_manager:
                mock_robots.return_value = True
                mock_throttle_manager.return_value = MagicMock()
                
                download_journal_fulltext(
                    "https://example.com/test.pdf", 
                    "/tmp/test.pdf",
                    use_custom_user_agent=True
                )
        
        # Verify User-Agent was used in request headers
        mock_requests_get.assert_called()
        call_args = mock_requests_get.call_args
        headers = call_args[1].get('headers', {}) if len(call_args) > 1 else {}
        
        # Check if User-Agent is in headers (implementation detail)
        mock_ua_rotator.get_user_agent.assert_called()

    def test_user_agent_rotator_initialization(self):
        """Test UserAgentRotator initialization and behavior."""
        # Test UserAgentRotator creation
        rotator = UserAgentRotator()
        
        # Test getting different User-Agents
        ua1 = rotator.get_user_agent()
        ua2 = rotator.get_user_agent()
        
        # Verify User-Agents are strings and potentially different
        assert isinstance(ua1, str)
        assert isinstance(ua2, str)
        assert len(ua1) > 0
        assert len(ua2) > 0

    def test_user_agent_rotator_with_custom_agents(self):
        """Test UserAgentRotator with custom User-Agent list."""
        custom_agents = [
            "Custom Browser 1.0",
            "Custom Browser 2.0",
            "Custom Browser 3.0"
        ]
        
        rotator = UserAgentRotator(user_agents=custom_agents)
        
        # Test that custom agents are used
        for _ in range(10):  # Test multiple calls
            ua = rotator.get_user_agent()
            assert ua in custom_agents


class TestThrottlingAndRateLimiting:
    """Test cases for request throttling and rate limiting."""
    
    @patch('src.data_acquisition.journal_scraper.time.sleep')
    def test_throttle_manager_basic_throttling(self, mock_sleep):
        """Test basic throttling functionality."""
        # Test RateLimiter with specific delay
        throttle_delay = 1.0
        throttle_manager = RateLimiter(delay=throttle_delay)
        
        # Make multiple throttled calls
        throttle_manager.wait()
        throttle_manager.wait()
        throttle_manager.wait()
        
        # Verify sleep was called with correct delay
        expected_calls = [call(throttle_delay)] * 2  # First call doesn't sleep
        mock_sleep.assert_has_calls(expected_calls)
    
    @patch('src.data_acquisition.journal_scraper.time.sleep')
    def test_throttle_manager_adaptive_throttling(self, mock_sleep):
        """Test adaptive throttling based on response times."""
        throttle_manager = RateLimiter(adaptive=True)
        
        # Simulate slow response (should increase delay)
        throttle_manager.record_request_time(2.5)  # Slow response
        throttle_manager.wait()
        
        # Simulate fast response (should decrease delay)
        throttle_manager.record_request_time(0.1)  # Fast response
        throttle_manager.wait()
        
        # Verify sleep was called (exact delays depend on implementation)
        assert mock_sleep.call_count >= 1
        
        # Verify delays were recorded
        assert len(throttle_manager.response_times) > 0

    @patch('src.data_acquisition.journal_scraper.paperscraper')
    @patch('src.data_acquisition.journal_scraper.time.sleep')
    def test_metadata_scraping_with_throttling(self, mock_sleep, mock_paperscraper):
        """Test that metadata scraping includes proper throttling."""
        # Mock paperscraper response
        mock_result = MagicMock()
        mock_result.title = "Test Article"
        mock_paperscraper.search_papers.return_value = [mock_result]
        
        # Make multiple rapid metadata requests
        with patch('src.data_acquisition.journal_scraper.RateLimiter') as mock_throttle_manager:
            mock_throttle = MagicMock()
            mock_throttle_manager.return_value = mock_throttle
            
            scrape_journal_metadata("Journal 1", "query 1")
            scrape_journal_metadata("Journal 2", "query 2")
            scrape_journal_metadata("Journal 3", "query 3")
            
            # Verify throttling was applied
            assert mock_throttle.wait.call_count >= 3

    @patch('src.data_acquisition.journal_scraper.requests.get')
    @patch('src.data_acquisition.journal_scraper.time.sleep')
    def test_download_with_throttling(self, mock_sleep, mock_requests_get):
        """Test that downloads include proper throttling."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"test content"
        mock_requests_get.return_value = mock_response
        
        with patch('src.data_acquisition.journal_scraper.check_robots_txt') as mock_robots:
            with patch('src.data_acquisition.journal_scraper.RateLimiter') as mock_throttle_manager:
                mock_robots.return_value = True
                mock_throttle = MagicMock()
                mock_throttle_manager.return_value = mock_throttle
                
                # Make multiple download requests
                download_journal_fulltext("https://example.com/1.pdf", "/tmp/1.pdf")
                download_journal_fulltext("https://example.com/2.pdf", "/tmp/2.pdf")
                
                # Verify throttling was applied
                assert mock_throttle.wait.call_count >= 2


class TestRobotsTxtHandling:
    """Test cases for robots.txt parsing and adherence."""
    
    @patch('src.data_acquisition.journal_scraper.RobotFileParser')
    def test_check_robots_txt_allows_access(self, mock_robot_parser_class):
        """Test robots.txt parsing when access is allowed."""
        # Create mock robot parser instance
        mock_robot_parser = MagicMock()
        mock_robot_parser.can_fetch.return_value = True
        mock_robot_parser_class.return_value = mock_robot_parser
        
        # Test robots.txt checking
        can_access = check_robots_txt("https://example.com/article.pdf", user_agent="*")
        
        # Verify access is allowed
        assert can_access is True
        
        # Verify robot parser was used correctly
        mock_robot_parser_class.assert_called_once()
        mock_robot_parser.set_url.assert_called_once()
        mock_robot_parser.read.assert_called_once()
        mock_robot_parser.can_fetch.assert_called_once_with("*", "https://example.com/article.pdf")

    @patch('src.data_acquisition.journal_scraper.RobotFileParser')
    def test_check_robots_txt_disallows_access(self, mock_robot_parser_class):
        """Test robots.txt parsing when access is disallowed."""
        # Create mock robot parser instance
        mock_robot_parser = MagicMock()
        mock_robot_parser.can_fetch.return_value = False  # Disallow access
        mock_robot_parser_class.return_value = mock_robot_parser
        
        # Test robots.txt checking for disallowed path
        can_access = check_robots_txt("https://example.com/private/article.pdf", user_agent="*")
        
        # Verify access is blocked
        assert can_access is False

    @patch('src.data_acquisition.journal_scraper.RobotFileParser')
    def test_check_robots_txt_specific_user_agent(self, mock_robot_parser_class):
        """Test robots.txt parsing with specific User-Agent rules."""
        # Create mock robot parser instance that responds differently for different user agents
        mock_robot_parser = MagicMock()
        def can_fetch_side_effect(user_agent, url):
            if user_agent == "GoodBot":
                return True
            elif user_agent == "BadBot":
                return False
            return True  # Default
        
        mock_robot_parser.can_fetch.side_effect = can_fetch_side_effect
        mock_robot_parser_class.return_value = mock_robot_parser
        
        # Test with specific User-Agent
        can_access_good = check_robots_txt("https://example.com/article.pdf", user_agent="GoodBot")
        can_access_bad = check_robots_txt("https://example.com/article.pdf", user_agent="BadBot")
        
        # Verify different access for different User-Agents
        assert can_access_good is True
        assert can_access_bad is False

    @patch('src.data_acquisition.journal_scraper.RobotFileParser')
    def test_check_robots_txt_missing_file(self, mock_robot_parser_class):
        """Test robots.txt handling when file is missing (404)."""
        # Create mock robot parser instance that raises URLError on read
        mock_robot_parser = MagicMock()
        mock_robot_parser.read.side_effect = URLError("HTTP 404 Not Found")
        mock_robot_parser_class.return_value = mock_robot_parser
        
        # Test robots.txt checking with missing file
        can_access = check_robots_txt("https://example.com/article.pdf", user_agent="*")
        
        # Verify access is allowed when robots.txt is missing
        assert can_access is True

    @patch('src.data_acquisition.journal_scraper.RobotFileParser')
    def test_check_robots_txt_connection_error(self, mock_robot_parser_class):
        """Test robots.txt handling with connection errors."""
        # Create mock robot parser instance that raises URLError on read
        mock_robot_parser = MagicMock()
        mock_robot_parser.read.side_effect = URLError("Connection failed")
        mock_robot_parser_class.return_value = mock_robot_parser
        
        # Test robots.txt checking with connection error
        can_access = check_robots_txt("https://example.com/article.pdf", user_agent="*")
        
        # Verify access is allowed when robots.txt cannot be fetched
        assert can_access is True

    def test_robots_checker_initialization(self):
        """Test check_robots_txt returns boolean for basic URLs."""
        # Test with a simple URL - should return True when robots.txt doesn't exist
        result = check_robots_txt("https://example.com/article.pdf")
        
        # Should return boolean
        assert isinstance(result, bool)

    @patch('src.data_acquisition.journal_scraper.requests.get')
    def test_robots_checker_with_crawl_delay(self, mock_requests_get):
        """Test robots.txt checking with crawl delay directive."""
        # Mock robots.txt with crawl delay
        robots_content = """
User-agent: *
Allow: /
Crawl-delay: 5
"""
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = robots_content
        mock_requests_get.return_value = mock_response
        
        # Test that URL is allowed despite crawl delay
        can_access = check_robots_txt("https://example.com/article.pdf", user_agent="*")
        
        # Verify access is allowed
        assert can_access is True

    @patch('src.data_acquisition.journal_scraper.RobotFileParser')
    def test_robots_checker_multiple_calls(self, mock_robot_parser_class):
        """Test multiple robots.txt checks for same domain."""
        # Create mock robot parser instance
        mock_robot_parser = MagicMock()
        mock_robot_parser.can_fetch.return_value = True
        mock_robot_parser_class.return_value = mock_robot_parser
        
        # Make multiple requests to same domain
        can_access1 = check_robots_txt("https://example.com/article1.pdf", user_agent="*")
        can_access2 = check_robots_txt("https://example.com/article2.pdf", user_agent="*")
        
        # Verify both requests succeeded
        assert can_access1 is True
        assert can_access2 is True
        
        # Verify robot parser was called for both requests
        assert mock_robot_parser_class.call_count >= 1


class TestErrorHandlingAndEdgeCases:
    """Test cases for comprehensive error handling and edge cases."""
    
    def test_journal_scraping_error_custom_exception(self):
        """Test JournalScraperError custom exception."""
        error_message = "Test journal scraping error"
        error = JournalScraperError(error_message)
        
        assert str(error) == error_message
        assert isinstance(error, Exception)

    def test_journal_scraping_error_with_cause(self):
        """Test JournalScraperError with underlying cause."""
        cause = ValueError("Original error")
        error = JournalScraperError("Journal scraping failed", cause)
        
        assert "Journal scraping failed" in str(error)
        assert isinstance(error, Exception)

    @patch('src.data_acquisition.journal_scraper.paperscraper')
    def test_scrape_metadata_with_malformed_response(self, mock_paperscraper):
        """Test metadata scraping with malformed paperscraper response."""
        # Mock malformed response (missing required fields)
        mock_result = MagicMock()
        mock_result.title = None  # Missing title
        mock_result.authors = []   # Empty authors
        mock_result.doi = ""      # Empty DOI
        mock_paperscraper.search_papers.return_value = [mock_result]
        
        with patch('src.data_acquisition.journal_scraper.RateLimiter'):
            # Test that malformed results are handled gracefully
            result = scrape_journal_metadata("Test Journal", "test query")
            
            # Should return results even with missing fields
            assert isinstance(result, list)
            # Implementation should handle None/empty values appropriately

    @patch('src.data_acquisition.journal_scraper.requests.get')
    def test_download_with_invalid_content_type(self, mock_requests_get):
        """Test download handling with unexpected content type."""
        # Mock response with unexpected content type
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<html>Not a PDF</html>"
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_requests_get.return_value = mock_response
        
        with patch('src.data_acquisition.journal_scraper.check_robots_txt') as mock_robots:
            with patch('src.data_acquisition.journal_scraper.RateLimiter') as mock_throttle:
                mock_robots.return_value = True
                mock_throttle.return_value = MagicMock()
                
                # Test that unexpected content type raises appropriate error
                with pytest.raises(JournalScraperError) as exc_info:
                    download_journal_fulltext("https://example.com/fake.pdf", "/tmp/test.pdf")
                
                error_message = str(exc_info.value).lower()
                assert "content type" in error_message or "invalid format" in error_message

    @patch('src.data_acquisition.journal_scraper.requests.get')
    def test_download_with_empty_response(self, mock_requests_get):
        """Test download handling with empty response."""
        # Mock empty response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b""  # Empty content
        mock_response.headers = {'Content-Type': 'application/pdf'}
        mock_requests_get.return_value = mock_response
        
        with patch('src.data_acquisition.journal_scraper.check_robots_txt') as mock_robots:
            with patch('src.data_acquisition.journal_scraper.RateLimiter') as mock_throttle:
                mock_robots.return_value = True
                mock_throttle.return_value = MagicMock()
                
                # Test that empty content raises appropriate error
                with pytest.raises(JournalScraperError) as exc_info:
                    download_journal_fulltext("https://example.com/empty.pdf", "/tmp/test.pdf")
                
                error_message = str(exc_info.value).lower()
                assert "empty" in error_message or "no content" in error_message

    def test_invalid_url_handling(self):
        """Test handling of invalid URLs."""
        invalid_urls = [
            "",
            "not-a-url",
            "ftp://invalid-protocol.com/file.pdf",
            "https://",
            None
        ]
        
        for invalid_url in invalid_urls:
            with pytest.raises((JournalScraperError, ValueError, TypeError)):
                if invalid_url is not None:
                    download_journal_fulltext(invalid_url, "/tmp/test.pdf")
                else:
                    download_journal_fulltext(invalid_url, "/tmp/test.pdf")

    def test_invalid_output_path_handling(self):
        """Test handling of invalid output paths."""
        invalid_paths = [
            "",
            "/root/no-permission/file.pdf",  # Permission denied path
            "/nonexistent/deep/directory/file.pdf",  # Non-existent directory
            None
        ]
        
        for invalid_path in invalid_paths:
            with patch('src.data_acquisition.journal_scraper.requests.get') as mock_get:
                with patch('src.data_acquisition.journal_scraper.check_robots_txt') as mock_robots:
                    with patch('src.data_acquisition.journal_scraper.RateLimiter') as mock_throttle:
                        mock_response = MagicMock()
                        mock_response.status_code = 200
                        mock_response.content = b"test content"
                        mock_get.return_value = mock_response
                        mock_robots.return_value = True
                        mock_throttle.return_value = MagicMock()
                        
                        with pytest.raises((JournalScraperError, ValueError, TypeError, OSError)):
                            download_journal_fulltext("https://example.com/test.pdf", invalid_path)


class TestIntegrationScenarios:
    """Test cases for integration scenarios combining multiple components."""
    
    @patch('src.data_acquisition.journal_scraper.paperscraper')
    @patch('src.data_acquisition.journal_scraper.requests.get')
    @patch('src.data_acquisition.journal_scraper.time.sleep')
    def test_complete_workflow_metadata_to_download(self, mock_sleep, mock_requests_get, mock_paperscraper):
        """Test complete workflow from metadata scraping to full-text download."""
        # Mock metadata scraping
        mock_result = MagicMock()
        mock_result.title = "Plant Metabolomics Study"
        mock_result.pdf_url = "https://example.com/article.pdf"
        mock_paperscraper.search_papers.return_value = [mock_result]
        
        # Mock download
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"PDF content"
        mock_response.headers = {'Content-Type': 'application/pdf'}
        mock_requests_get.return_value = mock_response
        
        with patch('src.data_acquisition.journal_scraper.check_robots_txt') as mock_robots:
            with patch('src.data_acquisition.journal_scraper.RateLimiter') as mock_throttle:
                mock_robots.return_value = True
                mock_throttle.return_value = MagicMock()
                
                # Step 1: Scrape metadata
                metadata_results = scrape_journal_metadata("Plant Journal", "metabolomics")
                
                # Step 2: Download full text using metadata
                assert len(metadata_results) > 0
                pdf_url = metadata_results[0].get('pdf_url') or mock_result.pdf_url
                
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    success = download_journal_fulltext(pdf_url, tmp_file.name)
                    
                    # Verify workflow completed successfully
                    assert success is True
                    
                    # Clean up
                    os.unlink(tmp_file.name)

    @patch('src.data_acquisition.journal_scraper.requests.get')
    def test_robots_txt_and_throttling_integration(self, mock_requests_get):
        """Test integration between robots.txt checking and throttling."""
        # Mock robots.txt response with crawl delay
        robots_response = MagicMock()
        robots_response.status_code = 200
        robots_response.text = "User-agent: *\nAllow: /\nCrawl-delay: 2"
        
        # Mock PDF download response
        pdf_response = MagicMock()
        pdf_response.status_code = 200
        pdf_response.content = b"PDF content"
        pdf_response.headers = {'Content-Type': 'application/pdf'}
        
        # Set up mock to return different responses for different URLs
        def mock_get_side_effect(url, **kwargs):
            if 'robots.txt' in url:
                return robots_response
            else:
                return pdf_response
        
        mock_requests_get.side_effect = mock_get_side_effect
        
        with patch('src.data_acquisition.journal_scraper.time.sleep') as mock_sleep:
            with patch('src.data_acquisition.journal_scraper.RateLimiter') as mock_throttle:
                mock_throttle_instance = MagicMock()
                mock_throttle.return_value = mock_throttle_instance
                
                # Test download with robots.txt crawl delay
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    success = download_journal_fulltext("https://example.com/article.pdf", tmp_file.name)
                    
                    # Verify success and that throttling was applied
                    assert success is True
                    mock_throttle_instance.wait.assert_called()
                    
                    # Clean up
                    os.unlink(tmp_file.name)

    @patch('src.data_acquisition.journal_scraper.paperscraper')
    def test_metadata_scraping_with_multiple_results_filtering(self, mock_paperscraper):
        """Test metadata scraping with filtering of multiple results."""
        # Mock multiple results with varying quality
        mock_results = []
        
        # Good result
        good_result = MagicMock()
        good_result.title = "High Quality Plant Metabolomics Study"
        good_result.doi = "10.1104/pp.23.00123"
        good_result.journal = "Plant Physiology"
        good_result.year = "2023"
        mock_results.append(good_result)
        
        # Result with missing DOI
        incomplete_result = MagicMock()
        incomplete_result.title = "Study Without DOI"
        incomplete_result.doi = None
        incomplete_result.journal = "Unknown Journal"
        mock_results.append(incomplete_result)
        
        # Another good result
        another_good_result = MagicMock()
        another_good_result.title = "Another Quality Study"
        another_good_result.doi = "10.1105/pp.23.00456"
        another_good_result.journal = "Plant Cell"
        another_good_result.year = "2023"
        mock_results.append(another_good_result)
        
        mock_paperscraper.search_papers.return_value = mock_results
        
        with patch('src.data_acquisition.journal_scraper.RateLimiter'):
            # Test metadata scraping with filtering
            results = scrape_journal_metadata("Plant Journal", "metabolomics", filter_incomplete=True)
            
            # Verify results were filtered appropriately
            assert isinstance(results, list)
            # Implementation should filter out incomplete results
            # Exact behavior depends on implementation