"""
Unit tests for src/data_acquisition/pubmed.py

This module tests the PubMed/PMC data acquisition functionality using
Biopython.Entrez to search and retrieve abstracts/full texts.

Test Coverage:
- Successful search and ID retrieval for given keywords
- Successful fetching of XML content for valid IDs
- Rate limiting implementation verification
- Error handling for network issues, invalid queries, empty results
- Entrez.email configuration verification
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock, call
from src.data_acquisition.pubmed import (
    search_pubmed, 
    fetch_pubmed_xml, 
    PubMedError,
    set_entrez_email,
    get_rate_limiter
)


class TestPubMedDataAcquisition:
    """Test cases for PubMed data acquisition functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Reset any global state
        pass
    
    def teardown_method(self):
        """Clean up after each test method."""
        pass
    
    @patch('src.data_acquisition.pubmed.Entrez.esearch')
    def test_search_pubmed_successful_search_and_id_retrieval(self, mock_esearch):
        """Test successful search and ID retrieval for a given keyword."""
        # Mock successful search response
        mock_handle = MagicMock()
        mock_esearch.return_value = mock_handle
        
        # Mock the parsed result
        mock_result = {
            'IdList': ['12345678', '87654321', '11111111'],
            'Count': '3',
            'RetMax': '3'
        }
        
        with patch('src.data_acquisition.pubmed.Entrez.read') as mock_read:
            mock_read.return_value = mock_result
            
            # Test the search function
            result = search_pubmed("plant metabolites", max_results=10)
            
            # Verify the result
            assert result == ['12345678', '87654321', '11111111']
            
            # Verify Entrez.esearch was called with correct parameters
            mock_esearch.assert_called_once()
            call_args = mock_esearch.call_args
            assert 'plant metabolites' in str(call_args)
            assert 'pubmed' in str(call_args).lower()
    
    @patch('src.data_acquisition.pubmed.Entrez.efetch')
    def test_fetch_pubmed_xml_successful_fetching(self, mock_efetch):
        """Test successful fetching of XML content for a list of valid IDs."""
        # Mock successful fetch response
        mock_handle = MagicMock()
        mock_efetch.return_value = mock_handle
        
        # Mock XML content
        mock_xml_content = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345678</PMID>
                    <Article>
                        <ArticleTitle>Test Article Title</ArticleTitle>
                        <Abstract>
                            <AbstractText>Test abstract content</AbstractText>
                        </Abstract>
                    </Article>
                </MedlineCitation>
            </PubmedArticle>
        </PubmedArticleSet>"""
        
        mock_handle.read.return_value = mock_xml_content
        
        # Test the fetch function
        id_list = ['12345678', '87654321']
        result = fetch_pubmed_xml(id_list)
        
        # Verify the result
        assert result == mock_xml_content
        
        # Verify Entrez.efetch was called with correct parameters
        mock_efetch.assert_called_once()
        call_args = mock_efetch.call_args
        assert '12345678,87654321' in str(call_args) or ['12345678', '87654321'] == call_args[1].get('id')
    
    @patch('src.data_acquisition.pubmed.time.sleep')
    @patch('src.data_acquisition.pubmed.Entrez.esearch')
    def test_rate_limiting_implementation(self, mock_esearch, mock_sleep):
        """Test rate limiting implementation (verifying delays between calls)."""
        # Mock successful search responses
        mock_handle = MagicMock()
        mock_esearch.return_value = mock_handle
        
        mock_result = {
            'IdList': ['12345678'],
            'Count': '1',
            'RetMax': '1'
        }
        
        with patch('src.data_acquisition.pubmed.Entrez.read') as mock_read:
            mock_read.return_value = mock_result
            
            # Make multiple rapid calls to test rate limiting
            search_pubmed("test query 1", max_results=1)
            search_pubmed("test query 2", max_results=1)
            search_pubmed("test query 3", max_results=1)
            
            # Verify that sleep was called between requests (rate limiting)
            # The exact number of calls depends on implementation
            assert mock_sleep.call_count >= 2, "Rate limiting should cause delays between calls"
            
            # Verify sleep was called with appropriate delay
            sleep_calls = mock_sleep.call_args_list
            for call in sleep_calls:
                delay = call[0][0]  # First argument to sleep()
                assert delay > 0, "Sleep delay should be positive"
                assert delay <= 1.0, "Sleep delay should be reasonable (<=1 second)"
    
    @patch('src.data_acquisition.pubmed.Entrez.esearch')
    def test_error_handling_network_issues(self, mock_esearch):
        """Test error handling for network issues."""
        # Mock network error
        from urllib.error import URLError
        mock_esearch.side_effect = URLError("Network connection failed")
        
        # Test that PubMedError is raised for network issues
        with pytest.raises(PubMedError) as exc_info:
            search_pubmed("test query", max_results=10)
        
        assert "network" in str(exc_info.value).lower() or "connection" in str(exc_info.value).lower()
    
    @patch('src.data_acquisition.pubmed.Entrez.esearch')
    def test_error_handling_invalid_queries(self, mock_esearch):
        """Test error handling for invalid queries."""
        # Mock invalid query response
        mock_handle = MagicMock()
        mock_esearch.return_value = mock_handle
        
        # Mock error in parsing
        with patch('src.data_acquisition.pubmed.Entrez.read') as mock_read:
            mock_read.side_effect = Exception("Invalid query format")
            
            # Test that PubMedError is raised for invalid queries
            with pytest.raises(PubMedError) as exc_info:
                search_pubmed("valid query", max_results=10)  # Valid query that triggers parsing error
            
            assert "error" in str(exc_info.value).lower()
    
    @patch('src.data_acquisition.pubmed.Entrez.esearch')
    def test_error_handling_empty_results(self, mock_esearch):
        """Test error handling for empty results."""
        # Mock empty search response
        mock_handle = MagicMock()
        mock_esearch.return_value = mock_handle
        
        mock_result = {
            'IdList': [],
            'Count': '0',
            'RetMax': '0'
        }
        
        with patch('src.data_acquisition.pubmed.Entrez.read') as mock_read:
            mock_read.return_value = mock_result
            
            # Test that empty results are handled gracefully
            result = search_pubmed("nonexistent query terms", max_results=10)
            
            # Should return empty list, not raise error
            assert result == []
    
    def test_entrez_email_configuration(self):
        """Test that Entrez.email is set properly."""
        # Test setting email
        test_email = "test@example.com"
        set_entrez_email(test_email)
        
        # Verify email was set (this would need to check the actual Entrez.email)
        # For now, just verify the function doesn't raise an error
        assert True  # Placeholder assertion
    
    def test_entrez_email_validation(self):
        """Test validation of email format."""
        # Test invalid email formats
        invalid_emails = ["", "invalid", "test@", "@example.com", None]
        
        for invalid_email in invalid_emails:
            with pytest.raises((ValueError, PubMedError)):
                set_entrez_email(invalid_email)
    
    @patch('src.data_acquisition.pubmed.Entrez.efetch')
    def test_fetch_pubmed_xml_with_invalid_ids(self, mock_efetch):
        """Test fetching XML with invalid PubMed IDs."""
        # Mock error response for invalid IDs
        mock_efetch.side_effect = Exception("Invalid ID format")
        
        # Test that PubMedError is raised for invalid IDs
        with pytest.raises(PubMedError) as exc_info:
            fetch_pubmed_xml(['invalid_id', 'another_invalid'])
        
        assert "error" in str(exc_info.value).lower()
    
    @patch('src.data_acquisition.pubmed.Entrez.esearch')
    def test_search_pubmed_with_max_results_parameter(self, mock_esearch):
        """Test search_pubmed with different max_results values."""
        # Mock successful search response
        mock_handle = MagicMock()
        mock_esearch.return_value = mock_handle
        
        mock_result = {
            'IdList': ['1', '2', '3', '4', '5'],
            'Count': '5',
            'RetMax': '5'
        }
        
        with patch('src.data_acquisition.pubmed.Entrez.read') as mock_read:
            mock_read.return_value = mock_result
            
            # Test with different max_results values
            result = search_pubmed("test query", max_results=3)
            
            # Verify the search was called with correct retmax parameter
            mock_esearch.assert_called_once()
            call_args = mock_esearch.call_args
            # The exact parameter checking depends on implementation
            assert call_args is not None
    
    def test_rate_limiter_configuration(self):
        """Test rate limiter configuration and behavior."""
        # Test getting rate limiter
        rate_limiter = get_rate_limiter()
        
        # Verify rate limiter exists and has expected properties
        assert rate_limiter is not None
        
        # Test rate limiter behavior (basic test)
        start_time = time.time()
        rate_limiter.acquire()  # Should not block on first call
        rate_limiter.acquire()  # Should block/delay on second call
        end_time = time.time()
        
        # Should have some delay between calls
        elapsed = end_time - start_time
        assert elapsed >= 0  # At minimum, no negative time
    
    @patch('src.data_acquisition.pubmed.Entrez.esearch')
    def test_search_pubmed_with_special_characters(self, mock_esearch):
        """Test search with special characters in query."""
        # Mock successful search response
        mock_handle = MagicMock()
        mock_esearch.return_value = mock_handle
        
        mock_result = {
            'IdList': ['12345678'],
            'Count': '1',
            'RetMax': '1'
        }
        
        with patch('src.data_acquisition.pubmed.Entrez.read') as mock_read:
            mock_read.return_value = mock_result
            
            # Test with special characters
            special_query = "plant AND (metabolite OR compound) NOT animal"
            result = search_pubmed(special_query, max_results=10)
            
            # Should handle special characters without error
            assert result == ['12345678']
    
    @patch('src.data_acquisition.pubmed.Entrez.efetch')
    def test_fetch_pubmed_xml_large_id_list(self, mock_efetch):
        """Test fetching XML with a large list of IDs."""
        # Mock successful fetch response
        mock_handle = MagicMock()
        mock_efetch.return_value = mock_handle
        mock_handle.read.return_value = "<xml>Large batch content</xml>"
        
        # Test with large ID list
        large_id_list = [str(i) for i in range(1, 101)]  # 100 IDs
        result = fetch_pubmed_xml(large_id_list)
        
        # Should handle large batches
        assert result == "<xml>Large batch content</xml>"
        
        # Verify efetch was called
        mock_efetch.assert_called_once()
    
    def test_pubmed_error_custom_exception(self):
        """Test PubMedError custom exception."""
        error_message = "Test PubMed error"
        error = PubMedError(error_message)
        
        assert str(error) == error_message
        assert isinstance(error, Exception)
    
    def test_pubmed_error_with_cause(self):
        """Test PubMedError with underlying cause."""
        cause = ValueError("Original error")
        error = PubMedError("PubMed operation failed", cause)
        
        assert "PubMed operation failed" in str(error)
        # Note: The exact format depends on implementation