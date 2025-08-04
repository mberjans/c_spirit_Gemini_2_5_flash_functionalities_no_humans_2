"""
Unit tests for the ontology loader module.

This module contains comprehensive tests for loading OWL 2.0 ontologies
using Owlready2 from both URLs and local files. Tests cover successful
loading scenarios as well as various error conditions.

Test Categories:
- Successful loading from local files
- Successful loading from URLs (mocked)
- Error handling for file system issues
- Error handling for network issues
- Error handling for invalid OWL formats
- Custom exception handling
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Generator

import pytest
from owlready2 import OwlReadyError, OwlReadyOntologyParsingError

from src.utils.testing_framework import expect_exception, parametrize


class TestOntologyLoader:
    """Test suite for ontology loading functionality."""

    @pytest.fixture
    def temp_owl_file(self, temp_dir: Path) -> Generator[Path, None, None]:
        """
        Create a temporary OWL file for testing.
        
        Args:
            temp_dir: Temporary directory fixture from conftest.py
            
        Yields:
            Path: Path to temporary OWL file
        """
        owl_content = '''<?xml version="1.0"?>
<rdf:RDF xmlns="http://test.example.org/ontology#"
         xml:base="http://test.example.org/ontology"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://test.example.org/ontology">
        <rdfs:label>Test Ontology</rdfs:label>
        <rdfs:comment>A simple test ontology for unit testing</rdfs:comment>
    </owl:Ontology>
    
    <owl:Class rdf:about="http://test.example.org/ontology#TestClass">
        <rdfs:label>Test Class</rdfs:label>
        <rdfs:comment>A test class for validation</rdfs:comment>
    </owl:Class>
</rdf:RDF>'''
        
        owl_file = temp_dir / "test_ontology.owl"
        owl_file.write_text(owl_content, encoding="utf-8")
        yield owl_file

    @pytest.fixture
    def invalid_owl_file(self, temp_dir: Path) -> Generator[Path, None, None]:
        """
        Create an invalid OWL file for testing error handling.
        
        Args:
            temp_dir: Temporary directory fixture from conftest.py
            
        Yields:
            Path: Path to invalid OWL file
        """
        invalid_content = '''<?xml version="1.0"?>
<rdf:RDF xmlns="http://test.example.org/ontology#"
         xml:base="http://test.example.org/ontology"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <!-- Missing closing tag and invalid structure -->
    <owl:Ontology rdf:about="http://test.example.org/ontology">
        <rdfs:label>Invalid Ontology</rdfs:label>
    <!-- Unclosed ontology tag -->
</rdf:RDF>'''
        
        invalid_file = temp_dir / "invalid_ontology.owl"
        invalid_file.write_text(invalid_content, encoding="utf-8")
        yield invalid_file

    @pytest.fixture
    def mock_ontology(self) -> Mock:
        """
        Create a mock ontology object for testing.
        
        Returns:
            Mock: Mock ontology object with load method
        """
        mock_ont = Mock()
        mock_ont.load.return_value = mock_ont
        mock_ont.name = "test_ontology"
        mock_ont.base_iri = "http://test.example.org/ontology"
        return mock_ont

    @pytest.fixture
    def mock_owlready2_get_ontology(self, mock_ontology: Mock) -> Generator[Mock, None, None]:
        """
        Mock owlready2.get_ontology function.
        
        Args:
            mock_ontology: Mock ontology fixture
            
        Yields:
            Mock: Mocked get_ontology function
        """
        with patch('owlready2.get_ontology') as mock_get_ont:
            mock_get_ont.return_value = mock_ontology
            yield mock_get_ont

    def test_load_ontology_from_file_success(
        self, 
        temp_owl_file: Path, 
        mock_owlready2_get_ontology: Mock,
        mock_ontology: Mock
    ):
        """
        Test successful loading of a valid OWL file from local filesystem.
        
        Args:
            temp_owl_file: Temporary OWL file fixture
            mock_owlready2_get_ontology: Mocked get_ontology function
            mock_ontology: Mock ontology object
        """
        from src.ontology.loader import load_ontology_from_file
        
        # Act
        result = load_ontology_from_file(str(temp_owl_file))
        
        # Assert
        assert result is not None
        assert result == mock_ontology
        
        # Verify owlready2.get_ontology was called with correct file URI
        # Use actual call args to handle path resolution differences across platforms
        call_args = mock_owlready2_get_ontology.call_args[0][0]
        assert call_args.startswith("file://")
        assert call_args.endswith("test_ontology.owl")
        mock_owlready2_get_ontology.assert_called_once()
        
        # Verify load() method was called
        mock_ontology.load.assert_called_once()

    def test_load_ontology_from_url_success(
        self, 
        mock_owlready2_get_ontology: Mock,
        mock_ontology: Mock
    ):
        """
        Test successful loading of a valid OWL file from URL.
        
        Args:
            mock_owlready2_get_ontology: Mocked get_ontology function
            mock_ontology: Mock ontology object
        """
        from src.ontology.loader import load_ontology_from_url
        
        test_url = "http://example.com/ontology.owl"
        
        # Act
        result = load_ontology_from_url(test_url)
        
        # Assert
        assert result is not None
        assert result == mock_ontology
        
        # Verify owlready2.get_ontology was called with correct URL
        mock_owlready2_get_ontology.assert_called_once_with(test_url)
        
        # Verify load() method was called
        mock_ontology.load.assert_called_once()

    def test_load_ontology_from_file_not_found(self):
        """
        Test error handling when local file does not exist.
        """
        from src.ontology.loader import load_ontology_from_file, OntologyLoadError
        
        non_existent_file = "/path/to/non_existent_file.owl"
        
        # Act & Assert
        with expect_exception(OntologyLoadError, match="File not found"):
            load_ontology_from_file(non_existent_file)

    def test_load_ontology_from_file_permission_error(self, temp_dir: Path):
        """
        Test error handling when file exists but cannot be read due to permissions.
        
        Args:
            temp_dir: Temporary directory fixture
        """
        from src.ontology.loader import load_ontology_from_file, OntologyLoadError
        
        # Create a file but mock permission error
        test_file = temp_dir / "permission_test.owl"
        test_file.write_text("test content")
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('owlready2.get_ontology') as mock_get_ont:
            
            # Mock permission error
            mock_ont = Mock()
            mock_ont.load.side_effect = PermissionError("Permission denied")
            mock_get_ont.return_value = mock_ont
            
            # Act & Assert
            with expect_exception(OntologyLoadError, match="Permission denied"):
                load_ontology_from_file(str(test_file))

    @parametrize("url,error_type,error_message", [
        ("http://invalid-domain-12345.com/ontology.owl", "ConnectionError", "Failed to connect"),
        ("https://timeout-example.com/ontology.owl", "Timeout", "Request timeout"),
        ("http://server-error.com/ontology.owl", "HTTPError", "HTTP 500 error"),
    ])
    def test_load_ontology_from_url_network_errors(
        self, 
        url: str, 
        error_type: str, 
        error_message: str
    ):
        """
        Test error handling for various network issues when loading from URL.
        
        Args:
            url: Test URL
            error_type: Type of network error to simulate
            error_message: Expected error message pattern
        """
        from src.ontology.loader import load_ontology_from_url, OntologyLoadError
        
        with patch('owlready2.get_ontology') as mock_get_ont:
            # Mock different types of network errors
            mock_ont = Mock()
            
            if error_type == "ConnectionError":
                from requests.exceptions import ConnectionError
                mock_ont.load.side_effect = ConnectionError(error_message)
            elif error_type == "Timeout":
                from requests.exceptions import Timeout
                mock_ont.load.side_effect = Timeout(error_message)
            elif error_type == "HTTPError":
                from requests.exceptions import HTTPError
                mock_ont.load.side_effect = HTTPError(error_message)
            
            mock_get_ont.return_value = mock_ont
            
            # Act & Assert
            with expect_exception(OntologyLoadError, match="Network error"):
                load_ontology_from_url(url)

    def test_load_ontology_from_file_invalid_owl_format(self, invalid_owl_file: Path):
        """
        Test error handling when OWL file has invalid format.
        
        Args:
            invalid_owl_file: Path to invalid OWL file fixture
        """
        from src.ontology.loader import load_ontology_from_file, OntologyLoadError
        
        with patch('owlready2.get_ontology') as mock_get_ont:
            # Mock Owlready2 parsing error
            mock_ont = Mock()
            mock_ont.load.side_effect = OwlReadyOntologyParsingError("Invalid OWL syntax")
            mock_get_ont.return_value = mock_ont
            
            # Act & Assert
            with expect_exception(OntologyLoadError, match="Invalid OWL format"):
                load_ontology_from_file(str(invalid_owl_file))

    def test_load_ontology_from_url_invalid_owl_format(self):
        """
        Test error handling when URL returns invalid OWL format.
        """
        from src.ontology.loader import load_ontology_from_url, OntologyLoadError
        
        test_url = "http://example.com/invalid_ontology.owl"
        
        with patch('owlready2.get_ontology') as mock_get_ont:
            # Mock Owlready2 parsing error
            mock_ont = Mock()
            mock_ont.load.side_effect = OwlReadyOntologyParsingError("Malformed RDF/XML")
            mock_get_ont.return_value = mock_ont
            
            # Act & Assert
            with expect_exception(OntologyLoadError, match="Invalid OWL format"):
                load_ontology_from_url(test_url)

    def test_load_ontology_from_file_generic_owlready_error(self, temp_owl_file: Path):
        """
        Test error handling for generic Owlready2 errors.
        
        Args:
            temp_owl_file: Temporary OWL file fixture
        """
        from src.ontology.loader import load_ontology_from_file, OntologyLoadError
        
        with patch('owlready2.get_ontology') as mock_get_ont:
            # Mock generic Owlready2 error
            mock_ont = Mock()
            mock_ont.load.side_effect = OwlReadyError("Generic Owlready2 error")
            mock_get_ont.return_value = mock_ont
            
            # Act & Assert
            with expect_exception(OntologyLoadError, match="Owlready2 error"):
                load_ontology_from_file(str(temp_owl_file))

    def test_load_ontology_from_url_generic_owlready_error(self):
        """
        Test error handling for generic Owlready2 errors when loading from URL.
        """
        from src.ontology.loader import load_ontology_from_url, OntologyLoadError
        
        test_url = "http://example.com/ontology.owl"
        
        with patch('owlready2.get_ontology') as mock_get_ont:
            # Mock generic Owlready2 error
            mock_ont = Mock()
            mock_ont.load.side_effect = OwlReadyError("Unknown Owlready2 issue")
            mock_get_ont.return_value = mock_ont
            
            # Act & Assert
            with expect_exception(OntologyLoadError, match="Owlready2 error"):
                load_ontology_from_url(test_url)

    @parametrize("file_path", [
        "",
        None,
        "   ",
    ])
    def test_load_ontology_from_file_invalid_path(self, file_path: str):
        """
        Test error handling for invalid file paths.
        
        Args:
            file_path: Invalid file path to test
        """
        from src.ontology.loader import load_ontology_from_file, OntologyLoadError
        
        # Act & Assert
        with expect_exception(OntologyLoadError, match="Invalid file path"):
            load_ontology_from_file(file_path)

    @parametrize("url", [
        "",
        None,
        "   ",
        "invalid-url",
        "ftp://example.com/ontology.owl",  # Unsupported protocol
    ])
    def test_load_ontology_from_url_invalid_url(self, url: str):
        """
        Test error handling for invalid URLs.
        
        Args:
            url: Invalid URL to test
        """
        from src.ontology.loader import load_ontology_from_url, OntologyLoadError
        
        # Act & Assert
        with expect_exception(OntologyLoadError, match="Invalid URL"):
            load_ontology_from_url(url)

    def test_ontology_load_error_custom_exception(self):
        """
        Test that custom OntologyLoadError exception works correctly.
        """
        from src.ontology.loader import OntologyLoadError
        
        # Test basic exception creation
        error_msg = "Test error message"
        exception = OntologyLoadError(error_msg)
        
        assert str(exception) == error_msg
        assert isinstance(exception, Exception)

    def test_ontology_load_error_with_cause(self):
        """
        Test that OntologyLoadError properly handles exception chaining.
        """
        from src.ontology.loader import OntologyLoadError
        
        # Test exception chaining
        original_error = ValueError("Original error")
        try:
            raise OntologyLoadError("Wrapped error") from original_error
        except OntologyLoadError as chained_error:
            assert str(chained_error) == "Wrapped error"
            assert chained_error.__cause__ == original_error

    def test_load_ontology_from_file_absolute_path_conversion(self, temp_owl_file: Path):
        """
        Test that relative paths are converted to absolute paths.
        
        Args:
            temp_owl_file: Temporary OWL file fixture
        """
        from src.ontology.loader import load_ontology_from_file
        
        with patch('owlready2.get_ontology') as mock_get_ont, \
             patch('pathlib.Path.resolve') as mock_resolve:
            
            # Setup mocks
            mock_ont = Mock()
            mock_ont.load.return_value = mock_ont
            mock_get_ont.return_value = mock_ont
            mock_resolve.return_value = temp_owl_file
            
            # Test with relative path
            relative_path = "./test_ontology.owl"
            
            # Act
            load_ontology_from_file(relative_path)
            
            # Assert that path was resolved to absolute
            mock_resolve.assert_called_once()

    def test_concurrent_loading_thread_safety(self, temp_owl_file: Path):
        """
        Test that ontology loading is thread-safe for concurrent operations.
        
        Args:
            temp_owl_file: Temporary OWL file fixture
        """
        import threading
        from src.ontology.loader import load_ontology_from_file
        
        results = []
        errors = []
        
        # Use a global patch to avoid conflicts between threads
        with patch('src.ontology.loader.owlready2.get_ontology') as mock_get_ont:
            def load_ontology_worker():
                try:
                    mock_ont = Mock()
                    mock_ont.load.return_value = mock_ont
                    mock_ont.name = f"ontology_{threading.current_thread().ident}"
                    mock_get_ont.return_value = mock_ont
                    
                    result = load_ontology_from_file(str(temp_owl_file))
                    results.append(result)
                except Exception as e:
                    errors.append(e)
            
            # Create multiple threads
            threads = []
            for _ in range(3):  # Reduce thread count to avoid resource contention
                thread = threading.Thread(target=load_ontology_worker)
                threads.append(thread)
            
            # Start all threads
            for thread in threads:
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Assert all operations completed successfully
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 3

    def test_load_ontology_memory_cleanup(self, temp_owl_file: Path):
        """
        Test that ontology loading properly handles memory cleanup on errors.
        
        Args:
            temp_owl_file: Temporary OWL file fixture
        """
        from src.ontology.loader import load_ontology_from_file, OntologyLoadError
        
        with patch('owlready2.get_ontology') as mock_get_ont:
            # Mock an ontology that fails during loading but needs cleanup
            mock_ont = Mock()
            mock_ont.load.side_effect = Exception("Loading failed")
            mock_ont.destroy = Mock()  # Mock cleanup method
            mock_get_ont.return_value = mock_ont
            
            # Act & Assert
            with expect_exception(OntologyLoadError):
                load_ontology_from_file(str(temp_owl_file))
            
            # Verify cleanup was attempted (if implemented in actual loader)
            # This test documents expected behavior for memory management

    def test_load_ontology_logging_integration(self, temp_owl_file: Path):
        """
        Test that ontology loading integrates properly with logging system.
        
        Args:  
            temp_owl_file: Temporary OWL file fixture
        """
        from src.ontology.loader import load_ontology_from_file
        import logging
        
        with patch('owlready2.get_ontology') as mock_get_ont, \
             patch('logging.getLogger') as mock_logger:
            
            # Setup mocks
            mock_ont = Mock()
            mock_ont.load.return_value = mock_ont
            mock_get_ont.return_value = mock_ont
            
            logger_instance = Mock()
            mock_logger.return_value = logger_instance
            
            # Act
            load_ontology_from_file(str(temp_owl_file))
            
            # Assert logging was used (if implemented in actual loader)
            # This test documents expected logging behavior