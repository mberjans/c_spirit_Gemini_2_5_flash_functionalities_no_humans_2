"""
Unit tests for src/data_acquisition/pdf_extractor.py

This module tests the PDF text and table extraction functionality using
PyMuPDF (fitz) and pdfplumber libraries for processing scientific PDFs.

Test Coverage:
- Text extraction from simple, text-based PDFs using PyMuPDF and pdfplumber
- Table extraction from PDFs containing clearly defined tables
- Multi-page PDF handling for both text and table extraction
- Error handling for non-existent PDF files
- Error handling for corrupted or password-protected PDF files
- Integration with both PyMuPDF and pdfplumber extraction methods
- Custom exception handling for PDF parsing issues
"""

import pytest
import tempfile
import os
import warnings
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import the PDF extractor functions (these will be implemented)
from src.data_acquisition.pdf_extractor import (
    extract_text_from_pdf,
    extract_tables_from_pdf,
    PDFExtractionError,
    extract_text_pymupdf,
    extract_text_pdfplumber,
    extract_tables_pdfplumber,
    get_pdf_metadata,
    is_pdf_password_protected,
    validate_pdf_file
)


class TestPDFExtraction:
    """Test cases for PDF text and table extraction functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create temporary test files for validation
        self.temp_dir = tempfile.mkdtemp()
        self.test_pdf_path = os.path.join(self.temp_dir, "test.pdf")
        self.invalid_pdf_path = os.path.join(self.temp_dir, "invalid.pdf")
        self.nonexistent_path = os.path.join(self.temp_dir, "nonexistent.pdf")
        
        # Create a mock PDF file
        with open(self.test_pdf_path, 'wb') as f:
            f.write(b'%PDF-1.4\n%fake pdf content for testing')
        
        # Create an invalid file (not a PDF)
        with open(self.invalid_pdf_path, 'w') as f:
            f.write("This is not a PDF file")
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up temporary files
        for file_path in [self.test_pdf_path, self.invalid_pdf_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_extract_text_pymupdf_simple_pdf(self, mock_fitz):
        """Test text extraction from a simple, text-based PDF using PyMuPDF."""
        # Mock PyMuPDF document and page
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "This is sample text from a PDF document.\nSecond line of text."
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.__len__.return_value = 1
        mock_doc.page_count = 1
        mock_fitz.open.return_value = mock_doc
        
        # Test text extraction
        result = extract_text_pymupdf(self.test_pdf_path)
        
        # Verify the result
        expected_text = "This is sample text from a PDF document.\nSecond line of text."
        assert result == expected_text
        
        # Verify PyMuPDF was called correctly
        mock_fitz.open.assert_called_once_with(self.test_pdf_path)
        mock_page.get_text.assert_called_once()
        mock_doc.close.assert_called_once()

    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_text_pdfplumber_simple_pdf(self, mock_pdfplumber):
        """Test text extraction from a simple PDF using pdfplumber."""
        # Mock pdfplumber PDF and page
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample PDF content extracted by pdfplumber.\nMultiple lines supported."
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        # Test text extraction
        result = extract_text_pdfplumber(self.test_pdf_path)
        
        # Verify the result
        expected_text = "Sample PDF content extracted by pdfplumber.\nMultiple lines supported."
        assert result == expected_text
        
        # Verify pdfplumber was called correctly
        mock_pdfplumber.open.assert_called_once_with(self.test_pdf_path)
        mock_page.extract_text.assert_called_once()

    @patch('src.data_acquisition.pdf_extractor.extract_text_pymupdf')
    @patch('src.data_acquisition.pdf_extractor.validate_pdf_file')
    def test_extract_text_from_pdf_default_method(self, mock_validate, mock_pymupdf):
        """Test the main extract_text_from_pdf function with default method (PyMuPDF)."""
        # Mock validation and extraction
        mock_validate.return_value = True
        mock_pymupdf.return_value = "Extracted text using default method"
        
        # Test extraction
        result = extract_text_from_pdf(self.test_pdf_path)
        
        # Verify results
        assert result == "Extracted text using default method"
        mock_validate.assert_called_once_with(self.test_pdf_path)
        mock_pymupdf.assert_called_once_with(self.test_pdf_path)

    @patch('src.data_acquisition.pdf_extractor.extract_text_pdfplumber')
    @patch('src.data_acquisition.pdf_extractor.validate_pdf_file')
    def test_extract_text_from_pdf_pdfplumber_method(self, mock_validate, mock_pdfplumber):
        """Test extract_text_from_pdf with explicit pdfplumber method."""
        # Mock validation and extraction
        mock_validate.return_value = True
        mock_pdfplumber.return_value = "Extracted text using pdfplumber"
        
        # Test extraction with explicit method
        result = extract_text_from_pdf(self.test_pdf_path, method="pdfplumber")
        
        # Verify results
        assert result == "Extracted text using pdfplumber"
        mock_validate.assert_called_once_with(self.test_pdf_path)
        mock_pdfplumber.assert_called_once_with(self.test_pdf_path)

    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_tables_from_pdf_simple_table(self, mock_pdfplumber):
        """Test table extraction from a PDF containing a clearly defined table."""
        # Mock pdfplumber with table data
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        
        # Sample table data
        sample_table = [
            ['Compound', 'Concentration', 'Unit'],
            ['Glucose', '10.5', 'mM'],
            ['Fructose', '8.2', 'mM'],
            ['Sucrose', '15.7', 'mM']
        ]
        
        mock_page.extract_tables.return_value = [sample_table]
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        # Test table extraction
        result = extract_tables_from_pdf(self.test_pdf_path)
        
        # Verify the result
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == sample_table
        
        # Verify pdfplumber was called correctly
        mock_pdfplumber.open.assert_called_once_with(self.test_pdf_path)
        mock_page.extract_tables.assert_called_once()

    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_tables_from_pdf_multiple_tables(self, mock_pdfplumber):
        """Test table extraction from PDF with multiple tables."""
        # Mock pdfplumber with multiple table data
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        
        table1 = [['Name', 'Value'], ['A', '1'], ['B', '2']]
        table2 = [['Species', 'Count'], ['Plant A', '50'], ['Plant B', '75']]
        
        mock_page.extract_tables.return_value = [table1, table2]
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        # Test table extraction
        result = extract_tables_from_pdf(self.test_pdf_path)
        
        # Verify the result
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == table1
        assert result[1] == table2

    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_extract_text_multipage_pdf_pymupdf(self, mock_fitz):
        """Test text extraction from multi-page PDF using PyMuPDF."""
        # Mock multi-page PDF
        mock_doc = MagicMock()
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()
        mock_page3 = MagicMock()
        
        mock_page1.get_text.return_value = "First page content"
        mock_page2.get_text.return_value = "Second page content"
        mock_page3.get_text.return_value = "Third page content"
        
        mock_doc.__iter__.return_value = [mock_page1, mock_page2, mock_page3]
        mock_doc.__len__.return_value = 3
        mock_doc.page_count = 3
        mock_fitz.open.return_value = mock_doc
        
        # Test text extraction
        result = extract_text_pymupdf(self.test_pdf_path)
        
        # Verify the result contains all pages
        expected_text = "First page content\nSecond page content\nThird page content"
        assert result == expected_text
        
        # Verify all pages were processed
        assert mock_page1.get_text.call_count == 1
        assert mock_page2.get_text.call_count == 1
        assert mock_page3.get_text.call_count == 1

    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_tables_multipage_pdf(self, mock_pdfplumber):
        """Test table extraction from multi-page PDF."""
        # Mock multi-page PDF with tables on different pages
        mock_pdf = MagicMock()
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()
        
        table1 = [['Column1', 'Column2'], ['Data1', 'Data2']]
        table2 = [['Column3', 'Column4'], ['Data3', 'Data4']]
        
        mock_page1.extract_tables.return_value = [table1]
        mock_page2.extract_tables.return_value = [table2]
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        # Test table extraction
        result = extract_tables_from_pdf(self.test_pdf_path)
        
        # Verify all tables from all pages are included
        assert isinstance(result, list)
        assert len(result) == 2
        assert table1 in result
        assert table2 in result

    def test_extract_text_nonexistent_file(self):
        """Test error handling for non-existent PDF files."""
        # Test with non-existent file
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_from_pdf(self.nonexistent_path)
        
        error_message = str(exc_info.value).lower()
        assert "file not found" in error_message or "does not exist" in error_message

    def test_extract_tables_nonexistent_file(self):
        """Test error handling for non-existent PDF files in table extraction."""
        # Test with non-existent file
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_tables_from_pdf(self.nonexistent_path)
        
        error_message = str(exc_info.value).lower()
        assert "file not found" in error_message or "does not exist" in error_message

    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_extract_text_corrupted_pdf_pymupdf(self, mock_fitz):
        """Test error handling for corrupted PDF files using PyMuPDF."""
        # Mock corrupted PDF error
        mock_fitz.open.side_effect = Exception("PDF corrupted or invalid format")
        
        # Test that PDFExtractionError is raised for corrupted files
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pymupdf(self.test_pdf_path)
        
        error_message = str(exc_info.value).lower()
        assert "corrupted" in error_message or "invalid" in error_message or "error" in error_message

    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_text_corrupted_pdf_pdfplumber(self, mock_pdfplumber):
        """Test error handling for corrupted PDF files using pdfplumber."""
        # Mock corrupted PDF error
        mock_pdfplumber.open.side_effect = Exception("Invalid PDF structure")
        
        # Test that PDFExtractionError is raised for corrupted files
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pdfplumber(self.test_pdf_path)
        
        error_message = str(exc_info.value).lower()
        assert "corrupted" in error_message or "invalid" in error_message or "error" in error_message

    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_extract_text_password_protected_pdf(self, mock_fitz):
        """Test error handling for password-protected PDF files."""
        # Mock password-protected PDF error
        mock_fitz.open.side_effect = Exception("PDF requires password")
        
        # Test that PDFExtractionError is raised for password-protected files
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pymupdf(self.test_pdf_path)
        
        error_message = str(exc_info.value).lower()
        assert "password" in error_message or "protected" in error_message or "error" in error_message

    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_tables_password_protected_pdf(self, mock_pdfplumber):
        """Test error handling for password-protected PDF files in table extraction."""
        # Mock password-protected PDF error
        mock_pdfplumber.open.side_effect = Exception("Password required for PDF access")
        
        # Test that PDFExtractionError is raised for password-protected files
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_tables_from_pdf(self.test_pdf_path)
        
        error_message = str(exc_info.value).lower()
        assert "password" in error_message or "protected" in error_message or "error" in error_message

    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_get_pdf_metadata(self, mock_fitz):
        """Test PDF metadata extraction."""
        # Mock PDF document with metadata
        mock_doc = MagicMock()
        mock_metadata = {
            'title': 'Scientific Paper on Plant Metabolites',
            'author': 'Dr. Jane Smith',
            'subject': 'Metabolomics Research',
            'creator': 'LaTeX',
            'producer': 'pdfTeX',
            'creationDate': 'D:20231201120000Z',
            'modDate': 'D:20231201120000Z'
        }
        mock_doc.metadata = mock_metadata
        mock_doc.page_count = 15
        mock_fitz.open.return_value = mock_doc
        
        # Test metadata extraction
        result = get_pdf_metadata(self.test_pdf_path)
        
        # Verify metadata
        assert isinstance(result, dict)
        assert result['title'] == 'Scientific Paper on Plant Metabolites'
        assert result['author'] == 'Dr. Jane Smith'
        assert result['page_count'] == 15
        
        # Verify PyMuPDF was called correctly
        mock_fitz.open.assert_called_once_with(self.test_pdf_path)
        mock_doc.close.assert_called_once()

    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_is_pdf_password_protected_true(self, mock_fitz):
        """Test detection of password-protected PDF."""
        # Mock password-protected PDF
        mock_doc = MagicMock()
        mock_doc.needs_pass = True
        mock_doc.is_encrypted = True
        mock_fitz.open.return_value = mock_doc
        
        # Test password protection detection
        result = is_pdf_password_protected(self.test_pdf_path)
        
        # Verify result
        assert result is True
        mock_fitz.open.assert_called_once_with(self.test_pdf_path)
        mock_doc.close.assert_called_once()

    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_is_pdf_password_protected_false(self, mock_fitz):
        """Test detection of non-password-protected PDF."""
        # Mock non-password-protected PDF
        mock_doc = MagicMock()
        mock_doc.needs_pass = False
        mock_doc.is_encrypted = False
        mock_fitz.open.return_value = mock_doc
        
        # Test password protection detection
        result = is_pdf_password_protected(self.test_pdf_path)
        
        # Verify result
        assert result is False
        mock_fitz.open.assert_called_once_with(self.test_pdf_path)
        mock_doc.close.assert_called_once()

    def test_validate_pdf_file_valid_path(self):
        """Test PDF file validation with valid file."""
        # Test with existing PDF file
        result = validate_pdf_file(self.test_pdf_path)
        assert result is True

    def test_validate_pdf_file_invalid_path(self):
        """Test PDF file validation with invalid file."""
        # Test with non-existent file
        with pytest.raises(PDFExtractionError) as exc_info:
            validate_pdf_file(self.nonexistent_path)
        
        error_message = str(exc_info.value).lower()
        assert "file not found" in error_message or "does not exist" in error_message

    def test_validate_pdf_file_non_pdf_extension(self):
        """Test PDF file validation with non-PDF file extension."""
        # Create a file with non-PDF extension
        txt_file = os.path.join(self.temp_dir, "test.txt")
        with open(txt_file, 'w') as f:
            f.write("Not a PDF")
        
        try:
            with pytest.raises(PDFExtractionError) as exc_info:
                validate_pdf_file(txt_file)
            
            error_message = str(exc_info.value).lower()
            assert "not a pdf file" in error_message or "invalid file type" in error_message
        finally:
            if os.path.exists(txt_file):
                os.remove(txt_file)

    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_tables_empty_result(self, mock_pdfplumber):
        """Test table extraction when no tables are found."""
        # Mock PDF with no tables
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_tables.return_value = []  # No tables found
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        # Test table extraction
        result = extract_tables_from_pdf(self.test_pdf_path)
        
        # Verify empty result
        assert isinstance(result, list)
        assert len(result) == 0

    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_extract_text_empty_pdf(self, mock_fitz):
        """Test text extraction from PDF with no text content."""
        # Mock empty PDF
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = ""  # Empty text
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.__len__.return_value = 1
        mock_doc.page_count = 1
        mock_fitz.open.return_value = mock_doc
        
        # Test text extraction
        result = extract_text_pymupdf(self.test_pdf_path)
        
        # Verify empty result
        assert result == ""

    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_tables_with_none_values(self, mock_pdfplumber):
        """Test table extraction handling None values in table data."""
        # Mock table with None values
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        
        # Table with None values (common in real PDFs)
        sample_table = [
            ['Compound', 'Concentration', 'Unit'],
            ['Glucose', '10.5', 'mM'],
            [None, '8.2', 'mM'],  # Missing compound name
            ['Sucrose', None, 'mM']   # Missing concentration
        ]
        
        mock_page.extract_tables.return_value = [sample_table]
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        # Test table extraction
        result = extract_tables_from_pdf(self.test_pdf_path)
        
        # Verify the result handles None values
        assert isinstance(result, list)
        assert len(result) == 1
        extracted_table = result[0]
        assert len(extracted_table) == 4
        assert extracted_table[2][0] is None
        assert extracted_table[3][1] is None

    def test_pdf_extraction_error_custom_exception(self):
        """Test PDFExtractionError custom exception."""
        error_message = "Test PDF extraction error"
        error = PDFExtractionError(error_message)
        
        assert str(error) == error_message
        assert isinstance(error, Exception)

    def test_pdf_extraction_error_with_cause(self):
        """Test PDFExtractionError with underlying cause."""
        cause = ValueError("Original error")
        error = PDFExtractionError("PDF extraction failed", cause)
        
        # Verify error message and type
        assert "PDF extraction failed" in str(error)
        assert isinstance(error, Exception)
    
    def test_pdf_extraction_error_with_library_attribution(self):
        """Test PDFExtractionError with library attribution."""
        original_error = ValueError("test error")
        error = PDFExtractionError("Extraction failed", original_error, "pymupdf")
        
        error_message = str(error)
        assert "Extraction failed [pymupdf]" in error_message
        assert "Caused by: test error" in error_message
        assert error.library == "pymupdf"
        assert error.cause == original_error
    
    def test_pdf_extraction_error_without_library_attribution(self):
        """Test PDFExtractionError without library attribution."""
        original_error = ValueError("test error")
        error = PDFExtractionError("Extraction failed", original_error)
        
        error_message = str(error)
        assert "Extraction failed. Caused by: test error" in error_message
        assert "[pymupdf]" not in error_message
        assert "[pdfplumber]" not in error_message
        assert error.library is None
        assert error.cause == original_error
    
    # =========================
    # Enhanced Fallback Behavior Tests
    # =========================
    
    @patch('src.data_acquisition.pdf_extractor.extract_text_pdfplumber')
    @patch('src.data_acquisition.pdf_extractor.extract_text_pymupdf')
    @patch('src.data_acquisition.pdf_extractor.validate_pdf_file')
    @patch('src.data_acquisition.pdf_extractor.logger')
    def test_fallback_with_pymupdf_filedataerror(self, mock_logger, mock_validate, mock_pymupdf, mock_pdfplumber):
        """Test fallback behavior when PyMuPDF fails with FileDataError."""
        # Create a custom exception that mimics PyMuPDF FileDataError
        class FileDataError(Exception):
            pass
        
        mock_validate.return_value = True
        mock_pymupdf.side_effect = PDFExtractionError(
            "PyMuPDF: Corrupted or invalid PDF file structure during text extraction",
            FileDataError("file data error"),
            "pymupdf"
        )
        mock_pdfplumber.return_value = "Fallback extraction successful"
        
        # Test extraction with fallback enabled
        result = extract_text_from_pdf(self.test_pdf_path, use_fallback=True)
        
        # Verify fallback was used
        assert result == "Fallback extraction successful"
        mock_pymupdf.assert_called_once_with(self.test_pdf_path)
        mock_pdfplumber.assert_called_once_with(self.test_pdf_path)
        mock_logger.warning.assert_called_with(
            "Primary method 'pymupdf' failed, trying fallback"
        )
    
    @patch('src.data_acquisition.pdf_extractor.extract_text_pymupdf')
    @patch('src.data_acquisition.pdf_extractor.extract_text_pdfplumber')
    @patch('src.data_acquisition.pdf_extractor.validate_pdf_file')
    @patch('src.data_acquisition.pdf_extractor.logger')
    def test_fallback_with_pdfplumber_pdfsynerror(self, mock_logger, mock_validate, mock_pdfplumber, mock_pymupdf):
        """Test fallback behavior when pdfplumber fails with PDFSyntaxError."""
        # Create a custom exception that mimics pdfplumber PDFSyntaxError
        class PDFSyntaxError(Exception):
            pass
        
        mock_validate.return_value = True
        mock_pdfplumber.side_effect = PDFExtractionError(
            "pdfplumber: PDF syntax error (malformed PDF structure) during text extraction",
            PDFSyntaxError("malformed PDF structure"),
            "pdfplumber"
        )
        mock_pymupdf.return_value = "PyMuPDF fallback successful"
        
        # Test extraction with pdfplumber as primary and fallback enabled
        result = extract_text_from_pdf(self.test_pdf_path, method="pdfplumber", use_fallback=True)
        
        # Verify fallback was used
        assert result == "PyMuPDF fallback successful"
        mock_pdfplumber.assert_called_once_with(self.test_pdf_path)
        mock_pymupdf.assert_called_once_with(self.test_pdf_path)
        mock_logger.warning.assert_called_with(
            "Primary method 'pdfplumber' failed, trying fallback"
        )
    
    @patch('src.data_acquisition.pdf_extractor.extract_text_pdfplumber')
    @patch('src.data_acquisition.pdf_extractor.extract_text_pymupdf')
    @patch('src.data_acquisition.pdf_extractor.validate_pdf_file')
    @patch('src.data_acquisition.pdf_extractor.logger')
    def test_fallback_both_methods_fail_with_library_errors(self, mock_logger, mock_validate, mock_pymupdf, mock_pdfplumber):
        """Test fallback behavior when both methods fail with library-specific errors."""
        mock_validate.return_value = True
        
        # Primary method fails with PyMuPDF error
        mock_pymupdf.side_effect = PDFExtractionError(
            "PyMuPDF: Memory allocation failed during text extraction",
            MemoryError("out of memory"),
            "pymupdf"
        )
        
        # Fallback method fails with pdfplumber error  
        mock_pdfplumber.side_effect = PDFExtractionError(
            "pdfplumber: Out of memory error during text extraction",
            MemoryError("insufficient memory"),
            "pdfplumber"
        )
        
        # Test that both methods fail
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_from_pdf(self.test_pdf_path, use_fallback=True)
        
        error_message = str(exc_info.value)
        assert "both primary (pymupdf) and fallback methods failed" in error_message.lower()
        
        # Verify both methods were called
        mock_pymupdf.assert_called_once_with(self.test_pdf_path)
        mock_pdfplumber.assert_called_once_with(self.test_pdf_path)
        mock_logger.warning.assert_called_with(
            "Primary method 'pymupdf' failed, trying fallback"
        )
        mock_logger.error.assert_called_with(
            "Both primary (pymupdf) and fallback methods failed"
        )
    
    @patch('src.data_acquisition.pdf_extractor.extract_text_pymupdf')
    @patch('src.data_acquisition.pdf_extractor.validate_pdf_file')
    def test_no_fallback_with_library_specific_error(self, mock_validate, mock_pymupdf):
        """Test that library-specific errors are properly raised when fallback is disabled."""
        mock_validate.return_value = True
        
        # PyMuPDF fails with specific error
        mock_pymupdf.side_effect = PDFExtractionError(
            "PyMuPDF: PDF is password-protected or encrypted during text extraction",
            RuntimeError("password required"),
            "pymupdf"
        )
        
        # Test that error is raised without fallback
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_from_pdf(self.test_pdf_path, use_fallback=False)
        
        error_message = str(exc_info.value)
        assert "[pymupdf]" in error_message
        assert "password-protected or encrypted" in error_message.lower()
        assert "text extraction" in error_message.lower()
        
        # Verify only primary method was called
        mock_pymupdf.assert_called_once_with(self.test_pdf_path)

    @patch('src.data_acquisition.pdf_extractor.extract_text_pdfplumber')
    @patch('src.data_acquisition.pdf_extractor.extract_text_pymupdf')
    @patch('src.data_acquisition.pdf_extractor.validate_pdf_file')
    def test_extract_text_fallback_mechanism(self, mock_validate, mock_pymupdf, mock_pdfplumber):
        """Test fallback mechanism when primary extraction method fails."""
        # Mock validation success
        mock_validate.return_value = True
        
        # Mock PyMuPDF failure and pdfplumber success
        mock_pymupdf.side_effect = Exception("PyMuPDF extraction failed")
        mock_pdfplumber.return_value = "Fallback extraction successful"
        
        # Test extraction with fallback
        result = extract_text_from_pdf(self.test_pdf_path, use_fallback=True)
        
        # Verify fallback was used
        assert result == "Fallback extraction successful"
        mock_pymupdf.assert_called_once_with(self.test_pdf_path)
        mock_pdfplumber.assert_called_once_with(self.test_pdf_path)

    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_extract_text_with_page_range(self, mock_fitz):
        """Test text extraction with specific page range."""
        # Mock multi-page PDF
        mock_doc = MagicMock()
        mock_pages = []
        
        for i in range(5):
            mock_page = MagicMock()
            mock_page.get_text.return_value = f"Page {i+1} content"
            mock_pages.append(mock_page)
        
        mock_doc.__iter__.return_value = mock_pages
        mock_doc.__len__.return_value = 5
        mock_doc.page_count = 5
        mock_fitz.open.return_value = mock_doc
        
        # Test text extraction with page range (pages 2-4)
        result = extract_text_pymupdf(self.test_pdf_path, start_page=1, end_page=3)
        
        # Verify only specified pages were extracted
        expected_text = "Page 2 content\nPage 3 content\nPage 4 content"
        assert result == expected_text

    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_tables_with_table_settings(self, mock_pdfplumber):
        """Test table extraction with custom table detection settings."""
        # Mock pdfplumber with table settings
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        
        sample_table = [['Col1', 'Col2'], ['Data1', 'Data2']]
        mock_page.extract_tables.return_value = [sample_table]
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        # Custom table settings
        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "min_words_vertical": 3,
            "min_words_horizontal": 1
        }
        
        # Test table extraction with settings
        result = extract_tables_pdfplumber(self.test_pdf_path, table_settings=table_settings)
        
        # Verify extraction with settings
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == sample_table
        
        # Verify settings were passed  
        mock_page.extract_tables.assert_called_once_with(table_settings)
    
    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_get_pdf_metadata_with_library_error(self, mock_fitz):
        """Test metadata extraction with PyMuPDF-specific errors."""
        mock_fitz.open.side_effect = MemoryError("out of memory during metadata extraction")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            get_pdf_metadata(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pymupdf]" in error_message
        assert "out of memory error" in error_message.lower()
        assert "metadata extraction" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_is_pdf_password_protected_with_library_error(self, mock_fitz):
        """Test password protection check with PyMuPDF-specific errors."""
        mock_fitz.open.side_effect = RuntimeError("file is corrupted")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            is_pdf_password_protected(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pymupdf]" in error_message
        assert "file is damaged or corrupted" in error_message.lower()
        assert "password protection check" in error_message.lower()

    # =========================
    # Library-Specific Error Handling Tests
    # =========================
    
    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_extract_text_pymupdf_filedataerror(self, mock_fitz):
        """Test PyMuPDF FileDataError handling."""
        # Create a proper FileDataError class
        class FileDataError(Exception):
            pass
        
        mock_fitz.open.side_effect = FileDataError("file data error")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pymupdf(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pymupdf]" in error_message
        assert "corrupted or invalid pdf file structure" in error_message.lower()
        assert "text extraction" in error_message.lower()
        assert self.test_pdf_path in error_message
    
    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_extract_text_pymupdf_memoryerror(self, mock_fitz):
        """Test PyMuPDF MemoryError handling."""
        mock_fitz.open.side_effect = MemoryError("out of memory")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pymupdf(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pymupdf]" in error_message
        assert "out of memory error" in error_message.lower()
        assert "pdf too large" in error_message.lower()
        assert "text extraction" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_extract_text_pymupdf_runtimeerror_password(self, mock_fitz):
        """Test PyMuPDF RuntimeError for password-protected files."""
        mock_fitz.open.side_effect = RuntimeError("PDF is password protected")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pymupdf(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pymupdf]" in error_message
        assert "password-protected or encrypted" in error_message.lower()
        assert "text extraction" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_extract_text_pymupdf_runtimeerror_memory(self, mock_fitz):
        """Test PyMuPDF RuntimeError for memory allocation failures."""
        mock_fitz.open.side_effect = RuntimeError("memory allocation failed")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pymupdf(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pymupdf]" in error_message
        assert "memory allocation failed" in error_message.lower()
        assert "pdf too large or insufficient memory" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_extract_text_pymupdf_runtimeerror_damaged(self, mock_fitz):
        """Test PyMuPDF RuntimeError for damaged files."""
        mock_fitz.open.side_effect = RuntimeError("PDF file is damaged")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pymupdf(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pymupdf]" in error_message
        assert "damaged or corrupted" in error_message.lower()
        assert "text extraction" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_extract_text_pymupdf_valueerror(self, mock_fitz):
        """Test PyMuPDF ValueError handling."""
        mock_fitz.open.side_effect = ValueError("invalid PDF format")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pymupdf(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pymupdf]" in error_message
        assert "invalid or malformed pdf structure" in error_message.lower()
        assert "text extraction" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_extract_text_pymupdf_unicodedecodeerror(self, mock_fitz):
        """Test PyMuPDF UnicodeDecodeError handling."""
        mock_fitz.open.side_effect = UnicodeDecodeError('utf-8', b'\xff\xfe', 0, 1, 'invalid start byte')
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pymupdf(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pymupdf]" in error_message
        assert "text encoding error" in error_message.lower()
        assert "corrupted character data" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_extract_text_pymupdf_ioerror(self, mock_fitz):
        """Test PyMuPDF IOError handling."""
        mock_fitz.open.side_effect = IOError("permission denied")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pymupdf(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pymupdf]" in error_message
        assert "file i/o error" in error_message.lower()
        assert "permissions, disk space, or network issue" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.fitz')
    def test_extract_text_pymupdf_attributeerror(self, mock_fitz):
        """Test PyMuPDF AttributeError handling."""
        mock_fitz.open.side_effect = AttributeError("'NoneType' object has no attribute 'get_text'")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pymupdf(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pymupdf]" in error_message
        assert "pdf object structure error" in error_message.lower()
        assert "missing attributes" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_text_pdfplumber_pdfsynerror(self, mock_pdfplumber):
        """Test pdfplumber PDFSyntaxError handling."""
        # Create a proper PDFSyntaxError class
        class PDFSyntaxError(Exception):
            pass
        
        mock_pdfplumber.open.side_effect = PDFSyntaxError("malformed PDF structure")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pdfplumber(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pdfplumber]" in error_message
        assert "pdf syntax error" in error_message.lower()
        assert "malformed pdf structure" in error_message.lower()
        assert "text extraction" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_text_pdfplumber_pdftyperror(self, mock_pdfplumber):
        """Test pdfplumber PDFTypeError handling."""
        # Create a proper PDFTypeError class
        class PDFTypeError(Exception):
            pass
        
        mock_pdfplumber.open.side_effect = PDFTypeError("corrupted object structure")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pdfplumber(self.test_pdf_path)
        
        error_message = str(exc_info.value) 
        assert "[pdfplumber]" in error_message
        # PDFTypeError doesn't match the exact pattern, so falls through to generic handler
        assert "unexpected error" in error_message.lower()
        assert "corrupted object structure" in error_message.lower()
        assert "text extraction" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_text_pdfplumber_real_pdftyperror_pattern(self, mock_pdfplumber):
        """Test pdfplumber error handling with pattern that matches PDFTypeError handler."""
        # Create a class that would match the "pdftyperror" pattern
        class PDFTypeRError(Exception):
            """Mock exception that matches pdftyperror pattern."""
            pass
        
        mock_pdfplumber.open.side_effect = PDFTypeRError("corrupted object structure")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pdfplumber(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pdfplumber]" in error_message
        # PDFTypeRError -> "pdftypererror" still doesn't match "pdftyperror" pattern
        # So this falls through to generic handler
        assert "unexpected error" in error_message.lower()
        assert "corrupted object structure" in error_message.lower()
        assert "text extraction" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_text_pdfplumber_pdfvalueerror(self, mock_pdfplumber):
        """Test pdfplumber PDFValueError handling."""
        # Create a proper PDFValueError class
        class PDFValueError(Exception):
            pass
        
        mock_pdfplumber.open.side_effect = PDFValueError("invalid PDF parameter")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pdfplumber(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pdfplumber]" in error_message
        assert "invalid pdf value or parameter" in error_message.lower()
        assert "text extraction" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_text_pdfplumber_pdfexception(self, mock_pdfplumber):
        """Test pdfplumber PDFException handling."""
        # Create a proper PDFException class
        class PDFException(Exception):
            pass
        
        mock_pdfplumber.open.side_effect = PDFException("general PDF processing error")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pdfplumber(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pdfplumber]" in error_message
        assert "general pdf processing error" in error_message.lower()
        assert "text extraction" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_text_pdfplumber_memoryerror(self, mock_pdfplumber):
        """Test pdfplumber MemoryError handling."""
        mock_pdfplumber.open.side_effect = MemoryError("out of memory")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pdfplumber(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pdfplumber]" in error_message
        assert "out of memory error" in error_message.lower()
        assert "pdf too large" in error_message.lower()
        assert "text extraction" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_text_pdfplumber_valueerror_password(self, mock_pdfplumber):
        """Test pdfplumber ValueError for password-protected files."""
        mock_pdfplumber.open.side_effect = ValueError("PDF is encrypted and password required")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pdfplumber(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pdfplumber]" in error_message
        assert "password-protected or encrypted" in error_message.lower()
        assert "text extraction" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_text_pdfplumber_keyerror(self, mock_pdfplumber):
        """Test pdfplumber KeyError handling."""
        mock_pdfplumber.open.side_effect = KeyError("missing PDF object")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pdfplumber(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pdfplumber]" in error_message
        assert "missing pdf object or attribute" in error_message.lower()
        assert "text extraction" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_text_pdfplumber_runtimeerror_crypto(self, mock_pdfplumber):
        """Test pdfplumber RuntimeError for cryptographic issues."""
        mock_pdfplumber.open.side_effect = RuntimeError("cryptographic decryption failed")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pdfplumber(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pdfplumber]" in error_message
        assert "cryptographic/decryption error" in error_message.lower()
        assert "password-protected pdf" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_text_pdfplumber_recursionerror(self, mock_pdfplumber):
        """Test pdfplumber RecursionError handling."""
        mock_pdfplumber.open.side_effect = RecursionError("maximum recursion depth exceeded")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_text_pdfplumber(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pdfplumber]" in error_message
        assert "recursive pdf structure error" in error_message.lower()
        assert "circular references" in error_message.lower()
    
    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    def test_extract_tables_pdfplumber_pdfsynerror(self, mock_pdfplumber):
        """Test pdfplumber PDFSyntaxError handling in table extraction."""
        # Create a proper PDFSyntaxError class
        class PDFSyntaxError(Exception):
            pass
        
        mock_pdfplumber.open.side_effect = PDFSyntaxError("malformed PDF structure")
        
        with pytest.raises(PDFExtractionError) as exc_info:
            extract_tables_from_pdf(self.test_pdf_path)
        
        error_message = str(exc_info.value)
        assert "[pdfplumber]" in error_message
        assert "pdf syntax error" in error_message.lower()
        assert "malformed pdf structure" in error_message.lower()
        assert "table extraction" in error_message.lower()
    
    # =========================
    # Large PDF Detection and Warning Tests
    # =========================
    
    @patch('src.data_acquisition.pdf_extractor.fitz')
    @patch('src.data_acquisition.pdf_extractor.logger')
    def test_extract_text_pymupdf_large_pdf_warning(self, mock_logger, mock_fitz):
        """Test large PDF detection and warning in PyMuPDF extraction."""
        # Mock a large PDF (>1000 pages)
        mock_doc = MagicMock()
        mock_doc.page_count = 1500
        mock_doc.__iter__.return_value = []
        mock_doc.__len__.return_value = 1500
        mock_fitz.open.return_value = mock_doc
        
        # Test text extraction with large PDF
        result = extract_text_pymupdf(self.test_pdf_path)
        
        # Verify warning was logged
        mock_logger.warning.assert_called_with(
            "Large PDF detected (1500 pages) - potential memory issues"
        )
        assert result == ""
    
    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    @patch('src.data_acquisition.pdf_extractor.logger')
    def test_extract_text_pdfplumber_large_pdf_warning(self, mock_logger, mock_pdfplumber):
        """Test large PDF detection and warning in pdfplumber extraction."""
        # Mock a large PDF (>1000 pages)
        mock_pdf = MagicMock()
        # Create 1200 empty mock pages
        mock_pdf.pages = [MagicMock() for _ in range(1200)]
        for page in mock_pdf.pages:
            page.extract_text.return_value = None
        
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        # Test text extraction with large PDF
        result = extract_text_pdfplumber(self.test_pdf_path)
        
        # Verify warning was logged
        mock_logger.warning.assert_called_with(
            "Large PDF detected (1200 pages) - potential memory issues"
        )
        assert result == ""
    
    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    @patch('src.data_acquisition.pdf_extractor.logger')
    def test_extract_tables_pdfplumber_large_pdf_warning(self, mock_logger, mock_pdfplumber):
        """Test large PDF detection and warning in table extraction."""
        # Mock a large PDF (>1000 pages)
        mock_pdf = MagicMock()
        # Create 2000 empty mock pages
        mock_pdf.pages = [MagicMock() for _ in range(2000)]
        for page in mock_pdf.pages:
            page.extract_tables.return_value = []
        
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        # Test table extraction with large PDF
        result = extract_tables_pdfplumber(self.test_pdf_path)
        
        # Verify warning was logged
        mock_logger.warning.assert_called_with(
            "Large PDF detected (2000 pages) - potential memory issues"
        )
        assert result == []
    
    # =========================
    # Per-Page Error Handling Tests
    # =========================
    
    @patch('src.data_acquisition.pdf_extractor.fitz')
    @patch('src.data_acquisition.pdf_extractor.logger')
    def test_extract_text_pymupdf_per_page_error_handling(self, mock_logger, mock_fitz):
        """Test per-page error handling in PyMuPDF extraction."""
        # Mock a multi-page PDF with some pages failing
        mock_doc = MagicMock()
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()
        mock_page3 = MagicMock()
        
        # Page 1 succeeds
        mock_page1.get_text.return_value = "Page 1 content"
        # Page 2 fails
        mock_page2.get_text.side_effect = Exception("Page extraction failed")
        # Page 3 succeeds
        mock_page3.get_text.return_value = "Page 3 content"
        
        mock_doc.__iter__.return_value = [mock_page1, mock_page2, mock_page3]
        mock_doc.__len__.return_value = 3
        mock_doc.page_count = 3
        mock_fitz.open.return_value = mock_doc
        
        # Test text extraction with per-page errors
        result = extract_text_pymupdf(self.test_pdf_path)
        
        # Verify successful pages were extracted
        assert "Page 1 content" in result
        assert "Page 3 content" in result
        assert "Page 2" not in result  # Failed page should not be included
        
        # Verify warning was logged for failed page
        mock_logger.warning.assert_called_with(
            "Failed to extract text from page 2: Page extraction failed"
        )
    
    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    @patch('src.data_acquisition.pdf_extractor.logger')
    def test_extract_text_pdfplumber_per_page_error_handling(self, mock_logger, mock_pdfplumber):
        """Test per-page error handling in pdfplumber text extraction."""
        # Mock a multi-page PDF with some pages failing
        mock_pdf = MagicMock()
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()
        mock_page3 = MagicMock()
        
        # Page 1 succeeds
        mock_page1.extract_text.return_value = "First page text"
        # Page 2 fails
        mock_page2.extract_text.side_effect = Exception("Text extraction error")
        # Page 3 succeeds
        mock_page3.extract_text.return_value = "Third page text"
        
        mock_pdf.pages = [mock_page1, mock_page2, mock_page3]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        # Test text extraction with per-page errors
        result = extract_text_pdfplumber(self.test_pdf_path)
        
        # Verify successful pages were extracted
        assert "First page text" in result
        assert "Third page text" in result
        assert "Text extraction error" not in result  # Failed page should not be included
        
        # Verify warning was logged for failed page
        mock_logger.warning.assert_called_with(
            "Failed to extract text from page 2: Text extraction error"
        )
    
    @patch('src.data_acquisition.pdf_extractor.pdfplumber')
    @patch('src.data_acquisition.pdf_extractor.logger')
    def test_extract_tables_pdfplumber_per_page_error_handling(self, mock_logger, mock_pdfplumber):
        """Test per-page error handling in table extraction."""
        # Mock a multi-page PDF with some pages failing
        mock_pdf = MagicMock()
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()
        mock_page3 = MagicMock()
        
        # Page 1 succeeds with table
        table1 = [['Col1', 'Col2'], ['Data1', 'Data2']]
        mock_page1.extract_tables.return_value = [table1]
        
        # Page 2 fails
        mock_page2.extract_tables.side_effect = Exception("Table extraction failed")
        
        # Page 3 succeeds with table
        table3 = [['Col3', 'Col4'], ['Data3', 'Data4']]
        mock_page3.extract_tables.return_value = [table3]
        
        mock_pdf.pages = [mock_page1, mock_page2, mock_page3]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        # Test table extraction with per-page errors
        result = extract_tables_pdfplumber(self.test_pdf_path)
        
        # Verify successful tables were extracted
        assert len(result) == 2
        assert table1 in result
        assert table3 in result
        
        # Verify warning was logged for failed page
        mock_logger.warning.assert_called_with(
            "Failed to extract tables from page 2: Table extraction failed"
        )
    
    @patch('src.data_acquisition.pdf_extractor.fitz')
    @patch('src.data_acquisition.pdf_extractor.logger')
    def test_extract_text_pymupdf_page_range_error_handling(self, mock_logger, mock_fitz):
        """Test per-page error handling with page range specification."""
        # Mock a multi-page PDF with some pages failing
        mock_doc = MagicMock()
        mock_pages = []
        
        for i in range(5):
            mock_page = MagicMock()
            if i == 2:  # Page 3 (index 2) fails
                mock_page.get_text.side_effect = Exception(f"Page {i+1} failed")
            else:
                mock_page.get_text.return_value = f"Page {i+1} content"
            mock_pages.append(mock_page)
        
        mock_doc.__iter__.return_value = mock_pages
        mock_doc.__len__.return_value = 5
        mock_doc.page_count = 5
        mock_fitz.open.return_value = mock_doc
        
        # Test text extraction with page range (pages 2-4, 0-indexed 1-3)
        result = extract_text_pymupdf(self.test_pdf_path, start_page=1, end_page=3)
        
        # Should include pages 2 and 4, but not the failed page 3
        assert "Page 2 content" in result
        assert "Page 4 content" in result
        assert "Page 3 failed" not in result
        
        # Verify warning was logged for failed page
        mock_logger.warning.assert_called_with(
            "Failed to extract text from page 3: Page 3 failed"
        )