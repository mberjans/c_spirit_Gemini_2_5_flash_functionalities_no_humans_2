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