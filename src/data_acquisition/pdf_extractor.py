"""
PDF Text and Table Extraction Module

This module provides functionality to extract text and tables from PDF documents
using PyMuPDF (fitz) and pdfplumber libraries. It implements comprehensive error
handling and fallback mechanisms for robust PDF processing.

Key Features:
- Text extraction using PyMuPDF (primary) and pdfplumber (fallback)
- Table extraction using pdfplumber with customizable settings
- Multi-page PDF support with optional page range specification
- PDF metadata extraction and validation
- Password protection detection
- Comprehensive error handling for corrupted and invalid PDFs

Dependencies:
- PyMuPDF (fitz) for primary text extraction and metadata
- pdfplumber for table extraction and text fallback
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import fitz  # PyMuPDF
except ImportError as e:
    raise ImportError(f"PyMuPDF is required for PDF text extraction: {e}")

try:
    import pdfplumber
except ImportError as e:
    raise ImportError(f"pdfplumber is required for PDF table extraction: {e}")

# Set up logging
logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Custom exception for PDF extraction-related errors."""
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        """
        Initialize PDFExtractionError.
        
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


def validate_pdf_file(file_path: str) -> bool:
    """
    Validate that the file exists and has a PDF extension.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        bool: True if file is valid
        
    Raises:
        PDFExtractionError: If file is invalid or doesn't exist
    """
    if not file_path or not isinstance(file_path, str):
        raise PDFExtractionError("File path must be a non-empty string")
    
    file_path = file_path.strip()
    if not file_path:
        raise PDFExtractionError("File path cannot be empty or whitespace")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise PDFExtractionError(f"File not found: {file_path}")
    
    # Check if it's a file (not a directory)
    if not os.path.isfile(file_path):
        raise PDFExtractionError(f"Path is not a file: {file_path}")
    
    # Check PDF extension
    path_obj = Path(file_path)
    if path_obj.suffix.lower() != '.pdf':
        raise PDFExtractionError(f"Not a PDF file: {file_path}")
    
    return True


def get_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dict[str, Any]: Dictionary containing PDF metadata
        
    Raises:
        PDFExtractionError: If metadata extraction fails
    """
    logger.info(f"Extracting metadata from PDF: {file_path}")
    
    try:
        doc = fitz.open(file_path)
        metadata = doc.metadata.copy() if doc.metadata else {}
        metadata['page_count'] = doc.page_count
        doc.close()
        
        logger.debug(f"Successfully extracted metadata: {len(metadata)} fields")
        return metadata
        
    except Exception as e:
        error_msg = f"Failed to extract PDF metadata from {file_path}: {e}"
        logger.error(error_msg)
        raise PDFExtractionError(error_msg, e)


def is_pdf_password_protected(file_path: str) -> bool:
    """
    Check if a PDF file is password protected.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        bool: True if password protected, False otherwise
        
    Raises:
        PDFExtractionError: If password protection check fails
    """
    logger.debug(f"Checking password protection for PDF: {file_path}")
    
    try:
        doc = fitz.open(file_path)
        is_protected = doc.needs_pass or doc.is_encrypted
        doc.close()
        
        logger.debug(f"Password protection status: {is_protected}")
        return is_protected
        
    except Exception as e:
        error_msg = f"Failed to check password protection for {file_path}: {e}"
        logger.error(error_msg)
        raise PDFExtractionError(error_msg, e)


def extract_text_pymupdf(file_path: str, start_page: Optional[int] = None, 
                        end_page: Optional[int] = None) -> str:
    """
    Extract text from PDF using PyMuPDF (fitz).
    
    Args:
        file_path: Path to the PDF file
        start_page: Starting page number (0-indexed, inclusive)
        end_page: Ending page number (0-indexed, inclusive)
        
    Returns:
        str: Extracted text content
        
    Raises:
        PDFExtractionError: If text extraction fails
    """
    logger.info(f"Extracting text from PDF using PyMuPDF: {file_path}")
    
    try:
        doc = fitz.open(file_path)
        
        # Determine page range
        total_pages = doc.page_count
        start = start_page if start_page is not None else 0
        end = end_page if end_page is not None else total_pages - 1
        
        # Validate page range
        start = max(0, min(start, total_pages - 1))
        end = max(start, min(end, total_pages - 1))
        
        text_parts = []
        
        # Extract text from specified page range
        if start_page is not None or end_page is not None:
            # Page range specified - iterate through document and select pages
            pages = list(doc)  # Convert to list to handle mocking properly
            for page_num in range(start, end + 1):
                if page_num < len(pages):
                    page = pages[page_num]
                    page_text = page.get_text()
                    if page_text:  # Include empty pages in range extraction
                        text_parts.append(page_text)
        else:
            # No page range - iterate through all pages
            for page in doc:
                page_text = page.get_text()
                if page_text:  # Include all pages when no range specified
                    text_parts.append(page_text)
        
        doc.close()
        
        # Join pages with newlines
        extracted_text = '\n'.join(text_parts)
        
        logger.info(f"Successfully extracted {len(extracted_text)} characters from {len(text_parts)} pages")
        return extracted_text
        
    except Exception as e:
        error_msg = f"Failed to extract text using PyMuPDF from {file_path}: {e}"
        logger.error(error_msg)
        raise PDFExtractionError(error_msg, e)


def extract_text_pdfplumber(file_path: str) -> str:
    """
    Extract text from PDF using pdfplumber.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        str: Extracted text content
        
    Raises:
        PDFExtractionError: If text extraction fails
    """
    logger.info(f"Extracting text from PDF using pdfplumber: {file_path}")
    
    try:
        text_parts = []
        
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():  # Only add non-empty pages
                    text_parts.append(page_text)
        
        # Join pages with newlines
        extracted_text = '\n'.join(text_parts)
        
        logger.info(f"Successfully extracted {len(extracted_text)} characters from {len(text_parts)} pages")
        return extracted_text
        
    except Exception as e:
        error_msg = f"Failed to extract text using pdfplumber from {file_path}: {e}"
        logger.error(error_msg)
        raise PDFExtractionError(error_msg, e)


def extract_text_from_pdf(file_path: str, method: str = "pymupdf", 
                         use_fallback: bool = False) -> str:
    """
    Extract text from PDF with method selection and fallback support.
    
    Args:
        file_path: Path to the PDF file
        method: Extraction method ("pymupdf" or "pdfplumber")
        use_fallback: Whether to use fallback method if primary fails
        
    Returns:
        str: Extracted text content
        
    Raises:
        PDFExtractionError: If text extraction fails
    """
    logger.info(f"Extracting text from PDF: {file_path} (method: {method}, fallback: {use_fallback})")
    
    # Validate file first
    validate_pdf_file(file_path)
    
    # Try primary method
    try:
        if method.lower() == "pymupdf":
            return extract_text_pymupdf(file_path)
        elif method.lower() == "pdfplumber":
            return extract_text_pdfplumber(file_path)
        else:
            raise PDFExtractionError(f"Unsupported extraction method: {method}")
            
    except Exception as e:
        if not use_fallback:
            if isinstance(e, PDFExtractionError):
                raise e
            else:
                raise PDFExtractionError(f"Text extraction failed: {e}", e)
        
        # Try fallback method
        logger.warning(f"Primary method '{method}' failed, trying fallback")
        
        try:
            if method.lower() == "pymupdf":
                return extract_text_pdfplumber(file_path)
            else:
                return extract_text_pymupdf(file_path)
        except Exception as fallback_error:
            error_msg = f"Both primary ({method}) and fallback methods failed"
            logger.error(error_msg)
            raise PDFExtractionError(error_msg, fallback_error)


def extract_tables_pdfplumber(file_path: str, 
                             table_settings: Optional[Dict[str, Any]] = None) -> List[List[List[str]]]:
    """
    Extract tables from PDF using pdfplumber with custom settings.
    
    Args:
        file_path: Path to the PDF file
        table_settings: Optional dictionary of table detection settings
        
    Returns:
        List[List[List[str]]]: List of tables, where each table is a list of rows,
                              and each row is a list of cell values
        
    Raises:
        PDFExtractionError: If table extraction fails
    """
    logger.info(f"Extracting tables from PDF using pdfplumber: {file_path}")
    
    try:
        all_tables = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    # Extract tables with optional settings
                    if table_settings:
                        page_tables = page.extract_tables(table_settings)
                    else:
                        page_tables = page.extract_tables()
                    
                    if page_tables:
                        all_tables.extend(page_tables)
                        logger.debug(f"Found {len(page_tables)} tables on page {page_num + 1}")
                
                except Exception as page_error:
                    logger.warning(f"Failed to extract tables from page {page_num + 1}: {page_error}")
                    continue
        
        logger.info(f"Successfully extracted {len(all_tables)} tables from PDF")
        return all_tables
        
    except Exception as e:
        error_msg = f"Failed to extract tables using pdfplumber from {file_path}: {e}"
        logger.error(error_msg)
        raise PDFExtractionError(error_msg, e)


def extract_tables_from_pdf(file_path: str) -> List[List[List[str]]]:
    """
    Extract tables from PDF using pdfplumber.
    
    This is a convenience wrapper around extract_tables_pdfplumber with default settings.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List[List[List[str]]]: List of tables, where each table is a list of rows,
                              and each row is a list of cell values
        
    Raises:
        PDFExtractionError: If table extraction fails
    """
    logger.info(f"Extracting tables from PDF: {file_path}")
    
    # Validate file first
    validate_pdf_file(file_path)
    
    return extract_tables_pdfplumber(file_path)


# Module initialization
logger.info("PDF extraction module loaded successfully")