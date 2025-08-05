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
import sys

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
    
    def __init__(self, message: str, cause: Optional[Exception] = None, library: Optional[str] = None):
        """
        Initialize PDFExtractionError.
        
        Args:
            message: Error message
            cause: Optional underlying exception that caused this error
            library: Optional library name that caused the error (pymupdf, pdfplumber)
        """
        super().__init__(message)
        self.cause = cause
        self.library = library
        if cause:
            library_info = f" [{library}]" if library else ""
            self.message = f"{message}{library_info}. Caused by: {str(cause)}"
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
        # Handle PyMuPDF-specific errors
        error_msg = _handle_pymupdf_error(e, file_path, "metadata extraction")
        logger.error(error_msg)
        raise PDFExtractionError(error_msg, e, "pymupdf")


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
        # Handle PyMuPDF-specific errors
        error_msg = _handle_pymupdf_error(e, file_path, "password protection check")
        logger.error(error_msg)
        raise PDFExtractionError(error_msg, e, "pymupdf")


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
    
    doc = None
    try:
        doc = fitz.open(file_path)
        
        # Check for memory issues with large PDFs
        if doc.page_count > 1000:
            logger.warning(f"Large PDF detected ({doc.page_count} pages) - potential memory issues")
        
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
                    try:
                        page = pages[page_num]
                        page_text = page.get_text()
                        if page_text:  # Include empty pages in range extraction
                            text_parts.append(page_text)
                    except Exception as page_error:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {page_error}")
                        continue
        else:
            # No page range - iterate through all pages
            for page_num, page in enumerate(doc):
                try:
                    page_text = page.get_text()
                    if page_text:  # Include all pages when no range specified
                        text_parts.append(page_text)
                except Exception as page_error:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {page_error}")
                    continue
        
        # Join pages with newlines
        extracted_text = '\n'.join(text_parts)
        
        logger.info(f"Successfully extracted {len(extracted_text)} characters from {len(text_parts)} pages")
        return extracted_text
        
    except Exception as e:
        # Handle PyMuPDF-specific errors
        error_msg = _handle_pymupdf_error(e, file_path, "text extraction")
        logger.error(error_msg)
        raise PDFExtractionError(error_msg, e, "pymupdf")
    finally:
        # Ensure document is closed even if error occurs
        if doc is not None:
            try:
                doc.close()
            except Exception as close_error:
                logger.warning(f"Failed to close PDF document: {close_error}")


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
            # Check for memory issues with large PDFs
            if len(pdf.pages) > 1000:
                logger.warning(f"Large PDF detected ({len(pdf.pages)} pages) - potential memory issues")
            
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():  # Only add non-empty pages
                        text_parts.append(page_text)
                except Exception as page_error:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {page_error}")
                    continue
        
        # Join pages with newlines
        extracted_text = '\n'.join(text_parts)
        
        logger.info(f"Successfully extracted {len(extracted_text)} characters from {len(text_parts)} pages")
        return extracted_text
        
    except Exception as e:
        # Handle pdfplumber-specific errors
        error_msg = _handle_pdfplumber_error(e, file_path, "text extraction")
        logger.error(error_msg)
        raise PDFExtractionError(error_msg, e, "pdfplumber")


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
            # Check for memory issues with large PDFs
            if len(pdf.pages) > 1000:
                logger.warning(f"Large PDF detected ({len(pdf.pages)} pages) - potential memory issues")
            
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
        # Handle pdfplumber-specific errors
        error_msg = _handle_pdfplumber_error(e, file_path, "table extraction")
        logger.error(error_msg)
        raise PDFExtractionError(error_msg, e, "pdfplumber")


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


def _handle_pymupdf_error(error: Exception, file_path: str, operation: str) -> str:
    """
    Handle PyMuPDF-specific errors and provide informative error messages.
    
    Args:
        error: The original exception
        file_path: Path to the PDF file
        operation: Description of the operation that failed
        
    Returns:
        str: Informative error message
    """
    error_str = str(error).lower()
    error_type = type(error).__name__
    
    # Check for specific PyMuPDF errors
    if "filedataerror" in error_type.lower() or "file data error" in error_str:
        return f"PyMuPDF: Corrupted or invalid PDF file structure during {operation} - {file_path}"
    
    elif "filenotfounderror" in error_type.lower() or "file not found" in error_str:
        return f"PyMuPDF: PDF file not found during {operation} - {file_path}"
    
    elif "runtimeerror" in error_type.lower():
        if "password" in error_str or "encrypted" in error_str:
            return f"PyMuPDF: PDF is password-protected or encrypted during {operation} - {file_path}"
        elif "damaged" in error_str or "corrupt" in error_str:
            return f"PyMuPDF: PDF file is damaged or corrupted during {operation} - {file_path}"
        elif "memory" in error_str or "malloc" in error_str:
            return f"PyMuPDF: Memory allocation failed (PDF too large or insufficient memory) during {operation} - {file_path}"
        else:
            return f"PyMuPDF: Runtime error during {operation} - {file_path}: {error}"
    
    elif "memoryerror" in error_type.lower():
        return f"PyMuPDF: Out of memory error (PDF too large) during {operation} - {file_path}"
    
    elif "valueerror" in error_type.lower():
        if "invalid" in error_str or "malformed" in error_str:
            return f"PyMuPDF: Invalid or malformed PDF structure during {operation} - {file_path}"
        else:
            return f"PyMuPDF: Invalid parameter or PDF format during {operation} - {file_path}: {error}"
    
    elif "unicodedecodeerror" in error_type.lower() or "unicodeerror" in error_type.lower():
        return f"PyMuPDF: Text encoding error (corrupted character data) during {operation} - {file_path}"
    
    elif "ioerror" in error_type.lower() or "oserror" in error_type.lower():
        return f"PyMuPDF: File I/O error (permissions, disk space, or network issue) during {operation} - {file_path}"
    
    elif "attributeerror" in error_type.lower():
        return f"PyMuPDF: PDF object structure error (missing attributes) during {operation} - {file_path}"
    
    else:
        # Generic PyMuPDF error
        return f"PyMuPDF: Unexpected error during {operation} - {file_path}: {error}"


def _handle_pdfplumber_error(error: Exception, file_path: str, operation: str) -> str:
    """
    Handle pdfplumber-specific errors and provide informative error messages.
    
    Args:
        error: The original exception
        file_path: Path to the PDF file
        operation: Description of the operation that failed
        
    Returns:
        str: Informative error message
    """
    error_str = str(error).lower()
    error_type = type(error).__name__
    
    # Check for specific pdfplumber errors
    if "pdfplumbererror" in error_type.lower():
        return f"pdfplumber: PDF parsing error during {operation} - {file_path}: {error}"
    
    elif "pdfsynerror" in error_type.lower() or "pdfsyntaxerror" in error_type.lower():
        return f"pdfplumber: PDF syntax error (malformed PDF structure) during {operation} - {file_path}"
    
    elif "pdftyperror" in error_type.lower():
        return f"pdfplumber: PDF object type error (corrupted object structure) during {operation} - {file_path}"
    
    elif "pdfvalueerror" in error_type.lower():
        return f"pdfplumber: Invalid PDF value or parameter during {operation} - {file_path}: {error}"
    
    elif "pdfexception" in error_type.lower():
        return f"pdfplumber: General PDF processing error during {operation} - {file_path}: {error}"
    
    elif "memoryerror" in error_type.lower():
        return f"pdfplumber: Out of memory error (PDF too large) during {operation} - {file_path}"
    
    elif "valueerror" in error_type.lower():
        if "password" in error_str or "encrypted" in error_str:
            return f"pdfplumber: PDF is password-protected or encrypted during {operation} - {file_path}"
        elif "invalid" in error_str or "malformed" in error_str:
            return f"pdfplumber: Invalid or malformed PDF content during {operation} - {file_path}"
        else:
            return f"pdfplumber: Invalid parameter or PDF format during {operation} - {file_path}: {error}"
    
    elif "keyerror" in error_type.lower():
        return f"pdfplumber: Missing PDF object or attribute during {operation} - {file_path}: {error}"
    
    elif "attributeerror" in error_type.lower():
        return f"pdfplumber: PDF object structure error (missing methods/attributes) during {operation} - {file_path}"
    
    elif "unicodedecodeerror" in error_type.lower() or "unicodeerror" in error_type.lower():
        return f"pdfplumber: Text encoding error (corrupted character data) during {operation} - {file_path}"
    
    elif "ioerror" in error_type.lower() or "oserror" in error_type.lower():
        return f"pdfplumber: File I/O error (permissions, disk space, or network issue) during {operation} - {file_path}"
    
    elif "runtimeerror" in error_type.lower():
        if "cryptographic" in error_str or "decrypt" in error_str:
            return f"pdfplumber: Cryptographic/decryption error (password-protected PDF) during {operation} - {file_path}"
        else:
            return f"pdfplumber: Runtime error during {operation} - {file_path}: {error}"
    
    elif "recursionerror" in error_type.lower():
        return f"pdfplumber: Recursive PDF structure error (circular references) during {operation} - {file_path}"
    
    else:
        # Generic pdfplumber error
        return f"pdfplumber: Unexpected error during {operation} - {file_path}: {error}"


# Module initialization
logger.info("PDF extraction module loaded successfully")