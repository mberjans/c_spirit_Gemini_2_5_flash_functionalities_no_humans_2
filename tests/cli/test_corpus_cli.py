"""
Integration tests for CLI corpus management commands.

This module tests the command-line interface for corpus data acquisition,
including PubMed downloads, PDF extraction, and journal scraping operations.

Test Coverage:
- corpus pubmed-download --query <query> --output <dir> command
- corpus pdf-extract --input <file> --output <dir> command
- corpus journal-scrape --url <url> --output <dir> command
- Invalid arguments and error message handling
- Proper mocking of data acquisition modules
- Cleanup of temporary files and directories
"""

import pytest
import tempfile
import os
import subprocess
import sys
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open


class TestCorpusCLI:
    """Integration tests for corpus CLI commands."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.temp_files = []
        self.temp_dirs = []
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up temporary files
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def create_temp_directory(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def create_dummy_pdf_file(self):
        """Create a dummy PDF file for testing."""
        temp_file = tempfile.mktemp(suffix='.pdf')
        self.temp_files.append(temp_file)
        
        # Create a minimal PDF-like file
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
>>
endobj
xref
0 4
0000000000 65535 f 
0000000015 00000 n 
0000000074 00000 n 
0000000120 00000 n 
trailer
<<
/Size 4
/Root 1 0 R
>>
startxref
197
%%EOF"""
        
        with open(temp_file, 'wb') as f:
            f.write(pdf_content)
        
        return temp_file
    
    def run_cli_command(self, args):
        """Run CLI command and return result."""
        # Construct the command to run the CLI
        cmd = [sys.executable, '-m', 'src.cli'] + args
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.getcwd()
            )
            return result
        except subprocess.TimeoutExpired:
            pytest.fail("CLI command timed out")
        except Exception as e:
            pytest.fail(f"Failed to run CLI command: {e}")
    
    @patch('src.data_acquisition.pubmed.search_and_fetch')
    def test_corpus_pubmed_download_command_success(self, mock_search_fetch):
        """Test corpus pubmed-download command with successful execution."""
        # Setup
        output_dir = self.create_temp_directory()
        query = "plant metabolomics"
        
        # Mock the PubMed search and fetch function
        mock_xml_content = """<?xml version="1.0"?>
<PubmedArticleSet>
    <PubmedArticle>
        <MedlineCitation>
            <PMID>12345678</PMID>
            <Article>
                <ArticleTitle>Test Article Title</ArticleTitle>
                <Abstract>
                    <AbstractText>Test abstract content about plant metabolomics.</AbstractText>
                </Abstract>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
</PubmedArticleSet>"""
        mock_search_fetch.return_value = mock_xml_content
        
        # Run CLI command
        result = self.run_cli_command([
            'corpus', 'pubmed-download', 
            '--query', query, 
            '--output', output_dir
        ])
        
        # Verify command executed successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"
        
        # Verify search_and_fetch was called with correct parameters
        mock_search_fetch.assert_called_once()
        call_args = mock_search_fetch.call_args
        assert query in str(call_args)
        
        # Verify output contains success message
        output_text = result.stdout.lower()
        assert any(keyword in output_text for keyword in ['downloaded', 'success', 'completed'])
    
    @patch('src.data_acquisition.pubmed.search_and_fetch')
    def test_corpus_pubmed_download_with_max_results(self, mock_search_fetch):
        """Test corpus pubmed-download command with max results parameter."""
        # Setup
        output_dir = self.create_temp_directory()
        query = "plant metabolomics"
        max_results = 50
        
        # Mock the PubMed function
        mock_search_fetch.return_value = "<PubmedArticleSet></PubmedArticleSet>"
        
        # Run CLI command with max results
        result = self.run_cli_command([
            'corpus', 'pubmed-download',
            '--query', query,
            '--output', output_dir,
            '--max-results', str(max_results)
        ])
        
        # Verify command executed successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"
        
        # Verify search_and_fetch was called with max_results parameter
        mock_search_fetch.assert_called_once()
        call_args = mock_search_fetch.call_args
        assert max_results in call_args[1].values() or max_results in call_args[0]
    
    @patch('src.data_acquisition.pdf_extractor.extract_text_from_pdf')
    @patch('src.data_acquisition.pdf_extractor.extract_tables_from_pdf')
    def test_corpus_pdf_extract_command_success(self, mock_extract_tables, mock_extract_text):
        """Test corpus pdf-extract command with successful execution."""
        # Setup
        input_pdf = self.create_dummy_pdf_file()
        output_dir = self.create_temp_directory()
        
        # Mock the PDF extraction functions
        mock_extract_text.return_value = "Extracted text content from PDF document about plant research."
        mock_extract_tables.return_value = [
            [["Header 1", "Header 2"], ["Row 1 Col 1", "Row 1 Col 2"]],
            [["Table 2 Header", "Value"], ["Data", "123"]]
        ]
        
        # Run CLI command
        result = self.run_cli_command([
            'corpus', 'pdf-extract',
            '--input', input_pdf,
            '--output', output_dir
        ])
        
        # Verify command executed successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"
        
        # Verify extraction functions were called
        mock_extract_text.assert_called_once_with(input_pdf)
        mock_extract_tables.assert_called_once_with(input_pdf)
        
        # Verify output contains success message
        output_text = result.stdout.lower()
        assert any(keyword in output_text for keyword in ['extracted', 'success', 'completed'])
    
    @patch('src.data_acquisition.pdf_extractor.extract_text_from_pdf')
    def test_corpus_pdf_extract_text_only_mode(self, mock_extract_text):
        """Test corpus pdf-extract command with text-only extraction mode."""
        # Setup
        input_pdf = self.create_dummy_pdf_file()
        output_dir = self.create_temp_directory()
        
        # Mock the PDF text extraction function
        mock_extract_text.return_value = "Sample text content from PDF."
        
        # Run CLI command with text-only flag
        result = self.run_cli_command([
            'corpus', 'pdf-extract',
            '--input', input_pdf,
            '--output', output_dir,
            '--text-only'
        ])
        
        # Verify command executed successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"
        
        # Verify only text extraction was called
        mock_extract_text.assert_called_once_with(input_pdf)
    
    @patch('src.data_acquisition.journal_scraper.scrape_journal_metadata')
    @patch('src.data_acquisition.journal_scraper.download_journal_fulltext')
    def test_corpus_journal_scrape_command_success(self, mock_download_fulltext, mock_scrape_metadata):
        """Test corpus journal-scrape command with successful execution."""
        # Setup
        url = "https://example-journal.com/article/123"
        output_dir = self.create_temp_directory()
        
        # Mock the journal scraping functions
        mock_scrape_metadata.return_value = {
            "title": "Sample Article Title",
            "authors": ["Author 1", "Author 2"],
            "abstract": "Sample abstract content",
            "doi": "10.1000/example.doi",
            "journal": "Example Journal",
            "year": 2023
        }
        mock_download_fulltext.return_value = True
        
        # Run CLI command
        result = self.run_cli_command([
            'corpus', 'journal-scrape',
            '--url', url,
            '--output', output_dir
        ])
        
        # Verify command executed successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"
        
        # Verify scraping functions were called
        mock_scrape_metadata.assert_called_once()
        mock_download_fulltext.assert_called_once()
        
        # Check that URL was passed to the functions
        metadata_call_args = mock_scrape_metadata.call_args
        download_call_args = mock_download_fulltext.call_args
        assert url in str(metadata_call_args) or url in str(download_call_args)
        
        # Verify output contains success message
        output_text = result.stdout.lower()
        assert any(keyword in output_text for keyword in ['scraped', 'success', 'completed'])
    
    @patch('src.data_acquisition.journal_scraper.scrape_journal_metadata')
    def test_corpus_journal_scrape_metadata_only_mode(self, mock_scrape_metadata):
        """Test corpus journal-scrape command with metadata-only extraction."""
        # Setup
        url = "https://example-journal.com/article/456"
        output_dir = self.create_temp_directory()
        
        # Mock the metadata scraping function
        mock_scrape_metadata.return_value = {
            "title": "Metadata Only Article",
            "authors": ["Researcher A"],
            "abstract": "Abstract content for metadata test"
        }
        
        # Run CLI command with metadata-only flag
        result = self.run_cli_command([
            'corpus', 'journal-scrape',
            '--url', url,
            '--output', output_dir,
            '--metadata-only'
        ])
        
        # Verify command executed successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"
        
        # Verify only metadata scraping was called
        mock_scrape_metadata.assert_called_once()
    
    def test_corpus_pubmed_download_missing_query_argument(self):
        """Test corpus pubmed-download command with missing query argument."""
        output_dir = self.create_temp_directory()
        
        # Run CLI command without query argument
        result = self.run_cli_command([
            'corpus', 'pubmed-download',
            '--output', output_dir
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with missing query"
        
        # Verify error message mentions missing argument
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['query', 'required', 'missing', 'argument'])
    
    def test_corpus_pubmed_download_missing_output_argument(self):
        """Test corpus pubmed-download command with missing output argument."""
        # Run CLI command without output argument
        result = self.run_cli_command([
            'corpus', 'pubmed-download',
            '--query', 'test query'
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with missing output"
        
        # Verify error message mentions missing argument
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['output', 'required', 'missing', 'argument'])
    
    def test_corpus_pdf_extract_with_non_existent_file(self):
        """Test corpus pdf-extract command with non-existent input file."""
        non_existent_file = "/path/to/non/existent/file.pdf"
        output_dir = self.create_temp_directory()
        
        # Run CLI command with non-existent file
        result = self.run_cli_command([
            'corpus', 'pdf-extract',
            '--input', non_existent_file,
            '--output', output_dir
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with non-existent file"
        
        # Verify error message is displayed
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['not found', 'does not exist', 'error', 'file'])
    
    def test_corpus_pdf_extract_missing_input_argument(self):
        """Test corpus pdf-extract command with missing input argument."""
        output_dir = self.create_temp_directory()
        
        # Run CLI command without input argument
        result = self.run_cli_command([
            'corpus', 'pdf-extract',
            '--output', output_dir
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with missing input"
        
        # Verify error message mentions missing argument
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['input', 'required', 'missing', 'argument'])
    
    def test_corpus_journal_scrape_with_invalid_url(self):
        """Test corpus journal-scrape command with invalid URL format."""
        invalid_url = "not-a-valid-url"
        output_dir = self.create_temp_directory()
        
        # Run CLI command with invalid URL
        result = self.run_cli_command([
            'corpus', 'journal-scrape',
            '--url', invalid_url,
            '--output', output_dir
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with invalid URL"
        
        # Verify error message mentions URL format
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['invalid', 'url', 'format', 'error'])
    
    def test_corpus_journal_scrape_missing_url_argument(self):
        """Test corpus journal-scrape command with missing URL argument."""
        output_dir = self.create_temp_directory()
        
        # Run CLI command without URL argument
        result = self.run_cli_command([
            'corpus', 'journal-scrape',
            '--output', output_dir
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with missing URL"
        
        # Verify error message mentions missing argument
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['url', 'required', 'missing', 'argument'])
    
    def test_invalid_corpus_subcommand(self):
        """Test invalid corpus subcommand and ensure proper error message."""
        # Run CLI command with invalid subcommand
        result = self.run_cli_command(['corpus', 'invalid_command'])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with invalid subcommand"
        
        # Verify error message is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['invalid', 'unknown', 'command', 'usage', 'help'])
    
    def test_corpus_command_without_subcommand(self):
        """Test corpus command without any subcommand."""
        # Run CLI command without subcommand
        result = self.run_cli_command(['corpus'])
        
        # Should show help or usage information
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'commands', 'subcommand'])
        
        # Should mention available corpus commands
        assert any(keyword in output for keyword in ['pubmed', 'pdf', 'journal'])
    
    @patch('src.data_acquisition.pubmed.search_and_fetch')
    def test_corpus_pubmed_download_with_api_error(self, mock_search_fetch):
        """Test corpus pubmed-download command handling API errors."""
        # Setup
        output_dir = self.create_temp_directory()
        query = "test query"
        
        # Mock the PubMed function to raise an exception
        from src.data_acquisition.pubmed import PubMedError
        mock_search_fetch.side_effect = PubMedError("API request failed")
        
        # Run CLI command
        result = self.run_cli_command([
            'corpus', 'pubmed-download',
            '--query', query,
            '--output', output_dir
        ])
        
        # Verify command failed gracefully
        assert result.returncode != 0, "Command should have failed with API error"
        
        # Verify error message is displayed
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['error', 'failed', 'api'])
    
    @patch('src.data_acquisition.pdf_extractor.extract_text_from_pdf')
    def test_corpus_pdf_extract_with_extraction_error(self, mock_extract_text):
        """Test corpus pdf-extract command handling extraction errors."""
        # Setup
        input_pdf = self.create_dummy_pdf_file()
        output_dir = self.create_temp_directory()
        
        # Mock the PDF extraction function to raise an exception
        from src.data_acquisition.pdf_extractor import PDFExtractionError
        mock_extract_text.side_effect = PDFExtractionError("Failed to extract text")
        
        # Run CLI command
        result = self.run_cli_command([
            'corpus', 'pdf-extract',
            '--input', input_pdf,
            '--output', output_dir
        ])
        
        # Verify command failed gracefully
        assert result.returncode != 0, "Command should have failed with extraction error"
        
        # Verify error message is displayed
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['error', 'failed', 'extract'])
    
    @patch('src.data_acquisition.journal_scraper.scrape_journal_metadata')
    def test_corpus_journal_scrape_with_network_error(self, mock_scrape_metadata):
        """Test corpus journal-scrape command handling network errors."""
        # Setup
        url = "https://example-journal.com/article/123"
        output_dir = self.create_temp_directory()
        
        # Mock the scraping function to raise a network exception
        import requests
        mock_scrape_metadata.side_effect = requests.ConnectionError("Network connection failed")
        
        # Run CLI command
        result = self.run_cli_command([
            'corpus', 'journal-scrape',
            '--url', url,
            '--output', output_dir
        ])
        
        # Verify command failed gracefully
        assert result.returncode != 0, "Command should have failed with network error"
        
        # Verify error message is displayed
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['error', 'failed', 'network', 'connection'])
    
    def test_corpus_help_command(self):
        """Test corpus help command displays available options."""
        # Run corpus help command
        result = self.run_cli_command(['corpus', '--help'])
        
        # Verify help is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'commands'])
        
        # Should mention corpus subcommands
        assert any(keyword in output for keyword in ['pubmed', 'pdf', 'journal'])
    
    @patch('src.data_acquisition.pubmed.search_and_fetch')
    def test_corpus_pubmed_download_with_verbose_output(self, mock_search_fetch):
        """Test corpus pubmed-download command with verbose output."""
        # Setup
        output_dir = self.create_temp_directory()
        query = "plant metabolomics"
        
        # Mock the PubMed function
        mock_search_fetch.return_value = "<PubmedArticleSet><PubmedArticle></PubmedArticle></PubmedArticleSet>"
        
        # Run CLI command with verbose flag
        result = self.run_cli_command([
            'corpus', 'pubmed-download',
            '--query', query,
            '--output', output_dir,
            '--verbose'
        ])
        
        # Verify command executed successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"
        
        # Verify verbose output is provided
        assert len(result.stdout) > 0, "Verbose output should be provided"
        
        # Verify verbose information is included
        output_text = result.stdout.lower()
        assert any(keyword in output_text for keyword in ['processing', 'query', 'downloading'])
    
    @patch('src.data_acquisition.pdf_extractor.extract_text_from_pdf')
    def test_corpus_pdf_extract_with_output_format_options(self, mock_extract_text):
        """Test corpus pdf-extract command with different output format options."""
        # Setup
        input_pdf = self.create_dummy_pdf_file()
        output_dir = self.create_temp_directory()
        
        # Mock the PDF extraction function
        mock_extract_text.return_value = "Extracted text content"
        
        # Test different output formats
        formats = ['txt', 'json', 'xml']
        
        for fmt in formats:
            # Run CLI command with specific format
            result = self.run_cli_command([
                'corpus', 'pdf-extract',
                '--input', input_pdf,
                '--output', output_dir,
                '--format', fmt
            ])
            
            # Verify command handles the format appropriately
            assert result.returncode == 0 or "format" in result.stderr.lower()
    
    @patch('src.data_acquisition.journal_scraper.scrape_journal_metadata')
    def test_corpus_journal_scrape_with_custom_headers(self, mock_scrape_metadata):
        """Test corpus journal-scrape command with custom user agent headers."""
        # Setup
        url = "https://example-journal.com/article/789"
        output_dir = self.create_temp_directory()
        
        # Mock the scraping function
        mock_scrape_metadata.return_value = {"title": "Test Article", "authors": ["Author"]}
        
        # Run CLI command with custom user agent
        result = self.run_cli_command([
            'corpus', 'journal-scrape',
            '--url', url,
            '--output', output_dir,
            '--user-agent', 'CustomBot/1.0'
        ])
        
        # Verify command executed successfully or handled custom headers
        assert result.returncode == 0 or "user-agent" in result.stderr.lower()
    
    def test_corpus_output_directory_creation(self):
        """Test that corpus commands can create output directories if they don't exist."""
        # Create a non-existent output directory path
        base_temp_dir = self.create_temp_directory()
        output_dir = os.path.join(base_temp_dir, 'new_subdir', 'corpus_output')
        
        with patch('src.data_acquisition.pubmed.search_and_fetch') as mock_search_fetch:
            mock_search_fetch.return_value = "<PubmedArticleSet></PubmedArticleSet>"
            
            # Run CLI command with non-existent output directory
            result = self.run_cli_command([
                'corpus', 'pubmed-download',
                '--query', 'test',
                '--output', output_dir
            ])
            
            # Command should either create the directory or handle the error gracefully
            assert result.returncode == 0 or "directory" in result.stderr.lower()
    
    def test_all_corpus_commands_with_help_flag(self):
        """Test that all corpus subcommands respond to --help flag."""
        subcommands = ['pubmed-download', 'pdf-extract', 'journal-scrape']
        
        for subcommand in subcommands:
            # Run each subcommand with --help
            result = self.run_cli_command(['corpus', subcommand, '--help'])
            
            # Verify help is displayed (should not fail)
            output = (result.stderr + result.stdout).lower()
            assert any(keyword in output for keyword in ['usage', 'help', 'options', 'arguments'])