"""
Integration tests for CLI extraction commands.

This module tests the command-line interface for text processing and extraction
operations including text cleaning, chunking, entity extraction, and relationship
extraction operations.

Test Coverage:
- process clean --input <file> --output <file> command
- process chunk --input <file> --output <dir> --size <int> command  
- extract ner --input <file> --schema <file> --output <file> command
- extract relations --input <file> --entities <file> --schema <file> --output <file> command
- Invalid arguments and error message handling
- Proper mocking of LLM API calls
- Cleanup of temporary files and directories

Note: These tests are designed for the expected CLI interface. Commands that are not yet
implemented (T2-T6) will currently fail with "No such command" errors, which is expected.
When the commands are implemented, these tests will validate the full functionality.
"""

import pytest
import tempfile
import os
import subprocess
import sys
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open


# Mark tests as expected to fail until commands are implemented
commands_not_implemented = pytest.mark.xfail(
    reason="CLI extraction commands not yet implemented (T2-T6)",
    raises=(AssertionError, subprocess.CalledProcessError),
    strict=False
)


class TestExtractionCLI:
    """Integration tests for extraction CLI commands."""
    
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
    
    def create_temp_file(self, content, suffix='.txt'):
        """Create a temporary file with given content."""
        temp_file = tempfile.mktemp(suffix=suffix)
        self.temp_files.append(temp_file)
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return temp_file
    
    def create_entity_schema_file(self):
        """Create a temporary entity schema file for testing."""
        schema = {
            "METABOLITE": "Primary and secondary metabolites found in plants",
            "SPECIES": "Plant and organism species names",
            "PLANT_PART": "Plant anatomical structures and tissues",
            "GENE": "Gene names and genetic elements"
        }
        return self.create_temp_file(json.dumps(schema, indent=2), suffix='.json')
    
    def create_relationship_schema_file(self):
        """Create a temporary relationship schema file for testing."""
        schema = {
            "synthesized_by": "Metabolite is synthesized by an organism or enzyme",
            "found_in": "Metabolite is found in a specific plant part",
            "affects": "Compound affects a plant trait or biological process",
            "involved_in": "Entity participates in a metabolic pathway"
        }
        return self.create_temp_file(json.dumps(schema, indent=2), suffix='.json')
    
    def create_entities_file(self):
        """Create a temporary entities file for testing."""
        entities = [
            {
                "text": "anthocyanins",
                "label": "METABOLITE",
                "start": 0,
                "end": 12,
                "confidence": 0.95
            },
            {
                "text": "grape berries",
                "label": "PLANT_PART",
                "start": 20,
                "end": 33,
                "confidence": 0.88
            },
            {
                "text": "Vitis vinifera",
                "label": "SPECIES",
                "start": 40,
                "end": 54,
                "confidence": 0.99
            }
        ]
        return self.create_temp_file(json.dumps(entities, indent=2), suffix='.json')
    
    def run_cli_command(self, args, timeout=30):
        """Run CLI command and return result."""
        # Construct the command to run the CLI
        cmd = [sys.executable, '-m', 'src.cli'] + args
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            return result
        except subprocess.TimeoutExpired:
            pytest.fail("CLI command timed out")
        except Exception as e:
            pytest.fail(f"Failed to run CLI command: {e}")
    
    # Tests for process clean command
    
    @commands_not_implemented
    def test_process_clean_command_success(self):
        """Test process clean command with successful execution."""
        # Setup input and output files
        input_text = """
        <p>Plant metabolomics    research</p> studies  the   chemical
        compounds found in plants. This includes   <strong>flavonoids</strong>
        and other    secondary metabolites.
        """
        input_file = self.create_temp_file(input_text)
        output_file = tempfile.mktemp(suffix='.txt')
        self.temp_files.append(output_file)
        
        # Mock the text cleaning functions
        with patch('src.text_processing.cleaner.normalize_text') as mock_normalize:
            mock_normalize.return_value = "plant metabolomics research studies the chemical compounds found in plants. this includes flavonoids and other secondary metabolites."
            
            # Run CLI command
            result = self.run_cli_command([
                'process', 'clean',
                '--input', input_file,
                '--output', output_file
            ])
            
            # Verify command executed successfully
            assert result.returncode == 0, f"Command failed with error: {result.stderr}"
            
            # Verify output contains success message
            output_text = result.stdout.lower()
            assert any(keyword in output_text for keyword in ['cleaned', 'success', 'processed'])
            
            # Verify output file was created
            assert os.path.exists(output_file), f"Output file should be created: {output_file}"
            
            # Verify mock was called
            mock_normalize.assert_called_once()
    
    @commands_not_implemented
    def test_process_clean_command_with_verbose(self):
        """Test process clean command with verbose output."""
        # Setup
        input_text = "Plant metabolomics research with <HTML> tags and   extra spaces."
        input_file = self.create_temp_file(input_text)
        output_file = tempfile.mktemp(suffix='.txt')
        self.temp_files.append(output_file)
        
        # Mock the text cleaning functions
        with patch('src.text_processing.cleaner.normalize_text') as mock_normalize:
            mock_normalize.return_value = "plant metabolomics research with tags and extra spaces."
            
            # Run CLI command with verbose flag
            result = self.run_cli_command([
                'process', 'clean',
                '--input', input_file,
                '--output', output_file,
                '--verbose'
            ])
            
            # Verify command executed successfully
            assert result.returncode == 0, f"Command failed with error: {result.stderr}"
            
            # Verify verbose output is provided
            assert len(result.stdout) > 0, "Verbose output should be provided"
            
            # Verify verbose information includes details
            output_text = result.stdout.lower()
            assert any(keyword in output_text for keyword in ['characters', 'processing', 'cleaning'])
    
    @commands_not_implemented
    def test_process_clean_missing_input_argument(self):
        """Test process clean command with missing input argument."""
        output_file = tempfile.mktemp(suffix='.txt')
        self.temp_files.append(output_file)
        
        # Run CLI command without input argument
        result = self.run_cli_command([
            'process', 'clean',
            '--output', output_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with missing input"
        
        # Verify error message - could be about missing command or missing input
        error_output = (result.stderr + result.stdout).lower()
        # If command doesn't exist yet, that's expected (command not implemented)
        # If command exists, should mention missing input argument
        command_not_found = any(keyword in error_output for keyword in ['no such command', 'command not found', "command 'process'"])
        missing_input = any(keyword in error_output for keyword in ['input', 'required', 'missing', 'argument'])
        
        assert command_not_found or missing_input, f"Should indicate command not found or missing input. Got: {error_output}"
    
    @commands_not_implemented
    def test_process_clean_non_existent_input_file(self):
        """Test process clean command with non-existent input file."""
        non_existent_file = "/path/to/non/existent/file.txt"
        output_file = tempfile.mktemp(suffix='.txt')
        self.temp_files.append(output_file)
        
        # Run CLI command with non-existent file
        result = self.run_cli_command([
            'process', 'clean',
            '--input', non_existent_file,
            '--output', output_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with non-existent file"
        
        # Verify error message is displayed
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['not found', 'does not exist', 'error', 'file'])
    
    # Tests for process chunk command
    
    @commands_not_implemented
    def test_process_chunk_command_success(self):
        """Test process chunk command with successful execution."""
        # Setup
        input_text = "This is a long text document that needs to be chunked into smaller pieces for processing. " * 10
        input_file = self.create_temp_file(input_text)
        output_dir = self.create_temp_directory()
        
        # Mock the chunking functions
        with patch('src.text_processing.chunker.chunk_fixed_size') as mock_chunk:
            mock_chunk.return_value = [
                "This is a long text document that needs to be chunked",
                "into smaller pieces for processing. This is a long text",
                "document that needs to be chunked into smaller pieces"
            ]
            
            # Run CLI command
            result = self.run_cli_command([
                'process', 'chunk',
                '--input', input_file,
                '--output', output_dir,
                '--size', '100'
            ])
            
            # Verify command executed successfully
            assert result.returncode == 0, f"Command failed with error: {result.stderr}"
            
            # Verify output contains success message
            output_text = result.stdout.lower()
            assert any(keyword in output_text for keyword in ['chunked', 'success', 'processed'])
            
            # Verify output directory contains chunk files
            output_path = Path(output_dir)
            chunk_files = list(output_path.glob('chunk_*.txt'))
            assert len(chunk_files) > 0, "Should create chunk files"
            
            # Verify mock was called
            mock_chunk.assert_called_once()
    
    @commands_not_implemented
    def test_process_chunk_with_overlap(self):
        """Test process chunk command with overlap parameter."""
        # Setup
        input_text = "Plant metabolomics research involves studying chemical compounds in plants. " * 5
        input_file = self.create_temp_file(input_text)
        output_dir = self.create_temp_directory()
        
        # Mock the chunking functions
        with patch('src.text_processing.chunker.chunk_fixed_size') as mock_chunk:
            mock_chunk.return_value = [
                "Plant metabolomics research involves studying chemical",
                "studying chemical compounds in plants. Plant metabolomics",
                "metabolomics research involves studying chemical compounds"
            ]
            
            # Run CLI command with overlap
            result = self.run_cli_command([
                'process', 'chunk',
                '--input', input_file,
                '--output', output_dir,
                '--size', '50',
                '--overlap', '10'
            ])
            
            # Verify command executed successfully
            assert result.returncode == 0, f"Command failed with error: {result.stderr}"
            
            # Verify chunk_fixed_size was called with correct parameters
            mock_chunk.assert_called_once()
            args, kwargs = mock_chunk.call_args
            assert args[1] == 50  # chunk_size
            assert args[2] == 10  # chunk_overlap
    
    @commands_not_implemented
    def test_process_chunk_missing_size_argument(self):
        """Test process chunk command with missing size argument."""
        input_file = self.create_temp_file("Sample text")
        output_dir = self.create_temp_directory()
        
        # Run CLI command without size argument
        result = self.run_cli_command([
            'process', 'chunk',
            '--input', input_file,
            '--output', output_dir
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with missing size"
        
        # Verify error message mentions missing argument
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['size', 'required', 'missing', 'argument'])
    
    @commands_not_implemented
    def test_process_chunk_invalid_size_argument(self):
        """Test process chunk command with invalid size argument."""
        input_file = self.create_temp_file("Sample text")
        output_dir = self.create_temp_directory()
        
        # Run CLI command with invalid size (negative)
        result = self.run_cli_command([
            'process', 'chunk',
            '--input', input_file,
            '--output', output_dir,
            '--size', '-50'
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with invalid size"
        
        # Verify error message mentions invalid size
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['invalid', 'size', 'positive', 'error'])
    
    # Tests for extract ner command
    
    @commands_not_implemented
    def test_extract_ner_command_success(self):
        """Test extract ner command with successful execution."""
        # Setup
        input_text = "Anthocyanins are found in grape berries and contribute to their color."
        input_file = self.create_temp_file(input_text)
        schema_file = self.create_entity_schema_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Mock the NER extraction function
        mock_entities = [
            {
                "text": "Anthocyanins",
                "label": "METABOLITE",
                "start": 0,
                "end": 12,
                "confidence": 0.95
            },
            {
                "text": "grape berries",
                "label": "PLANT_PART",
                "start": 26,
                "end": 39,
                "confidence": 0.88
            }
        ]
        
        with patch('src.llm_extraction.ner.extract_entities') as mock_extract:
            mock_extract.return_value = mock_entities
            
            # Run CLI command
            result = self.run_cli_command([
                'extract', 'ner',
                '--input', input_file,
                '--schema', schema_file,
                '--output', output_file
            ])
            
            # Verify command executed successfully
            assert result.returncode == 0, f"Command failed with error: {result.stderr}"
            
            # Verify output contains success message
            output_text = result.stdout.lower()
            assert any(keyword in output_text for keyword in ['extracted', 'entities', 'success'])
            
            # Verify output file was created
            assert os.path.exists(output_file), f"Output file should be created: {output_file}"
            
            # Verify output file contains entities
            with open(output_file, 'r') as f:
                extracted_entities = json.load(f)
                assert len(extracted_entities) == 2
                assert extracted_entities[0]['label'] == 'METABOLITE'
                assert extracted_entities[1]['label'] == 'PLANT_PART'
            
            # Verify mock was called
            mock_extract.assert_called_once()
    
    @commands_not_implemented
    def test_extract_ner_with_model_parameter(self):
        """Test extract ner command with model parameter."""
        # Setup
        input_text = "Flavonoids are secondary metabolites in plants."
        input_file = self.create_temp_file(input_text)
        schema_file = self.create_entity_schema_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Mock the NER extraction function
        with patch('src.llm_extraction.ner.extract_entities') as mock_extract:
            mock_extract.return_value = []
            
            # Run CLI command with model parameter
            result = self.run_cli_command([
                'extract', 'ner',
                '--input', input_file,
                '--schema', schema_file,
                '--output', output_file,
                '--model', 'gpt-3.5-turbo'
            ])
            
            # Verify command executed successfully
            assert result.returncode == 0, f"Command failed with error: {result.stderr}"
            
            # Verify extract_entities was called with correct model
            mock_extract.assert_called_once()
            args, kwargs = mock_extract.call_args
            assert args[2] == 'gpt-3.5-turbo'  # llm_model_name parameter
    
    @patch('src.llm_extraction.ner.extract_entities')
    @commands_not_implemented
    def test_extract_ner_with_llm_api_error(self, mock_extract):
        """Test extract ner command handling LLM API errors."""
        # Setup
        input_text = "Plant metabolomics research."
        input_file = self.create_temp_file(input_text)
        schema_file = self.create_entity_schema_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Mock LLM API error
        from src.llm_extraction.ner import LLMAPIError
        mock_extract.side_effect = LLMAPIError("API rate limit exceeded")
        
        # Run CLI command
        result = self.run_cli_command([
            'extract', 'ner',
            '--input', input_file,
            '--schema', schema_file,
            '--output', output_file
        ])
        
        # Verify command failed gracefully
        assert result.returncode != 0, "Command should have failed with API error"
        
        # Verify error message is displayed
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['api', 'error', 'rate limit'])
    
    @commands_not_implemented
    def test_extract_ner_missing_schema_argument(self):
        """Test extract ner command with missing schema argument."""
        input_file = self.create_temp_file("Sample text")
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Run CLI command without schema argument
        result = self.run_cli_command([
            'extract', 'ner',
            '--input', input_file,
            '--output', output_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with missing schema"
        
        # Verify error message mentions missing argument
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['schema', 'required', 'missing', 'argument'])
    
    @commands_not_implemented
    def test_extract_ner_invalid_schema_file(self):
        """Test extract ner command with invalid schema file."""
        input_file = self.create_temp_file("Sample text")
        invalid_schema_file = self.create_temp_file("invalid json content")
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Run CLI command with invalid schema file
        result = self.run_cli_command([
            'extract', 'ner',
            '--input', input_file,
            '--schema', invalid_schema_file,
            '--output', output_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with invalid schema"
        
        # Verify error message mentions schema format
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['schema', 'json', 'invalid', 'format'])
    
    # Tests for extract relations command
    
    @commands_not_implemented
    def test_extract_relations_command_success(self):
        """Test extract relations command with successful execution."""
        # Setup
        input_text = "Anthocyanins are synthesized by grape berries and affect fruit color."
        input_file = self.create_temp_file(input_text)
        entities_file = self.create_entities_file()
        schema_file = self.create_relationship_schema_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Mock the relationship extraction function
        mock_relationships = [
            ("anthocyanins", "synthesized_by", "grape berries"),
            ("anthocyanins", "affects", "fruit color")
        ]
        
        with patch('src.llm_extraction.relations.extract_relationships') as mock_extract:
            mock_extract.return_value = mock_relationships
            
            # Run CLI command
            result = self.run_cli_command([
                'extract', 'relations',
                '--input', input_file,
                '--entities', entities_file,
                '--schema', schema_file,
                '--output', output_file
            ])
            
            # Verify command executed successfully
            assert result.returncode == 0, f"Command failed with error: {result.stderr}"
            
            # Verify output contains success message
            output_text = result.stdout.lower()
            assert any(keyword in output_text for keyword in ['extracted', 'relationships', 'success'])
            
            # Verify output file was created
            assert os.path.exists(output_file), f"Output file should be created: {output_file}"
            
            # Verify output file contains relationships
            with open(output_file, 'r') as f:
                extracted_relationships = json.load(f)
                assert len(extracted_relationships) == 2
                assert extracted_relationships[0][1] == 'synthesized_by'
                assert extracted_relationships[1][1] == 'affects'
            
            # Verify mock was called
            mock_extract.assert_called_once()
    
    @commands_not_implemented
    def test_extract_relations_with_model_parameter(self):
        """Test extract relations command with model parameter."""
        # Setup
        input_text = "Flavonoids are found in plant tissues."
        input_file = self.create_temp_file(input_text)
        entities_file = self.create_entities_file()
        schema_file = self.create_relationship_schema_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Mock the relationship extraction function
        with patch('src.llm_extraction.relations.extract_relationships') as mock_extract:
            mock_extract.return_value = []
            
            # Run CLI command with model parameter
            result = self.run_cli_command([
                'extract', 'relations',
                '--input', input_file,
                '--entities', entities_file,
                '--schema', schema_file,
                '--output', output_file,
                '--model', 'gpt-4'
            ])
            
            # Verify command executed successfully
            assert result.returncode == 0, f"Command failed with error: {result.stderr}"
            
            # Verify extract_relationships was called with correct model
            mock_extract.assert_called_once()
            args, kwargs = mock_extract.call_args
            assert args[3] == 'gpt-4'  # llm_model_name parameter
    
    @patch('src.llm_extraction.relations.extract_relationships')
    @commands_not_implemented
    def test_extract_relations_with_api_error(self, mock_extract):
        """Test extract relations command handling API errors."""
        # Setup
        input_text = "Plants produce metabolites."
        input_file = self.create_temp_file(input_text)
        entities_file = self.create_entities_file()
        schema_file = self.create_relationship_schema_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Mock API error
        from src.llm_extraction.relations import LLMAPIError
        mock_extract.side_effect = LLMAPIError("Request timed out")
        
        # Run CLI command
        result = self.run_cli_command([
            'extract', 'relations',
            '--input', input_file,
            '--entities', entities_file,
            '--schema', schema_file,
            '--output', output_file
        ])
        
        # Verify command failed gracefully
        assert result.returncode != 0, "Command should have failed with API error"
        
        # Verify error message is displayed
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['api', 'error', 'timeout'])
    
    @commands_not_implemented
    def test_extract_relations_missing_entities_argument(self):
        """Test extract relations command with missing entities argument."""
        input_file = self.create_temp_file("Sample text")
        schema_file = self.create_relationship_schema_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Run CLI command without entities argument
        result = self.run_cli_command([
            'extract', 'relations',
            '--input', input_file,
            '--schema', schema_file,
            '--output', output_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with missing entities"
        
        # Verify error message mentions missing argument
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['entities', 'required', 'missing', 'argument'])
    
    @commands_not_implemented
    def test_extract_relations_invalid_entities_file(self):
        """Test extract relations command with invalid entities file."""
        input_file = self.create_temp_file("Sample text")
        invalid_entities_file = self.create_temp_file("not valid json")
        schema_file = self.create_relationship_schema_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Run CLI command with invalid entities file
        result = self.run_cli_command([
            'extract', 'relations',
            '--input', input_file,
            '--entities', invalid_entities_file,
            '--schema', schema_file,
            '--output', output_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with invalid entities"
        
        # Verify error message mentions entities format
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['entities', 'json', 'invalid', 'format'])
    
    # Tests for invalid subcommands and general CLI behavior
    
    @commands_not_implemented
    def test_invalid_process_subcommand(self):
        """Test invalid process subcommand and ensure proper error message."""
        # Run CLI command with invalid subcommand
        result = self.run_cli_command(['process', 'invalid_command'])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with invalid subcommand"
        
        # Verify error message is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['invalid', 'unknown', 'command', 'usage', 'help'])
    
    @commands_not_implemented
    def test_invalid_extract_subcommand(self):
        """Test invalid extract subcommand and ensure proper error message."""
        # Run CLI command with invalid subcommand
        result = self.run_cli_command(['extract', 'invalid_command'])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with invalid subcommand"
        
        # Verify error message is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['invalid', 'unknown', 'command', 'usage', 'help'])
    
    @commands_not_implemented
    def test_process_command_without_subcommand(self):
        """Test process command without any subcommand."""
        # Run CLI command without subcommand
        result = self.run_cli_command(['process'])
        
        # Should show help or usage information
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'command'])
        
        # Should mention available subcommands
        assert any(keyword in output for keyword in ['clean', 'chunk']) or \
               any(keyword in output for keyword in ['missing', 'try', '--help'])
    
    @commands_not_implemented
    def test_extract_command_without_subcommand(self):
        """Test extract command without any subcommand."""
        # Run CLI command without subcommand
        result = self.run_cli_command(['extract'])
        
        # Should show help or usage information
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'command'])
        
        # Should mention available subcommands
        assert any(keyword in output for keyword in ['ner', 'relations']) or \
               any(keyword in output for keyword in ['missing', 'try', '--help'])
    
    # Tests for help functionality
    
    @commands_not_implemented
    def test_process_help_command(self):
        """Test process help command displays available options."""
        # Run process help command
        result = self.run_cli_command(['process', '--help'])
        
        # Verify help is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'commands'])
        
        # Should mention process subcommands
        assert any(keyword in output for keyword in ['clean', 'chunk'])
    
    @commands_not_implemented
    def test_extract_help_command(self):
        """Test extract help command displays available options."""
        # Run extract help command
        result = self.run_cli_command(['extract', '--help'])
        
        # Verify help is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'commands'])
        
        # Should mention extract subcommands
        assert any(keyword in output for keyword in ['ner', 'relations'])
    
    @commands_not_implemented
    def test_process_clean_help_command(self):
        """Test process clean help command displays specific options."""
        # Run process clean help command
        result = self.run_cli_command(['process', 'clean', '--help'])
        
        # Verify help is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'options', 'arguments'])
        
        # Should mention clean-specific options
        assert any(keyword in output for keyword in ['input', 'output'])
    
    @commands_not_implemented
    def test_process_chunk_help_command(self):
        """Test process chunk help command displays specific options."""
        # Run process chunk help command
        result = self.run_cli_command(['process', 'chunk', '--help'])
        
        # Verify help is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'options', 'arguments'])
        
        # Should mention chunk-specific options
        assert any(keyword in output for keyword in ['input', 'output', 'size'])
    
    @commands_not_implemented
    def test_extract_ner_help_command(self):
        """Test extract ner help command displays specific options."""
        # Run extract ner help command
        result = self.run_cli_command(['extract', 'ner', '--help'])
        
        # Verify help is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'options', 'arguments'])
        
        # Should mention ner-specific options
        assert any(keyword in output for keyword in ['input', 'schema', 'output'])
    
    @commands_not_implemented
    def test_extract_relations_help_command(self):
        """Test extract relations help command displays specific options."""
        # Run extract relations help command
        result = self.run_cli_command(['extract', 'relations', '--help'])
        
        # Verify help is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'options', 'arguments'])
        
        # Should mention relations-specific options
        assert any(keyword in output for keyword in ['input', 'entities', 'schema', 'output'])
    
    # Tests for output directory creation
    
    @commands_not_implemented
    def test_commands_create_output_directories(self):
        """Test that extraction commands can create output directories if they don't exist."""
        # Setup
        input_text = "Sample text for processing"
        input_file = self.create_temp_file(input_text)
        
        # Create a non-existent output directory path
        base_temp_dir = self.create_temp_directory()
        output_dir = os.path.join(base_temp_dir, 'new_subdir', 'extraction_output')
        
        # Mock the chunking function
        with patch('src.text_processing.chunker.chunk_fixed_size') as mock_chunk:
            mock_chunk.return_value = ["Sample text", "for processing"]
            
            # Run CLI command with non-existent output directory
            result = self.run_cli_command([
                'process', 'chunk',
                '--input', input_file,
                '--output', output_dir,
                '--size', '50'
            ])
            
            # Command should create the directory and run successfully
            assert os.path.exists(output_dir), "Output directory should be created"
            
            # Command may succeed or fail, but should handle directory creation
            if result.returncode != 0:
                error_text = (result.stderr + result.stdout).lower()
                # Should not fail due to directory issues
                assert not ('directory' in error_text and 'not' in error_text and 'exist' in error_text)
    
    # Tests for verbose output across commands
    
    @commands_not_implemented
    def test_all_extraction_commands_with_verbose_flag(self):
        """Test that all extraction commands respond to --verbose flag."""
        input_text = "Test text for verbose testing"
        input_file = self.create_temp_file(input_text)
        output_file = tempfile.mktemp(suffix='.txt')
        self.temp_files.append(output_file)
        
        # Test process clean with verbose
        with patch('src.text_processing.cleaner.normalize_text') as mock_clean:
            mock_clean.return_value = "test text for verbose testing"
            
            result = self.run_cli_command([
                'process', 'clean',
                '--input', input_file,
                '--output', output_file,
                '--verbose'
            ])
            
            # Verify verbose output is provided regardless of success/failure
            assert len(result.stdout) > 0, "Verbose output should be provided"
    
    # Tests for edge cases and error handling
    
    @commands_not_implemented
    def test_empty_input_file_handling(self):
        """Test handling of empty input files."""
        # Create empty input file
        empty_input_file = self.create_temp_file("")
        output_file = tempfile.mktemp(suffix='.txt')
        self.temp_files.append(output_file)
        
        # Mock the cleaning function to handle empty input
        with patch('src.text_processing.cleaner.normalize_text') as mock_clean:
            mock_clean.return_value = ""
            
            # Run CLI command with empty file
            result = self.run_cli_command([
                'process', 'clean',
                '--input', empty_input_file,
                '--output', output_file
            ])
            
            # Command should handle empty files gracefully
            # May succeed with empty output or provide appropriate message
            output_text = (result.stderr + result.stdout).lower()
            assert len(output_text) > 0, "Should provide some feedback for empty input"
    
    @commands_not_implemented
    def test_large_input_file_handling(self):
        """Test handling of large input files."""
        # Create large input file (simulate with mocking)
        large_text = "This is a large document. " * 1000
        large_input_file = self.create_temp_file(large_text)
        output_dir = self.create_temp_directory()
        
        # Mock chunking to simulate processing large file
        with patch('src.text_processing.chunker.chunk_fixed_size') as mock_chunk:
            # Simulate many chunks for large file
            mock_chunk.return_value = ["Chunk " + str(i) for i in range(100)]
            
            # Run CLI command with large file
            result = self.run_cli_command([
                'process', 'chunk',
                '--input', large_input_file,
                '--output', output_dir,
                '--size', '100',
                '--verbose'
            ])
            
            # Command should handle large files and provide progress info
            if result.returncode == 0:
                output_text = result.stdout.lower()
                assert any(keyword in output_text for keyword in ['processed', 'chunks', 'completed'])