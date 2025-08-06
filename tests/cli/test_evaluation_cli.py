"""
Integration tests for CLI evaluation commands.

This module tests the command-line interface for evaluation operations including
benchmarking LLM performance against gold standard datasets and curation of
LLM-generated extractions.

Test Coverage:
- eval benchmark --gold <file> --predicted <file> command
- eval curate --input <file> --output <file> command
- Invalid arguments and error message handling
- Proper mocking of evaluation modules
- Cleanup of temporary files and directories

The tests verify that the CLI properly integrates with the evaluation modules
(benchmarker.py and curation_tool.py) without requiring actual LLM calls or
external dependencies.

Author: AIM2-ODIE System
Date: 2025-08-06
"""

import pytest
import tempfile
import os
import subprocess
import sys
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestEvaluationCLI:
    """Integration tests for evaluation CLI commands."""
    
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
    
    def create_temp_file(self, content, suffix='.json'):
        """Create a temporary file with given content."""
        temp_file = tempfile.mktemp(suffix=suffix)
        self.temp_files.append(temp_file)
        
        if content:
            with open(temp_file, 'w', encoding='utf-8') as f:
                if isinstance(content, str):
                    f.write(content)
                else:
                    json.dump(content, f, indent=2)
        
        return temp_file
    
    def create_gold_standard_file(self):
        """Create a temporary gold standard file for benchmarking."""
        gold_data = [
            {
                "text": "Quercetin is a flavonoid found in Arabidopsis thaliana leaves.",
                "entities": [
                    {
                        "entity_type": "COMPOUND",
                        "text": "Quercetin",
                        "start_char": 0,
                        "end_char": 9
                    },
                    {
                        "entity_type": "COMPOUND",
                        "text": "flavonoid",
                        "start_char": 15,
                        "end_char": 24
                    },
                    {
                        "entity_type": "SPECIES",
                        "text": "Arabidopsis thaliana",
                        "start_char": 34,
                        "end_char": 54
                    },
                    {
                        "entity_type": "PLANT_PART",
                        "text": "leaves",
                        "start_char": 55,
                        "end_char": 61
                    }
                ],
                "relations": [
                    ["Quercetin", "found_in", "Arabidopsis thaliana"],
                    ["Quercetin", "located_in", "leaves"]
                ]
            }
        ]
        return self.create_temp_file(gold_data)
    
    def create_predicted_file(self):
        """Create a temporary predicted results file for benchmarking."""
        predicted_data = [
            {
                "text": "Quercetin is a flavonoid found in Arabidopsis thaliana leaves.",
                "entities": [
                    {
                        "entity_type": "COMPOUND",
                        "text": "Quercetin",
                        "start_char": 0,
                        "end_char": 9
                    },
                    {
                        "entity_type": "SPECIES",
                        "text": "Arabidopsis thaliana",
                        "start_char": 34,
                        "end_char": 54
                    },
                    {
                        "entity_type": "PLANT_PART",
                        "text": "leaves",
                        "start_char": 55,
                        "end_char": 61
                    }
                ],
                "relations": [
                    ["Quercetin", "found_in", "Arabidopsis thaliana"]
                ]
            }
        ]
        return self.create_temp_file(predicted_data)
    
    def create_llm_extraction_file(self):
        """Create a temporary LLM extraction file for curation."""
        llm_data = {
            "text": "Anthocyanins are pigments produced by grape berries during ripening.",
            "entities": [
                {
                    "entity_type": "COMPOUND",
                    "text": "Anthocyanins",
                    "start_char": 0,
                    "end_char": 12
                },
                {
                    "entity_type": "COMPOUND",
                    "text": "pigments",
                    "start_char": 17,
                    "end_char": 25
                },
                {
                    "entity_type": "PLANT_PART",
                    "text": "grape berries",
                    "start_char": 38,
                    "end_char": 51
                }
            ],
            "relations": [
                ["Anthocyanins", "produced_by", "grape berries"],
                ["pigments", "produced_by", "grape berries"]
            ]
        }
        return self.create_temp_file(llm_data)
    
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
    
    # Tests for eval benchmark command
    
    @patch('src.evaluation.benchmarker.calculate_ner_metrics')
    @patch('src.evaluation.benchmarker.calculate_relation_metrics')
    def test_eval_benchmark_command_success(self, mock_relation_metrics, mock_ner_metrics):
        """Test eval benchmark command with successful execution."""
        # Setup mock return values
        mock_ner_metrics.return_value = {
            'precision': 0.85,
            'recall': 0.80,
            'f1': 0.82
        }
        mock_relation_metrics.return_value = {
            'precision': 0.75,
            'recall': 0.70,
            'f1': 0.72
        }
        
        # Setup input files
        gold_file = self.create_gold_standard_file()
        predicted_file = self.create_predicted_file()
        
        # Run CLI command
        result = self.run_cli_command([
            'eval', 'benchmark',
            '--gold', gold_file,
            '--predicted', predicted_file
        ])
        
        # Verify command executed successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"
        
        # Verify output contains evaluation results
        output_text = result.stdout.lower()
        assert any(keyword in output_text for keyword in ['benchmark', 'evaluation', 'metrics'])
        assert any(keyword in output_text for keyword in ['precision', 'recall', 'f1'])
        
        # Verify metrics functions were called
        mock_ner_metrics.assert_called_once()
        mock_relation_metrics.assert_called_once()
    
    @patch('src.evaluation.benchmarker.calculate_ner_metrics')
    @patch('src.evaluation.benchmarker.calculate_relation_metrics')
    def test_eval_benchmark_with_output_file(self, mock_relation_metrics, mock_ner_metrics):
        """Test eval benchmark command with output file specification."""
        # Setup mock return values
        mock_ner_metrics.return_value = {'precision': 0.9, 'recall': 0.8, 'f1': 0.85}
        mock_relation_metrics.return_value = {'precision': 0.8, 'recall': 0.7, 'f1': 0.75}
        
        # Setup input files
        gold_file = self.create_gold_standard_file()
        predicted_file = self.create_predicted_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Run CLI command with output file
        result = self.run_cli_command([
            'eval', 'benchmark',
            '--gold', gold_file,
            '--predicted', predicted_file,
            '--output', output_file
        ])
        
        # Verify command executed successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"
        
        # Verify output file was created
        assert os.path.exists(output_file), f"Output file should be created: {output_file}"
        
        # Verify output file contains results
        with open(output_file, 'r') as f:
            results = json.load(f)
            assert 'ner_metrics' in results
            assert 'relation_metrics' in results
            assert results['ner_metrics']['f1'] == 0.85
            assert results['relation_metrics']['f1'] == 0.75
    
    @patch('src.evaluation.benchmarker.calculate_ner_metrics')
    def test_eval_benchmark_with_verbose_flag(self, mock_ner_metrics):
        """Test eval benchmark command with verbose output."""
        # Setup mock return value
        mock_ner_metrics.return_value = {'precision': 0.9, 'recall': 0.8, 'f1': 0.85}
        
        # Setup input files
        gold_file = self.create_gold_standard_file()
        predicted_file = self.create_predicted_file()
        
        # Run CLI command with verbose flag
        result = self.run_cli_command([
            'eval', 'benchmark',
            '--gold', gold_file,
            '--predicted', predicted_file,
            '--verbose'
        ])
        
        # Verify command executed successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"
        
        # Verify verbose output is provided
        assert len(result.stdout) > 0, "Verbose output should be provided"
        
        # Verify verbose information includes processing details
        output_text = result.stdout.lower()
        assert any(keyword in output_text for keyword in ['processing', 'loading', 'calculating'])
    
    def test_eval_benchmark_missing_gold_argument(self):
        """Test eval benchmark command with missing gold argument."""
        predicted_file = self.create_predicted_file()
        
        # Run CLI command without gold argument
        result = self.run_cli_command([
            'eval', 'benchmark',
            '--predicted', predicted_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with missing gold argument"
        
        # Verify error message mentions missing argument
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['gold', 'required', 'missing', 'argument'])
    
    def test_eval_benchmark_missing_predicted_argument(self):
        """Test eval benchmark command with missing predicted argument."""
        gold_file = self.create_gold_standard_file()
        
        # Run CLI command without predicted argument
        result = self.run_cli_command([
            'eval', 'benchmark',
            '--gold', gold_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with missing predicted argument"
        
        # Verify error message mentions missing argument
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['predicted', 'required', 'missing', 'argument'])
    
    def test_eval_benchmark_non_existent_gold_file(self):
        """Test eval benchmark command with non-existent gold file."""
        non_existent_file = "/path/to/non/existent/gold.json"
        predicted_file = self.create_predicted_file()
        
        # Run CLI command with non-existent gold file
        result = self.run_cli_command([
            'eval', 'benchmark',
            '--gold', non_existent_file,
            '--predicted', predicted_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with non-existent gold file"
        
        # Verify error message mentions file not found
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['not found', 'does not exist', 'file', 'error'])
    
    def test_eval_benchmark_non_existent_predicted_file(self):
        """Test eval benchmark command with non-existent predicted file."""
        gold_file = self.create_gold_standard_file()
        non_existent_file = "/path/to/non/existent/predicted.json"
        
        # Run CLI command with non-existent predicted file
        result = self.run_cli_command([
            'eval', 'benchmark',
            '--gold', gold_file,
            '--predicted', non_existent_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with non-existent predicted file"
        
        # Verify error message mentions file not found
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['not found', 'does not exist', 'file', 'error'])
    
    def test_eval_benchmark_invalid_gold_file_format(self):
        """Test eval benchmark command with invalid gold file format."""
        invalid_gold_file = self.create_temp_file("invalid json content")
        predicted_file = self.create_predicted_file()
        
        # Run CLI command with invalid gold file
        result = self.run_cli_command([
            'eval', 'benchmark',
            '--gold', invalid_gold_file,
            '--predicted', predicted_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with invalid gold file format"
        
        # Verify error message mentions format issue
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['json', 'format', 'invalid', 'parse'])
    
    def test_eval_benchmark_invalid_predicted_file_format(self):
        """Test eval benchmark command with invalid predicted file format."""
        gold_file = self.create_gold_standard_file()
        invalid_predicted_file = self.create_temp_file("not valid json")
        
        # Run CLI command with invalid predicted file
        result = self.run_cli_command([
            'eval', 'benchmark',
            '--gold', gold_file,
            '--predicted', invalid_predicted_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with invalid predicted file format"
        
        # Verify error message mentions format issue
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['json', 'format', 'invalid', 'parse'])
    
    # Tests for eval curate command
    
    @patch('src.evaluation.curation_tool.load_llm_output')
    @patch('src.evaluation.curation_tool.display_for_review')
    @patch('src.evaluation.curation_tool.save_curated_output')
    def test_eval_curate_command_success(self, mock_save, mock_display, mock_load):
        """Test eval curate command with successful execution."""
        # Setup mock return values
        mock_llm_data = {
            'text': 'Sample text for curation',
            'entities': [{'entity_type': 'COMPOUND', 'text': 'test', 'start_char': 0, 'end_char': 4}],
            'relations': [('test', 'relation', 'object')]
        }
        mock_load.return_value = mock_llm_data
        mock_save.return_value = True
        
        # Setup input and output files
        input_file = self.create_llm_extraction_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Run CLI command
        result = self.run_cli_command([
            'eval', 'curate',
            '--input', input_file,
            '--output', output_file
        ])
        
        # Verify command executed successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"
        
        # Verify output contains success message
        output_text = result.stdout.lower()
        assert any(keyword in output_text for keyword in ['curate', 'review', 'success', 'completed'])
        
        # Verify curation functions were called
        mock_load.assert_called_once_with(input_file)
        mock_display.assert_called_once()
        mock_save.assert_called_once()
    
    @patch('src.evaluation.curation_tool.load_llm_output')
    @patch('src.evaluation.curation_tool.display_for_review')
    def test_eval_curate_with_verbose_flag(self, mock_display, mock_load):
        """Test eval curate command with verbose output."""
        # Setup mock return value
        mock_llm_data = {
            'text': 'Sample text',
            'entities': [],
            'relations': []
        }
        mock_load.return_value = mock_llm_data
        
        # Setup input and output files
        input_file = self.create_llm_extraction_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Run CLI command with verbose flag
        result = self.run_cli_command([
            'eval', 'curate',
            '--input', input_file,
            '--output', output_file,
            '--verbose'
        ])
        
        # Verify command executed successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"
        
        # Verify verbose output is provided
        assert len(result.stdout) > 0, "Verbose output should be provided"
        
        # Verify verbose information includes processing details
        output_text = result.stdout.lower()
        assert any(keyword in output_text for keyword in ['loading', 'processing', 'displaying'])
    
    def test_eval_curate_missing_input_argument(self):
        """Test eval curate command with missing input argument."""
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Run CLI command without input argument
        result = self.run_cli_command([
            'eval', 'curate',
            '--output', output_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with missing input argument"
        
        # Verify error message mentions missing argument
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['input', 'required', 'missing', 'argument'])
    
    def test_eval_curate_missing_output_argument(self):
        """Test eval curate command with missing output argument."""
        input_file = self.create_llm_extraction_file()
        
        # Run CLI command without output argument
        result = self.run_cli_command([
            'eval', 'curate',
            '--input', input_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with missing output argument"
        
        # Verify error message mentions missing argument
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['output', 'required', 'missing', 'argument'])
    
    def test_eval_curate_non_existent_input_file(self):
        """Test eval curate command with non-existent input file."""
        non_existent_file = "/path/to/non/existent/input.json"
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Run CLI command with non-existent input file
        result = self.run_cli_command([
            'eval', 'curate',
            '--input', non_existent_file,
            '--output', output_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with non-existent input file"
        
        # Verify error message mentions file not found
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['not found', 'does not exist', 'file', 'error'])
    
    def test_eval_curate_invalid_input_file_format(self):
        """Test eval curate command with invalid input file format."""
        invalid_input_file = self.create_temp_file("invalid json content")
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Run CLI command with invalid input file
        result = self.run_cli_command([
            'eval', 'curate',
            '--input', invalid_input_file,
            '--output', output_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with invalid input file format"
        
        # Verify error message mentions format issue
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['json', 'format', 'invalid', 'parse'])
    
    @patch('src.evaluation.curation_tool.load_llm_output')
    def test_eval_curate_with_curation_tool_error(self, mock_load):
        """Test eval curate command handling curation tool errors."""
        # Setup mock to raise error
        from src.evaluation.curation_tool import CurationError
        mock_load.side_effect = CurationError("Invalid extraction format")
        
        # Setup files
        input_file = self.create_llm_extraction_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Run CLI command
        result = self.run_cli_command([
            'eval', 'curate',
            '--input', input_file,
            '--output', output_file
        ])
        
        # Verify command failed gracefully
        assert result.returncode != 0, "Command should have failed with curation tool error"
        
        # Verify error message is displayed
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['curation', 'error', 'invalid', 'format'])
    
    # Tests for invalid subcommands and general CLI behavior
    
    def test_invalid_eval_subcommand(self):
        """Test invalid eval subcommand and ensure proper error message."""
        # Run CLI command with invalid subcommand
        result = self.run_cli_command(['eval', 'invalid_command'])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with invalid subcommand"
        
        # Verify error message is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['invalid', 'unknown', 'command', 'usage', 'help'])
    
    def test_eval_command_without_subcommand(self):
        """Test eval command without any subcommand."""
        # Run CLI command without subcommand
        result = self.run_cli_command(['eval'])
        
        # Should show help or usage information
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'command'])
        
        # Should mention available subcommands
        assert any(keyword in output for keyword in ['benchmark', 'curate']) or \
               any(keyword in output for keyword in ['missing', 'try', '--help'])
    
    # Tests for help functionality
    
    def test_eval_help_command(self):
        """Test eval help command displays available options."""
        # Run eval help command
        result = self.run_cli_command(['eval', '--help'])
        
        # Verify help is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'commands'])
        
        # Should mention eval subcommands
        assert any(keyword in output for keyword in ['benchmark', 'curate'])
    
    def test_eval_benchmark_help_command(self):
        """Test eval benchmark help command displays specific options."""
        # Run eval benchmark help command
        result = self.run_cli_command(['eval', 'benchmark', '--help'])
        
        # Verify help is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'options', 'arguments'])
        
        # Should mention benchmark-specific options
        assert any(keyword in output for keyword in ['gold', 'predicted', 'output'])
    
    def test_eval_curate_help_command(self):
        """Test eval curate help command displays specific options."""
        # Run eval curate help command
        result = self.run_cli_command(['eval', 'curate', '--help'])
        
        # Verify help is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'options', 'arguments'])
        
        # Should mention curate-specific options
        assert any(keyword in output for keyword in ['input', 'output'])
    
    # Tests for edge cases and error handling
    
    def test_empty_gold_standard_file_handling(self):
        """Test handling of empty gold standard files."""
        # Create empty gold standard file
        empty_gold_file = self.create_temp_file({"documents": []})
        predicted_file = self.create_predicted_file()
        
        # Mock benchmarker functions to handle empty data
        with patch('src.evaluation.benchmarker.calculate_ner_metrics') as mock_ner:
            with patch('src.evaluation.benchmarker.calculate_relation_metrics') as mock_rel:
                mock_ner.return_value = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                mock_rel.return_value = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                
                # Run CLI command with empty file
                result = self.run_cli_command([
                    'eval', 'benchmark',
                    '--gold', empty_gold_file,
                    '--predicted', predicted_file
                ])
                
                # Command should handle empty files gracefully
                # May succeed with zero metrics or provide appropriate message
                output_text = (result.stderr + result.stdout).lower()
                assert len(output_text) > 0, "Should provide some feedback for empty input"
    
    def test_empty_llm_extraction_file_handling(self):
        """Test handling of empty LLM extraction files."""
        # Create empty extraction file
        empty_extraction = {
            "text": "",
            "entities": [],
            "relations": []
        }
        empty_input_file = self.create_temp_file(empty_extraction)
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Mock curation tool functions to handle empty data
        with patch('src.evaluation.curation_tool.load_llm_output') as mock_load:
            with patch('src.evaluation.curation_tool.display_for_review') as mock_display:
                with patch('src.evaluation.curation_tool.save_curated_output') as mock_save:
                    mock_load.return_value = empty_extraction
                    mock_save.return_value = True
                    
                    # Run CLI command with empty file
                    result = self.run_cli_command([
                        'eval', 'curate',
                        '--input', empty_input_file,
                        '--output', output_file
                    ])
                    
                    # Command should handle empty files gracefully
                    output_text = (result.stderr + result.stdout).lower()
                    assert len(output_text) > 0, "Should provide feedback for empty extraction"
    
    # Tests for output directory creation
    
    def test_commands_create_output_directories(self):
        """Test that evaluation commands can create output directories if they don't exist."""
        # Setup
        gold_file = self.create_gold_standard_file()
        predicted_file = self.create_predicted_file()
        
        # Create a non-existent output directory path
        base_temp_dir = self.create_temp_directory()
        output_file = os.path.join(base_temp_dir, 'new_subdir', 'benchmark_results.json')
        
        # Mock benchmarker functions
        with patch('src.evaluation.benchmarker.calculate_ner_metrics') as mock_ner:
            with patch('src.evaluation.benchmarker.calculate_relation_metrics') as mock_rel:
                mock_ner.return_value = {'precision': 0.9, 'recall': 0.8, 'f1': 0.85}
                mock_rel.return_value = {'precision': 0.8, 'recall': 0.7, 'f1': 0.75}
                
                # Run CLI command with non-existent output directory
                result = self.run_cli_command([
                    'eval', 'benchmark',
                    '--gold', gold_file,
                    '--predicted', predicted_file,
                    '--output', output_file
                ])
                
                # Command should create the directory and run successfully
                if result.returncode == 0:
                    assert os.path.exists(os.path.dirname(output_file)), "Output directory should be created"
                else:
                    error_text = (result.stderr + result.stdout).lower()
                    # Should not fail due to directory issues
                    assert not ('directory' in error_text and 'not' in error_text and 'exist' in error_text)
    
    # Integration tests for module interaction
    
    @patch('src.evaluation.benchmarker.calculate_ner_metrics')
    @patch('src.evaluation.benchmarker.calculate_relation_metrics')
    def test_benchmark_module_integration(self, mock_relation_metrics, mock_ner_metrics):
        """Test that benchmark command properly integrates with benchmarker module."""
        # Setup realistic mock data
        mock_ner_metrics.return_value = {
            'precision': 0.92,
            'recall': 0.88,
            'f1': 0.90
        }
        mock_relation_metrics.return_value = {
            'precision': 0.85,
            'recall': 0.82,
            'f1': 0.83
        }
        
        # Setup test files
        gold_file = self.create_gold_standard_file()
        predicted_file = self.create_predicted_file()
        
        # Run benchmark command
        result = self.run_cli_command([
            'eval', 'benchmark',
            '--gold', gold_file,
            '--predicted', predicted_file,
            '--verbose'
        ])
        
        # Verify successful integration
        assert result.returncode == 0, f"Integration test failed: {result.stderr}"
        
        # Verify both metric functions were called
        mock_ner_metrics.assert_called_once()
        mock_relation_metrics.assert_called_once()
        
        # Verify the functions were called with correct argument types
        ner_call_args = mock_ner_metrics.call_args[0]
        assert len(ner_call_args) == 2, "calculate_ner_metrics should be called with 2 arguments"
        assert isinstance(ner_call_args[0], list), "First argument should be a list (gold entities)"
        assert isinstance(ner_call_args[1], list), "Second argument should be a list (predicted entities)"
        
        rel_call_args = mock_relation_metrics.call_args[0]
        assert len(rel_call_args) == 2, "calculate_relation_metrics should be called with 2 arguments"
        assert isinstance(rel_call_args[0], list), "First argument should be a list (gold relations)"
        assert isinstance(rel_call_args[1], list), "Second argument should be a list (predicted relations)"
    
    @patch('src.evaluation.curation_tool.load_llm_output')
    @patch('src.evaluation.curation_tool.display_for_review')
    @patch('src.evaluation.curation_tool.save_curated_output')
    def test_curation_module_integration(self, mock_save, mock_display, mock_load):
        """Test that curate command properly integrates with curation_tool module."""
        # Setup realistic mock data
        mock_extraction_data = {
            'text': 'Quercetin is found in onion bulbs and has antioxidant properties.',
            'entities': [
                {
                    'entity_type': 'COMPOUND',
                    'text': 'Quercetin',
                    'start_char': 0,
                    'end_char': 9
                },
                {
                    'entity_type': 'PLANT_PART',
                    'text': 'onion bulbs',
                    'start_char': 22,
                    'end_char': 33
                }
            ],
            'relations': [
                ('Quercetin', 'found_in', 'onion bulbs')
            ]
        }
        
        mock_load.return_value = mock_extraction_data
        mock_save.return_value = True
        
        # Setup test files
        input_file = self.create_llm_extraction_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Run curate command
        result = self.run_cli_command([
            'eval', 'curate',
            '--input', input_file,
            '--output', output_file,
            '--verbose'
        ])
        
        # Verify successful integration
        assert result.returncode == 0, f"Integration test failed: {result.stderr}"
        
        # Verify all curation functions were called in correct order
        mock_load.assert_called_once_with(input_file)
        mock_display.assert_called_once()
        mock_save.assert_called_once()
        
        # Verify functions were called with correct arguments
        display_call_args = mock_display.call_args[0]
        assert len(display_call_args) == 3, "display_for_review should be called with 3 arguments"
        assert isinstance(display_call_args[0], str), "First argument should be text string"
        assert isinstance(display_call_args[1], list), "Second argument should be entities list"
        assert isinstance(display_call_args[2], list), "Third argument should be relations list"
        
        save_call_args = mock_save.call_args[0]
        assert len(save_call_args) == 2, "save_curated_output should be called with 2 arguments"
        assert isinstance(save_call_args[0], dict), "First argument should be curated data dict"
        assert save_call_args[1] == output_file, "Second argument should be output file path"