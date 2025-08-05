"""
Integration tests for CLI ontology management commands.

This module tests the command-line interface for ontology management,
including load, trim, and export operations.

Test Coverage:
- ontology load <file_path> command with dummy OWL file
- ontology trim <file_path> --keyword <keyword> command with filtering
- ontology export <input_file> <output_file> command
- Invalid arguments and error message handling
"""

import pytest
import tempfile
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestOntologyCLI:
    """Integration tests for ontology CLI commands."""
    
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
                import shutil
                shutil.rmtree(temp_dir)
    
    def create_dummy_owl_file(self, content=None):
        """Create a dummy OWL file for testing."""
        temp_file = tempfile.mktemp(suffix='.owl')
        self.temp_files.append(temp_file)
        
        if content is None:
            content = '''<?xml version="1.0"?>
<rdf:RDF xmlns="http://test.example.org/ontology#"
     xml:base="http://test.example.org/ontology"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://test.example.org/ontology"/>
    
    <owl:Class rdf:about="http://test.example.org/ontology#TestClass">
        <rdfs:label>Test Class</rdfs:label>
    </owl:Class>
    
    <owl:Class rdf:about="http://test.example.org/ontology#AnotherClass">
        <rdfs:label>Another Class</rdfs:label>
    </owl:Class>
</rdf:RDF>'''
        
        with open(temp_file, 'w') as f:
            f.write(content)
        
        return temp_file
    
    def create_dummy_ontology_with_keywords(self):
        """Create a dummy ontology with specific keywords for trimming tests."""
        content = '''<?xml version="1.0"?>
<rdf:RDF xmlns="http://test.example.org/ontology#"
     xml:base="http://test.example.org/ontology"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://test.example.org/ontology"/>
    
    <owl:Class rdf:about="http://test.example.org/ontology#PlantMetabolite">
        <rdfs:label>Plant Metabolite</rdfs:label>
        <rdfs:comment>A metabolite found in plants</rdfs:comment>
    </owl:Class>
    
    <owl:Class rdf:about="http://test.example.org/ontology#AnimalProtein">
        <rdfs:label>Animal Protein</rdfs:label>
        <rdfs:comment>A protein found in animals</rdfs:comment>
    </owl:Class>
    
    <owl:Class rdf:about="http://test.example.org/ontology#PlantCompound">
        <rdfs:label>Plant Compound</rdfs:label>
        <rdfs:comment>A chemical compound from plants</rdfs:comment>
    </owl:Class>
</rdf:RDF>'''
        return self.create_dummy_owl_file(content)
    
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
    
    @patch('src.ontology.loader.load_ontology')
    def test_ontology_load_command_with_dummy_owl_file(self, mock_load_ontology):
        """Test ontology load <file_path> command with a dummy OWL file."""
        # Create dummy OWL file
        dummy_file = self.create_dummy_owl_file()
        
        # Mock the loader function
        mock_ontology = MagicMock()
        mock_load_ontology.return_value = mock_ontology
        
        # Run CLI command
        result = self.run_cli_command(['ontology', 'load', dummy_file])
        
        # Verify command executed successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"
        
        # Verify loader was called with correct file path
        mock_load_ontology.assert_called_once_with(dummy_file)
        
        # Verify output contains success message
        assert "loaded" in result.stdout.lower() or "success" in result.stdout.lower()
    
    @patch('src.ontology.trimmer.trim_ontology')
    @patch('src.ontology.loader.load_ontology')
    def test_ontology_trim_command_with_keyword_filtering(self, mock_load_ontology, mock_trim_ontology):
        """Test ontology trim <file_path> --keyword <keyword> command with filtering criteria."""
        # Create dummy ontology with keywords
        dummy_file = self.create_dummy_ontology_with_keywords()
        
        # Mock the loader and trimmer functions
        mock_ontology = MagicMock()
        mock_load_ontology.return_value = mock_ontology
        mock_trimmed_ontology = MagicMock()
        mock_trim_ontology.return_value = mock_trimmed_ontology
        
        # Run CLI command with keyword filter
        keyword = "plant"
        result = self.run_cli_command(['ontology', 'trim', dummy_file, '--keyword', keyword])
        
        # Verify command executed successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"
        
        # Verify loader was called
        mock_load_ontology.assert_called_once_with(dummy_file)
        
        # Verify trimmer was called with keyword
        mock_trim_ontology.assert_called_once()
        call_args = mock_trim_ontology.call_args
        assert keyword in str(call_args) or any(keyword in str(arg) for arg in call_args[0])
        
        # Verify output contains success message
        assert "trimmed" in result.stdout.lower() or "filtered" in result.stdout.lower()
    
    @patch('src.ontology.exporter.export_ontology')
    @patch('src.ontology.loader.load_ontology')
    def test_ontology_export_command_to_temporary_file(self, mock_load_ontology, mock_export_ontology):
        """Test ontology export <input_file> <output_file> command and verify output."""
        # Create dummy input file
        input_file = self.create_dummy_owl_file()
        
        # Create temporary output file path
        output_file = tempfile.mktemp(suffix='.owl')
        self.temp_files.append(output_file)
        
        # Mock the loader and exporter functions
        mock_ontology = MagicMock()
        mock_load_ontology.return_value = mock_ontology
        mock_export_ontology.return_value = True
        
        # Run CLI command
        result = self.run_cli_command(['ontology', 'export', input_file, output_file])
        
        # Verify command executed successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"
        
        # Verify loader was called with input file
        mock_load_ontology.assert_called_once_with(input_file)
        
        # Verify exporter was called with output file
        mock_export_ontology.assert_called_once()
        call_args = mock_export_ontology.call_args
        assert output_file in str(call_args)
        
        # Verify output contains success message
        assert "exported" in result.stdout.lower() or "saved" in result.stdout.lower()
    
    def test_ontology_load_with_non_existent_file(self):
        """Test ontology load with non-existent file and ensure proper error message."""
        non_existent_file = "/path/to/non/existent/file.owl"
        
        # Run CLI command with non-existent file
        result = self.run_cli_command(['ontology', 'load', non_existent_file])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with non-existent file"
        
        # Verify error message is displayed
        error_output = result.stderr.lower()
        assert any(keyword in error_output for keyword in ['not found', 'does not exist', 'error', 'file'])
    
    def test_ontology_export_with_invalid_input_format(self):
        """Test ontology export with incorrect format and ensure proper error message."""
        # Create a non-OWL file (plain text)
        invalid_file = tempfile.mktemp(suffix='.txt')
        self.temp_files.append(invalid_file)
        
        with open(invalid_file, 'w') as f:
            f.write("This is not an OWL file")
        
        output_file = tempfile.mktemp(suffix='.owl')
        self.temp_files.append(output_file)
        
        # Run CLI command with invalid input
        result = self.run_cli_command(['ontology', 'export', invalid_file, output_file])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with invalid input format"
        
        # Verify error message is displayed
        error_output = result.stderr.lower()
        assert any(keyword in error_output for keyword in ['invalid', 'format', 'error', 'parse'])
    
    def test_ontology_trim_with_missing_keyword_argument(self):
        """Test ontology trim without required keyword argument."""
        dummy_file = self.create_dummy_owl_file()
        
        # Run CLI command without keyword argument
        result = self.run_cli_command(['ontology', 'trim', dummy_file])
        
        # Verify command failed or shows help
        assert result.returncode != 0 or "usage" in result.stdout.lower() or "help" in result.stdout.lower()
        
        # Verify error message mentions missing argument
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['keyword', 'required', 'missing', 'argument'])
    
    def test_invalid_ontology_subcommand(self):
        """Test invalid ontology subcommand and ensure proper error message."""
        # Run CLI command with invalid subcommand
        result = self.run_cli_command(['ontology', 'invalid_command'])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with invalid subcommand"
        
        # Verify error message is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['invalid', 'unknown', 'command', 'usage', 'help'])
    
    def test_ontology_command_without_subcommand(self):
        """Test ontology command without any subcommand."""
        # Run CLI command without subcommand
        result = self.run_cli_command(['ontology'])
        
        # Should show help or usage information
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'commands', 'subcommand'])
    
    def test_cli_help_command(self):
        """Test CLI help command displays available options."""
        # Run CLI help command
        result = self.run_cli_command(['--help'])
        
        # Verify help is displayed
        output = result.stdout.lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'commands', 'ontology'])
        
        # Should mention ontology commands
        assert 'ontology' in output
    
    @patch('src.ontology.loader.load_ontology')
    def test_ontology_load_with_verbose_output(self, mock_load_ontology):
        """Test ontology load command with verbose output."""
        dummy_file = self.create_dummy_owl_file()
        
        # Mock the loader function
        mock_ontology = MagicMock()
        mock_load_ontology.return_value = mock_ontology
        
        # Run CLI command with verbose flag
        result = self.run_cli_command(['ontology', 'load', dummy_file, '--verbose'])
        
        # Verify command executed successfully
        assert result.returncode == 0, f"Command failed with error: {result.stderr}"
        
        # Verify verbose output is provided
        assert len(result.stdout) > 0, "Verbose output should be provided"
    
    def test_multiple_keyword_filters_in_trim_command(self):
        """Test ontology trim command with multiple keyword filters."""
        dummy_file = self.create_dummy_ontology_with_keywords()
        
        # This test structure assumes the CLI supports multiple keywords
        # The actual implementation may vary
        with patch('src.ontology.loader.load_ontology') as mock_load, \
             patch('src.ontology.trimmer.trim_ontology') as mock_trim:
            
            mock_ontology = MagicMock()
            mock_load.return_value = mock_ontology
            mock_trim.return_value = mock_ontology
            
            # Run CLI command with multiple keywords
            result = self.run_cli_command([
                'ontology', 'trim', dummy_file, 
                '--keyword', 'plant', 
                '--keyword', 'metabolite'
            ])
            
            # The command should handle multiple keywords appropriately
            # (either succeed or provide clear error message about syntax)
            assert result.returncode == 0 or "keyword" in result.stderr.lower()
    
    def test_export_to_different_formats(self):
        """Test ontology export to different output formats."""
        input_file = self.create_dummy_owl_file()
        
        # Test different output formats
        formats = ['.owl', '.rdf', '.ttl']
        
        for fmt in formats:
            output_file = tempfile.mktemp(suffix=fmt)
            self.temp_files.append(output_file)
            
            with patch('src.ontology.loader.load_ontology') as mock_load, \
                 patch('src.ontology.exporter.export_ontology') as mock_export:
                
                mock_ontology = MagicMock()
                mock_load.return_value = mock_ontology
                mock_export.return_value = True
                
                # Run CLI command
                result = self.run_cli_command(['ontology', 'export', input_file, output_file])
                
                # Verify command handles the format appropriately
                assert result.returncode == 0 or "format" in result.stderr.lower()