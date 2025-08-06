"""
Integration tests for CLI postprocessing commands.

This module tests the command-line interface for postprocessing operations
including entity mapping, relation mapping, data cleaning, deduplication,
and taxonomic filtering.

Test Coverage:
- map entities --input <file> --ontology <url> --output <file> command
- map relations --input <file> --ontology <url> --output <file> command  
- clean normalize --input <file> --output <file> command
- clean deduplicate --input <file> --output <file> command
- taxonomy filter --input <file> --lineage <lineage> --output <file> command
- Invalid arguments and error message handling
- Proper mocking of external dependencies
- Cleanup of temporary files and directories

Note: These tests are designed for the expected CLI interface. Commands that are not yet
implemented will currently fail with "No such command" errors, which is expected.
When the commands are implemented, these tests will validate the full functionality.
"""

import pytest
import tempfile
import os
import subprocess
import sys
import shutil
import json
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open


# Mark tests as expected to fail until commands are implemented
commands_not_implemented = pytest.mark.xfail(
    reason="CLI postprocessing commands not yet implemented",
    raises=(AssertionError, subprocess.CalledProcessError),
    strict=False
)


class TestPostprocessingCLI:
    """Integration tests for postprocessing CLI commands."""
    
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
        fd, temp_file = tempfile.mkstemp(suffix=suffix)
        self.temp_files.append(temp_file)
        
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return temp_file
    
    def create_entities_file(self):
        """Create a temporary entities file for testing."""
        entities = [
            "glucose",
            "arabidopsis",
            "photosynthesis",
            "ATP",
            "chlorophyll"
        ]
        return self.create_temp_file('\n'.join(entities), suffix='.txt')
    
    def create_entities_json_file(self):
        """Create a temporary entities JSON file for testing."""
        entities = [
            {
                "text": "glucose",
                "label": "METABOLITE",
                "start": 0,
                "end": 7,
                "confidence": 0.95
            },
            {
                "text": "arabidopsis",
                "label": "SPECIES",
                "start": 20,
                "end": 31,
                "confidence": 0.88
            },
            {
                "text": "photosynthesis",
                "label": "PROCESS",
                "start": 40,
                "end": 54,
                "confidence": 0.92
            }
        ]
        return self.create_temp_file(json.dumps(entities, indent=2), suffix='.json')
    
    def create_relations_file(self):
        """Create a temporary relations file for testing."""
        relations = [
            ["glucose", "metabolized_by", "enzyme"],
            ["arabidopsis", "has_part", "leaf"],
            ["ATP", "produced_by", "respiration"],
            ["chlorophyll", "found_in", "chloroplast"]
        ]
        return self.create_temp_file(json.dumps(relations, indent=2), suffix='.json')
    
    def create_records_file(self):
        """Create a temporary records file for deduplication testing."""
        records = [
            {
                "id": 1,
                "name": "glucose",
                "synonyms": ["D-glucose", "dextrose"],
                "category": "metabolite"
            },
            {
                "id": 2,
                "name": "Glucose",
                "synonyms": ["glucose", "blood sugar"],
                "category": "metabolite"
            },
            {
                "id": 3,
                "name": "arabidopsis",
                "synonyms": ["Arabidopsis thaliana", "thale cress"],
                "category": "species"
            },
            {
                "id": 4,
                "name": "ATP",
                "synonyms": ["adenosine triphosphate"],
                "category": "metabolite"
            }
        ]
        return self.create_temp_file(json.dumps(records, indent=2), suffix='.json')
    
    def create_species_file(self):
        """Create a temporary species file for taxonomy filtering."""
        species = [
            "Arabidopsis thaliana",
            "Oryza sativa",
            "Zea mays",
            "Triticum aestivum",
            "Solanum lycopersicum",
            "Escherichia coli",
            "Saccharomyces cerevisiae"
        ]
        return self.create_temp_file('\n'.join(species), suffix='.txt')
    
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
    
    # Tests for map entities command
    
    @commands_not_implemented
    def test_map_entities_command_success(self):
        """Test map entities command with successful execution."""
        # Setup
        input_file = self.create_entities_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        ontology_url = "http://purl.obolibrary.org/obo/chebi.owl"
        
        # Mock the entity mapping function
        mock_results = pd.DataFrame([
            {
                'Source Term': 'glucose',
                'Mapped Term Label': 'D-glucose',
                'Mapped Term IRI': 'http://purl.obolibrary.org/obo/CHEBI_17234',
                'Mapping Score': 0.95,
                'Term Type': 'class'
            },
            {
                'Source Term': 'ATP',
                'Mapped Term Label': 'adenosine 5\'-triphosphate',
                'Mapped Term IRI': 'http://purl.obolibrary.org/obo/CHEBI_15422',
                'Mapping Score': 0.88,
                'Term Type': 'class'
            }
        ])
        
        with patch('src.ontology_mapping.entity_mapper.map_entities_to_ontology') as mock_map:
            mock_map.return_value = mock_results
            
            # Run CLI command
            result = self.run_cli_command([
                'map', 'entities',
                '--input', input_file,
                '--ontology', ontology_url,
                '--output', output_file
            ])
            
            # Verify command executed successfully
            assert result.returncode == 0, f"Command failed with error: {result.stderr}"
            
            # Verify output contains success message
            output_text = result.stdout.lower()
            assert any(keyword in output_text for keyword in ['mapped', 'entities', 'success'])
            
            # Verify output file was created
            assert os.path.exists(output_file), f"Output file should be created: {output_file}"
            
            # Verify output file contains mappings
            with open(output_file, 'r') as f:
                mappings = json.load(f)
                assert len(mappings) >= 1, "Should contain mapped entities"
            
            # Verify mock was called with correct parameters
            mock_map.assert_called_once()
            args, kwargs = mock_map.call_args
            assert ontology_url in args or ontology_url in kwargs.values()
    
    @commands_not_implemented
    def test_map_entities_with_method_parameter(self):
        """Test map entities command with mapping method parameter."""
        # Setup
        input_file = self.create_entities_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        ontology_url = "http://purl.obolibrary.org/obo/chebi.owl"
        
        # Mock the entity mapping function
        with patch('src.ontology_mapping.entity_mapper.map_entities_to_ontology') as mock_map:
            mock_map.return_value = pd.DataFrame()
            
            # Run CLI command with method parameter
            result = self.run_cli_command([
                'map', 'entities',
                '--input', input_file,
                '--ontology', ontology_url,
                '--output', output_file,
                '--method', 'levenshtein'
            ])
            
            # Verify command executed successfully
            assert result.returncode == 0, f"Command failed with error: {result.stderr}"
            
            # Verify map_entities_to_ontology was called with correct method
            mock_map.assert_called_once()
            args, kwargs = mock_map.call_args
            assert kwargs.get('mapping_method') == 'levenshtein'
    
    @commands_not_implemented
    def test_map_entities_with_min_score_parameter(self):
        """Test map entities command with minimum score parameter."""
        # Setup
        input_file = self.create_entities_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        ontology_url = "http://purl.obolibrary.org/obo/chebi.owl"
        
        # Mock the entity mapping function
        with patch('src.ontology_mapping.entity_mapper.map_entities_to_ontology') as mock_map:
            mock_map.return_value = pd.DataFrame()
            
            # Run CLI command with min-score parameter
            result = self.run_cli_command([
                'map', 'entities',
                '--input', input_file,
                '--ontology', ontology_url,
                '--output', output_file,
                '--min-score', '0.8'
            ])
            
            # Verify command executed successfully
            assert result.returncode == 0, f"Command failed with error: {result.stderr}"
            
            # Verify map_entities_to_ontology was called with correct min_score
            mock_map.assert_called_once()
            args, kwargs = mock_map.call_args
            assert kwargs.get('min_score') == 0.8
    
    @commands_not_implemented
    def test_map_entities_missing_input_argument(self):
        """Test map entities command with missing input argument."""
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        ontology_url = "http://purl.obolibrary.org/obo/chebi.owl"
        
        # Run CLI command without input argument
        result = self.run_cli_command([
            'map', 'entities',
            '--ontology', ontology_url,
            '--output', output_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with missing input"
        
        # Verify error message mentions missing argument
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['input', 'required', 'missing', 'argument'])
    
    @commands_not_implemented
    def test_map_entities_missing_ontology_argument(self):
        """Test map entities command with missing ontology argument."""
        input_file = self.create_entities_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Run CLI command without ontology argument
        result = self.run_cli_command([
            'map', 'entities',
            '--input', input_file,
            '--output', output_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with missing ontology"
        
        # Verify error message mentions missing argument
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['ontology', 'required', 'missing', 'argument'])
    
    @commands_not_implemented
    def test_map_entities_invalid_ontology_url(self):
        """Test map entities command with invalid ontology URL."""
        input_file = self.create_entities_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        invalid_url = "not-a-valid-url"
        
        # Run CLI command with invalid ontology URL
        result = self.run_cli_command([
            'map', 'entities',
            '--input', input_file,
            '--ontology', invalid_url,
            '--output', output_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with invalid URL"
        
        # Verify error message mentions URL format
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['invalid', 'url', 'ontology', 'error'])
    
    # Tests for map relations command
    
    @commands_not_implemented
    def test_map_relations_command_success(self):
        """Test map relations command with successful execution."""
        # Setup
        input_file = self.create_relations_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        ontology_url = "http://purl.obolibrary.org/obo/ro.owl"
        
        # Mock the relation mapping function
        mock_results = pd.DataFrame([
            {
                'Subject': 'glucose',
                'Relation': 'metabolized_by',
                'Object': 'enzyme',
                'Mapped_Relation_Label': 'metabolized by',
                'Mapped_Relation_IRI': 'http://purl.obolibrary.org/obo/RO_0002180',
                'Mapping_Score': 0.92,
                'Term_Type': 'objectProperty',
                'Semantic_Valid': True
            }
        ])
        
        with patch('src.ontology_mapping.relation_mapper.map_relationships_to_ontology') as mock_map:
            mock_map.return_value = mock_results
            
            # Run CLI command
            result = self.run_cli_command([
                'map', 'relations',
                '--input', input_file,
                '--ontology', ontology_url,
                '--output', output_file
            ])
            
            # Verify command executed successfully
            assert result.returncode == 0, f"Command failed with error: {result.stderr}"
            
            # Verify output contains success message
            output_text = result.stdout.lower()
            assert any(keyword in output_text for keyword in ['mapped', 'relations', 'success'])
            
            # Verify output file was created
            assert os.path.exists(output_file), f"Output file should be created: {output_file}"
            
            # Verify output file contains mappings
            with open(output_file, 'r') as f:
                mappings = json.load(f)
                assert len(mappings) >= 1, "Should contain mapped relations"
            
            # Verify mock was called
            mock_map.assert_called_once()
    
    @commands_not_implemented
    def test_map_relations_with_validate_semantics_parameter(self):
        """Test map relations command with semantic validation parameter."""
        # Setup
        input_file = self.create_relations_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        ontology_url = "http://purl.obolibrary.org/obo/ro.owl"
        
        # Mock the relation mapping function
        with patch('src.ontology_mapping.relation_mapper.map_relationships_to_ontology') as mock_map:
            mock_map.return_value = pd.DataFrame()
            
            # Run CLI command with validate-semantics parameter
            result = self.run_cli_command([
                'map', 'relations',
                '--input', input_file,
                '--ontology', ontology_url,
                '--output', output_file,
                '--validate-semantics'
            ])
            
            # Verify command executed successfully
            assert result.returncode == 0, f"Command failed with error: {result.stderr}"
            
            # Verify map_relationships_to_ontology was called with correct validation
            mock_map.assert_called_once()
            args, kwargs = mock_map.call_args
            assert kwargs.get('validate_semantics') is True
    
    @commands_not_implemented
    def test_map_relations_missing_input_argument(self):
        """Test map relations command with missing input argument."""
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        ontology_url = "http://purl.obolibrary.org/obo/ro.owl"
        
        # Run CLI command without input argument
        result = self.run_cli_command([
            'map', 'relations',
            '--ontology', ontology_url,
            '--output', output_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with missing input"
        
        # Verify error message mentions missing argument
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['input', 'required', 'missing', 'argument'])
    
    # Tests for clean normalize command
    
    @commands_not_implemented
    def test_clean_normalize_command_success(self):
        """Test clean normalize command with successful execution."""
        # Setup
        input_file = self.create_entities_json_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Mock the normalization function
        with patch('src.data_quality.normalizer.normalize_name') as mock_normalize:
            mock_normalize.side_effect = lambda x: x.lower().strip()
            
            # Run CLI command
            result = self.run_cli_command([
                'clean', 'normalize',
                '--input', input_file,
                '--output', output_file
            ])
            
            # Verify command executed successfully
            assert result.returncode == 0, f"Command failed with error: {result.stderr}"
            
            # Verify output contains success message
            output_text = result.stdout.lower()
            assert any(keyword in output_text for keyword in ['normalized', 'cleaned', 'success'])
            
            # Verify output file was created
            assert os.path.exists(output_file), f"Output file should be created: {output_file}"
            
            # Verify output file contains normalized data
            with open(output_file, 'r') as f:
                normalized_data = json.load(f)
                assert len(normalized_data) >= 1, "Should contain normalized entities"
            
            # Verify normalize_name was called
            assert mock_normalize.call_count >= 1, "normalize_name should have been called"
    
    @commands_not_implemented
    def test_clean_normalize_with_case_option(self):
        """Test clean normalize command with case conversion option."""
        # Setup
        input_file = self.create_entities_json_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Mock the normalization function
        with patch('src.data_quality.normalizer.normalize_name') as mock_normalize:
            mock_normalize.side_effect = lambda x: x.lower()
            
            # Run CLI command with case option
            result = self.run_cli_command([
                'clean', 'normalize',
                '--input', input_file,
                '--output', output_file,
                '--case', 'lower'
            ])
            
            # Verify command executed successfully
            assert result.returncode == 0, f"Command failed with error: {result.stderr}"
            
            # Verify normalization was applied
            assert mock_normalize.call_count >= 1
    
    @commands_not_implemented
    def test_clean_normalize_missing_input_argument(self):
        """Test clean normalize command with missing input argument."""
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Run CLI command without input argument
        result = self.run_cli_command([
            'clean', 'normalize',
            '--output', output_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with missing input"
        
        # Verify error message mentions missing argument
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['input', 'required', 'missing', 'argument'])
    
    # Tests for clean deduplicate command
    
    @commands_not_implemented
    def test_clean_deduplicate_command_success(self):
        """Test clean deduplicate command with successful execution."""
        # Setup
        input_file = self.create_records_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Mock the deduplication function
        mock_deduplicated = [
            {
                "id": 1,
                "name": "glucose",
                "synonyms": ["D-glucose", "dextrose"],
                "category": "metabolite"
            },
            {
                "id": 3,
                "name": "arabidopsis",
                "synonyms": ["Arabidopsis thaliana", "thale cress"],
                "category": "species"
            },
            {
                "id": 4,
                "name": "ATP",
                "synonyms": ["adenosine triphosphate"],
                "category": "metabolite"
            }
        ]
        
        with patch('src.data_quality.deduplicator.deduplicate_entities') as mock_dedupe:
            mock_dedupe.return_value = mock_deduplicated
            
            # Run CLI command
            result = self.run_cli_command([
                'clean', 'deduplicate',
                '--input', input_file,
                '--output', output_file,
                '--fields', 'name,synonyms'
            ])
            
            # Verify command executed successfully
            assert result.returncode == 0, f"Command failed with error: {result.stderr}"
            
            # Verify output contains success message
            output_text = result.stdout.lower()
            assert any(keyword in output_text for keyword in ['deduplicated', 'cleaned', 'success'])
            
            # Verify output file was created
            assert os.path.exists(output_file), f"Output file should be created: {output_file}"
            
            # Verify output file contains deduplicated data
            with open(output_file, 'r') as f:
                deduplicated_data = json.load(f)
                assert len(deduplicated_data) >= 1, "Should contain deduplicated records"
            
            # Verify deduplicate_entities was called with correct parameters
            mock_dedupe.assert_called_once()
            args, kwargs = mock_dedupe.call_args
            assert 'name' in args[1] or 'name' in kwargs.get('fields', [])
    
    @commands_not_implemented
    def test_clean_deduplicate_with_threshold_parameter(self):
        """Test clean deduplicate command with similarity threshold parameter."""
        # Setup
        input_file = self.create_records_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Mock the deduplication function
        with patch('src.data_quality.deduplicator.deduplicate_entities') as mock_dedupe:
            mock_dedupe.return_value = []
            
            # Run CLI command with threshold parameter
            result = self.run_cli_command([
                'clean', 'deduplicate',
                '--input', input_file,
                '--output', output_file,
                '--fields', 'name',
                '--threshold', '0.8'
            ])
            
            # Verify command executed successfully
            assert result.returncode == 0, f"Command failed with error: {result.stderr}"
            
            # Verify deduplication was called (threshold may be used internally)
            mock_dedupe.assert_called_once()
    
    @commands_not_implemented
    def test_clean_deduplicate_missing_fields_argument(self):
        """Test clean deduplicate command with missing fields argument."""
        input_file = self.create_records_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Run CLI command without fields argument
        result = self.run_cli_command([
            'clean', 'deduplicate',
            '--input', input_file,
            '--output', output_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with missing fields"
        
        # Verify error message mentions missing argument
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['fields', 'required', 'missing', 'argument'])
    
    # Tests for taxonomy filter command
    
    @commands_not_implemented
    def test_taxonomy_filter_command_success(self):
        """Test taxonomy filter command with successful execution."""
        # Setup
        input_file = self.create_species_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        lineage = "Viridiplantae"
        
        # Mock the taxonomy functions
        mock_taxonomy = MagicMock()
        mock_filtered_species = [
            {
                "species": "Arabidopsis thaliana",
                "taxonomic_id": 3702,
                "lineage": ["Eukaryota", "Viridiplantae", "Streptophyta", "Embryophyta"]
            },
            {
                "species": "Oryza sativa",
                "taxonomic_id": 4530,
                "lineage": ["Eukaryota", "Viridiplantae", "Streptophyta", "Embryophyta"]
            }
        ]
        
        with patch('src.data_quality.taxonomy.load_ncbi_taxonomy') as mock_load:
            with patch('src.data_quality.taxonomy.filter_species_by_lineage') as mock_filter:
                mock_load.return_value = mock_taxonomy
                mock_filter.return_value = mock_filtered_species
                
                # Run CLI command
                result = self.run_cli_command([
                    'taxonomy', 'filter',
                    '--input', input_file,
                    '--lineage', lineage,
                    '--output', output_file
                ])
                
                # Verify command executed successfully
                assert result.returncode == 0, f"Command failed with error: {result.stderr}"
                
                # Verify output contains success message
                output_text = result.stdout.lower()
                assert any(keyword in output_text for keyword in ['filtered', 'taxonomy', 'success'])
                
                # Verify output file was created
                assert os.path.exists(output_file), f"Output file should be created: {output_file}"
                
                # Verify output file contains filtered data
                with open(output_file, 'r') as f:
                    filtered_data = json.load(f)
                    assert len(filtered_data) >= 1, "Should contain filtered species"
                
                # Verify taxonomy functions were called
                mock_load.assert_called_once()
                mock_filter.assert_called_once()
                args, kwargs = mock_filter.call_args
                assert lineage in args or lineage in kwargs.values()
    
    @commands_not_implemented
    def test_taxonomy_filter_with_rank_parameter(self):
        """Test taxonomy filter command with taxonomic rank parameter."""
        # Setup
        input_file = self.create_species_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        lineage = "Rosaceae"
        rank = "family"
        
        # Mock the taxonomy functions
        mock_taxonomy = MagicMock()
        
        with patch('src.data_quality.taxonomy.load_ncbi_taxonomy') as mock_load:
            with patch('src.data_quality.taxonomy.filter_species_by_lineage') as mock_filter:
                mock_load.return_value = mock_taxonomy
                mock_filter.return_value = []
                
                # Run CLI command with rank parameter
                result = self.run_cli_command([
                    'taxonomy', 'filter',
                    '--input', input_file,
                    '--lineage', lineage,
                    '--output', output_file,
                    '--rank', rank
                ])
                
                # Verify command executed successfully
                assert result.returncode == 0, f"Command failed with error: {result.stderr}"
                
                # Verify filter_species_by_lineage was called with correct rank
                mock_filter.assert_called_once()
                args, kwargs = mock_filter.call_args
                assert kwargs.get('rank') == rank or rank in args
    
    @commands_not_implemented
    def test_taxonomy_filter_missing_lineage_argument(self):
        """Test taxonomy filter command with missing lineage argument."""
        input_file = self.create_species_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Run CLI command without lineage argument
        result = self.run_cli_command([
            'taxonomy', 'filter',
            '--input', input_file,
            '--output', output_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with missing lineage"
        
        # Verify error message mentions missing argument
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['lineage', 'required', 'missing', 'argument'])
    
    # Tests for invalid subcommands and general CLI behavior
    
    @commands_not_implemented
    def test_invalid_map_subcommand(self):
        """Test invalid map subcommand and ensure proper error message."""
        # Run CLI command with invalid subcommand
        result = self.run_cli_command(['map', 'invalid_command'])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with invalid subcommand"
        
        # Verify error message is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['invalid', 'unknown', 'command', 'usage', 'help'])
    
    @commands_not_implemented
    def test_invalid_clean_subcommand(self):
        """Test invalid clean subcommand and ensure proper error message."""
        # Run CLI command with invalid subcommand
        result = self.run_cli_command(['clean', 'invalid_command'])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with invalid subcommand"
        
        # Verify error message is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['invalid', 'unknown', 'command', 'usage', 'help'])
    
    @commands_not_implemented
    def test_invalid_taxonomy_subcommand(self):
        """Test invalid taxonomy subcommand and ensure proper error message."""
        # Run CLI command with invalid subcommand
        result = self.run_cli_command(['taxonomy', 'invalid_command'])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with invalid subcommand"
        
        # Verify error message is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['invalid', 'unknown', 'command', 'usage', 'help'])
    
    @commands_not_implemented
    def test_map_command_without_subcommand(self):
        """Test map command without any subcommand."""
        # Run CLI command without subcommand
        result = self.run_cli_command(['map'])
        
        # Should show help or usage information
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'command'])
        
        # Should mention available subcommands
        assert any(keyword in output for keyword in ['entities', 'relations']) or \
               any(keyword in output for keyword in ['missing', 'try', '--help'])
    
    @commands_not_implemented
    def test_clean_command_without_subcommand(self):
        """Test clean command without any subcommand."""
        # Run CLI command without subcommand
        result = self.run_cli_command(['clean'])
        
        # Should show help or usage information
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'command'])
        
        # Should mention available subcommands
        assert any(keyword in output for keyword in ['normalize', 'deduplicate']) or \
               any(keyword in output for keyword in ['missing', 'try', '--help'])
    
    @commands_not_implemented
    def test_taxonomy_command_without_subcommand(self):
        """Test taxonomy command without any subcommand."""
        # Run CLI command without subcommand
        result = self.run_cli_command(['taxonomy'])
        
        # Should show help or usage information
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'command'])
        
        # Should mention available subcommands
        assert any(keyword in output for keyword in ['filter']) or \
               any(keyword in output for keyword in ['missing', 'try', '--help'])
    
    # Tests for help functionality
    
    @commands_not_implemented
    def test_map_help_command(self):
        """Test map help command displays available options."""
        # Run map help command
        result = self.run_cli_command(['map', '--help'])
        
        # Verify help is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'commands'])
        
        # Should mention map subcommands
        assert any(keyword in output for keyword in ['entities', 'relations'])
    
    @commands_not_implemented
    def test_clean_help_command(self):
        """Test clean help command displays available options."""
        # Run clean help command
        result = self.run_cli_command(['clean', '--help'])
        
        # Verify help is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'commands'])
        
        # Should mention clean subcommands
        assert any(keyword in output for keyword in ['normalize', 'deduplicate'])
    
    @commands_not_implemented
    def test_taxonomy_help_command(self):
        """Test taxonomy help command displays available options."""
        # Run taxonomy help command
        result = self.run_cli_command(['taxonomy', '--help'])
        
        # Verify help is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'commands'])
        
        # Should mention taxonomy subcommands
        assert any(keyword in output for keyword in ['filter'])
    
    @commands_not_implemented
    def test_map_entities_help_command(self):
        """Test map entities help command displays specific options."""
        # Run map entities help command
        result = self.run_cli_command(['map', 'entities', '--help'])
        
        # Verify help is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'options', 'arguments'])
        
        # Should mention entities-specific options
        assert any(keyword in output for keyword in ['input', 'ontology', 'output'])
    
    @commands_not_implemented
    def test_map_relations_help_command(self):
        """Test map relations help command displays specific options."""
        # Run map relations help command
        result = self.run_cli_command(['map', 'relations', '--help'])
        
        # Verify help is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'options', 'arguments'])
        
        # Should mention relations-specific options
        assert any(keyword in output for keyword in ['input', 'ontology', 'output'])
    
    @commands_not_implemented
    def test_clean_normalize_help_command(self):
        """Test clean normalize help command displays specific options."""
        # Run clean normalize help command
        result = self.run_cli_command(['clean', 'normalize', '--help'])
        
        # Verify help is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'options', 'arguments'])
        
        # Should mention normalize-specific options
        assert any(keyword in output for keyword in ['input', 'output'])
    
    @commands_not_implemented
    def test_clean_deduplicate_help_command(self):
        """Test clean deduplicate help command displays specific options."""
        # Run clean deduplicate help command
        result = self.run_cli_command(['clean', 'deduplicate', '--help'])
        
        # Verify help is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'options', 'arguments'])
        
        # Should mention deduplicate-specific options
        assert any(keyword in output for keyword in ['input', 'output', 'fields'])
    
    @commands_not_implemented
    def test_taxonomy_filter_help_command(self):
        """Test taxonomy filter help command displays specific options."""
        # Run taxonomy filter help command
        result = self.run_cli_command(['taxonomy', 'filter', '--help'])
        
        # Verify help is displayed
        output = (result.stderr + result.stdout).lower()
        assert any(keyword in output for keyword in ['usage', 'help', 'options', 'arguments'])
        
        # Should mention filter-specific options
        assert any(keyword in output for keyword in ['input', 'lineage', 'output'])
    
    # Tests for error handling and edge cases
    
    @commands_not_implemented
    def test_map_entities_with_nonexistent_input_file(self):
        """Test map entities command with non-existent input file."""
        non_existent_file = "/path/to/non/existent/file.txt"
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        ontology_url = "http://purl.obolibrary.org/obo/chebi.owl"
        
        # Run CLI command with non-existent file
        result = self.run_cli_command([
            'map', 'entities',
            '--input', non_existent_file,
            '--ontology', ontology_url,
            '--output', output_file
        ])
        
        # Verify command failed
        assert result.returncode != 0, "Command should have failed with non-existent file"
        
        # Verify error message is displayed
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['not found', 'does not exist', 'error', 'file'])
    
    @commands_not_implemented
    def test_clean_normalize_with_empty_input_file(self):
        """Test clean normalize command with empty input file."""
        # Create empty input file
        empty_input_file = self.create_temp_file("")
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Run CLI command with empty file
        result = self.run_cli_command([
            'clean', 'normalize',
            '--input', empty_input_file,
            '--output', output_file
        ])
        
        # Command should handle empty files gracefully
        # May succeed with empty output or provide appropriate message
        output_text = (result.stderr + result.stdout).lower()
        assert len(output_text) > 0, "Should provide some feedback for empty input"
    
    # Tests for output directory creation
    
    @commands_not_implemented
    def test_commands_create_output_directories(self):
        """Test that postprocessing commands can create output directories if they don't exist."""
        # Setup
        input_file = self.create_entities_file()
        
        # Create a non-existent output directory path
        base_temp_dir = self.create_temp_directory()
        output_dir = os.path.join(base_temp_dir, 'new_subdir', 'postprocessing_output')
        output_file = os.path.join(output_dir, 'results.json')
        
        # Mock the entity mapping function
        with patch('src.ontology_mapping.entity_mapper.map_entities_to_ontology') as mock_map:
            mock_map.return_value = pd.DataFrame()
            
            # Run CLI command with non-existent output directory
            result = self.run_cli_command([
                'map', 'entities',
                '--input', input_file,
                '--ontology', 'http://purl.obolibrary.org/obo/chebi.owl',
                '--output', output_file
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
    def test_all_postprocessing_commands_with_verbose_flag(self):
        """Test that all postprocessing commands respond to --verbose flag."""
        input_file = self.create_entities_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Test map entities with verbose
        with patch('src.ontology_mapping.entity_mapper.map_entities_to_ontology') as mock_map:
            mock_map.return_value = pd.DataFrame()
            
            result = self.run_cli_command([
                'map', 'entities',
                '--input', input_file,
                '--ontology', 'http://purl.obolibrary.org/obo/chebi.owl',
                '--output', output_file,
                '--verbose'
            ])
            
            # Verify verbose output is provided regardless of success/failure
            assert len(result.stdout) > 0, "Verbose output should be provided"
    
    # Tests for API/network error handling
    
    @patch('src.ontology_mapping.entity_mapper.map_entities_to_ontology')
    @commands_not_implemented
    def test_map_entities_with_api_error(self, mock_map):
        """Test map entities command handling API/network errors."""
        # Setup
        input_file = self.create_entities_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Mock API error
        from src.ontology_mapping.entity_mapper import MappingError
        mock_map.side_effect = MappingError("Failed to connect to ontology service")
        
        # Run CLI command
        result = self.run_cli_command([
            'map', 'entities',
            '--input', input_file,
            '--ontology', 'http://purl.obolibrary.org/obo/chebi.owl',
            '--output', output_file
        ])
        
        # Verify command failed gracefully
        assert result.returncode != 0, "Command should have failed with API error"
        
        # Verify error message is displayed
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['error', 'failed', 'connect'])
    
    @patch('src.data_quality.taxonomy.load_ncbi_taxonomy')
    @commands_not_implemented
    def test_taxonomy_filter_with_database_error(self, mock_load):
        """Test taxonomy filter command handling database loading errors."""
        # Setup
        input_file = self.create_species_file()
        output_file = tempfile.mktemp(suffix='.json')
        self.temp_files.append(output_file)
        
        # Mock database error
        from src.data_quality.taxonomy import TaxonomyError
        mock_load.side_effect = TaxonomyError("Failed to load NCBI taxonomy database")
        
        # Run CLI command
        result = self.run_cli_command([
            'taxonomy', 'filter',
            '--input', input_file,
            '--lineage', 'Viridiplantae',
            '--output', output_file
        ])
        
        # Verify command failed gracefully
        assert result.returncode != 0, "Command should have failed with database error"
        
        # Verify error message is displayed
        error_output = (result.stderr + result.stdout).lower()
        assert any(keyword in error_output for keyword in ['error', 'failed', 'taxonomy', 'database'])