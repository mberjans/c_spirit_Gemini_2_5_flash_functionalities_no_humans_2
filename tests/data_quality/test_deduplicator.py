"""
Unit tests for src/data_quality/deduplicator.py

This module tests the data quality deduplication functionality for identifying and
consolidating duplicate entity records in the AIM2-ODIE ontology development and
information extraction system. The deduplicator uses both exact matching and 
fuzzy matching to identify records that represent the same entity.

Test Coverage:
- Entity deduplication: exact duplicates, approximate matches using dedupe/recordlinkage
- Output format validation: list of unique consolidated entities
- Empty input handling: empty lists, None values
- Integration with normalizer: preprocessing with normalize_name function
- Error handling: invalid inputs, type mismatches, field validation
- Performance considerations: large datasets, memory efficiency
- Edge cases: malformed records, missing fields, special characters
- Mock external libraries: proper mocking of dedupe and recordlinkage

Functions Under Test:
- deduplicate_entities(records: list[dict], fields: list[str], settings_file: str = None, 
  training_file: str = None) -> list[dict]: Core deduplication functionality

Classes Under Test:
- DeduplicationError: Custom exception for deduplication-related errors

Dependencies:
- src.data_quality.normalizer.normalize_name: Name normalization preprocessing
- dedupe or recordlinkage: External fuzzy matching libraries
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import os
import json

# Import the data quality deduplicator functions (will be implemented)
from src.data_quality.deduplicator import (
    deduplicate_entities,
    DeduplicationError
)


class TestDeduplicateEntitiesBasic:
    """Test cases for basic deduplication functionality."""
    
    def test_deduplicate_entities_exact_duplicates(self):
        """Test deduplication of exact duplicate records."""
        records = [
            {"id": 1, "name": "Glucose", "formula": "C6H12O6", "mass": 180.16},
            {"id": 2, "name": "Glucose", "formula": "C6H12O6", "mass": 180.16},
            {"id": 3, "name": "Fructose", "formula": "C6H12O6", "mass": 180.16},
            {"id": 4, "name": "Glucose", "formula": "C6H12O6", "mass": 180.16}
        ]
        fields = ["name", "formula"]
        
        # Mock normalize_name to return input unchanged for this test
        with patch('src.data_quality.deduplicator.normalize_name', side_effect=lambda x: x):
            # Mock dedupe library
            with patch('src.data_quality.deduplicator.dedupe') as mock_dedupe_lib:
                mock_deduper = MagicMock()
                mock_dedupe_lib.Dedupe.return_value = mock_deduper
                
                # Mock clustering results - records 0, 1, 3 are duplicates
                mock_deduper.partition.return_value = [
                    ([0, 1, 3], [0.95, 0.95, 0.95]),  # Glucose cluster
                    ([2], [1.0])  # Fructose cluster (single record)
                ]
                
                result = deduplicate_entities(records, fields)
        
        # Should return 2 unique entities (Glucose and Fructose)
        assert len(result) == 2
        assert isinstance(result, list)
        assert all(isinstance(record, dict) for record in result)
        
        # Check that we have one Glucose and one Fructose
        names = [record['name'] for record in result]
        assert 'Glucose' in names
        assert 'Fructose' in names
        assert names.count('Glucose') == 1
        assert names.count('Fructose') == 1
    
    def test_deduplicate_entities_no_duplicates(self):
        """Test deduplication when no duplicates exist."""
        records = [
            {"id": 1, "name": "Glucose", "formula": "C6H12O6"},
            {"id": 2, "name": "Fructose", "formula": "C6H12O6"},  
            {"id": 3, "name": "Sucrose", "formula": "C12H22O11"}
        ]
        fields = ["name"]
        
        with patch('src.data_quality.deduplicator.normalize_name', side_effect=lambda x: x):
            with patch('src.data_quality.deduplicator.dedupe') as mock_dedupe_lib:
                mock_deduper = MagicMock()
                mock_dedupe_lib.Dedupe.return_value = mock_deduper
                
                # Mock clustering - each record is its own cluster
                mock_deduper.partition.return_value = [
                    ([0], [1.0]),
                    ([1], [1.0]),
                    ([2], [1.0])
                ]
                
                result = deduplicate_entities(records, fields)
        
        # Should return all 3 original records
        assert len(result) == 3
        assert result == records
    
    def test_deduplicate_entities_single_record(self):
        """Test deduplication with single record."""
        records = [{"id": 1, "name": "Glucose", "formula": "C6H12O6"}]
        fields = ["name"]
        
        with patch('src.data_quality.deduplicator.normalize_name', side_effect=lambda x: x):
            with patch('src.data_quality.deduplicator.dedupe') as mock_dedupe_lib:
                mock_deduper = MagicMock()
                mock_dedupe_lib.Dedupe.return_value = mock_deduper
                
                # Single record cluster
                mock_deduper.partition.return_value = [([0], [1.0])]
                
                result = deduplicate_entities(records, fields)
        
        # Should return the single record unchanged
        assert len(result) == 1
        assert result == records
    
    def test_deduplicate_entities_empty_input(self):
        """Test deduplication with empty input list."""
        records = []
        fields = ["name"]
        
        result = deduplicate_entities(records, fields)
        
        # Should return empty list
        assert result == []
        assert isinstance(result, list)
    
    def test_deduplicate_entities_output_format(self):
        """Test that output format matches expected structure."""
        records = [
            {"id": 1, "name": "Alpha-Glucose", "type": "sugar", "mass": 180.16},
            {"id": 2, "name": "alpha-glucose", "type": "sugar", "mass": 180.16}
        ]
        fields = ["name", "type"]
        
        with patch('src.data_quality.deduplicator.normalize_name', side_effect=lambda x: x.lower()):
            with patch('src.data_quality.deduplicator.dedupe') as mock_dedupe_lib:
                mock_deduper = MagicMock()
                mock_dedupe_lib.Dedupe.return_value = mock_deduper
                
                # Mock clustering - both records are duplicates
                mock_deduper.partition.return_value = [([0, 1], [0.95, 0.95])]
                
                result = deduplicate_entities(records, fields)
        
        # Should return list of dictionaries
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        
        # Consolidated record should contain all expected fields
        consolidated = result[0]
        assert "id" in consolidated
        assert "name" in consolidated
        assert "type" in consolidated
        assert "mass" in consolidated


class TestDeduplicateEntitiesApproximateMatches:
    """Test cases for approximate matching and minor variations."""
    
    @patch('src.data_quality.deduplicator.normalize_name')
    @patch('src.data_quality.deduplicator.dedupe')
    def test_deduplicate_entities_minor_variations(self, mock_dedupe_lib, mock_normalize):
        """Test deduplication of records with minor variations."""
        records = [
            {"id": 1, "name": "Arabidopsis thaliana", "type": "plant"},
            {"id": 2, "name": "Arabidopsis Thaliana", "type": "plant"},
            {"id": 3, "name": "Arabidopsis  thaliana ", "type": "plant"},  # Extra whitespace
            {"id": 4, "name": "Brassica napus", "type": "plant"}
        ]
        fields = ["name", "type"]
        
        # Mock normalize_name to return standardized names
        mock_normalize.side_effect = lambda x: x.strip().title()
        
        mock_deduper = MagicMock()
        mock_dedupe_lib.Dedupe.return_value = mock_deduper
        
        # Mock clustering - first 3 records are similar enough to be duplicates
        mock_deduper.partition.return_value = [
            ([0, 1, 2], [0.92, 0.88, 0.90]),  # Arabidopsis cluster
            ([3], [1.0])  # Brassica cluster (single record)
        ]
        
        result = deduplicate_entities(records, fields)
        
        # Should return 2 unique entities
        assert len(result) == 2
        
        # Check that normalization was called
        assert mock_normalize.call_count >= 4  # At least once per record name
        
        # Verify dedupe was initialized with correct fields
        mock_dedupe_lib.Dedupe.assert_called_once()
        call_args = mock_dedupe_lib.Dedupe.call_args[0][0]
        assert any(field_def['field'] in ['name', 'type'] for field_def in call_args)
    
    @patch('src.data_quality.deduplicator.normalize_name')
    @patch('src.data_quality.deduplicator.dedupe')
    def test_deduplicate_entities_chemical_variations(self, mock_dedupe_lib, mock_normalize):
        """Test deduplication of chemical compounds with variations."""
        records = [
            {"id": 1, "name": "alpha-D-glucose", "formula": "C6H12O6"},
            {"id": 2, "name": "α-D-glucose", "formula": "C6H12O6"},  # Greek letter
            {"id": 3, "name": "Alpha-d-Glucose", "formula": "C6H12O6"},  # Case variation
            {"id": 4, "name": "beta-carotene", "formula": "C40H56"}
        ]
        fields = ["name", "formula"]
        
        # Mock normalize_name to handle special characters
        mock_normalize.side_effect = lambda x: x.replace('α', 'alpha').title()
        
        mock_deduper = MagicMock()
        mock_dedupe_lib.Dedupe.return_value = mock_deduper
        
        # Mock clustering - first 3 records are glucose variants
        mock_deduper.partition.return_value = [
            ([0, 1, 2], [0.95, 0.90, 0.93]),  # Glucose cluster
            ([3], [1.0])  # Beta-carotene cluster
        ]
        
        result = deduplicate_entities(records, fields)
        
        assert len(result) == 2
        
        # Verify the glucose variants were consolidated
        glucose_records = [r for r in result if 'glucose' in r['name'].lower()]
        carotene_records = [r for r in result if 'carotene' in r['name'].lower()]
        
        assert len(glucose_records) == 1
        assert len(carotene_records) == 1
    
    @patch('src.data_quality.deduplicator.normalize_name')
    @patch('src.data_quality.deduplicator.recordlinkage')
    def test_deduplicate_entities_with_recordlinkage(self, mock_rl, mock_normalize):
        """Test using recordlinkage library instead of dedupe."""
        records = [
            {"id": 1, "name": "John Smith", "age": 30, "city": "New York"},
            {"id": 2, "name": "Jon Smith", "age": 30, "city": "New York"},  # Typo
            {"id": 3, "name": "Jane Doe", "age": 25, "city": "Boston"}
        ]
        fields = ["name", "age", "city"]
        
        mock_normalize.side_effect = lambda x: x.strip().title()
        
        # Mock recordlinkage components
        mock_indexer = MagicMock()
        mock_compare = MagicMock()
        mock_classifier = MagicMock()
        
        mock_rl.Index.return_value = mock_indexer
        mock_rl.Compare.return_value = mock_compare
        mock_rl.NaiveBayesClassifier.return_value = mock_classifier
        
        # Mock the comparison results
        mock_indexer.index.return_value = [(0, 1), (0, 2), (1, 2)]  # All pairs
        mock_compare.compute.return_value = MagicMock()  # Comparison vectors
        mock_classifier.predict.return_value = [True, False, False]  # Only (0,1) match
        
        # Mock this test to use recordlinkage by patching dedupe to not be available
        with patch('src.data_quality.deduplicator.dedupe', side_effect=ImportError):
            result = deduplicate_entities(records, fields)
        
        # Should still work with recordlinkage
        assert isinstance(result, list)
        assert len(result) <= len(records)  # Should not increase records


class TestDeduplicateEntitiesIntegration:
    """Test cases for integration with normalizer and external libraries."""
    
    @patch('src.data_quality.deduplicator.dedupe')
    def test_deduplicate_entities_normalizer_integration(self, mock_dedupe_lib):
        """Test integration with normalize_name function."""
        records = [
            {"id": 1, "name": "KING arthur", "title": "legendary king"},
            {"id": 2, "name": "king ARTHUR", "title": "Legendary King"},
            {"id": 3, "name": "merlin", "title": "wizard"}
        ]
        fields = ["name", "title"]
        
        mock_deduper = MagicMock()
        mock_dedupe_lib.Dedupe.return_value = mock_deduper
        
        # Mock clustering - first 2 records are duplicates after normalization
        mock_deduper.partition.return_value = [
            ([0, 1], [0.95, 0.95]),  # King Arthur cluster
            ([2], [1.0])  # Merlin cluster
        ]
        
        # Test with real normalize_name function
        from src.data_quality.normalizer import normalize_name
        
        result = deduplicate_entities(records, fields)
        
        # Should return 2 unique entities
        assert len(result) == 2
        
        # Verify that normalization was applied
        # The test implicitly checks this through the mocked clustering results
        assert any('arthur' in str(record).lower() for record in result)
        assert any('merlin' in str(record).lower() for record in result)
    
    @patch('src.data_quality.deduplicator.normalize_name')
    @patch('src.data_quality.deduplicator.dedupe')
    def test_deduplicate_entities_with_settings_file(self, mock_dedupe_lib, mock_normalize):
        """Test deduplication with custom settings file."""
        records = [
            {"id": 1, "name": "Test Entity 1", "category": "A"},
            {"id": 2, "name": "Test Entity 2", "category": "B"}
        ]
        fields = ["name", "category"]
        settings_file = "/path/to/settings.json"
        
        mock_normalize.side_effect = lambda x: x
        
        mock_deduper = MagicMock()
        mock_dedupe_lib.Dedupe.return_value = mock_deduper
        mock_deduper.partition.return_value = [([0], [1.0]), ([1], [1.0])]
        
        # Mock file operations
        with patch('builtins.open', mock_open(read_data='{"threshold": 0.8}')):
            with patch('os.path.exists', return_value=True):
                result = deduplicate_entities(records, fields, settings_file=settings_file)
        
        assert len(result) == 2
        
        # Verify settings file was used
        mock_deduper.prepare_training.assert_called_once()
    
    @patch('src.data_quality.deduplicator.normalize_name')
    @patch('src.data_quality.deduplicator.dedupe')
    def test_deduplicate_entities_with_training_file(self, mock_dedupe_lib, mock_normalize):
        """Test deduplication with training data file."""
        records = [
            {"id": 1, "name": "Entity A", "type": "compound"},
            {"id": 2, "name": "Entity B", "type": "compound"}
        ]
        fields = ["name", "type"]
        training_file = "/path/to/training.json"
        
        mock_normalize.side_effect = lambda x: x
        
        mock_deduper = MagicMock()
        mock_dedupe_lib.Dedupe.return_value = mock_deduper
        mock_deduper.partition.return_value = [([0], [1.0]), ([1], [1.0])]
        
        # Mock file operations
        with patch('builtins.open', mock_open(read_data='[]')):
            with patch('os.path.exists', return_value=True):
                result = deduplicate_entities(records, fields, training_file=training_file)
        
        assert len(result) == 2
        
        # Verify training data was used
        mock_deduper.prepare_training.assert_called_once()


class TestDeduplicateEntitiesErrorHandling:
    """Test cases for error handling and input validation."""
    
    def test_deduplicate_entities_none_records(self):
        """Test error handling for None records input."""
        with pytest.raises(DeduplicationError, match="Records cannot be None"):
            deduplicate_entities(None, ["name"])
    
    def test_deduplicate_entities_non_list_records(self):
        """Test error handling for non-list records input."""
        with pytest.raises(DeduplicationError, match="Records must be a list"):
            deduplicate_entities("not a list", ["name"])
        
        with pytest.raises(DeduplicationError, match="Records must be a list"):
            deduplicate_entities({"not": "a list"}, ["name"])
    
    def test_deduplicate_entities_none_fields(self):
        """Test error handling for None fields input."""
        records = [{"name": "test"}]
        
        with pytest.raises(DeduplicationError, match="Fields cannot be None"):
            deduplicate_entities(records, None)
    
    def test_deduplicate_entities_non_list_fields(self):
        """Test error handling for non-list fields input."""
        records = [{"name": "test"}]
        
        with pytest.raises(DeduplicationError, match="Fields must be a list"):
            deduplicate_entities(records, "name")
    
    def test_deduplicate_entities_empty_fields(self):
        """Test error handling for empty fields list."""
        records = [{"name": "test"}]
        
        with pytest.raises(DeduplicationError, match="Fields list cannot be empty"):
            deduplicate_entities(records, [])
    
    def test_deduplicate_entities_non_dict_records(self):
        """Test error handling for non-dict items in records."""
        records = [
            {"name": "valid record"},
            "invalid record",  # Not a dict
            {"name": "another valid record"}
        ]
        fields = ["name"]
        
        with pytest.raises(DeduplicationError, match="All records must be dictionaries"):
            deduplicate_entities(records, fields)
    
    def test_deduplicate_entities_missing_fields(self):
        """Test error handling for records missing required fields."""
        records = [
            {"name": "Test Entity", "type": "A"},
            {"name": "Another Entity"},  # Missing 'type' field
            {"type": "B"}  # Missing 'name' field
        ]
        fields = ["name", "type"]
        
        with pytest.raises(DeduplicationError, match="Record .* missing required field"):
            deduplicate_entities(records, fields)
    
    def test_deduplicate_entities_non_string_field_values(self):
        """Test error handling for unsupported field value types."""
        records = [
            {"name": "Valid Entity", "priority": 1},
            {"name": "Another Entity", "priority": {"level": "high"}}  # Dict value not supported
        ]
        fields = ["name", "priority"]
        
        with pytest.raises(DeduplicationError, match="Field .* must be a string, int, or float"):
            deduplicate_entities(records, fields)
    
    def test_deduplicate_entities_invalid_settings_file(self):
        """Test error handling for invalid settings file."""
        records = [{"name": "test"}]
        fields = ["name"]
        
        # Test non-existent file
        with pytest.raises(DeduplicationError, match="Settings file .* does not exist"):
            deduplicate_entities(records, fields, settings_file="/nonexistent/file.json")
    
    def test_deduplicate_entities_invalid_training_file(self):
        """Test error handling for invalid training file."""
        records = [{"name": "test"}]
        fields = ["name"]
        
        # Test non-existent file
        with pytest.raises(DeduplicationError, match="Training file .* does not exist"):
            deduplicate_entities(records, fields, training_file="/nonexistent/file.json")
    
    @patch('src.data_quality.deduplicator.normalize_name')
    def test_deduplicate_entities_normalization_error(self, mock_normalize):
        """Test error handling when normalization fails."""
        records = [{"name": "test entity", "type": "test"}]
        fields = ["name"]
        
        # Mock normalize_name to raise an error
        mock_normalize.side_effect = Exception("Normalization failed")
        
        with pytest.raises(DeduplicationError, match="Error during name normalization"):
            deduplicate_entities(records, fields)
    
    @patch('src.data_quality.deduplicator.normalize_name')
    @patch('src.data_quality.deduplicator.dedupe')
    def test_deduplicate_entities_library_error(self, mock_dedupe_lib, mock_normalize):
        """Test error handling when deduplication library fails."""
        records = [{"name": "test", "type": "A"}]
        fields = ["name"]
        
        mock_normalize.side_effect = lambda x: x
        
        # Mock dedupe to raise an error
        mock_dedupe_lib.Dedupe.side_effect = Exception("Dedupe library error")
        
        with pytest.raises(DeduplicationError, match="Error during deduplication"):
            deduplicate_entities(records, fields)


class TestDeduplicateEntitiesEdgeCases:
    """Test cases for edge cases and special scenarios."""
    
    @patch('src.data_quality.deduplicator.normalize_name')
    @patch('src.data_quality.deduplicator.dedupe')
    def test_deduplicate_entities_large_dataset(self, mock_dedupe_lib, mock_normalize):
        """Test deduplication with large dataset."""
        # Create a large dataset
        records = [
            {"id": i, "name": f"Entity {i}", "category": f"Cat {i % 10}"}
            for i in range(1000)
        ]
        fields = ["name", "category"]
        
        mock_normalize.side_effect = lambda x: x
        
        mock_deduper = MagicMock()
        mock_dedupe_lib.Dedupe.return_value = mock_deduper
        
        # Mock clustering - each record is unique
        mock_deduper.partition.return_value = [([i], [1.0]) for i in range(1000)]
        
        result = deduplicate_entities(records, fields)
        
        # Should handle large datasets without errors
        assert len(result) == 1000
        assert isinstance(result, list)
    
    @patch('src.data_quality.deduplicator.normalize_name')
    @patch('src.data_quality.deduplicator.dedupe')
    def test_deduplicate_entities_unicode_content(self, mock_dedupe_lib, mock_normalize):
        """Test deduplication with Unicode content."""
        records = [
            {"id": 1, "name": "café", "description": "coffee shop"},
            {"id": 2, "name": "cafe", "description": "coffee shop"},  # Without accent
            {"id": 3, "name": "naïve", "description": "innocent"},
            {"id": 4, "name": "α-glucose", "description": "sugar molecule"}
        ]
        fields = ["name", "description"]
        
        # Mock normalize_name to handle Unicode
        mock_normalize.side_effect = lambda x: x.lower().replace('é', 'e').replace('ï', 'i')
        
        mock_deduper = MagicMock()
        mock_dedupe_lib.Dedupe.return_value = mock_deduper
        
        # Mock clustering - café and cafe are duplicates
        mock_deduper.partition.return_value = [
            ([0, 1], [0.95, 0.95]),  # Café cluster
            ([2], [1.0]),  # Naïve cluster
            ([3], [1.0])   # α-glucose cluster
        ]
        
        result = deduplicate_entities(records, fields)
        
        assert len(result) == 3
        assert all(isinstance(record, dict) for record in result)
    
    @patch('src.data_quality.deduplicator.normalize_name')
    @patch('src.data_quality.deduplicator.dedupe')
    def test_deduplicate_entities_special_characters(self, mock_dedupe_lib, mock_normalize):
        """Test deduplication with special characters and punctuation."""
        records = [
            {"id": 1, "name": "α-D-glucose", "formula": "C6H12O6"},
            {"id": 2, "name": "alpha-D-glucose", "formula": "C6H12O6"},
            {"id": 3, "name": "compound-123", "formula": "Unknown"},
            {"id": 4, "name": "compound_123", "formula": "Unknown"}  # Underscore vs hyphen
        ]
        fields = ["name", "formula"]
        
        mock_normalize.side_effect = lambda x: x.replace('α', 'alpha').replace('_', '-')
        
        mock_deduper = MagicMock()
        mock_dedupe_lib.Dedupe.return_value = mock_deduper
        
        # Mock clustering - similar compounds are duplicates
        mock_deduper.partition.return_value = [
            ([0, 1], [0.92, 0.92]),  # Glucose variants
            ([2, 3], [0.90, 0.90])   # Compound variants
        ]
        
        result = deduplicate_entities(records, fields)
        
        assert len(result) == 2
    
    @patch('src.data_quality.deduplicator.normalize_name')
    @patch('src.data_quality.deduplicator.dedupe')
    def test_deduplicate_entities_mixed_data_types(self, mock_dedupe_lib, mock_normalize):
        """Test deduplication with mixed data types in non-comparison fields."""
        records = [
            {"id": 1, "name": "Entity A", "count": 10, "active": True, "metadata": {"key": "value"}},
            {"id": 2, "name": "Entity A", "count": 15, "active": False, "metadata": {"key": "other"}},
            {"id": 3, "name": "Entity B", "count": 5, "active": True, "metadata": None}
        ]
        fields = ["name"]  # Only compare name field
        
        mock_normalize.side_effect = lambda x: x
        
        mock_deduper = MagicMock()
        mock_dedupe_lib.Dedupe.return_value = mock_deduper
        
        # Mock clustering - first two records are duplicates
        mock_deduper.partition.return_value = [
            ([0, 1], [0.95, 0.95]),  # Entity A cluster
            ([2], [1.0])  # Entity B cluster
        ]
        
        result = deduplicate_entities(records, fields)
        
        assert len(result) == 2
        
        # Consolidated Entity A should preserve non-comparison fields from first record
        entity_a = next(r for r in result if r['name'] == 'Entity A')
        assert 'count' in entity_a
        assert 'active' in entity_a
        assert 'metadata' in entity_a


class TestDeduplicateEntitiesPerformance:
    """Test cases for performance considerations and memory usage."""
    
    @patch('src.data_quality.deduplicator.normalize_name')
    @patch('src.data_quality.deduplicator.dedupe')
    def test_deduplicate_entities_memory_efficiency(self, mock_dedupe_lib, mock_normalize):
        """Test memory efficiency with multiple deduplication operations."""
        import sys
        
        # Create test data
        records = [
            {"id": i, "name": f"Test Entity {i}", "data": "x" * 100}
            for i in range(100)
        ]
        fields = ["name"]
        
        mock_normalize.side_effect = lambda x: x
        
        mock_deduper = MagicMock()
        mock_dedupe_lib.Dedupe.return_value = mock_deduper
        mock_deduper.partition.return_value = [([i], [1.0]) for i in range(100)]
        
        # Get initial memory snapshot
        initial_refs = sys.getrefcount(records)
        
        # Perform multiple deduplication operations
        for _ in range(10):
            result = deduplicate_entities(records.copy(), fields)
            assert len(result) == 100
        
        # Memory should not grow excessively
        final_refs = sys.getrefcount(records)
        assert final_refs <= initial_refs + 20  # Allow reasonable growth
    
    @patch('src.data_quality.deduplicator.normalize_name')
    @patch('src.data_quality.deduplicator.dedupe')
    def test_deduplicate_entities_timeout_handling(self, mock_dedupe_lib, mock_normalize):
        """Test handling of long-running operations."""
        records = [{"id": i, "name": f"Entity {i}"} for i in range(10)]
        fields = ["name"]
        
        mock_normalize.side_effect = lambda x: x
        
        mock_deduper = MagicMock()
        mock_dedupe_lib.Dedupe.return_value = mock_deduper
        
        # Mock a slow partition operation
        import time
        def slow_partition(data):
            time.sleep(0.1)  # Simulate slow operation
            return [([i], [1.0]) for i in range(len(data))]
        
        mock_deduper.partition.side_effect = slow_partition
        
        # Should complete without timeout errors (basic test)
        result = deduplicate_entities(records, fields)
        assert len(result) == 10


class TestDeduplicationErrorClass:
    """Test cases for DeduplicationError exception class.""" 
    
    def test_deduplication_error_inheritance(self):
        """Test that DeduplicationError properly inherits from Exception."""
        error = DeduplicationError("Test error message")
        assert isinstance(error, Exception)
        assert str(error) == "Test error message"
    
    def test_deduplication_error_empty_message(self):
        """Test DeduplicationError with empty message."""
        error = DeduplicationError("")
        assert isinstance(error, Exception)
        assert str(error) == ""
    
    def test_deduplication_error_with_details(self):
        """Test DeduplicationError with detailed message."""
        details = "Field 'name' is missing from record at index 5"
        error = DeduplicationError(f"Validation failed: {details}")
        assert str(error) == f"Validation failed: {details}"


# Fixtures for common test data
@pytest.fixture
def sample_entity_records():
    """Fixture providing sample entity records for testing."""
    return [
        {"id": 1, "name": "Glucose", "formula": "C6H12O6", "type": "sugar", "mass": 180.16},
        {"id": 2, "name": "glucose", "formula": "C6H12O6", "type": "Sugar", "mass": 180.16},
        {"id": 3, "name": "Fructose", "formula": "C6H12O6", "type": "sugar", "mass": 180.16},
        {"id": 4, "name": "Alpha-D-Glucose", "formula": "C6H12O6", "type": "sugar", "mass": 180.16},
        {"id": 5, "name": "Sucrose", "formula": "C12H22O11", "type": "sugar", "mass": 342.30}
    ]


@pytest.fixture
def sample_organism_records():
    """Fixture providing sample organism records for testing."""
    return [
        {"id": 1, "name": "Arabidopsis thaliana", "kingdom": "Plantae", "family": "Brassicaceae"},
        {"id": 2, "name": "Arabidopsis Thaliana", "kingdom": "Plantae", "family": "Brassicaceae"},
        {"id": 3, "name": "Arabidopsis  thaliana ", "kingdom": "Plantae", "family": "Brassicaceae"},
        {"id": 4, "name": "Homo sapiens", "kingdom": "Animalia", "family": "Hominidae"},
        {"id": 5, "name": "Escherichia coli", "kingdom": "Bacteria", "family": "Enterobacteriaceae"}
    ]


@pytest.fixture
def sample_chemical_records():
    """Fixture providing sample chemical compound records for testing."""
    return [
        {"id": 1, "name": "α-D-glucose", "iupac": "alpha-D-glucopyranose", "formula": "C6H12O6"},
        {"id": 2, "name": "alpha-D-glucose", "iupac": "alpha-D-glucopyranose", "formula": "C6H12O6"},
        {"id": 3, "name": "α-d-glucose", "iupac": "Alpha-D-Glucopyranose", "formula": "C6H12O6"},
        {"id": 4, "name": "β-carotene", "iupac": "beta-carotene", "formula": "C40H56"},
        {"id": 5, "name": "beta-carotene", "iupac": "beta-carotene", "formula": "C40H56"}
    ]


@pytest.fixture
def sample_fields_basic():
    """Fixture providing basic field configuration for testing."""
    return ["name", "type"]


@pytest.fixture
def sample_fields_chemical():
    """Fixture providing chemical-specific field configuration."""
    return ["name", "formula", "iupac"]


@pytest.fixture
def sample_fields_organism():
    """Fixture providing organism-specific field configuration."""
    return ["name", "kingdom", "family"]


@pytest.fixture
def temp_settings_file():
    """Fixture providing temporary settings file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        settings = {
            "threshold": 0.85,
            "algorithm": "dedupe",
            "fields": {
                "name": {"type": "String", "weight": 1.0},
                "type": {"type": "String", "weight": 0.8}
            }
        }
        json.dump(settings, f)
        f.flush()
        
        yield f.name
        
        # Cleanup
        try:
            os.unlink(f.name)
        except OSError:
            pass


@pytest.fixture  
def temp_training_file():
    """Fixture providing temporary training file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        training_data = [
            {"distinct": [0, 1], "match": [2, 3]},  # Example training pairs
            {"distinct": [0, 4], "match": [1, 2]}
        ]
        json.dump(training_data, f)
        f.flush()
        
        yield f.name
        
        # Cleanup
        try:
            os.unlink(f.name)
        except OSError:
            pass


# Parametrized test configurations
@pytest.mark.parametrize("fields,expected_unique", [
    (["name"], 3),  # Should dedupe based on name only
    (["name", "formula"], 3),  # Should dedupe based on name and formula
    (["formula"], 2),  # Should dedupe based on formula only (glucose vs sucrose)
])
@patch('src.data_quality.deduplicator.normalize_name')
@patch('src.data_quality.deduplicator.dedupe')
def test_deduplicate_entities_parametrized_fields(mock_dedupe_lib, mock_normalize, 
                                                  sample_entity_records, fields, expected_unique):
    """Parametrized test for different field combinations."""
    mock_normalize.side_effect = lambda x: x.lower().strip()
    
    mock_deduper = MagicMock()
    mock_dedupe_lib.Dedupe.return_value = mock_deduper
    
    # Mock appropriate clustering based on expected unique count
    if expected_unique == 3:
        # Glucose variants, Fructose, Sucrose
        mock_deduper.partition.return_value = [
            ([0, 1, 3], [0.95, 0.95, 0.90]),  # Glucose cluster
            ([2], [1.0]),  # Fructose
            ([4], [1.0])   # Sucrose
        ]
    elif expected_unique == 2:
        # All C6H12O6 compounds vs C12H22O11
        mock_deduper.partition.return_value = [
            ([0, 1, 2, 3], [0.95, 0.95, 0.85, 0.90]),  # C6H12O6 cluster
            ([4], [1.0])  # C12H22O11
        ]
    
    result = deduplicate_entities(sample_entity_records, fields)
    assert len(result) == expected_unique


# Mark all tests in this module as data quality related
pytestmark = pytest.mark.unit