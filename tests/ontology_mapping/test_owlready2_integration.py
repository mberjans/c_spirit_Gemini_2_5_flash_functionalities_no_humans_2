"""
Integration tests for Owlready2 functionality in entity_mapper.

This module tests the new Owlready2 integration features that allow users to pass
Owlready2 ontology objects directly to the mapping function, providing better
integration for users who already have loaded ontologies.

Test Coverage:
- Owlready2 ontology object detection
- IRI extraction from Owlready2 objects
- Target ontology validation with both string IRIs and Owlready2 objects
- Error handling for invalid Owlready2 objects
- Backward compatibility with string IRIs
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.ontology_mapping.entity_mapper import (
    map_entities_to_ontology,
    _is_owlready2_ontology,
    _extract_iri_from_owlready2_ontology,
    _validate_target_ontology,
    InvalidOwlready2ObjectError,
    OWLREADY2_AVAILABLE
)


class TestOwlready2Detection:
    """Test Owlready2 ontology object detection functionality."""
    
    def test_is_owlready2_ontology_when_not_available(self):
        """Test detection when Owlready2 is not available."""
        # Mock OWLREADY2_AVAILABLE to be False
        with patch('src.ontology_mapping.entity_mapper.OWLREADY2_AVAILABLE', False):
            result = _is_owlready2_ontology("any_object")
            assert result is False
    
    def test_is_owlready2_ontology_with_non_ontology(self):
        """Test detection with non-ontology objects."""
        test_objects = [
            "string",
            123,
            [],
            {},
            None,
            object()
        ]
        
        for obj in test_objects:
            result = _is_owlready2_ontology(obj)
            assert result is False
    
    @patch('src.ontology_mapping.entity_mapper.OWLREADY2_AVAILABLE', True)
    def test_is_owlready2_ontology_with_mock_ontology(self):
        """Test detection with mocked Owlready2 ontology."""
        # Create a mock ontology object
        mock_ontology = Mock()
        
        # Since OWLREADY2_AVAILABLE is True but owlready2 isn't actually available,
        # this should return False when isinstance check fails
        result = _is_owlready2_ontology(mock_ontology)
        assert result is False
        
        # Test that the function handles exceptions gracefully
        with patch('src.ontology_mapping.entity_mapper.owlready2', None):
            result = _is_owlready2_ontology(mock_ontology)
            assert result is False


class TestIRIExtraction:
    """Test IRI extraction from Owlready2 ontology objects."""
    
    def test_extract_iri_when_owlready2_not_available(self):
        """Test IRI extraction when Owlready2 is not available."""
        with patch('src.ontology_mapping.entity_mapper.OWLREADY2_AVAILABLE', False):
            with pytest.raises(InvalidOwlready2ObjectError) as exc_info:
                _extract_iri_from_owlready2_ontology("any_object")
            
            assert "Owlready2 is not available" in str(exc_info.value)
            assert "pip install owlready2" in str(exc_info.value)
    
    @patch('src.ontology_mapping.entity_mapper.OWLREADY2_AVAILABLE', True)
    @patch('src.ontology_mapping.entity_mapper._is_owlready2_ontology')
    def test_extract_iri_with_invalid_object(self, mock_is_ontology):
        """Test IRI extraction with invalid object."""
        mock_is_ontology.return_value = False
        
        with pytest.raises(InvalidOwlready2ObjectError) as exc_info:
            _extract_iri_from_owlready2_ontology("not_an_ontology")
        
        assert "not a valid Owlready2 ontology" in str(exc_info.value)
    
    @patch('src.ontology_mapping.entity_mapper.OWLREADY2_AVAILABLE', True)
    @patch('src.ontology_mapping.entity_mapper._is_owlready2_ontology')
    def test_extract_iri_success(self, mock_is_ontology):
        """Test successful IRI extraction."""
        mock_is_ontology.return_value = True
        
        # Create mock ontology with base_iri
        mock_ontology = Mock()
        mock_ontology.base_iri = "http://example.org/ontology.owl"
        
        result = _extract_iri_from_owlready2_ontology(mock_ontology)
        assert result == "http://example.org/ontology.owl"
    
    @patch('src.ontology_mapping.entity_mapper.OWLREADY2_AVAILABLE', True)
    @patch('src.ontology_mapping.entity_mapper._is_owlready2_ontology')
    def test_extract_iri_removes_trailing_slash(self, mock_is_ontology):
        """Test that trailing slash is removed from IRI."""
        mock_is_ontology.return_value = True
        
        mock_ontology = Mock()
        mock_ontology.base_iri = "http://example.org/ontology.owl/"
        
        result = _extract_iri_from_owlready2_ontology(mock_ontology)
        assert result == "http://example.org/ontology.owl"
    
    @patch('src.ontology_mapping.entity_mapper.OWLREADY2_AVAILABLE', True)
    @patch('src.ontology_mapping.entity_mapper._is_owlready2_ontology')
    def test_extract_iri_no_base_iri(self, mock_is_ontology):
        """Test IRI extraction when ontology has no base_iri."""
        mock_is_ontology.return_value = True
        
        mock_ontology = Mock()
        mock_ontology.base_iri = None
        
        with pytest.raises(InvalidOwlready2ObjectError) as exc_info:
            _extract_iri_from_owlready2_ontology(mock_ontology)
        
        assert "does not have a valid base IRI" in str(exc_info.value)
    
    @patch('src.ontology_mapping.entity_mapper.OWLREADY2_AVAILABLE', True)
    @patch('src.ontology_mapping.entity_mapper._is_owlready2_ontology')
    def test_extract_iri_attribute_error(self, mock_is_ontology):
        """Test IRI extraction when base_iri attribute is missing."""
        mock_is_ontology.return_value = True
        
        mock_ontology = Mock()
        del mock_ontology.base_iri  # Remove the attribute
        
        with pytest.raises(InvalidOwlready2ObjectError) as exc_info:
            _extract_iri_from_owlready2_ontology(mock_ontology)
        
        assert "Unable to extract IRI" in str(exc_info.value)


class TestTargetOntologyValidation:
    """Test target ontology validation with both string IRIs and Owlready2 objects."""
    
    def test_validate_target_ontology_none(self):
        """Test validation with None input."""
        with pytest.raises(ValueError) as exc_info:
            _validate_target_ontology(None)
        
        assert "Invalid ontology IRI: cannot be None" in str(exc_info.value)
    
    def test_validate_target_ontology_string_iri(self):
        """Test validation with valid string IRI."""
        test_iri = "http://example.org/ontology.owl"
        result = _validate_target_ontology(test_iri)
        assert result == test_iri
    
    def test_validate_target_ontology_invalid_string(self):
        """Test validation with invalid string IRI."""
        with pytest.raises(ValueError) as exc_info:
            _validate_target_ontology("invalid_url")
        
        assert "Invalid ontology IRI" in str(exc_info.value)
    
    @patch('src.ontology_mapping.entity_mapper._is_owlready2_ontology')
    @patch('src.ontology_mapping.entity_mapper._extract_iri_from_owlready2_ontology')
    def test_validate_target_ontology_owlready2(self, mock_extract, mock_is_ontology):
        """Test validation with Owlready2 ontology object."""
        mock_is_ontology.return_value = True
        mock_extract.return_value = "http://example.org/ontology.owl"
        
        mock_ontology = Mock()
        result = _validate_target_ontology(mock_ontology)
        
        assert result == "http://example.org/ontology.owl"
        mock_is_ontology.assert_called_once_with(mock_ontology)
        mock_extract.assert_called_once_with(mock_ontology)
    
    def test_validate_target_ontology_invalid_type(self):
        """Test validation with invalid object type."""
        with pytest.raises(ValueError) as exc_info:
            _validate_target_ontology(123)
        
        error_message = str(exc_info.value)
        assert "Invalid ontology IRI" in error_message
        assert "must be a string IRI or Owlready2 ontology object" in error_message
        assert "<class 'int'>" in error_message


class TestOwlready2Integration:
    """Test integration of Owlready2 functionality with main mapping function."""
    
    @patch('src.ontology_mapping.entity_mapper.text2term.map_terms')
    def test_map_entities_with_string_iri_backward_compatibility(self, mock_map_terms):
        """Test that string IRIs still work (backward compatibility)."""
        # Setup mock return value
        mock_df = pd.DataFrame({
            'Source Term': ['glucose'],
            'Mapped Term IRI': ['http://example.org/glucose'],
            'Mapping Score': [0.9]
        })
        mock_map_terms.return_value = mock_df
        
        entities = ['glucose']
        ontology_iri = 'http://example.org/ontology.owl'
        
        result = map_entities_to_ontology(
            entities=entities,
            target_ontology=ontology_iri
        )
        
        # Verify text2term was called with the correct IRI
        mock_map_terms.assert_called_once()
        call_args = mock_map_terms.call_args[1]
        assert call_args['target_ontology'] == ontology_iri
        assert call_args['source_terms'] == entities
        
        # Verify result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    @patch('src.ontology_mapping.entity_mapper.text2term.map_terms')
    @patch('src.ontology_mapping.entity_mapper._is_owlready2_ontology')
    @patch('src.ontology_mapping.entity_mapper._extract_iri_from_owlready2_ontology')
    def test_map_entities_with_owlready2_object(self, mock_extract, mock_is_ontology, mock_map_terms):
        """Test mapping with Owlready2 ontology object."""
        # Setup mocks
        mock_is_ontology.return_value = True
        mock_extract.return_value = "http://example.org/ontology.owl"
        
        mock_df = pd.DataFrame({
            'Source Term': ['glucose'],
            'Mapped Term IRI': ['http://example.org/glucose'],
            'Mapping Score': [0.9]
        })
        mock_map_terms.return_value = mock_df
        
        # Create mock ontology
        mock_ontology = Mock()
        entities = ['glucose']
        
        result = map_entities_to_ontology(
            entities=entities,
            target_ontology=mock_ontology
        )
        
        # Verify the IRI was extracted and used
        mock_is_ontology.assert_called_once_with(mock_ontology)
        mock_extract.assert_called_once_with(mock_ontology)
        
        # Verify text2term was called with extracted IRI
        mock_map_terms.assert_called_once()
        call_args = mock_map_terms.call_args[1]
        assert call_args['target_ontology'] == "http://example.org/ontology.owl"
        
        # Verify result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_map_entities_with_invalid_owlready2_object(self):
        """Test error handling with invalid Owlready2 object."""
        entities = ['glucose']
        invalid_object = object()  # Not an Owlready2 ontology
        
        with pytest.raises(ValueError) as exc_info:
            map_entities_to_ontology(
                entities=entities,
                target_ontology=invalid_object
            )
        
        error_message = str(exc_info.value)
        assert "Invalid ontology IRI" in error_message
        assert "must be a string IRI or Owlready2 ontology object" in error_message


class TestBackwardCompatibility:
    """Test that all existing functionality remains unchanged."""
    
    @patch('src.ontology_mapping.entity_mapper.text2term.map_terms')
    def test_all_existing_parameters_work(self, mock_map_terms):
        """Test that all existing parameters and functionality work unchanged."""
        mock_df = pd.DataFrame({
            'Source Term': ['glucose', 'fructose'],
            'Mapped Term IRI': ['http://example.org/glucose', 'http://example.org/fructose'],
            'Mapping Score': [0.9, 0.8]
        })
        mock_map_terms.return_value = mock_df
        
        entities = ['glucose', 'fructose']
        ontology_iri = 'http://example.org/ontology.owl'
        
        result = map_entities_to_ontology(
            entities=entities,
            target_ontology=ontology_iri,  # Changed parameter name but same functionality
            mapping_method='levenshtein',
            min_score=0.7,
            term_type='class'
        )
        
        # Verify all parameters were passed correctly
        mock_map_terms.assert_called_once()
        call_args = mock_map_terms.call_args[1]
        assert call_args['source_terms'] == entities
        assert call_args['target_ontology'] == ontology_iri
        assert call_args['min_score'] == 0.7
        assert call_args['term_type'] == 'class'
        
        # Verify result processing still works
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2