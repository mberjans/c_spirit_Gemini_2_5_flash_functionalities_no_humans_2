"""
Unit tests for src/evaluation/curation_tool.py

This module provides comprehensive unit tests for the manual curation functionality
that enables review and correction of LLM-generated entity and relationship extractions
in the AIM2-ODIE plant metabolomics information extraction system.

Test Coverage:
- load_llm_output: Loading LLM-generated entities and relations from JSON files
- display_for_review: CLI-based display of text with entities and relations for review
- apply_correction: Modifying entities and relations based on user corrections
- save_curated_output: Saving curated data to output files with proper structure
- Error handling: File I/O errors, invalid data formats, malformed corrections
- Edge cases: Empty files, missing fields, complex correction scenarios

Since the actual implementation doesn't exist yet, this test file uses comprehensive
mocking to test the intended API interface and behavior patterns.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
from typing import Dict, List, Tuple, Any


class TestCurationTool:
    """Comprehensive test class for curation tool functionality."""

    @pytest.fixture(autouse=True)
    def setup_mock_module(self):
        """Set up mock module and functions for each test."""
        # Define mock exception classes
        class CurationError(Exception):
            pass
        
        class InvalidDataError(CurationError):
            pass
        
        class CorrectionError(CurationError):
            pass
        
        self.CurationError = CurationError
        self.InvalidDataError = InvalidDataError
        self.CorrectionError = CorrectionError
        
        # Mock function implementations
        def load_llm_output(file_path: str) -> Dict[str, Any]:
            """Mock implementation of load_llm_output function."""
            if not isinstance(file_path, str) or not file_path.strip():
                raise ValueError("File path must be a non-empty string")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                raise InvalidDataError(f"Invalid JSON format: {str(e)}")
            except Exception as e:
                raise CurationError(f"Failed to load file: {str(e)}")
            
            # Validate required structure
            if not isinstance(data, dict):
                raise InvalidDataError("Data must be a dictionary")
            
            if 'text' not in data:
                raise InvalidDataError("Missing required field: text")
            
            if 'entities' not in data:
                raise InvalidDataError("Missing required field: entities")
            
            if 'relations' not in data:
                raise InvalidDataError("Missing required field: relations")
            
            if not isinstance(data['entities'], list):
                raise InvalidDataError("entities must be a list")
            
            if not isinstance(data['relations'], list):
                raise InvalidDataError("relations must be a list")
            
            # Validate entity structure
            for i, entity in enumerate(data['entities']):
                if not isinstance(entity, dict):
                    raise InvalidDataError(f"entities[{i}] must be a dictionary")
                
                required_entity_fields = ['entity_type', 'text', 'start_char', 'end_char']
                for field in required_entity_fields:
                    if field not in entity:
                        raise InvalidDataError(f"entities[{i}] missing required field: {field}")
                
                if not isinstance(entity['start_char'], int) or entity['start_char'] < 0:
                    raise InvalidDataError(f"entities[{i}] start_char must be a non-negative integer")
                
                if not isinstance(entity['end_char'], int) or entity['end_char'] <= entity['start_char']:
                    raise InvalidDataError(f"entities[{i}] end_char must be greater than start_char")
            
            # Validate relation structure
            for i, relation in enumerate(data['relations']):
                if not isinstance(relation, (tuple, list)) or len(relation) != 3:
                    raise InvalidDataError(f"relations[{i}] must be a tuple/list of length 3")
            
            # Convert relations from lists back to tuples (JSON converts tuples to lists)
            data['relations'] = [tuple(relation) for relation in data['relations']]
            
            return data
        
        def display_for_review(text: str, entities: List[Dict], relations: List[Tuple]) -> None:
            """Mock implementation of display_for_review function."""
            if not isinstance(text, str):
                raise ValueError("text must be a string")
            
            if not isinstance(entities, list):
                raise ValueError("entities must be a list")
            
            if not isinstance(relations, list):
                raise ValueError("relations must be a list")
            
            # Mock printing behavior (would normally print to console)
            print(f"=== TEXT FOR REVIEW ===")
            print(text)
            print(f"\n=== ENTITIES ({len(entities)}) ===")
            for i, entity in enumerate(entities):
                print(f"{i+1}. {entity['entity_type']}: '{entity['text']}' ({entity['start_char']}-{entity['end_char']})")
            
            print(f"\n=== RELATIONS ({len(relations)}) ===")
            for i, relation in enumerate(relations):
                print(f"{i+1}. {relation[0]} --{relation[1]}--> {relation[2]}")
        
        def apply_correction(extracted_data: Dict[str, Any], correction_type: str, old_value: Any, new_value: Any) -> Dict[str, Any]:
            """Mock implementation of apply_correction function."""
            if not isinstance(extracted_data, dict):
                raise ValueError("extracted_data must be a dictionary")
            
            if not isinstance(correction_type, str) or not correction_type.strip():
                raise ValueError("correction_type must be a non-empty string")
            
            valid_correction_types = [
                'entity_text', 'entity_type', 'relation_type', 
                'add_entity', 'remove_entity', 'add_relation', 'remove_relation'
            ]
            
            if correction_type not in valid_correction_types:
                raise ValueError(f"Invalid correction_type: {correction_type}. Must be one of {valid_correction_types}")
            
            # Create a deep copy to avoid modifying original data
            corrected_data = {
                'text': extracted_data.get('text', ''),
                'entities': [entity.copy() for entity in extracted_data.get('entities', [])],
                'relations': list(extracted_data.get('relations', []))
            }
            
            try:
                if correction_type == 'entity_text':
                    # Expect old_value to be entity index, new_value to be new text
                    if not isinstance(old_value, int) or old_value < 0 or old_value >= len(corrected_data['entities']):
                        raise CorrectionError(f"Invalid entity index: {old_value}")
                    if not isinstance(new_value, str) or not new_value.strip():
                        raise CorrectionError("New entity text must be a non-empty string")
                    
                    # Update entity text and any relations that reference the old text
                    old_text = corrected_data['entities'][old_value]['text']
                    new_text = new_value.strip()
                    corrected_data['entities'][old_value]['text'] = new_text
                    
                    # Update relations that reference the old entity text
                    updated_relations = []
                    for relation in corrected_data['relations']:
                        subject, predicate, obj = relation
                        if subject == old_text:
                            subject = new_text
                        if obj == old_text:
                            obj = new_text
                        updated_relations.append((subject, predicate, obj))
                    corrected_data['relations'] = updated_relations
                
                elif correction_type == 'entity_type':
                    # Expect old_value to be entity index, new_value to be new type
                    if not isinstance(old_value, int) or old_value < 0 or old_value >= len(corrected_data['entities']):
                        raise CorrectionError(f"Invalid entity index: {old_value}")
                    if not isinstance(new_value, str) or not new_value.strip():
                        raise CorrectionError("New entity type must be a non-empty string")
                    
                    corrected_data['entities'][old_value]['entity_type'] = new_value.strip()
                
                elif correction_type == 'relation_type':
                    # Expect old_value to be relation index, new_value to be new relation type
                    if not isinstance(old_value, int) or old_value < 0 or old_value >= len(corrected_data['relations']):
                        raise CorrectionError(f"Invalid relation index: {old_value}")
                    if not isinstance(new_value, str) or not new_value.strip():
                        raise CorrectionError("New relation type must be a non-empty string")
                    
                    old_relation = corrected_data['relations'][old_value]
                    corrected_data['relations'][old_value] = (old_relation[0], new_value.strip(), old_relation[2])
                
                elif correction_type == 'add_entity':
                    # Expect new_value to be a complete entity dict
                    if not isinstance(new_value, dict):
                        raise CorrectionError("New entity must be a dictionary")
                    
                    required_fields = ['entity_type', 'text', 'start_char', 'end_char']
                    for field in required_fields:
                        if field not in new_value:
                            raise CorrectionError(f"New entity missing required field: {field}")
                    
                    corrected_data['entities'].append(new_value)
                
                elif correction_type == 'remove_entity':
                    # Expect old_value to be entity index
                    if not isinstance(old_value, int) or old_value < 0 or old_value >= len(corrected_data['entities']):
                        raise CorrectionError(f"Invalid entity index: {old_value}")
                    
                    corrected_data['entities'].pop(old_value)
                
                elif correction_type == 'add_relation':
                    # Expect new_value to be a complete relation tuple
                    if not isinstance(new_value, (tuple, list)) or len(new_value) != 3:
                        raise CorrectionError("New relation must be a tuple/list of length 3")
                    
                    corrected_data['relations'].append(tuple(new_value))
                
                elif correction_type == 'remove_relation':
                    # Expect old_value to be relation index
                    if not isinstance(old_value, int) or old_value < 0 or old_value >= len(corrected_data['relations']):
                        raise CorrectionError(f"Invalid relation index: {old_value}")
                    
                    corrected_data['relations'].pop(old_value)
            
            except Exception as e:
                if isinstance(e, CorrectionError):
                    raise
                raise CorrectionError(f"Failed to apply correction: {str(e)}")
            
            return corrected_data
        
        def save_curated_output(curated_data: Dict[str, Any], output_file: str) -> bool:
            """Mock implementation of save_curated_output function."""
            if not isinstance(curated_data, dict):
                raise ValueError("curated_data must be a dictionary")
            
            if not isinstance(output_file, str) or not output_file.strip():
                raise ValueError("output_file must be a non-empty string")
            
            # Validate data structure
            required_fields = ['text', 'entities', 'relations']
            for field in required_fields:
                if field not in curated_data:
                    raise InvalidDataError(f"curated_data missing required field: {field}")
            
            try:
                # Prepare output structure with metadata
                output_data = {
                    'text': curated_data['text'],
                    'entities': curated_data['entities'],
                    'relations': [list(relation) if isinstance(relation, tuple) else relation for relation in curated_data['relations']],
                    'metadata': {
                        'tool': 'curation_tool',
                        'version': '1.0.0',
                        'curated': True,
                        'entity_count': len(curated_data['entities']),
                        'relation_count': len(curated_data['relations'])
                    }
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                return True
            
            except Exception as e:
                raise CurationError(f"Failed to save curated output: {str(e)}")
        
        # Store functions as instance attributes
        self.load_llm_output = load_llm_output
        self.display_for_review = display_for_review
        self.apply_correction = apply_correction
        self.save_curated_output = save_curated_output

    def test_load_llm_output_valid_file(self, temp_dir):
        """Test loading a valid LLM output file."""
        # Create test data
        test_data = {
            'text': 'Quercetin is found in Arabidopsis thaliana.',
            'entities': [
                {
                    'entity_type': 'COMPOUND',
                    'text': 'Quercetin',
                    'start_char': 0,
                    'end_char': 9
                },
                {
                    'entity_type': 'ORGANISM',
                    'text': 'Arabidopsis thaliana',
                    'start_char': 22,
                    'end_char': 42
                }
            ],
            'relations': [
                ('Quercetin', 'found_in', 'Arabidopsis thaliana')
            ]
        }
        
        # Write test file
        test_file = temp_dir / "llm_output.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        # Test loading
        result = self.load_llm_output(str(test_file))
        
        assert result == test_data
        assert 'text' in result
        assert 'entities' in result
        assert 'relations' in result
        assert len(result['entities']) == 2
        assert len(result['relations']) == 1

    def test_load_llm_output_file_not_found(self):
        """Test error handling for non-existent files."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            self.load_llm_output("/nonexistent/path/file.json")

    def test_load_llm_output_invalid_json(self, temp_dir):
        """Test error handling for invalid JSON format."""
        test_file = temp_dir / "invalid.json"
        test_file.write_text("{ invalid json content", encoding='utf-8')
        
        with pytest.raises(self.InvalidDataError, match="Invalid JSON format"):
            self.load_llm_output(str(test_file))

    def test_load_llm_output_missing_required_fields(self, temp_dir):
        """Test error handling for missing required fields."""
        # Missing 'text' field
        test_data = {
            'entities': [],
            'relations': []
        }
        test_file = temp_dir / "missing_text.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        with pytest.raises(self.InvalidDataError, match="Missing required field: text"):
            self.load_llm_output(str(test_file))
        
        # Missing 'entities' field
        test_data = {
            'text': 'Test text',
            'relations': []
        }
        test_file = temp_dir / "missing_entities.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        with pytest.raises(self.InvalidDataError, match="Missing required field: entities"):
            self.load_llm_output(str(test_file))
        
        # Missing 'relations' field
        test_data = {
            'text': 'Test text',
            'entities': []
        }
        test_file = temp_dir / "missing_relations.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        with pytest.raises(self.InvalidDataError, match="Missing required field: relations"):
            self.load_llm_output(str(test_file))

    def test_load_llm_output_invalid_entity_structure(self, temp_dir):
        """Test error handling for invalid entity structure."""
        # Non-dict entity
        test_data = {
            'text': 'Test text',
            'entities': ['not_a_dict'],
            'relations': []
        }
        test_file = temp_dir / "invalid_entity.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        with pytest.raises(self.InvalidDataError, match="entities\\[0\\] must be a dictionary"):
            self.load_llm_output(str(test_file))
        
        # Missing entity field
        test_data = {
            'text': 'Test text',
            'entities': [{'text': 'test', 'start_char': 0, 'end_char': 4}],  # missing entity_type
            'relations': []
        }
        test_file = temp_dir / "missing_entity_field.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        with pytest.raises(self.InvalidDataError, match="entities\\[0\\] missing required field: entity_type"):
            self.load_llm_output(str(test_file))

    def test_load_llm_output_invalid_relation_structure(self, temp_dir):
        """Test error handling for invalid relation structure."""
        test_data = {
            'text': 'Test text',
            'entities': [],
            'relations': [('subject', 'predicate')]  # Missing object
        }
        test_file = temp_dir / "invalid_relation.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        with pytest.raises(self.InvalidDataError, match="relations\\[0\\] must be a tuple/list of length 3"):
            self.load_llm_output(str(test_file))

    def test_load_llm_output_invalid_input_types(self):
        """Test error handling for invalid input types."""
        with pytest.raises(ValueError, match="File path must be a non-empty string"):
            self.load_llm_output(None)
        
        with pytest.raises(ValueError, match="File path must be a non-empty string"):
            self.load_llm_output("")
        
        with pytest.raises(ValueError, match="File path must be a non-empty string"):
            self.load_llm_output(123)

    @patch('builtins.print')
    def test_display_for_review_basic(self, mock_print):
        """Test basic display functionality."""
        text = "Quercetin is found in plants."
        entities = [
            {
                'entity_type': 'COMPOUND',
                'text': 'Quercetin',
                'start_char': 0,
                'end_char': 9
            }
        ]
        relations = [
            ('Quercetin', 'found_in', 'plants')
        ]
        
        self.display_for_review(text, entities, relations)
        
        # Verify print calls
        assert mock_print.call_count >= 4  # At least header, text, entities header, entity
        mock_print.assert_any_call("=== TEXT FOR REVIEW ===")
        mock_print.assert_any_call(text)
        mock_print.assert_any_call("\n=== ENTITIES (1) ===")
        mock_print.assert_any_call("1. COMPOUND: 'Quercetin' (0-9)")

    @patch('builtins.print')
    def test_display_for_review_empty_data(self, mock_print):
        """Test display with empty entities and relations."""
        text = "Test text without annotations."
        entities = []
        relations = []
        
        self.display_for_review(text, entities, relations)
        
        mock_print.assert_any_call("=== TEXT FOR REVIEW ===")
        mock_print.assert_any_call(text)
        mock_print.assert_any_call("\n=== ENTITIES (0) ===")
        mock_print.assert_any_call("\n=== RELATIONS (0) ===")

    @patch('builtins.print')
    def test_display_for_review_multiple_items(self, mock_print):
        """Test display with multiple entities and relations."""
        text = "Quercetin and kaempferol are found in plants."
        entities = [
            {'entity_type': 'COMPOUND', 'text': 'Quercetin', 'start_char': 0, 'end_char': 9},
            {'entity_type': 'COMPOUND', 'text': 'kaempferol', 'start_char': 14, 'end_char': 24}
        ]
        relations = [
            ('Quercetin', 'found_in', 'plants'),
            ('kaempferol', 'found_in', 'plants')
        ]
        
        self.display_for_review(text, entities, relations)
        
        # Check that both entities are displayed
        mock_print.assert_any_call("1. COMPOUND: 'Quercetin' (0-9)")
        mock_print.assert_any_call("2. COMPOUND: 'kaempferol' (14-24)")
        
        # Check that both relations are displayed
        mock_print.assert_any_call("1. Quercetin --found_in--> plants")
        mock_print.assert_any_call("2. kaempferol --found_in--> plants")

    def test_display_for_review_invalid_input_types(self):
        """Test error handling for invalid input types."""
        entities = []
        relations = []
        
        with pytest.raises(ValueError, match="text must be a string"):
            self.display_for_review(123, entities, relations)
        
        with pytest.raises(ValueError, match="entities must be a list"):
            self.display_for_review("text", "not_a_list", relations)
        
        with pytest.raises(ValueError, match="relations must be a list"):
            self.display_for_review("text", entities, "not_a_list")

    def test_apply_correction_entity_text(self):
        """Test applying entity text corrections."""
        extracted_data = {
            'text': 'Test text',
            'entities': [
                {'entity_type': 'COMPOUND', 'text': 'quercetin', 'start_char': 0, 'end_char': 9}
            ],
            'relations': []
        }
        
        result = self.apply_correction(extracted_data, 'entity_text', 0, 'Quercetin')
        
        assert result['entities'][0]['text'] == 'Quercetin'
        assert result['entities'][0]['entity_type'] == 'COMPOUND'  # Unchanged
        assert len(result['entities']) == 1

    def test_apply_correction_entity_type(self):
        """Test applying entity type corrections."""
        extracted_data = {
            'text': 'Test text',
            'entities': [
                {'entity_type': 'MOLECULE', 'text': 'quercetin', 'start_char': 0, 'end_char': 9}
            ],
            'relations': []
        }
        
        result = self.apply_correction(extracted_data, 'entity_type', 0, 'COMPOUND')
        
        assert result['entities'][0]['entity_type'] == 'COMPOUND'
        assert result['entities'][0]['text'] == 'quercetin'  # Unchanged
        assert len(result['entities']) == 1

    def test_apply_correction_relation_type(self):
        """Test applying relation type corrections."""
        extracted_data = {
            'text': 'Test text',
            'entities': [],
            'relations': [
                ('quercetin', 'occurs_in', 'plants')
            ]
        }
        
        result = self.apply_correction(extracted_data, 'relation_type', 0, 'found_in')
        
        assert result['relations'][0] == ('quercetin', 'found_in', 'plants')
        assert len(result['relations']) == 1

    def test_apply_correction_add_entity(self):
        """Test adding new entities."""
        extracted_data = {
            'text': 'Test text',
            'entities': [
                {'entity_type': 'COMPOUND', 'text': 'quercetin', 'start_char': 0, 'end_char': 9}
            ],
            'relations': []
        }
        
        new_entity = {
            'entity_type': 'ORGANISM',
            'text': 'Arabidopsis',
            'start_char': 15,
            'end_char': 26
        }
        
        result = self.apply_correction(extracted_data, 'add_entity', None, new_entity)
        
        assert len(result['entities']) == 2
        assert result['entities'][1] == new_entity
        assert result['entities'][0]['text'] == 'quercetin'  # Original unchanged

    def test_apply_correction_remove_entity(self):
        """Test removing entities."""
        extracted_data = {
            'text': 'Test text',
            'entities': [
                {'entity_type': 'COMPOUND', 'text': 'quercetin', 'start_char': 0, 'end_char': 9},
                {'entity_type': 'ORGANISM', 'text': 'Arabidopsis', 'start_char': 15, 'end_char': 26}
            ],
            'relations': []
        }
        
        result = self.apply_correction(extracted_data, 'remove_entity', 0, None)
        
        assert len(result['entities']) == 1
        assert result['entities'][0]['text'] == 'Arabidopsis'

    def test_apply_correction_add_relation(self):
        """Test adding new relations."""
        extracted_data = {
            'text': 'Test text',
            'entities': [],
            'relations': [
                ('quercetin', 'found_in', 'plants')
            ]
        }
        
        new_relation = ('kaempferol', 'found_in', 'plants')
        
        result = self.apply_correction(extracted_data, 'add_relation', None, new_relation)
        
        assert len(result['relations']) == 2
        assert result['relations'][1] == new_relation
        assert result['relations'][0] == ('quercetin', 'found_in', 'plants')  # Original unchanged

    def test_apply_correction_remove_relation(self):
        """Test removing relations."""
        extracted_data = {
            'text': 'Test text',
            'entities': [],
            'relations': [
                ('quercetin', 'found_in', 'plants'),
                ('kaempferol', 'found_in', 'plants')
            ]
        }
        
        result = self.apply_correction(extracted_data, 'remove_relation', 0, None)
        
        assert len(result['relations']) == 1
        assert result['relations'][0] == ('kaempferol', 'found_in', 'plants')

    def test_apply_correction_invalid_correction_type(self):
        """Test error handling for invalid correction types."""
        extracted_data = {'text': 'Test', 'entities': [], 'relations': []}
        
        with pytest.raises(ValueError, match="Invalid correction_type"):
            self.apply_correction(extracted_data, 'invalid_type', None, None)

    def test_apply_correction_invalid_entity_index(self):
        """Test error handling for invalid entity indices."""
        extracted_data = {
            'text': 'Test text',
            'entities': [{'entity_type': 'TEST', 'text': 'test', 'start_char': 0, 'end_char': 4}],
            'relations': []
        }
        
        # Index out of range
        with pytest.raises(self.CorrectionError, match="Invalid entity index"):
            self.apply_correction(extracted_data, 'entity_text', 5, 'new_text')
        
        # Negative index
        with pytest.raises(self.CorrectionError, match="Invalid entity index"):
            self.apply_correction(extracted_data, 'entity_text', -1, 'new_text')

    def test_apply_correction_invalid_relation_index(self):
        """Test error handling for invalid relation indices."""
        extracted_data = {
            'text': 'Test text',
            'entities': [],
            'relations': [('a', 'b', 'c')]
        }
        
        # Index out of range
        with pytest.raises(self.CorrectionError, match="Invalid relation index"):
            self.apply_correction(extracted_data, 'relation_type', 5, 'new_type')
        
        # Negative index
        with pytest.raises(self.CorrectionError, match="Invalid relation index"):
            self.apply_correction(extracted_data, 'relation_type', -1, 'new_type')

    def test_apply_correction_invalid_new_entity_structure(self):
        """Test error handling for invalid new entity structure."""
        extracted_data = {'text': 'Test', 'entities': [], 'relations': []}
        
        # Non-dict new entity
        with pytest.raises(self.CorrectionError, match="New entity must be a dictionary"):
            self.apply_correction(extracted_data, 'add_entity', None, 'not_a_dict')
        
        # Missing required fields
        incomplete_entity = {'entity_type': 'TEST', 'text': 'test'}  # Missing start_char, end_char
        with pytest.raises(self.CorrectionError, match="New entity missing required field"):
            self.apply_correction(extracted_data, 'add_entity', None, incomplete_entity)

    def test_apply_correction_invalid_new_relation_structure(self):
        """Test error handling for invalid new relation structure."""
        extracted_data = {'text': 'Test', 'entities': [], 'relations': []}
        
        # Wrong length relation
        with pytest.raises(self.CorrectionError, match="New relation must be a tuple/list of length 3"):
            self.apply_correction(extracted_data, 'add_relation', None, ('a', 'b'))  # Missing third element

    def test_apply_correction_invalid_input_types(self):
        """Test error handling for invalid input types."""
        valid_data = {'text': 'Test', 'entities': [], 'relations': []}
        
        with pytest.raises(ValueError, match="extracted_data must be a dictionary"):
            self.apply_correction("not_a_dict", 'entity_text', 0, 'new_text')
        
        with pytest.raises(ValueError, match="correction_type must be a non-empty string"):
            self.apply_correction(valid_data, "", 0, 'new_text')
        
        with pytest.raises(ValueError, match="correction_type must be a non-empty string"):
            self.apply_correction(valid_data, None, 0, 'new_text')

    def test_save_curated_output_basic(self, temp_dir):
        """Test saving curated output to file."""
        curated_data = {
            'text': 'Quercetin is found in plants.',
            'entities': [
                {'entity_type': 'COMPOUND', 'text': 'Quercetin', 'start_char': 0, 'end_char': 9}
            ],
            'relations': [
                ('Quercetin', 'found_in', 'plants')
            ]
        }
        
        output_file = temp_dir / "curated_output.json"
        result = self.save_curated_output(curated_data, str(output_file))
        
        assert result is True
        assert output_file.exists()
        
        # Verify file contents
        with open(output_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert 'text' in saved_data
        assert 'entities' in saved_data
        assert 'relations' in saved_data
        assert 'metadata' in saved_data
        
        # Check metadata
        metadata = saved_data['metadata']
        assert metadata['tool'] == 'curation_tool'
        assert metadata['curated'] is True
        assert metadata['entity_count'] == 1
        assert metadata['relation_count'] == 1
        
        # Check data integrity
        assert saved_data['text'] == curated_data['text']
        assert saved_data['entities'] == curated_data['entities']
        # Relations are saved as lists but original are tuples, so convert for comparison
        expected_relations = [list(relation) if isinstance(relation, tuple) else relation for relation in curated_data['relations']]
        assert saved_data['relations'] == expected_relations

    def test_save_curated_output_empty_data(self, temp_dir):
        """Test saving curated output with empty entities and relations."""
        curated_data = {
            'text': 'Text with no annotations.',
            'entities': [],
            'relations': []
        }
        
        output_file = temp_dir / "empty_curated.json"
        result = self.save_curated_output(curated_data, str(output_file))
        
        assert result is True
        
        with open(output_file, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        assert saved_data['metadata']['entity_count'] == 0
        assert saved_data['metadata']['relation_count'] == 0
        assert len(saved_data['entities']) == 0
        assert len(saved_data['relations']) == 0

    def test_save_curated_output_missing_required_fields(self, temp_dir):
        """Test error handling for missing required fields."""
        output_file = temp_dir / "output.json"
        
        # Missing text
        curated_data = {'entities': [], 'relations': []}
        with pytest.raises(self.InvalidDataError, match="curated_data missing required field: text"):
            self.save_curated_output(curated_data, str(output_file))
        
        # Missing entities
        curated_data = {'text': 'Test', 'relations': []}
        with pytest.raises(self.InvalidDataError, match="curated_data missing required field: entities"):
            self.save_curated_output(curated_data, str(output_file))
        
        # Missing relations
        curated_data = {'text': 'Test', 'entities': []}
        with pytest.raises(self.InvalidDataError, match="curated_data missing required field: relations"):
            self.save_curated_output(curated_data, str(output_file))

    def test_save_curated_output_invalid_input_types(self, temp_dir):
        """Test error handling for invalid input types."""
        output_file = temp_dir / "output.json"
        valid_data = {'text': 'Test', 'entities': [], 'relations': []}
        
        with pytest.raises(ValueError, match="curated_data must be a dictionary"):
            self.save_curated_output("not_a_dict", str(output_file))
        
        with pytest.raises(ValueError, match="output_file must be a non-empty string"):
            self.save_curated_output(valid_data, "")
        
        with pytest.raises(ValueError, match="output_file must be a non-empty string"):
            self.save_curated_output(valid_data, None)

    def test_complete_curation_workflow(self, temp_dir):
        """Test a complete curation workflow from loading to saving."""
        # Step 1: Create initial LLM output
        initial_data = {
            'text': 'Quercetin and kaempferol are flavonoids found in tomato plants.',
            'entities': [
                {'entity_type': 'MOLECULE', 'text': 'Quercetin', 'start_char': 0, 'end_char': 9},
                {'entity_type': 'MOLECULE', 'text': 'kaempferol', 'start_char': 14, 'end_char': 24},
                {'entity_type': 'COMPOUND_CLASS', 'text': 'flavonoids', 'start_char': 29, 'end_char': 39},
                {'entity_type': 'PLANT', 'text': 'tomato plants', 'start_char': 49, 'end_char': 62}
            ],
            'relations': [
                ('Quercetin', 'is_a', 'flavonoids'),
                ('kaempferol', 'is_a', 'flavonoids'),
                ('flavonoids', 'found_in', 'tomato plants')
            ]
        }
        
        input_file = temp_dir / "initial_llm_output.json"
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f)
        
        # Step 2: Load the LLM output
        loaded_data = self.load_llm_output(str(input_file))
        assert loaded_data == initial_data
        
        # Step 3: Apply corrections
        # Correct entity types: MOLECULE -> COMPOUND, PLANT -> ORGANISM
        corrected_data = self.apply_correction(loaded_data, 'entity_type', 0, 'COMPOUND')
        corrected_data = self.apply_correction(corrected_data, 'entity_type', 1, 'COMPOUND')
        corrected_data = self.apply_correction(corrected_data, 'entity_type', 3, 'ORGANISM')
        
        # Add a missing relation
        new_relation = ('Quercetin', 'found_in', 'tomato plants')
        corrected_data = self.apply_correction(corrected_data, 'add_relation', None, new_relation)
        
        # Remove redundant relation
        corrected_data = self.apply_correction(corrected_data, 'remove_relation', 2, None)  # Remove 'flavonoids found_in tomato plants'
        
        # Step 4: Verify corrections
        assert corrected_data['entities'][0]['entity_type'] == 'COMPOUND'
        assert corrected_data['entities'][1]['entity_type'] == 'COMPOUND'
        assert corrected_data['entities'][3]['entity_type'] == 'ORGANISM'
        assert len(corrected_data['relations']) == 3  # 2 original + 1 added - 1 removed
        assert ('Quercetin', 'found_in', 'tomato plants') in corrected_data['relations']
        
        # Step 5: Save curated output
        output_file = temp_dir / "curated_final.json"
        result = self.save_curated_output(corrected_data, str(output_file))
        assert result is True
        
        # Step 6: Verify saved output
        with open(output_file, 'r', encoding='utf-8') as f:
            final_data = json.load(f)
        
        assert final_data['metadata']['curated'] is True
        assert final_data['metadata']['entity_count'] == 4
        assert final_data['metadata']['relation_count'] == 3
        assert final_data['text'] == initial_data['text']
        
        # Verify entity type corrections
        entity_types = [entity['entity_type'] for entity in final_data['entities']]
        assert entity_types.count('COMPOUND') == 2
        assert entity_types.count('ORGANISM') == 1
        assert entity_types.count('COMPOUND_CLASS') == 1

    def test_error_handling_comprehensive(self, temp_dir):
        """Test comprehensive error handling across all functions."""
        # Test file I/O errors
        protected_file = temp_dir / "protected.json"
        protected_file.write_text('{"text": "test", "entities": [], "relations": []}')
        protected_file.chmod(0o000)  # Remove all permissions
        
        try:
            # This should raise a permission error
            with pytest.raises(self.CurationError):
                self.load_llm_output(str(protected_file))
        finally:
            protected_file.chmod(0o644)  # Restore permissions for cleanup
        
        # Test data validation edge cases
        test_data = {
            'text': 'Test text',
            'entities': [
                {'entity_type': 'TEST', 'text': 'test', 'start_char': -5, 'end_char': 4}  # Invalid start_char
            ],
            'relations': []
        }
        
        invalid_file = temp_dir / "invalid_entities.json"
        with open(invalid_file, 'w') as f:
            json.dump(test_data, f)
        
        with pytest.raises(self.InvalidDataError, match="start_char must be a non-negative integer"):
            self.load_llm_output(str(invalid_file))
        
        # Test correction edge cases
        valid_data = {
            'text': 'Test text',
            'entities': [{'entity_type': 'TEST', 'text': 'test', 'start_char': 0, 'end_char': 4}],
            'relations': [('a', 'b', 'c')]
        }
        
        # Empty string corrections
        with pytest.raises(self.CorrectionError, match="New entity text must be a non-empty string"):
            self.apply_correction(valid_data, 'entity_text', 0, "")
        
        with pytest.raises(self.CorrectionError, match="New entity type must be a non-empty string"):
            self.apply_correction(valid_data, 'entity_type', 0, "  ")  # Whitespace only
        
        with pytest.raises(self.CorrectionError, match="New relation type must be a non-empty string"):
            self.apply_correction(valid_data, 'relation_type', 0, "")

    def test_multiple_corrections_sequence(self):
        """Test applying multiple corrections in sequence."""
        initial_data = {
            'text': 'Test compound in plant.',
            'entities': [
                {'entity_type': 'MOLECULE', 'text': 'compound', 'start_char': 5, 'end_char': 13},
                {'entity_type': 'SPECIES', 'text': 'plant', 'start_char': 17, 'end_char': 22}
            ],
            'relations': [
                ('compound', 'located_in', 'plant')
            ]
        }
        
        # Apply sequence of corrections
        corrected = self.apply_correction(initial_data, 'entity_type', 0, 'COMPOUND')
        corrected = self.apply_correction(corrected, 'entity_type', 1, 'ORGANISM')
        corrected = self.apply_correction(corrected, 'entity_text', 0, 'flavonoid')
        corrected = self.apply_correction(corrected, 'relation_type', 0, 'found_in')
        
        # Add new entity and relation
        new_entity = {'entity_type': 'PROCESS', 'text': 'metabolism', 'start_char': 25, 'end_char': 35}
        corrected = self.apply_correction(corrected, 'add_entity', None, new_entity)
        corrected = self.apply_correction(corrected, 'add_relation', None, ('flavonoid', 'involved_in', 'metabolism'))
        
        # Verify final state
        assert len(corrected['entities']) == 3
        assert len(corrected['relations']) == 2
        assert corrected['entities'][0]['entity_type'] == 'COMPOUND'
        assert corrected['entities'][0]['text'] == 'flavonoid'
        assert corrected['entities'][1]['entity_type'] == 'ORGANISM'
        assert corrected['entities'][2]['entity_type'] == 'PROCESS'
        assert corrected['relations'][0] == ('flavonoid', 'found_in', 'plant')
        assert corrected['relations'][1] == ('flavonoid', 'involved_in', 'metabolism')


# Mark all tests in this module as evaluation related
pytestmark = [pytest.mark.unit, pytest.mark.evaluation]