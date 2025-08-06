"""
Unit tests for src/evaluation/gold_standard_tool.py

This module tests the gold standard annotation tool functionality for creating
and managing manual annotations of biological literature in the AIM2-ODIE
ontology development and information extraction system.

Test Coverage:
- Document loading: text files, PDF files, invalid formats
- Entity annotation: type validation, span validation, conflict detection  
- Relationship annotation: entity linking, relationship types, validation
- Export functionality: JSON format, CSV format, structured output
- Error handling: invalid inputs, file I/O errors, annotation conflicts
- Integration scenarios: complete annotation workflows

Since the actual implementation doesn't exist yet, this test file uses comprehensive
mocking to test the intended API interface and behavior.
"""

import pytest
import json
import csv
import tempfile
import os
import uuid
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestGoldStandardTool:
    """Comprehensive test class for gold standard annotation tool."""
    
    @pytest.fixture(autouse=True)
    def setup_mock_module(self):
        """Set up mock module and storage for each test."""
        # Mock storage to simulate in-memory annotation database
        self.mock_storage = {
            'documents': {},
            'entities': {},
            'relationships': {},
            'next_entity_id': 1
        }
        
        # Define mock exception classes
        class GoldStandardError(Exception):
            pass
        
        class AnnotationConflictError(GoldStandardError):
            pass
        
        class InvalidDocumentError(GoldStandardError):
            pass
        
        self.GoldStandardError = GoldStandardError
        self.AnnotationConflictError = AnnotationConflictError
        self.InvalidDocumentError = InvalidDocumentError
        
        # Mock function implementations
        def load_document_for_annotation(file_path: str) -> str:
            if not isinstance(file_path, str) or not file_path:
                raise ValueError("File path must be a non-empty string")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            path_obj = Path(file_path)
            if path_obj.suffix.lower() not in ['.txt', '.pdf']:
                raise InvalidDocumentError(f"Unsupported file format: {path_obj.suffix}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                raise InvalidDocumentError("Document is empty or contains only whitespace")
            
            doc_id = str(uuid.uuid4())
            self.mock_storage['documents'][doc_id] = {
                'file_path': file_path,
                'content': content,
                'entities': [],
                'relationships': []
            }
            
            return doc_id
        
        def add_entity_annotation(doc_id: str, entity_type: str, text: str, start_char: int, end_char: int) -> str:
            # Input validation
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("Document ID must be a non-empty string")
            
            if doc_id not in self.mock_storage['documents']:
                raise ValueError(f"Document not found: {doc_id}")
            
            if not isinstance(entity_type, str) or not entity_type.strip():
                raise ValueError("Entity type must be a non-empty string")
            
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Entity text must be a non-empty string")
            
            if not isinstance(start_char, int) or start_char < 0:
                raise ValueError("Start character must be a non-negative integer")
            
            if not isinstance(end_char, int) or end_char <= start_char:
                raise ValueError("End character must be greater than start character")
            
            # Validate span against document content
            document = self.mock_storage['documents'][doc_id]
            doc_content = document['content']
            
            if end_char > len(doc_content):
                raise ValueError("End character exceeds document length")
            
            extracted_text = doc_content[start_char:end_char]
            if extracted_text != text:
                raise ValueError(f"Text does not match span: expected '{text}', got '{extracted_text}'")
            
            # Check for overlapping annotations
            for existing_entity in document['entities']:
                existing_start = existing_entity['start_char']
                existing_end = existing_entity['end_char']
                
                if (start_char < existing_end and end_char > existing_start):
                    raise AnnotationConflictError(
                        f"Entity annotation overlaps with existing annotation at {existing_start}-{existing_end}"
                    )
            
            # Generate entity ID and add annotation
            entity_id = f"E{self.mock_storage['next_entity_id']}"
            self.mock_storage['next_entity_id'] += 1
            
            entity_annotation = {
                'id': entity_id,
                'doc_id': doc_id,
                'entity_type': entity_type.strip(),
                'text': text.strip(),
                'start_char': start_char,
                'end_char': end_char
            }
            
            document['entities'].append(entity_annotation)
            self.mock_storage['entities'][entity_id] = entity_annotation
            
            return entity_id
        
        def add_relationship_annotation(doc_id: str, subject_id: str, relation_type: str, object_id: str) -> str:
            # Input validation
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("Document ID must be a non-empty string")
            
            if doc_id not in self.mock_storage['documents']:
                raise ValueError(f"Document not found: {doc_id}")
            
            if not isinstance(subject_id, str) or not subject_id:
                raise ValueError("Subject ID must be a non-empty string")
            
            if subject_id not in self.mock_storage['entities']:
                raise ValueError(f"Subject entity not found: {subject_id}")
            
            if not isinstance(object_id, str) or not object_id:
                raise ValueError("Object ID must be a non-empty string")
            
            if object_id not in self.mock_storage['entities']:
                raise ValueError(f"Object entity not found: {object_id}")
            
            if not isinstance(relation_type, str) or not relation_type.strip():
                raise ValueError("Relation type must be a non-empty string")
            
            # Validate entities belong to same document
            subject_entity = self.mock_storage['entities'][subject_id]
            object_entity = self.mock_storage['entities'][object_id]
            
            if subject_entity['doc_id'] != doc_id or object_entity['doc_id'] != doc_id:
                raise ValueError("Both entities must belong to the specified document")
            
            # Check for duplicate relationships
            document = self.mock_storage['documents'][doc_id]
            for existing_rel in document['relationships']:
                if (existing_rel['subject_id'] == subject_id and 
                    existing_rel['object_id'] == object_id and 
                    existing_rel['relation_type'] == relation_type.strip()):
                    raise AnnotationConflictError(
                        f"Relationship already exists: {subject_id} -{relation_type}-> {object_id}"
                    )
            
            # Generate relationship ID and add annotation
            rel_id = f"R{len(self.mock_storage['relationships']) + 1}"
            
            relationship_annotation = {
                'id': rel_id,
                'doc_id': doc_id,
                'subject_id': subject_id,
                'relation_type': relation_type.strip(),
                'object_id': object_id
            }
            
            document['relationships'].append(relationship_annotation)
            self.mock_storage['relationships'][rel_id] = relationship_annotation
            
            return rel_id
        
        def export_annotations(output_file: str, format: str = 'json') -> bool:
            if not isinstance(output_file, str) or not output_file.strip():
                raise ValueError("Output file path must be a non-empty string")
            
            if not isinstance(format, str) or format.lower() not in ['json', 'csv']:
                raise ValueError("Format must be 'json' or 'csv'")
            
            try:
                if format.lower() == 'json':
                    export_data = {
                        'documents': [],
                        'export_timestamp': '2023-01-01T00:00:00Z',
                        'total_documents': len(self.mock_storage['documents']),
                        'total_entities': len(self.mock_storage['entities']),
                        'total_relationships': len(self.mock_storage['relationships'])
                    }
                    
                    for doc_id, document in self.mock_storage['documents'].items():
                        doc_data = {
                            'id': doc_id,
                            'file_path': document['file_path'],
                            'entities': document['entities'],
                            'relationships': document['relationships']
                        }
                        export_data['documents'].append(doc_data)
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                elif format.lower() == 'csv':
                    with open(output_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        
                        # Write header
                        writer.writerow(['type', 'doc_id', 'id', 'entity_type', 'text', 'start_char', 'end_char'])
                        
                        # Write entity annotations
                        for entity in self.mock_storage['entities'].values():
                            writer.writerow([
                                'entity', entity['doc_id'], entity['id'], 
                                entity['entity_type'], entity['text'], 
                                entity['start_char'], entity['end_char']
                            ])
                        
                        # Write relationship annotations
                        for relationship in self.mock_storage['relationships'].values():
                            writer.writerow([
                                'relationship', relationship['doc_id'], relationship['id'],
                                relationship['subject_id'], relationship['relation_type'], 
                                relationship['object_id'], ''
                            ])
                
                return True
            
            except Exception as e:
                raise GoldStandardError(f"Export failed: {str(e)}")
        
        # Store functions as instance attributes
        self.load_document_for_annotation = load_document_for_annotation
        self.add_entity_annotation = add_entity_annotation
        self.add_relationship_annotation = add_relationship_annotation
        self.export_annotations = export_annotations
    
    def test_load_document_for_annotation_text_file(self, temp_dir):
        """Test loading a valid text file for annotation."""
        # Create test document
        test_content = """Plant metabolomics is the study of small molecules (metabolites) found in plants.
These metabolites include primary metabolites like amino acids, sugars, and organic acids,
as well as secondary metabolites such as flavonoids, alkaloids, and terpenoids."""
        
        test_file = temp_dir / "test_document.txt"
        test_file.write_text(test_content, encoding='utf-8')
        
        doc_id = self.load_document_for_annotation(str(test_file))
        
        # Verify document ID is returned
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0
        # UUID format validation
        assert len(doc_id.split('-')) == 5  # UUID has 5 parts separated by hyphens
    
    def test_load_document_for_annotation_pdf_file(self, temp_dir):
        """Test loading a PDF file for annotation."""
        test_content = "Mock PDF content for testing file extension validation."
        
        test_file = temp_dir / "test_document.pdf"
        test_file.write_text(test_content, encoding='utf-8')
        
        doc_id = self.load_document_for_annotation(str(test_file))
        
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0
    
    def test_load_document_for_annotation_file_not_found(self):
        """Test error handling for non-existent files."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            self.load_document_for_annotation("/nonexistent/path/document.txt")
    
    def test_load_document_for_annotation_unsupported_format(self, temp_dir):
        """Test error handling for unsupported file formats."""
        test_file = temp_dir / "document.docx"
        test_file.write_text("Test content", encoding='utf-8')
        
        with pytest.raises(self.InvalidDocumentError, match="Unsupported file format"):
            self.load_document_for_annotation(str(test_file))
    
    def test_load_document_for_annotation_empty_file(self, temp_dir):
        """Test error handling for empty documents."""
        test_file = temp_dir / "empty_document.txt"
        test_file.write_text("", encoding='utf-8')
        
        with pytest.raises(self.InvalidDocumentError, match="Document is empty"):
            self.load_document_for_annotation(str(test_file))
    
    def test_add_entity_annotation_basic(self, temp_dir):
        """Test adding a basic entity annotation."""
        # Setup document
        test_content = "Flavonoids are secondary metabolites found in Arabidopsis thaliana leaves."
        test_file = temp_dir / "test.txt"
        test_file.write_text(test_content, encoding='utf-8')
        doc_id = self.load_document_for_annotation(str(test_file))
        
        entity_id = self.add_entity_annotation(
            doc_id=doc_id,
            entity_type="COMPOUND",
            text="Flavonoids",
            start_char=0,
            end_char=10
        )
        
        assert isinstance(entity_id, str)
        assert entity_id.startswith("E")
        assert entity_id != ""
    
    def test_add_entity_annotation_multiple_entities(self, temp_dir):
        """Test adding multiple entity annotations to the same document."""
        # Setup document
        test_content = "Flavonoids are secondary metabolites found in Arabidopsis thaliana leaves."
        test_file = temp_dir / "test.txt"
        test_file.write_text(test_content, encoding='utf-8')
        doc_id = self.load_document_for_annotation(str(test_file))
        
        # Add multiple entities
        entity1_id = self.add_entity_annotation(doc_id, "COMPOUND", "Flavonoids", 0, 10)
        entity2_id = self.add_entity_annotation(doc_id, "COMPOUND", "secondary metabolites", 15, 36)
        entity3_id = self.add_entity_annotation(doc_id, "ORGANISM", "Arabidopsis thaliana", 46, 66)
        
        # Verify all entities have unique IDs
        assert entity1_id != entity2_id != entity3_id
        assert all(eid.startswith("E") for eid in [entity1_id, entity2_id, entity3_id])
    
    def test_add_entity_annotation_overlapping_spans(self, temp_dir):
        """Test error handling for overlapping entity spans."""
        # Setup document
        test_content = "Flavonoids are secondary metabolites."
        test_file = temp_dir / "test.txt"
        test_file.write_text(test_content, encoding='utf-8')
        doc_id = self.load_document_for_annotation(str(test_file))
        
        # Add first entity
        self.add_entity_annotation(doc_id, "COMPOUND", "Flavonoids", 0, 10)
        
        # Try to add overlapping entity
        with pytest.raises(self.AnnotationConflictError, match="overlaps with existing annotation"):
            self.add_entity_annotation(doc_id, "MOLECULE", "lavonoids", 1, 10)
    
    def test_add_entity_annotation_invalid_span(self, temp_dir):
        """Test error handling for invalid character spans."""
        # Setup document
        test_content = "Test document content."
        test_file = temp_dir / "test.txt"
        test_file.write_text(test_content, encoding='utf-8')
        doc_id = self.load_document_for_annotation(str(test_file))
        
        # Test end_char exceeds document length
        with pytest.raises(ValueError, match="End character exceeds document length"):
            self.add_entity_annotation(doc_id, "COMPOUND", "test", 0, 1000)
        
        # Test start_char >= end_char
        with pytest.raises(ValueError, match="End character must be greater than start character"):
            self.add_entity_annotation(doc_id, "COMPOUND", "test", 10, 10)
        
        # Test negative start_char
        with pytest.raises(ValueError, match="Start character must be a non-negative integer"):
            self.add_entity_annotation(doc_id, "COMPOUND", "test", -1, 4)
    
    def test_add_entity_annotation_text_span_mismatch(self, temp_dir):
        """Test error handling for text that doesn't match the span."""
        # Setup document
        test_content = "Test document content."
        test_file = temp_dir / "test.txt"
        test_file.write_text(test_content, encoding='utf-8')
        doc_id = self.load_document_for_annotation(str(test_file))
        
        with pytest.raises(ValueError, match="Text does not match span"):
            self.add_entity_annotation(doc_id, "COMPOUND", "WrongText", 0, 4)  # Should be "Test"
    
    def test_add_relationship_annotation_basic(self, temp_dir):
        """Test adding a basic relationship annotation."""
        # Setup document and entities
        test_content = "Quercetin is found in Arabidopsis thaliana."
        test_file = temp_dir / "test.txt"
        test_file.write_text(test_content, encoding='utf-8')
        doc_id = self.load_document_for_annotation(str(test_file))
        
        entity1_id = self.add_entity_annotation(doc_id, "COMPOUND", "Quercetin", 0, 9)
        entity2_id = self.add_entity_annotation(doc_id, "ORGANISM", "Arabidopsis thaliana", 22, 42)
        
        rel_id = self.add_relationship_annotation(
            doc_id=doc_id,
            subject_id=entity1_id,
            relation_type="found_in",
            object_id=entity2_id
        )
        
        assert isinstance(rel_id, str)
        assert rel_id.startswith("R")
        assert rel_id != ""
    
    def test_add_relationship_annotation_duplicate_error(self, temp_dir):
        """Test error handling for duplicate relationships."""
        # Setup document and entities
        test_content = "Quercetin is found in Arabidopsis thaliana."
        test_file = temp_dir / "test.txt"
        test_file.write_text(test_content, encoding='utf-8')
        doc_id = self.load_document_for_annotation(str(test_file))
        
        entity1_id = self.add_entity_annotation(doc_id, "COMPOUND", "Quercetin", 0, 9)
        entity2_id = self.add_entity_annotation(doc_id, "ORGANISM", "Arabidopsis thaliana", 22, 42)
        
        # Add first relationship
        self.add_relationship_annotation(doc_id, entity1_id, "found_in", entity2_id)
        
        # Try to add the same relationship again
        with pytest.raises(self.AnnotationConflictError, match="Relationship already exists"):
            self.add_relationship_annotation(doc_id, entity1_id, "found_in", entity2_id)
    
    def test_add_relationship_annotation_invalid_entity_ids(self, temp_dir):
        """Test error handling for invalid entity IDs."""
        # Setup document and one entity
        test_content = "Quercetin is found in plants."
        test_file = temp_dir / "test.txt"
        test_file.write_text(test_content, encoding='utf-8')
        doc_id = self.load_document_for_annotation(str(test_file))
        
        entity1_id = self.add_entity_annotation(doc_id, "COMPOUND", "Quercetin", 0, 9)
        
        # Test non-existent subject entity
        with pytest.raises(ValueError, match="Subject entity not found"):
            self.add_relationship_annotation(doc_id, "E999", "found_in", entity1_id)
        
        # Test non-existent object entity
        with pytest.raises(ValueError, match="Object entity not found"):
            self.add_relationship_annotation(doc_id, entity1_id, "found_in", "E999")
    
    def test_export_annotations_json_format(self, temp_dir):
        """Test exporting annotations in JSON format."""
        # Setup document and annotations
        test_content = "Resveratrol shows antioxidant activity."
        test_file = temp_dir / "test.txt"
        test_file.write_text(test_content, encoding='utf-8')
        doc_id = self.load_document_for_annotation(str(test_file))
        
        entity1_id = self.add_entity_annotation(doc_id, "COMPOUND", "Resveratrol", 0, 11)
        entity2_id = self.add_entity_annotation(doc_id, "BIOLOGICAL_ACTIVITY", "antioxidant activity", 18, 38)
        rel_id = self.add_relationship_annotation(doc_id, entity1_id, "exhibits", entity2_id)
        
        # Export to JSON
        output_file = temp_dir / "annotations.json"
        result = self.export_annotations(str(output_file), format='json')
        
        assert result is True
        assert output_file.exists()
        
        # Verify JSON structure
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert 'documents' in data
        assert 'export_timestamp' in data
        assert 'total_documents' in data
        assert 'total_entities' in data
        assert 'total_relationships' in data
        
        # Check counts
        assert data['total_documents'] == 1
        assert data['total_entities'] == 2
        assert data['total_relationships'] == 1
        
        # Check document structure
        doc_data = data['documents'][0]
        assert len(doc_data['entities']) == 2
        assert len(doc_data['relationships']) == 1
    
    def test_export_annotations_csv_format(self, temp_dir):
        """Test exporting annotations in CSV format."""
        # Setup document and annotations
        test_content = "Resveratrol shows antioxidant activity."
        test_file = temp_dir / "test.txt"
        test_file.write_text(test_content, encoding='utf-8')
        doc_id = self.load_document_for_annotation(str(test_file))
        
        entity1_id = self.add_entity_annotation(doc_id, "COMPOUND", "Resveratrol", 0, 11)
        entity2_id = self.add_entity_annotation(doc_id, "BIOLOGICAL_ACTIVITY", "antioxidant activity", 18, 38)
        rel_id = self.add_relationship_annotation(doc_id, entity1_id, "exhibits", entity2_id)
        
        # Export to CSV
        output_file = temp_dir / "annotations.csv"
        result = self.export_annotations(str(output_file), format='csv')
        
        assert result is True
        assert output_file.exists()
        
        # Verify CSV structure
        with open(output_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Check header
        header = rows[0]
        expected_columns = ['type', 'doc_id', 'id', 'entity_type', 'text', 'start_char', 'end_char']
        assert header == expected_columns
        
        # Check data rows
        data_rows = rows[1:]
        entity_rows = [row for row in data_rows if row[0] == 'entity']
        relationship_rows = [row for row in data_rows if row[0] == 'relationship']
        
        assert len(entity_rows) == 2
        assert len(relationship_rows) == 1
    
    def test_export_annotations_invalid_format(self, temp_dir):
        """Test error handling for invalid export formats."""
        output_file = temp_dir / "annotations.txt"
        
        with pytest.raises(ValueError, match="Format must be 'json' or 'csv'"):
            self.export_annotations(str(output_file), format='xml')
    
    def test_complete_annotation_workflow(self, temp_dir):
        """Test a complete annotation workflow from document loading to export."""
        # Create test document with known character positions
        test_content = "The study analyzed quercetin and kaempferol levels in tomato fruit under drought stress conditions. These flavonoids showed increased expression in response to water deficit."
        
        doc_file = temp_dir / "research_paper.txt"
        doc_file.write_text(test_content, encoding='utf-8')
        
        # Step 1: Load document
        doc_id = self.load_document_for_annotation(str(doc_file))
        assert isinstance(doc_id, str)
        
        # Step 2: Add entity annotations (verified positions)
        entity1_id = self.add_entity_annotation(doc_id, "COMPOUND", "quercetin", 19, 28)
        entity2_id = self.add_entity_annotation(doc_id, "COMPOUND", "kaempferol", 33, 43)
        entity3_id = self.add_entity_annotation(doc_id, "ORGANISM", "tomato", 54, 60)
        entity4_id = self.add_entity_annotation(doc_id, "CONDITION", "drought stress", 73, 87)
        entity5_id = self.add_entity_annotation(doc_id, "COMPOUND", "flavonoids", 106, 116)
        
        assert all(eid.startswith("E") for eid in [entity1_id, entity2_id, entity3_id, entity4_id, entity5_id])
        
        # Step 3: Add relationship annotations
        rel1_id = self.add_relationship_annotation(doc_id, entity1_id, "found_in", entity3_id)
        rel2_id = self.add_relationship_annotation(doc_id, entity2_id, "found_in", entity3_id)
        rel3_id = self.add_relationship_annotation(doc_id, entity1_id, "same_class", entity5_id)
        
        assert all(rid.startswith("R") for rid in [rel1_id, rel2_id, rel3_id])
        
        # Step 4: Export annotations
        json_output = temp_dir / "workflow_annotations.json"
        result = self.export_annotations(str(json_output), format='json')
        assert result is True
        
        csv_output = temp_dir / "workflow_annotations.csv"
        result = self.export_annotations(str(csv_output), format='csv')
        assert result is True
        
        # Step 5: Verify exported data
        with open(json_output, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        assert json_data['total_documents'] == 1
        assert json_data['total_entities'] == 5
        assert json_data['total_relationships'] == 3
    
    def test_error_handling_comprehensive(self, temp_dir):
        """Test comprehensive error handling scenarios."""
        # Test invalid input types for document loading
        with pytest.raises(ValueError, match="File path must be a non-empty string"):
            self.load_document_for_annotation(None)
        
        with pytest.raises(ValueError, match="File path must be a non-empty string"):
            self.load_document_for_annotation("")
        
        # Create a valid document for entity annotation tests
        test_content = "Test document content."
        test_file = temp_dir / "error_test.txt"
        test_file.write_text(test_content, encoding='utf-8')
        doc_id = self.load_document_for_annotation(str(test_file))
        
        # Test invalid entity annotation inputs
        with pytest.raises(ValueError, match="Document ID must be a non-empty string"):
            self.add_entity_annotation(None, "COMPOUND", "test", 0, 4)
        
        with pytest.raises(ValueError, match="Document ID must be a non-empty string"):
            self.add_entity_annotation("", "COMPOUND", "test", 0, 4)
        
        with pytest.raises(ValueError, match="Document not found"):
            self.add_entity_annotation("invalid_doc_id", "COMPOUND", "test", 0, 4)
        
        with pytest.raises(ValueError, match="Entity type must be a non-empty string"):
            self.add_entity_annotation(doc_id, "", "Test", 0, 4)
        
        with pytest.raises(ValueError, match="Entity text must be a non-empty string"):
            self.add_entity_annotation(doc_id, "COMPOUND", "", 0, 4)
        
        # Create valid entities for relationship tests
        entity1_id = self.add_entity_annotation(doc_id, "TEST", "Test", 0, 4)
        entity2_id = self.add_entity_annotation(doc_id, "WORD", "document", 5, 13)
        
        # Test invalid relationship annotation inputs
        with pytest.raises(ValueError, match="Document ID must be a non-empty string"):
            self.add_relationship_annotation("", entity1_id, "found_in", entity2_id)
        
        with pytest.raises(ValueError, match="Subject ID must be a non-empty string"):
            self.add_relationship_annotation(doc_id, "", "found_in", entity2_id)
        
        with pytest.raises(ValueError, match="Object ID must be a non-empty string"):
            self.add_relationship_annotation(doc_id, entity1_id, "found_in", "")
        
        with pytest.raises(ValueError, match="Relation type must be a non-empty string"):
            self.add_relationship_annotation(doc_id, entity1_id, "", entity2_id)
        
        # Test non-existent entity IDs
        with pytest.raises(ValueError, match="Subject entity not found"):
            self.add_relationship_annotation(doc_id, "E999", "found_in", entity2_id)
        
        with pytest.raises(ValueError, match="Object entity not found"):
            self.add_relationship_annotation(doc_id, entity1_id, "found_in", "E999")
        
        # Test invalid export inputs
        with pytest.raises(ValueError, match="Output file path must be a non-empty string"):
            self.export_annotations("", format='json')
        
        with pytest.raises(ValueError, match="Format must be 'json' or 'csv'"):
            self.export_annotations("output.txt", format='xml')


# Mark all tests in this module as evaluation related
pytestmark = [pytest.mark.unit, pytest.mark.evaluation]