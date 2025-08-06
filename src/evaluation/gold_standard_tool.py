"""
Gold standard annotation tool for the AIM2-ODIE project.

This module provides a simple CLI tool and Python API for facilitating manual
annotation of biological literature to create gold standard test sets for
Named Entity Recognition (NER) and relationship extraction evaluation.

The tool supports:
- Loading text and PDF documents for annotation
- Adding entity annotations with span validation
- Adding relationship annotations between entities
- Exporting annotations in structured formats (JSON Lines, CSV)
- In-memory storage during annotation sessions

Example Usage:
    # As a Python module
    from src.evaluation.gold_standard_tool import GoldStandardTool
    
    tool = GoldStandardTool()
    doc_id = tool.load_document_for_annotation("paper.txt")
    entity_id = tool.add_entity_annotation(doc_id, "COMPOUND", "quercetin", 0, 9)
    tool.export_annotations("annotations.json")
    
    # As a CLI (when run directly)
    python gold_standard_tool.py --help

Author: AIM2 Development Team
Date: August 2025
"""

import json
import csv
import os
import uuid
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GoldStandardError(Exception):
    """Base exception for gold standard annotation tool errors."""
    pass


class AnnotationConflictError(GoldStandardError):
    """Exception raised when annotation conflicts occur (e.g., overlapping spans)."""
    pass


class InvalidDocumentError(GoldStandardError):
    """Exception raised for invalid document formats or content."""
    pass


class GoldStandardTool:
    """
    Gold standard annotation tool for manual annotation of biological literature.
    
    This tool allows users to load documents, annotate entities and relationships,
    and export the annotations in structured formats for evaluation purposes.
    """
    
    def __init__(self):
        """Initialize the gold standard annotation tool with empty storage."""
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relationships: Dict[str, Dict[str, Any]] = {}
        self.next_entity_id = 1
        self.next_relationship_id = 1
        
        logger.info("Gold standard annotation tool initialized")
    
    def load_document_for_annotation(self, file_path: str) -> str:
        """
        Load a document for annotation from a text or PDF file.
        
        Args:
            file_path (str): Path to the document file (.txt or .pdf)
            
        Returns:
            str: Unique document ID for referencing in annotations
            
        Raises:
            ValueError: If file_path is invalid
            FileNotFoundError: If the file doesn't exist
            InvalidDocumentError: If file format is unsupported or document is empty
        """
        # Input validation
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("File path must be a non-empty string")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Validate file format
        path_obj = Path(file_path)
        supported_formats = {'.txt', '.pdf'}
        if path_obj.suffix.lower() not in supported_formats:
            raise InvalidDocumentError(
                f"Unsupported file format: {path_obj.suffix}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )
        
        # Read file content
        try:
            if path_obj.suffix.lower() == '.pdf':
                # For PDF files, we'll read as text for now
                # In a production system, you'd use a PDF library like PyMuPDF or pdfplumber
                content = self._read_pdf_as_text(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
        except UnicodeDecodeError:
            raise InvalidDocumentError(f"Unable to decode file: {file_path}")
        except Exception as e:
            raise InvalidDocumentError(f"Error reading file: {file_path}. {str(e)}")
        
        # Validate content
        if not content.strip():
            raise InvalidDocumentError("Document is empty or contains only whitespace")
        
        # Generate unique document ID and store
        doc_id = str(uuid.uuid4())
        self.documents[doc_id] = {
            'id': doc_id,
            'file_path': os.path.abspath(file_path),
            'content': content,
            'entities': [],
            'relationships': [],
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Loaded document {doc_id} from {file_path}")
        return doc_id
    
    def _read_pdf_as_text(self, file_path: str) -> str:
        """
        Read PDF file as text. Placeholder implementation for demo purposes.
        
        In production, this should use a proper PDF parsing library.
        """
        # For now, read as text file for testing purposes
        # In production, use PyMuPDF, pdfplumber, or similar
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # If binary PDF, return a placeholder
            return "PDF content would be extracted here using a PDF library like PyMuPDF"
    
    def add_entity_annotation(
        self,
        doc_id: str,
        entity_type: str,
        text: str,
        start_char: int,
        end_char: int
    ) -> str:
        """
        Add an entity annotation to a document.
        
        Args:
            doc_id (str): Document ID returned by load_document_for_annotation
            entity_type (str): Type/category of the entity (e.g., "COMPOUND", "ORGANISM")
            text (str): The text span being annotated
            start_char (int): Starting character position (0-based)
            end_char (int): Ending character position (exclusive)
            
        Returns:
            str: Unique entity ID for the annotation
            
        Raises:
            ValueError: If any input parameters are invalid
            AnnotationConflictError: If the annotation overlaps with existing ones
        """
        # Input validation
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("Document ID must be a non-empty string")
        
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")
        
        if not isinstance(entity_type, str) or not entity_type.strip():
            raise ValueError("Entity type must be a non-empty string")
        
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Entity text must be a non-empty string")
        
        if not isinstance(start_char, int) or start_char < 0:
            raise ValueError("Start character must be a non-negative integer")
        
        if not isinstance(end_char, int) or end_char <= start_char:
            raise ValueError("End character must be greater than start character")
        
        # Get document and validate span
        document = self.documents[doc_id]
        doc_content = document['content']
        
        if end_char > len(doc_content):
            raise ValueError(
                f"End character ({end_char}) exceeds document length ({len(doc_content)})"
            )
        
        # Validate that text matches the span
        extracted_text = doc_content[start_char:end_char]
        if extracted_text != text:
            raise ValueError(
                f"Text does not match span: expected '{text}', got '{extracted_text}'"
            )
        
        # Check for overlapping annotations
        for existing_entity in document['entities']:
            existing_start = existing_entity['start_char']
            existing_end = existing_entity['end_char']
            
            # Check for overlap: new span overlaps if it starts before existing ends
            # and ends after existing starts
            if start_char < existing_end and end_char > existing_start:
                raise AnnotationConflictError(
                    f"Entity annotation ({start_char}-{end_char}) overlaps with "
                    f"existing annotation {existing_entity['id']} "
                    f"({existing_start}-{existing_end})"
                )
        
        # Generate entity ID and create annotation
        entity_id = f"E{self.next_entity_id}"
        self.next_entity_id += 1
        
        entity_annotation = {
            'id': entity_id,
            'doc_id': doc_id,
            'entity_type': entity_type.strip(),
            'text': text.strip(),
            'start_char': start_char,
            'end_char': end_char,
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Store annotation
        document['entities'].append(entity_annotation)
        self.entities[entity_id] = entity_annotation
        
        logger.info(f"Added entity annotation {entity_id}: {entity_type} '{text}' at {start_char}-{end_char}")
        return entity_id
    
    def add_relationship_annotation(
        self,
        doc_id: str,
        subject_id: str,
        relation_type: str,
        object_id: str
    ) -> str:
        """
        Add a relationship annotation between two entities.
        
        Args:
            doc_id (str): Document ID containing both entities
            subject_id (str): Entity ID of the subject/source entity
            relation_type (str): Type of relationship (e.g., "found_in", "inhibits")
            object_id (str): Entity ID of the object/target entity
            
        Returns:
            str: Unique relationship ID for the annotation
            
        Raises:
            ValueError: If any input parameters are invalid or entities don't exist
            AnnotationConflictError: If the relationship already exists
        """
        # Input validation
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("Document ID must be a non-empty string")
        
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")
        
        if not isinstance(subject_id, str) or not subject_id:
            raise ValueError("Subject ID must be a non-empty string")
        
        if subject_id not in self.entities:
            raise ValueError(f"Subject entity not found: {subject_id}")
        
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("Object ID must be a non-empty string")
        
        if object_id not in self.entities:
            raise ValueError(f"Object entity not found: {object_id}")
        
        if not isinstance(relation_type, str) or not relation_type.strip():
            raise ValueError("Relation type must be a non-empty string")
        
        # Validate that both entities belong to the specified document
        subject_entity = self.entities[subject_id]
        object_entity = self.entities[object_id]
        
        if subject_entity['doc_id'] != doc_id:
            raise ValueError(f"Subject entity {subject_id} does not belong to document {doc_id}")
        
        if object_entity['doc_id'] != doc_id:
            raise ValueError(f"Object entity {object_id} does not belong to document {doc_id}")
        
        # Check for duplicate relationships
        document = self.documents[doc_id]
        relation_type_clean = relation_type.strip()
        
        for existing_rel in document['relationships']:
            if (existing_rel['subject_id'] == subject_id and 
                existing_rel['object_id'] == object_id and 
                existing_rel['relation_type'] == relation_type_clean):
                raise AnnotationConflictError(
                    f"Relationship already exists: {subject_id} -{relation_type_clean}-> {object_id}"
                )
        
        # Generate relationship ID and create annotation
        rel_id = f"R{self.next_relationship_id}"
        self.next_relationship_id += 1
        
        relationship_annotation = {
            'id': rel_id,
            'doc_id': doc_id,
            'subject_id': subject_id,
            'relation_type': relation_type_clean,
            'object_id': object_id,
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Store annotation
        document['relationships'].append(relationship_annotation)
        self.relationships[rel_id] = relationship_annotation
        
        logger.info(f"Added relationship annotation {rel_id}: {subject_id} -{relation_type_clean}-> {object_id}")
        return rel_id
    
    def export_annotations(self, output_file: str, format: str = 'json') -> bool:
        """
        Export all annotations to a structured file format.
        
        Args:
            output_file (str): Path to the output file
            format (str): Export format - 'json' (default), 'csv', or 'jsonl'
            
        Returns:
            bool: True if export was successful
            
        Raises:
            ValueError: If output file path or format is invalid
            GoldStandardError: If export operation fails
        """
        # Input validation
        if not isinstance(output_file, str) or not output_file.strip():
            raise ValueError("Output file path must be a non-empty string")
        
        if not isinstance(format, str) or format.lower() not in ['json', 'csv', 'jsonl']:
            raise ValueError("Format must be 'json', 'csv', or 'jsonl'")
        
        try:
            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            format_lower = format.lower()
            
            if format_lower == 'json':
                self._export_json(output_file)
            elif format_lower == 'csv':
                self._export_csv(output_file)
            elif format_lower == 'jsonl':
                self._export_jsonl(output_file)
            
            logger.info(f"Successfully exported annotations to {output_file} in {format} format")
            return True
        
        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            logger.error(error_msg)
            raise GoldStandardError(error_msg)
    
    def _export_json(self, output_file: str) -> None:
        """Export annotations in JSON format."""
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now(timezone.utc).isoformat(),
                'tool_version': '1.0.0',
                'format_version': '1.0'
            },
            'statistics': {
                'total_documents': len(self.documents),
                'total_entities': len(self.entities),
                'total_relationships': len(self.relationships)
            },
            'documents': []
        }
        
        # Add document data
        for doc_id, document in self.documents.items():
            doc_data = {
                'id': doc_id,
                'file_path': document['file_path'],
                'created_at': document['created_at'],
                'entities': document['entities'].copy(),
                'relationships': document['relationships'].copy()
            }
            export_data['documents'].append(doc_data)
        
        # Write JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def _export_csv(self, output_file: str) -> None:
        """Export annotations in CSV format."""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['type', 'doc_id', 'id', 'entity_type', 'text', 'start_char', 'end_char']
            writer.writerow(header)
            
            # Write entity annotations
            for entity in self.entities.values():
                writer.writerow([
                    'entity',
                    entity['doc_id'],
                    entity['id'],
                    entity['entity_type'],
                    entity['text'],
                    entity['start_char'],
                    entity['end_char']
                ])
            
            # Write relationship annotations
            for relationship in self.relationships.values():
                # For relationships, we use the same column structure but adapt the content
                writer.writerow([
                    'relationship',
                    relationship['doc_id'],
                    relationship['id'],
                    relationship['subject_id'],  # reuse entity_type column
                    relationship['relation_type'],  # reuse text column
                    relationship['object_id'],  # reuse start_char column
                    ''  # empty end_char for relationships
                ])
    
    def _export_jsonl(self, output_file: str) -> None:
        """Export annotations in JSON Lines format."""
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write each document as a separate JSON line
            for doc_id, document in self.documents.items():
                doc_data = {
                    'id': doc_id,
                    'file_path': document['file_path'],
                    'created_at': document['created_at'],
                    'entities': document['entities'].copy(),
                    'relationships': document['relationships'].copy()
                }
                # Write as a single line JSON object
                f.write(json.dumps(doc_data, ensure_ascii=False) + '\n')
    
    def get_document_info(self, doc_id: str) -> Dict[str, Any]:
        """
        Get information about a loaded document.
        
        Args:
            doc_id (str): Document ID
            
        Returns:
            dict: Document information including entity and relationship counts
            
        Raises:
            ValueError: If document ID is invalid
        """
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("Document ID must be a non-empty string")
        
        if doc_id not in self.documents:
            raise ValueError(f"Document not found: {doc_id}")
        
        document = self.documents[doc_id]
        return {
            'id': doc_id,
            'file_path': document['file_path'],
            'content_length': len(document['content']),
            'entity_count': len(document['entities']),
            'relationship_count': len(document['relationships']),
            'created_at': document['created_at']
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all loaded documents with summary information.
        
        Returns:
            list: List of document summaries
        """
        return [self.get_document_info(doc_id) for doc_id in self.documents.keys()]
    
    def clear_all_annotations(self) -> None:
        """Clear all documents and annotations from memory."""
        self.documents.clear()
        self.entities.clear()
        self.relationships.clear()
        self.next_entity_id = 1
        self.next_relationship_id = 1
        logger.info("Cleared all annotations from memory")


def main():
    """Command-line interface for the gold standard annotation tool."""
    parser = argparse.ArgumentParser(
        description="Gold Standard Annotation Tool for AIM2-ODIE Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load a document and start annotation session
  python gold_standard_tool.py load document.txt
  
  # Export annotations
  python gold_standard_tool.py export annotations.json --format json
  
  # Show help for specific commands
  python gold_standard_tool.py load --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load a document for annotation')
    load_parser.add_argument('file_path', help='Path to the document file (.txt or .pdf)')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export annotations')
    export_parser.add_argument('output_file', help='Output file path')
    export_parser.add_argument('--format', choices=['json', 'csv', 'jsonl'], default='json',
                              help='Export format (default: json)')
    
    # List command
    subparsers.add_parser('list', help='List loaded documents')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Start interactive annotation session')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    tool = GoldStandardTool()
    
    try:
        if args.command == 'load':
            doc_id = tool.load_document_for_annotation(args.file_path)
            print(f"Document loaded successfully. ID: {doc_id}")
            
        elif args.command == 'export':
            success = tool.export_annotations(args.output_file, args.format)
            if success:
                print(f"Annotations exported successfully to {args.output_file}")
            
        elif args.command == 'list':
            documents = tool.list_documents()
            if not documents:
                print("No documents loaded.")
            else:
                print(f"Loaded documents ({len(documents)}):")
                for doc in documents:
                    print(f"  {doc['id']}: {doc['file_path']} "
                          f"({doc['entity_count']} entities, "
                          f"{doc['relationship_count']} relationships)")
        
        elif args.command == 'interactive':
            print("Interactive annotation mode not implemented in CLI version.")
            print("Use the Python API for interactive annotation.")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())