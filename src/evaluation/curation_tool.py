"""
Manual curation and feedback loop tool for the AIM2-ODIE project.

This module provides functionality for human experts to review, correct, and curate
LLM-generated extractions of entities and relationships from biological literature.
The tool enables systematic feedback collection to improve prompt engineering and
model performance in plant metabolomics research.

Key Functions:
- load_llm_output: Load LLM-generated extraction results from JSON files
- display_for_review: CLI-based display of text and extracted items for review
- apply_correction: Apply human corrections to entities and relationships
- save_curated_output: Save curated data with applied corrections

The tool supports different correction types for entities (modify, add, remove)
and relations (modify, add, remove), allowing comprehensive curation of
extracted information.

Author: AIM2-ODIE System
Date: 2025-08-06
"""

import json
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timezone
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CurationError(Exception):
    """Base exception for curation tool errors."""
    pass


class InvalidExtractionFormatError(CurationError):
    """Exception raised when extraction data has invalid format."""
    pass


class CorrectionError(CurationError):
    """Exception raised when correction operations fail."""
    pass


def load_llm_output(file_path: str) -> dict:
    """
    Load LLM-generated extraction results from a JSON file.

    Reads and validates LLM-generated entities and relationships from a structured
    JSON file. The expected format includes document text, extracted entities with
    position information, and relationships as tuples.

    Args:
        file_path (str): Path to the JSON file containing LLM extraction results.
                        Expected structure:
                        {
                            "text": "document text",
                            "entities": [
                                {
                                    "entity_type": "COMPOUND",
                                    "text": "quercetin",
                                    "start_char": 0,
                                    "end_char": 9
                                }
                            ],
                            "relations": [
                                ["quercetin", "found_in", "Arabidopsis thaliana"]
                            ]
                        }

    Returns:
        dict: Dictionary containing validated extraction data with keys:
            - text (str): The source document text
            - entities (List[Dict]): List of entity dictionaries
            - relations (List[Tuple]): List of relation tuples

    Raises:
        ValueError: If file_path is invalid or file doesn't exist
        InvalidExtractionFormatError: If JSON structure is invalid or missing
                                     required fields
        CurationError: If file reading or JSON parsing fails

    Examples:
        >>> data = load_llm_output("extractions.json")
        >>> print(f"Found {len(data['entities'])} entities")
        >>> print(f"Found {len(data['relations'])} relations")
    """
    # Input validation
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("File path must be a non-empty string")
    
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    
    # Read and parse JSON file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise CurationError(f"Invalid JSON format in {file_path}: {str(e)}")
    except Exception as e:
        raise CurationError(f"Error reading file {file_path}: {str(e)}")
    
    # Validate data structure
    if not isinstance(data, dict):
        raise InvalidExtractionFormatError("Root element must be a dictionary")
    
    # Required fields validation
    required_fields = ['text', 'entities', 'relations']
    for field in required_fields:
        if field not in data:
            raise InvalidExtractionFormatError(
                f"Missing required field: {field}"
            )
    
    # Validate text field
    if not isinstance(data['text'], str):
        raise InvalidExtractionFormatError("'text' field must be a string")
    
    # Validate entities field
    if not isinstance(data['entities'], list):
        raise InvalidExtractionFormatError("'entities' field must be a list")
    
    # Validate entity structure
    entity_required_fields = ['entity_type', 'text', 'start_char', 'end_char']
    for i, entity in enumerate(data['entities']):
        if not isinstance(entity, dict):
            raise InvalidExtractionFormatError(
                f"entities[{i}] must be a dictionary"
            )
        
        for field in entity_required_fields:
            if field not in entity:
                raise InvalidExtractionFormatError(
                    f"entities[{i}] missing required field: {field}"
                )
        
        # Validate field types
        if not isinstance(entity['entity_type'], str):
            raise InvalidExtractionFormatError(
                f"entities[{i}]['entity_type'] must be a string"
            )
        if not isinstance(entity['text'], str):
            raise InvalidExtractionFormatError(
                f"entities[{i}]['text'] must be a string"
            )
        if not isinstance(entity['start_char'], int):
            raise InvalidExtractionFormatError(
                f"entities[{i}]['start_char'] must be an integer"
            )
        if not isinstance(entity['end_char'], int):
            raise InvalidExtractionFormatError(
                f"entities[{i}]['end_char'] must be an integer"
            )
        
        # Validate character positions
        if entity['start_char'] < 0:
            raise InvalidExtractionFormatError(
                f"entities[{i}]['start_char'] must be non-negative"
            )
        if entity['end_char'] <= entity['start_char']:
            raise InvalidExtractionFormatError(
                f"entities[{i}]['end_char'] must be greater than start_char"
            )
    
    # Validate relations field
    if not isinstance(data['relations'], list):
        raise InvalidExtractionFormatError("'relations' field must be a list")
    
    # Validate relation structure
    for i, relation in enumerate(data['relations']):
        if not isinstance(relation, (list, tuple)):
            raise InvalidExtractionFormatError(
                f"relations[{i}] must be a list or tuple"
            )
        if len(relation) != 3:
            raise InvalidExtractionFormatError(
                f"relations[{i}] must have exactly 3 elements "
                "(subject, relation_type, object)"
            )
        
        # Validate all elements are strings
        for j, element in enumerate(relation):
            if not isinstance(element, str):
                raise InvalidExtractionFormatError(
                    f"relations[{i}][{j}] must be a string"
                )
    
    # Convert relations to tuples for consistency
    normalized_relations = [tuple(rel) for rel in data['relations']]
    
    result = {
        'text': data['text'],
        'entities': data['entities'],
        'relations': normalized_relations
    }
    
    logger.info(f"Loaded LLM output from {file_path}: "
               f"{len(result['entities'])} entities, "
               f"{len(result['relations'])} relations")
    
    return result


def display_for_review(
    text: str, 
    entities: List[dict], 
    relations: List[Tuple[str, str, str]]
) -> None:
    """
    Display text and extracted items in a clear, reviewable format for CLI.

    Presents the source text along with extracted entities and relationships
    in an organized, human-readable format suitable for expert review and
    correction. Shows entities with their types and positions, and relationships
    as subject-predicate-object triples.

    Args:
        text (str): The source document text to display
        entities (List[dict]): List of entity dictionaries containing:
            - entity_type (str): Entity category/type
            - text (str): Entity text span
            - start_char (int): Starting character position
            - end_char (int): Ending character position
        relations (List[Tuple[str, str, str]]): List of relationship tuples
                                                (subject, relation_type, object)

    Raises:
        ValueError: If input parameters have invalid types or structure

    Examples:
        >>> entities = [{"entity_type": "COMPOUND", "text": "quercetin", 
        ...              "start_char": 0, "end_char": 9}]
        >>> relations = [("quercetin", "found_in", "Arabidopsis")]
        >>> display_for_review("quercetin study", entities, relations)
        # Displays formatted output to console
    """
    # Input validation
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    
    if not isinstance(entities, list):
        raise ValueError("entities must be a list")
    
    if not isinstance(relations, list):
        raise ValueError("relations must be a list")
    
    # Validate entities structure (basic validation for display)
    for i, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise ValueError(f"entities[{i}] must be a dictionary")
        required_fields = ['entity_type', 'text', 'start_char', 'end_char']
        for field in required_fields:
            if field not in entity:
                raise ValueError(f"entities[{i}] missing field: {field}")
    
    # Validate relations structure (basic validation for display)
    for i, relation in enumerate(relations):
        if not isinstance(relation, (tuple, list)):
            raise ValueError(f"relations[{i}] must be a tuple or list")
        if len(relation) != 3:
            raise ValueError(f"relations[{i}] must have 3 elements")
    
    # Display header
    print("=" * 80)
    print("LLM EXTRACTION REVIEW")
    print("=" * 80)
    print()
    
    # Display source text
    print("SOURCE TEXT:")
    print("-" * 40)
    print(text)
    print()
    
    # Display entities
    print(f"EXTRACTED ENTITIES ({len(entities)}):")
    print("-" * 40)
    if not entities:
        print("No entities found.")
    else:
        # Sort entities by start position for logical display
        sorted_entities = sorted(entities, key=lambda x: x['start_char'])
        
        for i, entity in enumerate(sorted_entities, 1):
            entity_text = entity['text']
            entity_type = entity['entity_type']
            start_pos = entity['start_char']
            end_pos = entity['end_char']
            
            print(f"{i:2}. [{entity_type}] '{entity_text}' "
                  f"(pos: {start_pos}-{end_pos})")
            
            # Show context (10 characters before and after)
            context_start = max(0, start_pos - 10)
            context_end = min(len(text), end_pos + 10)
            context = text[context_start:context_end]
            
            # Highlight the entity in context
            entity_in_context = context.replace(
                entity_text, f"**{entity_text}**"
            )
            print(f"    Context: ...{entity_in_context}...")
            print()
    
    # Display relations
    print(f"EXTRACTED RELATIONS ({len(relations)}):")
    print("-" * 40)
    if not relations:
        print("No relations found.")
    else:
        for i, relation in enumerate(relations, 1):
            subject, relation_type, obj = relation
            print(f"{i:2}. {subject} --[{relation_type}]--> {obj}")
    
    print()
    print("=" * 80)
    print("END OF REVIEW")
    print("=" * 80)
    
    logger.info(f"Displayed for review: {len(entities)} entities, "
               f"{len(relations)} relations")


def apply_correction(
    extracted_data: dict, 
    correction_type: str, 
    old_value: Any, 
    new_value: Any
) -> dict:
    """
    Apply human correction to extracted entities or relationships.

    Modifies the extracted data based on expert corrections, supporting
    addition, removal, and modification of entities and relationships.
    Returns a new dictionary with the corrections applied.

    Args:
        extracted_data (dict): Original extraction data containing:
            - text (str): Source document text
            - entities (List[dict]): Entity list
            - relations (List[Tuple]): Relation list
        correction_type (str): Type of correction to apply:
            - "modify_entity": Modify existing entity
            - "add_entity": Add new entity
            - "remove_entity": Remove existing entity
            - "modify_relation": Modify existing relation
            - "add_relation": Add new relation
            - "remove_relation": Remove existing relation
        old_value (Any): For modify/remove operations, the item to change/remove.
                        For entities: dict with entity data
                        For relations: tuple (subject, relation_type, object)
        new_value (Any): For modify/add operations, the new/replacement item.
                        Same format as old_value.

    Returns:
        dict: New dictionary with corrections applied, maintaining original
              structure but with modified entities/relations lists

    Raises:
        ValueError: If input parameters are invalid
        CorrectionError: If correction operation fails (e.g., item not found
                        for modification/removal)

    Examples:
        >>> data = {"text": "test", "entities": [...], "relations": [...]}
        >>> # Add new entity
        >>> corrected = apply_correction(data, "add_entity", None, 
        ...                             {"entity_type": "COMPOUND", "text": "new",
        ...                              "start_char": 0, "end_char": 3})
        >>> # Remove relation
        >>> corrected = apply_correction(data, "remove_relation",
        ...                             ("old", "relation", "value"), None)
    """
    # Input validation
    if not isinstance(extracted_data, dict):
        raise ValueError("extracted_data must be a dictionary")
    
    required_fields = ['text', 'entities', 'relations']
    for field in required_fields:
        if field not in extracted_data:
            raise ValueError(f"extracted_data missing required field: {field}")
    
    if not isinstance(correction_type, str):
        raise ValueError("correction_type must be a string")
    
    valid_correction_types = [
        'modify_entity', 'add_entity', 'remove_entity',
        'modify_relation', 'add_relation', 'remove_relation'
    ]
    
    if correction_type not in valid_correction_types:
        raise ValueError(
            f"Invalid correction_type: {correction_type}. "
            f"Must be one of: {valid_correction_types}"
        )
    
    # Create deep copy to avoid modifying original data
    corrected_data = {
        'text': extracted_data['text'],
        'entities': extracted_data['entities'].copy(),
        'relations': [tuple(rel) for rel in extracted_data['relations']]
    }
    
    try:
        # Handle entity corrections
        if correction_type.endswith('_entity'):
            if correction_type == 'add_entity':
                if new_value is None:
                    raise CorrectionError("new_value cannot be None for add_entity")
                
                # Validate new entity structure
                if not isinstance(new_value, dict):
                    raise CorrectionError("new_value must be a dictionary for entities")
                
                required_entity_fields = ['entity_type', 'text', 'start_char', 'end_char']
                for field in required_entity_fields:
                    if field not in new_value:
                        raise CorrectionError(f"new_value missing field: {field}")
                
                corrected_data['entities'].append(new_value.copy())
                logger.info(f"Added entity: {new_value['entity_type']} '{new_value['text']}'")
            
            elif correction_type == 'remove_entity':
                if old_value is None:
                    raise CorrectionError("old_value cannot be None for remove_entity")
                
                # Find and remove entity
                entity_found = False
                for i, entity in enumerate(corrected_data['entities']):
                    if entity == old_value:
                        corrected_data['entities'].pop(i)
                        entity_found = True
                        logger.info(f"Removed entity: {old_value}")
                        break
                
                if not entity_found:
                    raise CorrectionError(f"Entity not found for removal: {old_value}")
            
            elif correction_type == 'modify_entity':
                if old_value is None or new_value is None:
                    raise CorrectionError("Both old_value and new_value required for modify_entity")
                
                # Find and modify entity
                entity_found = False
                for i, entity in enumerate(corrected_data['entities']):
                    if entity == old_value:
                        corrected_data['entities'][i] = new_value.copy()
                        entity_found = True
                        logger.info(f"Modified entity: {old_value} -> {new_value}")
                        break
                
                if not entity_found:
                    raise CorrectionError(f"Entity not found for modification: {old_value}")
        
        # Handle relation corrections
        elif correction_type.endswith('_relation'):
            if correction_type == 'add_relation':
                if new_value is None:
                    raise CorrectionError("new_value cannot be None for add_relation")
                
                # Validate new relation structure
                if not isinstance(new_value, (tuple, list)):
                    raise CorrectionError("new_value must be a tuple or list for relations")
                
                if len(new_value) != 3:
                    raise CorrectionError("new_value must have 3 elements for relations")
                
                new_relation = tuple(new_value)
                corrected_data['relations'].append(new_relation)
                logger.info(f"Added relation: {new_relation}")
            
            elif correction_type == 'remove_relation':
                if old_value is None:
                    raise CorrectionError("old_value cannot be None for remove_relation")
                
                # Find and remove relation
                old_relation = tuple(old_value) if isinstance(old_value, list) else old_value
                
                if old_relation in corrected_data['relations']:
                    corrected_data['relations'].remove(old_relation)
                    logger.info(f"Removed relation: {old_relation}")
                else:
                    raise CorrectionError(f"Relation not found for removal: {old_relation}")
            
            elif correction_type == 'modify_relation':
                if old_value is None or new_value is None:
                    raise CorrectionError("Both old_value and new_value required for modify_relation")
                
                # Find and modify relation
                old_relation = tuple(old_value) if isinstance(old_value, list) else old_value
                new_relation = tuple(new_value) if isinstance(new_value, list) else new_value
                
                relation_found = False
                for i, relation in enumerate(corrected_data['relations']):
                    if relation == old_relation:
                        corrected_data['relations'][i] = new_relation
                        relation_found = True
                        logger.info(f"Modified relation: {old_relation} -> {new_relation}")
                        break
                
                if not relation_found:
                    raise CorrectionError(f"Relation not found for modification: {old_relation}")
    
    except Exception as e:
        if isinstance(e, CorrectionError):
            raise
        else:
            raise CorrectionError(f"Error applying correction: {str(e)}")
    
    return corrected_data


def save_curated_output(curated_data: dict, output_file: str) -> bool:
    """
    Save curated extraction data with corrections to a JSON file.

    Exports the curated data to a structured JSON file with metadata
    about the curation process, including timestamp and original vs
    corrected counts for tracking purposes.

    Args:
        curated_data (dict): Curated extraction data containing:
            - text (str): Source document text
            - entities (List[dict]): Curated entity list
            - relations (List[Tuple]): Curated relation list
        output_file (str): Path to the output JSON file

    Returns:
        bool: True if save operation was successful

    Raises:
        ValueError: If input parameters are invalid
        CurationError: If file writing fails

    Examples:
        >>> curated = {"text": "sample", "entities": [...], "relations": [...]}
        >>> success = save_curated_output(curated, "curated_output.json")
        >>> print(f"Save successful: {success}")
        True
    """
    # Input validation
    if not isinstance(curated_data, dict):
        raise ValueError("curated_data must be a dictionary")
    
    required_fields = ['text', 'entities', 'relations']
    for field in required_fields:
        if field not in curated_data:
            raise ValueError(f"curated_data missing required field: {field}")
    
    if not isinstance(output_file, str) or not output_file.strip():
        raise ValueError("output_file must be a non-empty string")
    
    # Prepare output data with metadata
    output_data = {
        'metadata': {
            'curation_timestamp': datetime.now(timezone.utc).isoformat(),
            'tool_name': 'AIM2-ODIE Curation Tool',
            'tool_version': '1.0.0',
            'format_version': '1.0'
        },
        'statistics': {
            'total_entities': len(curated_data['entities']),
            'total_relations': len(curated_data['relations']),
            'text_length': len(curated_data['text'])
        },
        'curated_data': {
            'text': curated_data['text'],
            'entities': curated_data['entities'],
            'relations': [list(rel) for rel in curated_data['relations']]  # Convert tuples to lists for JSON
        }
    }
    
    try:
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully saved curated output to {output_file}: "
                   f"{len(curated_data['entities'])} entities, "
                   f"{len(curated_data['relations'])} relations")
        
        return True
    
    except Exception as e:
        error_msg = f"Error saving curated output to {output_file}: {str(e)}"
        logger.error(error_msg)
        raise CurationError(error_msg)