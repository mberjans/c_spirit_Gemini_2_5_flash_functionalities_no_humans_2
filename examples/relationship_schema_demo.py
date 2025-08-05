#!/usr/bin/env python3
"""
Demonstration script for relationship schemas in plant metabolomics research.

This script shows how to use the relationship schema system to:
1. Validate relationship patterns
2. Find compatible relationships for entity types
3. Work with domain-specific schemas
4. Use the schemas for relationship extraction

Usage:
    python examples/relationship_schema_demo.py
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_extraction.relationship_schemas import (
    get_plant_metabolomics_relationship_schema,
    get_basic_relationship_schema,
    validate_relationship_pattern,
    get_compatible_relationships,
    get_domain_specific_schema,
    convert_schema_to_simple_dict,
    get_relationship_statistics,
    METABOLOMICS_RELATIONSHIP_SCHEMA,
    GENETICS_RELATIONSHIP_SCHEMA
)


def main():
    """Main demonstration function."""
    print("=== Plant Metabolomics Relationship Schema Demonstration ===\n")
    
    # 1. Show basic schema information
    print("1. Basic Schema Information")
    print("-" * 40)
    
    full_schema = get_plant_metabolomics_relationship_schema()
    basic_schema = get_basic_relationship_schema()
    
    print(f"Full schema contains {len(full_schema)} relationship types")
    print(f"Basic schema contains {len(basic_schema)} relationship types")
    
    # Show some relationship types
    print("\nSample relationship types:")
    for i, (rel_type, pattern) in enumerate(list(full_schema.items())[:5]):
        print(f"- {rel_type}: {pattern.description[:80]}...")
    
    print("\n")
    
    # 2. Demonstrate specific requested relationships
    print("2. Key Requested Relationships")
    print("-" * 40)
    
    # Compound-Affects-Trait relationship
    print("Compound-Affects-Trait relationship:")
    affects_pattern = full_schema["affects"]
    print(f"Description: {affects_pattern.description}")
    print(f"Domain (subjects): {', '.join(list(affects_pattern.domain)[:5])}...")
    print(f"Range (objects): {', '.join(list(affects_pattern.range)[:5])}...")
    
    print("\nMetabolite-InvolvedIn-BiologicalProcess relationship:")
    involved_pattern = full_schema["involved_in_biological_process"]
    print(f"Description: {involved_pattern.description}")
    print(f"Domain (subjects): {', '.join(list(involved_pattern.domain)[:5])}...")
    print(f"Range (objects): {', '.join(list(involved_pattern.range)[:3])}...")
    
    print("\n")
    
    # 3. Demonstrate relationship pattern validation
    print("3. Relationship Pattern Validation")
    print("-" * 40)
    
    # Valid patterns
    valid_patterns = [
        ("COMPOUND", "affects", "PLANT_TRAIT"),
        ("METABOLITE", "found_in", "LEAF"),
        ("GENE", "encodes", "PROTEIN"),
        ("METABOLITE", "involved_in_biological_process", "METABOLIC_PATHWAY")
    ]
    
    print("Valid relationship patterns:")
    for subject, relation, obj in valid_patterns:
        is_valid = validate_relationship_pattern(subject, relation, obj)
        print(f"✓ {subject} --{relation}--> {obj}: {is_valid}")
    
    # Invalid patterns
    invalid_patterns = [
        ("GENE", "accumulates_in", "SPECIES"),  # Wrong domain/range
        ("METABOLITE", "invalid_relation", "PLANT_TRAIT"),  # Non-existent relation
        ("INVALID_TYPE", "affects", "PLANT_TRAIT")  # Invalid entity type
    ]
    
    print("\nInvalid relationship patterns:")
    for subject, relation, obj in invalid_patterns:
        is_valid = validate_relationship_pattern(subject, relation, obj)
        print(f"✗ {subject} --{relation}--> {obj}: {is_valid}")
    
    print("\n")
    
    # 4. Find compatible relationships
    print("4. Finding Compatible Relationships")
    print("-" * 40)
    
    entity_pairs = [
        ("METABOLITE", "SPECIES"),
        ("COMPOUND", "PLANT_TRAIT"),
        ("GENE", "PROTEIN"),
        ("ENZYME", "METABOLIC_PATHWAY")
    ]
    
    for subject_type, object_type in entity_pairs:
        compatible = get_compatible_relationships(subject_type, object_type)
        print(f"{subject_type} -> {object_type}:")
        print(f"  Compatible relations: {', '.join(compatible[:3])}{'...' if len(compatible) > 3 else ''}")
        print(f"  Total: {len(compatible)} relationships")
    
    print("\n")
    
    # 5. Domain-specific schemas
    print("5. Domain-Specific Schemas")
    print("-" * 40)
    
    domains = ["metabolomics", "genetics", "biochemistry"]
    
    for domain in domains:
        domain_schema = get_domain_specific_schema(domain)
        print(f"{domain.capitalize()} domain:")
        print(f"  Relationships: {len(domain_schema)}")
        print(f"  Types: {', '.join(list(domain_schema.keys())[:4])}...")
    
    print("\n")
    
    # 6. Schema statistics
    print("6. Schema Statistics")
    print("-" * 40)
    
    stats = get_relationship_statistics(full_schema)
    print(f"Total relationships: {stats['total_relationships']}")
    print(f"Unique domain types: {stats['unique_domain_types']}")
    print(f"Unique range types: {stats['unique_range_types']}")
    print(f"Symmetric relationships: {stats['symmetric_relationships']}")
    print(f"Transitive relationships: {stats['transitive_relationships']}")
    print(f"Relationships with inverse: {stats['relationships_with_inverse']}")
    
    print("\n")
    
    # 7. Practical usage example
    print("7. Practical Usage Example")
    print("-" * 40)
    
    # Simulate extracted entities from text
    entities = [
        {"text": "anthocyanins", "label": "FLAVONOID"},
        {"text": "grape berries", "label": "FRUIT"},
        {"text": "flower color", "label": "MORPHOLOGICAL_TRAIT"},
        {"text": "flavonoid biosynthesis pathway", "label": "METABOLIC_PATHWAY"}
    ]
    
    print("Given entities from text:")
    for entity in entities:
        print(f"- {entity['text']} ({entity['label']})")
    
    print("\nPossible relationships:")
    
    # Check all possible entity pairs
    for i, entity1 in enumerate(entities):
        for entity2 in entities[i+1:]:
            compatible_forward = get_compatible_relationships(entity1["label"], entity2["label"])
            compatible_reverse = get_compatible_relationships(entity2["label"], entity1["label"])
            
            if compatible_forward:
                for rel in compatible_forward[:2]:  # Show first 2 relationships
                    print(f"- {entity1['text']} --{rel}--> {entity2['text']}")
            
            if compatible_reverse:
                for rel in compatible_reverse[:2]:  # Show first 2 relationships
                    print(f"- {entity2['text']} --{rel}--> {entity1['text']}")
    
    print("\n")
    
    # 8. Converting to simple format for LLM usage
    print("8. Converting Schema for LLM Usage")
    print("-" * 40)
    
    simple_metabolomics = convert_schema_to_simple_dict(METABOLOMICS_RELATIONSHIP_SCHEMA)
    print("Metabolomics schema in simple format:")
    for rel_type, description in list(simple_metabolomics.items())[:3]:
        print(f"- {rel_type}: {description[:60]}...")
    
    print(f"\nThis simple format can be directly used with LLM relationship extraction.")
    
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()