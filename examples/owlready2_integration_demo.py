#!/usr/bin/env python3
"""
Demo script showing Owlready2 integration with entity_mapper.

This script demonstrates the new functionality that allows users to pass
Owlready2 ontology objects directly to the entity mapping function, providing
better integration for users who already have loaded ontologies.

Features demonstrated:
1. Loading ontologies with Owlready2
2. Using loaded ontologies directly with entity_mapper
3. Comparison with traditional string IRI approach
4. Error handling for invalid objects

Note: This is a demonstration script. The actual mapping calls are mocked
since real ontologies require network access and text2term installation.
"""

import sys
from typing import List
from unittest.mock import patch, Mock
import pandas as pd

# Add the project root to the path to import our modules
sys.path.append('/Users/Mark/Research/C-Spirit/c_spirit_Gemini_2_5_flash_functionalities_no_humans_2')

from src.ontology_mapping.entity_mapper import (
    map_entities_to_ontology,
    _is_owlready2_ontology,
    _extract_iri_from_owlready2_ontology,
    _validate_target_ontology,
    InvalidOwlready2ObjectError,
    OWLREADY2_AVAILABLE
)


def demo_traditional_approach():
    """Demonstrate the traditional string IRI approach (backward compatible)."""
    print("=" * 60)
    print("1. TRADITIONAL APPROACH (String IRI)")
    print("=" * 60)
    
    entities = ["glucose", "fructose", "arabidopsis"]
    ontology_iri = "http://purl.obolibrary.org/obo/chebi.owl"
    
    print(f"Entities to map: {entities}")
    print(f"Ontology IRI: {ontology_iri}")
    
    # Mock the text2term call since we don't have actual ontologies
    mock_results = pd.DataFrame({
        'Source Term': ['glucose', 'fructose'],
        'Mapped Term Label': ['D-glucose', 'D-fructose'],
        'Mapped Term IRI': ['http://purl.obolibrary.org/obo/CHEBI_4167', 'http://purl.obolibrary.org/obo/CHEBI_15824'],
        'Mapping Score': [0.95, 0.92],
        'Term Type': ['class', 'class']
    })
    
    with patch('src.ontology_mapping.entity_mapper.text2term.map_terms', return_value=mock_results):
        try:
            results = map_entities_to_ontology(
                entities=entities,
                target_ontology=ontology_iri,  # String IRI (traditional approach)
                mapping_method='tfidf',
                min_score=0.8
            )
            
            print("✅ Mapping successful!")
            print(f"Results shape: {results.shape}")
            print("Sample results:")
            print(results.to_string(index=False))
            
        except Exception as e:
            print(f"❌ Error: {e}")


def demo_owlready2_integration():
    """Demonstrate the new Owlready2 integration approach."""
    print("\n" + "=" * 60)
    print("2. NEW OWLREADY2 INTEGRATION APPROACH")
    print("=" * 60)
    
    entities = ["glucose", "fructose", "arabidopsis"]
    
    print(f"Entities to map: {entities}")
    print(f"Owlready2 available: {OWLREADY2_AVAILABLE}")
    
    if not OWLREADY2_AVAILABLE:
        print("⚠️  Owlready2 not installed. Simulating with mock objects...")
        
        # Create a mock ontology object that behaves like owlready2.Ontology
        mock_ontology = Mock()
        mock_ontology.base_iri = "http://purl.obolibrary.org/obo/chebi.owl"
        
        # Mock the Owlready2 detection and IRI extraction
        with patch('src.ontology_mapping.entity_mapper._is_owlready2_ontology', return_value=True), \
             patch('src.ontology_mapping.entity_mapper._extract_iri_from_owlready2_ontology', 
                   return_value="http://purl.obolibrary.org/obo/chebi.owl"):
            
            print(f"Mock ontology IRI: {mock_ontology.base_iri}")
            
            mock_results = pd.DataFrame({
                'Source Term': ['glucose', 'fructose'],
                'Mapped Term Label': ['D-glucose', 'D-fructose'],
                'Mapped Term IRI': ['http://purl.obolibrary.org/obo/CHEBI_4167', 'http://purl.obolibrary.org/obo/CHEBI_15824'],
                'Mapping Score': [0.95, 0.92],
                'Term Type': ['class', 'class']
            })
            
            with patch('src.ontology_mapping.entity_mapper.text2term.map_terms', return_value=mock_results):
                try:
                    results = map_entities_to_ontology(
                        entities=entities,
                        target_ontology=mock_ontology,  # Owlready2 object (new approach)
                        mapping_method='tfidf',
                        min_score=0.8
                    )
                    
                    print("✅ Mapping successful with Owlready2 object!")
                    print(f"Results shape: {results.shape}")
                    print("Sample results:")
                    print(results.to_string(index=False))
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
    
    else:
        print("✅ Owlready2 is available! You can use real ontology objects.")
        print("Example code:")
        print("""
import owlready2
onto = owlready2.get_ontology("http://purl.obolibrary.org/obo/chebi.owl").load()
results = map_entities_to_ontology(
    entities=entities,
    target_ontology=onto,  # Direct ontology object!
    mapping_method='tfidf',
    min_score=0.8
)
        """)


def demo_error_handling():
    """Demonstrate error handling for invalid objects."""
    print("\n" + "=" * 60)
    print("3. ERROR HANDLING DEMONSTRATION")
    print("=" * 60)
    
    entities = ["glucose"]
    
    # Test various invalid inputs
    test_cases = [
        (None, "None value"),
        (123, "Integer"),
        ([], "Empty list"),
        ({}, "Empty dictionary"),
        ("invalid_url", "Invalid URL"),
        (object(), "Generic object")
    ]
    
    for invalid_input, description in test_cases:
        print(f"\nTesting {description}: {invalid_input}")
        try:
            result = _validate_target_ontology(invalid_input)
            print(f"  ✅ Accepted: {result}")
        except Exception as e:
            print(f"  ❌ Rejected: {type(e).__name__}: {e}")


def demo_iri_extraction():
    """Demonstrate IRI extraction from mock Owlready2 objects."""
    print("\n" + "=" * 60)
    print("4. IRI EXTRACTION DEMONSTRATION")
    print("=" * 60)
    
    # Test cases for IRI extraction
    test_cases = [
        ("http://example.org/ontology.owl", "Standard IRI"),
        ("http://example.org/ontology.owl/", "IRI with trailing slash"),
        ("https://purl.obolibrary.org/obo/chebi.owl", "HTTPS IRI"),
        ("file:///path/to/local/ontology.owl", "File IRI")
    ]
    
    for test_iri, description in test_cases:
        print(f"\nTesting {description}: {test_iri}")
        
        # Create mock ontology
        mock_ontology = Mock()
        mock_ontology.base_iri = test_iri
        
        with patch('src.ontology_mapping.entity_mapper._is_owlready2_ontology', return_value=True):
            try:
                extracted_iri = _extract_iri_from_owlready2_ontology(mock_ontology)
                print(f"  ✅ Extracted IRI: {extracted_iri}")
                
                # Show that trailing slashes are removed
                if test_iri.endswith('/') and not extracted_iri.endswith('/'):
                    print("  📝 Note: Trailing slash was automatically removed")
                    
            except Exception as e:
                print(f"  ❌ Error: {type(e).__name__}: {e}")


def main():
    """Run all demonstrations."""
    print("\n🔬 OWLREADY2 INTEGRATION DEMONSTRATION")
    print("This demo shows how the entity_mapper now supports both:")
    print("  • Traditional string IRIs (backward compatible)")
    print("  • Owlready2 ontology objects (new integration)")
    
    try:
        demo_traditional_approach()
        demo_owlready2_integration()
        demo_error_handling()
        demo_iri_extraction()
        
        print("\n" + "=" * 60)
        print("🎉 DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print("\nKey benefits of the new Owlready2 integration:")
        print("✅ Better integration for existing Owlready2 workflows")
        print("✅ Automatic IRI extraction from loaded ontologies")
        print("✅ Comprehensive error handling and validation")
        print("✅ Full backward compatibility with string IRIs")
        print("✅ Consistent API with enhanced functionality")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())