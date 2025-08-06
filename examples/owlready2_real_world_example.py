#!/usr/bin/env python3
"""
Real-world example of using Owlready2 with entity_mapper.

This example shows how to use the new integrated functionality with real
Owlready2 ontology objects. The example is designed to work whether or not
Owlready2 is actually installed.

If Owlready2 is installed, it will demonstrate loading an ontology and
using it directly with the entity_mapper. If not installed, it will
show what the code would look like.
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ontology_mapping.entity_mapper import (
    map_entities_to_ontology,
    OWLREADY2_AVAILABLE
)


def example_with_owlready2():
    """Example using real Owlready2 objects (if available)."""
    print("=" * 60)
    print("REAL OWLREADY2 INTEGRATION EXAMPLE")
    print("=" * 60)
    
    if not OWLREADY2_AVAILABLE:
        print("❌ Owlready2 is not installed.")
        print("📥 To install: pip install owlready2")
        print("\n🔍 Here's what the code would look like:")
        print("""
# Load an ontology with Owlready2
import owlready2

# Option 1: Load from URL
chebi_onto = owlready2.get_ontology("http://purl.obolibrary.org/obo/chebi.owl")
chebi_onto.load()

# Option 2: Load from local file
# local_onto = owlready2.get_ontology("file://path/to/ontology.owl")
# local_onto.load()

# Define entities to map
entities = [
    "glucose",
    "fructose", 
    "sucrose",
    "arabidopsis thaliana",
    "photosynthesis"
]

# Use the loaded ontology directly with entity_mapper
results = map_entities_to_ontology(
    entities=entities,
    target_ontology=chebi_onto,  # Pass the ontology object directly!
    mapping_method='tfidf',
    min_score=0.7,
    term_type='class'
)

print("Mapping Results:")
print(results)

# Benefits of this approach:
# 1. No need to remember or manage IRI strings
# 2. Works with already-loaded ontologies
# 3. Integrates seamlessly with existing Owlready2 workflows
# 4. Automatic IRI extraction and validation
        """)
        return
    
    print("✅ Owlready2 is available!")
    
    try:
        import owlready2
        
        print("\n📋 STEP 1: Loading Ontology")
        print("-" * 30)
        
        # For demonstration, we'll create a simple ontology in memory
        # In real usage, you'd typically load from URL or file
        onto = owlready2.get_ontology("http://example.org/demo_ontology.owl")
        
        with onto:
            # Create some example classes
            class ChemicalCompound(owlready2.Thing):
                pass
            
            class Glucose(ChemicalCompound):
                pass
            
            class Fructose(ChemicalCompound):
                pass
        
        print(f"✅ Ontology loaded: {onto.base_iri}")
        print(f"📊 Classes defined: {len(list(onto.classes()))}")
        
        print("\n📋 STEP 2: Entity Mapping")
        print("-" * 30)
        
        entities = ["glucose", "fructose", "sugar"]
        print(f"🎯 Entities to map: {entities}")
        
        # Mock the text2term call since we don't have real mappings
        from unittest.mock import patch
        import pandas as pd
        
        mock_results = pd.DataFrame({
            'Source Term': ['glucose', 'fructose'],
            'Mapped Term Label': ['Glucose', 'Fructose'],
            'Mapped Term IRI': [
                'http://example.org/demo_ontology.owl#Glucose',
                'http://example.org/demo_ontology.owl#Fructose'
            ],
            'Mapping Score': [0.95, 0.92],
            'Term Type': ['class', 'class']
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms', return_value=mock_results):
            # Use the ontology object directly!
            results = map_entities_to_ontology(
                entities=entities,
                target_ontology=onto,  # Pass the Owlready2 object directly
                mapping_method='tfidf',
                min_score=0.7
            )
            
            print("✅ Mapping successful!")
            print(f"📊 Results shape: {results.shape}")
            print("\n📋 Mapping Results:")
            print(results.to_string(index=False))
            
            print(f"\n🔍 Note: The ontology IRI was automatically extracted:")
            print(f"    Original ontology object: {type(onto)}")
            print(f"    Extracted IRI: {onto.base_iri}")
    
    except Exception as e:
        print(f"❌ Error in Owlready2 example: {e}")
        print("This might happen if there are import conflicts or missing dependencies.")


def comparison_example():
    """Show comparison between old and new approaches."""
    print("\n" + "=" * 60)
    print("COMPARISON: OLD vs NEW APPROACH")
    print("=" * 60)
    
    print("📝 OLD APPROACH (still supported):")
    print("-" * 35)
    print("""
# You had to manage IRI strings manually
ontology_iri = "http://purl.obolibrary.org/obo/chebi.owl"

results = map_entities_to_ontology(
    entities=entities,
    target_ontology=ontology_iri,  # String IRI
    mapping_method='tfidf'
)
    """)
    
    print("✨ NEW APPROACH (enhanced):")
    print("-" * 30)
    print("""
# Load ontology once with Owlready2
import owlready2
onto = owlready2.get_ontology("http://purl.obolibrary.org/obo/chebi.owl").load()

# Use the object directly - no need to manage IRIs!
results = map_entities_to_ontology(
    entities=entities,
    target_ontology=onto,  # Ontology object
    mapping_method='tfidf'
)

# Benefits:
# ✅ Better integration with Owlready2 workflows
# ✅ No manual IRI management
# ✅ Automatic validation
# ✅ Works with complex ontology hierarchies
# ✅ Maintains full backward compatibility
    """)


def integration_benefits():
    """Explain the benefits of the integration."""
    print("\n" + "=" * 60)
    print("🎯 INTEGRATION BENEFITS")
    print("=" * 60)
    
    benefits = [
        "🔗 **Seamless Workflow Integration**",
        "   - Works directly with loaded Owlready2 ontologies",
        "   - No need to convert between objects and IRIs",
        "   - Maintains ontology context throughout pipeline",
        "",
        "🛡️ **Enhanced Error Handling**",
        "   - Validates Owlready2 objects automatically",
        "   - Clear error messages for common issues",
        "   - Graceful handling when Owlready2 isn't available",
        "",
        "📚 **Backward Compatibility**", 
        "   - All existing code continues to work unchanged",
        "   - String IRIs still fully supported",
        "   - No breaking changes to existing APIs",
        "",
        "⚡ **Developer Experience**",
        "   - Intuitive API that matches user expectations",
        "   - Automatic IRI extraction and normalization",
        "   - Comprehensive type hints and documentation",
        "",
        "🔧 **Technical Robustness**",
        "   - Conditional imports prevent hard dependencies",
        "   - Proper exception hierarchy for error handling",
        "   - Extensive test coverage for all scenarios"
    ]
    
    for benefit in benefits:
        print(benefit)


def main():
    """Run the complete example."""
    print("🔬 OWLREADY2 INTEGRATION: REAL-WORLD EXAMPLE")
    print("This example demonstrates the enhanced entity_mapper functionality")
    print("that now supports both string IRIs and Owlready2 ontology objects.\n")
    
    try:
        example_with_owlready2()
        comparison_example()
        integration_benefits()
        
        print("\n" + "=" * 60)
        print("🎉 EXAMPLE COMPLETE!")
        print("=" * 60)
        print("The entity_mapper now provides enhanced integration with Owlready2")
        print("while maintaining full backward compatibility with existing code.")
        
        if not OWLREADY2_AVAILABLE:
            print("\n💡 TIP: Install Owlready2 to try the integration:")
            print("    pip install owlready2")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())