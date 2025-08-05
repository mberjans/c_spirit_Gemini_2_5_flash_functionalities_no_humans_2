#!/usr/bin/env python3
"""
Demo script showing sophisticated relationship extraction templates in action.

This script demonstrates the new relationship extraction templates with:
1. Hierarchical differentiation between broad and specific relationships
2. Contextual understanding with conditional relationships
3. Few-shot learning with domain-specific examples
4. Multi-type relationship extraction
5. Template management and selection
"""

from src.llm_extraction.prompt_templates import (
    get_relationship_template,
    list_available_relationship_templates,
    generate_relationship_examples,
    format_relationship_schema_for_template,
    select_optimal_relationship_template,
    validate_relationship_template_inputs
)
from src.llm_extraction.relationship_schemas import PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA


def demo_template_selection():
    """Demonstrate intelligent template selection based on text characteristics."""
    print("=== DEMO: Intelligent Template Selection ===")
    
    test_cases = [
        {
            "description": "Simple metabolomics text",
            "characteristics": {
                "complexity": "low",
                "domain": "metabolomics",
                "context_dependency": False,
                "relationship_density": "low"
            }
        },
        {
            "description": "Complex experimental study with conditions",
            "characteristics": {
                "complexity": "high", 
                "domain": "metabolomics",
                "context_dependency": True,
                "temporal_markers": True,
                "conditional_statements": True,
                "relationship_density": "high"
            }
        },
        {
            "description": "High-density biological pathway text",
            "characteristics": {
                "complexity": "high",
                "domain": "general",
                "context_dependency": False,
                "relationship_density": "high"
            }
        }
    ]
    
    for case in test_cases:
        recommended = select_optimal_relationship_template(case["characteristics"])
        print(f"\nText type: {case['description']}")
        print(f"Characteristics: {case['characteristics']}")
        print(f"Recommended template: {recommended}")


def demo_hierarchical_template():
    """Demonstrate hierarchical relationship differentiation."""
    print("\n=== DEMO: Hierarchical Relationship Differentiation ===")
    
    # Get the hierarchical template
    template = get_relationship_template("hierarchical")
    
    # Sample text with relationships at different specificity levels
    sample_text = "CYP75A upregulates anthocyanin biosynthesis genes in response to cold stress, while resveratrol accumulates in grape skins."
    
    sample_entities = [
        {"text": "CYP75A", "label": "ENZYME"},
        {"text": "anthocyanin biosynthesis", "label": "PATHWAY"},
        {"text": "cold stress", "label": "STRESS_CONDITION"},
        {"text": "resveratrol", "label": "PHENOLIC_COMPOUND"},
        {"text": "grape skins", "label": "PLANT_TISSUE"}
    ]
    
    # Format schema for template
    schema_formatted = format_relationship_schema_for_template(PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA)
    
    print("Sample text:", sample_text)
    print("\nEntities:", [f"{e['text']} ({e['label']})" for e in sample_entities])
    print("\nHierarchical template emphasizes:")
    print("- Choosing 'upregulates' over generic 'affects'")
    print("- Using 'accumulates_in' instead of broad 'found_in'")
    print("- Providing specificity justification")
    
    # Show template structure
    print(f"\nTemplate includes {template.count('SPECIFIC')} references to specificity")
    print(f"Template includes {template.count('BROAD')} references to broad relationships")


def demo_contextual_template():
    """Demonstrate contextual relationship understanding.""" 
    print("\n=== DEMO: Contextual Relationship Understanding ===")
    
    # Get contextual template
    template = get_relationship_template("contextual")
    
    # Sample text with contextual conditions
    sample_text = "Under drought stress, proline levels increased 3-fold in root tissues after 24 hours, while being regulated by P5CS enzyme only during water deficit conditions."
    
    sample_entities = [
        {"text": "drought stress", "label": "STRESS_CONDITION"},
        {"text": "proline", "label": "AMINO_ACID"},
        {"text": "root tissues", "label": "PLANT_TISSUE"},
        {"text": "P5CS", "label": "ENZYME"},
        {"text": "water deficit", "label": "STRESS_CONDITION"}
    ]
    
    print("Sample text:", sample_text)
    print("\nEntities:", [f"{e['text']} ({e['label']})" for e in sample_entities])
    print("\nContextual template captures:")
    print("- Environmental conditions: 'under drought stress'")
    print("- Quantitative context: '3-fold increase'")
    print("- Temporal context: 'after 24 hours'")
    print("- Conditional relationships: 'only during water deficit'")
    print("- Spatial context: 'in root tissues'")
    
    print(f"\nTemplate includes {template.count('context')} references to context")
    print(f"Template includes {template.count('condition')} references to conditions")


def demo_few_shot_templates():
    """Demonstrate few-shot templates with domain examples."""
    print("\n=== DEMO: Few-Shot Templates with Domain Examples ===")
    
    # Show available few-shot templates
    templates = list_available_relationship_templates()
    few_shot_templates = [t for t in templates if t["type"] == "few-shot"]
    
    print("Available few-shot templates:")
    for template in few_shot_templates:
        print(f"- {template['name']}: {template['description']}")
    
    # Get few-shot metabolomics template
    metabolomics_template = get_relationship_template("few_shot_metabolomics")
    
    print(f"\nFew-shot metabolomics template includes:")
    print(f"- {metabolomics_template.count('Example')} learning examples")
    print(f"- Domain-specific relationships like 'synthesized_by', 'accumulates_in'")
    print(f"- Evidence-based confidence scoring")
    
    # Generate examples for specific relationship type
    examples = generate_relationship_examples("synthesized_by", ["METABOLITE", "ENZYME"], count=2)
    print(f"\nGenerated examples for 'synthesized_by' relationship:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['subject_entity']['text']} synthesized_by {example['object_entity']['text']}")
        print(f"   Evidence: {example['evidence']}")


def demo_multi_type_template():
    """Demonstrate multi-type comprehensive relationship extraction."""
    print("\n=== DEMO: Multi-Type Comprehensive Relationship Extraction ===")
    
    template = get_relationship_template("multi_type")
    
    # Sample complex text with multiple relationship types
    sample_text = "Chalcone synthase catalyzes the first step of flavonoid biosynthesis, converting coumaroyl-CoA to naringenin chalcone, which accumulates in flower petals under UV stress and is detected by HPLC analysis."
    
    print("Sample complex text:", sample_text)
    print("\nMulti-type template extracts:")
    print("1. FUNCTIONAL: chalcone_synthase 'catalyzes' flavonoid_biosynthesis")
    print("2. BIOSYNTHETIC: coumaroyl-CoA 'converted_to' naringenin_chalcone")
    print("3. LOCALIZATION: naringenin_chalcone 'accumulates_in' flower_petals")
    print("4. ENVIRONMENTAL: naringenin_chalcone 'increases_under' UV_stress")
    print("5. ANALYTICAL: naringenin_chalcone 'detected_by' HPLC")
    
    print(f"\nTemplate includes cross-validation and relationship summary")
    print(f"Template categorizes relationships into 6 types:")
    print("- biosynthetic, regulatory, localization, functional, environmental, analytical")


def demo_template_validation():
    """Demonstrate template input validation."""
    print("\n=== DEMO: Template Input Validation ===")
    
    template = get_relationship_template("basic")
    
    # Test cases
    test_cases = [
        {
            "name": "Valid inputs",
            "template": template,
            "text": "Anthocyanins are synthesized by CHS enzyme in grape berries.",
            "entities": [{"text": "anthocyanins", "label": "METABOLITE"}, {"text": "CHS", "label": "ENZYME"}],
            "schema": {"synthesized_by": {"description": "synthesis relationship"}}
        },
        {
            "name": "Too short text",
            "template": template,
            "text": "Short",
            "entities": [{"text": "metabolite", "label": "METABOLITE"}],
            "schema": {}
        },
        {
            "name": "No entities", 
            "template": template,
            "text": "This is a longer text about plant metabolomics research.",
            "entities": [],
            "schema": {}
        }
    ]
    
    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        validation = validate_relationship_template_inputs(
            case["template"], case["text"], case["entities"], case["schema"]
        )
        print(f"Valid: {validation['valid']}")
        if validation['errors']:
            print(f"Errors: {validation['errors']}")
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")
        if validation['recommendations']:
            print(f"Recommendations: {validation['recommendations']}")


def main():
    """Run all relationship template demos."""
    print("SOPHISTICATED RELATIONSHIP EXTRACTION TEMPLATES DEMO")
    print("=" * 60)
    
    demo_template_selection()
    demo_hierarchical_template()
    demo_contextual_template()
    demo_few_shot_templates()
    demo_multi_type_template()
    demo_template_validation()
    
    print("\n" + "=" * 60)
    print("Demo completed! The new relationship extraction templates provide:")
    print("✓ Hierarchical differentiation (specific vs broad relationships)")
    print("✓ Contextual understanding (conditional, temporal, spatial)")
    print("✓ Few-shot learning with domain-specific examples")
    print("✓ Multi-type comprehensive extraction")
    print("✓ Intelligent template selection")
    print("✓ Input validation and recommendations")


if __name__ == "__main__":
    main()