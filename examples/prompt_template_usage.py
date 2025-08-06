#!/usr/bin/env python3
"""
Example usage of zero-shot prompt templates for plant metabolomics NER.

This script demonstrates how to use the comprehensive prompt templates
with the existing extract_entities() function for different use cases
and research domains.

Usage:
    python examples/prompt_template_usage.py
"""

import sys
import os

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_extraction.prompt_templates import (
    get_basic_zero_shot_template,
    get_detailed_zero_shot_template,
    get_precision_focused_template,
    get_recall_focused_template,
    get_scientific_literature_template,
    get_domain_specific_template,
    get_template_for_use_case,
    get_recommended_template,
    customize_template,
    list_available_templates
)
from llm_extraction.entity_schemas import get_plant_metabolomics_schema, get_basic_schema
from llm_extraction.ner import _format_prompt


def demonstrate_basic_templates():
    """Demonstrate basic template functionality."""
    print("=== BASIC TEMPLATE FUNCTIONALITY ===\n")
    
    # Sample scientific text
    text = """
    LC-MS analysis revealed increased levels of quercetin and kaempferol in 
    Arabidopsis thaliana leaves under drought stress conditions. These flavonoid 
    compounds showed enhanced expression of CHS and F3H genes in the phenylpropanoid 
    biosynthesis pathway.
    """
    
    # Basic entity schema
    schema = {
        "METABOLITE": "Primary and secondary metabolites",
        "ANALYTICAL_METHOD": "Analytical techniques and instruments", 
        "SPECIES": "Plant and organism species",
        "PLANT_PART": "Plant anatomical structures",
        "STRESS_CONDITION": "Environmental stress conditions",
        "GENE": "Gene names and genetic elements",
        "PATHWAY": "Biochemical and metabolic pathways"
    }
    
    # Available templates
    print("Available templates:")
    for template_name in list_available_templates():
        print(f"  - {template_name}")
    print()
    
    # Basic zero-shot template
    print("1. BASIC ZERO-SHOT TEMPLATE:")
    basic_template = get_basic_zero_shot_template()
    basic_prompt = _format_prompt(basic_template, text.strip(), schema, None)
    print(f"Template length: {len(basic_template)} characters")
    print(f"Formatted prompt length: {len(basic_prompt)} characters")
    print(f"First 200 characters: {basic_prompt[:200]}...")
    print()
    
    # Detailed zero-shot template
    print("2. DETAILED ZERO-SHOT TEMPLATE:")
    detailed_template = get_detailed_zero_shot_template()
    detailed_prompt = _format_prompt(detailed_template, text.strip(), schema, None)
    print(f"Template length: {len(detailed_template)} characters")
    print(f"Formatted prompt length: {len(detailed_prompt)} characters")
    print(f"First 200 characters: {detailed_prompt[:200]}...")
    print()


def demonstrate_domain_specific_templates():
    """Demonstrate domain-specific template functionality."""
    print("=== DOMAIN-SPECIFIC TEMPLATES ===\n")
    
    text = "Anthocyanins and proanthocyanidins were analyzed using HPLC-MS in grape berry samples."
    schema = get_basic_schema()
    
    domains = ["metabolomics", "genetics", "plant_biology"]
    
    for domain in domains:
        print(f"{domain.upper()} DOMAIN TEMPLATE:")
        try:
            domain_template = get_domain_specific_template(domain)
            domain_prompt = _format_prompt(domain_template, text, schema, None)
            print(f"Template length: {len(domain_template)} characters")
            print(f"Domain-specific keywords: {domain} appears {domain_template.lower().count(domain)} times")
            print(f"First 150 characters: {domain_prompt[:150]}...")
        except Exception as e:
            print(f"Error with {domain} template: {e}")
        print()


def demonstrate_precision_recall_templates():
    """Demonstrate precision vs recall focused templates."""
    print("=== PRECISION vs RECALL TEMPLATES ===\n")
    
    text = """Catechin, epicatechin, and procyanidin B1 were identified as major 
    polyphenolic compounds. The antioxidant activity correlated with phenolic content."""
    
    schema = {
        "PHENOLIC_COMPOUND": "Phenolic compounds and derivatives",
        "BIOLOGICAL_ACTIVITY": "Biological activities and functions"
    }
    
    # Precision-focused template
    print("PRECISION-FOCUSED TEMPLATE (minimize false positives):")
    precision_template = get_precision_focused_template()
    precision_prompt = _format_prompt(precision_template, text, schema, None)
    print(f"Contains 'high confidence': {'high confidence' in precision_template.lower()}")
    print(f"Contains 'precise': {'precise' in precision_template.lower()}")
    print(f"First 200 characters: {precision_prompt[:200]}...")
    print()
    
    # Recall-focused template
    print("RECALL-FOCUSED TEMPLATE (capture more entities):")
    recall_template = get_recall_focused_template()
    recall_prompt = _format_prompt(recall_template, text, schema, None)
    print(f"Contains 'comprehensive': {'comprehensive' in recall_template.lower()}")
    print(f"Contains 'all': {'all' in recall_template.lower()}")
    print(f"First 200 characters: {recall_prompt[:200]}...")
    print()


def demonstrate_use_case_selection():
    """Demonstrate automatic template selection based on use case."""
    print("=== USE CASE-BASED TEMPLATE SELECTION ===\n")
    
    text = "Sample research text for analysis."
    schema = get_basic_schema()
    
    use_cases = [
        ("research_paper", None, "balanced"),
        ("quick_analysis", None, "balanced"), 
        ("comprehensive", None, "balanced"),
        ("analysis", "metabolomics", "balanced"),
        ("analysis", None, "precision"),
        ("analysis", None, "recall")
    ]
    
    for use_case, domain, priority in use_cases:
        print(f"USE CASE: {use_case}")
        if domain:
            print(f"DOMAIN: {domain}")
        print(f"PRIORITY: {priority}")
        
        try:
            template = get_template_for_use_case(use_case, domain, priority)
            prompt = _format_prompt(template, text, schema, None)
            print(f"Selected template length: {len(template)} characters")
            print(f"Template type indicators: ", end="")
            
            # Check for template type indicators
            template_lower = template.lower()
            indicators = []
            if "basic" in template_lower:
                indicators.append("basic")
            if "detailed" in template_lower or "comprehensive" in template_lower:
                indicators.append("detailed/comprehensive")
            if "precision" in template_lower:
                indicators.append("precision")
            if "recall" in template_lower:
                indicators.append("recall")
            if "scientific" in template_lower or "literature" in template_lower:
                indicators.append("scientific")
            if "metabol" in template_lower:
                indicators.append("metabolomics")
            
            print(", ".join(indicators) if indicators else "none detected")
            
        except Exception as e:
            print(f"Error: {e}")
        print()


def demonstrate_template_customization():
    """Demonstrate template customization features."""
    print("=== TEMPLATE CUSTOMIZATION ===\n")
    
    # Base template
    base_template = get_basic_zero_shot_template()
    text = "Quercetin and kaempferol levels increased in stressed plants."
    schema = {"METABOLITE": "Chemical metabolites"}
    
    print("1. ORIGINAL TEMPLATE:")
    original_prompt = _format_prompt(base_template, text, schema, None)
    print(f"Length: {len(original_prompt)} characters")
    print()
    
    # Customized with instructions
    print("2. CUSTOMIZED WITH INSTRUCTIONS:")
    custom_instructions = "Focus specifically on flavonoid compounds and their derivatives."
    customized_template = customize_template(base_template, custom_instructions=custom_instructions)
    custom_prompt = _format_prompt(customized_template, text, schema, None)
    print(f"Added instructions: {custom_instructions}")
    print(f"New length: {len(custom_prompt)} characters")
    print(f"Length increase: {len(custom_prompt) - len(original_prompt)} characters")
    print()
    
    # Customized with confidence threshold
    print("3. CUSTOMIZED WITH CONFIDENCE THRESHOLD:")
    threshold_template = customize_template(base_template, confidence_threshold=0.85)
    threshold_prompt = _format_prompt(threshold_template, text, schema, None)
    print(f"Confidence threshold: >= 0.85")
    print(f"Contains threshold instruction: {'confidence >= 0.85' in threshold_template}")
    print()
    
    # Customized with additional examples
    print("4. CUSTOMIZED WITH ADDITIONAL CONTEXT:")
    additional_examples = [
        "Consider metabolite-protein interactions",
        "Include biosynthetic pathway information"
    ]
    example_template = customize_template(base_template, additional_examples=additional_examples)
    example_prompt = _format_prompt(example_template, text, schema, None)
    print(f"Additional context items: {len(additional_examples)}")
    print(f"Contains additional context: {'ADDITIONAL CONTEXT' in example_template}")
    print()


def demonstrate_comprehensive_schema_integration():
    """Demonstrate integration with comprehensive plant metabolomics schema."""
    print("=== COMPREHENSIVE SCHEMA INTEGRATION ===\n")
    
    # Complex scientific text
    text = """
    HPLC-ESI-MS/MS analysis of Solanum lycopersicum fruit extracts revealed 127 metabolites
    including chlorogenic acid, rutin, and naringenin chalcone. Under heat stress conditions,
    the expression of PAL, C4H, and 4CL genes in the phenylpropanoid pathway increased 
    2.5-fold. These changes correlated with enhanced antioxidant capacity and improved
    stress tolerance in cv. Micro-Tom tomato plants.
    """
    
    # Comprehensive schema (117 entity types)
    comprehensive_schema = get_plant_metabolomics_schema()
    
    print(f"Text length: {len(text)} characters")
    print(f"Schema entity types: {len(comprehensive_schema)}")
    print()
    
    # Scientific literature template with comprehensive schema
    print("SCIENTIFIC LITERATURE TEMPLATE + COMPREHENSIVE SCHEMA:")
    scientific_template = get_scientific_literature_template()
    comprehensive_prompt = _format_prompt(scientific_template, text.strip(), comprehensive_schema, None)
    
    print(f"Template length: {len(scientific_template)} characters")
    print(f"Final prompt length: {len(comprehensive_prompt)} characters")
    print(f"Entity types in prompt: {len([line for line in comprehensive_prompt.split('\\n') if '- ' in line and ':' in line])}")
    
    # Show sample entity types included
    entity_lines = [line.strip() for line in comprehensive_prompt.split('\\n') if line.strip().startswith('- ')]
    print("\\nSample entity types included:")
    for i, line in enumerate(entity_lines[:10]):  # Show first 10
        print(f"  {line}")
    if len(entity_lines) > 10:
        print(f"  ... and {len(entity_lines) - 10} more")
    print()


def demonstrate_template_recommendations():
    """Demonstrate template recommendation system."""
    print("=== TEMPLATE RECOMMENDATION SYSTEM ===\n")
    
    scenarios = [
        (200, 5, None, "balanced", "Short text, few entities"),
        (1500, 25, "metabolomics", "balanced", "Medium text, metabolomics domain"),
        (3000, 50, None, "precision", "Long text, precision priority"),
        (800, 15, None, "recall", "Medium text, recall priority")
    ]
    
    for text_length, entity_count, domain, priority, description in scenarios:
        print(f"SCENARIO: {description}")
        print(f"  Text length: {text_length} characters")
        print(f"  Estimated entities: {entity_count}")
        print(f"  Domain: {domain or 'none'}")
        print(f"  Priority: {priority}")
        
        recommended = get_recommended_template(text_length, entity_count, domain, priority)
        
        # Identify template type
        template_type = "unknown"
        if recommended == get_basic_zero_shot_template():
            template_type = "basic"
        elif recommended == get_detailed_zero_shot_template():
            template_type = "detailed"
        elif recommended == get_scientific_literature_template():
            template_type = "scientific"
        elif recommended == get_precision_focused_template():
            template_type = "precision"
        elif recommended == get_recall_focused_template():
            template_type = "recall"
        elif domain and domain in recommended.lower():
            template_type = f"{domain} domain-specific"
        
        print(f"  RECOMMENDED: {template_type} template")
        print(f"  Template length: {len(recommended)} characters")
        print()


def main():
    """Run all demonstrations."""
    print("ZERO-SHOT PROMPT TEMPLATES FOR PLANT METABOLOMICS NER")
    print("=" * 60)
    print()
    
    try:
        demonstrate_basic_templates()
        print("\\n" + "=" * 60 + "\\n")
        
        demonstrate_domain_specific_templates()
        print("\\n" + "=" * 60 + "\\n")
        
        demonstrate_precision_recall_templates()
        print("\\n" + "=" * 60 + "\\n")
        
        demonstrate_use_case_selection()
        print("\\n" + "=" * 60 + "\\n")
        
        demonstrate_template_customization()
        print("\\n" + "=" * 60 + "\\n")
        
        demonstrate_comprehensive_schema_integration()
        print("\\n" + "=" * 60 + "\\n")
        
        demonstrate_template_recommendations()
        
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("\\nTo use these templates with the extract_entities() function:")
        print("1. Choose appropriate template using get_*_template() functions")
        print("2. Optionally customize with customize_template()")
        print("3. Pass template to extract_entities() as prompt_template parameter")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()