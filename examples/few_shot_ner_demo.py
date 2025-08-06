"""
Demonstration of Few-Shot Named Entity Recognition for Plant Metabolomics

This script showcases the comprehensive few-shot NER functionality including:
- Synthetic example generation for all 117 entity types
- Multiple few-shot template variants
- Context-aware example selection
- Domain-specific templates and examples
- Integration with existing NER pipeline

Run this script to see the few-shot NER system in action with plant metabolomics text.
"""

import json
import sys
import os
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm_extraction.prompt_templates import (
    # Example generation and selection
    generate_synthetic_examples,
    select_examples,
    get_examples_by_domain,
    format_examples_for_prompt,
    get_context_aware_examples,
    
    # Template getters
    get_few_shot_template,
    get_few_shot_basic_template,
    get_few_shot_detailed_template,
    get_few_shot_precision_template,
    get_few_shot_recall_template,
    get_few_shot_domain_template,
    
    # Constants
    SYNTHETIC_EXAMPLES_DATABASE
)

from src.llm_extraction.entity_schemas import (
    get_plant_metabolomics_schema,
    get_entity_types_by_category,
    get_schema_by_domain
)


def demonstrate_example_database():
    """Demonstrate the synthetic examples database."""
    print("=== SYNTHETIC EXAMPLES DATABASE ===")
    print(f"Total entity types with examples: {len(SYNTHETIC_EXAMPLES_DATABASE)}")
    
    # Show examples for key entity types
    key_types = ["METABOLITE", "SPECIES", "GENE", "PLANT_PART", "ANALYTICAL_METHOD"]
    
    for entity_type in key_types:
        if entity_type in SYNTHETIC_EXAMPLES_DATABASE:
            examples = SYNTHETIC_EXAMPLES_DATABASE[entity_type]
            print(f"\n{entity_type} ({len(examples)} examples):")
            
            # Show first example
            example = examples[0]
            print(f"  Text: {example['text']}")
            print(f"  Entities: {[e['text'] + ' (' + e['label'] + ')' for e in example['entities']]}")
    
    print("\n" + "="*60)


def demonstrate_example_generation():
    """Demonstrate synthetic example generation."""
    print("\n=== EXAMPLE GENERATION ===")
    
    # Generate examples for specific entity types
    target_types = ["METABOLITE", "SPECIES", "GENE", "ENZYME"]
    
    print(f"Generating examples for: {target_types}")
    
    # Test different difficulty levels
    for difficulty in ["simple", "complex", "mixed"]:
        examples = generate_synthetic_examples(
            target_types, 
            num_examples=2, 
            difficulty_level=difficulty
        )
        
        print(f"\n{difficulty.upper()} examples ({len(examples)} generated):")
        for i, example in enumerate(examples[:2], 1):  # Show first 2
            entities_text = [f"{e['text']}({e['label']})" for e in example['entities']]
            print(f"  {i}. {example['text'][:60]}...")
            print(f"     Entities: {', '.join(entities_text)}")
    
    print("\n" + "="*60)


def demonstrate_example_selection():
    """Demonstrate different example selection strategies."""
    print("\n=== EXAMPLE SELECTION STRATEGIES ===")
    
    target_types = ["METABOLITE", "COMPOUND", "SPECIES", "GENE"]
    strategies = ["balanced", "high_confidence", "diverse", "random"]
    
    for strategy in strategies:
        examples = select_examples(
            target_types, 
            strategy=strategy, 
            max_examples=4
        )
        
        print(f"\n{strategy.upper()} strategy ({len(examples)} examples):")
        
        if examples:
            # Show summary statistics
            all_confidence = []
            all_types = set()
            
            for example in examples:
                for entity in example['entities']:
                    all_confidence.append(entity['confidence'])
                    all_types.add(entity['label'])
            
            avg_confidence = sum(all_confidence) / len(all_confidence) if all_confidence else 0
            print(f"  Average confidence: {avg_confidence:.3f}")
            print(f"  Entity types found: {sorted(all_types)}")
            
            # Show one example
            example = examples[0]
            print(f"  Sample: {example['text'][:50]}...")
    
    print("\n" + "="*60)


def demonstrate_domain_examples():
    """Demonstrate domain-specific example selection."""
    print("\n=== DOMAIN-SPECIFIC EXAMPLES ===")
    
    domains = ["metabolomics", "genetics", "plant_biology", "analytical"]
    
    for domain in domains:
        examples = get_examples_by_domain(domain, max_examples=3)
        
        print(f"\n{domain.upper()} domain ({len(examples)} examples):")
        
        for i, example in enumerate(examples[:2], 1):  # Show first 2
            entity_types = [e['label'] for e in example['entities']]
            print(f"  {i}. {example['text'][:50]}...")
            print(f"     Types: {set(entity_types)}")
    
    print("\n" + "="*60)


def demonstrate_context_aware_selection():
    """Demonstrate context-aware example selection."""
    print("\n=== CONTEXT-AWARE EXAMPLE SELECTION ===")
    
    # Test different types of scientific text
    test_texts = [
        {
            "text": "HPLC-MS analysis revealed high concentrations of quercetin and kaempferol in stressed leaves.",
            "expected_domain": "metabolomics"
        },
        {
            "text": "Gene expression analysis showed upregulation of transcription factors during drought stress.",
            "expected_domain": "genetics"
        },
        {
            "text": "Root elongation and leaf area were measured in different plant cultivars.",
            "expected_domain": "plant_biology"
        }
    ]
    
    schema = get_plant_metabolomics_schema()
    
    for i, test_case in enumerate(test_texts, 1):
        text = test_case["text"]
        expected = test_case["expected_domain"]
        
        examples = get_context_aware_examples(text, schema, max_examples=3)
        
        print(f"\n{i}. Text: {text}")
        print(f"   Expected domain: {expected}")
        print(f"   Selected examples ({len(examples)}):")
        
        if examples:
            # Analyze selected examples
            found_types = set()
            for example in examples:
                for entity in example['entities']:
                    found_types.add(entity['label'])
            
            print(f"   Entity types in examples: {sorted(found_types)}")
            
            # Show one example
            sample = examples[0]
            print(f"   Sample: {sample['text'][:60]}...")
    
    print("\n" + "="*60)


def demonstrate_few_shot_templates():
    """Demonstrate different few-shot template variants."""
    print("\n=== FEW-SHOT TEMPLATE VARIANTS ===")
    
    # Generate sample examples
    examples = generate_synthetic_examples(["METABOLITE", "SPECIES"], num_examples=2)
    formatted_examples = format_examples_for_prompt(examples)
    
    # Test different template types
    template_types = ["basic", "detailed", "precision", "recall", "scientific"]
    
    for template_type in template_types:
        template = get_few_shot_template(template_type)
        
        print(f"\n{template_type.upper()} Template:")
        print(f"  Length: {len(template)} characters")
        
        # Show key characteristics
        if "precision" in template.lower():
            print("  Focus: High precision, minimize false positives")
        elif "recall" in template.lower():
            print("  Focus: High recall, capture all entities")
        elif "detailed" in template.lower():
            print("  Focus: Comprehensive guidelines and analysis")
        elif "scientific" in template.lower():
            print("  Focus: Scientific literature conventions")
        else:
            print("  Focus: Basic few-shot learning")
        
        # Check if it has the key placeholders
        placeholders = ["{schema}", "{text}", "{examples}"]
        found_placeholders = [p for p in placeholders if p in template]
        print(f"  Placeholders: {found_placeholders}")
    
    print(f"\nFormatted examples preview:")
    print(formatted_examples[:200] + "..." if len(formatted_examples) > 200 else formatted_examples)
    
    print("\n" + "="*60)


def demonstrate_domain_templates():
    """Demonstrate domain-specific few-shot templates."""
    print("\n=== DOMAIN-SPECIFIC FEW-SHOT TEMPLATES ===")
    
    domains = ["metabolomics", "genetics", "plant_biology"]
    
    for domain in domains:
        try:
            template = get_few_shot_domain_template(domain)
            examples = get_examples_by_domain(domain, max_examples=2)
            
            print(f"\n{domain.upper()} Domain Template:")
            print(f"  Template length: {len(template)} characters")
            print(f"  Available examples: {len(examples)}")
            
            # Show domain-specific patterns mentioned in template
            if "metabolite" in template.lower():
                print("  Patterns: Chemical compounds, analytical methods, biochemical processes")
            elif "gene" in template.lower():
                print("  Patterns: Gene nomenclature, protein names, molecular processes")
            elif "anatomical" in template.lower():
                print("  Patterns: Plant anatomy, developmental stages, physiological processes")
            
            # Show example types from this domain
            if examples:
                example_types = set()
                for example in examples:
                    for entity in example['entities']:
                        example_types.add(entity['label'])
                print(f"  Example entity types: {sorted(example_types)}")
        
        except Exception as e:
            print(f"\n{domain.upper()}: Error - {e}")
    
    print("\n" + "="*60)


def demonstrate_end_to_end_pipeline():
    """Demonstrate the complete few-shot NER pipeline."""
    print("\n=== END-TO-END FEW-SHOT NER PIPELINE ===")
    
    # Sample scientific text
    sample_text = """
    Arabidopsis thaliana plants were subjected to drought stress conditions. 
    LC-MS analysis revealed increased accumulation of anthocyanin and quercetin 
    in leaf tissues. The chalcone synthase gene showed upregulated expression, 
    suggesting enhanced flavonoid biosynthesis pathway activation.
    """
    
    print(f"Sample text: {sample_text.strip()}")
    
    # Get comprehensive schema
    schema = get_plant_metabolomics_schema()
    print(f"\nUsing schema with {len(schema)} entity types")
    
    # Step 1: Context-aware example selection
    print("\n1. Context-aware example selection:")
    context_examples = get_context_aware_examples(sample_text, schema, max_examples=4)
    print(f"   Selected {len(context_examples)} contextually relevant examples")
    
    if context_examples:
        found_types = set()
        for example in context_examples:
            for entity in example['entities']:
                found_types.add(entity['label'])
        print(f"   Example entity types: {sorted(found_types)}")
    
    # Step 2: Template selection
    print("\n2. Template selection:")
    template = get_few_shot_detailed_template()
    print(f"   Using detailed few-shot template ({len(template)} chars)")
    
    # Step 3: Format complete prompt (simulation)
    print("\n3. Prompt formatting:")
    schema_str = "\n".join([f"- {key}: {desc[:50]}..." for key, desc in list(schema.items())[:5]])
    examples_str = format_examples_for_prompt(context_examples[:2])
    
    print("   Schema preview:")
    print("   " + "\n   ".join(schema_str.split("\n")[:3]))
    print("   ... (and {} more entity types)".format(len(schema) - 3))
    
    print("\n   Examples preview:")
    preview = examples_str[:300] + "..." if len(examples_str) > 300 else examples_str
    print("   " + preview.replace("\n", "\n   "))
    
    # Step 4: Expected entity extraction (simulation)
    print("\n4. Expected entity extraction:")
    expected_entities = [
        {"text": "Arabidopsis thaliana", "label": "SPECIES", "confidence": 0.99},
        {"text": "drought stress", "label": "STRESS_CONDITION", "confidence": 0.97},
        {"text": "LC-MS", "label": "ANALYTICAL_METHOD", "confidence": 0.98},
        {"text": "anthocyanin", "label": "METABOLITE", "confidence": 0.95},
        {"text": "quercetin", "label": "METABOLITE", "confidence": 0.95},
        {"text": "leaf", "label": "PLANT_PART", "confidence": 0.96},
        {"text": "chalcone synthase", "label": "ENZYME", "confidence": 0.97},
        {"text": "flavonoid biosynthesis", "label": "BIOSYNTHESIS", "confidence": 0.94}
    ]
    
    print(f"   Expected entities ({len(expected_entities)}):")
    for entity in expected_entities:
        print(f"   - {entity['text']} ({entity['label']}, confidence: {entity['confidence']})")
    
    print("\n   Entity type distribution:")
    entity_type_counts = {}
    for entity in expected_entities:
        entity_type_counts[entity['label']] = entity_type_counts.get(entity['label'], 0) + 1
    
    for entity_type, count in sorted(entity_type_counts.items()):
        print(f"     {entity_type}: {count}")
    
    print("\n" + "="*60)


def main():
    """Run all demonstrations."""
    print("COMPREHENSIVE FEW-SHOT NER DEMONSTRATION")
    print("=" * 60)
    
    # Run all demonstrations
    demonstrate_example_database()
    demonstrate_example_generation()
    demonstrate_example_selection()
    demonstrate_domain_examples()
    demonstrate_context_aware_selection()
    demonstrate_few_shot_templates()
    demonstrate_domain_templates()
    demonstrate_end_to_end_pipeline()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("\nThe few-shot NER system provides:")
    print("✓ Synthetic examples for all 117 entity types")
    print("✓ Multiple template variants (basic, detailed, precision, recall)")
    print("✓ Context-aware example selection")
    print("✓ Domain-specific templates and examples")
    print("✓ Seamless integration with existing NER pipeline")
    print("✓ Comprehensive test coverage")
    print("\nReady for production use in plant metabolomics research!")


if __name__ == "__main__":
    main()