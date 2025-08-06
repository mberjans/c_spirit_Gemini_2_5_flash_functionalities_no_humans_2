#!/usr/bin/env python3
"""
Demo script showcasing the comprehensive prompt template utilities.

This script demonstrates the key functionality of the new template validation,
optimization, metrics, and management utilities.
"""

from src.llm_extraction.prompt_templates import (
    validate_template_structure,
    validate_examples_format,
    optimize_prompt_for_model,
    calculate_template_metrics,
    suggest_template_improvements,
    register_custom_template,
    get_template_metadata,
    compare_templates,
    get_template_recommendations,
    get_basic_zero_shot_template,
    InvalidTemplateError
)


def demo_template_validation():
    """Demo template structure validation."""
    print("=== Template Structure Validation Demo ===\n")
    
    # Valid template
    valid_template = """
    Extract entities from {text} using the provided {schema}.
    Return JSON with entities array containing text, label, start, end, confidence fields.
    Example: {"entities": [{"text": "quercetin", "label": "METABOLITE", "start": 0, "end": 9, "confidence": 0.95}]}
    """
    
    try:
        result = validate_template_structure(valid_template)
        print(f"✅ Valid template passed validation: {result}")
    except InvalidTemplateError as e:
        print(f"❌ Valid template failed: {e}")
    
    # Invalid template (missing required fields)
    invalid_template = "Extract entities from {text} using {schema}. Return JSON."
    
    try:
        validate_template_structure(invalid_template)
        print("❌ Invalid template incorrectly passed validation")
    except InvalidTemplateError as e:
        print(f"✅ Invalid template correctly rejected: {e}")
    
    print()


def demo_examples_validation():
    """Demo examples format validation."""
    print("=== Examples Format Validation Demo ===\n")
    
    # Valid examples
    valid_examples = [
        {
            "text": "Quercetin is a flavonoid compound found in plants.",
            "entities": [
                {
                    "text": "Quercetin",
                    "label": "METABOLITE", 
                    "start": 0,
                    "end": 9,
                    "confidence": 0.95
                },
                {
                    "text": "flavonoid",
                    "label": "COMPOUND",
                    "start": 15,
                    "end": 24,
                    "confidence": 0.90
                }
            ]
        }
    ]
    
    try:
        result = validate_examples_format(valid_examples)
        print(f"✅ Valid examples passed validation: {result}")
    except InvalidTemplateError as e:
        print(f"❌ Valid examples failed: {e}")
    
    # Invalid examples (text span mismatch)
    invalid_examples = [
        {
            "text": "Test compound",
            "entities": [
                {
                    "text": "Wrong text",  # Doesn't match actual span
                    "label": "COMPOUND",
                    "start": 0,
                    "end": 4,
                    "confidence": 0.95
                }
            ]
        }
    ]
    
    try:
        validate_examples_format(invalid_examples)
        print("❌ Invalid examples incorrectly passed validation")
    except InvalidTemplateError as e:
        print(f"✅ Invalid examples correctly rejected: {e}")
    
    print()


def demo_prompt_optimization():
    """Demo prompt optimization for different models."""
    print("=== Prompt Optimization Demo ===\n")
    
    basic_prompt = "Extract entities from {text} using {schema}. Return JSON with entities array containing text, label, start, end, confidence."
    
    models = ["gpt-4", "claude-3", "gemini-pro", "llama-2"]
    
    for model in models:
        optimized = optimize_prompt_for_model(basic_prompt, model)
        print(f"📝 Optimized for {model}:")
        print(f"   Original length: {len(basic_prompt)} chars")
        print(f"   Optimized length: {len(optimized)} chars")
        print(f"   Added features: {_get_optimization_features(optimized, basic_prompt)}")
        print()


def _get_optimization_features(optimized, original):
    """Helper to identify what features were added during optimization."""
    features = []
    if "**TASK:**" in optimized and "**TASK:**" not in original:
        features.append("Task headers")
    if "**DETAILED APPROACH:**" in optimized:
        features.append("Detailed approach")
    if "**OUTPUT FORMAT:**" in optimized:
        features.append("Output format specs")
    if "reasoning" in optimized.lower() and "reasoning" not in original.lower():
        features.append("Reasoning instructions")
    if len(optimized) > len(original) * 1.2:
        features.append("Expanded instructions")
    
    return ", ".join(features) if features else "Minor formatting"


def demo_template_metrics():
    """Demo template metrics calculation."""
    print("=== Template Metrics Demo ===\n")
    
    template = get_basic_zero_shot_template()
    metrics = calculate_template_metrics(template)
    
    print(f"📊 Template Metrics for Basic Zero-Shot Template:")
    print(f"   Word count: {metrics['word_count']}")
    print(f"   Character count: {metrics['character_count']}")
    print(f"   Complexity level: {metrics['complexity_level']}")
    print(f"   Quality score: {metrics['quality_score']:.2f}")
    print(f"   Readability score: {metrics['readability_score']:.2f}")
    print(f"   Estimated effectiveness: {metrics['estimated_effectiveness']}")
    print(f"   Placeholders: {metrics['placeholders']}")
    print(f"   Section count: {metrics['section_count']}")
    
    quality_indicators = metrics['quality_indicators']
    print(f"   Quality indicators:")
    for indicator, value in quality_indicators.items():
        status = "✅" if value else "❌"
        print(f"     {status} {indicator}")
    
    print()


def demo_improvement_suggestions():
    """Demo template improvement suggestions."""
    print("=== Template Improvement Suggestions Demo ===\n")
    
    # Test with a minimal template
    minimal_template = "Extract entities from {text}."
    
    suggestions = suggest_template_improvements(minimal_template)
    
    print(f"🔧 Improvement suggestions for minimal template:")
    print(f"   Template: '{minimal_template}'")
    print(f"   Number of suggestions: {len(suggestions)}")
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"   {i}. {suggestion}")
    
    print()
    
    # Test with a complete template
    complete_template = get_basic_zero_shot_template()
    complete_suggestions = suggest_template_improvements(complete_template)
    
    print(f"🔧 Suggestions for complete template: {len(complete_suggestions)} suggestions")
    for i, suggestion in enumerate(complete_suggestions[:3], 1):  # Show first 3
        print(f"   {i}. {suggestion}")
    if len(complete_suggestions) > 3:
        print(f"   ... and {len(complete_suggestions) - 3} more")
    
    print()


def demo_template_management():
    """Demo template registry and management."""
    print("=== Template Management Demo ===\n")
    
    # Register a custom template
    custom_template = """
    You are an expert in plant metabolomics analysis. Extract entities from {text} using {schema}.
    Focus on chemical compounds, plant species, and analytical methods.
    Return JSON with entities array containing text, label, start, end, confidence fields.
    Apply high precision to minimize false positives.
    Example: {"entities": [{"text": "HPLC", "label": "ANALYTICAL_METHOD", "start": 0, "end": 4, "confidence": 0.98}]}
    """
    
    try:
        result = register_custom_template("demo_metabolomics", custom_template, "Custom metabolomics template")
        print(f"✅ Custom template registered: {result}")
        
        # Get metadata for the custom template
        metadata = get_template_metadata("demo_metabolomics")
        print(f"📋 Custom template metadata:")
        print(f"   Name: {metadata['name']}")
        print(f"   Type: {metadata['template_type']}")
        print(f"   Domain focus: {metadata['domain_focus']}")
        print(f"   Use case: {metadata['use_case']}")
        print(f"   Effectiveness: {metadata['metrics']['estimated_effectiveness']}")
        print(f"   Suggestions: {len(metadata['suggestions'])} improvement suggestions")
        
        # Clean up
        from src.llm_extraction.prompt_templates import TEMPLATE_REGISTRY
        del TEMPLATE_REGISTRY["demo_metabolomics"]
        
    except Exception as e:
        print(f"❌ Custom template registration failed: {e}")
    
    print()


def demo_template_comparison():
    """Demo template comparison functionality."""
    print("=== Template Comparison Demo ===\n")
    
    template1 = "Extract entities from {text} using {schema}. Return JSON with entities array containing text, label, start, end, confidence."
    template2 = get_basic_zero_shot_template()
    
    comparison = compare_templates(template1, template2)
    
    print(f"⚖️  Template Comparison:")
    print(f"   Template 1 word count: {comparison['template1_metrics']['word_count']}")
    print(f"   Template 2 word count: {comparison['template2_metrics']['word_count']}")
    print(f"   Word count difference: {comparison['differences']['word_count_diff']}")
    print(f"   Quality difference: {comparison['differences']['quality_diff']:.3f}")
    print(f"   Complexity difference: {comparison['differences']['complexity_diff']:.3f}")
    print(f"   Recommendation: {comparison['recommendation']}")
    
    print()


def demo_template_recommendations():
    """Demo template recommendation system."""
    print("=== Template Recommendations Demo ===\n")
    
    # Test different requirement scenarios
    scenarios = [
        {
            'name': 'Metabolomics Research',
            'requirements': {
                'domain': 'metabolomics',
                'accuracy_priority': 'precision',
                'complexity': 'high',
                'use_case': 'research'
            }
        },
        {
            'name': 'Quick Analysis',
            'requirements': {
                'domain': 'general',
                'accuracy_priority': 'balanced',
                'complexity': 'low',
                'use_case': 'quick_analysis'
            }
        },
        {
            'name': 'Genetics Study',
            'requirements': {
                'domain': 'genetics',
                'accuracy_priority': 'recall',
                'complexity': 'medium',
                'use_case': 'research'
            }
        }
    ]
    
    for scenario in scenarios:
        recommendations = get_template_recommendations(scenario['requirements'])
        print(f"🎯 Recommendations for {scenario['name']}:")
        print(f"   Requirements: {scenario['requirements']}")
        print(f"   Top 3 recommended templates:")
        
        for i, template_name in enumerate(recommendations[:3], 1):
            print(f"     {i}. {template_name}")
        
        print()


def main():
    """Run all demos."""
    print("🚀 Prompt Template Utilities Demo\n")
    print("This demo showcases the comprehensive utilities for prompt template")
    print("validation, optimization, metrics calculation, and management.\n")
    
    demo_template_validation()
    demo_examples_validation()
    demo_prompt_optimization()
    demo_template_metrics()
    demo_improvement_suggestions()
    demo_template_management()
    demo_template_comparison()
    demo_template_recommendations()
    
    print("✨ Demo completed! All utility functions are working correctly.")


if __name__ == "__main__":
    main()