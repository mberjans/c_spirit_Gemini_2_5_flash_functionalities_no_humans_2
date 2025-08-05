#!/usr/bin/env python3
"""
Example usage of the extract ner CLI command.

This script demonstrates how to use the newly implemented extract ner subcommand
for Named Entity Recognition with various options and configurations.
"""

import subprocess
import os
import tempfile
from pathlib import Path

# Sample text for demonstration
SAMPLE_TEXT = """
Plant metabolomics studies often involve the analysis of flavonoids like quercetin 
and kaempferol in Arabidopsis thaliana. These secondary metabolites are produced 
through the phenylpropanoid pathway and can be analyzed using high-performance 
liquid chromatography (HPLC) and mass spectrometry (MS). The ATP synthase gene 
plays a crucial role in cellular energy production, while chlorophyll a and b 
are essential for photosynthesis in plant cells.
"""

def create_sample_file():
    """Create a temporary sample file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(SAMPLE_TEXT.strip())
        return f.name

def demonstrate_basic_usage():
    """Demonstrate basic NER extraction."""
    print("🧬 Basic NER Extraction Example")
    print("=" * 50)
    
    sample_file = create_sample_file()
    output_file = "basic_entities.json"
    
    try:
        # Basic extraction with default metabolomics schema
        cmd = [
            "python", "-m", "src.cli", "extract", "ner", 
            sample_file,
            "--output", output_file,
            "--verbose"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print("\nNote: This would normally require API keys for LLM access.")
        print("Expected behavior:")
        print("- Loads default plant metabolomics schema (67+ entity types)")
        print("- Uses basic zero-shot template")
        print("- Extracts entities like: quercetin (METABOLITE), Arabidopsis thaliana (ORGANISM)")
        print("- Saves results in JSON format with metadata")
        
    finally:
        # Cleanup
        if os.path.exists(sample_file):
            os.unlink(sample_file)
        if os.path.exists(output_file):
            os.unlink(output_file)

def demonstrate_domain_specific():
    """Demonstrate domain-specific extraction."""
    print("\n🌿 Domain-Specific Extraction Example")
    print("=" * 50)
    
    sample_file = create_sample_file()
    output_file = "metabolomics_entities.json"
    
    # Domain-specific metabolomics extraction
    cmd = [
        "python", "-m", "src.cli", "extract", "ner",
        sample_file,
        "--output", output_file,
        "--domain", "metabolomics",
        "--template-type", "scientific",
        "--verbose"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nExpected behavior:")
    print("- Uses metabolomics-specific schema and templates")
    print("- Optimized for metabolite, compound, and analytical method extraction")
    print("- Uses scientific literature conventions")
    
    # Cleanup
    if os.path.exists(sample_file):
        os.unlink(sample_file)

def demonstrate_few_shot():
    """Demonstrate few-shot learning."""
    print("\n🎯 Few-Shot Learning Example")
    print("=" * 50)
    
    sample_file = create_sample_file()
    output_file = "few_shot_entities.json"
    
    # Few-shot extraction with examples
    cmd = [
        "python", "-m", "src.cli", "extract", "ner",
        sample_file,
        "--output", output_file,
        "--few-shot",
        "--num-examples", "5",
        "--template-type", "detailed",
        "--domain", "metabolomics",
        "--verbose"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nExpected behavior:")
    print("- Includes 5 relevant examples in the prompt")
    print("- Uses detailed template for comprehensive extraction")
    print("- Combines domain expertise with example-based learning")
    
    # Cleanup
    if os.path.exists(sample_file):
        os.unlink(sample_file)

def demonstrate_custom_schema():
    """Demonstrate custom schema usage."""
    print("\n⚙️ Custom Schema Example")
    print("=" * 50)
    
    sample_file = create_sample_file()
    schema_file = "examples/sample_entity_schema.json"
    output_file = "custom_entities.json"
    
    # Custom schema extraction
    cmd = [
        "python", "-m", "src.cli", "extract", "ner",
        sample_file,
        "--output", output_file,
        "--schema", schema_file,
        "--template-type", "precision",
        "--confidence-threshold", "0.8",
        "--verbose"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nExpected behavior:")
    print("- Uses custom 8-entity schema instead of default")
    print("- High-precision template minimizes false positives")
    print("- Filters entities below 0.8 confidence threshold")
    
    # Cleanup
    if os.path.exists(sample_file):
        os.unlink(sample_file)

def show_help_examples():
    """Show help command examples."""
    print("\n📚 Help and Documentation")
    print("=" * 50)
    
    commands = [
        "python -m src.cli extract --help",
        "python -m src.cli extract ner --help"
    ]
    
    for cmd in commands:
        print(f"📖 {cmd}")
    
    print("\nAvailable template types: basic, detailed, precision, recall, scientific")
    print("Available domains: metabolomics, genetics, plant_biology")
    print("Supported models: gpt-3.5-turbo, gpt-4, claude-v1, etc.")

def main():
    """Run all demonstration examples."""
    print("🚀 AIM2-ODIE Extract NER Command Demonstration")
    print("=" * 60)
    print("This script shows how to use the extract ner subcommand")
    print("for Named Entity Recognition in plant metabolomics research.\n")
    
    demonstrate_basic_usage()
    demonstrate_domain_specific()
    demonstrate_few_shot()
    demonstrate_custom_schema()
    show_help_examples()
    
    print("\n✅ Demonstration completed!")
    print("\nNote: These examples require:")
    print("- LLM API access (OpenAI, Anthropic, etc.)")
    print("- Properly configured API keys")
    print("- Internet connection for API calls")
    print("\nFor testing without API calls, use mocked versions in the test suite.")

if __name__ == "__main__":
    main()