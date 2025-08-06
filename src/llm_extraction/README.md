# Zero-Shot Prompt Templates for Plant Metabolomics NER

This module provides comprehensive zero-shot prompt templates for Named Entity Recognition (NER) in plant metabolomics research, designed to work seamlessly with the existing `extract_entities()` function.

## Overview

The prompt templates system offers:

- **Multiple template variants** optimized for different use cases
- **Domain-specific templates** for specialized research areas  
- **Precision/recall optimization** options
- **Template customization** capabilities
- **Automatic template selection** based on text characteristics
- **Integration with comprehensive entity schemas** (117 entity types across 6 categories)

## Quick Start

```python
from llm_extraction.prompt_templates import get_basic_zero_shot_template
from llm_extraction.ner import extract_entities
from llm_extraction.entity_schemas import get_plant_metabolomics_schema

# Get template and schema
template = get_basic_zero_shot_template()
schema = get_plant_metabolomics_schema()

# Extract entities
text = "Quercetin levels increased in Arabidopsis leaves under drought stress."
entities = extract_entities(
    text=text,
    entity_schema=schema,
    llm_model_name="gpt-4",
    prompt_template=template
)
```

## Available Templates

### Core Templates

| Template | Description | Use Case |
|----------|-------------|----------|
| `get_basic_zero_shot_template()` | General-purpose template | Quick analysis, simple texts |
| `get_detailed_zero_shot_template()` | Comprehensive instructions | Thorough analysis, complex texts |
| `get_precision_focused_template()` | High precision, low false positives | Critical applications |
| `get_recall_focused_template()` | High recall, comprehensive extraction | Exploratory analysis |
| `get_scientific_literature_template()` | Academic writing conventions | Research papers, publications |

### Domain-Specific Templates

```python
from llm_extraction.prompt_templates import get_domain_specific_template

# Available domains
metabolomics_template = get_domain_specific_template("metabolomics")
genetics_template = get_domain_specific_template("genetics")
plant_biology_template = get_domain_specific_template("plant_biology")
```

**Supported domains:**
- `metabolomics`, `plant_metabolomics` - Focus on metabolites and analytical methods
- `genetics`, `genomics`, `molecular_biology` - Focus on genes and molecular processes
- `plant_biology`, `botany`, `plant_science` - Focus on plant anatomy and physiology

## Template Selection and Recommendations

### Automatic Use Case Selection

```python
from llm_extraction.prompt_templates import get_template_for_use_case

# Research paper analysis
template = get_template_for_use_case("research_paper")

# Quick analysis with precision focus
template = get_template_for_use_case(
    use_case="quick_analysis",
    precision_recall_balance="precision"
)

# Domain-specific analysis
template = get_template_for_use_case(
    use_case="analysis",
    domain="metabolomics"
)
```

### Smart Recommendations

```python
from llm_extraction.prompt_templates import get_recommended_template

# Get recommendation based on text characteristics
template = get_recommended_template(
    text_length=2000,           # Length of input text
    entity_count_estimate=30,   # Expected number of entities
    domain="metabolomics",      # Optional domain
    accuracy_priority="balanced" # "precision", "recall", or "balanced"
)
```

## Template Customization

```python
from llm_extraction.prompt_templates import customize_template, get_basic_zero_shot_template

base_template = get_basic_zero_shot_template()

# Add custom instructions
customized = customize_template(
    base_template,
    custom_instructions="Focus on secondary metabolites and stress responses",
    confidence_threshold=0.85,
    additional_examples=["Consider metabolite-protein interactions"]
)
```

## Integration with Entity Schemas

The templates work with all entity schemas, including the comprehensive plant metabolomics schema with 117 entity types:

```python
from llm_extraction.entity_schemas import (
    get_plant_metabolomics_schema,  # 117 entity types
    get_basic_schema,               # 8 core types
    get_schema_by_domain           # Domain-specific subsets
)

# Use comprehensive schema
comprehensive_schema = get_plant_metabolomics_schema()

# Use domain-specific subset
metabolomics_schema = get_schema_by_domain("metabolomics")
```

## Entity Categories Supported

The templates are optimized for extracting entities across six main categories:

1. **Plant Metabolites** (10 types)
   - METABOLITE, COMPOUND, PHENOLIC_COMPOUND, FLAVONOID, ALKALOID, etc.

2. **Species** (5 types)  
   - SPECIES, PLANT_SPECIES, ORGANISM, CULTIVAR, ECOTYPE

3. **Plant Anatomical Structures** (11 types)
   - PLANT_PART, PLANT_ORGAN, PLANT_TISSUE, ROOT, LEAF, STEM, etc.

4. **Experimental Conditions** (9 types)
   - EXPERIMENTAL_CONDITION, STRESS_CONDITION, TREATMENT, etc.

5. **Molecular Traits** (9 types)
   - MOLECULAR_TRAIT, GENE_EXPRESSION, ENZYME_ACTIVITY, etc.

6. **Plant Traits** (9 types)
   - PLANT_TRAIT, MORPHOLOGICAL_TRAIT, PHYSIOLOGICAL_TRAIT, etc.

Plus additional supporting categories for genetics, analytical methods, and bioactivity.

## Template Validation and Statistics

```python
from llm_extraction.prompt_templates import validate_template, get_template_statistics

# Validate template format
try:
    validate_template(custom_template)
    print("Template is valid")
except InvalidTemplateError as e:
    print(f"Template error: {e}")

# Get template statistics
stats = get_template_statistics(template)
print(f"Word count: {stats['word_count']}")
print(f"Complexity: {stats['estimated_complexity']}")
print(f"Sections: {stats['section_count']}")
```

## Error Handling

The module provides specific exception types:

```python
from llm_extraction.prompt_templates import (
    TemplateError,           # Base exception
    InvalidTemplateError,    # Invalid template format
    TemplateNotFoundError    # Template not found
)

try:
    template = get_domain_specific_template("invalid_domain")
except TemplateNotFoundError as e:
    print(f"Domain not supported: {e}")
```

## Template Design Principles

### Output Format Requirements

All templates ensure consistent JSON output format:

```json
{
  "entities": [
    {
      "text": "quercetin",
      "label": "METABOLITE", 
      "start": 15,
      "end": 24,
      "confidence": 0.95
    }
  ]
}
```

### Scientific Context Awareness

Templates include:
- Scientific nomenclature conventions (IUPAC, taxonomic)
- Domain-specific terminology patterns
- Academic writing style considerations
- Confidence calibration for scientific literature

### Robust Entity Handling

Templates address:
- Overlapping entity mentions
- Ambiguous terminology
- Multi-word entity spans
- Chemical formulas and systematic names
- Species names and abbreviations

## Examples and Usage Patterns

### Research Paper Analysis

```python
# For analyzing scientific publications
template = get_scientific_literature_template()
schema = get_plant_metabolomics_schema()

# Process research paper text
entities = extract_entities(text, schema, "gpt-4", template)
```

### High-Throughput Screening

```python
# For processing many samples quickly
template = get_basic_zero_shot_template()
schema = get_basic_schema()

# Process multiple texts efficiently
for text in text_samples:
    entities = extract_entities(text, schema, "gpt-3.5-turbo", template)
```

### Precision-Critical Applications

```python
# When false positives are costly
template = get_precision_focused_template()
customized = customize_template(template, confidence_threshold=0.9)

entities = extract_entities(text, schema, "gpt-4", customized)
```

### Exploratory Analysis

```python
# When discovering new entity types
template = get_recall_focused_template()
comprehensive_schema = get_plant_metabolomics_schema()

entities = extract_entities(text, comprehensive_schema, "gpt-4", template)
```

## Performance Considerations

- **Template Length**: Longer templates provide more guidance but increase token usage
- **Schema Size**: Comprehensive schemas improve coverage but increase prompt length
- **Model Selection**: GPT-4 works better with complex templates, GPT-3.5-turbo with simpler ones
- **Batch Processing**: Use basic templates for high-volume processing

## Testing and Validation

The module includes comprehensive tests covering:

- Template retrieval and validation
- Domain-specific functionality  
- Integration with NER system
- Error handling and edge cases
- Performance with large schemas

Run tests with:
```bash
pytest tests/llm_extraction/test_prompt_templates.py -v
```

## Contributing

When adding new templates:

1. Follow the established format with required placeholders (`{text}`, `{schema}`, `{examples}`)
2. Include proper JSON output format specification
3. Add comprehensive validation and error handling
4. Write corresponding test cases
5. Update documentation and examples

## API Reference

See the complete API documentation in the module docstrings and type hints. Key functions:

- `get_*_template()` - Template retrieval functions
- `get_template_for_use_case()` - Smart template selection
- `customize_template()` - Template customization
- `validate_template()` - Template validation
- `get_template_statistics()` - Template analysis

## License and Citation

This implementation is part of the AIM2-ODIE ontology development and information extraction system for plant metabolomics research.