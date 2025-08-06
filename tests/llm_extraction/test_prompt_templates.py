"""
Unit tests for src/llm_extraction/prompt_templates.py

This module tests the comprehensive zero-shot prompt templates for plant metabolomics
Named Entity Recognition (NER). The tests validate template functionality, format
compliance, integration with the existing NER system, and domain-specific requirements.

Test Coverage:
- Template retrieval and validation functions
- Template format and placeholder validation
- Integration with existing NER extract_entities function
- Domain-specific template selection and customization
- Template statistics and recommendation systems
- Error handling for invalid templates and parameters
- Edge cases and boundary conditions
"""

import pytest
import re
import json
from unittest.mock import Mock, patch
from typing import Dict, List, Any

# Import the prompt template functions and classes
from src.llm_extraction.prompt_templates import (
    get_basic_zero_shot_template,
    get_detailed_zero_shot_template,
    get_precision_focused_template,
    get_recall_focused_template,
    get_scientific_literature_template,
    get_domain_specific_template,
    get_template_by_name,
    list_available_templates,
    validate_template,
    get_template_for_use_case,
    customize_template,
    get_template_statistics,
    validate_template_output_format,
    get_recommended_template,
    TemplateError,
    InvalidTemplateError,
    TemplateNotFoundError,
    TemplateType,
    TEMPLATE_REGISTRY
)

# Import NER functions for integration testing
from src.llm_extraction.ner import extract_entities, _format_prompt
from src.llm_extraction.entity_schemas import get_plant_metabolomics_schema


class TestBasicTemplateRetrieval:
    """Test cases for basic template retrieval functions."""
    
    def test_get_basic_zero_shot_template(self):
        """Test retrieval of basic zero-shot template."""
        template = get_basic_zero_shot_template()
        
        assert isinstance(template, str)
        assert len(template.strip()) > 0
        assert "{text}" in template
        assert "{schema}" in template
        assert "{examples}" in template
        assert "json" in template.lower()
        assert "entities" in template.lower()
    
    def test_get_detailed_zero_shot_template(self):
        """Test retrieval of detailed zero-shot template."""
        template = get_detailed_zero_shot_template()
        
        assert isinstance(template, str)
        assert len(template) > len(get_basic_zero_shot_template())
        assert "{text}" in template
        assert "{schema}" in template
        assert "detailed" in template.lower() or "comprehensive" in template.lower()
        assert "confidence" in template.lower()
    
    def test_get_precision_focused_template(self):
        """Test retrieval of precision-focused template."""
        template = get_precision_focused_template()
        
        assert isinstance(template, str)
        assert "{text}" in template
        assert "{schema}" in template
        assert "precision" in template.lower()
        assert "confident" in template.lower() or "accuracy" in template.lower()
    
    def test_get_recall_focused_template(self):
        """Test retrieval of recall-focused template."""
        template = get_recall_focused_template()
        
        assert isinstance(template, str)
        assert "{text}" in template
        assert "{schema}" in template
        assert "recall" in template.lower() or "comprehensive" in template.lower()
        assert "all" in template.lower()
    
    def test_get_scientific_literature_template(self):
        """Test retrieval of scientific literature template."""
        template = get_scientific_literature_template()
        
        assert isinstance(template, str)
        assert "{text}" in template
        assert "{schema}" in template
        assert "scientific" in template.lower() or "literature" in template.lower()
        assert "nomenclature" in template.lower() or "academic" in template.lower()


class TestDomainSpecificTemplates:
    """Test cases for domain-specific template functionality."""
    
    def test_get_domain_specific_template_metabolomics(self):
        """Test retrieval of metabolomics domain template."""
        template = get_domain_specific_template("metabolomics")
        
        assert isinstance(template, str)
        assert "{text}" in template
        assert "{schema}" in template
        assert "metabol" in template.lower()
    
    def test_get_domain_specific_template_genetics(self):
        """Test retrieval of genetics domain template."""
        template = get_domain_specific_template("genetics")
        
        assert isinstance(template, str)
        assert "{text}" in template
        assert "{schema}" in template
        assert "gene" in template.lower() or "genetic" in template.lower()
    
    def test_get_domain_specific_template_plant_biology(self):
        """Test retrieval of plant biology domain template."""
        template = get_domain_specific_template("plant_biology")
        
        assert isinstance(template, str)
        assert "{text}" in template
        assert "{schema}" in template
        assert "plant" in template.lower()
    
    def test_get_domain_specific_template_case_insensitive(self):
        """Test that domain template retrieval is case insensitive."""
        domains = ["METABOLOMICS", "Genetics", "plant_Biology"]
        
        for domain in domains:
            template = get_domain_specific_template(domain)
            assert isinstance(template, str)
            assert len(template.strip()) > 0
    
    def test_get_domain_specific_template_aliases(self):
        """Test that domain aliases work correctly."""
        # Test metabolomics aliases
        metabolomics_aliases = ["metabolomics", "plant_metabolomics"]
        base_template = get_domain_specific_template("metabolomics")
        
        for alias in metabolomics_aliases:
            template = get_domain_specific_template(alias)
            assert template == base_template
        
        # Test genetics aliases
        genetics_aliases = ["genetics", "genomics", "molecular_biology"]
        base_genetics = get_domain_specific_template("genetics")
        
        for alias in genetics_aliases:
            template = get_domain_specific_template(alias)
            assert template == base_genetics
    
    def test_get_domain_specific_template_invalid_domain(self):
        """Test error handling for invalid domain names."""
        invalid_domains = ["invalid_domain", "chemistry", "physics", ""]
        
        for domain in invalid_domains:
            with pytest.raises(TemplateNotFoundError):
                get_domain_specific_template(domain)


class TestTemplateRegistry:
    """Test cases for template registry functionality."""
    
    def test_list_available_templates(self):
        """Test listing of available templates."""
        templates = list_available_templates()
        
        assert isinstance(templates, list)
        assert len(templates) > 0
        assert "basic" in templates
        assert "detailed" in templates
        assert "precision" in templates
        assert "recall" in templates
    
    def test_get_template_by_name_valid(self):
        """Test retrieval of templates by valid names."""
        template_names = ["basic", "detailed", "precision", "recall"]
        
        for name in template_names:
            template = get_template_by_name(name)
            assert isinstance(template, str)
            assert len(template.strip()) > 0
            assert "{text}" in template
            assert "{schema}" in template
    
    def test_get_template_by_name_case_insensitive(self):
        """Test that template name retrieval is case insensitive."""
        names = ["BASIC", "Detailed", "pRECISION"]
        
        for name in names:
            template = get_template_by_name(name)
            assert isinstance(template, str)
            assert len(template.strip()) > 0
    
    def test_get_template_by_name_invalid(self):
        """Test error handling for invalid template names."""
        invalid_names = ["invalid", "nonexistent", "", "123"]
        
        for name in invalid_names:
            with pytest.raises(TemplateNotFoundError):
                get_template_by_name(name)
    
    def test_template_registry_completeness(self):
        """Test that template registry contains all expected templates."""
        expected_templates = [
            "basic", "detailed", "precision", "recall", 
            "scientific", "metabolomics", "genetics", "plant_biology"
        ]
        
        available_templates = list_available_templates()
        
        for expected in expected_templates:
            assert expected in available_templates


class TestTemplateValidation:
    """Test cases for template validation functionality."""
    
    def test_validate_template_valid(self):
        """Test validation of valid templates."""
        valid_templates = [
            get_basic_zero_shot_template(),
            get_detailed_zero_shot_template(),
            get_precision_focused_template(),
            get_recall_focused_template()
        ]
        
        for template in valid_templates:
            # Should not raise any exception
            assert validate_template(template) is True
    
    def test_validate_template_missing_placeholders(self):
        """Test validation of templates with missing required placeholders."""
        invalid_templates = [
            # Missing {text}
            "Extract entities from the text. Schema: {schema}. Examples: {examples}",
            # Missing {schema}
            "Extract entities from: {text}. Examples: {examples}",
            # Missing both {text} and {schema}
            "Extract entities. Examples: {examples}"
        ]
        
        for template in invalid_templates:
            with pytest.raises(InvalidTemplateError, match="missing required placeholders"):
                validate_template(template)
    
    def test_validate_template_unknown_placeholders(self):
        """Test validation of templates with unknown placeholders."""
        invalid_template = "Extract {unknown} from {text} with {schema} and {examples}"
        
        with pytest.raises(InvalidTemplateError, match="unknown placeholders"):
            validate_template(invalid_template)
    
    def test_validate_template_empty_or_invalid_type(self):
        """Test validation of empty or non-string templates."""
        invalid_templates = [
            "",  # Empty string
            "   ",  # Whitespace only
            None,  # None
            123,  # Non-string
            [],  # List
            {}  # Dictionary
        ]
        
        for template in invalid_templates:
            with pytest.raises(InvalidTemplateError):
                validate_template(template)
    
    def test_validate_template_missing_json_specification(self):
        """Test validation requires JSON output specification."""
        template_without_json = "Extract {schema} entities from: {text}. Return results."
        
        with pytest.raises(InvalidTemplateError, match="JSON output format"):
            validate_template(template_without_json)
    
    def test_validate_template_missing_required_fields(self):
        """Test validation requires mention of required entity fields."""
        required_fields = ["text", "label", "start", "end", "confidence"]
        
        for field in required_fields:
            # Create template missing this field but having all others
            other_fields = [f for f in required_fields if f != field]
            template = f"Extract entities from {{text}} with {{schema}}. Return JSON with {', '.join(other_fields)} fields. Include array entities format."
            
            # Only test if the template is missing the field entirely
            if field not in template.lower():
                with pytest.raises(InvalidTemplateError, match=f"required field: {field}"):
                    validate_template(template)
    
    def test_validate_template_output_format_valid(self):
        """Test validation of templates with proper output format."""
        valid_template = """
        Extract entities from {text} using {schema}.
        Return JSON with entities array containing:
        - text: entity text
        - label: entity label  
        - start: start position
        - end: end position
        - confidence: confidence score
        
        Example: {"entities": [{"text": "compound", "label": "METABOLITE", "start": 0, "end": 8, "confidence": 0.9}]}
        """
        
        assert validate_template_output_format(valid_template) is True
    
    def test_validate_template_output_format_invalid(self):
        """Test validation fails for inadequate output format specification."""
        invalid_templates = [
            # No JSON mention
            "Extract entities from {text} using {schema}. Return results.",
            # Missing required fields
            "Extract entities from {text} using {schema}. Return JSON with text and label.",
            # No example
            "Extract entities from {text} using {schema}. Return JSON with text, label, start, end, confidence."
        ]
        
        for template in invalid_templates:
            with pytest.raises(InvalidTemplateError):
                validate_template_output_format(template)


class TestTemplateIntegration:
    """Test cases for template integration with NER system."""
    
    def test_template_integration_with_format_prompt(self):
        """Test that templates work with existing _format_prompt function."""
        template = get_basic_zero_shot_template()
        text = "Quercetin is a flavonoid found in plants."
        schema = {"METABOLITE": "Chemical compounds", "ORGANISM": "Species"}
        
        formatted_prompt = _format_prompt(template, text, schema, None)
        
        assert isinstance(formatted_prompt, str)
        assert text in formatted_prompt
        assert "METABOLITE" in formatted_prompt
        assert "Chemical compounds" in formatted_prompt
        assert "ORGANISM" in formatted_prompt
        assert "Species" in formatted_prompt
        # {text}, {schema}, {examples} should be replaced
        assert "{text}" not in formatted_prompt
        assert "{schema}" not in formatted_prompt
        assert "{examples}" not in formatted_prompt
    
    def test_template_integration_with_examples(self):
        """Test template integration with few-shot examples."""
        template = get_detailed_zero_shot_template()
        text = "Sample text"
        schema = {"COMPOUND": "Chemical compounds"}
        examples = [
            {
                "text": "Glucose is a sugar.",
                "entities": [{"text": "Glucose", "label": "COMPOUND"}]
            }
        ]
        
        formatted_prompt = _format_prompt(template, text, schema, examples)
        
        assert "Glucose" in formatted_prompt
        assert "sugar" in formatted_prompt
        assert "Examples" in formatted_prompt or "examples" in formatted_prompt
    
    @patch('src.llm_extraction.ner.requests.post')
    def test_template_integration_with_extract_entities(self, mock_post):
        """Test full integration with extract_entities function."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "entities": [
                {"text": "quercetin", "label": "METABOLITE", "start": 0, "end": 9, "confidence": 0.95}
            ]
        }
        mock_post.return_value = mock_response
        
        # Test with different templates
        templates = [
            get_basic_zero_shot_template(),
            get_detailed_zero_shot_template(),
            get_precision_focused_template()
        ]
        
        text = "Quercetin is a flavonoid compound."
        schema = {"METABOLITE": "Chemical metabolites"}
        
        for template in templates:
            result = extract_entities(text, schema, "gpt-4", template)
            
            assert len(result) == 1
            assert result[0]["text"] == "quercetin"
            assert result[0]["label"] == "METABOLITE"
    
    def test_template_integration_with_plant_metabolomics_schema(self):
        """Test template integration with comprehensive entity schema."""
        template = get_scientific_literature_template()
        schema = get_plant_metabolomics_schema()
        text = "Sample scientific text"
        
        formatted_prompt = _format_prompt(template, text, schema, None)
        
        # Should contain multiple entity types from schema
        entity_types = ["METABOLITE", "SPECIES", "PLANT_PART", "GENE", "PROTEIN"]
        for entity_type in entity_types:
            assert entity_type in formatted_prompt
        
        # Should be substantial in length due to comprehensive schema
        assert len(formatted_prompt) > 1000


class TestTemplateUseCaseSelection:
    """Test cases for use case-based template selection."""
    
    def test_get_template_for_use_case_research_paper(self):
        """Test template selection for research paper use case."""
        template = get_template_for_use_case("research_paper")
        expected = get_scientific_literature_template()
        
        assert template == expected
    
    def test_get_template_for_use_case_quick_analysis(self):
        """Test template selection for quick analysis use case."""
        template = get_template_for_use_case("quick_analysis")
        expected = get_basic_zero_shot_template()
        
        assert template == expected
    
    def test_get_template_for_use_case_with_domain(self):
        """Test template selection with domain specification."""
        template = get_template_for_use_case("analysis", domain="metabolomics")
        expected = get_domain_specific_template("metabolomics")
        
        assert template == expected
    
    def test_get_template_for_use_case_precision_recall(self):
        """Test template selection based on precision/recall preference."""
        precision_template = get_template_for_use_case("analysis", precision_recall_balance="precision")
        recall_template = get_template_for_use_case("analysis", precision_recall_balance="recall")
        
        assert precision_template == get_precision_focused_template()
        assert recall_template == get_recall_focused_template()
    
    def test_get_template_for_use_case_aliases(self):
        """Test that use case aliases work correctly."""
        scientific_aliases = ["research_paper", "scientific_literature", "publication"]
        expected = get_scientific_literature_template()
        
        for alias in scientific_aliases:
            template = get_template_for_use_case(alias)
            assert template == expected


class TestTemplateCustomization:
    """Test cases for template customization functionality."""
    
    def test_customize_template_with_custom_instructions(self):
        """Test adding custom instructions to templates."""
        base_template = get_basic_zero_shot_template()
        custom_instructions = "Focus specifically on plant secondary metabolites."
        
        customized = customize_template(base_template, custom_instructions=custom_instructions)
        
        assert custom_instructions in customized
        assert "CUSTOM INSTRUCTIONS" in customized
        assert len(customized) > len(base_template)
    
    def test_customize_template_with_confidence_threshold(self):
        """Test adding confidence threshold to templates."""
        base_template = get_basic_zero_shot_template()
        threshold = 0.85
        
        customized = customize_template(base_template, confidence_threshold=threshold)
        
        assert f"confidence >= {threshold:.2f}" in customized
        assert "CONFIDENCE THRESHOLD" in customized
    
    def test_customize_template_with_additional_examples(self):
        """Test adding additional examples to templates."""
        base_template = get_basic_zero_shot_template()
        additional_examples = ["Consider metabolite-protein interactions", "Include pathway information"]
        
        customized = customize_template(base_template, additional_examples=additional_examples)
        
        for example in additional_examples:
            assert example in customized
        assert "ADDITIONAL CONTEXT" in customized
    
    def test_customize_template_invalid_confidence_threshold(self):
        """Test error handling for invalid confidence thresholds."""
        base_template = get_basic_zero_shot_template()
        invalid_thresholds = [-0.1, 1.1, 2.0]
        
        for threshold in invalid_thresholds:
            with pytest.raises(InvalidTemplateError, match="Confidence threshold must be between"):
                customize_template(base_template, confidence_threshold=threshold)
    
    def test_customize_template_invalid_base_template(self):
        """Test error handling for invalid base templates."""
        invalid_base = "Invalid template without required placeholders"
        
        with pytest.raises(InvalidTemplateError):
            customize_template(invalid_base, custom_instructions="Test")


class TestTemplateStatistics:
    """Test cases for template statistics and analysis."""
    
    def test_get_template_statistics_basic(self):
        """Test basic template statistics calculation."""
        template = get_basic_zero_shot_template()
        stats = get_template_statistics(template)
        
        assert isinstance(stats, dict)
        assert "word_count" in stats
        assert "character_count" in stats
        assert "placeholders" in stats
        assert "placeholder_count" in stats
        assert "sections" in stats
        assert "instruction_density" in stats
        assert "estimated_complexity" in stats
        
        assert isinstance(stats["word_count"], int)
        assert stats["word_count"] > 0
        assert isinstance(stats["character_count"], int)
        assert stats["character_count"] > stats["word_count"]
        assert isinstance(stats["placeholders"], list)
        assert "{text}" in stats["placeholders"]
        assert "{schema}" in stats["placeholders"]
    
    def test_get_template_statistics_comparison(self):
        """Test statistics comparison between different templates."""
        basic_stats = get_template_statistics(get_basic_zero_shot_template())
        detailed_stats = get_template_statistics(get_detailed_zero_shot_template())
        
        # Detailed template should be more complex
        assert detailed_stats["word_count"] > basic_stats["word_count"]
        assert detailed_stats["character_count"] > basic_stats["character_count"]
        assert detailed_stats["section_count"] >= basic_stats["section_count"]
    
    def test_get_template_statistics_complexity_classification(self):
        """Test template complexity classification."""
        templates_and_expected_complexity = [
            (get_basic_zero_shot_template(), ["low", "medium"]),  # Could be either
            (get_detailed_zero_shot_template(), ["medium", "high"]),  # Should be medium or high
            (get_scientific_literature_template(), ["medium", "high"])  # Should be medium or high (adjusted expectation)
        ]
        
        for template, expected_complexities in templates_and_expected_complexity:
            stats = get_template_statistics(template)
            assert stats["estimated_complexity"] in expected_complexities


class TestTemplateRecommendations:
    """Test cases for template recommendation system."""
    
    def test_get_recommended_template_short_text(self):
        """Test template recommendation for short texts."""
        recommended = get_recommended_template(
            text_length=200,
            entity_count_estimate=3,
            domain=None,
            accuracy_priority="balanced"
        )
        
        expected = get_basic_zero_shot_template()
        assert recommended == expected
    
    def test_get_recommended_template_long_text(self):
        """Test template recommendation for long texts."""
        recommended = get_recommended_template(
            text_length=3000,
            entity_count_estimate=50,
            domain=None,
            accuracy_priority="balanced"
        )
        
        expected = get_scientific_literature_template()
        assert recommended == expected
    
    def test_get_recommended_template_with_domain(self):
        """Test template recommendation with domain specification."""
        recommended = get_recommended_template(
            text_length=1000,
            entity_count_estimate=20,
            domain="metabolomics",
            accuracy_priority="balanced"
        )
        
        expected = get_domain_specific_template("metabolomics")
        assert recommended == expected
    
    def test_get_recommended_template_precision_priority(self):
        """Test template recommendation with precision priority."""
        recommended = get_recommended_template(
            text_length=1000,
            entity_count_estimate=20,
            domain=None,
            accuracy_priority="precision"
        )
        
        expected = get_precision_focused_template()
        assert recommended == expected
    
    def test_get_recommended_template_recall_priority(self):
        """Test template recommendation with recall priority."""
        recommended = get_recommended_template(
            text_length=1000,
            entity_count_estimate=20,
            domain=None,
            accuracy_priority="recall"
        )
        
        expected = get_recall_focused_template()
        assert recommended == expected


class TestErrorHandling:
    """Test cases for error handling in template system."""
    
    def test_template_error_inheritance(self):
        """Test that TemplateError properly inherits from Exception."""
        error = TemplateError("Test template error")
        assert isinstance(error, Exception)
        assert str(error) == "Test template error"
    
    def test_invalid_template_error_inheritance(self):
        """Test that InvalidTemplateError properly inherits from TemplateError."""
        error = InvalidTemplateError("Invalid template")
        assert isinstance(error, TemplateError)
        assert isinstance(error, Exception)
        assert str(error) == "Invalid template"
    
    def test_template_not_found_error_inheritance(self):
        """Test that TemplateNotFoundError properly inherits from TemplateError."""
        error = TemplateNotFoundError("Template not found")
        assert isinstance(error, TemplateError)
        assert str(error) == "Template not found"
    
    def test_error_messages_are_descriptive(self):
        """Test that error messages provide helpful information."""
        # Test domain not found error
        try:
            get_domain_specific_template("invalid_domain")
        except TemplateNotFoundError as e:
            assert "invalid_domain" in str(e)
            assert "Available domains:" in str(e)
        
        # Test template not found error
        try:
            get_template_by_name("invalid_template")
        except TemplateNotFoundError as e:
            assert "invalid_template" in str(e)
            assert "Available templates:" in str(e)


class TestTemplateType:
    """Test cases for TemplateType enumeration."""
    
    def test_template_type_values(self):
        """Test that TemplateType enum has expected values."""
        expected_types = [
            "basic", "detailed", "precision", "recall", "scientific",
            "metabolomics", "genetics", "plant_biology", "biochemistry",
            "stress", "analytical"
        ]
        
        actual_types = [template_type.value for template_type in TemplateType]
        
        for expected in expected_types:
            assert expected in actual_types
    
    def test_template_type_enum_usage(self):
        """Test using TemplateType enum values."""
        # Should be able to access template using enum values
        basic_template = TEMPLATE_REGISTRY[TemplateType.BASIC.value]
        assert isinstance(basic_template, str)
        assert "{text}" in basic_template


class TestEdgeCasesAndBoundaryConditions:
    """Test cases for edge cases and boundary conditions."""
    
    def test_template_with_special_characters(self):
        """Test templates handle special characters correctly."""
        template = get_basic_zero_shot_template()
        text = "β-carotene and α-tocopherol in café extracts (n=10) [p<0.05]"
        schema = {"COMPOUND": "Chemical compounds"}
        
        formatted_prompt = _format_prompt(template, text, schema, None)
        
        # Should contain the special characters
        assert "β-carotene" in formatted_prompt
        assert "α-tocopherol" in formatted_prompt
        assert "café" in formatted_prompt
    
    def test_template_with_very_large_schema(self):
        """Test templates work with very large entity schemas."""
        template = get_detailed_zero_shot_template()
        large_schema = get_plant_metabolomics_schema()  # 117 entity types
        text = "Sample text"
        
        formatted_prompt = _format_prompt(template, text, large_schema, None)
        
        # Should contain many entity types
        entity_count = len([line for line in formatted_prompt.split('\n') if line.strip().startswith('- ')])
        assert entity_count > 50  # Should have many entity type descriptions
    
    def test_template_with_empty_schema(self):
        """Test template behavior with empty schema."""
        template = get_basic_zero_shot_template()
        text = "Sample text"
        empty_schema = {}
        
        formatted_prompt = _format_prompt(template, text, empty_schema, None)
        
        # Should handle empty schema gracefully
        assert isinstance(formatted_prompt, str)
        assert text in formatted_prompt
    
    def test_template_with_very_long_text(self):
        """Test templates with very long input texts."""
        template = get_basic_zero_shot_template()
        long_text = "Plant metabolomics research. " * 1000  # Very long text
        schema = {"METABOLITE": "Chemical compounds"}
        
        formatted_prompt = _format_prompt(template, long_text, schema, None)
        
        # Should handle long text without issues
        assert isinstance(formatted_prompt, str)
        assert len(formatted_prompt) > len(long_text)
        assert long_text in formatted_prompt
    
    def test_template_whitespace_handling(self):
        """Test that templates handle whitespace correctly."""
        templates = [
            get_basic_zero_shot_template(),
            get_detailed_zero_shot_template(),
            get_precision_focused_template()
        ]
        
        for template in templates:
            # Should not have excessive whitespace
            lines = template.split('\n')
            for line in lines:
                # No lines should have trailing whitespace
                assert line == line.rstrip()
    
    def test_template_placeholder_edge_cases(self):
        """Test template placeholder handling edge cases."""
        template = get_basic_zero_shot_template()
        
        # Test with text containing placeholder-like strings
        text = "Extract {entities} from {input} using {method}"
        schema = {"ENTITY": "Generic entities"}
        
        formatted_prompt = _format_prompt(template, text, schema, None)
        
        # Original placeholder-like strings in text should be preserved
        assert "{entities}" in formatted_prompt
        assert "{input}" in formatted_prompt
        assert "{method}" in formatted_prompt
        
        # But template placeholders should be replaced
        assert text in formatted_prompt  # Original text should be there
        assert "ENTITY" in formatted_prompt  # Schema should be formatted


# Fixtures for test data
@pytest.fixture
def sample_metabolomics_text():
    """Fixture providing sample metabolomics text."""
    return """
    LC-MS analysis revealed increased levels of quercetin and kaempferol in 
    Arabidopsis thaliana leaves under drought stress conditions. These flavonoid 
    compounds showed enhanced expression of CHS and F3H genes in the phenylpropanoid 
    biosynthesis pathway.
    """


@pytest.fixture
def comprehensive_entity_schema():
    """Fixture providing comprehensive entity schema."""
    return {
        "METABOLITE": "Primary and secondary metabolites",
        "ANALYTICAL_METHOD": "Analytical techniques and instruments",
        "SPECIES": "Plant and organism species",
        "PLANT_PART": "Plant anatomical structures",
        "STRESS_CONDITION": "Environmental stress conditions",
        "GENE": "Gene names and genetic elements",
        "PATHWAY": "Biochemical and metabolic pathways"
    }


@pytest.fixture
def sample_few_shot_examples():
    """Fixture providing sample few-shot examples."""
    return [
        {
            "text": "GC-MS detected anthocyanins in grape berries during ripening.",
            "entities": [
                {"text": "GC-MS", "label": "ANALYTICAL_METHOD"},
                {"text": "anthocyanins", "label": "METABOLITE"},
                {"text": "grape", "label": "SPECIES"},
                {"text": "berries", "label": "PLANT_PART"}
            ]
        }
    ]


# Mark all tests in this module as template related
pytestmark = pytest.mark.template