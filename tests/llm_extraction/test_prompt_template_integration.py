"""
Comprehensive integration tests for prompt templates with mock LLM calls.

This module provides comprehensive testing for the prompt template system to ensure
templates work correctly with mock LLM calls and produce expected output formats.
Tests validate integration with the existing NER pipeline and error handling.

Test Coverage:
- Template functionality tests with mock LLM responses
- Zero-shot and few-shot template variants validation
- Output format validation and entity parsing
- Integration with extract_entities() function
- Error handling for various failure scenarios
- Performance tests for template generation
- Domain-specific template testing
- Edge cases and boundary conditions

The tests use mock LLM responses to avoid API costs while ensuring the complete
pipeline works correctly from template generation to entity extraction.

How to run the tests:
- Run all tests: pytest tests/llm_extraction/test_prompt_template_integration.py -v
- Run specific test class: pytest tests/llm_extraction/test_prompt_template_integration.py::TestZeroShotTemplateIntegration -v
- Run with coverage: pytest tests/llm_extraction/test_prompt_template_integration.py --cov=src.llm_extraction
- Quick summary: pytest tests/llm_extraction/test_prompt_template_integration.py --tb=no
"""

import pytest
import json
import re
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List, Any, Optional
import requests
from requests.exceptions import RequestException, Timeout, HTTPError

# Import prompt template functions
from src.llm_extraction.prompt_templates import (
    get_basic_zero_shot_template,
    get_detailed_zero_shot_template,
    get_precision_focused_template,
    get_recall_focused_template,
    get_scientific_literature_template,
    get_domain_specific_template,
    get_few_shot_template,
    get_few_shot_basic_template,
    get_few_shot_detailed_template,
    get_few_shot_precision_template,
    get_few_shot_recall_template,
    get_few_shot_domain_template,
    generate_synthetic_examples,
    select_examples,
    format_examples_for_prompt,
    get_examples_by_domain,
    get_context_aware_examples,
    validate_template,
    get_template_by_name,
    list_available_templates,
    TemplateError,
    InvalidTemplateError,
    TemplateNotFoundError,
    TemplateType
)

# Import NER functions for integration testing
from src.llm_extraction.ner import (
    extract_entities,
    _format_prompt,
    _parse_llm_response,
    _make_llm_request,
    _validate_response_format,
    NERError,
    LLMAPIError,
    InvalidSchemaError,
    RateLimitError
)

# Import entity schemas
from src.llm_extraction.entity_schemas import (
    get_plant_metabolomics_schema,
    get_basic_schema,
    get_schema_by_domain,
    PLANT_METABOLOMICS_SCHEMA
)


class TestPromptTemplateIntegration:
    """Comprehensive integration tests for prompt templates with mock LLM calls."""

    @pytest.fixture
    def sample_text(self):
        """Sample scientific text for testing."""
        return """
        Flavonoids are secondary metabolites widely distributed in Arabidopsis thaliana.
        These compounds, including quercetin and kaempferol, are found in leaves and flowers.
        Under drought stress conditions, the expression of flavonoid biosynthesis genes
        like CHS and DFR is significantly upregulated in root tissues.
        """

    @pytest.fixture
    def basic_schema(self):
        """Basic entity schema for testing."""
        return {
            "METABOLITE": "Primary and secondary metabolites found in plants",
            "SPECIES": "Plant and organism species names",
            "PLANT_PART": "Plant anatomical structures and tissues",
            "GENE": "Gene names and genetic elements",
            "EXPERIMENTAL_CONDITION": "Experimental treatments and conditions"
        }

    @pytest.fixture
    def mock_successful_response(self):
        """Mock successful LLM API response with entities."""
        return {
            "entities": [
                {
                    "text": "Flavonoids",
                    "label": "METABOLITE",
                    "start": 9,
                    "end": 19,
                    "confidence": 0.95
                },
                {
                    "text": "Arabidopsis thaliana",
                    "label": "SPECIES",
                    "start": 67,
                    "end": 87,
                    "confidence": 0.98
                },
                {
                    "text": "quercetin",
                    "label": "METABOLITE", 
                    "start": 136,
                    "end": 145,
                    "confidence": 0.92
                },
                {
                    "text": "leaves",
                    "label": "PLANT_PART",
                    "start": 172,
                    "end": 178,
                    "confidence": 0.88
                },
                {
                    "text": "CHS",
                    "label": "GENE",
                    "start": 287,
                    "end": 290,
                    "confidence": 0.85
                },
                {
                    "text": "drought stress",
                    "label": "EXPERIMENTAL_CONDITION",
                    "start": 194,
                    "end": 208,
                    "confidence": 0.90
                }
            ]
        }

    @pytest.fixture
    def mock_empty_response(self):
        """Mock LLM response with no entities."""
        return {"entities": []}

    @pytest.fixture
    def mock_malformed_response(self):
        """Mock malformed LLM response."""
        return {"invalid_key": "This response is missing entities key"}

    @pytest.fixture
    def few_shot_examples(self):
        """Sample few-shot examples for testing."""
        return [
            {
                "text": "Anthocyanins accumulate in grape berries during ripening.",
                "entities": [
                    {"text": "Anthocyanins", "label": "METABOLITE", "start": 0, "end": 12},
                    {"text": "grape", "label": "SPECIES", "start": 26, "end": 31},
                    {"text": "berries", "label": "PLANT_PART", "start": 32, "end": 39}
                ]
            },
            {
                "text": "PAL enzyme activity increases under UV-B radiation in tomato leaves.",
                "entities": [
                    {"text": "PAL", "label": "GENE", "start": 0, "end": 3},
                    {"text": "UV-B radiation", "label": "EXPERIMENTAL_CONDITION", "start": 36, "end": 50},
                    {"text": "tomato", "label": "SPECIES", "start": 54, "end": 60},
                    {"text": "leaves", "label": "PLANT_PART", "start": 61, "end": 67}
                ]
            }
        ]


class TestZeroShotTemplateIntegration(TestPromptTemplateIntegration):
    """Test zero-shot template integration with mock LLM calls."""

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_basic_zero_shot_template_integration(self, mock_llm_request, sample_text, basic_schema, mock_successful_response):
        """Test basic zero-shot template with successful mock response."""
        # Setup mock response
        mock_llm_request.return_value = mock_successful_response

        # Get template and extract entities
        template = get_basic_zero_shot_template()
        result = extract_entities(
            text=sample_text,
            entity_schema=basic_schema,
            llm_model_name="gpt-3.5-turbo",
            prompt_template=template
        )

        # Verify API call was made
        assert mock_llm_request.called
        call_args = mock_llm_request.call_args

        # Verify prompt formatting
        prompt_content = call_args[0][0]  # First argument is the prompt
        
        assert "extract" in prompt_content.lower()
        assert sample_text.strip() in prompt_content
        assert "METABOLITE" in prompt_content
        assert "JSON" in prompt_content

        # Verify results
        assert len(result) == 6
        assert result[0]['text'] == "Flavonoids"
        assert result[0]['label'] == "METABOLITE"
        assert result[1]['text'] == "Arabidopsis thaliana"
        assert result[1]['label'] == "SPECIES"

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_detailed_zero_shot_template_integration(self, mock_llm_request, sample_text, basic_schema, mock_successful_response):
        """Test detailed zero-shot template with mock response."""
        mock_llm_request.return_value = mock_successful_response

        template = get_detailed_zero_shot_template()
        result = extract_entities(
            text=sample_text,
            entity_schema=basic_schema,
            llm_model_name="gpt-4",
            prompt_template=template
        )

        # Verify detailed instructions in prompt
        prompt_content = mock_llm_request.call_args[0][0]
        
        assert "instructions" in prompt_content.lower() or "guidelines" in prompt_content.lower()
        assert "overlapping" in prompt_content.lower() or "confidence" in prompt_content.lower()

        # Verify results
        assert len(result) == 6
        assert all(0.0 <= entity['confidence'] <= 1.0 for entity in result)

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_precision_focused_template_integration(self, mock_llm_request, sample_text, basic_schema, mock_successful_response):
        """Test precision-focused template with mock response."""
        mock_llm_request.return_value = mock_successful_response

        template = get_precision_focused_template()
        result = extract_entities(
            text=sample_text,
            entity_schema=basic_schema,
            llm_model_name="gpt-4",
            prompt_template=template
        )

        # Verify precision instructions in prompt
        prompt_content = mock_llm_request.call_args[0][0]
        
        assert "high confidence" in prompt_content.lower() or "precise" in prompt_content.lower()

        # Verify high confidence scores (precision focus)
        assert len(result) == 6
        high_confidence_entities = [e for e in result if e['confidence'] >= 0.8]
        assert len(high_confidence_entities) >= 4  # Most should be high confidence

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_recall_focused_template_integration(self, mock_llm_request, sample_text, basic_schema, mock_successful_response):
        """Test recall-focused template with mock response."""
        mock_llm_request.return_value = mock_successful_response

        template = get_recall_focused_template()
        result = extract_entities(
            text=sample_text,
            entity_schema=basic_schema,
            llm_model_name="gpt-4",
            prompt_template=template
        )

        # Verify recall instructions in prompt
        prompt_content = mock_llm_request.call_args[0][0]
        
        assert "comprehensive" in prompt_content.lower() or "all possible" in prompt_content.lower()

        # Verify results (recall focus should capture entities)
        assert len(result) == 6

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_scientific_literature_template_integration(self, mock_llm_request, sample_text, basic_schema, mock_successful_response):
        """Test scientific literature template with mock response."""
        mock_llm_request.return_value = mock_successful_response

        template = get_scientific_literature_template()
        result = extract_entities(
            text=sample_text,
            entity_schema=basic_schema,
            llm_model_name="gpt-4",
            prompt_template=template
        )

        # Verify scientific context in prompt
        prompt_content = mock_llm_request.call_args[0][0]
        
        assert "scientific" in prompt_content.lower()
        assert "research" in prompt_content.lower() or "literature" in prompt_content.lower()

        assert len(result) == 6

    @pytest.mark.parametrize("domain", ["metabolomics", "genetics", "plant_biology"])
    @patch('src.llm_extraction.ner._make_llm_request')
    def test_domain_specific_template_integration(self, mock_llm_request, domain, sample_text, mock_successful_response):
        """Test domain-specific templates with mock responses."""
        mock_llm_request.return_value = mock_successful_response

        template = get_domain_specific_template(domain)
        schema = get_schema_by_domain(domain)
        
        result = extract_entities(
            text=sample_text,
            entity_schema=schema,
            llm_model_name="gpt-4",
            prompt_template=template
        )

        # Verify domain-specific content in prompt
        prompt_content = mock_llm_request.call_args[0][0]
        
        assert domain.lower() in prompt_content.lower() or any(
            entity_type in prompt_content for entity_type in schema.keys()
        )

        assert len(result) == 6


class TestFewShotTemplateIntegration(TestPromptTemplateIntegration):
    """Test few-shot template integration with mock LLM calls."""

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_few_shot_basic_template_integration(self, mock_llm_request, sample_text, basic_schema, few_shot_examples, mock_successful_response):
        """Test basic few-shot template with examples."""
        mock_llm_request.return_value = mock_successful_response

        template = get_few_shot_basic_template()
        result = extract_entities(
            text=sample_text,
            entity_schema=basic_schema,
            llm_model_name="gpt-4",
            prompt_template=template,
            few_shot_examples=few_shot_examples
        )

        # Verify examples in prompt
        prompt_content = mock_llm_request.call_args[0][0]
        
        assert "Anthocyanins" in prompt_content  # From example
        assert "grape" in prompt_content  # From example
        assert "EXAMPLES:" in prompt_content or "Learning examples:" in prompt_content

        assert len(result) == 6

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_few_shot_detailed_template_integration(self, mock_llm_request, sample_text, basic_schema, few_shot_examples, mock_successful_response):
        """Test detailed few-shot template with examples."""
        mock_llm_request.return_value = mock_successful_response

        template = get_few_shot_detailed_template()
        result = extract_entities(
            text=sample_text,
            entity_schema=basic_schema,
            llm_model_name="gpt-4",
            prompt_template=template,
            few_shot_examples=few_shot_examples
        )

        # Verify detailed few-shot content
        prompt_content = mock_llm_request.call_args[0][0]
        
        assert "LEARNING EXAMPLES:" in prompt_content or "examples" in prompt_content.lower()
        assert "start" in prompt_content and "end" in prompt_content  # Check for position fields
        assert "Anthocyanins" in prompt_content  # Verify examples are included

        assert len(result) == 6

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_few_shot_precision_template_integration(self, mock_llm_request, sample_text, basic_schema, few_shot_examples, mock_successful_response):
        """Test precision-focused few-shot template."""
        mock_llm_request.return_value = mock_successful_response

        template = get_few_shot_precision_template()
        result = extract_entities(
            text=sample_text,
            entity_schema=basic_schema,
            llm_model_name="gpt-4",
            prompt_template=template,
            few_shot_examples=few_shot_examples
        )

        # Verify precision focus with examples
        prompt_content = mock_llm_request.call_args[0][0]
        
        assert "high confidence" in prompt_content.lower() or "precise" in prompt_content.lower()
        assert "Anthocyanins" in prompt_content  # Examples present

        assert len(result) == 6

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_few_shot_recall_template_integration(self, mock_llm_request, sample_text, basic_schema, few_shot_examples, mock_successful_response):
        """Test recall-focused few-shot template."""
        mock_llm_request.return_value = mock_successful_response

        template = get_few_shot_recall_template()
        result = extract_entities(
            text=sample_text,
            entity_schema=basic_schema,
            llm_model_name="gpt-4",
            prompt_template=template,
            few_shot_examples=few_shot_examples
        )

        # Verify recall focus with examples
        prompt_content = mock_llm_request.call_args[0][0]
        
        assert "comprehensive" in prompt_content.lower() or "all possible" in prompt_content.lower()
        assert "PAL" in prompt_content  # From examples

        assert len(result) == 6

    @pytest.mark.parametrize("domain", ["metabolomics", "genetics", "plant_biology"])
    @patch('src.llm_extraction.ner._make_llm_request')
    def test_few_shot_domain_template_integration(self, mock_llm_request, domain, sample_text, few_shot_examples, mock_successful_response):
        """Test domain-specific few-shot templates."""
        mock_llm_request.return_value = mock_successful_response

        template = get_few_shot_domain_template(domain)
        schema = get_schema_by_domain(domain)
        
        result = extract_entities(
            text=sample_text,
            entity_schema=schema,
            llm_model_name="gpt-4",
            prompt_template=template,
            few_shot_examples=few_shot_examples
        )

        # Verify domain-specific few-shot content
        prompt_content = mock_llm_request.call_args[0][0]
        
        assert domain.lower() in prompt_content.lower() or "plant" in prompt_content.lower()
        assert "EXAMPLES:" in prompt_content or "examples" in prompt_content.lower()

        assert len(result) == 6


class TestOutputFormatValidation(TestPromptTemplateIntegration):
    """Test output format validation with various mock responses."""

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_valid_output_format_parsing(self, mock_llm_request, sample_text, basic_schema, mock_successful_response):
        """Test parsing of valid output format."""
        mock_llm_request.return_value = mock_successful_response

        template = get_basic_zero_shot_template()
        result = extract_entities(
            text=sample_text,
            entity_schema=basic_schema,
            llm_model_name="gpt-4",
            prompt_template=template
        )

        # Verify all required fields present
        for entity in result:
            assert 'text' in entity
            assert 'label' in entity
            assert 'start' in entity
            assert 'end' in entity
            assert 'confidence' in entity
            
            # Verify field types
            assert isinstance(entity['text'], str)
            assert isinstance(entity['label'], str)
            assert isinstance(entity['start'], int)
            assert isinstance(entity['end'], int)
            assert isinstance(entity['confidence'], (int, float))
            
            # Verify field ranges
            assert entity['start'] >= 0
            assert entity['end'] >= entity['start']
            assert 0.0 <= entity['confidence'] <= 1.0

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_empty_response_handling(self, mock_llm_request, sample_text, basic_schema, mock_empty_response):
        """Test handling of empty entity response."""
        mock_llm_request.return_value = mock_empty_response

        template = get_basic_zero_shot_template()
        result = extract_entities(
            text=sample_text,
            entity_schema=basic_schema,
            llm_model_name="gpt-4",
            prompt_template=template
        )

        assert result == []

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_malformed_response_handling(self, mock_llm_request, sample_text, basic_schema, mock_malformed_response):
        """Test handling of malformed LLM response."""
        mock_llm_request.return_value = mock_malformed_response

        template = get_basic_zero_shot_template()
        
        with pytest.raises(LLMAPIError, match="missing 'entities' key"):
            extract_entities(
                text=sample_text,
                entity_schema=basic_schema,
                llm_model_name="gpt-4",
                prompt_template=template
            )

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_overlapping_entities_handling(self, mock_llm_request, sample_text, basic_schema):
        """Test handling of overlapping entities in response."""
        overlapping_response = {
            "entities": [
                {
                    "text": "Arabidopsis thaliana",
                    "label": "SPECIES",
                    "start": 67,
                    "end": 87,
                    "confidence": 0.98
                },
                {
                    "text": "thaliana",
                    "label": "SPECIES",  # Overlapping with above
                    "start": 79,
                    "end": 87,
                    "confidence": 0.85
                }
            ]
        }

        mock_llm_request.return_value = overlapping_response

        template = get_detailed_zero_shot_template()
        result = extract_entities(
            text=sample_text,
            entity_schema=basic_schema,
            llm_model_name="gpt-4",
            prompt_template=template
        )

        # Should handle overlapping entities gracefully
        assert len(result) == 2
        assert all('text' in entity for entity in result)

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_invalid_entity_fields_handling(self, mock_llm_request, sample_text, basic_schema):
        """Test handling of invalid entity field values."""
        invalid_response = {
            "entities": [
                {
                    "text": "Flavonoids",
                    "label": "METABOLITE",
                    "start": -1,  # Invalid negative start
                    "end": 19,
                    "confidence": 0.95
                },
                {
                    "text": "quercetin",
                    "label": "METABOLITE",
                    "start": 136,
                    "end": 130,  # Invalid: end < start
                    "confidence": 0.92
                },
                {
                    "text": "kaempferol",
                    "label": "METABOLITE",
                    "start": 150,
                    "end": 160,
                    "confidence": 1.5  # Invalid: confidence > 1.0
                }
            ]
        }

        mock_llm_request.return_value = invalid_response

        template = get_basic_zero_shot_template()
        
        with pytest.raises(LLMAPIError, match="Invalid field"):
            extract_entities(
                text=sample_text,
                entity_schema=basic_schema,
                llm_model_name="gpt-4",
                prompt_template=template
            )


class TestErrorHandlingIntegration(TestPromptTemplateIntegration):
    """Test error handling in template integration."""

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_api_timeout_handling(self, mock_llm_request, sample_text, basic_schema):
        """Test handling of API timeout errors."""
        mock_llm_request.side_effect = Timeout("Request timed out")

        template = get_basic_zero_shot_template()
        
        with pytest.raises(LLMAPIError, match="Request timed out"):
            extract_entities(
                text=sample_text,
                entity_schema=basic_schema,
                llm_model_name="gpt-4",
                prompt_template=template
            )

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_api_rate_limit_handling(self, mock_llm_request, sample_text, basic_schema):
        """Test handling of API rate limit errors."""
        mock_llm_request.side_effect = RateLimitError("Rate limit exceeded")

        template = get_basic_zero_shot_template()
        
        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            extract_entities(
                text=sample_text,
                entity_schema=basic_schema,
                llm_model_name="gpt-4",
                prompt_template=template
            )

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_api_http_error_handling(self, mock_llm_request, sample_text, basic_schema):
        """Test handling of HTTP errors."""
        mock_llm_request.side_effect = LLMAPIError("HTTP error occurred: Internal server error")

        template = get_basic_zero_shot_template()
        
        with pytest.raises(LLMAPIError, match="HTTP error occurred"):
            extract_entities(
                text=sample_text,
                entity_schema=basic_schema,
                llm_model_name="gpt-4",
                prompt_template=template
            )

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_retry_logic_integration(self, mock_llm_request, sample_text, basic_schema, mock_successful_response):
        """Test retry logic with eventual success."""
        # First two calls fail, third succeeds
        mock_llm_request.side_effect = [
            HTTPError("Server error"),
            HTTPError("Server error"),
            mock_successful_response
        ]

        template = get_basic_zero_shot_template()
        result = extract_entities(
            text=sample_text,
            entity_schema=basic_schema,
            llm_model_name="gpt-4",
            prompt_template=template
        )

        # Should succeed after retries
        assert len(result) == 6
        assert mock_llm_request.call_count == 3

    def test_invalid_template_placeholder_handling(self, sample_text, basic_schema):
        """Test handling of invalid template placeholders."""
        invalid_template = "Extract entities from {invalid_placeholder} using {schema}"
        
        with pytest.raises(Exception):  # Should fail during template formatting
            extract_entities(
                text=sample_text,
                entity_schema=basic_schema,
                llm_model_name="gpt-4",
                prompt_template=invalid_template
            )

    def test_missing_required_template_placeholders(self, sample_text, basic_schema):
        """Test handling of templates missing required placeholders."""
        incomplete_template = "Extract entities from the text."  # Missing {text} and {schema}
        
        # This should work but produce a poorly formatted prompt
        # The actual validation would depend on implementation details
        with patch('src.llm_extraction.ner._make_llm_request') as mock_llm_request:
            mock_llm_request.return_value = {"entities": []}
            
            result = extract_entities(
                text=sample_text,
                entity_schema=basic_schema,
                llm_model_name="gpt-4",
                prompt_template=incomplete_template
            )
            
            # Should handle gracefully with empty results
            assert result == []


class TestTemplateUtilityIntegration(TestPromptTemplateIntegration):
    """Test template utility functions with integration."""

    def test_generate_synthetic_examples_integration(self, basic_schema):
        """Test synthetic example generation for template integration."""
        examples = generate_synthetic_examples(list(basic_schema.keys()), num_examples=3)
        
        assert len(examples) >= 3  # Should generate examples for entity types
        for example in examples:
            assert 'text' in example
            assert 'entities' in example
            assert isinstance(example['entities'], list)
            
            for entity in example['entities']:
                assert entity['label'] in basic_schema.keys()

    def test_select_examples_integration(self, basic_schema):
        """Test example selection for template integration."""
        entity_types = list(basic_schema.keys())
        examples = select_examples(entity_types, strategy="balanced", max_examples=4)
        
        assert len(examples) <= 4
        for example in examples:
            assert 'text' in example
            assert 'entities' in example

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_context_aware_examples_integration(self, mock_llm_request, sample_text, basic_schema, mock_successful_response):
        """Test context-aware example selection integration."""
        mock_llm_request.return_value = mock_successful_response

        # Test with context-aware examples
        examples = get_context_aware_examples(sample_text, basic_schema, max_examples=2)
        
        template = get_few_shot_detailed_template()
        result = extract_entities(
            text=sample_text,
            entity_schema=basic_schema,
            llm_model_name="gpt-4",
            prompt_template=template,
            few_shot_examples=examples
        )

        assert len(result) == 6

    def test_format_examples_for_prompt_integration(self, few_shot_examples):
        """Test example formatting for prompt integration."""
        formatted_examples = format_examples_for_prompt(few_shot_examples)
        
        assert isinstance(formatted_examples, str)
        assert "Anthocyanins" in formatted_examples
        assert "METABOLITE" in formatted_examples
        assert "start" in formatted_examples
        assert "end" in formatted_examples

    def test_domain_examples_integration(self):
        """Test domain-specific example retrieval."""
        examples = get_examples_by_domain("metabolomics", max_examples=3)
        
        assert len(examples) <= 3
        for example in examples:
            assert 'text' in example
            assert 'entities' in example
            
            # Should contain metabolomics-relevant entities
            entity_labels = [e['label'] for e in example['entities']]
            metabolomics_entities = ['METABOLITE', 'COMPOUND', 'SPECIES', 'PLANT_PART', 'PHENOLIC_COMPOUND', 'FLAVONOID']
            assert any(label in metabolomics_entities for label in entity_labels), f"Found labels: {entity_labels}"


class TestTemplateValidationIntegration(TestPromptTemplateIntegration):
    """Test template validation with integration."""

    def test_template_validation_all_templates(self):
        """Test validation of all available templates."""
        templates = list_available_templates()
        
        for template_name in templates:
            template = get_template_by_name(template_name)
            
            # Basic validation
            assert isinstance(template, str)
            assert len(template) > 50  # Should be reasonably detailed
            
            # Should contain essential placeholders
            has_text_placeholder = '{text}' in template
            has_schema_placeholder = '{schema}' in template
            has_examples_placeholder = '{examples}' in template
            
            # At minimum should have text and schema
            assert has_text_placeholder
            assert has_schema_placeholder
            
            # Few-shot templates should have examples placeholder
            if 'few_shot' in template_name.lower():
                assert has_examples_placeholder

    def test_template_placeholder_validation(self):
        """Test template placeholder validation."""
        # Test actual templates from the system
        basic_template = get_basic_zero_shot_template()
        detailed_template = get_detailed_zero_shot_template()
        
        # Basic checks that these templates have required placeholders
        assert '{text}' in basic_template
        assert '{schema}' in basic_template
        assert '{text}' in detailed_template
        assert '{schema}' in detailed_template
        
        # Invalid template should fail validation
        invalid_template = "Extract entities from {invalid} using {wrong}."
        with pytest.raises(InvalidTemplateError):
            validate_template(invalid_template)

    def test_template_format_validation(self):
        """Test template format requirements."""
        # Template must be string
        with pytest.raises(InvalidTemplateError):
            validate_template(123)
        
        # Template cannot be empty
        with pytest.raises(InvalidTemplateError):
            validate_template("")
        
        # Template must have minimum content
        with pytest.raises(InvalidTemplateError):
            validate_template("short")

    @pytest.mark.parametrize("template_type", [
        TemplateType.BASIC,
        TemplateType.DETAILED,
        TemplateType.PRECISION,
        TemplateType.RECALL,
        TemplateType.SCIENTIFIC
    ])
    def test_zero_shot_template_type_validation(self, template_type):
        """Test validation of different zero-shot template types."""
        if template_type == TemplateType.BASIC:
            template = get_basic_zero_shot_template()
        elif template_type == TemplateType.DETAILED:
            template = get_detailed_zero_shot_template()
        elif template_type == TemplateType.PRECISION:
            template = get_precision_focused_template()
        elif template_type == TemplateType.RECALL:
            template = get_recall_focused_template()
        elif template_type == TemplateType.SCIENTIFIC:
            template = get_scientific_literature_template()
        
        assert validate_template(template)
        assert '{text}' in template
        assert '{schema}' in template

    @pytest.mark.parametrize("template_type", [
        "basic",
        "detailed"
    ])
    def test_few_shot_template_type_validation(self, template_type):
        """Test validation of different few-shot template types."""
        template = get_few_shot_template(template_type)
        
        # Check that template has required placeholders
        assert '{text}' in template
        assert '{schema}' in template  
        assert '{examples}' in template
        
        # Template should be substantial
        assert len(template) > 100


class TestPerformanceIntegration(TestPromptTemplateIntegration):
    """Test performance aspects of template integration."""

    @patch('src.llm_extraction.ner._make_llm_request')
    def test_large_text_performance(self, mock_llm_request, basic_schema, mock_successful_response):
        """Test template integration with large text inputs."""
        # Create large text (simulate scientific paper)
        large_text = """
        Flavonoids are secondary metabolites widely distributed in plants.
        """ * 1000  # Repeat to create large text
        
        mock_llm_request.return_value = mock_successful_response

        template = get_basic_zero_shot_template()
        
        import time
        start_time = time.time()
        result = extract_entities(
            text=large_text,
            entity_schema=basic_schema,
            llm_model_name="gpt-4",
            prompt_template=template
        )
        end_time = time.time()
        
        # Should complete within reasonable time (mostly network/mock time)
        assert end_time - start_time < 5.0  # 5 seconds max
        assert len(result) == 6

    def test_complex_schema_performance(self):
        """Test template integration with complex schemas."""
        complex_schema = get_plant_metabolomics_schema()  # 117 entity types
        
        template = get_detailed_zero_shot_template()
        
        import time
        start_time = time.time()
        
        # Just test template formatting performance
        formatted_prompt = _format_prompt(
            template=template,
            text="Test text with metabolites in Arabidopsis leaves.",
            schema=complex_schema
        )
        
        end_time = time.time()
        
        # Should format quickly even with large schema
        assert end_time - start_time < 1.0  # 1 second max
        assert len(formatted_prompt) > 1000  # Should be substantial
        assert "METABOLITE" in formatted_prompt
        assert "FLAVONOID" in formatted_prompt

    def test_many_examples_performance(self):
        """Test performance with many few-shot examples."""
        many_examples = []
        for i in range(20):  # 20 examples
            many_examples.append({
                "text": f"Example text {i} with metabolite compound in plant tissue.",
                "entities": [
                    {"text": "metabolite", "label": "METABOLITE", "start": 20, "end": 30},
                    {"text": "compound", "label": "COMPOUND", "start": 31, "end": 39},
                    {"text": "plant", "label": "SPECIES", "start": 43, "end": 48},
                    {"text": "tissue", "label": "PLANT_PART", "start": 49, "end": 55}
                ]
            })
        
        template = get_few_shot_detailed_template()
        basic_schema = get_basic_schema()
        
        import time
        start_time = time.time()
        
        formatted_prompt = _format_prompt(
            template=template,
            text="Test text.",
            schema=basic_schema,
            examples=many_examples
        )
        
        end_time = time.time()
        
        # Should handle many examples efficiently  
        assert end_time - start_time < 2.0  # 2 seconds max
        assert "Example text 0" in formatted_prompt
        assert "Example text 19" in formatted_prompt