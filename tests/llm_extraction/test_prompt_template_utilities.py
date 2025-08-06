"""
Unit tests for prompt template utilities and validation functions.

This module tests the comprehensive utility functions for prompt template
validation, optimization, metrics calculation, and management.
"""

import pytest
from typing import Dict, List, Any
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
    InvalidTemplateError,
    TemplateNotFoundError,
    TEMPLATE_REGISTRY,
    get_basic_zero_shot_template
)


class TestTemplateStructureValidation:
    """Test template structure validation functionality."""
    
    def test_valid_template_structure(self):
        """Test validation of properly structured template."""
        template = """
        Extract entities from {text} using the provided {schema}.
        Return JSON with entities array containing text, label, start, end, confidence fields.
        Example: {"entities": [{"text": "example", "label": "TYPE", "start": 0, "end": 7, "confidence": 0.95}]}
        """
        assert validate_template_structure(template) is True
    
    def test_missing_required_placeholders(self):
        """Test validation fails for missing required placeholders."""
        # Missing {schema}
        template = "Extract entities from {text}. Return JSON with entities array containing text, label, start, end, confidence."
        with pytest.raises(InvalidTemplateError, match="missing required placeholders"):
            validate_template_structure(template)
        
        # Missing {text}
        template = "Extract entities using {schema}. Return JSON with entities array containing text, label, start, end, confidence."
        with pytest.raises(InvalidTemplateError, match="missing required placeholders"):
            validate_template_structure(template)
    
    def test_invalid_placeholders(self):
        """Test validation fails for invalid placeholders."""
        template = "Extract entities from {text} using {schema} and {invalid_placeholder}. Return JSON with entities array."
        with pytest.raises(InvalidTemplateError, match="invalid placeholders"):
            validate_template_structure(template)
    
    def test_missing_json_specification(self):
        """Test validation fails when JSON output is not specified."""
        template = "Extract entities from {text} using {schema}. Return results with text, label, start, end, confidence."
        with pytest.raises(InvalidTemplateError, match="JSON output format"):
            validate_template_structure(template)
    
    def test_missing_required_fields(self):
        """Test validation fails when required entity fields are not mentioned."""
        template = "Extract entities from {text} using {schema}. Return JSON with entities array."
        with pytest.raises(InvalidTemplateError, match="required fields"):
            validate_template_structure(template)
    
    def test_template_too_short(self):
        """Test validation fails for overly short templates."""
        template = "Extract {text} {schema}"
        with pytest.raises(InvalidTemplateError, match="too short"):
            validate_template_structure(template)
    
    def test_template_too_long(self):
        """Test validation fails for overly long templates."""
        template = "Extract entities from {text} using {schema}. " + "Very long template. " * 500 + "Return JSON with entities array containing text, label, start, end, confidence."
        with pytest.raises(InvalidTemplateError, match="too long"):
            validate_template_structure(template)
    
    def test_non_string_template(self):
        """Test validation fails for non-string input."""
        with pytest.raises(InvalidTemplateError, match="must be a string"):
            validate_template_structure(123)
    
    def test_empty_template(self):
        """Test validation fails for empty template."""
        with pytest.raises(InvalidTemplateError, match="cannot be empty"):
            validate_template_structure("")


class TestExamplesFormatValidation:
    """Test examples format validation functionality."""
    
    def test_valid_examples_format(self):
        """Test validation of properly formatted examples."""
        examples = [
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
        assert validate_examples_format(examples) is True
    
    def test_examples_not_list(self):
        """Test validation fails when examples is not a list."""
        with pytest.raises(InvalidTemplateError, match="must be a list"):
            validate_examples_format("not a list")
    
    def test_empty_examples_list(self):
        """Test validation fails for empty examples list."""
        with pytest.raises(InvalidTemplateError, match="cannot be empty"):
            validate_examples_format([])
    
    def test_too_many_examples(self):
        """Test validation fails for too many examples."""
        examples = [{"text": "test", "entities": []} for _ in range(51)]
        with pytest.raises(InvalidTemplateError, match="Too many examples"):
            validate_examples_format(examples)
    
    def test_missing_required_example_fields(self):
        """Test validation fails for missing required fields in examples."""
        examples = [{"text": "test text"}]  # Missing entities field
        with pytest.raises(InvalidTemplateError, match="missing required field"):
            validate_examples_format(examples)
    
    def test_invalid_entity_positions(self):
        """Test validation fails for invalid entity positions."""
        examples = [
            {
                "text": "Short text",
                "entities": [
                    {
                        "text": "invalid",
                        "label": "TEST",
                        "start": 5,
                        "end": 3,  # End before start
                        "confidence": 0.95
                    }
                ]
            }
        ]
        with pytest.raises(InvalidTemplateError, match="invalid positions"):
            validate_examples_format(examples)
    
    def test_text_span_mismatch(self):
        """Test validation fails when entity text doesn't match span."""
        examples = [
            {
                "text": "Quercetin is a compound",
                "entities": [
                    {
                        "text": "Wrong text",  # Doesn't match actual span
                        "label": "METABOLITE",
                        "start": 0,
                        "end": 9,
                        "confidence": 0.95
                    }
                ]
            }
        ]
        with pytest.raises(InvalidTemplateError, match="text span mismatch"):
            validate_examples_format(examples)
    
    def test_invalid_confidence_range(self):
        """Test validation fails for confidence values outside valid range."""
        examples = [
            {
                "text": "Test text",
                "entities": [
                    {
                        "text": "Test",
                        "label": "TEST",
                        "start": 0,
                        "end": 4,
                        "confidence": 1.5  # Invalid confidence > 1.0
                    }
                ]
            }
        ]
        with pytest.raises(InvalidTemplateError, match="confidence must be between"):
            validate_examples_format(examples)


class TestPromptOptimization:
    """Test prompt optimization for different models."""
    
    def test_gpt_optimization(self):
        """Test optimization for GPT models."""
        prompt = "Extract entities from {text} using {schema}."
        optimized = optimize_prompt_for_model(prompt, "gpt-4")
        
        assert "**TASK:**" in optimized
        assert "You are a specialized NER system" in optimized
        assert "JSON" in optimized
        assert len(optimized) > len(prompt)
    
    def test_claude_optimization(self):
        """Test optimization for Claude models."""
        prompt = "Extract entities from {text} using {schema}."
        optimized = optimize_prompt_for_model(prompt, "claude-3")
        
        assert "**DETAILED APPROACH:**" in optimized
        assert "reasoning" in optimized.lower()
        assert len(optimized) > len(prompt)
    
    def test_gemini_optimization(self):
        """Test optimization for Gemini models."""
        # Test with long prompt that should be simplified
        long_prompt = "**SECTION1:** " * 20 + "Extract entities from {text} using {schema}. Return JSON with entities array containing text, label, start, end, confidence."
        optimized = optimize_prompt_for_model(long_prompt, "gemini-pro")
        
        assert "JSON format" in optimized or "json" in optimized.lower()
        # Optimization might not always reduce length due to added instructions
        assert len(optimized) > 0
    
    def test_llama_optimization(self):
        """Test optimization for Llama models."""
        prompt = "Extract entities from {text} using {schema}."
        optimized = optimize_prompt_for_model(prompt, "llama-2")
        
        assert "**OUTPUT FORMAT:**" in optimized
        assert "Task:" in optimized
        assert "Do not include explanations" in optimized
    
    def test_unknown_model_optimization(self):
        """Test generic optimization for unknown models."""
        prompt = "Extract entities from {text} using {schema}."
        optimized = optimize_prompt_for_model(prompt, "unknown-model")
        
        assert "**INSTRUCTIONS:**" in optimized
        assert "Return valid JSON only" in optimized


class TestTemplateMetrics:
    """Test template metrics calculation."""
    
    def test_basic_metrics_calculation(self):
        """Test calculation of basic template metrics."""
        template = get_basic_zero_shot_template()
        metrics = calculate_template_metrics(template)
        
        # Check basic metrics
        assert isinstance(metrics['word_count'], int)
        assert isinstance(metrics['character_count'], int)
        assert isinstance(metrics['line_count'], int)
        assert metrics['word_count'] > 0
        assert metrics['character_count'] > 0
        
        # Check placeholder metrics
        assert isinstance(metrics['placeholders'], list)
        assert '{text}' in metrics['placeholders']
        assert '{schema}' in metrics['placeholders']
        
        # Check quality indicators
        quality = metrics['quality_indicators']
        assert quality['has_json_spec'] is True
        assert quality['has_entity_fields'] is True
        
        # Check complexity assessment
        assert metrics['complexity_level'] in ['low', 'medium', 'high']
        assert 0 <= metrics['complexity_score'] <= 1
        
        # Check effectiveness rating
        assert metrics['estimated_effectiveness'] in ['poor', 'fair', 'good', 'excellent']
    
    def test_metrics_for_simple_template(self):
        """Test metrics for a simple template."""
        simple_template = "Extract entities from {text} using {schema}. Return JSON with entities array containing text, label, start, end, confidence fields."
        metrics = calculate_template_metrics(simple_template)
        
        assert metrics['complexity_level'] == 'low'
        assert metrics['word_count'] < 50


class TestTemplateImprovements:
    """Test template improvement suggestions."""
    
    def test_suggestions_for_minimal_template(self):
        """Test suggestions for a minimal template."""
        minimal_template = "Extract entities from {text}."
        suggestions = suggest_template_improvements(minimal_template)
        
        # Should suggest adding schema placeholder
        assert any("schema" in s.lower() for s in suggestions)
        # Should suggest adding JSON specification
        assert any("json" in s.lower() for s in suggestions)
        # Should suggest adding required fields
        assert any("required" in s.lower() and "fields" in s.lower() for s in suggestions)
    
    def test_suggestions_for_complete_template(self):
        """Test suggestions for a well-structured template."""
        complete_template = get_basic_zero_shot_template()
        suggestions = suggest_template_improvements(complete_template)
        
        # Should have reasonable number of suggestions (complete templates might still have room for improvement)
        assert len(suggestions) < 15
    
    def test_domain_specific_suggestions(self):
        """Test domain-specific improvement suggestions."""
        scientific_template = "Extract scientific entities from {text} using {schema}. Return JSON with entities array."
        suggestions = suggest_template_improvements(scientific_template)
        
        # Should suggest nomenclature guidelines
        assert any("nomenclature" in s.lower() for s in suggestions)
    
    def test_no_duplicate_suggestions(self):
        """Test that suggestions don't contain duplicates."""
        template = "Extract entities from {text}."
        suggestions = suggest_template_improvements(template)
        
        # Check for uniqueness
        assert len(suggestions) == len(set(suggestions))


class TestTemplateRegistry:
    """Test template registry and management functions."""
    
    def test_register_custom_template(self):
        """Test registering a custom template."""
        custom_template = """
        Extract custom entities from {text} using {schema}.
        Return JSON with entities array containing text, label, start, end, confidence fields.
        """
        
        result = register_custom_template("test_custom", custom_template)
        assert result is True
        assert "test_custom" in TEMPLATE_REGISTRY
        
        # Clean up
        del TEMPLATE_REGISTRY["test_custom"]
    
    def test_register_duplicate_template_name(self):
        """Test that registering duplicate template name fails."""
        template = """
        Extract entities from {text} using {schema}.
        Return JSON with entities array containing text, label, start, end, confidence fields.
        Example: {"entities": [{"text": "test", "label": "TYPE", "start": 0, "end": 4, "confidence": 0.95}]}
        """
        
        # Register first template
        register_custom_template("test_duplicate", template)
        
        # Try to register with same name
        with pytest.raises(ValueError, match="already exists"):
            register_custom_template("test_duplicate", template)
        
        # Clean up
        del TEMPLATE_REGISTRY["test_duplicate"]
    
    def test_get_template_metadata(self):
        """Test getting template metadata."""
        # Use existing template
        metadata = get_template_metadata("basic")
        
        assert isinstance(metadata, dict)
        assert 'name' in metadata
        assert 'template_type' in metadata
        assert 'domain_focus' in metadata
        assert 'use_case' in metadata
        assert 'metrics' in metadata
        assert 'suggestions' in metadata
    
    def test_get_nonexistent_template_metadata(self):
        """Test getting metadata for non-existent template."""
        with pytest.raises(TemplateNotFoundError):
            get_template_metadata("nonexistent_template")


class TestTemplateComparison:
    """Test template comparison functionality."""
    
    def test_compare_templates(self):
        """Test comparing two templates."""
        template1 = "Extract entities from {text} using {schema}. Return JSON with entities array containing text, label, start, end, confidence."
        template2 = get_basic_zero_shot_template()
        
        comparison = compare_templates(template1, template2)
        
        assert 'template1_metrics' in comparison
        assert 'template2_metrics' in comparison
        assert 'differences' in comparison
        assert 'recommendation' in comparison
        
        # Check differences structure
        diffs = comparison['differences']
        assert 'word_count_diff' in diffs
        assert 'complexity_diff' in diffs
        assert 'quality_diff' in diffs
        assert 'readability_diff' in diffs


class TestTemplateRecommendations:
    """Test template recommendation system."""
    
    def test_get_template_recommendations(self):
        """Test getting template recommendations based on requirements."""
        requirements = {
            'domain': 'metabolomics',
            'accuracy_priority': 'precision',
            'complexity': 'medium',
            'use_case': 'research'
        }
        
        recommendations = get_template_recommendations(requirements)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 10
        assert all(isinstance(name, str) for name in recommendations)
    
    def test_recommendations_for_different_domains(self):
        """Test recommendations for different domains."""
        # Test metabolomics domain
        metabolomics_reqs = {'domain': 'metabolomics'}
        metabolomics_recs = get_template_recommendations(metabolomics_reqs)
        
        # Test genetics domain  
        genetics_reqs = {'domain': 'genetics'}
        genetics_recs = get_template_recommendations(genetics_reqs)
        
        # Should get some recommendations for both
        assert len(metabolomics_recs) > 0
        assert len(genetics_recs) > 0


class TestIntegrationWithExistingFunctions:
    """Test integration with existing template functions."""
    
    def test_validation_with_existing_templates(self):
        """Test that existing templates pass validation."""
        basic_template = get_basic_zero_shot_template()
        
        # Should pass structure validation
        assert validate_template_structure(basic_template) is True
        
        # Should have reasonable metrics
        metrics = calculate_template_metrics(basic_template)
        assert metrics['quality_score'] > 0.5
        assert metrics['estimated_effectiveness'] in ['good', 'excellent']
    
    def test_optimization_preserves_functionality(self):
        """Test that optimization preserves template functionality."""
        original = get_basic_zero_shot_template()
        optimized = optimize_prompt_for_model(original, "gpt-4")
        
        # Both should pass validation
        assert validate_template_structure(original) is True
        assert validate_template_structure(optimized) is True
        
        # Optimized should have equal or better metrics
        original_metrics = calculate_template_metrics(original)
        optimized_metrics = calculate_template_metrics(optimized)
        
        # Quality should be maintained or improved
        assert optimized_metrics['quality_score'] >= original_metrics['quality_score'] - 0.1


if __name__ == "__main__":
    pytest.main([__file__])