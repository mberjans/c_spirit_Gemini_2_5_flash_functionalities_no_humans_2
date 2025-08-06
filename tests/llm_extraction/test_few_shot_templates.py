"""
Tests for few-shot prompt templates and synthetic example generation.

This module tests the comprehensive few-shot functionality including:
- Synthetic example generation for all 117 entity types
- Few-shot template variants (basic, detailed, precision, recall, domain-specific)
- Example selection algorithms (random, balanced, targeted)
- Integration with the extract_entities function
- Context-aware example selection
- Domain-specific template and example handling
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

from src.llm_extraction.prompt_templates import (
    # Template getters
    get_few_shot_template,
    get_few_shot_basic_template,
    get_few_shot_detailed_template,
    get_few_shot_precision_template,
    get_few_shot_recall_template,
    get_few_shot_scientific_template,
    get_few_shot_domain_template,
    
    # Example generation and selection
    generate_synthetic_examples,
    select_examples,
    get_examples_by_domain,
    format_examples_for_prompt,
    get_context_aware_examples,
    
    # Constants and databases
    SYNTHETIC_EXAMPLES_DATABASE,
    FEW_SHOT_BASIC_TEMPLATE,
    FEW_SHOT_DETAILED_TEMPLATE,
    FEW_SHOT_PRECISION_TEMPLATE,
    FEW_SHOT_RECALL_TEMPLATE,
    FEW_SHOT_METABOLOMICS_TEMPLATE,
    
    # Exceptions
    TemplateNotFoundError
)

from src.llm_extraction.ner import (
    extract_entities_few_shot,
    extract_entities_with_custom_examples,
    extract_entities_domain_specific,
    extract_entities_adaptive
)

from src.llm_extraction.entity_schemas import get_plant_metabolomics_schema


class TestSyntheticExampleDatabase:
    """Test the synthetic examples database."""
    
    def test_database_completeness(self):
        """Test that database contains examples for all major entity types."""
        # Check that we have examples for key entity categories
        expected_categories = [
            "METABOLITE", "COMPOUND", "PHENOLIC_COMPOUND", "FLAVONOID",
            "SPECIES", "PLANT_SPECIES", "ORGANISM",
            "PLANT_PART", "PLANT_ORGAN", "ROOT", "LEAF", "STEM",
            "EXPERIMENTAL_CONDITION", "STRESS_CONDITION", "TREATMENT",
            "MOLECULAR_TRAIT", "GENE_EXPRESSION", "ENZYME_ACTIVITY",
            "PLANT_TRAIT", "MORPHOLOGICAL_TRAIT", "GROWTH_TRAIT",
            "GENE", "PROTEIN", "ENZYME", "ANALYTICAL_METHOD"
        ]
        
        for category in expected_categories:
            assert category in SYNTHETIC_EXAMPLES_DATABASE, f"Missing examples for {category}"
            assert len(SYNTHETIC_EXAMPLES_DATABASE[category]) > 0, f"No examples for {category}"
    
    def test_example_format_validity(self):
        """Test that all examples follow the correct format."""
        for entity_type, examples in SYNTHETIC_EXAMPLES_DATABASE.items():
            for i, example in enumerate(examples):
                # Check required fields
                assert "text" in example, f"Missing 'text' field in {entity_type} example {i}"
                assert "entities" in example, f"Missing 'entities' field in {entity_type} example {i}"
                
                # Check text field
                assert isinstance(example["text"], str), f"Text should be string in {entity_type} example {i}"
                assert len(example["text"].strip()) > 0, f"Empty text in {entity_type} example {i}"
                
                # Check entities field
                assert isinstance(example["entities"], list), f"Entities should be list in {entity_type} example {i}"
                assert len(example["entities"]) > 0, f"No entities in {entity_type} example {i}"
                
                # Check each entity
                for j, entity in enumerate(example["entities"]):
                    required_fields = ["text", "label", "start", "end", "confidence"]
                    for field in required_fields:
                        assert field in entity, f"Missing '{field}' in {entity_type} example {i} entity {j}"
                    
                    # Check field types and ranges
                    assert isinstance(entity["text"], str), f"Entity text should be string"
                    assert isinstance(entity["label"], str), f"Entity label should be string"
                    assert isinstance(entity["start"], int), f"Entity start should be integer"
                    assert isinstance(entity["end"], int), f"Entity end should be integer"
                    assert isinstance(entity["confidence"], (int, float)), f"Confidence should be number"
                    
                    # Check ranges
                    assert entity["start"] >= 0, f"Start position should be non-negative"
                    assert entity["end"] > entity["start"], f"End should be greater than start"
                    assert 0.0 <= entity["confidence"] <= 1.0, f"Confidence should be between 0 and 1"
                    
                    # Check entity text matches the extracted span
                    extracted_text = example["text"][entity["start"]:entity["end"]]
                    assert extracted_text == entity["text"], f"Entity text mismatch: '{extracted_text}' vs '{entity['text']}'"


class TestExampleGeneration:
    """Test synthetic example generation functions."""
    
    def test_generate_synthetic_examples_basic(self):
        """Test basic example generation."""
        entity_types = ["METABOLITE", "SPECIES", "GENE"]
        examples = generate_synthetic_examples(entity_types, num_examples=2)
        
        assert isinstance(examples, list)
        assert len(examples) > 0
        
        # Check that we got examples for the requested types
        found_types = set()
        for example in examples:
            for entity in example["entities"]:
                found_types.add(entity["label"])
        
        for entity_type in entity_types:
            assert entity_type in found_types, f"No examples generated for {entity_type}"
    
    def test_generate_synthetic_examples_difficulty_levels(self):
        """Test example generation with different difficulty levels."""
        entity_types = ["METABOLITE", "COMPOUND"]
        
        # Test simple examples
        simple_examples = generate_synthetic_examples(
            entity_types, num_examples=2, difficulty_level="simple"
        )
        assert len(simple_examples) > 0
        
        # Test complex examples
        complex_examples = generate_synthetic_examples(
            entity_types, num_examples=2, difficulty_level="complex"
        )
        assert len(complex_examples) > 0
        
        # Test mixed examples
        mixed_examples = generate_synthetic_examples(
            entity_types, num_examples=2, difficulty_level="mixed"
        )
        assert len(mixed_examples) > 0
    
    def test_generate_synthetic_examples_invalid_types(self):
        """Test example generation with invalid entity types."""
        entity_types = ["INVALID_TYPE", "METABOLITE"]
        examples = generate_synthetic_examples(entity_types, num_examples=2)
        
        # Should still return examples for valid types
        assert len(examples) > 0
        
        # Should only contain valid entity types
        found_types = set()
        for example in examples:
            for entity in example["entities"]:
                found_types.add(entity["label"])
        
        assert "INVALID_TYPE" not in found_types
        assert "METABOLITE" in found_types


class TestExampleSelection:
    """Test example selection algorithms."""
    
    def test_select_examples_balanced(self):
        """Test balanced example selection strategy."""
        target_types = ["METABOLITE", "SPECIES", "GENE"]
        examples = select_examples(target_types, strategy="balanced", max_examples=6)
        
        assert isinstance(examples, list)
        assert len(examples) <= 6
        
        if examples:
            # Check that we have representation from multiple types
            found_types = set()
            for example in examples:
                for entity in example["entities"]:
                    found_types.add(entity["label"])
            
            # Should have at least some of the target types
            intersection = found_types.intersection(set(target_types))
            assert len(intersection) > 0
    
    def test_select_examples_high_confidence(self):
        """Test high confidence example selection."""
        target_types = ["METABOLITE", "COMPOUND"]
        examples = select_examples(
            target_types, 
            strategy="high_confidence", 
            max_examples=5
        )
        
        assert isinstance(examples, list)
        assert len(examples) <= 5
        
        if examples:
            # Check that examples have high average confidence
            for example in examples:
                avg_confidence = sum(e["confidence"] for e in example["entities"]) / len(example["entities"])
                # Should be reasonably high confidence
                assert avg_confidence >= 0.7
    
    def test_select_examples_with_confidence_filter(self):
        """Test example selection with confidence filtering."""
        target_types = ["METABOLITE", "SPECIES"]
        examples = select_examples(
            target_types,
            strategy="random",
            max_examples=5,
            confidence_filter=(0.9, 1.0)
        )
        
        # All entities should have confidence >= 0.9
        for example in examples:
            for entity in example["entities"]:
                assert entity["confidence"] >= 0.9
    
    def test_select_examples_diverse(self):
        """Test diverse example selection strategy."""
        target_types = ["METABOLITE", "SPECIES", "GENE", "PLANT_PART"]
        examples = select_examples(target_types, strategy="diverse", max_examples=8)
        
        assert isinstance(examples, list)
        assert len(examples) <= 8
        
        if examples:
            # Should have diverse entity types
            all_types = set()
            for example in examples:
                for entity in example["entities"]:
                    all_types.add(entity["label"])
            
            # Should have reasonable diversity
            assert len(all_types) >= min(3, len(examples))


class TestDomainSpecificExamples:
    """Test domain-specific example selection."""
    
    def test_get_examples_by_domain_metabolomics(self):
        """Test metabolomics domain example selection."""
        examples = get_examples_by_domain("metabolomics", max_examples=5)
        
        assert isinstance(examples, list)
        assert len(examples) <= 5
        
        if examples:
            # Should contain metabolomics-related entities
            metabolomics_types = {
                "METABOLITE", "COMPOUND", "PHENOLIC_COMPOUND", "FLAVONOID",
                "ALKALOID", "TERPENOID", "LIPID", "CARBOHYDRATE"
            }
            
            found_types = set()
            for example in examples:
                for entity in example["entities"]:
                    found_types.add(entity["label"])
            
            # Should have some overlap with metabolomics types
            assert len(found_types.intersection(metabolomics_types)) > 0
    
    def test_get_examples_by_domain_genetics(self):
        """Test genetics domain example selection."""
        examples = get_examples_by_domain("genetics", max_examples=4)
        
        assert isinstance(examples, list)
        assert len(examples) <= 4
        
        if examples:
            genetics_types = {
                "GENE", "PROTEIN", "ENZYME", "TRANSCRIPTION_FACTOR",
                "GENE_EXPRESSION", "MOLECULAR_TRAIT"
            }
            
            found_types = set()
            for example in examples:
                for entity in example["entities"]:
                    found_types.add(entity["label"])
            
            assert len(found_types.intersection(genetics_types)) > 0
    
    def test_get_examples_by_domain_invalid(self):
        """Test domain example selection with invalid domain."""
        examples = get_examples_by_domain("invalid_domain", max_examples=3)
        
        # Should still return examples (fallback)
        assert isinstance(examples, list)


class TestExampleFormatting:
    """Test example formatting for prompts."""
    
    def test_format_examples_for_prompt(self):
        """Test formatting examples for prompt inclusion."""
        examples = [
            {
                "text": "The leaves contain quercetin and kaempferol.",
                "entities": [
                    {"text": "quercetin", "label": "METABOLITE", "start": 20, "end": 29, "confidence": 0.95},
                    {"text": "kaempferol", "label": "METABOLITE", "start": 34, "end": 44, "confidence": 0.94}
                ]
            }
        ]
        
        formatted = format_examples_for_prompt(examples)
        
        assert isinstance(formatted, str)
        assert "Example 1:" in formatted
        assert "quercetin" in formatted
        assert "METABOLITE" in formatted
        assert "entities" in formatted.lower()
    
    def test_format_examples_empty(self):
        """Test formatting with empty examples."""
        formatted = format_examples_for_prompt([])
        assert formatted == ""


class TestContextAwareExamples:
    """Test context-aware example selection."""
    
    def test_get_context_aware_examples_metabolomics(self):
        """Test context-aware selection for metabolomics text."""
        text = "HPLC analysis revealed high concentrations of flavonoids and phenolic compounds in the extract."
        schema = {"METABOLITE": "Chemical compounds", "ANALYTICAL_METHOD": "Analysis techniques"}
        
        examples = get_context_aware_examples(text, schema, max_examples=4)
        
        assert isinstance(examples, list)
        assert len(examples) <= 4
    
    def test_get_context_aware_examples_genetics(self):
        """Test context-aware selection for genetics text."""
        text = "Gene expression analysis showed upregulation of transcription factors during stress response."
        schema = {"GENE": "Genetic elements", "PROTEIN": "Protein molecules"}
        
        examples = get_context_aware_examples(text, schema, max_examples=3)
        
        assert isinstance(examples, list)
        assert len(examples) <= 3


class TestFewShotTemplates:
    """Test few-shot template functionality."""
    
    def test_get_few_shot_basic_template(self):
        """Test basic few-shot template retrieval."""
        template = get_few_shot_basic_template()
        
        assert isinstance(template, str)
        assert "{schema}" in template
        assert "{text}" in template
        assert "{examples}" in template
        assert "EXAMPLES:" in template or "examples" in template.lower()
    
    def test_get_few_shot_detailed_template(self):
        """Test detailed few-shot template retrieval."""
        template = get_few_shot_detailed_template()
        
        assert isinstance(template, str)
        assert "{schema}" in template
        assert "{text}" in template
        assert "{examples}" in template
        assert len(template) > len(FEW_SHOT_BASIC_TEMPLATE)  # Should be more detailed
    
    def test_get_few_shot_precision_template(self):
        """Test precision-focused few-shot template."""
        template = get_few_shot_precision_template()
        
        assert isinstance(template, str)
        assert "precision" in template.lower()
        assert "high-confidence" in template.lower() or "confidence" in template.lower()
    
    def test_get_few_shot_recall_template(self):
        """Test recall-focused few-shot template."""
        template = get_few_shot_recall_template()
        
        assert isinstance(template, str)
        assert "recall" in template.lower() or "comprehensive" in template.lower()
    
    def test_get_few_shot_domain_template_valid(self):
        """Test domain-specific few-shot templates."""
        domains = ["metabolomics", "genetics", "plant_biology"]
        
        for domain in domains:
            template = get_few_shot_domain_template(domain)
            assert isinstance(template, str)
            assert "{schema}" in template
            assert "{text}" in template
            assert "{examples}" in template
    
    def test_get_few_shot_domain_template_invalid(self):
        """Test domain-specific template with invalid domain."""
        with pytest.raises(TemplateNotFoundError):
            get_few_shot_domain_template("invalid_domain")
    
    def test_get_few_shot_template_function(self):
        """Test the main few-shot template getter function."""
        # Test with different template types
        template_types = ["basic", "detailed", "precision", "recall", "scientific"]
        
        for template_type in template_types:
            template = get_few_shot_template(template_type)
            assert isinstance(template, str)
            assert "{schema}" in template
            assert "{text}" in template
            assert "{examples}" in template


class TestNERIntegration:
    """Test integration with NER functions."""
    
    @patch('src.llm_extraction.ner._make_llm_request')
    def test_extract_entities_few_shot_basic(self, mock_request):
        """Test basic few-shot entity extraction."""
        # Mock LLM response
        mock_response = {
            "entities": [
                {"text": "quercetin", "label": "METABOLITE", "start": 20, "end": 29, "confidence": 0.95}
            ]
        }
        mock_request.return_value = mock_response
        
        text = "The plant contains quercetin as a major flavonoid."
        schema = {"METABOLITE": "Chemical compounds and metabolites"}
        
        result = extract_entities_few_shot(
            text=text,
            entity_schema=schema,
            llm_model_name="test-model",
            template_type="basic",
            num_examples=2
        )
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == "quercetin"
        assert result[0]["label"] == "METABOLITE"
    
    @patch('src.llm_extraction.ner._make_llm_request')
    def test_extract_entities_with_custom_examples(self, mock_request):
        """Test entity extraction with custom examples."""
        mock_response = {
            "entities": [
                {"text": "anthocyanin", "label": "METABOLITE", "start": 15, "end": 26, "confidence": 0.92}
            ]
        }
        mock_request.return_value = mock_response
        
        text = "Red berries contain anthocyanin pigments."
        schema = {"METABOLITE": "Chemical compounds"}
        custom_examples = [
            {
                "text": "Leaves accumulate quercetin under stress.",
                "entities": [
                    {"text": "quercetin", "label": "METABOLITE", "start": 18, "end": 27, "confidence": 0.95}
                ]
            }
        ]
        
        result = extract_entities_with_custom_examples(
            text=text,
            entity_schema=schema,
            llm_model_name="test-model",
            examples=custom_examples,
            template_type="basic"
        )
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == "anthocyanin"
    
    @patch('src.llm_extraction.ner._make_llm_request')
    def test_extract_entities_domain_specific(self, mock_request):
        """Test domain-specific entity extraction."""
        mock_response = {
            "entities": [
                {"text": "CHS", "label": "GENE", "start": 4, "end": 7, "confidence": 0.96}
            ]
        }
        mock_request.return_value = mock_response
        
        text = "The CHS gene encodes chalcone synthase."
        schema = {"GENE": "Genetic elements"}
        
        result = extract_entities_domain_specific(
            text=text,
            entity_schema=schema,
            llm_model_name="test-model",
            domain="genetics",
            use_few_shot=True,
            num_examples=3
        )
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == "CHS"
        assert result[0]["label"] == "GENE"
    
    @patch('src.llm_extraction.ner._make_llm_request')
    def test_extract_entities_adaptive(self, mock_request):
        """Test adaptive entity extraction."""
        mock_response = {
            "entities": [
                {"text": "LC-MS", "label": "ANALYTICAL_METHOD", "start": 0, "end": 5, "confidence": 0.98}
            ]
        }
        mock_request.return_value = mock_response
        
        # Long text should trigger few-shot mode
        text = "LC-MS analysis was performed to identify metabolites in plant extracts. The chromatographic separation was optimized for better resolution of phenolic compounds."
        schema = {"ANALYTICAL_METHOD": "Analytical techniques"}
        
        result = extract_entities_adaptive(
            text=text,
            entity_schema=schema,
            llm_model_name="test-model",
            precision_recall_preference="balanced",
            auto_select_examples=True,
            max_examples=4
        )
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == "LC-MS"


class TestEndToEndIntegration:
    """Test complete end-to-end few-shot functionality."""
    
    @patch('src.llm_extraction.ner._make_llm_request')
    def test_full_few_shot_pipeline(self, mock_request):
        """Test complete few-shot NER pipeline."""
        # Mock comprehensive response
        mock_response = {
            "entities": [
                {"text": "Arabidopsis thaliana", "label": "SPECIES", "start": 0, "end": 20, "confidence": 0.99},
                {"text": "anthocyanin", "label": "METABOLITE", "start": 35, "end": 46, "confidence": 0.95},
                {"text": "chalcone synthase", "label": "ENZYME", "start": 60, "end": 77, "confidence": 0.97}
            ]
        }
        mock_request.return_value = mock_response
        
        # Complex scientific text
        text = "Arabidopsis thaliana plants showed increased anthocyanin levels when chalcone synthase was overexpressed."
        
        # Use comprehensive schema
        schema = get_plant_metabolomics_schema()
        
        # Test with different few-shot approaches
        approaches = [
            ("basic", "balanced", None),
            ("detailed", "high_confidence", None),
            ("precision", "diverse", None),
            ("scientific", "balanced", "metabolomics")
        ]
        
        for template_type, strategy, domain in approaches:
            if domain:
                result = extract_entities_domain_specific(
                    text=text,
                    entity_schema=schema,
                    llm_model_name="test-model",
                    domain=domain,
                    use_few_shot=True
                )
            else:
                result = extract_entities_few_shot(
                    text=text,
                    entity_schema=schema,
                    llm_model_name="test-model",
                    template_type=template_type,
                    example_strategy=strategy
                )
            
            # Verify results
            assert isinstance(result, list)
            assert len(result) == 3
            
            # Check specific entities
            entities_by_label = {e["label"]: e for e in result}
            assert "SPECIES" in entities_by_label
            assert "METABOLITE" in entities_by_label
            assert "ENZYME" in entities_by_label
            
            assert entities_by_label["SPECIES"]["text"] == "Arabidopsis thaliana"
            assert entities_by_label["METABOLITE"]["text"] == "anthocyanin"
            assert entities_by_label["ENZYME"]["text"] == "chalcone synthase"


if __name__ == "__main__":
    pytest.main([__file__])