"""
Unit tests for src/llm_extraction/relations.py

This module tests the Relationship Extraction functionality for extracting semantic
relationships between entities from scientific text in the AIM2-ODIE ontology development
and information extraction system. The relations module extracts domain-specific 
relationships such as "affects", "involved in", "upregulates", "downregulates",
"metabolized by", "produced by", etc.

Test Coverage:
- Basic relationship extraction with predefined schemas
- Zero-shot relationship extraction with example relationship types
- Few-shot relationship extraction with provided examples in prompts
- Output format validation for structured relationship triples
- Error handling for LLM API failures, invalid responses, and rate limits
- Hierarchical relationship testing (distinguishing specific vs general relationships)
- Edge cases: empty text, malformed schemas, network issues
- Performance considerations for large texts and batch processing
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any, Optional, Tuple
import requests
from requests.exceptions import RequestException, Timeout, HTTPError
import time

# Import the Relations functions (will be implemented in src/llm_extraction/relations.py)
from src.llm_extraction.relations import (
    extract_relationships,
    RelationsError,
    LLMAPIError,
    InvalidSchemaError,
    RateLimitError,
    InvalidEntitiesError,
    _format_prompt,
    _parse_llm_response,
    _validate_relationship_schema,
    _validate_entities_format,
    _validate_response_format
)


class TestExtractRelationshipsBasic:
    """Test cases for basic relationship extraction functionality."""
    
    def test_extract_relationships_simple_text_basic_schema(self):
        """Test extract_relationships with simple text and basic relationship schema."""
        text = "Quercetin affects antioxidant activity in plant cells."
        entities = [
            {"text": "Quercetin", "label": "COMPOUND", "start": 0, "end": 9, "confidence": 0.95},
            {"text": "antioxidant activity", "label": "BIOLOGICAL_ACTIVITY", "start": 18, "end": 38, "confidence": 0.90},
            {"text": "plant cells", "label": "BIOLOGICAL_SYSTEM", "start": 42, "end": 53, "confidence": 0.85}
        ]
        relationship_schema = {
            "affects": "One entity influences or impacts another entity",
            "involved_in": "One entity participates in or contributes to a process",
            "located_in": "One entity is spatially contained within another"
        }
        
        expected_response = {
            "relationships": [
                {
                    "subject": "Quercetin",
                    "relation": "affects", 
                    "object": "antioxidant activity",
                    "confidence": 0.92,
                    "context": "in plant cells"
                },
                {
                    "subject": "antioxidant activity",
                    "relation": "located_in",
                    "object": "plant cells", 
                    "confidence": 0.88,
                    "context": ""
                }
            ]
        }
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_post.return_value = mock_response
            
            result = extract_relationships(
                text=text,
                entities=entities,
                relationship_schema=relationship_schema,
                llm_model_name="gpt-3.5-turbo",
                prompt_template="Extract relationships from: {text}\nEntities: {entities}\nRelationship types: {schema}"
            )
            
            assert len(result) == 2
            assert result[0] == ("Quercetin", "affects", "antioxidant activity")
            assert result[1] == ("antioxidant activity", "located_in", "plant cells")
    
    def test_extract_relationships_plant_metabolomics_schema(self):
        """Test extract_relationships with comprehensive plant metabolomics schema."""
        text = """
        The CHS gene upregulates flavonoid biosynthesis in Arabidopsis leaves under UV stress.
        Quercetin metabolized by cytochrome P450 enzymes produces hydroxylated derivatives.
        """
        
        entities = [
            {"text": "CHS gene", "label": "GENE", "start": 8, "end": 16, "confidence": 0.98},
            {"text": "flavonoid biosynthesis", "label": "PATHWAY", "start": 28, "end": 50, "confidence": 0.95},
            {"text": "Arabidopsis", "label": "SPECIES", "start": 54, "end": 65, "confidence": 0.97},
            {"text": "leaves", "label": "PLANT_PART", "start": 66, "end": 72, "confidence": 0.92},
            {"text": "UV stress", "label": "EXPERIMENTAL_CONDITION", "start": 79, "end": 88, "confidence": 0.90},
            {"text": "Quercetin", "label": "METABOLITE", "start": 90, "end": 99, "confidence": 0.96},
            {"text": "cytochrome P450 enzymes", "label": "PROTEIN", "start": 114, "end": 137, "confidence": 0.94},
            {"text": "hydroxylated derivatives", "label": "METABOLITE", "start": 147, "end": 171, "confidence": 0.89}
        ]
        
        relationship_schema = {
            "upregulates": "One entity increases the expression or activity of another",
            "downregulates": "One entity decreases the expression or activity of another", 
            "metabolized_by": "One compound is processed or transformed by an enzyme",
            "produces": "One entity generates or creates another entity",
            "involved_in": "One entity participates in a biological process",
            "located_in": "One entity is found within or contained in another",
            "responds_to": "One entity reacts or changes in response to a stimulus"
        }
        
        expected_response = {
            "relationships": [
                {
                    "subject": "CHS gene",
                    "relation": "upregulates",
                    "object": "flavonoid biosynthesis",
                    "confidence": 0.95,
                    "context": "in Arabidopsis leaves under UV stress"
                },
                {
                    "subject": "flavonoid biosynthesis",
                    "relation": "located_in",
                    "object": "leaves",
                    "confidence": 0.90,
                    "context": "in Arabidopsis"
                },
                {
                    "subject": "flavonoid biosynthesis", 
                    "relation": "responds_to",
                    "object": "UV stress",
                    "confidence": 0.88,
                    "context": ""
                },
                {
                    "subject": "Quercetin",
                    "relation": "metabolized_by", 
                    "object": "cytochrome P450 enzymes",
                    "confidence": 0.93,
                    "context": ""
                },
                {
                    "subject": "cytochrome P450 enzymes",
                    "relation": "produces",
                    "object": "hydroxylated derivatives",
                    "confidence": 0.91,
                    "context": "from Quercetin"
                }
            ]
        }
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_post.return_value = mock_response
            
            result = extract_relationships(
                text=text,
                entities=entities,
                relationship_schema=relationship_schema,
                llm_model_name="gpt-4",
                prompt_template="Extract {schema} relationships from: {text}\nGiven entities: {entities}"
            )
            
            # Verify comprehensive relationship extraction
            assert len(result) == 5
            
            # Check specific relationship types
            upregulation_rels = [r for r in result if r[1] == "upregulates"]
            assert len(upregulation_rels) == 1
            assert upregulation_rels[0] == ("CHS gene", "upregulates", "flavonoid biosynthesis")
            
            metabolism_rels = [r for r in result if r[1] == "metabolized_by"]
            assert len(metabolism_rels) == 1
            assert metabolism_rels[0] == ("Quercetin", "metabolized_by", "cytochrome P450 enzymes")
            
            production_rels = [r for r in result if r[1] == "produces"]
            assert len(production_rels) == 1
            assert production_rels[0] == ("cytochrome P450 enzymes", "produces", "hydroxylated derivatives")
    
    def test_extract_relationships_output_format_validation(self):
        """Test that output format matches expected relationship triples format."""
        text = "Anthocyanins provide pigmentation in red flowers."
        entities = [
            {"text": "Anthocyanins", "label": "COMPOUND", "start": 0, "end": 12, "confidence": 0.97},
            {"text": "pigmentation", "label": "BIOLOGICAL_FUNCTION", "start": 21, "end": 33, "confidence": 0.90},
            {"text": "red flowers", "label": "PLANT_PART", "start": 37, "end": 48, "confidence": 0.85}
        ]
        relationship_schema = {"provides": "One entity supplies or gives another entity"}
        
        expected_response = {
            "relationships": [
                {
                    "subject": "Anthocyanins",
                    "relation": "provides",
                    "object": "pigmentation", 
                    "confidence": 0.94,
                    "context": "in red flowers"
                }
            ]
        }
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_post.return_value = mock_response
            
            result = extract_relationships(text, entities, relationship_schema, "gpt-3.5-turbo", "template")
            
            # Validate output format is list of tuples
            assert isinstance(result, list)
            assert len(result) == 1
            
            # Validate each relationship is a tuple with 3 elements
            for relationship in result:
                assert isinstance(relationship, tuple)
                assert len(relationship) == 3
                
                subject, relation, obj = relationship
                assert isinstance(subject, str)
                assert isinstance(relation, str) 
                assert isinstance(obj, str)
                
                # Validate relationship elements are non-empty
                assert subject.strip()
                assert relation.strip()
                assert obj.strip()
            
            # Validate specific relationship content
            assert result[0] == ("Anthocyanins", "provides", "pigmentation")


class TestHierarchicalRelationships:
    """Test cases for hierarchical relationship extraction and context distinction."""
    
    def test_hierarchical_relationships_general_vs_specific(self):
        """Test distinguishing between general and specific relationship types."""
        text = "The PAL enzyme upregulates phenylpropanoid biosynthesis and specifically increases lignin production."
        entities = [
            {"text": "PAL enzyme", "label": "PROTEIN", "start": 4, "end": 14, "confidence": 0.98},
            {"text": "phenylpropanoid biosynthesis", "label": "PATHWAY", "start": 27, "end": 55, "confidence": 0.95},
            {"text": "lignin production", "label": "PATHWAY", "start": 79, "end": 96, "confidence": 0.92}
        ]
        
        relationship_schema = {
            "involved_in": "General participation in a process",
            "upregulates": "Increases expression or activity (more specific than involved_in)",
            "increases": "Enhances or boosts (specific quantitative increase)",
            "regulates": "Controls or manages (general regulatory relationship)"
        }
        
        expected_response = {
            "relationships": [
                {
                    "subject": "PAL enzyme",
                    "relation": "upregulates",  # More specific than "involved_in"
                    "object": "phenylpropanoid biosynthesis",
                    "confidence": 0.96,
                    "context": ""
                },
                {
                    "subject": "PAL enzyme", 
                    "relation": "increases",  # Most specific for quantitative increase
                    "object": "lignin production",
                    "confidence": 0.94,
                    "context": "specifically"
                }
            ]
        }
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_post.return_value = mock_response
            
            result = extract_relationships(
                text=text,
                entities=entities,
                relationship_schema=relationship_schema,
                llm_model_name="gpt-4",
                prompt_template="Identify the most specific relationship type for each entity pair: {text}\nEntities: {entities}\nRelationships (ordered from general to specific): {schema}"
            )
            
            assert len(result) == 2
            
            # Verify that more specific relationships were chosen over general ones
            relations = [r[1] for r in result]
            assert "upregulates" in relations  # Specific rather than "involved_in"
            assert "increases" in relations    # Most specific for quantitative change
            assert "involved_in" not in relations  # General relationship should be avoided
            assert "regulates" not in relations    # General relationship should be avoided
    
    def test_context_dependent_relationship_selection(self):
        """Test that relationship selection depends on textual context."""
        text = "Drought stress downregulates photosynthesis while salt stress affects root development."
        entities = [
            {"text": "Drought stress", "label": "EXPERIMENTAL_CONDITION", "start": 0, "end": 14, "confidence": 0.95},
            {"text": "photosynthesis", "label": "BIOLOGICAL_PROCESS", "start": 27, "end": 41, "confidence": 0.98},
            {"text": "salt stress", "label": "EXPERIMENTAL_CONDITION", "start": 48, "end": 59, "confidence": 0.94},
            {"text": "root development", "label": "BIOLOGICAL_PROCESS", "start": 68, "end": 84, "confidence": 0.92}
        ]
        
        relationship_schema = {
            "affects": "General influence (ambiguous direction)",
            "downregulates": "Specifically decreases expression or activity", 
            "upregulates": "Specifically increases expression or activity",
            "inhibits": "Prevents or reduces activity",
            "enhances": "Improves or strengthens activity"
        }
        
        expected_response = {
            "relationships": [
                {
                    "subject": "Drought stress",
                    "relation": "downregulates",  # Specific based on context word "downregulates"
                    "object": "photosynthesis",
                    "confidence": 0.96,
                    "context": ""
                },
                {
                    "subject": "salt stress",
                    "relation": "affects",  # General based on context word "affects"
                    "object": "root development", 
                    "confidence": 0.88,
                    "context": ""
                }
            ]
        }
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_post.return_value = mock_response
            
            result = extract_relationships(
                text=text,
                entities=entities,
                relationship_schema=relationship_schema,
                llm_model_name="gpt-4",
                prompt_template="Extract relationships based on contextual clues: {text}\nEntities: {entities}\nAvailable relationships: {schema}"
            )
            
            assert len(result) == 2
            
            # Verify context-dependent relationship selection
            drought_rel = next(r for r in result if r[0] == "Drought stress")
            salt_rel = next(r for r in result if r[0] == "salt stress")
            
            assert drought_rel[1] == "downregulates"  # Specific from context
            assert salt_rel[1] == "affects"  # General from context
    
    def test_nested_hierarchical_relationships(self):
        """Test extraction of nested hierarchical relationships with multiple levels."""
        text = "The transcription factor MYB regulates anthocyanin biosynthesis by upregulating DFR gene expression."
        entities = [
            {"text": "transcription factor MYB", "label": "PROTEIN", "start": 4, "end": 29, "confidence": 0.97},
            {"text": "anthocyanin biosynthesis", "label": "PATHWAY", "start": 40, "end": 64, "confidence": 0.95},
            {"text": "DFR gene", "label": "GENE", "start": 79, "end": 87, "confidence": 0.98},
            {"text": "expression", "label": "MOLECULAR_PROCESS", "start": 88, "end": 98, "confidence": 0.90}
        ]
        
        relationship_schema = {
            "regulates": "General regulatory control (level 1 - most general)",
            "controls": "Direct control mechanism (level 2 - more specific)", 
            "upregulates": "Positive regulation (level 3 - specific direction)",
            "activates": "Direct activation (level 4 - most specific mechanism)"
        }
        
        expected_response = {
            "relationships": [
                {
                    "subject": "transcription factor MYB",
                    "relation": "regulates",  # General level for pathway
                    "object": "anthocyanin biosynthesis",
                    "confidence": 0.93,
                    "context": ""
                },
                {
                    "subject": "transcription factor MYB",
                    "relation": "upregulates",  # Specific level for gene
                    "object": "DFR gene",
                    "confidence": 0.95,
                    "context": "expression"
                },
                {
                    "subject": "DFR gene",
                    "relation": "involved_in", 
                    "object": "anthocyanin biosynthesis",
                    "confidence": 0.90,
                    "context": ""
                }
            ]
        }
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_post.return_value = mock_response
            
            result = extract_relationships(
                text=text,
                entities=entities,
                relationship_schema=relationship_schema,
                llm_model_name="gpt-4",
                prompt_template="Extract hierarchical relationships at appropriate specificity levels: {text}\nEntities: {entities}\nHierarchical schema: {schema}"
            )
            
            assert len(result) == 3
            
            # Verify hierarchical specificity matching
            myb_pathway_rel = next(r for r in result if r[0] == "transcription factor MYB" and r[2] == "anthocyanin biosynthesis")
            myb_gene_rel = next(r for r in result if r[0] == "transcription factor MYB" and r[2] == "DFR gene")
            
            assert myb_pathway_rel[1] == "regulates"    # General for complex pathway
            assert myb_gene_rel[1] == "upregulates"    # Specific for individual gene


class TestZeroShotRelationshipExtraction:
    """Test cases for zero-shot relationship extraction."""
    
    def test_zero_shot_basic_relationships(self):
        """Test zero-shot relationship extraction with basic relationship types."""
        text = "Chlorophyll absorbs light energy for photosynthesis in chloroplasts."
        entities = [
            {"text": "Chlorophyll", "label": "PIGMENT", "start": 0, "end": 11, "confidence": 0.99},
            {"text": "light energy", "label": "ENERGY_SOURCE", "start": 20, "end": 32, "confidence": 0.94},
            {"text": "photosynthesis", "label": "BIOLOGICAL_PROCESS", "start": 37, "end": 51, "confidence": 0.98},
            {"text": "chloroplasts", "label": "ORGANELLE", "start": 55, "end": 67, "confidence": 0.96}
        ]
        
        relationship_schema = {
            "absorbs": "One entity takes in or captures another entity",
            "used_for": "One entity serves a purpose for another process",
            "occurs_in": "One process takes place within a location"
        }
        
        expected_response = {
            "relationships": [
                {
                    "subject": "Chlorophyll",
                    "relation": "absorbs",
                    "object": "light energy",
                    "confidence": 0.96,
                    "context": ""
                },
                {
                    "subject": "light energy",
                    "relation": "used_for", 
                    "object": "photosynthesis",
                    "confidence": 0.94,
                    "context": ""
                },
                {
                    "subject": "photosynthesis",
                    "relation": "occurs_in",
                    "object": "chloroplasts",
                    "confidence": 0.92,
                    "context": ""
                }
            ]
        }
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_post.return_value = mock_response
            
            result = extract_relationships(
                text=text,
                entities=entities,
                relationship_schema=relationship_schema,
                llm_model_name="gpt-4",
                prompt_template="Identify {schema} relationships in: {text}\nEntities: {entities}",
                few_shot_examples=None  # Zero-shot
            )
            
            assert len(result) == 3
            assert ("Chlorophyll", "absorbs", "light energy") in result
            assert ("light energy", "used_for", "photosynthesis") in result
            assert ("photosynthesis", "occurs_in", "chloroplasts") in result
            
            # Verify API call was made without examples
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            request_data = json.loads(call_args[1]["data"])
            
            # Prompt should not contain examples
            prompt_content = request_data["messages"][0]["content"]
            assert "examples" not in prompt_content.lower()
    
    def test_zero_shot_domain_specific_relationships(self):
        """Test zero-shot extraction with domain-specific plant metabolomics relationships."""
        text = """
        HPLC analysis detected increased kaempferol levels in drought-stressed barley roots.
        The F3H enzyme catalyzes the conversion of naringenin to dihydrokaempferol.
        """
        
        entities = [
            {"text": "HPLC", "label": "ANALYTICAL_METHOD", "start": 8, "end": 12, "confidence": 0.98},
            {"text": "kaempferol", "label": "METABOLITE", "start": 32, "end": 42, "confidence": 0.97},
            {"text": "drought-stressed", "label": "EXPERIMENTAL_CONDITION", "start": 52, "end": 68, "confidence": 0.93},
            {"text": "barley", "label": "SPECIES", "start": 69, "end": 75, "confidence": 0.95},
            {"text": "roots", "label": "PLANT_PART", "start": 76, "end": 81, "confidence": 0.94},
            {"text": "F3H enzyme", "label": "PROTEIN", "start": 87, "end": 97, "confidence": 0.98},
            {"text": "naringenin", "label": "METABOLITE", "start": 133, "end": 143, "confidence": 0.96},
            {"text": "dihydrokaempferol", "label": "METABOLITE", "start": 147, "end": 164, "confidence": 0.95}
        ]
        
        relationship_schema = {
            "detected_by": "An analytical method identifies or measures a compound",
            "found_in": "A compound is present or located within a biological system",
            "responds_to": "An entity changes in response to an experimental condition",
            "catalyzes": "An enzyme facilitates a biochemical conversion",
            "converts_to": "One compound is transformed into another compound"
        }
        
        expected_response = {
            "relationships": [
                {
                    "subject": "kaempferol",
                    "relation": "detected_by",
                    "object": "HPLC",
                    "confidence": 0.95,
                    "context": "analysis"
                },
                {
                    "subject": "kaempferol", 
                    "relation": "found_in",
                    "object": "roots",
                    "confidence": 0.91,
                    "context": "in drought-stressed barley"
                },
                {
                    "subject": "kaempferol",
                    "relation": "responds_to",
                    "object": "drought-stressed",
                    "confidence": 0.89,
                    "context": "increased levels"
                },
                {
                    "subject": "F3H enzyme",
                    "relation": "catalyzes",
                    "object": "naringenin",
                    "confidence": 0.97,
                    "context": "conversion to dihydrokaempferol"
                },
                {
                    "subject": "naringenin",
                    "relation": "converts_to", 
                    "object": "dihydrokaempferol",
                    "confidence": 0.96,
                    "context": "catalyzed by F3H enzyme"
                }
            ]
        }
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_post.return_value = mock_response
            
            result = extract_relationships(
                text=text,
                entities=entities,
                relationship_schema=relationship_schema,
                llm_model_name="gpt-4",
                prompt_template="Extract {schema} from metabolomics text: {text}\nEntities: {entities}"
            )
            
            # Verify domain-specific relationship extraction
            assert len(result) == 5
            
            # Check analytical method relationships
            detection_rels = [r for r in result if r[1] == "detected_by"]
            assert len(detection_rels) == 1
            assert detection_rels[0] == ("kaempferol", "detected_by", "HPLC")
            
            # Check enzymatic relationships
            catalysis_rels = [r for r in result if r[1] == "catalyzes"]
            assert len(catalysis_rels) == 1
            assert catalysis_rels[0] == ("F3H enzyme", "catalyzes", "naringenin")
            
            # Check metabolic conversion
            conversion_rels = [r for r in result if r[1] == "converts_to"]
            assert len(conversion_rels) == 1
            assert conversion_rels[0] == ("naringenin", "converts_to", "dihydrokaempferol")


class TestFewShotRelationshipExtraction:
    """Test cases for few-shot relationship extraction with examples."""
    
    def test_few_shot_with_examples(self):
        """Test few-shot relationship extraction with provided examples."""
        text = "Salicylic acid induces defense responses in tobacco plants against pathogen attack."
        entities = [
            {"text": "Salicylic acid", "label": "COMPOUND", "start": 0, "end": 14, "confidence": 0.98},
            {"text": "defense responses", "label": "BIOLOGICAL_PROCESS", "start": 23, "end": 40, "confidence": 0.94},
            {"text": "tobacco plants", "label": "BIOLOGICAL_SYSTEM", "start": 44, "end": 58, "confidence": 0.92},
            {"text": "pathogen attack", "label": "STRESS_CONDITION", "start": 67, "end": 82, "confidence": 0.90}
        ]
        
        relationship_schema = {
            "induces": "One entity triggers or causes activation of another",
            "protects_against": "One entity provides defense or resistance to a threat",
            "occurs_in": "One process takes place within a biological system"
        }
        
        few_shot_examples = [
            {
                "text": "Jasmonic acid triggers wound responses in Arabidopsis leaves during herbivore feeding.",
                "entities": [
                    {"text": "Jasmonic acid", "label": "COMPOUND"},
                    {"text": "wound responses", "label": "BIOLOGICAL_PROCESS"},
                    {"text": "Arabidopsis leaves", "label": "BIOLOGICAL_SYSTEM"},
                    {"text": "herbivore feeding", "label": "STRESS_CONDITION"}
                ],
                "relationships": [
                    ("Jasmonic acid", "induces", "wound responses"),
                    ("wound responses", "occurs_in", "Arabidopsis leaves"),
                    ("wound responses", "protects_against", "herbivore feeding")
                ]
            },
            {
                "text": "Abscisic acid activates stomatal closure in rice leaves under drought conditions.",
                "entities": [
                    {"text": "Abscisic acid", "label": "COMPOUND"},
                    {"text": "stomatal closure", "label": "BIOLOGICAL_PROCESS"},
                    {"text": "rice leaves", "label": "BIOLOGICAL_SYSTEM"}, 
                    {"text": "drought conditions", "label": "STRESS_CONDITION"}
                ],
                "relationships": [
                    ("Abscisic acid", "induces", "stomatal closure"),
                    ("stomatal closure", "occurs_in", "rice leaves"),
                    ("stomatal closure", "protects_against", "drought conditions")
                ]
            }
        ]
        
        expected_response = {
            "relationships": [
                {
                    "subject": "Salicylic acid",
                    "relation": "induces",
                    "object": "defense responses",
                    "confidence": 0.96,
                    "context": ""
                },
                {
                    "subject": "defense responses",
                    "relation": "occurs_in",
                    "object": "tobacco plants",
                    "confidence": 0.93,
                    "context": ""
                },
                {
                    "subject": "defense responses",
                    "relation": "protects_against",
                    "object": "pathogen attack",
                    "confidence": 0.91,
                    "context": ""
                }
            ]
        }
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_post.return_value = mock_response
            
            result = extract_relationships(
                text=text,
                entities=entities,
                relationship_schema=relationship_schema,
                llm_model_name="gpt-4",
                prompt_template="Given examples: {examples}\nExtract {schema} from: {text}\nEntities: {entities}",
                few_shot_examples=few_shot_examples
            )
            
            assert len(result) == 3
            assert ("Salicylic acid", "induces", "defense responses") in result
            assert ("defense responses", "occurs_in", "tobacco plants") in result
            assert ("defense responses", "protects_against", "pathogen attack") in result
            
            # Verify API call included examples
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            request_data = json.loads(call_args[1]["data"])
            
            # Prompt should contain examples
            prompt_content = request_data["messages"][0]["content"]
            assert "Jasmonic acid" in prompt_content
            assert "wound responses" in prompt_content
            assert "Examples" in prompt_content or "examples" in prompt_content
    
    def test_few_shot_multiple_examples_learning(self):
        """Test few-shot learning from multiple examples for pattern recognition."""
        text = "Ethylene accelerates fruit ripening by activating ripening-related genes in tomato."
        entities = [
            {"text": "Ethylene", "label": "HORMONE", "start": 0, "end": 8, "confidence": 0.98},
            {"text": "fruit ripening", "label": "DEVELOPMENTAL_PROCESS", "start": 20, "end": 34, "confidence": 0.95},
            {"text": "ripening-related genes", "label": "GENE_SET", "start": 49, "end": 71, "confidence": 0.93},
            {"text": "tomato", "label": "SPECIES", "start": 75, "end": 81, "confidence": 0.97}
        ]
        
        relationship_schema = {
            "accelerates": "One entity speeds up or hastens a process",
            "activates": "One entity turns on or initiates another entity's function",
            "regulates": "One entity controls or manages another entity",
            "occurs_in": "One process takes place within an organism"
        }
        
        few_shot_examples = [
            {
                "text": "Gibberellin promotes stem elongation by activating growth genes in Arabidopsis.",
                "entities": [
                    {"text": "Gibberellin", "label": "HORMONE"},
                    {"text": "stem elongation", "label": "DEVELOPMENTAL_PROCESS"},
                    {"text": "growth genes", "label": "GENE_SET"},
                    {"text": "Arabidopsis", "label": "SPECIES"}
                ],
                "relationships": [
                    ("Gibberellin", "accelerates", "stem elongation"),
                    ("Gibberellin", "activates", "growth genes"),
                    ("stem elongation", "occurs_in", "Arabidopsis")
                ]
            },
            {
                "text": "Cytokinin enhances cell division by activating division genes in root meristems.",
                "entities": [
                    {"text": "Cytokinin", "label": "HORMONE"},
                    {"text": "cell division", "label": "DEVELOPMENTAL_PROCESS"},
                    {"text": "division genes", "label": "GENE_SET"},
                    {"text": "root meristems", "label": "TISSUE"}
                ],
                "relationships": [
                    ("Cytokinin", "accelerates", "cell division"),
                    ("Cytokinin", "activates", "division genes"),
                    ("cell division", "occurs_in", "root meristems")
                ]
            },
            {
                "text": "Auxin stimulates root development by activating developmental genes in seedlings.",
                "entities": [
                    {"text": "Auxin", "label": "HORMONE"},
                    {"text": "root development", "label": "DEVELOPMENTAL_PROCESS"},
                    {"text": "developmental genes", "label": "GENE_SET"},
                    {"text": "seedlings", "label": "BIOLOGICAL_SYSTEM"}
                ],
                "relationships": [
                    ("Auxin", "accelerates", "root development"),
                    ("Auxin", "activates", "developmental genes"),
                    ("root development", "occurs_in", "seedlings")
                ]
            }
        ]
        
        expected_response = {
            "relationships": [
                {
                    "subject": "Ethylene",
                    "relation": "accelerates",
                    "object": "fruit ripening",
                    "confidence": 0.97,
                    "context": ""
                },
                {
                    "subject": "Ethylene",
                    "relation": "activates",
                    "object": "ripening-related genes",
                    "confidence": 0.95,
                    "context": ""
                },
                {
                    "subject": "fruit ripening",
                    "relation": "occurs_in",
                    "object": "tomato",
                    "confidence": 0.92,
                    "context": ""
                }
            ]
        }
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_post.return_value = mock_response
            
            result = extract_relationships(
                text=text,
                entities=entities,
                relationship_schema=relationship_schema,
                llm_model_name="gpt-4",
                prompt_template="Learn from these patterns: {examples}\nNow extract {schema} relationships from: {text}\nEntities: {entities}",
                few_shot_examples=few_shot_examples
            )
            
            # Verify learning from pattern in examples (hormone -> accelerates process, hormone -> activates genes)
            assert len(result) == 3
            assert ("Ethylene", "accelerates", "fruit ripening") in result
            assert ("Ethylene", "activates", "ripening-related genes") in result
            assert ("fruit ripening", "occurs_in", "tomato") in result
            
            # Verify that the pattern from examples was followed
            hormone_accelerates = [r for r in result if r[0] == "Ethylene" and r[1] == "accelerates"]
            hormone_activates = [r for r in result if r[0] == "Ethylene" and r[1] == "activates"]
            process_occurs = [r for r in result if r[1] == "occurs_in"]
            
            assert len(hormone_accelerates) == 1
            assert len(hormone_activates) == 1
            assert len(process_occurs) == 1


class TestErrorHandling:
    """Test cases for error handling in relationship extraction functionality."""
    
    def test_llm_api_failure_handling(self):
        """Test error handling for LLM API failures."""
        text = "Sample text for testing"
        entities = [{"text": "compound", "label": "COMPOUND", "start": 0, "end": 8, "confidence": 0.9}]
        relationship_schema = {"affects": "One entity influences another"}
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            # Simulate API failure
            mock_post.side_effect = requests.exceptions.ConnectionError("API unavailable")
            
            with pytest.raises(LLMAPIError, match="LLM API request failed"):
                extract_relationships(text, entities, relationship_schema, "gpt-3.5-turbo", "template")
    
    def test_http_error_handling(self):
        """Test error handling for HTTP errors from LLM API."""
        text = "Sample text"
        entities = [{"text": "compound", "label": "COMPOUND", "start": 0, "end": 8, "confidence": 0.9}]
        relationship_schema = {"affects": "Influences"}
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            # Simulate HTTP 500 error
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = HTTPError("500 Server Error")
            mock_post.return_value = mock_response
            
            with pytest.raises(LLMAPIError, match="HTTP error occurred"):
                extract_relationships(text, entities, relationship_schema, "gpt-4", "template")
    
    def test_rate_limit_error_handling(self):
        """Test error handling for API rate limit exceeded."""
        text = "Sample text"
        entities = [{"text": "compound", "label": "COMPOUND", "start": 0, "end": 8, "confidence": 0.9}]
        relationship_schema = {"affects": "Influences"}
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            # Simulate rate limit error
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.json.return_value = {"error": "Rate limit exceeded"}
            mock_post.return_value = mock_response
            
            with pytest.raises(RateLimitError, match="Rate limit exceeded"):
                extract_relationships(text, entities, relationship_schema, "gpt-3.5-turbo", "template")
    
    def test_invalid_json_response_handling(self):
        """Test error handling for invalid JSON responses from LLM."""
        text = "Sample text"
        entities = [{"text": "compound", "label": "COMPOUND", "start": 0, "end": 8, "confidence": 0.9}]
        relationship_schema = {"affects": "Influences"}
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            mock_response.text = "Invalid JSON response"
            mock_post.return_value = mock_response
            
            with pytest.raises(LLMAPIError, match="Invalid JSON response"):
                extract_relationships(text, entities, relationship_schema, "gpt-4", "template")
    
    def test_malformed_relationship_response_handling(self):
        """Test error handling for malformed relationship responses."""
        text = "Sample text"
        entities = [{"text": "compound", "label": "COMPOUND", "start": 0, "end": 8, "confidence": 0.9}]
        relationship_schema = {"affects": "Influences"}
        
        malformed_responses = [
            # Missing relationships key
            {"result": []},
            # Relationships not a list
            {"relationships": "not a list"},
            # Relationship missing required fields
            {"relationships": [{"subject": "A", "relation": "affects"}]},  # missing object
            # Invalid field types
            {"relationships": [{"subject": 123, "relation": "affects", "object": "B", "confidence": 0.9}]}
        ]
        
        for malformed_response in malformed_responses:
            with patch('src.llm_extraction.relations.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = malformed_response
                mock_post.return_value = mock_response
                
                with pytest.raises(LLMAPIError):
                    extract_relationships(text, entities, relationship_schema, "gpt-4", "template")
    
    def test_request_timeout_handling(self):
        """Test error handling for request timeouts."""
        text = "Sample text"
        entities = [{"text": "compound", "label": "COMPOUND", "start": 0, "end": 8, "confidence": 0.9}]
        relationship_schema = {"affects": "Influences"}
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            mock_post.side_effect = Timeout("Request timed out")
            
            with pytest.raises(LLMAPIError, match="Request timed out"):
                extract_relationships(text, entities, relationship_schema, "gpt-3.5-turbo", "template")


class TestInputValidation:
    """Test cases for input validation and parameter checking."""
    
    def test_validate_relationship_schema_valid(self):
        """Test validation of valid relationship schemas."""
        valid_schemas = [
            {"affects": "One entity influences another"},
            {"upregulates": "Increases", "downregulates": "Decreases"},
            {"affects": "Influences", "involved_in": "Participates", "located_in": "Contained within"}
        ]
        
        for schema in valid_schemas:
            # Should not raise any exception
            _validate_relationship_schema(schema)
    
    def test_validate_relationship_schema_invalid(self):
        """Test validation of invalid relationship schemas."""
        invalid_schemas = [
            None,  # None schema
            {},    # Empty schema
            "not a dict",  # Not a dictionary
            {"": "Empty key"},  # Empty key
            {"affects": ""},  # Empty description
            {123: "Non-string key"}  # Non-string key
        ]
        
        for schema in invalid_schemas:
            with pytest.raises(InvalidSchemaError):
                _validate_relationship_schema(schema)
    
    def test_validate_entities_format_valid(self):
        """Test validation of valid entities format."""
        valid_entities = [
            [],  # Empty list is valid
            [{"text": "compound", "label": "COMPOUND", "start": 0, "end": 8, "confidence": 0.9}],
            [
                {"text": "gene", "label": "GENE", "start": 0, "end": 4, "confidence": 0.95},
                {"text": "protein", "label": "PROTEIN", "start": 10, "end": 17, "confidence": 0.88}
            ]
        ]
        
        for entities in valid_entities:
            # Should not raise any exception
            _validate_entities_format(entities)
    
    def test_validate_entities_format_invalid(self):
        """Test validation of invalid entities format."""
        invalid_entities = [
            None,  # None entities
            "not a list",  # Not a list
            [{"text": "entity"}],  # Missing required fields
            [{"text": 123, "label": "LABEL", "start": 0, "end": 3, "confidence": 0.9}],  # Invalid types
            [{"text": "entity", "label": "LABEL", "start": -1, "end": 3, "confidence": 0.9}]  # Invalid ranges
        ]
        
        for entities in invalid_entities:
            with pytest.raises(InvalidEntitiesError):
                _validate_entities_format(entities)
    
    def test_empty_text_input(self):
        """Test handling of empty text input."""
        entities = [{"text": "compound", "label": "COMPOUND", "start": 0, "end": 8, "confidence": 0.9}]
        relationship_schema = {"affects": "Influences"}
        
        result = extract_relationships("", entities, relationship_schema, "gpt-3.5-turbo", "template")
        assert result == []
    
    def test_none_text_input(self):
        """Test error handling for None text input."""
        entities = [{"text": "compound", "label": "COMPOUND", "start": 0, "end": 8, "confidence": 0.9}]
        relationship_schema = {"affects": "Influences"}
        
        with pytest.raises(ValueError, match="Text input cannot be None"):
            extract_relationships(None, entities, relationship_schema, "gpt-3.5-turbo", "template")
    
    def test_empty_entities_input(self):
        """Test handling of empty entities list."""
        text = "Sample text"
        relationship_schema = {"affects": "Influences"}
        
        result = extract_relationships(text, [], relationship_schema, "gpt-3.5-turbo", "template")
        assert result == []
    
    def test_invalid_llm_model_name(self):
        """Test error handling for invalid LLM model names."""
        text = "Sample text"
        entities = [{"text": "compound", "label": "COMPOUND", "start": 0, "end": 8, "confidence": 0.9}]
        relationship_schema = {"affects": "Influences"}
        
        invalid_models = [None, "", "invalid-model", 123]
        
        for model in invalid_models:
            with pytest.raises(ValueError, match="Invalid LLM model name"):
                extract_relationships(text, entities, relationship_schema, model, "template")
    
    def test_invalid_prompt_template(self):
        """Test error handling for invalid prompt templates."""
        text = "Sample text"
        entities = [{"text": "compound", "label": "COMPOUND", "start": 0, "end": 8, "confidence": 0.9}]
        relationship_schema = {"affects": "Influences"}
        
        invalid_templates = [None, "", 123]
        
        for template in invalid_templates:
            with pytest.raises(ValueError, match="Invalid prompt template"):
                extract_relationships(text, entities, relationship_schema, "gpt-4", template)
    
    def test_invalid_few_shot_examples_format(self):
        """Test error handling for invalid few-shot examples format."""
        text = "Sample text"
        entities = [{"text": "compound", "label": "COMPOUND", "start": 0, "end": 8, "confidence": 0.9}]
        relationship_schema = {"affects": "Influences"}
        
        invalid_examples = [
            "not a list",  # Not a list
            [{"text": "example"}],  # Missing required fields
            [{"entities": [], "relationships": []}],  # Missing text
            [{"text": "example", "entities": [], "relationships": "not a list"}],  # Relationships not a list
            [{"text": "example", "entities": [], "relationships": [("A", "affects")]}]  # Incomplete relationship tuple
        ]
        
        for examples in invalid_examples:
            with pytest.raises(ValueError, match="Invalid few-shot examples format"):
                extract_relationships(text, entities, relationship_schema, "gpt-4", "template", examples)


class TestPromptFormatting:
    """Test cases for prompt formatting functionality."""
    
    def test_format_prompt_basic(self):
        """Test basic prompt formatting without examples."""
        text = "Sample text"
        entities = [{"text": "compound", "label": "COMPOUND", "start": 0, "end": 8, "confidence": 0.9}]
        schema = {"affects": "One entity influences another"}
        template = "Extract {schema} relationships from: {text}\nEntities: {entities}"
        
        formatted_prompt = _format_prompt(template, text, entities, schema, None)
        
        assert "Sample text" in formatted_prompt
        assert "affects" in formatted_prompt
        assert "influences" in formatted_prompt
        assert "compound" in formatted_prompt
        assert "COMPOUND" in formatted_prompt
    
    def test_format_prompt_with_examples(self):
        """Test prompt formatting with few-shot examples."""
        text = "Sample text"
        entities = [{"text": "compound", "label": "COMPOUND", "start": 0, "end": 8, "confidence": 0.9}]
        schema = {"affects": "Influences"}
        template = "Examples: {examples}\nExtract {schema} from: {text}\nEntities: {entities}"
        examples = [
            {
                "text": "Drug X inhibits enzyme Y.",
                "entities": [
                    {"text": "Drug X", "label": "COMPOUND"},
                    {"text": "enzyme Y", "label": "PROTEIN"}
                ],
                "relationships": [("Drug X", "inhibits", "enzyme Y")]
            }
        ]
        
        formatted_prompt = _format_prompt(template, text, entities, schema, examples)
        
        assert "Examples:" in formatted_prompt
        assert "Drug X" in formatted_prompt
        assert "inhibits" in formatted_prompt
        assert "enzyme Y" in formatted_prompt
        assert "Sample text" in formatted_prompt
    
    def test_format_prompt_schema_formatting(self):
        """Test that relationship schema is properly formatted in prompts."""
        text = "Test"
        entities = [{"text": "A", "label": "COMPOUND", "start": 0, "end": 1, "confidence": 0.9}]
        schema = {
            "affects": "One entity influences another entity",
            "upregulates": "One entity increases expression of another",
            "located_in": "One entity is spatially contained within another"
        }
        template = "Relationship types: {schema}\nText: {text}\nEntities: {entities}"
        
        formatted_prompt = _format_prompt(template, text, entities, schema, None)
        
        # Should contain all schema keys and descriptions
        for key, description in schema.items():
            assert key in formatted_prompt
            assert description in formatted_prompt


class TestResponseParsing:
    """Test cases for LLM response parsing functionality."""
    
    def test_parse_llm_response_valid(self):
        """Test parsing of valid LLM responses."""
        valid_response = {
            "relationships": [
                {
                    "subject": "compound A",
                    "relation": "affects",
                    "object": "process B",
                    "confidence": 0.95,
                    "context": ""
                },
                {
                    "subject": "gene X",
                    "relation": "upregulates",
                    "object": "protein Y",
                    "confidence": 0.88,
                    "context": "in response to stress"
                }
            ]
        }
        
        result = _parse_llm_response(valid_response)
        
        assert len(result) == 2
        assert result[0] == ("compound A", "affects", "process B")
        assert result[1] == ("gene X", "upregulates", "protein Y")
    
    def test_parse_llm_response_empty_relationships(self):
        """Test parsing response with empty relationships list."""
        response = {"relationships": []}
        
        result = _parse_llm_response(response)
        assert result == []
    
    def test_parse_llm_response_invalid_format(self):
        """Test error handling for invalid response formats."""
        # Test missing relationships key
        with pytest.raises(LLMAPIError):
            _parse_llm_response({})
        
        # Test relationships not a list
        with pytest.raises(LLMAPIError):
            _parse_llm_response({"relationships": "not a list"})


class TestResponseFormatValidation:
    """Test cases for response format validation."""
    
    def test_validate_response_format_valid(self):
        """Test validation of valid response formats."""
        valid_relationships = [
            ("compound A", "affects", "process B"),
            ("gene X", "upregulates", "protein Y")
        ]
        
        # Should not raise any exception
        _validate_response_format(valid_relationships)
    
    def test_validate_response_format_invalid_structure(self):
        """Test validation of relationships with invalid structure."""
        invalid_relationships = [
            [("compound", "affects")],  # Missing third element
            [("compound", "affects", "process", "extra")],  # Too many elements
            ["not a tuple"],  # Not a tuple
            [("compound", 123, "process")]  # Non-string elements
        ]
        
        for relationships in invalid_relationships:
            with pytest.raises(LLMAPIError):
                _validate_response_format(relationships)
    
    def test_validate_response_format_empty_elements(self):
        """Test validation of relationships with empty string elements."""
        invalid_relationships = [
            [("", "affects", "process")],  # Empty subject
            [("compound", "", "process")],  # Empty relation
            [("compound", "affects", "")]   # Empty object
        ]
        
        for relationships in invalid_relationships:
            with pytest.raises(LLMAPIError):
                _validate_response_format(relationships)


class TestEdgeCases:
    """Test cases for edge cases and boundary conditions."""
    
    def test_very_long_text_input(self):
        """Test handling of very long text inputs."""
        # Create a very long text
        long_text = "Plant metabolomics analyzes relationships between compounds and processes. " * 1000
        entities = [
            {"text": "metabolomics", "label": "FIELD", "start": 6, "end": 18, "confidence": 0.85},
            {"text": "compounds", "label": "COMPOUND", "start": 50, "end": 59, "confidence": 0.90},
            {"text": "processes", "label": "PROCESS", "start": 64, "end": 73, "confidence": 0.88}
        ]
        relationship_schema = {"analyzes": "Studies or examines"}
        
        expected_response = {
            "relationships": [
                {
                    "subject": "metabolomics",
                    "relation": "analyzes",
                    "object": "compounds",
                    "confidence": 0.92,
                    "context": ""
                }
            ]
        }
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_post.return_value = mock_response
            
            result = extract_relationships(long_text, entities, relationship_schema, "gpt-4", "template")
            
            # Should handle long text without issues
            assert len(result) == 1
            mock_post.assert_called_once()
    
    def test_special_characters_in_text(self):
        """Test handling of special characters and Unicode in text."""
        text = "β-carotene affects α-tocopherol levels in <plant> tissues [p<0.05]."
        entities = [
            {"text": "β-carotene", "label": "COMPOUND", "start": 0, "end": 10, "confidence": 0.95},
            {"text": "α-tocopherol", "label": "COMPOUND", "start": 19, "end": 31, "confidence": 0.93},
            {"text": "plant", "label": "ORGANISM", "start": 42, "end": 47, "confidence": 0.90}
        ]
        relationship_schema = {"affects": "Influences"}
        
        expected_response = {
            "relationships": [
                {
                    "subject": "β-carotene",
                    "relation": "affects",
                    "object": "α-tocopherol",
                    "confidence": 0.91,
                    "context": "levels in plant tissues"
                }
            ]
        }
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_post.return_value = mock_response
            
            result = extract_relationships(text, entities, relationship_schema, "gpt-4", "template")
            
            assert len(result) == 1
            assert result[0] == ("β-carotene", "affects", "α-tocopherol")
    
    def test_no_relationships_found(self):
        """Test handling when no relationships are found between entities."""
        text = "The red fox jumped. The blue sky was clear."
        entities = [
            {"text": "red fox", "label": "ANIMAL", "start": 4, "end": 11, "confidence": 0.95},
            {"text": "blue sky", "label": "PHENOMENON", "start": 25, "end": 33, "confidence": 0.90}
        ]
        relationship_schema = {"affects": "Influences"}
        
        expected_response = {"relationships": []}
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_post.return_value = mock_response
            
            result = extract_relationships(text, entities, relationship_schema, "gpt-4", "template")
            
            assert result == []
    
    def test_single_entity_input(self):
        """Test handling of single entity input (no relationships possible)."""
        text = "Quercetin is a flavonoid."
        entities = [
            {"text": "Quercetin", "label": "COMPOUND", "start": 0, "end": 9, "confidence": 0.95}
        ]
        relationship_schema = {"is_a": "Type relationship"}
        
        result = extract_relationships(text, entities, relationship_schema, "gpt-4", "template")
        assert result == []


class TestRelationsErrorClasses:
    """Test cases for Relations-specific error classes."""
    
    def test_relations_error_inheritance(self):
        """Test that RelationsError properly inherits from Exception."""
        error = RelationsError("Test relations error")
        assert isinstance(error, Exception)
        assert str(error) == "Test relations error"
    
    def test_invalid_entities_error_inheritance(self):
        """Test that InvalidEntitiesError properly inherits from RelationsError."""
        error = InvalidEntitiesError("Entities error")
        assert isinstance(error, RelationsError)
        assert isinstance(error, Exception)
        assert str(error) == "Entities error"


class TestPerformanceAndIntegration:
    """Test cases for performance considerations and integration scenarios."""
    
    def test_batch_processing_multiple_texts(self):
        """Test processing multiple texts efficiently."""
        texts_and_entities = [
            ("Compound A affects process B.", [{"text": "Compound A", "label": "COMPOUND", "start": 0, "end": 10, "confidence": 0.9}]),
            ("Gene X upregulates protein Y.", [{"text": "Gene X", "label": "GENE", "start": 0, "end": 6, "confidence": 0.95}]),
            ("Enzyme Z catalyzes reaction W.", [{"text": "Enzyme Z", "label": "PROTEIN", "start": 0, "end": 8, "confidence": 0.92}])
        ]
        relationship_schema = {"affects": "Influences", "upregulates": "Increases", "catalyzes": "Facilitates"}
        
        expected_responses = [
            {"relationships": [{"subject": "Compound A", "relation": "affects", "object": "process B", "confidence": 0.9, "context": ""}]},
            {"relationships": [{"subject": "Gene X", "relation": "upregulates", "object": "protein Y", "confidence": 0.95, "context": ""}]},
            {"relationships": [{"subject": "Enzyme Z", "relation": "catalyzes", "object": "reaction W", "confidence": 0.92, "context": ""}]}
        ]
        
        with patch('src.llm_extraction.relations.requests.post') as mock_post:
            mock_responses = []
            for response_data in expected_responses:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = response_data
                mock_responses.append(mock_response)
            
            mock_post.side_effect = mock_responses
            
            # Process multiple text-entity pairs
            results = []
            for text, entities in texts_and_entities:
                result = extract_relationships(text, entities, relationship_schema, "gpt-4", "template")
                results.append(result)
            
            # Verify all texts were processed
            assert len(results) == 3
            assert mock_post.call_count == 3
            
            # Verify each result
            assert results[0] == [("Compound A", "affects", "process B")]
            assert results[1] == [("Gene X", "upregulates", "protein Y")]
            assert results[2] == [("Enzyme Z", "catalyzes", "reaction W")]
    
    def test_different_llm_models_compatibility(self):
        """Test compatibility with different LLM models."""
        text = "Quercetin inhibits inflammatory pathways."
        entities = [
            {"text": "Quercetin", "label": "COMPOUND", "start": 0, "end": 9, "confidence": 0.96},
            {"text": "inflammatory pathways", "label": "PATHWAY", "start": 19, "end": 40, "confidence": 0.93}
        ]
        relationship_schema = {"inhibits": "Prevents or reduces activity"}
        
        models = ["gpt-3.5-turbo", "gpt-4", "claude-2", "llama-2"]
        
        expected_response = {
            "relationships": [
                {
                    "subject": "Quercetin",
                    "relation": "inhibits",
                    "object": "inflammatory pathways",
                    "confidence": 0.94,
                    "context": ""
                }
            ]
        }
        
        for model in models:
            with patch('src.llm_extraction.relations.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = expected_response
                mock_post.return_value = mock_response
                
                result = extract_relationships(text, entities, relationship_schema, model, "template")
                
                assert len(result) == 1
                assert result[0] == ("Quercetin", "inhibits", "inflammatory pathways")
                
                # Verify correct model was used in API call
                call_args = mock_post.call_args
                request_data = json.loads(call_args[1]["data"])
                assert request_data["model"] == model


# Fixtures for test data
@pytest.fixture
def sample_plant_metabolomics_text():
    """Fixture providing sample plant metabolomics text for relationship testing."""
    return """
    The CHS gene upregulates anthocyanin biosynthesis in response to UV-B radiation.
    Quercetin metabolized by P450 enzymes produces hydroxylated derivatives that 
    enhance plant defense responses against oxidative stress in leaf tissues.
    """


@pytest.fixture
def comprehensive_relationship_schema():
    """Fixture providing comprehensive relationship schema for plant metabolomics."""
    return {
        "upregulates": "Increases the expression or activity of another entity",
        "downregulates": "Decreases the expression or activity of another entity",
        "metabolized_by": "One compound is processed or transformed by an enzyme",
        "produces": "One entity generates or creates another entity",
        "enhances": "One entity improves or strengthens another entity",
        "responds_to": "One entity reacts or changes in response to a stimulus",
        "protects_against": "One entity provides defense or resistance to a threat",
        "located_in": "One entity is spatially contained within another entity",
        "involved_in": "One entity participates in or contributes to a process",
        "catalyzes": "An enzyme facilitates a biochemical reaction or conversion",
        "inhibits": "One entity prevents or reduces the activity of another",
        "activates": "One entity turns on or initiates another entity's function"
    }


@pytest.fixture
def sample_entities():
    """Fixture providing sample entities for relationship testing."""
    return [
        {"text": "CHS gene", "label": "GENE", "start": 4, "end": 12, "confidence": 0.98},
        {"text": "anthocyanin biosynthesis", "label": "PATHWAY", "start": 24, "end": 48, "confidence": 0.95},
        {"text": "UV-B radiation", "label": "EXPERIMENTAL_CONDITION", "start": 65, "end": 79, "confidence": 0.92},
        {"text": "Quercetin", "label": "METABOLITE", "start": 85, "end": 94, "confidence": 0.96},
        {"text": "P450 enzymes", "label": "PROTEIN", "start": 109, "end": 121, "confidence": 0.94},
        {"text": "hydroxylated derivatives", "label": "METABOLITE", "start": 131, "end": 155, "confidence": 0.89},
        {"text": "plant defense responses", "label": "BIOLOGICAL_PROCESS", "start": 169, "end": 192, "confidence": 0.91},
        {"text": "oxidative stress", "label": "STRESS_CONDITION", "start": 201, "end": 217, "confidence": 0.88},
        {"text": "leaf tissues", "label": "PLANT_PART", "start": 221, "end": 233, "confidence": 0.93}
    ]


@pytest.fixture
def sample_few_shot_examples():
    """Fixture providing sample few-shot examples for relationship extraction."""
    return [
        {
            "text": "Jasmonic acid induces wound responses in Arabidopsis leaves during pathogen attack.",
            "entities": [
                {"text": "Jasmonic acid", "label": "HORMONE"},
                {"text": "wound responses", "label": "BIOLOGICAL_PROCESS"},
                {"text": "Arabidopsis leaves", "label": "BIOLOGICAL_SYSTEM"},
                {"text": "pathogen attack", "label": "STRESS_CONDITION"}
            ],
            "relationships": [
                ("Jasmonic acid", "induces", "wound responses"),
                ("wound responses", "occurs_in", "Arabidopsis leaves"),
                ("wound responses", "responds_to", "pathogen attack")
            ]
        },
        {
            "text": "The PAL enzyme catalyzes the conversion of phenylalanine to cinnamic acid in phenylpropanoid metabolism.",
            "entities": [
                {"text": "PAL enzyme", "label": "PROTEIN"},
                {"text": "phenylalanine", "label": "METABOLITE"},
                {"text": "cinnamic acid", "label": "METABOLITE"},
                {"text": "phenylpropanoid metabolism", "label": "PATHWAY"}
            ],
            "relationships": [
                ("PAL enzyme", "catalyzes", "phenylalanine"),
                ("phenylalanine", "converts_to", "cinnamic acid"),
                ("PAL enzyme", "involved_in", "phenylpropanoid metabolism")
            ]
        }
    ]


# Mark all tests in this module as relationship extraction related
pytestmark = pytest.mark.relations