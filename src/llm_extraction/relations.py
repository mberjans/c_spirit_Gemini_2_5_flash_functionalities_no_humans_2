"""
Relationship extraction module for LLM-based information extraction.

This module provides functionality for extracting relationships between entities from scientific text
using Large Language Models (LLMs). It supports both zero-shot and few-shot relationship extraction
approaches and is specifically designed for plant metabolomics and biological domains.

The module extracts complex relationships like "affects", "made_via", "accumulates_in", and
differentiates between broad ("involved in") and specific ("upregulates") associations.

Functions:
    extract_relationships: Main function for extracting relationships from text
    _format_prompt: Format prompts for LLM API calls
    _parse_llm_response: Parse and validate LLM responses
    _validate_relationship_schema: Validate relationship schema format
    _validate_response_format: Validate extracted relationship format
    _make_llm_request: Make API request for relationship extraction

Classes:
    RelationsError: Base exception for relationship-related errors
    InvalidEntitiesError: Exception for invalid entities format
    LLMAPIError: Exception for LLM API-related errors
    InvalidSchemaError: Exception for invalid relationship schema
    RateLimitError: Exception for API rate limit exceeded
"""

import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set
import requests
from requests.exceptions import RequestException, Timeout, HTTPError


class RelationsError(Exception):
    """Base exception class for relationship extraction errors."""
    pass


class InvalidEntitiesError(RelationsError):
    """Exception raised for invalid entities format."""
    pass


class LLMAPIError(RelationsError):
    """Exception raised for LLM API-related errors."""
    pass


class InvalidSchemaError(RelationsError):
    """Exception raised for invalid relationship schema."""
    pass


class RateLimitError(LLMAPIError):
    """Exception raised when API rate limit is exceeded."""
    pass


# Default relationship types for plant metabolomics research
DEFAULT_RELATIONSHIP_TYPES = {
    # Metabolite-related relationships
    "synthesized_by": "Metabolite is synthesized/produced by an organism or enzyme",
    "found_in": "Metabolite is found/detected in a specific plant part or species",
    "accumulates_in": "Metabolite accumulates in a specific plant part or tissue",
    "derived_from": "Metabolite is derived from another compound or precursor",
    "converted_to": "Metabolite is converted to another compound",
    "made_via": "Metabolite is produced via a specific pathway or process",
    
    # Gene/Protein-related relationships
    "encodes": "Gene encodes a specific protein or enzyme",
    "expressed_in": "Gene is expressed in a specific tissue or condition",
    "regulated_by": "Gene or protein is regulated by another factor",
    "upregulates": "Factor increases expression or activity of target",
    "downregulates": "Factor decreases expression or activity of target",
    "catalyzes": "Enzyme catalyzes a specific reaction or process",
    
    # Pathway relationships
    "involved_in": "Entity participates in a metabolic or biological pathway",
    "part_of": "Entity is a component of a larger system or pathway",
    "upstream_of": "Entity acts upstream in a pathway relative to another",
    "downstream_of": "Entity acts downstream in a pathway relative to another",
    
    # Experimental relationships
    "responds_to": "Entity responds to experimental treatment or condition",
    "affected_by": "Entity is affected by experimental treatment or stress",
    "increases_under": "Entity increases under specific conditions",
    "decreases_under": "Entity decreases under specific conditions",
    
    # Structural relationships
    "located_in": "Entity is located in a specific cellular or tissue location",
    "binds_to": "Molecule binds to another molecule or target",
    "interacts_with": "Entity interacts with another entity",
    
    # Phenotypic relationships
    "associated_with": "Entity is associated with a trait or phenotype",
    "contributes_to": "Entity contributes to a specific trait or function",
    "required_for": "Entity is required for a specific process or trait",
    
    # Analytical relationships
    "detected_by": "Entity is detected using a specific analytical method",
    "measured_with": "Entity is measured or quantified using a technique",
    "characterized_by": "Entity is characterized using analytical approaches"
}


def extract_relationships(
    text: str,
    entities: List[Dict[str, Any]],
    relationship_schema: Dict[str, str],
    llm_model_name: str,
    prompt_template: str,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Extract relationships between entities from text using LLM-based approach.
    
    This function identifies relationships between previously extracted entities
    based on the provided relationship schema. It supports both zero-shot and
    few-shot learning approaches.
    
    Args:
        text: Input text to extract relationships from
        entities: List of previously extracted entities with their positions
        relationship_schema: Dictionary mapping relationship types to descriptions
        llm_model_name: Name of the LLM model to use
        prompt_template: Template for formatting the prompt
        few_shot_examples: Optional list of examples for few-shot learning
        
    Returns:
        List of dictionaries containing extracted relationships with:
        - subject_entity: The source entity (dict with text, label, start, end)
        - relation_type: The relationship type/label
        - object_entity: The target entity (dict with text, label, start, end)
        - confidence: Confidence score (0.0-1.0)
        - context: Supporting context from text
        - evidence: Text span supporting the relationship
        
    Raises:
        ValueError: For invalid input parameters
        InvalidSchemaError: For invalid relationship schema
        LLMAPIError: For LLM API-related errors
        RateLimitError: For API rate limit errors
    """
    # Input validation
    if text is None:
        raise ValueError("Text input cannot be None")
    
    if not text.strip():
        return []
    
    if entities is None:
        raise ValueError("Entities list cannot be None")
    
    if not isinstance(entities, list):
        raise ValueError("Entities must be a list")
    
    if len(entities) < 2:
        return []  # Need at least 2 entities to form relationships
    
    if not isinstance(llm_model_name, str) or not llm_model_name.strip():
        raise ValueError("Invalid LLM model name")
    
    # Additional model name validation
    if isinstance(llm_model_name, str) and llm_model_name in ["", "invalid-model"] or isinstance(llm_model_name, (int, float)):
        raise ValueError("Invalid LLM model name")
    
    if not isinstance(prompt_template, str) or not prompt_template.strip():
        raise ValueError("Invalid prompt template")
    
    # Validate relationship schema
    _validate_relationship_schema(relationship_schema)
    
    # Validate entities format
    _validate_entities_format(entities)
    
    # Validate few-shot examples if provided
    if few_shot_examples is not None:
        _validate_few_shot_relationship_examples(few_shot_examples)
    
    # Format the prompt
    formatted_prompt = _format_prompt(
        prompt_template, text, entities, relationship_schema, few_shot_examples
    )
    
    # Make API request with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = _make_llm_request(formatted_prompt, llm_model_name)
            break
        except (HTTPError, RequestException) as e:
            if attempt == max_retries - 1:
                raise LLMAPIError(f"LLM API request failed after {max_retries} attempts: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    
    # Parse and validate response
    relationships = _parse_llm_response(response)
    _validate_response_format(relationships, entities)
    
    # Filter and enhance relationships
    relationships = _filter_valid_relationships(relationships, entities, text)
    relationships = _add_relationship_context(relationships, text)
    
    return relationships


def _format_prompt(
    template: str,
    text: str,
    entities: List[Dict[str, Any]],
    schema: Dict[str, str],
    examples: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Format the prompt for relationship extraction LLM API call.
    
    Args:
        template: Prompt template with placeholders
        text: Input text
        entities: List of extracted entities
        schema: Relationship schema
        examples: Optional few-shot examples
        
    Returns:
        Formatted prompt string
    """
    # Format entities as a readable string
    entities_str = _format_entities_for_prompt(entities)
    
    # Format schema as a readable string
    schema_str = "\n".join([f"- {key}: {desc}" for key, desc in schema.items()])
    
    # Format examples for few-shot templates
    examples_str = ""
    if examples:
        examples_list = []
        for example in examples:
            example_text = example.get("text", "")
            example_entities = example.get("entities", [])
            example_relationships = example.get("relationships", [])
            
            entities_formatted = _format_entities_for_prompt(example_entities)
            relationships_formatted = _format_relationships_for_prompt(example_relationships)
            
            example_str = f"Text: {example_text}\n"
            example_str += f"Entities: {entities_formatted}\n"
            example_str += f"Relationships: {relationships_formatted}"
            examples_list.append(example_str)
        
        examples_str = "\n\nExamples:\n" + "\n\n".join(examples_list)
    
    # Replace placeholders in template
    formatted_prompt = template.replace("{text}", text)
    formatted_prompt = formatted_prompt.replace("{entities}", entities_str)
    formatted_prompt = formatted_prompt.replace("{schema}", schema_str)
    formatted_prompt = formatted_prompt.replace("{examples}", examples_str)
    
    return formatted_prompt


def _format_entities_for_prompt(entities: List[Dict[str, Any]]) -> str:
    """
    Format entities for inclusion in prompt.
    
    Args:
        entities: List of entity dictionaries
        
    Returns:
        Formatted string representation of entities
    """
    entity_strings = []
    for i, entity in enumerate(entities):
        entity_str = f"[{i}] {entity['text']} ({entity['label']})"
        if 'start' in entity and 'end' in entity:
            entity_str += f" [pos: {entity['start']}-{entity['end']}]"
        entity_strings.append(entity_str)
    
    return "\n".join(entity_strings)


def _format_relationships_for_prompt(relationships: List[Dict[str, Any]]) -> str:
    """
    Format relationships for inclusion in prompt examples.
    
    Args:
        relationships: List of relationship dictionaries
        
    Returns:
        Formatted string representation of relationships
    """
    if not relationships:
        return "None"
    
    rel_strings = []
    for rel in relationships:
        subject = rel['subject_entity']['text']
        relation = rel['relation_type']
        obj = rel['object_entity']['text']
        rel_strings.append(f"{subject} --{relation}--> {obj}")
    
    return "; ".join(rel_strings)


def _make_llm_request(prompt: str, model_name: str) -> Dict[str, Any]:
    """
    Make request to LLM API for relationship extraction.
    
    Args:
        prompt: Formatted prompt
        model_name: Model name
        
    Returns:
        API response as dictionary
        
    Raises:
        LLMAPIError: For API-related errors
        RateLimitError: For rate limit errors
    """
    # Mock API endpoint - in real implementation this would be actual LLM API
    api_url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY"  # In real implementation, get from env
    }
    
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(
            api_url,
            headers=headers,
            data=json.dumps(data),
            timeout=30
        )
        
        if response.status_code == 429:
            raise RateLimitError("Rate limit exceeded")
        
        response.raise_for_status()
        
        return response.json()
        
    except Timeout:
        raise LLMAPIError("Request timed out")
    except HTTPError as e:
        raise LLMAPIError(f"HTTP error occurred: {e}")
    except RequestException as e:
        raise LLMAPIError(f"LLM API request failed: {e}")
    except json.JSONDecodeError:
        raise LLMAPIError("Invalid JSON response from LLM API")


def _parse_llm_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse LLM API response to extract relationships.
    
    Args:
        response: LLM API response
        
    Returns:
        List of extracted relationships
        
    Raises:
        LLMAPIError: For invalid response format
    """
    if not isinstance(response, dict):
        raise LLMAPIError("Invalid response format: response must be a dictionary")
    
    # Handle typical OpenAI API response format
    if "choices" in response:
        try:
            content = response["choices"][0]["message"]["content"]
            # Parse JSON content from the message
            import json
            relationships_data = json.loads(content)
            if "relationships" in relationships_data:
                relationships = relationships_data["relationships"]
            else:
                raise LLMAPIError("Invalid response format: missing 'relationships' key in content")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise LLMAPIError(f"Invalid OpenAI response format: {e}")
    else:
        # Handle direct format for backward compatibility
        if "relationships" not in response:
            raise LLMAPIError("Invalid response format: missing 'relationships' key")
        relationships = response["relationships"]
    
    if not isinstance(relationships, list):
        raise LLMAPIError("Invalid response format: 'relationships' must be a list")
    
    return relationships


def _validate_relationship_schema(schema: Dict[str, str]) -> None:
    """
    Validate relationship schema format.
    
    Args:
        schema: Relationship schema to validate
        
    Raises:
        InvalidSchemaError: For invalid schema format
    """
    if schema is None:
        raise InvalidSchemaError("Relationship schema cannot be None")
    
    if not isinstance(schema, dict):
        raise InvalidSchemaError("Relationship schema must be a dictionary")
    
    if not schema:
        raise InvalidSchemaError("Relationship schema cannot be empty")
    
    for key, value in schema.items():
        if not isinstance(key, str):
            raise InvalidSchemaError("Schema keys must be strings")
        
        if not key.strip():
            raise InvalidSchemaError("Schema keys cannot be empty")
        
        if not isinstance(value, str):
            raise InvalidSchemaError("Schema values must be strings")
        
        if not value.strip():
            raise InvalidSchemaError("Schema descriptions cannot be empty")


def _validate_entities_format(entities: List[Dict[str, Any]]) -> None:
    """
    Validate entities format for relationship extraction.
    
    Args:
        entities: List of entities
        
    Raises:
        InvalidEntitiesError: For invalid entity format
    """
    required_fields = ["text", "label"]
    
    for i, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise InvalidEntitiesError(f"Entity {i} must be a dictionary")
        
        # Check required fields
        for field in required_fields:
            if field not in entity:
                raise InvalidEntitiesError(f"Missing required field '{field}' in entity {i}")
        
        # Check field types
        if not isinstance(entity["text"], str):
            raise InvalidEntitiesError(f"Invalid field type: 'text' must be string in entity {i}")
        
        if not isinstance(entity["label"], str):
            raise InvalidEntitiesError(f"Invalid field type: 'label' must be string in entity {i}")
        
        if not entity["text"].strip():
            raise InvalidEntitiesError(f"Entity text cannot be empty in entity {i}")


def _validate_few_shot_relationship_examples(examples: List[Dict[str, Any]]) -> None:
    """
    Validate few-shot relationship examples format.
    
    Args:
        examples: List of few-shot examples
        
    Raises:
        ValueError: For invalid examples format
    """
    if not isinstance(examples, list):
        raise ValueError("Invalid few-shot examples format: must be a list")
    
    for i, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(f"Invalid few-shot examples format: example {i} must be a dictionary")
        
        if "text" not in example:
            raise ValueError(f"Invalid few-shot examples format: example {i} missing 'text' field")
        
        if "entities" not in example:
            raise ValueError(f"Invalid few-shot examples format: example {i} missing 'entities' field")
        
        if "relationships" not in example:
            raise ValueError(f"Invalid few-shot examples format: example {i} missing 'relationships' field")
        
        if not isinstance(example["entities"], list):
            raise ValueError(f"Invalid few-shot examples format: example {i} 'entities' must be a list")
        
        if not isinstance(example["relationships"], list):
            raise ValueError(f"Invalid few-shot examples format: example {i} 'relationships' must be a list")
        
        # Validate entities in example
        _validate_entities_format(example["entities"])
        
        # Validate relationships in example
        for j, relationship in enumerate(example["relationships"]):
            if not isinstance(relationship, dict):
                raise ValueError(f"Invalid few-shot examples format: example {i} relationship {j} must be a dictionary")
            
            required_rel_fields = ["subject_entity", "relation_type", "object_entity"]
            for field in required_rel_fields:
                if field not in relationship:
                    raise ValueError(f"Invalid few-shot examples format: example {i} relationship {j} missing '{field}' field")


def _validate_response_format(relationships: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> None:
    """
    Validate extracted relationships format.
    
    Args:
        relationships: List of extracted relationships
        entities: List of available entities
        
    Raises:
        LLMAPIError: For invalid relationship format
    """
    required_fields = ["subject_entity", "relation_type", "object_entity", "confidence"]
    
    for i, relationship in enumerate(relationships):
        if not isinstance(relationship, dict):
            raise LLMAPIError(f"Relationship {i} must be a dictionary")
        
        # Check required fields
        for field in required_fields:
            if field not in relationship:
                raise LLMAPIError(f"Missing required field '{field}' in relationship {i}")
        
        # Check field types
        if not isinstance(relationship["subject_entity"], dict):
            raise LLMAPIError(f"Invalid field type: 'subject_entity' must be dict in relationship {i}")
        
        if not isinstance(relationship["object_entity"], dict):
            raise LLMAPIError(f"Invalid field type: 'object_entity' must be dict in relationship {i}")
        
        if not isinstance(relationship["relation_type"], str):
            raise LLMAPIError(f"Invalid field type: 'relation_type' must be string in relationship {i}")
        
        if not isinstance(relationship["confidence"], (int, float)):
            raise LLMAPIError(f"Invalid field type: 'confidence' must be number in relationship {i}")
        
        # Check confidence range
        if not (0.0 <= relationship["confidence"] <= 1.0):
            raise LLMAPIError(f"Invalid field range: 'confidence' must be between 0.0 and 1.0 in relationship {i}")
        
        # Validate entity structures
        for entity_key in ["subject_entity", "object_entity"]:
            entity = relationship[entity_key]
            if "text" not in entity or "label" not in entity:
                raise LLMAPIError(f"Missing required fields in {entity_key} of relationship {i}")


def _filter_valid_relationships(
    relationships: List[Dict[str, Any]], 
    entities: List[Dict[str, Any]], 
    text: str
) -> List[Dict[str, Any]]:
    """
    Filter relationships to ensure they are valid and meaningful.
    
    Args:
        relationships: List of extracted relationships
        entities: List of available entities
        text: Original text
        
    Returns:
        Filtered list of valid relationships
    """
    valid_relationships = []
    entity_texts = {entity["text"].lower() for entity in entities}
    
    for relationship in relationships:
        # Check if entities exist in the original entity list
        subject_text = relationship["subject_entity"]["text"].lower()
        object_text = relationship["object_entity"]["text"].lower()
        
        if subject_text in entity_texts and object_text in entity_texts:
            # Avoid self-relationships
            if subject_text != object_text:
                # Check confidence threshold
                if relationship["confidence"] >= 0.3:  # Minimum confidence threshold
                    valid_relationships.append(relationship)
    
    return valid_relationships


def _add_relationship_context(relationships: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """
    Add contextual information to relationships.
    
    Args:
        relationships: List of relationships
        text: Original text
        
    Returns:
        Enhanced relationships with context
    """
    enhanced_relationships = []
    
    for relationship in relationships:
        enhanced_rel = relationship.copy()
        
        # Add context if not already present
        if "context" not in enhanced_rel:
            enhanced_rel["context"] = _extract_relationship_context(relationship, text)
        
        # Add evidence if not already present
        if "evidence" not in enhanced_rel:
            enhanced_rel["evidence"] = _extract_relationship_evidence(relationship, text)
        
        enhanced_relationships.append(enhanced_rel)
    
    return enhanced_relationships


def _extract_relationship_context(relationship: Dict[str, Any], text: str) -> str:
    """
    Extract context surrounding a relationship from text.
    
    Args:
        relationship: Relationship dictionary
        text: Original text
        
    Returns:
        Context string
    """
    subject_text = relationship["subject_entity"]["text"]
    object_text = relationship["object_entity"]["text"]
    
    # Use proper regex for scientific text sentence boundaries
    # This pattern handles abbreviations, numbers, and scientific notation better
    sentence_pattern = r'(?<!\b(?:Dr|Mr|Mrs|Ms|Prof|vs|etc|cf|e\.g|i\.e|al|Fig|Tab)\.)(?<!\b[A-Z]\.)(?<=\.|\!|\?)\s+'
    sentences = re.split(sentence_pattern, text)
    context_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if (sentence and 
            subject_text.lower() in sentence.lower() and 
            object_text.lower() in sentence.lower()):
            context_sentences.append(sentence)
    
    return " ".join(context_sentences[:2])  # Return up to 2 sentences of context


def _extract_relationship_evidence(relationship: Dict[str, Any], text: str) -> str:
    """
    Extract evidence span supporting a relationship.
    
    Args:
        relationship: Relationship dictionary
        text: Original text
        
    Returns:
        Evidence string
    """
    subject_text = relationship["subject_entity"]["text"]
    object_text = relationship["object_entity"]["text"]
    
    # Find the shortest span containing both entities
    subject_pos = text.lower().find(subject_text.lower())
    object_pos = text.lower().find(object_text.lower())
    
    if subject_pos != -1 and object_pos != -1:
        start_pos = min(subject_pos, object_pos)
        end_pos = max(
            subject_pos + len(subject_text),
            object_pos + len(object_text)
        )
        
        # Extend to word boundaries
        while start_pos > 0 and text[start_pos - 1] not in ' \n\t.':
            start_pos -= 1
        while end_pos < len(text) and text[end_pos] not in ' \n\t.':
            end_pos += 1
        
        return text[start_pos:end_pos].strip()
    
    return ""


# Helper functions for relationship extraction

def extract_relationships_with_default_schema(
    text: str,
    entities: List[Dict[str, Any]],
    llm_model_name: str,
    template_type: str = "basic",
    few_shot_examples: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Extract relationships using default relationship schema.
    
    Args:
        text: Input text
        entities: List of extracted entities
        llm_model_name: LLM model name
        template_type: Type of template to use
        few_shot_examples: Optional examples for few-shot learning
        
    Returns:
        List of extracted relationships
    """
    from .prompt_templates import get_relationship_template
    
    template = get_relationship_template(template_type)
    return extract_relationships(
        text, entities, DEFAULT_RELATIONSHIP_TYPES, 
        llm_model_name, template, few_shot_examples
    )


def extract_domain_specific_relationships(
    text: str,
    entities: List[Dict[str, Any]],
    llm_model_name: str,
    domain: str,
    use_few_shot: bool = True
) -> List[Dict[str, Any]]:
    """
    Extract relationships using domain-specific schema and templates.
    
    Args:
        text: Input text
        entities: List of extracted entities
        llm_model_name: LLM model name
        domain: Domain name (metabolomics, genetics, etc.)
        use_few_shot: Whether to use few-shot learning
        
    Returns:
        List of extracted relationships
    """
    # Define domain-specific relationship schemas
    domain_schemas = {
        "metabolomics": {
            "synthesized_by": "Metabolite is synthesized by an organism or enzyme",
            "found_in": "Metabolite is found in a specific plant part",
            "accumulates_in": "Metabolite accumulates in tissue or organ",
            "derived_from": "Metabolite is derived from precursor compound",
            "made_via": "Metabolite is produced via specific pathway"
        },
        "genetics": {
            "encodes": "Gene encodes protein or enzyme",
            "expressed_in": "Gene is expressed in specific tissue",
            "regulated_by": "Gene is regulated by transcription factor",
            "upregulates": "Factor increases gene expression",
            "downregulates": "Factor decreases gene expression"
        },
        "biochemistry": {
            "catalyzes": "Enzyme catalyzes biochemical reaction",
            "involved_in": "Entity participates in metabolic pathway",
            "upstream_of": "Entity acts upstream in pathway",
            "downstream_of": "Entity acts downstream in pathway"
        }
    }
    
    schema = domain_schemas.get(domain, DEFAULT_RELATIONSHIP_TYPES)
    
    from .prompt_templates import get_relationship_template
    template_name = f"relationship_{domain}" if use_few_shot else "relationship_basic"
    
    try:
        template = get_relationship_template(template_name)
    except:
        template = get_relationship_template("relationship_basic")
    
    examples = None
    if use_few_shot:
        examples = _get_domain_relationship_examples(domain)
    
    return extract_relationships(text, entities, schema, llm_model_name, template, examples)


def _get_domain_relationship_examples(domain: str) -> List[Dict[str, Any]]:
    """
    Get domain-specific relationship examples.
    
    Args:
        domain: Domain name
        
    Returns:
        List of example relationships
    """
    examples = {
        "metabolomics": [
            {
                "text": "Anthocyanins are synthesized in grape berries through the flavonoid biosynthesis pathway.",
                "entities": [
                    {"text": "Anthocyanins", "label": "FLAVONOID"},
                    {"text": "grape berries", "label": "FRUIT"},
                    {"text": "flavonoid biosynthesis pathway", "label": "METABOLIC_PATHWAY"}
                ],
                "relationships": [
                    {
                        "subject_entity": {"text": "Anthocyanins", "label": "FLAVONOID"},
                        "relation_type": "synthesized_by",
                        "object_entity": {"text": "grape berries", "label": "FRUIT"},
                        "confidence": 0.9
                    },
                    {
                        "subject_entity": {"text": "Anthocyanins", "label": "FLAVONOID"},
                        "relation_type": "made_via",
                        "object_entity": {"text": "flavonoid biosynthesis pathway", "label": "METABOLIC_PATHWAY"},
                        "confidence": 0.95
                    }
                ]
            }
        ],
        "genetics": [
            {
                "text": "The CHS gene encodes chalcone synthase and is highly expressed in flower petals.",
                "entities": [
                    {"text": "CHS gene", "label": "GENE"},
                    {"text": "chalcone synthase", "label": "ENZYME"},
                    {"text": "flower petals", "label": "PLANT_PART"}
                ],
                "relationships": [
                    {
                        "subject_entity": {"text": "CHS gene", "label": "GENE"},
                        "relation_type": "encodes",
                        "object_entity": {"text": "chalcone synthase", "label": "ENZYME"},
                        "confidence": 0.95
                    },
                    {
                        "subject_entity": {"text": "CHS gene", "label": "GENE"},
                        "relation_type": "expressed_in",
                        "object_entity": {"text": "flower petals", "label": "PLANT_PART"},
                        "confidence": 0.85
                    }
                ]
            }
        ]
    }
    
    return examples.get(domain, [])


def get_relationship_statistics(relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about extracted relationships.
    
    Args:
        relationships: List of relationships
        
    Returns:
        Dictionary with relationship statistics
    """
    if not relationships:
        return {
            "total_relationships": 0,
            "relation_types": [],
            "avg_confidence": 0.0,
            "entity_pairs": 0
        }
    
    relation_types = [rel["relation_type"] for rel in relationships]
    confidences = [rel["confidence"] for rel in relationships]
    
    # Count unique entity pairs
    entity_pairs = set()
    for rel in relationships:
        subject = rel["subject_entity"]["text"]
        obj = rel["object_entity"]["text"]
        entity_pairs.add((subject, obj))
    
    return {
        "total_relationships": len(relationships),
        "relation_types": list(set(relation_types)),
        "relation_type_counts": {rt: relation_types.count(rt) for rt in set(relation_types)},
        "avg_confidence": sum(confidences) / len(confidences),
        "min_confidence": min(confidences),
        "max_confidence": max(confidences),
        "entity_pairs": len(entity_pairs)
    }


def filter_relationships_by_confidence(
    relationships: List[Dict[str, Any]], 
    min_confidence: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Filter relationships by minimum confidence threshold.
    
    Args:
        relationships: List of relationships
        min_confidence: Minimum confidence threshold
        
    Returns:
        Filtered list of relationships
    """
    return [rel for rel in relationships if rel["confidence"] >= min_confidence]


def group_relationships_by_type(relationships: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group relationships by relation type.
    
    Args:
        relationships: List of relationships
        
    Returns:
        Dictionary mapping relation types to lists of relationships
    """
    grouped = {}
    for rel in relationships:
        relation_type = rel["relation_type"]
        if relation_type not in grouped:
            grouped[relation_type] = []
        grouped[relation_type].append(rel)
    
    return grouped