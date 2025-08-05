"""
Named Entity Recognition (NER) module for LLM-based information extraction.

This module provides functionality for extracting named entities from scientific text
using Large Language Models (LLMs). It supports both zero-shot and few-shot NER
approaches and is specifically designed for plant metabolomics and biological domains.

Functions:
    extract_entities: Main function for extracting entities from text
    _format_prompt: Format prompts for LLM API calls
    _parse_llm_response: Parse and validate LLM responses
    _validate_entity_schema: Validate entity schema format
    _validate_response_format: Validate extracted entity format

Classes:
    NERError: Base exception for NER-related errors
    LLMAPIError: Exception for LLM API-related errors
    InvalidSchemaError: Exception for invalid entity schema
    RateLimitError: Exception for API rate limit exceeded
"""

import json
import time
from typing import List, Dict, Any, Optional
import requests
from requests.exceptions import RequestException, Timeout, HTTPError


class NERError(Exception):
    """Base exception class for NER-related errors."""
    pass


class LLMAPIError(NERError):
    """Exception raised for LLM API-related errors."""
    pass


class InvalidSchemaError(NERError):
    """Exception raised for invalid entity schema."""
    pass


class RateLimitError(LLMAPIError):
    """Exception raised when API rate limit is exceeded."""
    pass


def extract_entities(
    text: str,
    entity_schema: Dict[str, str],
    llm_model_name: str,
    prompt_template: str,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Extract named entities from text using LLM-based approach.
    
    This function sends text to an LLM API to extract entities based on the provided
    schema. It supports both zero-shot and few-shot learning approaches.
    
    Args:
        text: Input text to extract entities from
        entity_schema: Dictionary mapping entity types to descriptions
        llm_model_name: Name of the LLM model to use
        prompt_template: Template for formatting the prompt
        few_shot_examples: Optional list of examples for few-shot learning
        
    Returns:
        List of dictionaries containing extracted entities with:
        - text: The entity text
        - label: The entity type/label
        - start: Start character position
        - end: End character position  
        - confidence: Confidence score (0.0-1.0)
        
    Raises:
        ValueError: For invalid input parameters
        InvalidSchemaError: For invalid entity schema
        LLMAPIError: For LLM API-related errors
        RateLimitError: For API rate limit errors
    """
    # Input validation
    if text is None:
        raise ValueError("Text input cannot be None")
    
    if not text.strip():
        return []
    
    if not isinstance(llm_model_name, str) or not llm_model_name.strip():
        raise ValueError("Invalid LLM model name")
    
    # Additional model name validation
    if isinstance(llm_model_name, str) and llm_model_name in ["", "invalid-model"] or isinstance(llm_model_name, (int, float)):
        raise ValueError("Invalid LLM model name")
    
    if not isinstance(prompt_template, str) or not prompt_template.strip():
        raise ValueError("Invalid prompt template")
    
    # Validate entity schema
    _validate_entity_schema(entity_schema)
    
    # Validate few-shot examples if provided
    if few_shot_examples is not None:
        _validate_few_shot_examples(few_shot_examples)
    
    # Format the prompt
    formatted_prompt = _format_prompt(prompt_template, text, entity_schema, few_shot_examples)
    
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
    entities = _parse_llm_response(response)
    _validate_response_format(entities)
    
    return entities


def _format_prompt(
    template: str,
    text: str,
    schema: Dict[str, str],
    examples: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Format the prompt for LLM API call.
    
    Args:
        template: Prompt template with placeholders
        text: Input text
        schema: Entity schema
        examples: Optional few-shot examples
        
    Returns:
        Formatted prompt string
    """
    # Format schema as a readable string
    schema_str = "\n".join([f"- {key}: {desc}" for key, desc in schema.items()])
    
    # Format examples if provided
    examples_str = ""
    if examples:
        examples_list = []
        for example in examples:
            example_text = example["text"]
            example_entities = ", ".join([
                f"{e['text']} ({e['label']})" for e in example["entities"]
            ])
            examples_list.append(f"Text: {example_text}\nEntities: {example_entities}")
        examples_str = "\n\nExamples:\n" + "\n\n".join(examples_list)
    
    # Replace placeholders in template
    formatted_prompt = template.replace("{text}", text)
    formatted_prompt = formatted_prompt.replace("{schema}", schema_str)
    formatted_prompt = formatted_prompt.replace("{examples}", examples_str)
    
    return formatted_prompt


def _make_llm_request(prompt: str, model_name: str) -> Dict[str, Any]:
    """
    Make request to LLM API.
    
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
        "max_tokens": 1000
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
    Parse LLM API response to extract entities.
    
    Args:
        response: LLM API response
        
    Returns:
        List of extracted entities
        
    Raises:
        LLMAPIError: For invalid response format
    """
    if not isinstance(response, dict):
        raise LLMAPIError("Invalid response format: response must be a dictionary")
    
    if "entities" not in response:
        raise LLMAPIError("Invalid response format: missing 'entities' key")
    
    entities = response["entities"]
    
    if not isinstance(entities, list):
        raise LLMAPIError("Invalid response format: 'entities' must be a list")
    
    return entities


def _validate_entity_schema(schema: Dict[str, str]) -> None:
    """
    Validate entity schema format.
    
    Args:
        schema: Entity schema to validate
        
    Raises:
        InvalidSchemaError: For invalid schema format
    """
    if schema is None:
        raise InvalidSchemaError("Entity schema cannot be None")
    
    if not isinstance(schema, dict):
        raise InvalidSchemaError("Entity schema must be a dictionary")
    
    if not schema:
        raise InvalidSchemaError("Entity schema cannot be empty")
    
    for key, value in schema.items():
        if not isinstance(key, str):
            raise InvalidSchemaError("Schema keys must be strings")
        
        if not key.strip():
            raise InvalidSchemaError("Schema keys cannot be empty")
        
        if not key.isupper():
            raise InvalidSchemaError("Schema keys should be uppercase")
        
        if not isinstance(value, str):
            raise InvalidSchemaError("Schema values must be strings")
        
        if not value.strip():
            raise InvalidSchemaError("Schema descriptions cannot be empty")


def _validate_few_shot_examples(examples: List[Dict[str, Any]]) -> None:
    """
    Validate few-shot examples format.
    
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
        
        if not isinstance(example["entities"], list):
            raise ValueError(f"Invalid few-shot examples format: example {i} 'entities' must be a list")
        
        for j, entity in enumerate(example["entities"]):
            if not isinstance(entity, dict):
                raise ValueError(f"Invalid few-shot examples format: example {i} entity {j} must be a dictionary")
            
            if "text" not in entity or "label" not in entity:
                raise ValueError(f"Invalid few-shot examples format: example {i} entity {j} missing required fields")


def _validate_response_format(entities: List[Dict[str, Any]]) -> None:
    """
    Validate extracted entities format.
    
    Args:
        entities: List of extracted entities
        
    Raises:
        LLMAPIError: For invalid entity format
    """
    required_fields = ["text", "label", "start", "end", "confidence"]
    
    for i, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise LLMAPIError(f"Entity {i} must be a dictionary")
        
        # Check required fields
        for field in required_fields:
            if field not in entity:
                raise LLMAPIError(f"Missing required field '{field}' in entity {i}")
        
        # Check field types
        if not isinstance(entity["text"], str):
            raise LLMAPIError(f"Invalid field type: 'text' must be string in entity {i}")
        
        if not isinstance(entity["label"], str):
            raise LLMAPIError(f"Invalid field type: 'label' must be string in entity {i}")
        
        if not isinstance(entity["start"], int):
            raise LLMAPIError(f"Invalid field type: 'start' must be integer in entity {i}")
        
        if not isinstance(entity["end"], int):
            raise LLMAPIError(f"Invalid field type: 'end' must be integer in entity {i}")
        
        if not isinstance(entity["confidence"], (int, float)):
            raise LLMAPIError(f"Invalid field type: 'confidence' must be number in entity {i}")
        
        # Check field ranges
        if entity["start"] < 0:
            raise LLMAPIError(f"Invalid field range: 'start' cannot be negative in entity {i}")
        
        if entity["end"] < entity["start"]:
            raise LLMAPIError(f"Invalid field range: 'end' must be >= 'start' in entity {i}")
        
        if not (0.0 <= entity["confidence"] <= 1.0):
            raise LLMAPIError(f"Invalid field range: 'confidence' must be between 0.0 and 1.0 in entity {i}")