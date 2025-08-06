"""
Benchmarker module for evaluating LLM performance on Named Entity Recognition (NER)
and Relationship Extraction (RE) tasks for the AIM2-ODIE ontology development system.

This module provides functionality to calculate precision, recall, and F1 scores for
both NER and RE tasks, enabling systematic evaluation of LLM performance against
gold standard datasets in plant metabolomics research.

Key Functions:
- calculate_ner_metrics: Evaluates NER performance with exact entity matching
- calculate_relation_metrics: Evaluates RE performance with exact tuple matching
- run_benchmark: Orchestrates end-to-end evaluation across multiple documents

The evaluation uses strict exact matching for both entities and relations to ensure
precise performance measurement.

Author: AIM2-ODIE System
Date: 2025-08-06
"""

from typing import List, Dict, Tuple, Callable, Any


def calculate_ner_metrics(
    gold_entities: List[Dict], predicted_entities: List[Dict]
) -> Dict[str, float]:
    """
    Calculate Named Entity Recognition (NER) evaluation metrics.

    Computes precision, recall, and F1 scores by comparing predicted entities
    against gold standard entities using exact matching on entity type, text,
    start character position, and end character position.

    Args:
        gold_entities: List of gold standard entity dictionaries, each containing:
            - entity_type (str): The type/category of the entity
            - text (str): The entity text span
            - start_char (int): Starting character position in the document
            - end_char (int): Ending character position in the document
        predicted_entities: List of predicted entity dictionaries with same
                            structure

    Returns:
        Dictionary containing evaluation metrics:
            - precision (float): True positives / (True positives + False
                                positives)
            - recall (float): True positives / (True positives + False negatives)
            - f1 (float): Harmonic mean of precision and recall

    Raises:
        ValueError: If inputs are not lists, contain non-dict elements, or
                   missing required fields

    Examples:
        >>> gold = [{'entity_type': 'COMPOUND', 'text': 'quercetin',
        ...          'start_char': 0, 'end_char': 9}]
        >>> pred = [{'entity_type': 'COMPOUND', 'text': 'quercetin',
        ...          'start_char': 0, 'end_char': 9}]
        >>> metrics = calculate_ner_metrics(gold, pred)
        >>> metrics['f1']
        1.0
    """
    # Input validation
    if not isinstance(gold_entities, list):
        raise ValueError("gold_entities must be a list")
    if not isinstance(predicted_entities, list):
        raise ValueError("predicted_entities must be a list")

    # Validate entity structure
    required_fields = ['entity_type', 'text', 'start_char', 'end_char']

    for i, entity in enumerate(gold_entities):
        if not isinstance(entity, dict):
            raise ValueError(f"gold_entities[{i}] must be a dictionary")
        for field in required_fields:
            if field not in entity:
                raise ValueError(
                    f"gold_entities[{i}] missing required field: {field}"
                )

    for i, entity in enumerate(predicted_entities):
        if not isinstance(entity, dict):
            raise ValueError(f"predicted_entities[{i}] must be a dictionary")
        for field in required_fields:
            if field not in entity:
                raise ValueError(
                    f"predicted_entities[{i}] missing required field: {field}"
                )

    # Handle empty inputs
    if len(gold_entities) == 0 and len(predicted_entities) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}

    if len(gold_entities) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    if len(predicted_entities) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    # Calculate matches based on exact entity comparison (type, text, span)
    gold_set = set()
    for entity in gold_entities:
        key = (
            entity['entity_type'],
            entity['text'],
            entity['start_char'],
            entity['end_char']
        )
        gold_set.add(key)

    predicted_set = set()
    for entity in predicted_entities:
        key = (
            entity['entity_type'],
            entity['text'],
            entity['start_char'],
            entity['end_char']
        )
        predicted_set.add(key)

    # Calculate true positives, false positives, and false negatives
    true_positives = len(gold_set.intersection(predicted_set))
    false_positives = len(predicted_set - gold_set)
    false_negatives = len(gold_set - predicted_set)

    # Calculate metrics
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0 else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0 else 0.0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_relation_metrics(
    gold_relations: List[Tuple], predicted_relations: List[Tuple]
) -> Dict[str, float]:
    """
    Calculate Relationship Extraction (RE) evaluation metrics.

    Computes precision, recall, and F1 scores by comparing predicted relations
    against gold standard relations using exact tuple matching.

    Args:
        gold_relations: List of gold standard relation tuples, each containing:
            (subject, relation_type, object) as strings
        predicted_relations: List of predicted relation tuples with same
                            structure

    Returns:
        Dictionary containing evaluation metrics:
            - precision (float): True positives / (True positives + False
                                positives)
            - recall (float): True positives / (True positives + False negatives)
            - f1 (float): Harmonic mean of precision and recall

    Raises:
        ValueError: If inputs are not lists, contain non-tuple elements, or
                   tuples don't have exactly 3 elements

    Examples:
        >>> gold = [('quercetin', 'found_in', 'Arabidopsis thaliana')]
        >>> pred = [('quercetin', 'found_in', 'Arabidopsis thaliana')]
        >>> metrics = calculate_relation_metrics(gold, pred)
        >>> metrics['f1']
        1.0
    """
    # Input validation
    if not isinstance(gold_relations, list):
        raise ValueError("gold_relations must be a list")
    if not isinstance(predicted_relations, list):
        raise ValueError("predicted_relations must be a list")

    # Validate relation structure (tuples with 3 elements)
    for i, relation in enumerate(gold_relations):
        if not isinstance(relation, tuple):
            raise ValueError(f"gold_relations[{i}] must be a tuple")
        if len(relation) != 3:
            raise ValueError(
                f"gold_relations[{i}] must be a tuple of length 3 "
                "(subject, relation, object)"
            )

    for i, relation in enumerate(predicted_relations):
        if not isinstance(relation, tuple):
            raise ValueError(f"predicted_relations[{i}] must be a tuple")
        if len(relation) != 3:
            raise ValueError(
                f"predicted_relations[{i}] must be a tuple of length 3 "
                "(subject, relation, object)"
            )

    # Handle empty inputs
    if len(gold_relations) == 0 and len(predicted_relations) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}

    if len(gold_relations) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    if len(predicted_relations) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    # Calculate matches based on exact tuple comparison
    gold_set = set(gold_relations)
    predicted_set = set(predicted_relations)

    # Calculate true positives, false positives, and false negatives
    true_positives = len(gold_set.intersection(predicted_set))
    false_positives = len(predicted_set - gold_set)
    false_negatives = len(gold_set - predicted_set)

    # Calculate metrics
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0 else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0 else 0.0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def run_benchmark(
    gold_standard_data: List[Dict],
    llm_ner_function: Callable[[str], List[Dict]],
    llm_relation_function: Callable[[str, List[Dict]], List[Tuple]]
) -> Dict[str, Any]:
    """
    Run comprehensive benchmark evaluation on gold standard dataset.

    Orchestrates end-to-end evaluation by calling provided LLM functions on each
    document in the gold standard dataset, then aggregating results to compute
    overall performance metrics for both NER and RE tasks.

    Args:
        gold_standard_data: List of gold standard document dictionaries, each
                           containing:
            - text (str): The document text for processing
            - entities (List[Dict]): Gold standard entities in the document
            - relations (List[Tuple]): Gold standard relations in the document
        llm_ner_function: Callable that takes document text and returns
                         predicted entities
        llm_relation_function: Callable that takes text and entities, returns
                              predicted relations

    Returns:
        Dictionary containing comprehensive benchmark results:
            - ner_metrics (Dict[str, float]): Aggregated NER precision, recall,
                                             F1
            - relation_metrics (Dict[str, float]): Aggregated RE precision,
                                                  recall, F1
            - documents_processed (int): Number of documents processed
            - total_gold_entities (int): Total gold entities across all documents
            - total_predicted_entities (int): Total predicted entities across all
                                             documents
            - total_gold_relations (int): Total gold relations across all
                                         documents
            - total_predicted_relations (int): Total predicted relations across
                                              all documents

    Raises:
        ValueError: If inputs have invalid types or structure
        RuntimeError: If LLM function calls fail during processing

    Examples:
        >>> def mock_ner(text): return [{'entity_type': 'COMPOUND',
        ...                             'text': 'test', 'start_char': 0,
        ...                             'end_char': 4}]
        >>> def mock_re(text, entities): return [('subj', 'rel', 'obj')]
        >>> gold_data = [{'text': 'test', 'entities': [], 'relations': []}]
        >>> result = run_benchmark(gold_data, mock_ner, mock_re)
        >>> result['documents_processed']
        1
    """
    # Input validation
    if not isinstance(gold_standard_data, list):
        raise ValueError("gold_standard_data must be a list")
    if not callable(llm_ner_function):
        raise ValueError("llm_ner_function must be callable")
    if not callable(llm_relation_function):
        raise ValueError("llm_relation_function must be callable")

    # Validate gold standard data structure
    required_fields = ['text', 'entities', 'relations']
    for i, document in enumerate(gold_standard_data):
        if not isinstance(document, dict):
            raise ValueError(f"gold_standard_data[{i}] must be a dictionary")
        for field in required_fields:
            if field not in document:
                raise ValueError(
                    f"gold_standard_data[{i}] missing required field: {field}"
                )

    # Handle empty input
    if len(gold_standard_data) == 0:
        return {
            'ner_metrics': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'relation_metrics': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'documents_processed': 0,
            'total_gold_entities': 0,
            'total_predicted_entities': 0,
            'total_gold_relations': 0,
            'total_predicted_relations': 0
        }

    # Initialize aggregated metrics
    all_gold_entities = []
    all_predicted_entities = []
    all_gold_relations = []
    all_predicted_relations = []

    # Process each document
    for document in gold_standard_data:
        text = document['text']
        gold_entities = document['entities']
        gold_relations = document['relations']

        # Call LLM functions with error handling
        try:
            predicted_entities = llm_ner_function(text)
            predicted_relations = llm_relation_function(text, predicted_entities)
        except Exception as e:
            raise RuntimeError(f"LLM function call failed: {str(e)}")

        # Aggregate for overall metrics
        all_gold_entities.extend(gold_entities)
        all_predicted_entities.extend(predicted_entities)
        all_gold_relations.extend(gold_relations)
        all_predicted_relations.extend(predicted_relations)

    # Calculate overall metrics
    ner_metrics = calculate_ner_metrics(all_gold_entities, all_predicted_entities)
    relation_metrics = calculate_relation_metrics(
        all_gold_relations, all_predicted_relations
    )

    return {
        'ner_metrics': ner_metrics,
        'relation_metrics': relation_metrics,
        'documents_processed': len(gold_standard_data),
        'total_gold_entities': len(all_gold_entities),
        'total_predicted_entities': len(all_predicted_entities),
        'total_gold_relations': len(all_gold_relations),
        'total_predicted_relations': len(all_predicted_relations)
    }
