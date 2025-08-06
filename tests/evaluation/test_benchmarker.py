"""
Unit tests for src/evaluation/benchmarker.py

This module provides comprehensive unit tests for the benchmarker functionality
that evaluates LLM performance on Named Entity Recognition (NER) and Relationship 
Extraction (RE) tasks for plant metabolomics information extraction.

Test Coverage:
- calculate_ner_metrics: precision, recall, F1 calculation with various scenarios
- calculate_relation_metrics: precision, recall, F1 for relationship extraction
- run_benchmark: end-to-end benchmarking with mock LLM functions
- Edge cases: empty inputs, no matches, partial matches
- Error handling: invalid inputs, malformed data

The benchmarker will be implemented in a later task (AIM2-ODIE-031-T2), so this
test file uses comprehensive mocking to define the expected API interface and behavior.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Tuple, Callable, Any


class TestBenchmarker:
    """Comprehensive test class for benchmarker functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_mock_benchmarker(self):
        """Set up mock benchmarker functions for each test."""
        
        def calculate_ner_metrics(gold_entities: List[Dict], predicted_entities: List[Dict]) -> Dict[str, float]:
            """Mock implementation of NER metrics calculation."""
            # Input validation
            if not isinstance(gold_entities, list):
                raise ValueError("gold_entities must be a list")
            if not isinstance(predicted_entities, list):
                raise ValueError("predicted_entities must be a list")
            
            # Validate entity structure
            for i, entity in enumerate(gold_entities):
                if not isinstance(entity, dict):
                    raise ValueError(f"gold_entities[{i}] must be a dictionary")
                required_fields = ['entity_type', 'text', 'start_char', 'end_char']
                for field in required_fields:
                    if field not in entity:
                        raise ValueError(f"gold_entities[{i}] missing required field: {field}")
            
            for i, entity in enumerate(predicted_entities):
                if not isinstance(entity, dict):
                    raise ValueError(f"predicted_entities[{i}] must be a dictionary")
                required_fields = ['entity_type', 'text', 'start_char', 'end_char']
                for field in required_fields:
                    if field not in entity:
                        raise ValueError(f"predicted_entities[{i}] missing required field: {field}")
            
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
                key = (entity['entity_type'], entity['text'], entity['start_char'], entity['end_char'])
                gold_set.add(key)
            
            predicted_set = set()
            for entity in predicted_entities:
                key = (entity['entity_type'], entity['text'], entity['start_char'], entity['end_char'])
                predicted_set.add(key)
            
            true_positives = len(gold_set.intersection(predicted_set))
            false_positives = len(predicted_set - gold_set)
            false_negatives = len(gold_set - predicted_set)
            
            # Calculate metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        def calculate_relation_metrics(gold_relations: List[Tuple], predicted_relations: List[Tuple]) -> Dict[str, float]:
            """Mock implementation of relation metrics calculation."""
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
                    raise ValueError(f"gold_relations[{i}] must be a tuple of length 3 (subject, relation, object)")
            
            for i, relation in enumerate(predicted_relations):
                if not isinstance(relation, tuple):
                    raise ValueError(f"predicted_relations[{i}] must be a tuple")
                if len(relation) != 3:
                    raise ValueError(f"predicted_relations[{i}] must be a tuple of length 3 (subject, relation, object)")
            
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
            
            true_positives = len(gold_set.intersection(predicted_set))
            false_positives = len(predicted_set - gold_set)
            false_negatives = len(gold_set - predicted_set)
            
            # Calculate metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
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
            """Mock implementation of benchmark runner."""
            # Input validation
            if not isinstance(gold_standard_data, list):
                raise ValueError("gold_standard_data must be a list")
            if not callable(llm_ner_function):
                raise ValueError("llm_ner_function must be callable")
            if not callable(llm_relation_function):
                raise ValueError("llm_relation_function must be callable")
            
            # Validate gold standard data structure
            for i, document in enumerate(gold_standard_data):
                if not isinstance(document, dict):
                    raise ValueError(f"gold_standard_data[{i}] must be a dictionary")
                required_fields = ['text', 'entities', 'relations']
                for field in required_fields:
                    if field not in document:
                        raise ValueError(f"gold_standard_data[{i}] missing required field: {field}")
            
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
                
                # Call LLM functions
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
            relation_metrics = calculate_relation_metrics(all_gold_relations, all_predicted_relations)
            
            return {
                'ner_metrics': ner_metrics,
                'relation_metrics': relation_metrics,
                'documents_processed': len(gold_standard_data),
                'total_gold_entities': len(all_gold_entities),
                'total_predicted_entities': len(all_predicted_entities),
                'total_gold_relations': len(all_gold_relations),
                'total_predicted_relations': len(all_predicted_relations)
            }
        
        # Store functions as instance attributes
        self.calculate_ner_metrics = calculate_ner_metrics
        self.calculate_relation_metrics = calculate_relation_metrics
        self.run_benchmark = run_benchmark

    def test_calculate_ner_metrics_perfect_match(self):
        """Test NER metrics calculation with perfect prediction."""
        gold_entities = [
            {
                'entity_type': 'COMPOUND',
                'text': 'quercetin',
                'start_char': 0,
                'end_char': 9
            },
            {
                'entity_type': 'ORGANISM',
                'text': 'Arabidopsis thaliana',
                'start_char': 20,
                'end_char': 40
            }
        ]
        
        predicted_entities = [
            {
                'entity_type': 'COMPOUND',
                'text': 'quercetin',
                'start_char': 0,
                'end_char': 9
            },
            {
                'entity_type': 'ORGANISM',
                'text': 'Arabidopsis thaliana',
                'start_char': 20,
                'end_char': 40
            }
        ]
        
        metrics = self.calculate_ner_metrics(gold_entities, predicted_entities)
        
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0

    def test_calculate_ner_metrics_no_match(self):
        """Test NER metrics calculation with no matches."""
        gold_entities = [
            {
                'entity_type': 'COMPOUND',
                'text': 'quercetin',
                'start_char': 0,
                'end_char': 9
            }
        ]
        
        predicted_entities = [
            {
                'entity_type': 'ORGANISM',
                'text': 'Arabidopsis',
                'start_char': 10,
                'end_char': 21
            }
        ]
        
        metrics = self.calculate_ner_metrics(gold_entities, predicted_entities)
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0

    def test_calculate_ner_metrics_partial_match(self):
        """Test NER metrics calculation with partial matches."""
        gold_entities = [
            {
                'entity_type': 'COMPOUND',
                'text': 'quercetin',
                'start_char': 0,
                'end_char': 9
            },
            {
                'entity_type': 'COMPOUND',
                'text': 'kaempferol',
                'start_char': 15,
                'end_char': 25
            }
        ]
        
        predicted_entities = [
            {
                'entity_type': 'COMPOUND',
                'text': 'quercetin',
                'start_char': 0,
                'end_char': 9
            },
            {
                'entity_type': 'ORGANISM',
                'text': 'tomato',
                'start_char': 30,
                'end_char': 36
            }
        ]
        
        metrics = self.calculate_ner_metrics(gold_entities, predicted_entities)
        
        # 1 true positive, 1 false positive, 1 false negative
        expected_precision = 1.0 / 2.0  # 0.5
        expected_recall = 1.0 / 2.0     # 0.5
        expected_f1 = 2 * (0.5 * 0.5) / (0.5 + 0.5)  # 0.5
        
        assert metrics['precision'] == expected_precision
        assert metrics['recall'] == expected_recall
        assert metrics['f1'] == expected_f1

    def test_calculate_ner_metrics_empty_gold(self):
        """Test NER metrics calculation with empty gold standard."""
        gold_entities = []
        predicted_entities = [
            {
                'entity_type': 'COMPOUND',
                'text': 'quercetin',
                'start_char': 0,
                'end_char': 9
            }
        ]
        
        metrics = self.calculate_ner_metrics(gold_entities, predicted_entities)
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0

    def test_calculate_ner_metrics_empty_predicted(self):
        """Test NER metrics calculation with empty predictions."""
        gold_entities = [
            {
                'entity_type': 'COMPOUND',
                'text': 'quercetin',
                'start_char': 0,
                'end_char': 9
            }
        ]
        predicted_entities = []
        
        metrics = self.calculate_ner_metrics(gold_entities, predicted_entities)
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0

    def test_calculate_ner_metrics_both_empty(self):
        """Test NER metrics calculation with both empty inputs."""
        gold_entities = []
        predicted_entities = []
        
        metrics = self.calculate_ner_metrics(gold_entities, predicted_entities)
        
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0

    def test_calculate_ner_metrics_type_mismatch(self):
        """Test NER metrics calculation with entity type mismatches."""
        gold_entities = [
            {
                'entity_type': 'COMPOUND',
                'text': 'quercetin',
                'start_char': 0,
                'end_char': 9
            }
        ]
        
        # Same text and span but different type - should not match
        predicted_entities = [
            {
                'entity_type': 'ORGANISM',
                'text': 'quercetin',
                'start_char': 0,
                'end_char': 9
            }
        ]
        
        metrics = self.calculate_ner_metrics(gold_entities, predicted_entities)
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0

    def test_calculate_ner_metrics_invalid_input_types(self):
        """Test NER metrics calculation with invalid input types."""
        # Test non-list inputs
        with pytest.raises(ValueError, match="gold_entities must be a list"):
            self.calculate_ner_metrics("not_a_list", [])
        
        with pytest.raises(ValueError, match="predicted_entities must be a list"):
            self.calculate_ner_metrics([], "not_a_list")
        
        # Test non-dict entities
        with pytest.raises(ValueError, match="gold_entities\\[0\\] must be a dictionary"):
            self.calculate_ner_metrics(["not_a_dict"], [])
        
        with pytest.raises(ValueError, match="predicted_entities\\[0\\] must be a dictionary"):
            self.calculate_ner_metrics([], ["not_a_dict"])

    def test_calculate_ner_metrics_missing_required_fields(self):
        """Test NER metrics calculation with missing required fields."""
        # Missing entity_type
        with pytest.raises(ValueError, match="missing required field: entity_type"):
            self.calculate_ner_metrics([{'text': 'test', 'start_char': 0, 'end_char': 4}], [])
        
        # Missing text
        with pytest.raises(ValueError, match="missing required field: text"):
            self.calculate_ner_metrics([{'entity_type': 'TEST', 'start_char': 0, 'end_char': 4}], [])
        
        # Missing start_char
        with pytest.raises(ValueError, match="missing required field: start_char"):
            self.calculate_ner_metrics([{'entity_type': 'TEST', 'text': 'test', 'end_char': 4}], [])
        
        # Missing end_char
        with pytest.raises(ValueError, match="missing required field: end_char"):
            self.calculate_ner_metrics([{'entity_type': 'TEST', 'text': 'test', 'start_char': 0}], [])

    def test_calculate_relation_metrics_perfect_match(self):
        """Test relation metrics calculation with perfect prediction."""
        gold_relations = [
            ('quercetin', 'found_in', 'Arabidopsis thaliana'),
            ('kaempferol', 'affects', 'drought tolerance')
        ]
        
        predicted_relations = [
            ('quercetin', 'found_in', 'Arabidopsis thaliana'),
            ('kaempferol', 'affects', 'drought tolerance')
        ]
        
        metrics = self.calculate_relation_metrics(gold_relations, predicted_relations)
        
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0

    def test_calculate_relation_metrics_no_match(self):
        """Test relation metrics calculation with no matches."""
        gold_relations = [
            ('quercetin', 'found_in', 'Arabidopsis thaliana')
        ]
        
        predicted_relations = [
            ('kaempferol', 'affects', 'drought tolerance')
        ]
        
        metrics = self.calculate_relation_metrics(gold_relations, predicted_relations)
        
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0

    def test_calculate_relation_metrics_partial_match(self):
        """Test relation metrics calculation with partial matches."""
        gold_relations = [
            ('quercetin', 'found_in', 'Arabidopsis thaliana'),
            ('kaempferol', 'affects', 'drought tolerance')
        ]
        
        predicted_relations = [
            ('quercetin', 'found_in', 'Arabidopsis thaliana'),
            ('resveratrol', 'exhibits', 'antioxidant activity')
        ]
        
        metrics = self.calculate_relation_metrics(gold_relations, predicted_relations)
        
        # 1 true positive, 1 false positive, 1 false negative
        expected_precision = 1.0 / 2.0  # 0.5
        expected_recall = 1.0 / 2.0     # 0.5
        expected_f1 = 2 * (0.5 * 0.5) / (0.5 + 0.5)  # 0.5
        
        assert metrics['precision'] == expected_precision
        assert metrics['recall'] == expected_recall
        assert metrics['f1'] == expected_f1

    def test_calculate_relation_metrics_empty_inputs(self):
        """Test relation metrics calculation with empty inputs."""
        # Both empty
        metrics = self.calculate_relation_metrics([], [])
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        
        # Empty gold
        metrics = self.calculate_relation_metrics([], [('a', 'b', 'c')])
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
        
        # Empty predicted
        metrics = self.calculate_relation_metrics([('a', 'b', 'c')], [])
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0

    def test_calculate_relation_metrics_invalid_input_types(self):
        """Test relation metrics calculation with invalid input types."""
        # Test non-list inputs
        with pytest.raises(ValueError, match="gold_relations must be a list"):
            self.calculate_relation_metrics("not_a_list", [])
        
        with pytest.raises(ValueError, match="predicted_relations must be a list"):
            self.calculate_relation_metrics([], "not_a_list")
        
        # Test non-tuple relations
        with pytest.raises(ValueError, match="gold_relations\\[0\\] must be a tuple"):
            self.calculate_relation_metrics(["not_a_tuple"], [])
        
        with pytest.raises(ValueError, match="predicted_relations\\[0\\] must be a tuple"):
            self.calculate_relation_metrics([], ["not_a_tuple"])

    def test_calculate_relation_metrics_invalid_tuple_length(self):
        """Test relation metrics calculation with invalid tuple lengths."""
        # Test tuple with wrong length
        with pytest.raises(ValueError, match="gold_relations\\[0\\] must be a tuple of length 3"):
            self.calculate_relation_metrics([('a', 'b')], [])
        
        with pytest.raises(ValueError, match="predicted_relations\\[0\\] must be a tuple of length 3"):
            self.calculate_relation_metrics([], [('a', 'b', 'c', 'd')])

    def test_run_benchmark_basic(self):
        """Test basic benchmark run with small dataset."""
        # Mock LLM functions
        mock_ner_function = Mock()
        mock_relation_function = Mock()
        
        # Set up mock return values
        mock_ner_function.return_value = [
            {
                'entity_type': 'COMPOUND',
                'text': 'quercetin',
                'start_char': 0,
                'end_char': 9
            }
        ]
        mock_relation_function.return_value = [
            ('quercetin', 'found_in', 'Arabidopsis thaliana')
        ]
        
        # Prepare gold standard data
        gold_standard_data = [
            {
                'text': 'Quercetin is found in Arabidopsis thaliana.',
                'entities': [
                    {
                        'entity_type': 'COMPOUND',
                        'text': 'quercetin',
                        'start_char': 0,
                        'end_char': 9
                    }
                ],
                'relations': [
                    ('quercetin', 'found_in', 'Arabidopsis thaliana')
                ]
            }
        ]
        
        result = self.run_benchmark(gold_standard_data, mock_ner_function, mock_relation_function)
        
        # Verify function calls
        mock_ner_function.assert_called_once_with('Quercetin is found in Arabidopsis thaliana.')
        mock_relation_function.assert_called_once()
        
        # Verify result structure
        assert 'ner_metrics' in result
        assert 'relation_metrics' in result
        assert 'documents_processed' in result
        assert 'total_gold_entities' in result
        assert 'total_predicted_entities' in result
        assert 'total_gold_relations' in result
        assert 'total_predicted_relations' in result
        
        # Check counts
        assert result['documents_processed'] == 1
        assert result['total_gold_entities'] == 1
        assert result['total_predicted_entities'] == 1
        assert result['total_gold_relations'] == 1
        assert result['total_predicted_relations'] == 1
        
        # Perfect match should give perfect scores
        assert result['ner_metrics']['f1'] == 1.0
        assert result['relation_metrics']['f1'] == 1.0

    def test_run_benchmark_multiple_documents(self):
        """Test benchmark run with multiple documents."""
        # Mock LLM functions
        mock_ner_function = Mock()
        mock_relation_function = Mock()
        
        # Set up mock return values for multiple calls
        mock_ner_function.side_effect = [
            [{'entity_type': 'COMPOUND', 'text': 'quercetin', 'start_char': 0, 'end_char': 9}],
            [{'entity_type': 'COMPOUND', 'text': 'kaempferol', 'start_char': 0, 'end_char': 10}]
        ]
        mock_relation_function.side_effect = [
            [('quercetin', 'found_in', 'plants')],
            [('kaempferol', 'affects', 'stress')]
        ]
        
        # Prepare gold standard data with multiple documents
        gold_standard_data = [
            {
                'text': 'Quercetin is found in plants.',
                'entities': [
                    {'entity_type': 'COMPOUND', 'text': 'quercetin', 'start_char': 0, 'end_char': 9}
                ],
                'relations': [
                    ('quercetin', 'found_in', 'plants')
                ]
            },
            {
                'text': 'Kaempferol affects stress responses.',
                'entities': [
                    {'entity_type': 'COMPOUND', 'text': 'kaempferol', 'start_char': 0, 'end_char': 10}
                ],
                'relations': [
                    ('kaempferol', 'affects', 'stress')
                ]
            }
        ]
        
        result = self.run_benchmark(gold_standard_data, mock_ner_function, mock_relation_function)
        
        # Verify function calls for both documents
        assert mock_ner_function.call_count == 2
        assert mock_relation_function.call_count == 2
        
        expected_ner_calls = [
            call('Quercetin is found in plants.'),
            call('Kaempferol affects stress responses.')
        ]
        mock_ner_function.assert_has_calls(expected_ner_calls)
        
        # Check counts
        assert result['documents_processed'] == 2
        assert result['total_gold_entities'] == 2
        assert result['total_predicted_entities'] == 2
        assert result['total_gold_relations'] == 2
        assert result['total_predicted_relations'] == 2

    def test_run_benchmark_empty_dataset(self):
        """Test benchmark run with empty dataset."""
        mock_ner_function = Mock()
        mock_relation_function = Mock()
        
        result = self.run_benchmark([], mock_ner_function, mock_relation_function)
        
        # No function calls should be made
        mock_ner_function.assert_not_called()
        mock_relation_function.assert_not_called()
        
        # All counts should be zero
        assert result['documents_processed'] == 0
        assert result['total_gold_entities'] == 0
        assert result['total_predicted_entities'] == 0
        assert result['total_gold_relations'] == 0
        assert result['total_predicted_relations'] == 0
        
        # Metrics should be zero for empty dataset
        assert result['ner_metrics']['precision'] == 0.0
        assert result['ner_metrics']['recall'] == 0.0
        assert result['ner_metrics']['f1'] == 0.0
        assert result['relation_metrics']['precision'] == 0.0
        assert result['relation_metrics']['recall'] == 0.0
        assert result['relation_metrics']['f1'] == 0.0

    def test_run_benchmark_llm_function_failure(self):
        """Test benchmark handling of LLM function failures."""
        # Mock LLM functions that raise exceptions
        mock_ner_function = Mock()
        mock_relation_function = Mock()
        
        mock_ner_function.side_effect = RuntimeError("NER model failed")
        
        gold_standard_data = [
            {
                'text': 'Test document',
                'entities': [],
                'relations': []
            }
        ]
        
        with pytest.raises(RuntimeError, match="LLM function call failed: NER model failed"):
            self.run_benchmark(gold_standard_data, mock_ner_function, mock_relation_function)

    def test_run_benchmark_invalid_inputs(self):
        """Test benchmark run with invalid inputs."""
        mock_ner_function = Mock()
        mock_relation_function = Mock()
        
        # Test non-list gold standard data
        with pytest.raises(ValueError, match="gold_standard_data must be a list"):
            self.run_benchmark("not_a_list", mock_ner_function, mock_relation_function)
        
        # Test non-callable LLM functions
        with pytest.raises(ValueError, match="llm_ner_function must be callable"):
            self.run_benchmark([], "not_callable", mock_relation_function)
        
        with pytest.raises(ValueError, match="llm_relation_function must be callable"):
            self.run_benchmark([], mock_ner_function, "not_callable")

    def test_run_benchmark_invalid_document_structure(self):
        """Test benchmark run with invalid document structure."""
        mock_ner_function = Mock()
        mock_relation_function = Mock()
        
        # Test non-dict document
        with pytest.raises(ValueError, match="gold_standard_data\\[0\\] must be a dictionary"):
            self.run_benchmark(["not_a_dict"], mock_ner_function, mock_relation_function)
        
        # Test missing required fields
        with pytest.raises(ValueError, match="missing required field: text"):
            self.run_benchmark([{'entities': [], 'relations': []}], mock_ner_function, mock_relation_function)
        
        with pytest.raises(ValueError, match="missing required field: entities"):
            self.run_benchmark([{'text': 'test', 'relations': []}], mock_ner_function, mock_relation_function)
        
        with pytest.raises(ValueError, match="missing required field: relations"):
            self.run_benchmark([{'text': 'test', 'entities': []}], mock_ner_function, mock_relation_function)

    def test_run_benchmark_complex_scenario(self):
        """Test benchmark with complex real-world scenario."""
        # Mock LLM functions with realistic behavior
        mock_ner_function = Mock()
        mock_relation_function = Mock()
        
        # Simulate imperfect NER: misses some entities, adds false positives
        mock_ner_function.return_value = [
            # Correctly identifies quercetin
            {
                'entity_type': 'COMPOUND',
                'text': 'quercetin',
                'start_char': 0,
                'end_char': 9
            },
            # Misses kaempferol, adds false positive
            {
                'entity_type': 'ORGANISM',
                'text': 'tomato',
                'start_char': 50,
                'end_char': 56
            }
        ]
        
        # Simulate imperfect RE: gets one relation right, adds false positive
        mock_relation_function.return_value = [
            ('quercetin', 'found_in', 'plants'),  # Correct
            ('tomato', 'contains', 'lycopene')    # False positive
        ]
        
        # Gold standard with multiple entities and relations
        gold_standard_data = [
            {
                'text': 'Quercetin and kaempferol are flavonoids found in plants.',
                'entities': [
                    {
                        'entity_type': 'COMPOUND',
                        'text': 'quercetin',
                        'start_char': 0,
                        'end_char': 9
                    },
                    {
                        'entity_type': 'COMPOUND',
                        'text': 'kaempferol',
                        'start_char': 14,
                        'end_char': 24
                    },
                    {
                        'entity_type': 'COMPOUND',
                        'text': 'flavonoids',
                        'start_char': 29,
                        'end_char': 39
                    }
                ],
                'relations': [
                    ('quercetin', 'found_in', 'plants'),
                    ('kaempferol', 'found_in', 'plants'),
                    ('quercetin', 'is_a', 'flavonoids'),
                    ('kaempferol', 'is_a', 'flavonoids')
                ]
            }
        ]
        
        result = self.run_benchmark(gold_standard_data, mock_ner_function, mock_relation_function)
        
        # Verify detailed metrics
        assert result['documents_processed'] == 1
        assert result['total_gold_entities'] == 3
        assert result['total_predicted_entities'] == 2
        assert result['total_gold_relations'] == 4
        assert result['total_predicted_relations'] == 2
        
        # NER metrics: 1 TP, 1 FP, 2 FN
        # Precision = 1/2 = 0.5, Recall = 1/3 ≈ 0.33, F1 = 2*(0.5*0.33)/(0.5+0.33) = 0.4
        ner_metrics = result['ner_metrics']
        assert ner_metrics['precision'] == 0.5
        assert abs(ner_metrics['recall'] - (1/3)) < 1e-10
        expected_ner_f1 = 2 * (0.5 * (1/3)) / (0.5 + (1/3))
        assert abs(ner_metrics['f1'] - expected_ner_f1) < 1e-10
        
        # Relation metrics: 1 TP, 1 FP, 3 FN
        # Precision = 1/2 = 0.5, Recall = 1/4 = 0.25, F1 = 2*(0.5*0.25)/(0.5+0.25) ≈ 0.33
        relation_metrics = result['relation_metrics']
        assert relation_metrics['precision'] == 0.5
        assert relation_metrics['recall'] == 0.25
        expected_rel_f1 = 2 * (0.5 * 0.25) / (0.5 + 0.25)
        assert abs(relation_metrics['f1'] - expected_rel_f1) < 1e-10


# Mark all tests in this module as evaluation related
pytestmark = [pytest.mark.unit, pytest.mark.evaluation]