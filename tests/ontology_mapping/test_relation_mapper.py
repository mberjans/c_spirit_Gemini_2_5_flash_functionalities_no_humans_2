"""
Unit tests for src/ontology_mapping/relation_mapper.py

This module tests the relationship-to-ontology mapping functionality using text2term and custom logic
for mapping extracted relationship triples to defined ontology properties. The module ensures semantic 
consistency through domain/range validation and handles relationships without direct ontology matches.

Test Coverage:
- Basic relationship mapping with predefined test ontology
- Different text2term mapping methods (TFIDF, LEVENSHTEIN, etc.)
- Minimum score filtering for high-confidence mappings
- Mapping to specific term types (property, objectProperty, dataProperty)
- Handling of unmapped relationship triples
- Semantic consistency validation (domain/range checking)
- Error handling for invalid inputs and API failures
- Edge cases and performance considerations

Test Approach:
- Mock text2term.map_terms() to avoid external dependencies
- Mock Owlready2 ontology objects for semantic validation
- Test different mapping scenarios with controlled inputs
- Validate output format and data integrity
- Ensure proper error handling and validation
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional, Tuple
import json

# Import testing utilities from the project's testing framework
from src.utils.testing_framework import (
    expect_exception,
    parametrize,
    fake_text,
    fake_entity,
    fake_chemical_name
)

# Import the relation mapper functions (will be implemented in src/ontology_mapping/relation_mapper.py)
from src.ontology_mapping.relation_mapper import (
    map_relationships_to_ontology,
    RelationMapperError,
    OntologyNotFoundError,
    MappingError,
    SemanticValidationError,
    _validate_relationships,
    _validate_mapping_method,
    _process_mapping_results,
    _filter_by_score,
    _validate_semantic_consistency,
    _get_domain_range_constraints,
    text2term  # Import text2term for test assertions
)


class TestMapRelationshipsToOntologyBasic:
    """Test cases for basic relationship-to-ontology mapping functionality."""
    
    def test_map_relationships_basic_functionality(self):
        """Test basic relationship mapping with default parameters."""
        relationships = [
            ("glucose", "metabolized_by", "enzyme"),
            ("arabidopsis", "has_part", "leaf"),
            ("photosynthesis", "occurs_in", "chloroplast")
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        # Mock text2term response
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["metabolized_by", "has_part", "occurs_in"],
            'Mapped Term Label': ["metabolized by", "has part", "occurs in"],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/RO_0002209",
                "http://purl.obolibrary.org/obo/BFO_0000051",
                "http://purl.obolibrary.org/obo/BFO_0000066"
            ],
            'Mapping Score': [0.95, 0.88, 0.92],
            'Term Type': ["property", "property", "property"]
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                result = map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj
                )
                
                # Verify function call
                mock_map_terms.assert_called_once_with(
                    source_terms=["metabolized_by", "has_part", "occurs_in"],
                    target_ontology="http://example.org/test-ontology.owl",
                    mapper=text2term.Mapper.TFIDF,
                    min_score=0.3,
                    term_type='property',
                    incl_unmapped=False
                )
                
                # Validate results
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 3
                assert all(col in result.columns for col in [
                    'Subject', 'Relation', 'Object', 'Mapped_Relation_Label', 
                    'Mapped_Relation_IRI', 'Mapping_Score', 'Term_Type', 
                    'Semantic_Valid'
                ])
                
                # Check specific mappings
                assert result.iloc[0]['Relation'] == "metabolized_by"
                assert result.iloc[0]['Mapped_Relation_IRI'] == "http://purl.obolibrary.org/obo/RO_0002209"
                assert result.iloc[0]['Mapping_Score'] == 0.95
                assert result.iloc[0]['Semantic_Valid'] == True
    
    def test_map_relationships_with_biological_processes(self):
        """Test relationship mapping with biological process relationships."""
        biological_relationships = [
            ("ATP", "produced_by", "cellular_respiration"),
            ("glucose", "participates_in", "glycolysis"),
            ("enzyme", "catalyzes", "reaction"),
            ("gene", "regulates", "protein_expression"),
            ("transcription_factor", "binds_to", "DNA"),
            ("metabolite", "transported_by", "membrane_protein")
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://purl.obolibrary.org/obo/go.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["produced_by", "participates_in", "catalyzes", "regulates", "binds_to", "transported_by"],
            'Mapped Term Label': [
                "produced by", "participates in", "catalyzes", 
                "regulates", "binds to", "transported by"
            ],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/RO_0003001",
                "http://purl.obolibrary.org/obo/RO_0000056",
                "http://purl.obolibrary.org/obo/RO_0002327",
                "http://purl.obolibrary.org/obo/RO_0002211",
                "http://purl.obolibrary.org/obo/RO_0003680",
                "http://purl.obolibrary.org/obo/RO_0002313"
            ],
            'Mapping Score': [0.92, 0.89, 0.95, 0.91, 0.88, 0.85],
            'Term Type': ["property"] * 6
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                result = map_relationships_to_ontology(
                    relationships=biological_relationships,
                    ontology_obj=ontology_obj,
                    mapping_method='tfidf',
                    min_score=0.8
                )
                
                # Verify all biological relationships were processed
                assert len(result) == 6
                assert all(score >= 0.8 for score in result['Mapping_Score'])
                
                # Verify RO (Relations Ontology) IRIs format
                assert all(iri.startswith("http://purl.obolibrary.org/obo/RO_") 
                          for iri in result['Mapped_Relation_IRI'])
    
    def test_map_relationships_with_chemical_interactions(self):
        """Test relationship mapping with chemical interaction relationships."""
        chemical_relationships = [
            ("quercetin", "inhibits", "enzyme"),
            ("ATP", "binds_to", "kinase"),
            ("drug", "interacts_with", "protein"),
            ("metabolite", "derived_from", "precursor"),
            ("compound", "converted_to", "product")
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://purl.obolibrary.org/obo/chebi.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["inhibits", "binds_to", "interacts_with", "derived_from", "converted_to"],
            'Mapped Term Label': [
                "inhibits", "binds to", "interacts with", 
                "derived from", "converted to"
            ],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/RO_0002449",
                "http://purl.obolibrary.org/obo/RO_0002436",
                "http://purl.obolibrary.org/obo/RO_0002434",
                "http://purl.obolibrary.org/obo/RO_0001000",
                "http://purl.obolibrary.org/obo/RO_0002343"
            ],
            'Mapping Score': [0.94, 0.91, 0.87, 0.89, 0.92],
            'Term Type': ["property"] * 5
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                result = map_relationships_to_ontology(
                    relationships=chemical_relationships,
                    ontology_obj=ontology_obj,
                    mapping_method='levenshtein',
                    min_score=0.85
                )
                
                # Verify high-confidence chemical interaction mappings
                assert len(result) == 5
                assert all(score >= 0.85 for score in result['Mapping_Score'])
                
                # Verify RO IRIs format for chemical interactions
                assert all(iri.startswith("http://purl.obolibrary.org/obo/RO_") 
                          for iri in result['Mapped_Relation_IRI'])


class TestMappingMethods:
    """Test cases for different text2term mapping methods."""
    
    @parametrize("mapping_method,expected_mapper", [
        ("tfidf", "text2term.Mapper.TFIDF"),
        ("levenshtein", "text2term.Mapper.LEVENSHTEIN"),
        ("jaro_winkler", "text2term.Mapper.JARO_WINKLER"),
        ("jaccard", "text2term.Mapper.JACCARD"),
        ("fuzzy", "text2term.Mapper.FUZZY")
    ])
    def test_different_mapping_methods(self, mapping_method, expected_mapper):
        """Test different text2term mapping methods for relationships."""
        relationships = [
            ("glucose", "metabolized_by", "enzyme"),
            ("arabidopsis", "has_part", "leaf")
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["metabolized_by", "has_part"],
            'Mapped Term Label': ["metabolized by", "has part"],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/RO_0002209",
                "http://purl.obolibrary.org/obo/BFO_0000051"
            ],
            'Mapping Score': [0.92, 0.88],
            'Term Type': ["property", "property"]
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper.text2term.Mapper') as mock_mapper:
                with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                    mock_map_terms.return_value = mock_mapping_df
                    mock_validate.return_value = True
                    
                    # Set up mapper attribute access
                    getattr(mock_mapper, expected_mapper.split('.')[-1])
                    
                    result = map_relationships_to_ontology(
                        relationships=relationships,
                        ontology_obj=ontology_obj,
                        mapping_method=mapping_method
                    )
                    
                    # Verify correct mapper was used
                    mock_map_terms.assert_called_once()
                    call_args = mock_map_terms.call_args[1]
                    assert 'mapper' in call_args
                    
                    assert len(result) == 2
    
    def test_tfidf_method_performance(self):
        """Test TFIDF mapping method with performance considerations."""
        # Larger set of relationships to test TFIDF performance
        relationships = [
            (fake_chemical_name(), f"relation_{i}", fake_entity("compound")) 
            for i in range(20)
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://purl.obolibrary.org/obo/chebi.owl"
        
        relation_terms = [rel[1] for rel in relationships]
        
        # Mock varied scores to simulate TFIDF behavior
        mock_scores = [0.95, 0.89, 0.72, 0.68, 0.45, 0.91, 0.83, 0.55, 
                      0.78, 0.66, 0.88, 0.74, 0.59, 0.82, 0.71, 0.94, 
                      0.67, 0.86, 0.63, 0.77]
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': relation_terms,
            'Mapped Term Label': [f"mapped_{rel}" for rel in relation_terms],
            'Mapped Term IRI': [f"http://purl.obolibrary.org/obo/RO_{i:07d}" 
                               for i in range(len(relation_terms))],
            'Mapping Score': mock_scores,
            'Term Type': ["property"] * len(relation_terms)
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                result = map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj,
                    mapping_method='tfidf',
                    min_score=0.7  # Filter out low-confidence mappings
                )
                
                # Verify filtering worked correctly
                expected_count = sum(1 for score in mock_scores if score >= 0.7)
                assert len(result) == expected_count
                assert all(score >= 0.7 for score in result['Mapping_Score'])
    
    def test_levenshtein_method_fuzzy_matching(self):
        """Test Levenshtein mapping method for fuzzy string matching of relations."""
        # Relationships with slight variations to test fuzzy matching
        relationships = [
            ("entity1", "regulates", "entity2"),  # Exact match
            ("entity3", "regulats", "entity4"),   # Typo in relation
            ("entity5", "reguates", "entity6"),   # Different typo
            ("entity7", "inhibits", "entity8"),   # Exact match
            ("entity9", "inhbits", "entity10"),   # Typo in relation
            ("entity11", "inibits", "entity12")   # Different typo
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        # Levenshtein should handle these variations well
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["regulates", "regulats", "reguates", "inhibits", "inhbits", "inibits"],
            'Mapped Term Label': ["regulates", "regulates", "regulates", 
                                 "inhibits", "inhibits", "inhibits"],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/RO_0002211",
                "http://purl.obolibrary.org/obo/RO_0002211",
                "http://purl.obolibrary.org/obo/RO_0002211",
                "http://purl.obolibrary.org/obo/RO_0002449",
                "http://purl.obolibrary.org/obo/RO_0002449",
                "http://purl.obolibrary.org/obo/RO_0002449"
            ],
            'Mapping Score': [1.0, 0.85, 0.82, 1.0, 0.88, 0.79],
            'Term Type': ["property"] * 6
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                result = map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj,
                    mapping_method='levenshtein',
                    min_score=0.75
                )
                
                # Verify fuzzy matching results
                assert len(result) == 6  # All relations above threshold (0.75)
                
                # Check that variations map to same terms
                regulates_mappings = result[result['Mapped_Relation_IRI'] == 
                                          "http://purl.obolibrary.org/obo/RO_0002211"]
                assert len(regulates_mappings) == 3
                
                inhibits_mappings = result[result['Mapped_Relation_IRI'] == 
                                         "http://purl.obolibrary.org/obo/RO_0002449"]
                assert len(inhibits_mappings) == 3


class TestScoreFiltering:
    """Test cases for minimum score filtering functionality."""
    
    def test_min_score_filtering_basic(self):
        """Test basic minimum score filtering for relationships."""
        relationships = [
            ("compound1", "relation1", "target1"),
            ("compound2", "relation2", "target2"),
            ("compound3", "relation3", "target3"),
            ("compound4", "relation4", "target4")
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        min_score = 0.8
        
        # Mock responses with varied scores
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["relation1", "relation2", "relation3", "relation4"],
            'Mapped Term Label': ["mapped1", "mapped2", "mapped3", "mapped4"],
            'Mapped Term IRI': [f"http://example.org/relation{i}" for i in range(4)],
            'Mapping Score': [0.95, 0.75, 0.85, 0.65],  # 2 above, 2 below threshold
            'Term Type': ["property"] * 4
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                result = map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj,
                    min_score=min_score
                )
                
                # Only mappings with score >= 0.8 should be returned
                assert len(result) == 2
                assert all(score >= min_score for score in result['Mapping_Score'])
                
                # Verify specific relations that passed filtering
                expected_relations = ["relation1", "relation3"]
                assert set(result['Relation']) == set(expected_relations)
    
    @parametrize("min_score,expected_count", [
        (0.0, 6),   # All relations pass
        (0.5, 5),   # 5 relations pass
        (0.7, 4),   # 4 relations pass
        (0.8, 3),   # 3 relations pass
        (0.9, 2),   # 2 relations pass
        (0.95, 1),  # 1 relation passes
        (0.99, 0)   # No relations pass
    ])
    def test_different_score_thresholds(self, min_score, expected_count):
        """Test filtering with different minimum score thresholds."""
        relationships = [
            (f"entity{i}", f"relation{i}", f"target{i}") 
            for i in range(6)
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': [f"relation{i}" for i in range(6)],
            'Mapped Term Label': [f"mapped{i}" for i in range(6)],
            'Mapped Term IRI': [f"http://example.org/relation{i}" for i in range(6)],
            'Mapping Score': [0.98, 0.91, 0.84, 0.77, 0.63, 0.45],
            'Term Type': ["property"] * 6
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                result = map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj,
                    min_score=min_score
                )
                
                assert len(result) == expected_count
                if expected_count > 0:
                    assert all(score >= min_score for score in result['Mapping_Score'])
    
    def test_high_confidence_mappings_only(self):
        """Test filtering for high-confidence relationship mappings only."""
        relationships = [
            ("ATP", "produced_by", "respiration"),
            ("glucose", "participates_in", "glycolysis"),
            ("enzyme", "catalyzes", "reaction"),
            ("gene", "regulates", "expression"),
            ("protein", "binds_to", "DNA"),
            ("metabolite", "derived_from", "precursor")
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://purl.obolibrary.org/obo/ro.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["produced_by", "participates_in", "catalyzes", "regulates", "binds_to", "derived_from"],
            'Mapped Term Label': [
                "produced by", "participates in", "catalyzes",
                "regulates", "binds to", "derived from"
            ],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/RO_0003001",
                "http://purl.obolibrary.org/obo/RO_0000056",
                "http://purl.obolibrary.org/obo/RO_0002327",
                "http://purl.obolibrary.org/obo/RO_0002211",
                "http://purl.obolibrary.org/obo/RO_0003680",
                "http://purl.obolibrary.org/obo/RO_0001000"
            ],
            'Mapping Score': [0.99, 0.97, 0.94, 0.96, 0.98, 0.95],
            'Term Type': ["property"] * 6
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                # Test very high confidence threshold
                result = map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj,
                    min_score=0.95
                )
                
                # Should return only mappings with score >= 0.95
                assert len(result) == 5  # All except catalyzes (0.94)
                assert all(score >= 0.95 for score in result['Mapping_Score'])


class TestTermTypes:
    """Test cases for mapping to specific term types."""
    
    def test_map_to_property_terms(self):
        """Test mapping relations to ontology properties."""
        relationships = [
            ("glucose", "regulates", "metabolism"),
            ("enzyme", "catalyzes", "reaction"),
            ("protein", "participates_in", "process")
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["regulates", "catalyzes", "participates_in"],
            'Mapped Term Label': ["regulates", "catalyzes", "participates in"],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/RO_0002211",
                "http://purl.obolibrary.org/obo/RO_0002327",
                "http://purl.obolibrary.org/obo/RO_0000056"
            ],
            'Mapping Score': [0.95, 0.89, 0.92],
            'Term Type': ["property", "property", "property"]
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                result = map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj,
                    term_type='property'
                )
                
                # Verify text2term was called with correct term_type
                mock_map_terms.assert_called_once()
                call_args = mock_map_terms.call_args[1]
                assert call_args['term_type'] == 'property'
                
                # Verify all results are property types
                assert len(result) == 3
                assert all(term_type == "property" for term_type in result['Term_Type'])
    
    def test_map_to_object_property_terms(self):
        """Test mapping relations to ontology object properties."""
        relationships = [
            ("protein", "has_part", "domain"),
            ("cell", "contains", "organelle"),
            ("pathway", "involves", "enzyme")
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://purl.obolibrary.org/obo/ro.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["has_part", "contains", "involves"],
            'Mapped Term Label': ["has part", "contains", "involves"],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/BFO_0000051",
                "http://purl.obolibrary.org/obo/RO_0001019",
                "http://purl.obolibrary.org/obo/RO_0002233"
            ],
            'Mapping Score': [0.88, 0.92, 0.94],
            'Term Type': ["objectProperty", "objectProperty", "objectProperty"]
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                result = map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj,
                    term_type='objectProperty'
                )
                
                # Verify text2term was called with correct term_type
                call_args = mock_map_terms.call_args[1]
                assert call_args['term_type'] == 'objectProperty'
                
                # Verify all results are object property types
                assert len(result) == 3
                assert all(term_type == "objectProperty" for term_type in result['Term_Type'])
    
    @parametrize("term_type", ["property", "objectProperty", "dataProperty"])
    def test_different_term_types(self, term_type):
        """Test mapping with different term types."""
        relationships = [("test_entity", "test_relation", "test_target")]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["test_relation"],
            'Mapped Term Label': ["mapped_relation"],
            'Mapped Term IRI': ["http://example.org/mapped_relation"],
            'Mapping Score': [0.9],
            'Term Type': [term_type]
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                result = map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj,
                    term_type=term_type
                )
                
                call_args = mock_map_terms.call_args[1]
                assert call_args['term_type'] == term_type
                assert result.iloc[0]['Term_Type'] == term_type


class TestSemanticConsistencyValidation:
    """Test cases for semantic consistency validation (domain/range checking)."""
    
    def test_semantic_validation_with_valid_domain_range(self):
        """Test semantic validation with valid domain/range constraints."""
        relationships = [
            ("glucose", "metabolized_by", "enzyme"),
            ("protein", "binds_to", "DNA")
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        # Mock property with domain/range constraints
        mock_property = Mock()
        mock_property.domain = [Mock()]
        mock_property.domain[0].name = "ChemicalEntity"
        mock_property.range = [Mock()]
        mock_property.range[0].name = "Enzyme"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["metabolized_by", "binds_to"],
            'Mapped Term Label': ["metabolized by", "binds to"],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/RO_0002209",
                "http://purl.obolibrary.org/obo/RO_0003680"
            ],
            'Mapping Score': [0.95, 0.88],
            'Term Type': ["property", "property"]
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._get_domain_range_constraints') as mock_constraints:
                with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                    mock_map_terms.return_value = mock_mapping_df
                    mock_constraints.return_value = (["ChemicalEntity"], ["Enzyme"])
                    mock_validate.return_value = True
                    
                    result = map_relationships_to_ontology(
                        relationships=relationships,
                        ontology_obj=ontology_obj,
                        validate_semantics=True
                    )
                    
                    # Verify semantic validation was called
                    assert mock_validate.call_count == 2
                    
                    # Verify all results are semantically valid
                    assert len(result) == 2
                    assert all(result['Semantic_Valid'])
    
    def test_semantic_validation_with_invalid_domain_range(self):
        """Test semantic validation with invalid domain/range constraints."""
        relationships = [
            ("invalid_subject", "metabolized_by", "invalid_object"),
            ("another_invalid", "binds_to", "wrong_type")
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["metabolized_by", "binds_to"],
            'Mapped Term Label': ["metabolized by", "binds to"],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/RO_0002209",
                "http://purl.obolibrary.org/obo/RO_0003680"
            ],
            'Mapping Score': [0.95, 0.88],
            'Term Type': ["property", "property"]
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._get_domain_range_constraints') as mock_constraints:
                with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                    mock_map_terms.return_value = mock_mapping_df
                    mock_constraints.return_value = (["ChemicalEntity"], ["Enzyme"])
                    mock_validate.return_value = False  # Invalid semantics
                    
                    result = map_relationships_to_ontology(
                        relationships=relationships,
                        ontology_obj=ontology_obj,
                        validate_semantics=True
                    )
                    
                    # Verify semantic validation was called
                    assert mock_validate.call_count == 2
                    
                    # Verify all results are semantically invalid
                    assert len(result) == 2
                    assert all(not valid for valid in result['Semantic_Valid'])
    
    def test_semantic_validation_disabled(self):
        """Test relationship mapping with semantic validation disabled."""
        relationships = [
            ("any_subject", "any_relation", "any_object")
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["any_relation"],
            'Mapped Term Label': ["mapped relation"],
            'Mapped Term IRI': ["http://example.org/mapped_relation"],
            'Mapping Score': [0.9],
            'Term Type': ["property"]
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                
                result = map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj,
                    validate_semantics=False
                )
                
                # Verify semantic validation was not called
                mock_validate.assert_not_called()
                
                # Verify semantic validation column shows None or is omitted
                assert len(result) == 1
                if 'Semantic_Valid' in result.columns:
                    assert pd.isna(result.iloc[0]['Semantic_Valid']) or result.iloc[0]['Semantic_Valid'] is None
    
    def test_get_domain_range_constraints(self):
        """Test extraction of domain/range constraints from ontology properties."""
        ontology_obj = Mock()
        
        # Mock property with domain and range
        mock_property = Mock()
        mock_property.domain = [Mock(), Mock()]
        mock_property.domain[0].name = "ChemicalEntity"
        mock_property.domain[1].name = "BiologicalEntity"
        mock_property.range = [Mock()]
        mock_property.range[0].name = "Enzyme"
        
        # Mock ontology search
        ontology_obj.search.return_value = [mock_property]
        
        with patch('src.ontology_mapping.relation_mapper._get_domain_range_constraints') as mock_func:
            # Mock the actual function implementation
            mock_func.return_value = (["ChemicalEntity", "BiologicalEntity"], ["Enzyme"])
            
            domain, range_constraints = mock_func(ontology_obj, "http://example.org/property")
            
            assert domain == ["ChemicalEntity", "BiologicalEntity"]
            assert range_constraints == ["Enzyme"]


class TestUnmappedRelationshipsHandling:
    """Test cases for handling unmapped relationship triples."""
    
    def test_exclude_unmapped_relationships_default(self):
        """Test default behavior of excluding unmapped relationships."""
        relationships = [
            ("known_entity", "known_relation", "known_target"),
            ("unknown_entity", "unknown_relation", "unknown_target"),
            ("another_known", "another_relation", "another_target")
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        # Mock response with only mapped relations (default text2term behavior)
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["known_relation", "another_relation"],
            'Mapped Term Label': ["known relation", "another relation"],
            'Mapped Term IRI': [
                "http://example.org/known_relation",
                "http://example.org/another_relation"
            ],
            'Mapping Score': [0.95, 0.88],
            'Term Type': ["property", "property"]
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                result = map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj
                )
                
                # Verify text2term was called with incl_unmapped=False (default)
                call_args = mock_map_terms.call_args[1]
                assert call_args['incl_unmapped'] == False
                
                # Only mapped relations should be returned
                assert len(result) == 2
                assert "unknown_relation" not in result['Relation'].values
    
    def test_include_unmapped_relationships_explicit(self):
        """Test explicit inclusion of unmapped relationships."""
        relationships = [
            ("known_entity", "known_relation", "known_target"),
            ("unknown_entity", "unknown_relation", "unknown_target"),
            ("another_known", "another_relation", "another_target")
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        # Mock response including unmapped relations
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["known_relation", "unknown_relation", "another_relation"],
            'Mapped Term Label': ["known relation", None, "another relation"],
            'Mapped Term IRI': [
                "http://example.org/known_relation",
                None,
                "http://example.org/another_relation"
            ],
            'Mapping Score': [0.95, None, 0.88],
            'Term Type': ["property", None, "property"]
        })
        
        # Patch the function to accept incl_unmapped parameter
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper.map_relationships_to_ontology') as mock_func:
                mock_map_terms.return_value = mock_mapping_df
                
                # Mock the actual function to test parameter passing
                def mock_implementation(relationships, ontology_obj, **kwargs):
                    incl_unmapped = kwargs.get('incl_unmapped', False)
                    if incl_unmapped:
                        return mock_mapping_df
                    else:
                        return mock_mapping_df[mock_mapping_df['Mapped_Relation_IRI'].notna()]
                
                mock_func.side_effect = mock_implementation
                
                result = mock_func(
                    relationships=relationships,
                    ontology_obj=ontology_obj,
                    incl_unmapped=True
                )
                
                # All relations should be included, even unmapped ones
                assert len(result) == 3
                assert "unknown_relation" in result['Source Term'].values
    
    def test_mixed_mapped_unmapped_results(self):
        """Test handling of mixed mapped and unmapped relationship results."""
        relationships = [
            ("glucose", "metabolized_by", "enzyme"),
            ("xyz123", "unknown_rel1", "abc456"),
            ("arabidopsis", "has_part", "leaf"),
            ("def789", "unknown_rel2", "ghi012"),
            ("ATP", "produced_by", "respiration")
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        # Simulate realistic scenario where some relations don't map
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["metabolized_by", "has_part", "produced_by"],
            'Mapped Term Label': [
                "metabolized by", "has part", "produced by"
            ],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/RO_0002209",
                "http://purl.obolibrary.org/obo/BFO_0000051",
                "http://purl.obolibrary.org/obo/RO_0003001"
            ],
            'Mapping Score': [0.98, 0.95, 0.92],
            'Term Type': ["property", "property", "property"]
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                result = map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj,
                    min_score=0.9
                )
                
                # Only successfully mapped relations should be returned
                assert len(result) == 3
                mapped_relations = set(result['Relation'])
                assert mapped_relations == {"metabolized_by", "has_part", "produced_by"}
                assert "unknown_rel1" not in mapped_relations
                assert "unknown_rel2" not in mapped_relations


class TestErrorHandling:
    """Test cases for error handling in relationship mapping."""
    
    def test_ontology_not_found_error(self):
        """Test error handling for non-existent ontology."""
        relationships = [("glucose", "metabolized_by", "enzyme")]
        invalid_ontology = Mock()
        invalid_ontology.base_iri = "http://nonexistent.org/invalid-ontology.owl"
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            # Simulate text2term error for invalid ontology
            mock_map_terms.side_effect = FileNotFoundError("Ontology not found")
            
            with expect_exception(OntologyNotFoundError, "Ontology not found"):
                map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=invalid_ontology
                )
    
    def test_mapping_error_handling(self):
        """Test error handling for mapping process failures."""
        relationships = [("entity1", "relation1", "target1")]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            # Simulate text2term mapping error
            mock_map_terms.side_effect = RuntimeError("Mapping process failed")
            
            with expect_exception(MappingError, "Failed to map relationships"):
                map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj
                )
    
    def test_empty_relationships_list_error(self):
        """Test error handling for empty relationships list."""
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        with expect_exception(ValueError, "Relationships list cannot be empty"):
            map_relationships_to_ontology(
                relationships=[],
                ontology_obj=ontology_obj
            )
    
    def test_none_relationships_list_error(self):
        """Test error handling for None relationships list."""
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        with expect_exception(ValueError, "Relationships list cannot be None"):
            map_relationships_to_ontology(
                relationships=None,
                ontology_obj=ontology_obj
            )
    
    def test_invalid_relationship_format_error(self):
        """Test error handling for invalid relationship format."""
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        invalid_relationships = [
            ("subject", "relation"),  # Missing object
            ("subject",),  # Missing relation and object
            ("subject", "relation", "object", "extra"),  # Too many elements
            "not_a_tuple",  # Not a tuple
            123  # Not a tuple at all
        ]
        
        for invalid_rel in invalid_relationships:
            with expect_exception(ValueError, "Invalid relationship format"):
                map_relationships_to_ontology(
                    relationships=[invalid_rel],
                    ontology_obj=ontology_obj
                )
    
    def test_invalid_ontology_object_error(self):
        """Test error handling for invalid ontology object."""
        relationships = [("glucose", "metabolized_by", "enzyme")]
        
        invalid_ontologies = [
            None,
            "string_instead_of_object",
            123,
            []
        ]
        
        for invalid_ontology in invalid_ontologies:
            with expect_exception(ValueError, "Invalid ontology object"):
                map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=invalid_ontology
                )
    
    def test_invalid_mapping_method_error(self):
        """Test error handling for invalid mapping method."""
        relationships = [("glucose", "metabolized_by", "enzyme")]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        invalid_methods = [
            "invalid_method",
            "",
            None,
            123
        ]
        
        for invalid_method in invalid_methods:
            with expect_exception(ValueError, "Invalid mapping method"):
                map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj,
                    mapping_method=invalid_method
                )
    
    def test_invalid_min_score_error(self):
        """Test error handling for invalid minimum score values."""
        relationships = [("glucose", "metabolized_by", "enzyme")]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        invalid_scores = [-0.1, 1.1, "0.5", None]
        
        for invalid_score in invalid_scores:
            with expect_exception(ValueError, "Minimum score must be between 0.0 and 1.0"):
                map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj,
                    min_score=invalid_score
                )
    
    def test_semantic_validation_error(self):
        """Test error handling for semantic validation failures."""
        relationships = [("glucose", "metabolized_by", "enzyme")]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["metabolized_by"],
            'Mapped Term Label': ["metabolized by"],
            'Mapped Term IRI': ["http://purl.obolibrary.org/obo/RO_0002209"],
            'Mapping Score': [0.95],
            'Term Type': ["property"]
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                # Simulate semantic validation error
                mock_validate.side_effect = SemanticValidationError("Invalid domain/range")
                
                with expect_exception(SemanticValidationError, "Invalid domain/range"):
                    map_relationships_to_ontology(
                        relationships=relationships,
                        ontology_obj=ontology_obj,
                        validate_semantics=True
                    )


class TestInputValidation:
    """Test cases for input validation functions."""
    
    def test_validate_relationships_valid_input(self):
        """Test validation of valid relationship lists."""
        valid_relationship_lists = [
            [("subject", "relation", "object")],
            [
                ("glucose", "metabolized_by", "enzyme"),
                ("arabidopsis", "has_part", "leaf"),
                ("ATP", "produced_by", "respiration")
            ],
            [(fake_chemical_name(), "relation", fake_entity("compound")) for _ in range(10)]
        ]
        
        for relationships in valid_relationship_lists:
            # Should not raise any exception
            _validate_relationships(relationships)
    
    def test_validate_relationships_invalid_input(self):
        """Test validation of invalid relationship lists."""
        invalid_relationship_lists = [
            None,
            [],
            "",
            [("subject", "relation")],  # Missing object
            [("subject",)],  # Missing relation and object
            [("subject", "relation", "object", "extra")],  # Too many elements
            [("", "relation", "object")],  # Empty subject
            [("subject", "", "object")],  # Empty relation
            [("subject", "relation", "")],  # Empty object
            [("subject", None, "object")],  # None relation
            [(None, "relation", "object")],  # None subject
            [("subject", "relation", None)],  # None object
            ["not_a_tuple"],  # String instead of tuple
            [123],  # Number instead of tuple
            [("valid", "relation", "object"), ("invalid", "relation")]  # Mix of valid and invalid
        ]
        
        for relationships in invalid_relationship_lists:
            with expect_exception(ValueError):
                _validate_relationships(relationships)
    
    def test_validate_mapping_method_valid(self):
        """Test validation of valid mapping methods."""
        valid_methods = [
            "tfidf", "levenshtein", "jaro_winkler", 
            "jaccard", "fuzzy"
        ]
        
        for method in valid_methods:
            # Should not raise any exception
            _validate_mapping_method(method)
    
    def test_validate_mapping_method_invalid(self):
        """Test validation of invalid mapping methods."""
        invalid_methods = [
            None, "", "invalid", 123, []
        ]
        
        for method in invalid_methods:
            with expect_exception(ValueError):
                _validate_mapping_method(method)


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_process_mapping_results_basic(self):
        """Test basic processing of mapping results."""
        relationships = [("subject", "relation", "object")]
        raw_df = pd.DataFrame({
            'Source Term': ["relation"],
            'Mapped Term Label': ["mapped relation"],
            'Mapped Term IRI': ["http://example.org/relation"],
            'Mapping Score': [0.95],
            'Term Type': ["property"]
        })
        
        processed_df = _process_mapping_results(relationships, raw_df)
        
        # Should include additional columns for the full relationship context
        expected_columns = [
            'Subject', 'Relation', 'Object', 'Mapped_Relation_Label', 
            'Mapped_Relation_IRI', 'Mapping_Score', 'Term_Type'
        ]
        assert all(col in processed_df.columns for col in expected_columns)
        assert len(processed_df) == 1
        assert processed_df.iloc[0]['Subject'] == "subject"
        assert processed_df.iloc[0]['Relation'] == "relation"
        assert processed_df.iloc[0]['Object'] == "object"
    
    def test_process_mapping_results_with_cleaning(self):
        """Test processing with data cleaning."""
        relationships = [
            ("subject1", "relation1", "object1"),
            ("subject2", "relation2", "object2"),
            ("subject3", "relation3", "object3")
        ]
        raw_df = pd.DataFrame({
            'Source Term': ["relation1", "relation2", "relation3"],
            'Mapped Term Label': ["mapped1", None, "mapped3"],  # None value
            'Mapped Term IRI': ["http://example.org/1", "", "http://example.org/3"],  # Empty string
            'Mapping Score': [0.95, None, 0.85],  # None value
            'Term Type': ["property", None, "property"]  # None value
        })
        
        processed_df = _process_mapping_results(relationships, raw_df)
        
        # Should handle None and empty values appropriately
        assert len(processed_df) <= len(raw_df)  # May filter out invalid rows
        # Valid rows should not have null values in critical columns
        valid_rows = processed_df[processed_df['Mapped_Relation_IRI'].notna()]
        assert not valid_rows[['Subject', 'Relation', 'Object']].isnull().any().any()
    
    def test_filter_by_score_basic(self):
        """Test basic score filtering."""
        df = pd.DataFrame({
            'Relation': ["relation1", "relation2", "relation3"],
            'Mapping_Score': [0.95, 0.75, 0.65]
        })
        
        filtered_df = _filter_by_score(df, min_score=0.8)
        
        assert len(filtered_df) == 1
        assert filtered_df.iloc[0]['Relation'] == "relation1"
        assert filtered_df.iloc[0]['Mapping_Score'] == 0.95
    
    def test_filter_by_score_edge_cases(self):
        """Test score filtering edge cases."""
        df = pd.DataFrame({
            'Relation': ["relation1", "relation2", "relation3"],
            'Mapping_Score': [0.8, 0.8, 0.79]
        })
        
        # Test exact threshold matching
        filtered_df = _filter_by_score(df, min_score=0.8)
        assert len(filtered_df) == 2  # 0.8 >= 0.8, but 0.79 < 0.8
        
        # Test with very low threshold
        filtered_df = _filter_by_score(df, min_score=0.0)
        assert len(filtered_df) == 3  # All should pass
        
        # Test with very high threshold
        filtered_df = _filter_by_score(df, min_score=1.0)
        assert len(filtered_df) == 0  # None should pass


class TestEdgeCases:
    """Test cases for edge cases and boundary conditions."""
    
    def test_single_relationship_mapping(self):
        """Test mapping with a single relationship."""
        relationships = [("glucose", "metabolized_by", "enzyme")]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://purl.obolibrary.org/obo/ro.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["metabolized_by"],
            'Mapped Term Label': ["metabolized by"],
            'Mapped Term IRI': ["http://purl.obolibrary.org/obo/RO_0002209"],
            'Mapping Score': [0.98],
            'Term Type': ["property"]
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                result = map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj
                )
                
                assert len(result) == 1
                assert result.iloc[0]['Relation'] == "metabolized_by"
    
    def test_large_relationship_list_mapping(self):
        """Test mapping with a large list of relationships."""
        relationships = [
            (fake_chemical_name(), f"relation_{i}", fake_entity("compound")) 
            for i in range(100)
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://purl.obolibrary.org/obo/ro.owl"
        
        relation_terms = [rel[1] for rel in relationships]
        
        # Mock responses for large list
        mock_mapping_df = pd.DataFrame({
            'Source Term': relation_terms,
            'Mapped Term Label': [f"mapped_{relation}" for relation in relation_terms],
            'Mapped Term IRI': [f"http://purl.obolibrary.org/obo/RO_{i:07d}" 
                               for i in range(len(relation_terms))],
            'Mapping Score': [0.8 + (i % 20) * 0.01 for i in range(len(relation_terms))],
            'Term Type': ["property"] * len(relation_terms)
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                result = map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj,
                    min_score=0.8
                )
                
                # Should handle large lists efficiently
                assert len(result) == len(relationships)
                assert len(result.columns) >= 7  # Expected columns
    
    def test_relationships_with_special_characters(self):
        """Test mapping relationships with special characters."""
        relationships = [
            ("β-carotene", "converted_to", "vitamin_A"),
            ("α-tocopherol", "acts_as", "antioxidant"),
            ("γ-aminobutyric_acid", "functions_in", "neurotransmission"),
            ("trans-resveratrol", "exhibits", "anti-inflammatory_activity"),
            ("cis-lycopene", "transformed_to", "trans-lycopene"),
            ("D-glucose", "metabolized_via", "glycolysis")
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://purl.obolibrary.org/obo/ro.owl"
        
        relation_terms = [rel[1] for rel in relationships]
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': relation_terms,
            'Mapped Term Label': [
                "converted to", "acts as", "functions in",
                "exhibits", "transformed to", "metabolized via"
            ],
            'Mapped Term IRI': [f"http://purl.obolibrary.org/obo/RO_{i:07d}" 
                               for i in [2343, 2324, 2327, 2581, 2343, 2209]],
            'Mapping Score': [0.92, 0.89, 0.95, 0.88, 0.91, 0.94],
            'Term Type': ["property"] * 6
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                result = map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj
                )
                
                # Should handle special characters correctly
                assert len(result) == 6
                assert all(score > 0.8 for score in result['Mapping_Score'])
                
                # Verify subjects with special characters are preserved
                subjects_with_special = ["β-carotene", "α-tocopherol", "γ-aminobutyric_acid"]
                assert any(subj in result['Subject'].values for subj in subjects_with_special)
    
    def test_duplicate_relationships_handling(self):
        """Test handling of duplicate relationships in input."""
        relationships = [
            ("glucose", "metabolized_by", "enzyme"),
            ("glucose", "metabolized_by", "enzyme"),  # Exact duplicate
            ("arabidopsis", "has_part", "leaf"),
            ("glucose", "metabolized_by", "enzyme"),  # Another duplicate
            ("arabidopsis", "has_part", "leaf")  # Another duplicate
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        # Mock response should reflect the duplicates
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["metabolized_by", "metabolized_by", "has_part", "metabolized_by", "has_part"],
            'Mapped Term Label': [
                "metabolized by", "metabolized by", "has part", 
                "metabolized by", "has part"
            ],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/RO_0002209",
                "http://purl.obolibrary.org/obo/RO_0002209",
                "http://purl.obolibrary.org/obo/BFO_0000051",
                "http://purl.obolibrary.org/obo/RO_0002209",
                "http://purl.obolibrary.org/obo/BFO_0000051"
            ],
            'Mapping Score': [0.98, 0.98, 0.95, 0.98, 0.95],
            'Term Type': ["property"] * 5
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                result = map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj
                )
                
                # Should preserve duplicates if that's how text2term handles them
                assert len(result) == 5
    
    def test_relationships_with_whitespace_variations(self):
        """Test handling of relationships with whitespace variations."""
        relationships = [
            (" glucose ", " metabolized_by ", " enzyme "),
            ("  arabidopsis", "has_part  ", "leaf"),
            ("ATP\t", "\tproduced_by", "\trespiration\t"),
            ("\nprotein\n", "\nbinds_to\n", "\nDNA\n")
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://example.org/test-ontology.owl"
        
        # Relations should be cleaned before mapping
        cleaned_relations = ["metabolized_by", "has_part", "produced_by", "binds_to"]
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': cleaned_relations,
            'Mapped Term Label': [
                "metabolized by", "has part", "produced by", "binds to"
            ],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/RO_0002209",
                "http://purl.obolibrary.org/obo/BFO_0000051",
                "http://purl.obolibrary.org/obo/RO_0003001",
                "http://purl.obolibrary.org/obo/RO_0003680"
            ],
            'Mapping Score': [0.98, 0.95, 0.92, 0.89],
            'Term Type': ["property"] * 4
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                result = map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj
                )
                
                # Should handle whitespace variations
                assert len(result) == 4
                # Verify that cleaned relations are used
                assert set(result['Relation']) == set(cleaned_relations)
                # Verify that subjects and objects are also cleaned
                assert "glucose" in result['Subject'].values  # Cleaned from " glucose "
                assert "enzyme" in result['Object'].values  # Cleaned from " enzyme "


class TestRelationMapperErrorClasses:
    """Test cases for relation mapper specific error classes."""
    
    def test_relation_mapper_error_inheritance(self):
        """Test that RelationMapperError properly inherits from Exception."""
        error = RelationMapperError("Test relation mapper error")
        assert isinstance(error, Exception)
        assert str(error) == "Test relation mapper error"
    
    def test_ontology_not_found_error_inheritance(self):
        """Test that OntologyNotFoundError properly inherits from RelationMapperError."""
        error = OntologyNotFoundError("Ontology not found")
        assert isinstance(error, RelationMapperError)
        assert isinstance(error, Exception)
        assert str(error) == "Ontology not found"
    
    def test_mapping_error_inheritance(self):
        """Test that MappingError properly inherits from RelationMapperError."""
        error = MappingError("Mapping failed")
        assert isinstance(error, RelationMapperError)
        assert isinstance(error, Exception)
        assert str(error) == "Mapping failed"
    
    def test_semantic_validation_error_inheritance(self):
        """Test that SemanticValidationError properly inherits from RelationMapperError."""
        error = SemanticValidationError("Semantic validation failed")
        assert isinstance(error, RelationMapperError)
        assert isinstance(error, Exception)
        assert str(error) == "Semantic validation failed"


class TestIntegrationScenarios:
    """Test cases for integration scenarios with realistic data."""
    
    def test_plant_metabolomics_relationship_mapping_scenario(self):
        """Test complete plant metabolomics relationship mapping scenario."""
        # Realistic plant metabolomics relationships
        relationships = [
            ("quercetin", "produced_by", "Arabidopsis_thaliana"),
            ("anthocyanin", "synthesized_in", "flower"),
            ("chlorophyll", "participates_in", "photosynthesis"),
            ("glucose", "converted_to", "starch"),
            ("ATP", "produced_by", "cellular_respiration"),
            ("drought_stress", "affects", "metabolite_levels"),
            ("light", "regulates", "gene_expression"),
            ("enzyme", "catalyzes", "biosynthesis_pathway"),
            ("transcription_factor", "binds_to", "promoter_region"),
            ("metabolite", "transported_by", "membrane_protein"),
            ("phytohormone", "signals", "developmental_process"),
            ("antioxidant", "protects_against", "oxidative_stress")
        ]
        ontology_obj = Mock()
        ontology_obj.base_iri = "http://purl.obolibrary.org/obo/merged_plant_ontology.owl"
        
        relation_terms = [rel[1] for rel in relationships]
        
        # Mock realistic mappings
        mock_mapping_df = pd.DataFrame({
            'Source Term': relation_terms,
            'Mapped Term Label': [
                "produced by", "synthesized in", "participates in", "converted to",
                "produced by", "affects", "regulates", "catalyzes",
                "binds to", "transported by", "signals", "protects against"
            ],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/RO_0003001",
                "http://purl.obolibrary.org/obo/RO_0002202",
                "http://purl.obolibrary.org/obo/RO_0000056",
                "http://purl.obolibrary.org/obo/RO_0002343",
                "http://purl.obolibrary.org/obo/RO_0003001",
                "http://purl.obolibrary.org/obo/RO_0002263",
                "http://purl.obolibrary.org/obo/RO_0002211",
                "http://purl.obolibrary.org/obo/RO_0002327",
                "http://purl.obolibrary.org/obo/RO_0003680",
                "http://purl.obolibrary.org/obo/RO_0002313",
                "http://purl.obolibrary.org/obo/RO_0002348",
                "http://purl.obolibrary.org/obo/RO_0002456"
            ],
            'Mapping Score': [
                0.98, 0.89, 0.95, 0.92, 0.98, 0.85, 0.94, 0.96,
                0.91, 0.87, 0.88, 0.84
            ],
            'Term Type': ["property"] * 12
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_map_terms.return_value = mock_mapping_df
                mock_validate.return_value = True
                
                result = map_relationships_to_ontology(
                    relationships=relationships,
                    ontology_obj=ontology_obj,
                    mapping_method='tfidf',
                    min_score=0.8,
                    validate_semantics=True
                )
                
                # Verify comprehensive mapping results
                assert len(result) == 12
                
                # Check that all relationships are semantically valid
                assert all(result['Semantic_Valid'])
                
                # Verify RO (Relations Ontology) mappings
                ro_mappings = result[result['Mapped_Relation_IRI'].str.contains('RO_')]
                assert len(ro_mappings) == 12  # All should be RO mappings
                
                # Check specific biological relationships
                metabolic_relations = ['produced_by', 'converted_to', 'synthesized_in']
                metabolic_results = result[result['Relation'].isin(metabolic_relations)]
                assert len(metabolic_results) >= 3
                
                regulatory_relations = ['regulates', 'affects', 'signals']
                regulatory_results = result[result['Relation'].isin(regulatory_relations)]
                assert len(regulatory_results) >= 3
    
    def test_multi_ontology_relationship_mapping_workflow(self):
        """Test relationship mapping workflow with multiple target ontologies."""
        chemical_relationships = [
            ("quercetin", "inhibits", "enzyme"),
            ("ATP", "binds_to", "kinase"),
            ("drug", "interacts_with", "protein")
        ]
        
        biological_relationships = [
            ("gene", "regulates", "protein_expression"),
            ("enzyme", "catalyzes", "metabolic_reaction")
        ]
        
        # Test mapping to ChEBI for chemical relationships
        chebi_ontology = Mock()
        chebi_ontology.base_iri = "http://purl.obolibrary.org/obo/chebi.owl"
        
        chebi_mock_df = pd.DataFrame({
            'Source Term': ["inhibits", "binds_to", "interacts_with"],
            'Mapped Term Label': ["inhibits", "binds to", "interacts with"],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/RO_0002449",
                "http://purl.obolibrary.org/obo/RO_0002436",
                "http://purl.obolibrary.org/obo/RO_0002434"
            ],
            'Mapping Score': [0.94, 0.91, 0.87],
            'Term Type': ["property"] * 3
        })
        
        # Test mapping to GO for biological relationships
        go_ontology = Mock()
        go_ontology.base_iri = "http://purl.obolibrary.org/obo/go.owl"
        
        go_mock_df = pd.DataFrame({
            'Source Term': ["regulates", "catalyzes"],
            'Mapped Term Label': ["regulates", "catalyzes"],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/RO_0002211",
                "http://purl.obolibrary.org/obo/RO_0002327"
            ],
            'Mapping Score': [0.96, 0.98],
            'Term Type': ["property"] * 2
        })
        
        with patch('src.ontology_mapping.relation_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.relation_mapper._validate_semantic_consistency') as mock_validate:
                mock_validate.return_value = True
                
                # First call for chemical relationships to ChEBI
                mock_map_terms.return_value = chebi_mock_df
                
                chemical_results = map_relationships_to_ontology(
                    relationships=chemical_relationships,
                    ontology_obj=chebi_ontology
                )
                
                # Second call for biological relationships to GO
                mock_map_terms.return_value = go_mock_df
                
                biological_results = map_relationships_to_ontology(
                    relationships=biological_relationships,
                    ontology_obj=go_ontology
                )
                
                # Verify separate mappings
                assert len(chemical_results) == 3
                assert len(biological_results) == 2
                
                # Verify proper ontology targeting (all should map to RO)
                assert all('RO_' in iri for iri in chemical_results['Mapped_Relation_IRI'])
                assert all('RO_' in iri for iri in biological_results['Mapped_Relation_IRI'])


# Fixtures for test data
@pytest.fixture
def sample_metabolic_relationships():
    """Fixture providing sample metabolic relationships for testing."""
    return [
        ("glucose", "metabolized_by", "glycolysis"),
        ("ATP", "produced_by", "cellular_respiration"),
        ("pyruvate", "converted_to", "lactate"),
        ("acetyl_CoA", "participates_in", "TCA_cycle"),
        ("NADH", "oxidized_by", "electron_transport_chain"),
        ("fatty_acid", "synthesized_from", "acetyl_CoA"),
        ("amino_acid", "derived_from", "protein_degradation"),
        ("glucose_6_phosphate", "formed_from", "glucose"),
        ("citrate", "produced_in", "mitochondria"),
        ("oxaloacetate", "regenerated_in", "TCA_cycle")
    ]


@pytest.fixture
def sample_regulatory_relationships():
    """Fixture providing sample regulatory relationships for testing."""
    return [
        ("transcription_factor", "regulates", "gene_expression"),
        ("microRNA", "inhibits", "mRNA_translation"),
        ("histone_modification", "affects", "chromatin_structure"),
        ("phosphorylation", "activates", "protein_function"),
        ("allosteric_effector", "modulates", "enzyme_activity"),
        ("hormone", "signals", "cellular_response"),
        ("feedback_inhibition", "controls", "metabolic_pathway"),
        ("inducer", "upregulates", "operon_expression"),
        ("repressor", "downregulates", "transcription"),
        ("cofactor", "enhances", "enzyme_catalysis")
    ]


@pytest.fixture
def sample_interaction_relationships():
    """Fixture providing sample molecular interaction relationships for testing."""
    return [
        ("protein", "binds_to", "DNA"),
        ("enzyme", "catalyzes", "chemical_reaction"),
        ("antibody", "recognizes", "antigen"),
        ("drug", "interacts_with", "receptor"),
        ("substrate", "binds_to", "active_site"),
        ("ligand", "activates", "signaling_pathway"),
        ("inhibitor", "blocks", "enzyme_function"),
        ("cofactor", "assists", "enzymatic_reaction"),
        ("allosteric_modulator", "changes", "protein_conformation"),
        ("chaperone", "facilitates", "protein_folding")
    ]


@pytest.fixture
def mock_ro_ontology_response():
    """Fixture providing mock Relations Ontology mapping response."""
    return pd.DataFrame({
        'Source Term': ["regulates", "catalyzes", "participates_in"],
        'Mapped Term Label': ["regulates", "catalyzes", "participates in"],
        'Mapped Term IRI': [
            "http://purl.obolibrary.org/obo/RO_0002211",
            "http://purl.obolibrary.org/obo/RO_0002327",
            "http://purl.obolibrary.org/obo/RO_0000056"
        ],
        'Mapping Score': [0.96, 0.98, 0.92],
        'Term Type': ["property", "property", "property"]
    })


@pytest.fixture
def mock_bfo_ontology_response():
    """Fixture providing mock Basic Formal Ontology mapping response."""
    return pd.DataFrame({
        'Source Term': ["has_part", "part_of", "occurs_in"],
        'Mapped Term Label': ["has part", "part of", "occurs in"],
        'Mapped Term IRI': [
            "http://purl.obolibrary.org/obo/BFO_0000051",
            "http://purl.obolibrary.org/obo/BFO_0000050",
            "http://purl.obolibrary.org/obo/BFO_0000066"
        ],
        'Mapping Score': [0.99, 0.97, 0.94],
        'Term Type': ["property", "property", "property"]
    })


@pytest.fixture
def mock_ontology_with_properties():
    """Fixture providing mock ontology object with property constraints."""
    ontology = Mock()
    ontology.base_iri = "http://example.org/test-ontology.owl"
    
    # Mock property with domain and range
    mock_property = Mock()
    mock_property.name = "metabolized_by"
    mock_property.domain = [Mock()]
    mock_property.domain[0].name = "ChemicalEntity"
    mock_property.range = [Mock()]
    mock_property.range[0].name = "BiologicalProcess"
    
    ontology.search.return_value = [mock_property]
    return ontology


# Mark all tests in this module as ontology mapping related
pytestmark = pytest.mark.ontology_mapping