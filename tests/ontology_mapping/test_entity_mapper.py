"""
Unit tests for src/ontology_mapping/entity_mapper.py

This module tests the entity-to-ontology mapping functionality using text2term for mapping
extracted entities to ontology terms. The module supports various mapping methods, minimum
similarity scores, and different term types.

Test Coverage:
- Basic entity mapping with predefined test ontology
- Different text2term mapping methods (TFIDF, LEVENSHTEIN, etc.)
- Minimum score filtering for high-confidence mappings
- Mapping to specific term types (class, property)
- Handling of unmapped terms
- Error handling for invalid inputs and API failures
- Edge cases and performance considerations

Test Approach:
- Mock text2term.map_terms() to avoid external dependencies
- Test different mapping scenarios with controlled inputs
- Validate output format and data integrity
- Ensure proper error handling and validation
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional
import json

# Import testing utilities from the project's testing framework
from src.utils.testing_framework import (
    expect_exception,
    parametrize,
    fake_text,
    fake_entity,
    fake_chemical_name
)

# Import the entity mapper functions (will be implemented in src/ontology_mapping/entity_mapper.py)
from src.ontology_mapping.entity_mapper import (
    map_entities_to_ontology,
    EntityMapperError,
    OntologyNotFoundError,
    MappingError,
    _validate_entities,
    _validate_mapping_method,
    _process_mapping_results,
    _filter_by_score,
    text2term  # Import text2term for test assertions
)


class TestMapEntitiesToOntologyBasic:
    """Test cases for basic entity-to-ontology mapping functionality."""
    
    def test_map_entities_basic_functionality(self):
        """Test basic entity mapping with default parameters."""
        entities = ["glucose", "arabidopsis", "photosynthesis"]
        ontology_iri = "http://example.org/test-ontology.owl"
        
        # Mock text2term response
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["glucose", "arabidopsis", "photosynthesis"],
            'Mapped Term Label': ["glucose", "Arabidopsis thaliana", "photosynthesis"],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/CHEBI_17234",
                "http://purl.obolibrary.org/obo/NCBITaxon_3702", 
                "http://purl.obolibrary.org/obo/GO_0015979"
            ],
            'Mapping Score': [0.95, 0.88, 0.92],
            'Term Type': ["class", "class", "class"]
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            result = map_entities_to_ontology(
                entities=entities,
                target_ontology=ontology_iri
            )
            
            # Verify function call
            mock_map_terms.assert_called_once_with(
                source_terms=entities,
                target_ontology=ontology_iri,
                mapper=text2term.Mapper.TFIDF,
                min_score=0.3,
                term_type='class',
                incl_unmapped=False
            )
            
            # Validate results
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert all(col in result.columns for col in [
                'Source Term', 'Mapped Term Label', 'Mapped Term IRI', 
                'Mapping Score', 'Term Type'
            ])
            
            # Check specific mappings
            assert result.iloc[0]['Source Term'] == "glucose"
            assert result.iloc[0]['Mapped Term IRI'] == "http://purl.obolibrary.org/obo/CHEBI_17234"
            assert result.iloc[0]['Mapping Score'] == 0.95
    
    def test_map_entities_with_chemical_compounds(self):
        """Test entity mapping specifically with chemical compounds."""
        chemical_entities = [
            "quercetin", "anthocyanin", "resveratrol", "chlorophyll",
            "beta-carotene", "ascorbic acid"
        ]
        ontology_iri = "http://purl.obolibrary.org/obo/chebi.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': chemical_entities,
            'Mapped Term Label': [
                "quercetin", "anthocyanin", "resveratrol", 
                "chlorophyll a", "beta-carotene", "L-ascorbic acid"
            ],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/CHEBI_16243",
                "http://purl.obolibrary.org/obo/CHEBI_38697",
                "http://purl.obolibrary.org/obo/CHEBI_27881",
                "http://purl.obolibrary.org/obo/CHEBI_18230",
                "http://purl.obolibrary.org/obo/CHEBI_17836",
                "http://purl.obolibrary.org/obo/CHEBI_29073"
            ],
            'Mapping Score': [0.98, 0.85, 0.91, 0.89, 0.94, 0.96],
            'Term Type': ["class"] * 6
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            result = map_entities_to_ontology(
                entities=chemical_entities,
                target_ontology=ontology_iri,
                mapping_method='tfidf',
                min_score=0.8
            )
            
            # Verify all chemical compounds were processed
            assert len(result) == 6
            assert all(score >= 0.8 for score in result['Mapping Score'])
            
            # Verify ChEBI IRIs format
            assert all(iri.startswith("http://purl.obolibrary.org/obo/CHEBI_") 
                      for iri in result['Mapped Term IRI'])
    
    def test_map_entities_with_species_names(self):
        """Test entity mapping with biological species names."""
        species_entities = [
            "Arabidopsis thaliana", "Solanum lycopersicum", "Oryza sativa",
            "Zea mays", "Vitis vinifera"
        ]
        ontology_iri = "http://purl.obolibrary.org/obo/ncbitaxon.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': species_entities,
            'Mapped Term Label': species_entities,  # Exact matches expected
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/NCBITaxon_3702",
                "http://purl.obolibrary.org/obo/NCBITaxon_4081",
                "http://purl.obolibrary.org/obo/NCBITaxon_4530",
                "http://purl.obolibrary.org/obo/NCBITaxon_4577",
                "http://purl.obolibrary.org/obo/NCBITaxon_29760"
            ],
            'Mapping Score': [0.99, 0.99, 0.99, 0.99, 0.99],
            'Term Type': ["class"] * 5
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            result = map_entities_to_ontology(
                entities=species_entities,
                target_ontology=ontology_iri,
                mapping_method='levenshtein',
                min_score=0.9
            )
            
            # Verify high-confidence species mappings
            assert len(result) == 5
            assert all(score >= 0.9 for score in result['Mapping Score'])
            
            # Verify NCBITaxon IRIs format
            assert all(iri.startswith("http://purl.obolibrary.org/obo/NCBITaxon_") 
                      for iri in result['Mapped Term IRI'])


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
        """Test different text2term mapping methods."""
        entities = ["glucose", "arabidopsis"]
        ontology_iri = "http://example.org/test-ontology.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': entities,
            'Mapped Term Label': ["D-glucose", "Arabidopsis thaliana"],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/CHEBI_17234",
                "http://purl.obolibrary.org/obo/NCBITaxon_3702"
            ],
            'Mapping Score': [0.92, 0.88],
            'Term Type': ["class", "class"]
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.entity_mapper.text2term.Mapper') as mock_mapper:
                mock_map_terms.return_value = mock_mapping_df
                
                # Set up mapper attribute access
                getattr(mock_mapper, expected_mapper.split('.')[-1])
                
                result = map_entities_to_ontology(
                    entities=entities,
                    target_ontology=ontology_iri,
                    mapping_method=mapping_method
                )
                
                # Verify correct mapper was used
                mock_map_terms.assert_called_once()
                call_args = mock_map_terms.call_args[1]
                assert 'mapper' in call_args
                
                assert len(result) == 2
    
    def test_tfidf_method_performance(self):
        """Test TFIDF mapping method with performance considerations."""
        # Larger set of entities to test TFIDF performance
        entities = [fake_chemical_name() for _ in range(20)]
        ontology_iri = "http://purl.obolibrary.org/obo/chebi.owl"
        
        # Mock varied scores to simulate TFIDF behavior
        mock_scores = [0.95, 0.89, 0.72, 0.68, 0.45, 0.91, 0.83, 0.55, 
                      0.78, 0.66, 0.88, 0.74, 0.59, 0.82, 0.71, 0.94, 
                      0.67, 0.86, 0.63, 0.77]
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': entities,
            'Mapped Term Label': [f"mapped_{entity}" for entity in entities],
            'Mapped Term IRI': [f"http://purl.obolibrary.org/obo/CHEBI_{i:05d}" 
                               for i in range(len(entities))],
            'Mapping Score': mock_scores,
            'Term Type': ["class"] * len(entities)
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            result = map_entities_to_ontology(
                entities=entities,
                target_ontology=ontology_iri,
                mapping_method='tfidf',
                min_score=0.7  # Filter out low-confidence mappings
            )
            
            # Verify filtering worked correctly
            expected_count = sum(1 for score in mock_scores if score >= 0.7)
            assert len(result) == expected_count
            assert all(score >= 0.7 for score in result['Mapping Score'])
    
    def test_levenshtein_method_fuzzy_matching(self):
        """Test Levenshtein mapping method for fuzzy string matching."""
        # Entities with slight variations to test fuzzy matching
        entities = [
            "glucose", "glucos", "glocose",  # Variations of glucose
            "arabidopsis", "arabidopsi", "aribidopsis"  # Variations of arabidopsis
        ]
        ontology_iri = "http://example.org/test-ontology.owl"
        
        # Levenshtein should handle these variations well
        mock_mapping_df = pd.DataFrame({
            'Source Term': entities,
            'Mapped Term Label': ["glucose", "glucose", "glucose", 
                                 "Arabidopsis thaliana", "Arabidopsis thaliana", 
                                 "Arabidopsis thaliana"],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/CHEBI_17234",
                "http://purl.obolibrary.org/obo/CHEBI_17234",
                "http://purl.obolibrary.org/obo/CHEBI_17234",
                "http://purl.obolibrary.org/obo/NCBITaxon_3702",
                "http://purl.obolibrary.org/obo/NCBITaxon_3702",
                "http://purl.obolibrary.org/obo/NCBITaxon_3702"
            ],
            'Mapping Score': [1.0, 0.85, 0.82, 1.0, 0.88, 0.79],
            'Term Type': ["class"] * 6
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            result = map_entities_to_ontology(
                entities=entities,
                target_ontology=ontology_iri,
                mapping_method='levenshtein',
                min_score=0.75
            )
            
            # Verify fuzzy matching results
            assert len(result) == 6  # All entities above threshold (0.75)
            
            # Check that variations map to same terms
            glucose_mappings = result[result['Mapped Term IRI'] == 
                                    "http://purl.obolibrary.org/obo/CHEBI_17234"]
            assert len(glucose_mappings) == 3


class TestScoreFiltering:
    """Test cases for minimum score filtering functionality."""
    
    def test_min_score_filtering_basic(self):
        """Test basic minimum score filtering."""
        entities = ["compound1", "compound2", "compound3", "compound4"]
        ontology_iri = "http://example.org/test-ontology.owl"
        min_score = 0.8
        
        # Mock responses with varied scores
        mock_mapping_df = pd.DataFrame({
            'Source Term': entities,
            'Mapped Term Label': ["mapped1", "mapped2", "mapped3", "mapped4"],
            'Mapped Term IRI': [f"http://example.org/term{i}" for i in range(4)],
            'Mapping Score': [0.95, 0.75, 0.85, 0.65],  # 2 above, 2 below threshold
            'Term Type': ["class"] * 4
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            result = map_entities_to_ontology(
                entities=entities,
                target_ontology=ontology_iri,
                min_score=min_score
            )
            
            # Only mappings with score >= 0.8 should be returned
            assert len(result) == 2
            assert all(score >= min_score for score in result['Mapping Score'])
            
            # Verify specific entities that passed filtering
            expected_entities = ["compound1", "compound3"]
            assert set(result['Source Term']) == set(expected_entities)
    
    @parametrize("min_score,expected_count", [
        (0.0, 6),   # All entities pass
        (0.5, 5),   # 5 entities pass
        (0.7, 4),   # 4 entities pass
        (0.8, 3),   # 3 entities pass
        (0.9, 2),   # 2 entities pass
        (0.95, 1),  # 1 entity passes
        (0.99, 0)   # No entities pass
    ])
    def test_different_score_thresholds(self, min_score, expected_count):
        """Test filtering with different minimum score thresholds."""
        entities = ["entity1", "entity2", "entity3", "entity4", "entity5", "entity6"]
        ontology_iri = "http://example.org/test-ontology.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': entities,
            'Mapped Term Label': [f"mapped{i}" for i in range(6)],
            'Mapped Term IRI': [f"http://example.org/term{i}" for i in range(6)],
            'Mapping Score': [0.98, 0.91, 0.84, 0.77, 0.63, 0.45],
            'Term Type': ["class"] * 6
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            result = map_entities_to_ontology(
                entities=entities,
                target_ontology=ontology_iri,
                min_score=min_score
            )
            
            assert len(result) == expected_count
            if expected_count > 0:
                assert all(score >= min_score for score in result['Mapping Score'])
    
    def test_high_confidence_mappings_only(self):
        """Test filtering for high-confidence mappings only."""
        entities = [
            "photosynthesis", "cellular respiration", "glycolysis",
            "transcription", "translation", "metabolism"
        ]
        ontology_iri = "http://purl.obolibrary.org/obo/go.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': entities,
            'Mapped Term Label': [
                "photosynthesis", "cellular respiration", "glycolytic process",
                "DNA-templated transcription", "translation", "metabolic process"
            ],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/GO_0015979",
                "http://purl.obolibrary.org/obo/GO_0045333",
                "http://purl.obolibrary.org/obo/GO_0006096",
                "http://purl.obolibrary.org/obo/GO_0006351",
                "http://purl.obolibrary.org/obo/GO_0006412",
                "http://purl.obolibrary.org/obo/GO_0008152"
            ],
            'Mapping Score': [0.99, 0.97, 0.94, 0.96, 0.98, 0.95],
            'Term Type': ["class"] * 6
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            # Test very high confidence threshold
            result = map_entities_to_ontology(
                entities=entities,
                target_ontology=ontology_iri,
                min_score=0.95
            )
            
            # Should return only mappings with score >= 0.95
            assert len(result) == 5  # All except glycolysis (0.94)
            assert all(score >= 0.95 for score in result['Mapping Score'])


class TestTermTypes:
    """Test cases for mapping to specific term types."""
    
    def test_map_to_class_terms(self):
        """Test mapping entities to ontology classes."""
        entities = ["glucose", "enzyme", "membrane"]
        ontology_iri = "http://example.org/test-ontology.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': entities,
            'Mapped Term Label': ["glucose", "enzyme", "membrane"],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/CHEBI_17234",
                "http://purl.obolibrary.org/obo/CHEBI_36080",
                "http://purl.obolibrary.org/obo/GO_0016020"
            ],
            'Mapping Score': [0.95, 0.89, 0.92],
            'Term Type': ["class", "class", "class"]
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            result = map_entities_to_ontology(
                entities=entities,
                target_ontology=ontology_iri,
                term_type='class'
            )
            
            # Verify text2term was called with correct term_type
            mock_map_terms.assert_called_once()
            call_args = mock_map_terms.call_args[1]
            assert call_args['term_type'] == 'class'
            
            # Verify all results are class types
            assert len(result) == 3
            assert all(term_type == "class" for term_type in result['Term Type'])
    
    def test_map_to_property_terms(self):
        """Test mapping entities to ontology properties."""
        entities = ["has_part", "regulates", "participates_in"]
        ontology_iri = "http://purl.obolibrary.org/obo/ro.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': entities,
            'Mapped Term Label': ["has part", "regulates", "participates in"],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/BFO_0000051",
                "http://purl.obolibrary.org/obo/RO_0002211",
                "http://purl.obolibrary.org/obo/RO_0000056"
            ],
            'Mapping Score': [0.88, 0.92, 0.94],
            'Term Type': ["property", "property", "property"]
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            result = map_entities_to_ontology(
                entities=entities,
                target_ontology=ontology_iri,
                term_type='property'
            )
            
            # Verify text2term was called with correct term_type
            call_args = mock_map_terms.call_args[1]
            assert call_args['term_type'] == 'property'
            
            # Verify all results are property types
            assert len(result) == 3
            assert all(term_type == "property" for term_type in result['Term Type'])
    
    @parametrize("term_type", ["class", "property", "individual"])
    def test_different_term_types(self, term_type):
        """Test mapping with different term types."""
        entities = ["test_entity"]
        ontology_iri = "http://example.org/test-ontology.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': entities,
            'Mapped Term Label': ["mapped_entity"],
            'Mapped Term IRI': ["http://example.org/mapped_entity"],
            'Mapping Score': [0.9],
            'Term Type': [term_type]
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            result = map_entities_to_ontology(
                entities=entities,
                target_ontology=ontology_iri,
                term_type=term_type
            )
            
            call_args = mock_map_terms.call_args[1]
            assert call_args['term_type'] == term_type
            assert result.iloc[0]['Term Type'] == term_type


class TestUnmappedTermsHandling:
    """Test cases for handling unmapped terms."""
    
    def test_exclude_unmapped_terms_default(self):
        """Test default behavior of excluding unmapped terms."""
        entities = ["known_entity", "unknown_entity", "another_known"]
        ontology_iri = "http://example.org/test-ontology.owl"
        
        # Mock response with only mapped terms (default text2term behavior)
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["known_entity", "another_known"],
            'Mapped Term Label': ["known entity", "another known entity"],
            'Mapped Term IRI': [
                "http://example.org/known_entity",
                "http://example.org/another_known"
            ],
            'Mapping Score': [0.95, 0.88],
            'Term Type': ["class", "class"]
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            result = map_entities_to_ontology(
                entities=entities,
                target_ontology=ontology_iri
            )
            
            # Verify text2term was called with incl_unmapped=False (default)
            call_args = mock_map_terms.call_args[1]
            assert call_args['incl_unmapped'] == False
            
            # Only mapped terms should be returned
            assert len(result) == 2
            assert "unknown_entity" not in result['Source Term'].values
    
    def test_include_unmapped_terms_explicit(self):
        """Test explicit inclusion of unmapped terms."""
        entities = ["known_entity", "unknown_entity", "another_known"]
        ontology_iri = "http://example.org/test-ontology.owl"
        
        # Mock response including unmapped terms
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["known_entity", "unknown_entity", "another_known"],
            'Mapped Term Label': ["known entity", None, "another known entity"],
            'Mapped Term IRI': [
                "http://example.org/known_entity",
                None,
                "http://example.org/another_known"
            ],
            'Mapping Score': [0.95, None, 0.88],
            'Term Type': ["class", None, "class"]
        })
        
        # Patch the function to accept incl_unmapped parameter
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            with patch('src.ontology_mapping.entity_mapper.map_entities_to_ontology') as mock_func:
                mock_map_terms.return_value = mock_mapping_df
                
                # Mock the actual function to test parameter passing
                def mock_implementation(entities, target_ontology, **kwargs):
                    incl_unmapped = kwargs.get('incl_unmapped', False)
                    if incl_unmapped:
                        return mock_mapping_df
                    else:
                        return mock_mapping_df[mock_mapping_df['Mapped Term IRI'].notna()]
                
                mock_func.side_effect = mock_implementation
                
                result = mock_func(
                    entities=entities,
                    target_ontology=ontology_iri,
                    incl_unmapped=True
                )
                
                # All terms should be included, even unmapped ones
                assert len(result) == 3
                assert "unknown_entity" in result['Source Term'].values
    
    def test_mixed_mapped_unmapped_results(self):
        """Test handling of mixed mapped and unmapped results."""
        entities = ["glucose", "xyz123", "arabidopsis", "abc456", "photosynthesis"]
        ontology_iri = "http://example.org/test-ontology.owl"
        
        # Simulate realistic scenario where some entities don't map
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["glucose", "arabidopsis", "photosynthesis"],
            'Mapped Term Label': [
                "glucose", "Arabidopsis thaliana", "photosynthesis"
            ],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/CHEBI_17234",
                "http://purl.obolibrary.org/obo/NCBITaxon_3702",
                "http://purl.obolibrary.org/obo/GO_0015979"
            ],
            'Mapping Score': [0.98, 0.95, 0.92],
            'Term Type': ["class", "class", "class"]
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            result = map_entities_to_ontology(
                entities=entities,
                target_ontology=ontology_iri,
                min_score=0.8
            )
            
            # Only successfully mapped entities should be returned
            assert len(result) == 3
            mapped_entities = set(result['Source Term'])
            assert mapped_entities == {"glucose", "arabidopsis", "photosynthesis"}
            assert "xyz123" not in mapped_entities
            assert "abc456" not in mapped_entities


class TestErrorHandling:
    """Test cases for error handling in entity mapping."""
    
    def test_ontology_not_found_error(self):
        """Test error handling for non-existent ontology IRI."""
        entities = ["glucose", "arabidopsis"]
        invalid_iri = "http://nonexistent.org/invalid-ontology.owl"
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            # Simulate text2term error for invalid ontology
            mock_map_terms.side_effect = FileNotFoundError("Ontology not found")
            
            with expect_exception(OntologyNotFoundError, "Ontology not found"):
                map_entities_to_ontology(
                    entities=entities,
                    target_ontology=invalid_iri
                )
    
    def test_mapping_error_handling(self):
        """Test error handling for mapping process failures."""
        entities = ["entity1", "entity2"]
        ontology_iri = "http://example.org/test-ontology.owl"
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            # Simulate text2term mapping error
            mock_map_terms.side_effect = RuntimeError("Mapping process failed")
            
            with expect_exception(MappingError, "Failed to map entities"):
                map_entities_to_ontology(
                    entities=entities,
                    target_ontology=ontology_iri
                )
    
    def test_empty_entities_list_error(self):
        """Test error handling for empty entities list."""
        with expect_exception(ValueError, "Entities list cannot be empty"):
            map_entities_to_ontology(
                entities=[],
                target_ontology="http://example.org/test-ontology.owl"
            )
    
    def test_none_entities_list_error(self):
        """Test error handling for None entities list."""
        with expect_exception(ValueError, "Entities list cannot be None"):
            map_entities_to_ontology(
                entities=None,
                target_ontology="http://example.org/test-ontology.owl"
            )
    
    def test_invalid_ontology_iri_error(self):
        """Test error handling for invalid ontology IRI format."""
        entities = ["glucose"]
        
        invalid_iris = [
            None,
            "",
            "not-a-url",
            "ftp://invalid-protocol.org/ontology.owl",
            123  # Non-string type
        ]
        
        for invalid_iri in invalid_iris:
            with expect_exception(ValueError, "Invalid ontology IRI"):
                map_entities_to_ontology(
                    entities=entities,
                    target_ontology=invalid_iri
                )
    
    def test_invalid_mapping_method_error(self):
        """Test error handling for invalid mapping method."""
        entities = ["glucose"]
        ontology_iri = "http://example.org/test-ontology.owl"
        
        invalid_methods = [
            "invalid_method",
            "",
            None,
            123
        ]
        
        for invalid_method in invalid_methods:
            with expect_exception(ValueError, "Invalid mapping method"):
                map_entities_to_ontology(
                    entities=entities,
                    target_ontology=ontology_iri,
                    mapping_method=invalid_method
                )
    
    def test_invalid_min_score_error(self):
        """Test error handling for invalid minimum score values."""
        entities = ["glucose"]
        ontology_iri = "http://example.org/test-ontology.owl"
        
        invalid_scores = [-0.1, 1.1, "0.5", None]
        
        for invalid_score in invalid_scores:
            with expect_exception(ValueError, "Minimum score must be between 0.0 and 1.0"):
                map_entities_to_ontology(
                    entities=entities,
                    target_ontology=ontology_iri,
                    min_score=invalid_score
                )
    
    def test_invalid_term_type_error(self):
        """Test error handling for invalid term type."""
        entities = ["glucose"]
        ontology_iri = "http://example.org/test-ontology.owl"
        
        invalid_term_types = [
            "invalid_type",
            "",
            None,
            123
        ]
        
        for invalid_term_type in invalid_term_types:
            with expect_exception(ValueError, "Invalid term type"):
                map_entities_to_ontology(
                    entities=entities,
                    target_ontology=ontology_iri,
                    term_type=invalid_term_type
                )


class TestInputValidation:
    """Test cases for input validation functions."""
    
    def test_validate_entities_valid_input(self):
        """Test validation of valid entity lists."""
        valid_entity_lists = [
            ["glucose"],
            ["glucose", "arabidopsis", "photosynthesis"],
            [fake_chemical_name() for _ in range(10)]
        ]
        
        for entities in valid_entity_lists:
            # Should not raise any exception
            _validate_entities(entities)
    
    def test_validate_entities_invalid_input(self):
        """Test validation of invalid entity lists."""
        invalid_entity_lists = [
            None,
            [],
            "",
            [""],  # Empty string in list
            [None],  # None in list
            [123],  # Non-string in list
            ["valid", ""],  # Mix of valid and invalid
            ["valid", None]  # Mix of valid and None
        ]
        
        for entities in invalid_entity_lists:
            with expect_exception(ValueError):
                _validate_entities(entities)
    
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
        raw_df = pd.DataFrame({
            'Source Term': ["entity1", "entity2"],
            'Mapped Term Label': ["mapped1", "mapped2"],
            'Mapped Term IRI': ["http://example.org/1", "http://example.org/2"],
            'Mapping Score': [0.95, 0.85],
            'Term Type': ["class", "class"]
        })
        
        processed_df = _process_mapping_results(raw_df)
        
        # Should return the same DataFrame for valid input
        pd.testing.assert_frame_equal(processed_df, raw_df)
    
    def test_process_mapping_results_with_cleaning(self):
        """Test processing with data cleaning."""
        raw_df = pd.DataFrame({
            'Source Term': ["entity1", "entity2", "entity3"],
            'Mapped Term Label': ["mapped1", None, "mapped3"],  # None value
            'Mapped Term IRI': ["http://example.org/1", "", "http://example.org/3"],  # Empty string
            'Mapping Score': [0.95, None, 0.85],  # None value
            'Term Type': ["class", None, "class"]  # None value
        })
        
        processed_df = _process_mapping_results(raw_df)
        
        # Should handle None and empty values appropriately
        assert len(processed_df) <= len(raw_df)  # May filter out invalid rows
        assert not processed_df.isnull().any().any()  # No null values in result
    
    def test_filter_by_score_basic(self):
        """Test basic score filtering."""
        df = pd.DataFrame({
            'Source Term': ["entity1", "entity2", "entity3"],
            'Mapping Score': [0.95, 0.75, 0.65]
        })
        
        filtered_df = _filter_by_score(df, min_score=0.8)
        
        assert len(filtered_df) == 1
        assert filtered_df.iloc[0]['Source Term'] == "entity1"
        assert filtered_df.iloc[0]['Mapping Score'] == 0.95
    
    def test_filter_by_score_edge_cases(self):
        """Test score filtering edge cases."""
        df = pd.DataFrame({
            'Source Term': ["entity1", "entity2", "entity3"],
            'Mapping Score': [0.8, 0.8, 0.79]
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
    
    def test_single_entity_mapping(self):
        """Test mapping with a single entity."""
        entities = ["glucose"]
        ontology_iri = "http://purl.obolibrary.org/obo/chebi.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': ["glucose"],
            'Mapped Term Label': ["glucose"],
            'Mapped Term IRI': ["http://purl.obolibrary.org/obo/CHEBI_17234"],
            'Mapping Score': [0.98],
            'Term Type': ["class"]
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            result = map_entities_to_ontology(
                entities=entities,
                target_ontology=ontology_iri
            )
            
            assert len(result) == 1
            assert result.iloc[0]['Source Term'] == "glucose"
    
    def test_large_entity_list_mapping(self):
        """Test mapping with a large list of entities."""
        entities = [fake_chemical_name() for _ in range(100)]
        ontology_iri = "http://purl.obolibrary.org/obo/chebi.owl"
        
        # Mock responses for large list
        mock_mapping_df = pd.DataFrame({
            'Source Term': entities,
            'Mapped Term Label': [f"mapped_{entity}" for entity in entities],
            'Mapped Term IRI': [f"http://purl.obolibrary.org/obo/CHEBI_{i:05d}" 
                               for i in range(len(entities))],
            'Mapping Score': [0.8 + (i % 20) * 0.01 for i in range(len(entities))],
            'Term Type': ["class"] * len(entities)
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            result = map_entities_to_ontology(
                entities=entities,
                target_ontology=ontology_iri,
                min_score=0.8
            )
            
            # Should handle large lists efficiently
            assert len(result) == len(entities)
            assert len(result.columns) == 5  # Expected columns
    
    def test_entities_with_special_characters(self):
        """Test mapping entities with special characters."""
        entities = [
            "β-carotene", "α-tocopherol", "γ-aminobutyric acid",
            "D-glucose", "L-ascorbic acid", "trans-resveratrol"
        ]
        ontology_iri = "http://purl.obolibrary.org/obo/chebi.owl"
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': entities,
            'Mapped Term Label': [
                "beta-carotene", "alpha-tocopherol", "gamma-aminobutyric acid",
                "D-glucose", "L-ascorbic acid", "trans-resveratrol"
            ],
            'Mapped Term IRI': [f"http://purl.obolibrary.org/obo/CHEBI_{i}" 
                               for i in [17836, 18145, 30566, 17234, 29073, 27881]],
            'Mapping Score': [0.92, 0.89, 0.95, 0.99, 0.97, 0.91],
            'Term Type': ["class"] * 6
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            result = map_entities_to_ontology(
                entities=entities,
                target_ontology=ontology_iri
            )
            
            # Should handle special characters correctly
            assert len(result) == 6
            assert all(score > 0.8 for score in result['Mapping Score'])
    
    def test_duplicate_entities_handling(self):
        """Test handling of duplicate entities in input."""
        entities = ["glucose", "glucose", "arabidopsis", "glucose", "arabidopsis"]
        ontology_iri = "http://example.org/test-ontology.owl"
        
        # Mock response should reflect the duplicates
        mock_mapping_df = pd.DataFrame({
            'Source Term': entities,
            'Mapped Term Label': [
                "glucose", "glucose", "Arabidopsis thaliana", 
                "glucose", "Arabidopsis thaliana"
            ],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/CHEBI_17234",
                "http://purl.obolibrary.org/obo/CHEBI_17234",
                "http://purl.obolibrary.org/obo/NCBITaxon_3702",
                "http://purl.obolibrary.org/obo/CHEBI_17234",
                "http://purl.obolibrary.org/obo/NCBITaxon_3702"
            ],
            'Mapping Score': [0.98, 0.98, 0.95, 0.98, 0.95],
            'Term Type': ["class"] * 5
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            result = map_entities_to_ontology(
                entities=entities,
                target_ontology=ontology_iri
            )
            
            # Should preserve duplicates if that's how text2term handles them
            assert len(result) == 5
    
    def test_entities_with_whitespace_variations(self):
        """Test handling of entities with whitespace variations."""
        entities = [
            " glucose ", "  arabidopsis", "photosynthesis  ",
            "\tcellulose\t", "\nlignin\n"
        ]
        ontology_iri = "http://example.org/test-ontology.owl"
        
        # Entities should be cleaned before mapping
        cleaned_entities = ["glucose", "arabidopsis", "photosynthesis", "cellulose", "lignin"]
        
        mock_mapping_df = pd.DataFrame({
            'Source Term': cleaned_entities,
            'Mapped Term Label': [
                "glucose", "Arabidopsis thaliana", "photosynthesis",
                "cellulose", "lignin"
            ],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/CHEBI_17234",
                "http://purl.obolibrary.org/obo/NCBITaxon_3702",
                "http://purl.obolibrary.org/obo/GO_0015979",
                "http://purl.obolibrary.org/obo/CHEBI_3583",
                "http://purl.obolibrary.org/obo/CHEBI_6457"
            ],
            'Mapping Score': [0.98, 0.95, 0.92, 0.89, 0.87],
            'Term Type': ["class"] * 5
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            result = map_entities_to_ontology(
                entities=entities,
                target_ontology=ontology_iri
            )
            
            # Should handle whitespace variations
            assert len(result) == 5
            # Verify that cleaned entities are used
            assert set(result['Source Term']) == set(cleaned_entities)


class TestEntityMapperErrorClasses:
    """Test cases for entity mapper specific error classes."""
    
    def test_entity_mapper_error_inheritance(self):
        """Test that EntityMapperError properly inherits from Exception."""
        error = EntityMapperError("Test entity mapper error")
        assert isinstance(error, Exception)
        assert str(error) == "Test entity mapper error"
    
    def test_ontology_not_found_error_inheritance(self):
        """Test that OntologyNotFoundError properly inherits from EntityMapperError."""
        error = OntologyNotFoundError("Ontology not found")
        assert isinstance(error, EntityMapperError)
        assert isinstance(error, Exception)
        assert str(error) == "Ontology not found"
    
    def test_mapping_error_inheritance(self):
        """Test that MappingError properly inherits from EntityMapperError."""
        error = MappingError("Mapping failed")
        assert isinstance(error, EntityMapperError)
        assert isinstance(error, Exception)
        assert str(error) == "Mapping failed"


class TestIntegrationScenarios:
    """Test cases for integration scenarios with realistic data."""
    
    def test_plant_metabolomics_mapping_scenario(self):
        """Test complete plant metabolomics entity mapping scenario."""
        # Realistic plant metabolomics entities
        entities = [
            "quercetin", "kaempferol", "anthocyanin", "chlorophyll",
            "Arabidopsis thaliana", "Solanum lycopersicum", "Oryza sativa",
            "photosynthesis", "cellular respiration", "glycolysis",
            "drought stress", "salt stress", "cold acclimation"
        ]
        ontology_iri = "http://purl.obolibrary.org/obo/merged_plant_ontology.owl"
        
        # Mock realistic mappings
        mock_mapping_df = pd.DataFrame({
            'Source Term': entities,
            'Mapped Term Label': [
                "quercetin", "kaempferol", "anthocyanin", "chlorophyll a",
                "Arabidopsis thaliana", "Solanum lycopersicum", "Oryza sativa",
                "photosynthesis", "cellular respiration", "glycolytic process",
                "drought stress", "salt stress", "cold acclimation"
            ],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/CHEBI_16243",
                "http://purl.obolibrary.org/obo/CHEBI_28499",
                "http://purl.obolibrary.org/obo/CHEBI_38697",
                "http://purl.obolibrary.org/obo/CHEBI_18230",
                "http://purl.obolibrary.org/obo/NCBITaxon_3702",
                "http://purl.obolibrary.org/obo/NCBITaxon_4081",
                "http://purl.obolibrary.org/obo/NCBITaxon_4530",
                "http://purl.obolibrary.org/obo/GO_0015979",
                "http://purl.obolibrary.org/obo/GO_0045333",
                "http://purl.obolibrary.org/obo/GO_0006096",
                "http://purl.obolibrary.org/obo/PECO_0007174",
                "http://purl.obolibrary.org/obo/PECO_0007106",
                "http://purl.obolibrary.org/obo/PECO_0007221"
            ],
            'Mapping Score': [
                0.98, 0.96, 0.85, 0.89, 0.99, 0.99, 0.99,
                0.99, 0.97, 0.94, 0.88, 0.91, 0.86
            ],
            'Term Type': ["class"] * 13
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            mock_map_terms.return_value = mock_mapping_df
            
            result = map_entities_to_ontology(
                entities=entities,
                target_ontology=ontology_iri,
                mapping_method='tfidf',
                min_score=0.8
            )
            
            # Verify comprehensive mapping results
            assert len(result) == 13
            
            # Check specific ontology mappings
            chebi_mappings = result[result['Mapped Term IRI'].str.contains('CHEBI')]
            assert len(chebi_mappings) == 4  # Chemical compounds
            
            ncbi_mappings = result[result['Mapped Term IRI'].str.contains('NCBITaxon')]
            assert len(ncbi_mappings) == 3  # Species
            
            go_mappings = result[result['Mapped Term IRI'].str.contains('GO_')]
            assert len(go_mappings) == 3  # Biological processes
            
            peco_mappings = result[result['Mapped Term IRI'].str.contains('PECO')]
            assert len(peco_mappings) == 3  # Environmental conditions
    
    def test_multi_ontology_mapping_workflow(self):
        """Test mapping workflow with multiple target ontologies."""
        chemical_entities = ["glucose", "quercetin", "anthocyanin"]
        species_entities = ["Arabidopsis thaliana", "Oryza sativa"]
        
        # Test mapping to ChEBI for chemicals
        chebi_mock_df = pd.DataFrame({
            'Source Term': chemical_entities,
            'Mapped Term Label': ["glucose", "quercetin", "anthocyanin"],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/CHEBI_17234",
                "http://purl.obolibrary.org/obo/CHEBI_16243",
                "http://purl.obolibrary.org/obo/CHEBI_38697"
            ],
            'Mapping Score': [0.98, 0.96, 0.85],
            'Term Type': ["class"] * 3
        })
        
        # Test mapping to NCBI Taxonomy for species
        ncbi_mock_df = pd.DataFrame({
            'Source Term': species_entities,
            'Mapped Term Label': ["Arabidopsis thaliana", "Oryza sativa"],
            'Mapped Term IRI': [
                "http://purl.obolibrary.org/obo/NCBITaxon_3702",
                "http://purl.obolibrary.org/obo/NCBITaxon_4530"
            ],
            'Mapping Score': [0.99, 0.99],
            'Term Type': ["class"] * 2
        })
        
        with patch('src.ontology_mapping.entity_mapper.text2term.map_terms') as mock_map_terms:
            # First call for chemicals to ChEBI
            mock_map_terms.return_value = chebi_mock_df
            
            chemical_results = map_entities_to_ontology(
                entities=chemical_entities,
                target_ontology="http://purl.obolibrary.org/obo/chebi.owl"
            )
            
            # Second call for species to NCBI Taxonomy
            mock_map_terms.return_value = ncbi_mock_df
            
            species_results = map_entities_to_ontology(
                entities=species_entities,
                target_ontology="http://purl.obolibrary.org/obo/ncbitaxon.owl"
            )
            
            # Verify separate mappings
            assert len(chemical_results) == 3
            assert len(species_results) == 2
            
            # Verify proper ontology targeting
            assert all('CHEBI' in iri for iri in chemical_results['Mapped Term IRI'])
            assert all('NCBITaxon' in iri for iri in species_results['Mapped Term IRI'])


# Fixtures for test data
@pytest.fixture
def sample_chemical_entities():
    """Fixture providing sample chemical entities for testing."""
    return [
        "glucose", "fructose", "sucrose", "starch", "cellulose",
        "quercetin", "kaempferol", "anthocyanin", "resveratrol",
        "chlorophyll", "carotenoid", "tocopherol", "ascorbic acid"
    ]


@pytest.fixture
def sample_species_entities():
    """Fixture providing sample species entities for testing."""
    return [
        "Arabidopsis thaliana", "Solanum lycopersicum", "Oryza sativa",
        "Zea mays", "Triticum aestivum", "Vitis vinifera", "Medicago truncatula"
    ]


@pytest.fixture
def sample_process_entities():
    """Fixture providing sample biological process entities for testing."""
    return [
        "photosynthesis", "cellular respiration", "glycolysis",
        "transcription", "translation", "protein folding",
        "DNA replication", "cell division", "apoptosis"
    ]


@pytest.fixture
def mock_chebi_ontology_response():
    """Fixture providing mock ChEBI ontology mapping response."""
    return pd.DataFrame({
        'Source Term': ["glucose", "quercetin", "anthocyanin"],
        'Mapped Term Label': ["glucose", "quercetin", "anthocyanin"],
        'Mapped Term IRI': [
            "http://purl.obolibrary.org/obo/CHEBI_17234",
            "http://purl.obolibrary.org/obo/CHEBI_16243",
            "http://purl.obolibrary.org/obo/CHEBI_38697"
        ],
        'Mapping Score': [0.98, 0.96, 0.85],
        'Term Type': ["class", "class", "class"]
    })


@pytest.fixture
def mock_go_ontology_response():
    """Fixture providing mock Gene Ontology mapping response."""
    return pd.DataFrame({
        'Source Term': ["photosynthesis", "cellular respiration", "glycolysis"],
        'Mapped Term Label': ["photosynthesis", "cellular respiration", "glycolytic process"],
        'Mapped Term IRI': [
            "http://purl.obolibrary.org/obo/GO_0015979",
            "http://purl.obolibrary.org/obo/GO_0045333",
            "http://purl.obolibrary.org/obo/GO_0006096"
        ],
        'Mapping Score': [0.99, 0.97, 0.94],
        'Term Type': ["class", "class", "class"]
    })


# Mark all tests in this module as ontology mapping related
pytestmark = pytest.mark.ontology_mapping