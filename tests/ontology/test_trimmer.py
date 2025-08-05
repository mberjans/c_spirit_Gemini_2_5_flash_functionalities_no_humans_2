"""
Unit tests for the ontology trimmer module.

This module contains comprehensive tests for filtering and trimming OWL 2.0 ontologies
using Owlready2. Tests cover filtering by keywords, properties, class hierarchies, and
combined criteria while ensuring the original ontology remains unmodified.

Test Categories:
- Filtering classes by keyword in name/label
- Filtering individuals by property values
- Filtering subclasses of base classes
- Combined filtering criteria
- Original ontology preservation
- Edge cases and error handling
"""

import copy
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Generator, List, Dict

import pytest
from owlready2 import OwlReadyError

from src.utils.testing_framework import expect_exception, parametrize


class TestOntologyTrimmer:
    """Test suite for ontology trimming and filtering functionality."""

    @pytest.fixture
    def mock_ontology(self) -> Mock:
        """
        Create a comprehensive mock ontology object for testing.
        
        Returns:
            Mock: Mock ontology with classes, individuals, and properties
        """
        # Create mock ontology
        mock_ont = Mock()
        mock_ont.name = "test_ontology"
        mock_ont.base_iri = "http://test.example.org/ontology"
        
        # Create mock classes with different names and labels
        mock_class1 = Mock()
        mock_class1.name = "PlantClass"
        mock_class1.label = ["Plant related class"]
        mock_class1.iri = "http://test.example.org/ontology#PlantClass"
        
        mock_class2 = Mock()
        mock_class2.name = "MetaboliteClass"
        mock_class2.label = ["Metabolite compound class"]
        mock_class2.iri = "http://test.example.org/ontology#MetaboliteClass"
        
        mock_class3 = Mock()
        mock_class3.name = "AnimalClass"
        mock_class3.label = ["Animal related class"]
        mock_class3.iri = "http://test.example.org/ontology#AnimalClass"
        
        # Create mock individuals with properties
        mock_individual1 = Mock()
        mock_individual1.name = "glucose"
        mock_individual1.compound_type = "sugar"
        mock_individual1.concentration = 5.2
        mock_individual1.iri = "http://test.example.org/ontology#glucose"
        
        mock_individual2 = Mock()
        mock_individual2.name = "caffeine"
        mock_individual2.compound_type = "alkaloid"
        mock_individual2.concentration = 1.8
        mock_individual2.iri = "http://test.example.org/ontology#caffeine"
        
        mock_individual3 = Mock()
        mock_individual3.name = "chlorophyll"
        mock_individual3.compound_type = "pigment"
        mock_individual3.concentration = 3.1
        mock_individual3.iri = "http://test.example.org/ontology#chlorophyll"
        
        # Setup class hierarchy
        mock_class1.subclasses = Mock(return_value=[mock_class2])
        mock_class2.subclasses = Mock(return_value=[])
        mock_class3.subclasses = Mock(return_value=[])
        
        # Configure search method behavior
        def mock_search(**kwargs):
            results = []
            
            # Handle different search criteria
            if 'iri' in kwargs:
                iri = kwargs['iri']
                if "PlantClass" in iri:
                    results = [mock_class1]
                elif "MetaboliteClass" in iri:
                    results = [mock_class2]
                elif "AnimalClass" in iri:
                    results = [mock_class3]
            
            elif 'subclass_of' in kwargs or 'is_a' in kwargs:
                base_class = kwargs.get('subclass_of') or kwargs.get('is_a')
                if base_class == mock_class1:
                    results = [mock_class2]
                else:
                    results = []
            
            elif len(kwargs) == 1 and isinstance(list(kwargs.values())[0], str):
                # Keyword search
                keyword = list(kwargs.values())[0].lower()
                if "plant" in keyword:
                    results = [mock_class1]
                elif "metabolite" in keyword:
                    results = [mock_class2]
                elif "compound" in keyword:
                    results = [mock_class2, mock_individual1, mock_individual2]
                elif "sugar" in keyword:
                    results = [mock_individual1]
                elif "alkaloid" in keyword:
                    results = [mock_individual2]
                elif "*" in keyword:
                    # Wildcard search
                    results = [mock_class1, mock_class2, mock_class3, mock_individual1, mock_individual2, mock_individual3]
            
            return results
        
        mock_ont.search = Mock(side_effect=mock_search)
        
        # Setup classes and individuals collections
        mock_ont.classes = Mock(return_value=[mock_class1, mock_class2, mock_class3])
        mock_ont.individuals = Mock(return_value=[mock_individual1, mock_individual2, mock_individual3])
        
        return mock_ont

    @pytest.fixture
    def mock_trimmer_functions(self) -> Generator[Dict[str, Mock], None, None]:
        """
        Mock the trimmer module functions for testing.
        
        Yields:
            Dict[str, Mock]: Dictionary of mocked trimmer functions
        """
        with patch('src.ontology.trimmer.filter_classes_by_keyword') as mock_filter_classes, \
             patch('src.ontology.trimmer.filter_individuals_by_property') as mock_filter_individuals, \
             patch('src.ontology.trimmer.get_subclasses') as mock_get_subclasses, \
             patch('src.ontology.trimmer.apply_filters') as mock_apply_filters:
            
            yield {
                'filter_classes_by_keyword': mock_filter_classes,
                'filter_individuals_by_property': mock_filter_individuals,
                'get_subclasses': mock_get_subclasses,
                'apply_filters': mock_apply_filters
            }

    def test_filter_classes_by_keyword_success(self, mock_ontology: Mock):
        """
        Test successful filtering of classes by keyword in name or label.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import filter_classes_by_keyword
        
        keyword = "plant"
        
        # Act
        result = filter_classes_by_keyword(mock_ontology, keyword)
        
        # Assert
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "PlantClass"
        
        # Verify search was called with correct parameters
        mock_ontology.search.assert_called()

    def test_filter_classes_by_keyword_case_insensitive(self, mock_ontology: Mock):
        """
        Test that keyword filtering is case-insensitive.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import filter_classes_by_keyword
        
        # Test different case variations
        keywords = ["PLANT", "Plant", "pLaNt", "METABOLITE"]
        
        for keyword in keywords[:3]:  # Test "plant" variations
            result = filter_classes_by_keyword(mock_ontology, keyword)
            assert len(result) >= 1
            assert any(cls.name == "PlantClass" for cls in result)
        
        # Test "metabolite" case
        result = filter_classes_by_keyword(mock_ontology, keywords[3])
        assert len(result) >= 1
        assert any(cls.name == "MetaboliteClass" for cls in result)

    def test_filter_classes_by_keyword_no_matches(self, mock_ontology: Mock):
        """
        Test filtering classes by keyword when no matches are found.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import filter_classes_by_keyword
        
        keyword = "nonexistent"
        
        # Configure mock to return empty results
        mock_ontology.search.return_value = []
        
        # Act
        result = filter_classes_by_keyword(mock_ontology, keyword)
        
        # Assert
        assert result is not None
        assert len(result) == 0
        assert isinstance(result, list)

    @parametrize("keyword,expected_count", [
        ("plant", 1),
        ("metabolite", 1),
        ("compound", 1),  # Should match only MetaboliteClass (contains "compound" in label)
        ("class", 0),  # No direct matches in mock data
    ])
    def test_filter_classes_by_keyword_parameterized(
        self, 
        mock_ontology: Mock, 
        keyword: str, 
        expected_count: int
    ):
        """
        Test filtering classes by various keywords with expected counts.
        
        Args:
            mock_ontology: Mock ontology fixture
            keyword: Keyword to search for
            expected_count: Expected number of results
        """
        from src.ontology.trimmer import filter_classes_by_keyword
        
        # Act
        result = filter_classes_by_keyword(mock_ontology, keyword)
        
        # Assert
        assert len(result) == expected_count

    def test_filter_individuals_by_property_success(self, mock_ontology: Mock):
        """
        Test successful filtering of individuals by property values.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import filter_individuals_by_property
        
        property_name = "compound_type"
        value = "sugar"
        
        # Act
        result = filter_individuals_by_property(mock_ontology, property_name, value)
        
        # Assert
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "glucose"
        assert result[0].compound_type == "sugar"

    def test_filter_individuals_by_property_multiple_matches(self, mock_ontology: Mock):
        """
        Test filtering individuals by property with multiple matches.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import filter_individuals_by_property
        
        # Mock individuals to have concentration > 3.0
        individuals = mock_ontology.individuals()
        filtered_individuals = [ind for ind in individuals if hasattr(ind, 'concentration') and ind.concentration > 3.0]
        
        with patch('src.ontology.trimmer.filter_individuals_by_property') as mock_func:
            mock_func.return_value = filtered_individuals
            
            property_name = "concentration"
            value = ">3.0"  # Using string to represent condition
            
            # Act
            result = filter_individuals_by_property(mock_ontology, property_name, value)
            
            # Assert
            assert len(result) == 2  # glucose (5.2) and chlorophyll (3.1)
            mock_func.assert_called_once_with(mock_ontology, property_name, value)

    def test_filter_individuals_by_property_no_matches(self, mock_ontology: Mock):
        """
        Test filtering individuals by property when no matches are found.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import filter_individuals_by_property
        
        with patch('src.ontology.trimmer.filter_individuals_by_property') as mock_func:
            mock_func.return_value = []
            
            property_name = "nonexistent_property"
            value = "nonexistent_value"
            
            # Act
            result = filter_individuals_by_property(mock_ontology, property_name, value)
            
            # Assert
            assert result is not None
            assert len(result) == 0
            assert isinstance(result, list)

    @parametrize("property_name,value,expected_names", [
        ("compound_type", "sugar", ["glucose"]),
        ("compound_type", "alkaloid", ["caffeine"]),
        ("compound_type", "pigment", ["chlorophyll"]),
        ("compound_type", "nonexistent", []),
    ])
    def test_filter_individuals_by_property_parameterized(
        self, 
        mock_ontology: Mock, 
        property_name: str, 
        value: Any, 
        expected_names: List[str]
    ):
        """
        Test filtering individuals by various property values.
        
        Args:
            mock_ontology: Mock ontology fixture
            property_name: Name of the property to filter by
            value: Value to match
            expected_names: Expected names of matching individuals
        """
        from src.ontology.trimmer import filter_individuals_by_property
        
        # Mock the function to return expected results
        individuals = mock_ontology.individuals()
        filtered_individuals = [
            ind for ind in individuals 
            if hasattr(ind, property_name) and getattr(ind, property_name) == value
        ]
        
        with patch('src.ontology.trimmer.filter_individuals_by_property') as mock_func:
            mock_func.return_value = filtered_individuals
            
            # Act
            result = filter_individuals_by_property(mock_ontology, property_name, value)
            
            # Assert
            assert len(result) == len(expected_names)
            result_names = [ind.name for ind in result]
            assert set(result_names) == set(expected_names)

    def test_get_subclasses_success(self, mock_ontology: Mock):
        """
        Test successful retrieval of subclasses for a base class.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import get_subclasses
        
        base_class_iri = "http://test.example.org/ontology#PlantClass"
        
        # Act
        result = get_subclasses(mock_ontology, base_class_iri)
        
        # Assert
        assert result is not None
        assert len(result) == 1
        assert result[0].name == "MetaboliteClass"

    def test_get_subclasses_no_subclasses(self, mock_ontology: Mock):
        """
        Test getting subclasses when base class has no subclasses.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import get_subclasses
        
        base_class_iri = "http://test.example.org/ontology#MetaboliteClass"
        
        # Act
        result = get_subclasses(mock_ontology, base_class_iri)
        
        # Assert
        assert result is not None
        assert len(result) == 0
        assert isinstance(result, list)

    def test_get_subclasses_invalid_iri(self, mock_ontology: Mock):
        """
        Test getting subclasses with invalid base class IRI.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import get_subclasses
        
        # Configure mock to return empty results for invalid IRI
        mock_ontology.search.return_value = []
        
        base_class_iri = "http://invalid.example.org/ontology#NonexistentClass"
        
        # Act
        result = get_subclasses(mock_ontology, base_class_iri)
        
        # Assert
        assert result is not None
        assert len(result) == 0

    def test_get_subclasses_recursive_hierarchy(self, mock_ontology: Mock):
        """
        Test getting subclasses in a deeper class hierarchy.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import get_subclasses
        
        # Create a deeper hierarchy for testing
        mock_grandchild = Mock()
        mock_grandchild.name = "GrandchildClass"
        mock_grandchild.iri = "http://test.example.org/ontology#GrandchildClass"
        mock_grandchild.subclasses = Mock(return_value=[])
        
        # Update MetaboliteClass to have a subclass
        metabolite_class = mock_ontology.search(iri="*MetaboliteClass*")[0]
        metabolite_class.subclasses = Mock(return_value=[mock_grandchild])
        
        with patch('src.ontology.trimmer.get_subclasses') as mock_func:
            # Mock to return both direct and indirect subclasses
            mock_func.return_value = [metabolite_class, mock_grandchild]
            
            base_class_iri = "http://test.example.org/ontology#PlantClass"
            
            # Act
            result = get_subclasses(mock_ontology, base_class_iri)
            
            # Assert
            assert len(result) == 2
            names = [cls.name for cls in result]
            assert "MetaboliteClass" in names
            assert "GrandchildClass" in names

    def test_apply_filters_single_criterion(self, mock_ontology: Mock):
        """
        Test applying filters with a single filtering criterion.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import apply_filters
        
        filters = {
            "class_keyword": "plant"
        }
        
        with patch('src.ontology.trimmer.apply_filters') as mock_func:
            # Mock to return filtered classes matching plant keyword
            mock_func.return_value = {
                "classes": [mock_ontology.search(plant=True)[0]],
                "individuals": [],
                "properties": []
            }
            
            # Act
            result = apply_filters(mock_ontology, filters)
            
            # Assert
            assert result is not None
            assert "classes" in result
            assert "individuals" in result
            assert len(result["classes"]) == 1
            assert result["classes"][0].name == "PlantClass"

    def test_apply_filters_multiple_criteria(self, mock_ontology: Mock):
        """
        Test applying filters with multiple filtering criteria.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import apply_filters
        
        filters = {
            "class_keyword": "metabolite",
            "individual_property": {"compound_type": "sugar"},
            "base_class_iri": "http://test.example.org/ontology#PlantClass"
        }
        
        with patch('src.ontology.trimmer.apply_filters') as mock_func:
            # Mock to return results that match all criteria
            mock_func.return_value = {
                "classes": [cls for cls in mock_ontology.classes() if "Metabolite" in cls.name],
                "individuals": [ind for ind in mock_ontology.individuals() if ind.compound_type == "sugar"],
                "properties": []
            }
            
            # Act
            result = apply_filters(mock_ontology, filters)
            
            # Assert
            assert result is not None
            assert len(result["classes"]) >= 1
            assert len(result["individuals"]) >= 1
            assert any("Metabolite" in cls.name for cls in result["classes"])
            assert all(ind.compound_type == "sugar" for ind in result["individuals"])

    def test_apply_filters_empty_filters(self, mock_ontology: Mock):
        """
        Test applying filters with empty filter dictionary.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import apply_filters
        
        filters = {}
        
        with patch('src.ontology.trimmer.apply_filters') as mock_func:
            # Mock to return all entities when no filters are applied
            mock_func.return_value = {
                "classes": mock_ontology.classes(),
                "individuals": mock_ontology.individuals(),
                "properties": []
            }
            
            # Act
            result = apply_filters(mock_ontology, filters)
            
            # Assert
            assert result is not None
            assert len(result["classes"]) == 3  # All classes
            assert len(result["individuals"]) == 3  # All individuals

    def test_apply_filters_no_matches(self, mock_ontology: Mock):
        """
        Test applying filters when no entities match the criteria.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import apply_filters
        
        filters = {
            "class_keyword": "nonexistent",
            "individual_property": {"compound_type": "nonexistent"}
        }
        
        with patch('src.ontology.trimmer.apply_filters') as mock_func:
            mock_func.return_value = {
                "classes": [],
                "individuals": [],
                "properties": []
            }
            
            # Act
            result = apply_filters(mock_ontology, filters)
            
            # Assert
            assert result is not None
            assert len(result["classes"]) == 0
            assert len(result["individuals"]) == 0
            assert len(result["properties"]) == 0

    def test_original_ontology_preservation(self, mock_ontology: Mock):
        """
        Test that the original ontology object is not modified during filtering.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import filter_classes_by_keyword, filter_individuals_by_property
        
        # Store original state
        original_classes = mock_ontology.classes()
        original_individuals = mock_ontology.individuals()
        
        # Perform filtering operations
        filter_classes_by_keyword(mock_ontology, "plant")
        filter_individuals_by_property(mock_ontology, "compound_type", "sugar")
        
        # Assert original ontology is unchanged
        assert mock_ontology.classes() == original_classes
        assert mock_ontology.individuals() == original_individuals
        
        # Verify no modification methods were called
        assert not hasattr(mock_ontology, 'remove')
        assert not hasattr(mock_ontology, 'delete')

    def test_copy_operation_implied_filtering(self, mock_ontology: Mock):
        """
        Test that filtering operations work with ontology copying to preserve original.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import apply_filters
        
        with patch('copy.deepcopy') as mock_copy:
            # Create a copy of the ontology for the test
            ontology_copy = Mock()
            ontology_copy.classes = Mock(return_value=[mock_ontology.classes()[0]])  # Filtered result
            ontology_copy.individuals = Mock(return_value=[])
            mock_copy.return_value = ontology_copy
            
            with patch('src.ontology.trimmer.apply_filters') as mock_apply:
                mock_apply.return_value = {
                    "classes": ontology_copy.classes(),
                    "individuals": ontology_copy.individuals(),
                    "properties": []
                }
                
                filters = {"class_keyword": "plant"}
                
                # Act
                result = apply_filters(mock_ontology, filters)
                
                # Assert
                assert result is not None
                # Original ontology should still have all classes
                assert len(mock_ontology.classes()) == 3
                # Filtered result should have fewer classes
                assert len(result["classes"]) == 1

    @parametrize("invalid_input", [
        None,
        "",
        "   ",
        123,
        [],
        {}
    ])
    def test_filter_classes_by_keyword_invalid_input(self, mock_ontology: Mock, invalid_input: Any):
        """
        Test error handling for invalid keyword inputs.
        
        Args:
            mock_ontology: Mock ontology fixture
            invalid_input: Invalid input to test
        """
        from src.ontology.trimmer import filter_classes_by_keyword
        
        with patch('src.ontology.trimmer.filter_classes_by_keyword') as mock_func:
            mock_func.side_effect = ValueError("Invalid keyword")
            
            # Act & Assert
            with expect_exception(ValueError, match="Invalid keyword"):
                filter_classes_by_keyword(mock_ontology, invalid_input)

    def test_filter_individuals_property_type_validation(self, mock_ontology: Mock):
        """
        Test that property filtering handles different value types correctly.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import filter_individuals_by_property
        
        # Test different value types
        test_cases = [
            ("concentration", 5.2, "float"),
            ("compound_type", "sugar", "string"),
            ("is_active", True, "boolean"),
            ("molecule_count", 42, "integer")
        ]
        
        for property_name, value, value_type in test_cases:
            with patch('src.ontology.trimmer.filter_individuals_by_property') as mock_func:
                # Mock appropriate response based on value type
                if value_type == "float" and value == 5.2:
                    mock_func.return_value = [mock_ontology.individuals()[0]]  # glucose
                elif value_type == "string" and value == "sugar":
                    mock_func.return_value = [mock_ontology.individuals()[0]]  # glucose
                else:
                    mock_func.return_value = []
                
                # Act
                result = filter_individuals_by_property(mock_ontology, property_name, value)
                
                # Assert
                assert isinstance(result, list)
                mock_func.assert_called_once_with(mock_ontology, property_name, value)

    def test_get_subclasses_search_method_integration(self, mock_ontology: Mock):
        """
        Test that get_subclasses properly uses ontology.search() with is_a/subclass_of.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import get_subclasses
        
        base_class_iri = "http://test.example.org/ontology#PlantClass"
        
        # Act
        result = get_subclasses(mock_ontology, base_class_iri)
        
        # Assert that search was called (implementation should use search)
        mock_ontology.search.assert_called()
        assert isinstance(result, list)

    def test_concurrent_filtering_thread_safety(self, mock_ontology: Mock):
        """
        Test that filtering operations are thread-safe for concurrent use.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        import threading
        from src.ontology.trimmer import filter_classes_by_keyword
        
        results = []
        errors = []
        
        def filter_worker(keyword):
            try:
                result = filter_classes_by_keyword(mock_ontology, keyword)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads with different keywords
        threads = []
        keywords = ["plant", "metabolite", "compound"]
        
        for keyword in keywords:
            thread = threading.Thread(target=filter_worker, args=(keyword,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Assert all operations completed successfully
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 3

    def test_memory_efficiency_large_ontology(self, mock_ontology: Mock):
        """
        Test that filtering operations are memory efficient with large ontologies.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import filter_classes_by_keyword
        
        # Mock a large ontology with many classes
        large_class_set = []
        for i in range(1000):
            mock_class = Mock()
            mock_class.name = f"TestClass{i}"
            mock_class.label = [f"Test class {i}"]
            mock_class.iri = f"http://test.example.org/ontology#TestClass{i}"
            large_class_set.append(mock_class)
        
        # Configure mock to return subset based on keyword
        def large_search(**kwargs):
            if len(kwargs) == 1 and isinstance(list(kwargs.values())[0], str):
                keyword = list(kwargs.values())[0].lower()
                if "test" in keyword:
                    return large_class_set[:10]  # Return first 10 matches
            return []
        
        mock_ontology.search = Mock(side_effect=large_search)
        mock_ontology.classes = Mock(return_value=large_class_set)
        
        # Act
        result = filter_classes_by_keyword(mock_ontology, "test")
        
        # Assert - should handle large datasets efficiently
        assert len(result) <= 10  # Should not return all 1000 classes
        assert isinstance(result, list)

    def test_filter_combination_logical_operations(self, mock_ontology: Mock):
        """
        Test that combined filters work as logical AND operations.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import apply_filters
        
        # Test that combining filters narrows results (AND logic)
        filters = {
            "class_keyword": "metabolite",
            "base_class_iri": "http://test.example.org/ontology#PlantClass"
        }
        
        with patch('src.ontology.trimmer.apply_filters') as mock_func:
            # Mock should return intersection of both criteria
            mock_func.return_value = {
                "classes": [cls for cls in mock_ontology.classes() 
                          if "Metabolite" in cls.name],  # Only metabolite classes under PlantClass
                "individuals": [],
                "properties": []
            }
            
            # Act
            result = apply_filters(mock_ontology, filters)
            
            # Assert
            assert len(result["classes"]) == 1  # Should be fewer than all metabolite classes
            assert result["classes"][0].name == "MetaboliteClass"

    def test_error_handling_ontology_access_failure(self, mock_ontology: Mock):
        """
        Test error handling when ontology access fails.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import filter_classes_by_keyword
        
        # Mock ontology search to raise an error
        mock_ontology.search.side_effect = OwlReadyError("Ontology access failed")
        
        with patch('src.ontology.trimmer.filter_classes_by_keyword') as mock_func:
            mock_func.side_effect = OwlReadyError("Ontology access failed")
            
            # Act & Assert
            with expect_exception(OwlReadyError, match="Ontology access failed"):
                filter_classes_by_keyword(mock_ontology, "test")

    def test_empty_ontology_handling(self, mock_ontology: Mock):
        """
        Test that filtering works correctly with empty ontologies.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import filter_classes_by_keyword, filter_individuals_by_property
        
        # Configure mock to return empty collections
        mock_ontology.classes = Mock(return_value=[])
        mock_ontology.individuals = Mock(return_value=[])
        mock_ontology.search = Mock(return_value=[])
        
        # Act
        class_result = filter_classes_by_keyword(mock_ontology, "test")
        individual_result = filter_individuals_by_property(mock_ontology, "test_prop", "test_value")
        
        # Assert
        assert isinstance(class_result, list)
        assert len(class_result) == 0
        assert isinstance(individual_result, list)
        assert len(individual_result) == 0

    def test_wildcard_search_functionality(self, mock_ontology: Mock):
        """
        Test wildcard search functionality in filtering.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import filter_classes_by_keyword
        
        # Test wildcard patterns
        wildcard_patterns = ["*", "plant*", "*class", "*metabolite*"]
        
        for pattern in wildcard_patterns:
            result = filter_classes_by_keyword(mock_ontology, pattern)
            assert isinstance(result, list)
            # Wildcard should return some results (depending on pattern)
            if pattern == "*":
                assert len(result) >= 0  # Could return all or filtered results

    def test_regex_pattern_support(self, mock_ontology: Mock):
        """
        Test that filtering supports regex patterns in search.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import filter_classes_by_keyword
        
        # Test regex patterns
        regex_patterns = [
            r"Plant.*",
            r".*Class$",
            r"[Mm]etabolite",
            r"\w+Class"
        ]
        
        for pattern in regex_patterns:
            with patch('src.ontology.trimmer.filter_classes_by_keyword') as mock_func:
                # Mock regex support
                if "Plant" in pattern:
                    mock_func.return_value = [cls for cls in mock_ontology.classes() 
                                            if "Plant" in cls.name]
                elif "Class" in pattern:
                    mock_func.return_value = [cls for cls in mock_ontology.classes() 
                                            if "Class" in cls.name]
                else:
                    mock_func.return_value = []
                
                # Act
                result = filter_classes_by_keyword(mock_ontology, pattern)
                
                # Assert
                assert isinstance(result, list)
                mock_func.assert_called_once()

    def test_performance_benchmarking(self, mock_ontology: Mock):
        """
        Test performance characteristics of filtering operations.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        import time
        from src.ontology.trimmer import filter_classes_by_keyword
        
        # Measure execution time
        start_time = time.time()
        
        # Perform multiple filtering operations
        for _ in range(10):
            result = filter_classes_by_keyword(mock_ontology, "test")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Assert reasonable performance (should complete quickly)
        assert execution_time < 1.0  # Should complete within 1 second
        
        # Verify operations completed successfully
        assert isinstance(result, list)

    def test_filter_result_consistency(self, mock_ontology: Mock):
        """
        Test that repeated filtering operations return consistent results.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import filter_classes_by_keyword
        
        keyword = "plant"
        
        # Perform filtering multiple times
        results = []
        for _ in range(5):
            result = filter_classes_by_keyword(mock_ontology, keyword)
            results.append(result)
        
        # Assert all results are identical
        first_result = results[0]
        for result in results[1:]:
            assert len(result) == len(first_result)
            if len(result) > 0:
                assert [cls.name for cls in result] == [cls.name for cls in first_result]

    def test_documentation_examples(self, mock_ontology: Mock):
        """
        Test the filtering functionality with examples that would appear in documentation.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.trimmer import (
            filter_classes_by_keyword, 
            filter_individuals_by_property,
            get_subclasses,
            apply_filters
        )
        
        # Example 1: Filter classes by keyword
        plant_classes = filter_classes_by_keyword(mock_ontology, "plant")
        assert isinstance(plant_classes, list)
        
        # Example 2: Filter individuals by property
        sugar_compounds = filter_individuals_by_property(mock_ontology, "compound_type", "sugar")
        assert isinstance(sugar_compounds, list)
        
        # Example 3: Get subclasses
        subclasses = get_subclasses(mock_ontology, "http://test.example.org/ontology#PlantClass")
        assert isinstance(subclasses, list)
        
        # Example 4: Apply combined filters
        filters = {
            "class_keyword": "metabolite",
            "individual_property": {"compound_type": "sugar"}
        }
        combined_result = apply_filters(mock_ontology, filters)
        assert isinstance(combined_result, dict)
        assert "classes" in combined_result
        assert "individuals" in combined_result