"""
Unit tests for the ontology scheme structural module.

This module contains comprehensive tests for creating and managing structural
annotation classes in OWL 2.0 ontologies using Owlready2. Tests cover the
creation of Chemont, NP Classifier, and PMN (Plant Metabolic Network) categories
within target ontologies, including hierarchical relationships and validation.

Test Categories:
- ChemontClass creation and validation
- NPClass creation and hierarchical relationships
- PMNCompound creation and categorization
- Class accessibility and namespace integration
- Hierarchical relationship validation (is_a relationships)
- Error handling for invalid operations
- Integration with Owlready2 Thing inheritance
- Custom exception handling
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import Any, Generator, List, Dict, Set

import pytest
from owlready2 import OwlReadyError, OwlReadyOntologyParsingError, Thing

from src.utils.testing_framework import expect_exception, parametrize


class TestSchemeStructural:
    """Test suite for structural annotation class creation and management."""

    @pytest.fixture
    def mock_ontology(self) -> Mock:
        """
        Create a mock ontology object for testing.
        
        Returns:
            Mock: Mock ontology object with namespace and class creation capabilities
        """
        mock_ont = Mock()
        mock_ont.name = "test_ontology"
        mock_ont.base_iri = "http://test.example.org/ontology"
        mock_ont.get_namespace = Mock()
        
        # Mock namespace with type creation capabilities
        mock_namespace = Mock()
        mock_namespace.base_iri = "http://test.example.org/ontology#"
        mock_ont.get_namespace.return_value = mock_namespace
        
        # Mock classes container
        mock_ont.classes = Mock()
        mock_ont.classes.return_value = []
        
        return mock_ont

    @pytest.fixture
    def mock_namespace(self, mock_ontology: Mock) -> Mock:
        """
        Create a mock namespace for class creation.
        
        Args:
            mock_ontology: Mock ontology fixture
            
        Returns:
            Mock: Mock namespace object
        """
        return mock_ontology.get_namespace()

    @pytest.fixture
    def mock_chemont_class(self) -> Mock:
        """
        Create a mock ChemontClass for testing.
        
        Returns:
            Mock: Mock ChemontClass object
        """
        mock_class = Mock()
        mock_class.name = "ChemontClass"
        mock_class.label = ["Chemical Entity Class (Chemont)"]
        mock_class.comment = ["Base class for chemical entity classification based on ChEMONT ontology"]
        mock_class.is_a = [Thing]
        mock_class.equivalent_to = []
        mock_class.instances = Mock(return_value=[])
        return mock_class

    @pytest.fixture
    def mock_np_class(self) -> Mock:
        """
        Create a mock NPClass for testing.
        
        Returns:
            Mock: Mock NPClass object
        """
        mock_class = Mock()
        mock_class.name = "NPClass"
        mock_class.label = ["Natural Product Class"]
        mock_class.comment = ["Base class for natural product classification based on NP Classifier"]
        mock_class.is_a = [Thing]
        mock_class.equivalent_to = []
        mock_class.instances = Mock(return_value=[])
        return mock_class

    @pytest.fixture
    def mock_pmn_compound(self) -> Mock:
        """
        Create a mock PMNCompound for testing.
        
        Returns:
            Mock: Mock PMNCompound object
        """
        mock_class = Mock()
        mock_class.name = "PMNCompound"
        mock_class.label = ["Plant Metabolic Network Compound"]
        mock_class.comment = ["Base class for plant metabolic compounds from PMN database"]
        mock_class.is_a = [Thing]
        mock_class.equivalent_to = []
        mock_class.instances = Mock(return_value=[])
        return mock_class

    @pytest.fixture
    def mock_chemical_class(self) -> Mock:
        """
        Create a mock ChemicalClass parent for hierarchical testing.
        
        Returns:
            Mock: Mock ChemicalClass object
        """
        mock_class = Mock()
        mock_class.name = "ChemicalClass"
        mock_class.label = ["Chemical Class"]
        mock_class.comment = ["General class for chemical entities"]
        mock_class.is_a = [Thing]
        return mock_class

    def test_create_chemont_class_success(
        self, 
        mock_ontology: Mock, 
        mock_namespace: Mock,
        mock_chemont_class: Mock
    ):
        """
        Test successful creation of a ChemontClass in the target ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_namespace: Mock namespace fixture
            mock_chemont_class: Mock ChemontClass fixture
        """
        from src.ontology.scheme_structural import create_chemont_class
        
        # Mock the types() function to return our mock class
        with patch('owlready2.types.new_class') as mock_new_class:
            mock_new_class.return_value = mock_chemont_class
            
            # Act
            result = create_chemont_class(mock_ontology, "ChemontClass")
            
            # Assert
            assert result is not None
            assert result == mock_chemont_class
            assert result.name == "ChemontClass"
            assert "Chemical Entity Class (Chemont)" in result.label
            
            # Verify new_class was called with correct parameters
            mock_new_class.assert_called_once()
            args, kwargs = mock_new_class.call_args
            assert args[0] == "ChemontClass"  # Class name
            assert Thing in args[1]  # Inherits from Thing

    def test_create_np_class_success(
        self, 
        mock_ontology: Mock, 
        mock_namespace: Mock,
        mock_np_class: Mock
    ):
        """
        Test successful creation of an NPClass in the target ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_namespace: Mock namespace fixture
            mock_np_class: Mock NPClass fixture
        """
        from src.ontology.scheme_structural import create_np_class
        
        # Mock the types() function to return our mock class
        with patch('owlready2.types.new_class') as mock_new_class:
            mock_new_class.return_value = mock_np_class
            
            # Act
            result = create_np_class(mock_ontology, "NPClass")
            
            # Assert
            assert result is not None
            assert result == mock_np_class
            assert result.name == "NPClass"
            assert "Natural Product Class" in result.label
            
            # Verify new_class was called with correct parameters
            mock_new_class.assert_called_once()
            args, kwargs = mock_new_class.call_args
            assert args[0] == "NPClass"  # Class name
            assert Thing in args[1]  # Inherits from Thing

    def test_create_pmn_compound_success(
        self, 
        mock_ontology: Mock, 
        mock_namespace: Mock,
        mock_pmn_compound: Mock
    ):
        """
        Test successful creation of a PMNCompound in the target ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_namespace: Mock namespace fixture
            mock_pmn_compound: Mock PMNCompound fixture
        """
        from src.ontology.scheme_structural import create_pmn_compound
        
        # Mock the types() function to return our mock class
        with patch('owlready2.types.new_class') as mock_new_class:
            mock_new_class.return_value = mock_pmn_compound
            
            # Act
            result = create_pmn_compound(mock_ontology, "PMNCompound")
            
            # Assert
            assert result is not None
            assert result == mock_pmn_compound
            assert result.name == "PMNCompound"
            assert "Plant Metabolic Network Compound" in result.label
            
            # Verify new_class was called with correct parameters
            mock_new_class.assert_called_once()
            args, kwargs = mock_new_class.call_args
            assert args[0] == "PMNCompound"  # Class name
            assert Thing in args[1]  # Inherits from Thing

    def test_create_hierarchical_np_class_relationship(
        self, 
        mock_ontology: Mock, 
        mock_np_class: Mock,
        mock_chemical_class: Mock
    ):
        """
        Test creation of NPClass with hierarchical relationship to ChemicalClass.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_np_class: Mock NPClass fixture
            mock_chemical_class: Mock ChemicalClass fixture
        """
        from src.ontology.scheme_structural import create_np_class_with_parent
        
        # Mock ontology search to find parent class
        mock_ontology.search_one.return_value = mock_chemical_class
        
        with patch('owlready2.types.new_class') as mock_new_class:
            # Configure mock to set is_a relationship
            def configure_hierarchy(name, bases, namespace):
                mock_np_class.is_a = list(bases)
                return mock_np_class
            
            mock_new_class.side_effect = configure_hierarchy
            
            # Act
            result = create_np_class_with_parent(
                mock_ontology, 
                "NPClass", 
                parent_class_name="ChemicalClass"
            )
            
            # Assert
            assert result is not None
            assert result == mock_np_class
            assert mock_chemical_class in result.is_a
            
            # Verify parent class was searched for
            mock_ontology.search_one.assert_called_once_with(iri="*ChemicalClass")

    def test_verify_class_accessibility_success(
        self, 
        mock_ontology: Mock,
        mock_chemont_class: Mock
    ):
        """
        Test verification that created classes are accessible in the ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_chemont_class: Mock ChemontClass fixture
        """
        from src.ontology.scheme_structural import verify_class_accessibility
        
        # Mock ontology search to find the class
        mock_ontology.search_one.return_value = mock_chemont_class
        
        # Act
        result = verify_class_accessibility(mock_ontology, "ChemontClass")
        
        # Assert
        assert result is True
        mock_ontology.search_one.assert_called_once_with(iri="*ChemontClass")

    def test_verify_class_accessibility_failure(self, mock_ontology: Mock):
        """
        Test verification failure when class is not accessible in the ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_structural import verify_class_accessibility
        
        # Mock ontology search to return None (class not found)
        mock_ontology.search_one.return_value = None
        
        # Act
        result = verify_class_accessibility(mock_ontology, "NonExistentClass")
        
        # Assert
        assert result is False
        mock_ontology.search_one.assert_called_once_with(iri="*NonExistentClass")

    def test_get_class_hierarchy_depth(
        self, 
        mock_ontology: Mock,
        mock_np_class: Mock,
        mock_chemical_class: Mock
    ):
        """
        Test calculation of class hierarchy depth for structural classes.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_np_class: Mock NPClass fixture
            mock_chemical_class: Mock ChemicalClass fixture
        """
        from src.ontology.scheme_structural import get_class_hierarchy_depth
        
        # Set up hierarchy: NPClass -> ChemicalClass -> Thing
        mock_np_class.is_a = [mock_chemical_class]
        mock_chemical_class.is_a = [Thing]
        
        mock_ontology.search_one.return_value = mock_np_class
        
        # Act
        depth = get_class_hierarchy_depth(mock_ontology, "NPClass")
        
        # Assert
        assert depth == 2  # NPClass is 2 levels below Thing

    @parametrize("class_name,expected_category", [
        ("ChemontClass", "structural_classification"),
        ("NPClass", "natural_product_classification"),  
        ("PMNCompound", "plant_metabolic_classification"),
        ("UnknownClass", "unknown_classification")
    ])
    def test_classify_structural_annotation_type(
        self, 
        class_name: str, 
        expected_category: str,
        mock_ontology: Mock
    ):
        """
        Test classification of structural annotation types based on class names.
        
        Args:
            class_name: Name of the class to classify
            expected_category: Expected classification category
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_structural import classify_structural_annotation_type
        
        # Mock class with appropriate name
        mock_class = Mock()
        mock_class.name = class_name
        mock_ontology.search_one.return_value = mock_class
        
        # Act
        category = classify_structural_annotation_type(mock_ontology, class_name)
        
        # Assert
        assert category == expected_category

    def test_create_multiple_structural_classes_batch(self, mock_ontology: Mock):
        """
        Test batch creation of multiple structural annotation classes.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_structural import create_structural_classes_batch
        
        class_specs = [
            {"name": "ChemontClass", "type": "chemont", "parent": None},
            {"name": "NPClass", "type": "np_classifier", "parent": "ChemicalClass"},
            {"name": "PMNCompound", "type": "pmn", "parent": "ChemicalClass"}
        ]
        
        created_classes = []
        
        def mock_class_factory(name, bases, namespace):
            mock_class = Mock()
            mock_class.name = name
            mock_class.is_a = list(bases)
            created_classes.append(mock_class)
            return mock_class
        
        with patch('owlready2.types.new_class', side_effect=mock_class_factory):
            # Act
            results = create_structural_classes_batch(mock_ontology, class_specs)
            
            # Assert
            assert len(results) == 3
            assert all(result is not None for result in results)
            assert len(created_classes) == 3
            
            # Verify class names
            class_names = [cls.name for cls in created_classes]
            assert "ChemontClass" in class_names
            assert "NPClass" in class_names
            assert "PMNCompound" in class_names

    def test_validate_structural_class_properties(
        self, 
        mock_ontology: Mock,
        mock_chemont_class: Mock
    ):
        """
        Test validation of required properties for structural annotation classes.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_chemont_class: Mock ChemontClass fixture
        """
        from src.ontology.scheme_structural import validate_structural_class_properties
        
        # Configure mock class with required properties
        mock_chemont_class.label = ["Chemical Entity Class (Chemont)"]
        mock_chemont_class.comment = ["Base class for chemical entity classification"]
        mock_chemont_class.is_a = [Thing]
        
        mock_ontology.search_one.return_value = mock_chemont_class
        
        # Act
        is_valid = validate_structural_class_properties(mock_ontology, "ChemontClass")
        
        # Assert
        assert is_valid is True

    def test_validate_structural_class_properties_missing_label(
        self, 
        mock_ontology: Mock,
        mock_chemont_class: Mock
    ):
        """
        Test validation failure when structural class is missing required label.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_chemont_class: Mock ChemontClass fixture
        """
        from src.ontology.scheme_structural import validate_structural_class_properties
        
        # Configure mock class with missing label
        mock_chemont_class.label = []  # Missing label
        mock_chemont_class.comment = ["Base class for chemical entity classification"]
        mock_chemont_class.is_a = [Thing]
        
        mock_ontology.search_one.return_value = mock_chemont_class
        
        # Act
        is_valid = validate_structural_class_properties(mock_ontology, "ChemontClass")
        
        # Assert
        assert is_valid is False

    def test_create_class_with_invalid_ontology(self):
        """
        Test error handling when trying to create class with invalid ontology.
        """
        from src.ontology.scheme_structural import create_chemont_class, StructuralClassError
        
        # Act & Assert
        with expect_exception(StructuralClassError, match="Invalid ontology"):
            create_chemont_class(None, "ChemontClass")

    def test_create_class_with_empty_name(self, mock_ontology: Mock):
        """
        Test error handling when trying to create class with empty name.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_structural import create_chemont_class, StructuralClassError
        
        # Act & Assert
        with expect_exception(StructuralClassError, match="Invalid class name"):
            create_chemont_class(mock_ontology, "")

    @parametrize("invalid_name", [
        None,
        "",
        "   ",
        "123InvalidName",  # Starts with number
        "Invalid Name",   # Contains space
        "Invalid-Name",   # Contains hyphen
    ])
    def test_create_class_with_invalid_names(
        self, 
        invalid_name: str,
        mock_ontology: Mock
    ):
        """
        Test error handling for various invalid class names.
        
        Args:
            invalid_name: Invalid class name to test
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_structural import create_np_class, StructuralClassError
        
        # Act & Assert
        with expect_exception(StructuralClassError, match="Invalid class name"):
            create_np_class(mock_ontology, invalid_name)

    def test_create_class_owlready_error_handling(self, mock_ontology: Mock):
        """
        Test error handling when Owlready2 operations fail.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_structural import create_pmn_compound, StructuralClassError
        
        with patch('owlready2.types.new_class') as mock_new_class:
            # Mock Owlready2 error
            mock_new_class.side_effect = OwlReadyError("Owlready2 operation failed")
            
            # Act & Assert
            with expect_exception(StructuralClassError, match="Owlready2 error"):
                create_pmn_compound(mock_ontology, "PMNCompound")

    def test_verify_thing_inheritance(
        self, 
        mock_ontology: Mock,
        mock_chemont_class: Mock
    ):
        """
        Test verification that structural classes properly inherit from Thing.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_chemont_class: Mock ChemontClass fixture
        """
        from src.ontology.scheme_structural import verify_thing_inheritance
        
        # Configure class to inherit from Thing
        mock_chemont_class.is_a = [Thing]
        mock_ontology.search_one.return_value = mock_chemont_class
        
        # Act
        inherits_from_thing = verify_thing_inheritance(mock_ontology, "ChemontClass")
        
        # Assert
        assert inherits_from_thing is True

    def test_verify_thing_inheritance_failure(
        self, 
        mock_ontology: Mock,
        mock_chemont_class: Mock
    ):
        """
        Test verification failure when class doesn't inherit from Thing.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_chemont_class: Mock ChemontClass fixture
        """
        from src.ontology.scheme_structural import verify_thing_inheritance
        
        # Configure class to not inherit from Thing
        mock_other_class = Mock()
        mock_chemont_class.is_a = [mock_other_class]  # Doesn't inherit from Thing
        mock_ontology.search_one.return_value = mock_chemont_class
        
        # Act
        inherits_from_thing = verify_thing_inheritance(mock_ontology, "ChemontClass")
        
        # Assert
        assert inherits_from_thing is False

    def test_namespace_integration(
        self, 
        mock_ontology: Mock, 
        mock_namespace: Mock,
        mock_chemont_class: Mock
    ):
        """
        Test that classes are properly integrated with ontology namespace.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_namespace: Mock namespace fixture
            mock_chemont_class: Mock ChemontClass fixture
        """
        from src.ontology.scheme_structural import create_chemont_class
        
        with patch('owlready2.types.new_class') as mock_new_class:
            mock_new_class.return_value = mock_chemont_class
            
            # Act
            result = create_chemont_class(mock_ontology, "ChemontClass")
            
            # Assert
            assert result is not None
            
            # Verify namespace was accessed
            mock_ontology.get_namespace.assert_called_once()
            
            # Verify class creation used correct namespace
            mock_new_class.assert_called_once()
            args, kwargs = mock_new_class.call_args
            # Namespace parameter should be the third argument or in kwargs
            assert len(args) >= 3 or 'namespace' in kwargs

    def test_concurrent_class_creation_thread_safety(self, mock_ontology: Mock):
        """
        Test thread safety when creating multiple structural classes concurrently.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        import threading
        from src.ontology.scheme_structural import create_chemont_class
        
        results = []
        errors = []
        
        def create_class_worker(class_name: str):
            try:
                with patch('owlready2.types.new_class') as mock_new_class:
                    mock_class = Mock()
                    mock_class.name = class_name
                    mock_new_class.return_value = mock_class
                    
                    result = create_chemont_class(mock_ontology, class_name)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        class_names = ["ChemontClass1", "ChemontClass2", "ChemontClass3"]
        
        for class_name in class_names:
            thread = threading.Thread(target=create_class_worker, args=(class_name,))
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

    def test_structural_class_error_custom_exception(self):
        """
        Test that custom StructuralClassError exception works correctly.
        """
        from src.ontology.scheme_structural import StructuralClassError
        
        # Test basic exception creation
        error_msg = "Test structural class error"
        exception = StructuralClassError(error_msg)
        
        assert str(exception) == error_msg
        assert isinstance(exception, Exception)

    def test_structural_class_error_with_cause(self):
        """
        Test that StructuralClassError properly handles exception chaining.
        """
        from src.ontology.scheme_structural import StructuralClassError
        
        # Test exception chaining
        original_error = ValueError("Original error")
        try:
            raise StructuralClassError("Wrapped structural error") from original_error
        except StructuralClassError as chained_error:
            assert str(chained_error) == "Wrapped structural error"
            assert chained_error.__cause__ == original_error

    def test_get_all_structural_classes(self, mock_ontology: Mock):
        """
        Test retrieval of all structural annotation classes from ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_structural import get_all_structural_classes
        
        # Mock structural classes
        mock_chemont = Mock()
        mock_chemont.name = "ChemontClass"
        mock_np = Mock()
        mock_np.name = "NPClass"
        mock_pmn = Mock()
        mock_pmn.name = "PMNCompound"
        
        # Mock ontology search to return structural classes
        mock_ontology.classes.return_value = [mock_chemont, mock_np, mock_pmn]
        
        # Act
        structural_classes = get_all_structural_classes(mock_ontology)
        
        # Assert
        assert len(structural_classes) == 3
        class_names = [cls.name for cls in structural_classes]
        assert "ChemontClass" in class_names
        assert "NPClass" in class_names
        assert "PMNCompound" in class_names

    def test_structural_class_metadata_validation(
        self, 
        mock_ontology: Mock,
        mock_chemont_class: Mock
    ):
        """
        Test validation of structural class metadata and annotations.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_chemont_class: Mock ChemontClass fixture
        """
        from src.ontology.scheme_structural import validate_class_metadata
        
        # Configure class with complete metadata
        mock_chemont_class.label = ["Chemical Entity Class (Chemont)"]
        mock_chemont_class.comment = ["Base class for chemical entity classification based on ChEMONT ontology"]
        mock_chemont_class.iri = "http://test.example.org/ontology#ChemontClass"
        
        # Mock custom annotations
        mock_chemont_class.classification_system = ["ChEMONT"]
        mock_chemont_class.version = ["1.0"]
        
        mock_ontology.search_one.return_value = mock_chemont_class
        
        # Act
        metadata_valid = validate_class_metadata(mock_ontology, "ChemontClass")
        
        # Assert
        assert metadata_valid is True

    def test_cleanup_structural_classes(self, mock_ontology: Mock):
        """
        Test cleanup of structural annotation classes from ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_structural import cleanup_structural_classes
        
        # Mock structural classes to be cleaned up
        mock_classes = [Mock(), Mock(), Mock()]
        for i, mock_class in enumerate(mock_classes):
            mock_class.name = f"StructuralClass{i}"
            mock_class.destroy = Mock()
        
        mock_ontology.classes.return_value = mock_classes
        
        # Act
        cleanup_count = cleanup_structural_classes(mock_ontology)
        
        # Assert
        assert cleanup_count == 3
        for mock_class in mock_classes:
            mock_class.destroy.assert_called_once()

    def test_integration_with_owlready2_thing(self, mock_ontology: Mock):
        """
        Test integration with Owlready2 Thing class for proper inheritance.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_structural import create_chemont_class
        
        with patch('owlready2.types.new_class') as mock_new_class:
            # Verify Thing is imported and used correctly
            mock_class = Mock()
            mock_class.is_a = [Thing]
            mock_new_class.return_value = mock_class
            
            # Act
            result = create_chemont_class(mock_ontology, "ChemontClass")
            
            # Assert
            assert result is not None
            assert Thing in result.is_a
            
            # Verify new_class was called with Thing as base
            args, kwargs = mock_new_class.call_args
            assert Thing in args[1]  # Base classes tuple