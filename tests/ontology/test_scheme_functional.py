"""
Unit tests for the ontology scheme functional module.

This module contains comprehensive tests for creating and managing functional
annotation classes in OWL 2.0 ontologies using Owlready2. Tests cover the
creation of GO (Gene Ontology), Trait Ontology, and ChemFont categories
within target ontologies, including hierarchical relationships and validation.

Test Categories:
- MolecularTrait class creation and validation
- PlantTrait class creation and hierarchical relationships
- HumanTrait class creation and categorization
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


class TestSchemeFunctional:
    """Test suite for functional annotation class creation and management."""

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
        
        # Mock context manager protocol
        mock_ont.__enter__ = Mock(return_value=mock_ont)
        mock_ont.__exit__ = Mock(return_value=None)
        
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
    def mock_molecular_trait_class(self) -> Mock:
        """
        Create a mock MolecularTrait class for testing.
        
        Returns:
            Mock: Mock MolecularTrait object
        """
        mock_class = Mock()
        mock_class.name = "MolecularTrait"
        mock_class.label = ["Molecular Function Trait"]
        mock_class.comment = ["Base class for molecular-level functional traits from GO"]
        mock_class.is_a = [Thing]
        mock_class.equivalent_to = []
        mock_class.instances = Mock(return_value=[])
        return mock_class

    @pytest.fixture
    def mock_plant_trait_class(self) -> Mock:
        """
        Create a mock PlantTrait class for testing.
        
        Returns:
            Mock: Mock PlantTrait object
        """
        mock_class = Mock()
        mock_class.name = "PlantTrait"
        mock_class.label = ["Plant Functional Trait"]
        mock_class.comment = ["Base class for plant-specific functional traits from Trait Ontology"]
        mock_class.is_a = [Thing]
        mock_class.equivalent_to = []
        mock_class.instances = Mock(return_value=[])
        return mock_class

    @pytest.fixture
    def mock_human_trait_class(self) -> Mock:
        """
        Create a mock HumanTrait class for testing.
        
        Returns:
            Mock: Mock HumanTrait object
        """
        mock_class = Mock()
        mock_class.name = "HumanTrait"
        mock_class.label = ["Human Functional Trait"]
        mock_class.comment = ["Base class for human-related functional traits from ChemFont"]
        mock_class.is_a = [Thing]
        mock_class.equivalent_to = []
        mock_class.instances = Mock(return_value=[])
        return mock_class

    @pytest.fixture
    def mock_drought_tolerance_class(self) -> Mock:
        """
        Create a mock DroughtTolerance class for hierarchical testing.
        
        Returns:
            Mock: Mock DroughtTolerance class object
        """
        mock_class = Mock()
        mock_class.name = "DroughtTolerance"
        mock_class.label = ["Drought Tolerance Trait"]
        mock_class.comment = ["Plant trait related to drought stress tolerance"]
        mock_class.is_a = [Thing]
        return mock_class

    @pytest.fixture
    def mock_atpase_activity_class(self) -> Mock:
        """
        Create a mock ATPase_activity class for hierarchical testing.
        
        Returns:
            Mock: Mock ATPase_activity class object
        """
        mock_class = Mock()
        mock_class.name = "ATPase_activity"
        mock_class.label = ["ATPase Activity"]
        mock_class.comment = ["Molecular function related to ATP hydrolysis"]
        mock_class.is_a = [Thing]
        return mock_class

    def test_create_molecular_trait_class_success(
        self, 
        mock_ontology: Mock, 
        mock_namespace: Mock,
        mock_molecular_trait_class: Mock
    ):
        """
        Test successful creation of a MolecularTrait class in the target ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_namespace: Mock namespace fixture
            mock_molecular_trait_class: Mock MolecularTrait fixture
        """
        from src.ontology.scheme_functional import create_molecular_trait_class
        
        # Mock the types() function to return our mock class
        with patch('owlready2.types.new_class') as mock_new_class:
            mock_new_class.return_value = mock_molecular_trait_class
            
            # Act
            result = create_molecular_trait_class(mock_ontology, "MolecularTrait")
            
            # Assert
            assert result is not None
            assert result == mock_molecular_trait_class
            assert result.name == "MolecularTrait"
            assert "Molecular Function Trait" in result.label
            
            # Verify new_class was called with correct parameters
            mock_new_class.assert_called_once()
            args, kwargs = mock_new_class.call_args
            assert args[0] == "MolecularTrait"  # Class name
            assert Thing in args[1]  # Inherits from Thing

    def test_create_plant_trait_class_success(
        self, 
        mock_ontology: Mock, 
        mock_namespace: Mock,
        mock_plant_trait_class: Mock
    ):
        """
        Test successful creation of a PlantTrait class in the target ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_namespace: Mock namespace fixture
            mock_plant_trait_class: Mock PlantTrait fixture
        """
        from src.ontology.scheme_functional import create_plant_trait_class
        
        # Mock the types() function to return our mock class
        with patch('owlready2.types.new_class') as mock_new_class:
            mock_new_class.return_value = mock_plant_trait_class
            
            # Act
            result = create_plant_trait_class(mock_ontology, "PlantTrait")
            
            # Assert
            assert result is not None
            assert result == mock_plant_trait_class
            assert result.name == "PlantTrait"
            assert "Plant Functional Trait" in result.label
            
            # Verify new_class was called with correct parameters
            mock_new_class.assert_called_once()
            args, kwargs = mock_new_class.call_args
            assert args[0] == "PlantTrait"  # Class name
            assert Thing in args[1]  # Inherits from Thing

    def test_create_human_trait_class_success(
        self, 
        mock_ontology: Mock, 
        mock_namespace: Mock,
        mock_human_trait_class: Mock
    ):
        """
        Test successful creation of a HumanTrait class in the target ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_namespace: Mock namespace fixture
            mock_human_trait_class: Mock HumanTrait fixture
        """
        from src.ontology.scheme_functional import create_human_trait_class
        
        # Mock the types() function to return our mock class
        with patch('owlready2.types.new_class') as mock_new_class:
            mock_new_class.return_value = mock_human_trait_class
            
            # Act
            result = create_human_trait_class(mock_ontology, "HumanTrait")
            
            # Assert
            assert result is not None
            assert result == mock_human_trait_class
            assert result.name == "HumanTrait"
            assert "Human Functional Trait" in result.label
            
            # Verify new_class was called with correct parameters
            mock_new_class.assert_called_once()
            args, kwargs = mock_new_class.call_args
            assert args[0] == "HumanTrait"  # Class name
            assert Thing in args[1]  # Inherits from Thing

    def test_create_hierarchical_drought_tolerance_relationship(
        self, 
        mock_ontology: Mock, 
        mock_plant_trait_class: Mock,
        mock_drought_tolerance_class: Mock
    ):
        """
        Test creation of DroughtTolerance class with hierarchical relationship to PlantTrait.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_plant_trait_class: Mock PlantTrait fixture
            mock_drought_tolerance_class: Mock DroughtTolerance fixture
        """
        from src.ontology.scheme_functional import create_drought_tolerance_class_with_parent
        
        # Mock ontology search to find parent class
        mock_ontology.search_one.return_value = mock_plant_trait_class
        
        with patch('owlready2.types.new_class') as mock_new_class:
            # Configure mock to set is_a relationship
            def configure_hierarchy(name, bases, namespace):
                mock_drought_tolerance_class.is_a = list(bases)
                return mock_drought_tolerance_class
            
            mock_new_class.side_effect = configure_hierarchy
            
            # Act
            result = create_drought_tolerance_class_with_parent(
                mock_ontology, 
                "DroughtTolerance", 
                parent_class_name="PlantTrait"
            )
            
            # Assert
            assert result is not None
            assert result == mock_drought_tolerance_class
            assert mock_plant_trait_class in result.is_a
            
            # Verify parent class was searched for
            mock_ontology.search_one.assert_called_once_with(iri="*PlantTrait")

    def test_create_hierarchical_atpase_activity_relationship(
        self, 
        mock_ontology: Mock, 
        mock_molecular_trait_class: Mock,
        mock_atpase_activity_class: Mock
    ):
        """
        Test creation of ATPase_activity class with hierarchical relationship to MolecularTrait.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_molecular_trait_class: Mock MolecularTrait fixture
            mock_atpase_activity_class: Mock ATPase_activity fixture
        """
        from src.ontology.scheme_functional import create_atpase_activity_class_with_parent
        
        # Mock ontology search to find parent class
        mock_ontology.search_one.return_value = mock_molecular_trait_class
        
        with patch('owlready2.types.new_class') as mock_new_class:
            # Configure mock to set is_a relationship
            def configure_hierarchy(name, bases, namespace):
                mock_atpase_activity_class.is_a = list(bases)
                return mock_atpase_activity_class
            
            mock_new_class.side_effect = configure_hierarchy
            
            # Act
            result = create_atpase_activity_class_with_parent(
                mock_ontology, 
                "ATPase_activity", 
                parent_class_name="MolecularTrait"
            )
            
            # Assert
            assert result is not None
            assert result == mock_atpase_activity_class
            assert mock_molecular_trait_class in result.is_a
            
            # Verify parent class was searched for
            mock_ontology.search_one.assert_called_once_with(iri="*MolecularTrait")

    def test_verify_class_accessibility_success(
        self, 
        mock_ontology: Mock,
        mock_molecular_trait_class: Mock
    ):
        """
        Test verification that created classes are accessible in the ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_molecular_trait_class: Mock MolecularTrait fixture
        """
        from src.ontology.scheme_functional import verify_class_accessibility
        
        # Mock ontology search to find the class
        mock_ontology.search_one.return_value = mock_molecular_trait_class
        
        # Act
        result = verify_class_accessibility(mock_ontology, "MolecularTrait")
        
        # Assert
        assert result is True
        mock_ontology.search_one.assert_called_once_with(iri="*MolecularTrait")

    def test_verify_class_accessibility_failure(self, mock_ontology: Mock):
        """
        Test verification failure when class is not accessible in the ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_functional import verify_class_accessibility
        
        # Mock ontology search to return None (class not found)
        mock_ontology.search_one.return_value = None
        
        # Act
        result = verify_class_accessibility(mock_ontology, "NonExistentTrait")
        
        # Assert
        assert result is False
        mock_ontology.search_one.assert_called_once_with(iri="*NonExistentTrait")

    def test_get_class_hierarchy_depth(
        self, 
        mock_ontology: Mock,
        mock_drought_tolerance_class: Mock,
        mock_plant_trait_class: Mock
    ):
        """
        Test calculation of class hierarchy depth for functional classes.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_drought_tolerance_class: Mock DroughtTolerance fixture
            mock_plant_trait_class: Mock PlantTrait fixture
        """
        from src.ontology.scheme_functional import get_class_hierarchy_depth
        
        # Set up hierarchy: DroughtTolerance -> PlantTrait -> Thing
        mock_drought_tolerance_class.is_a = [mock_plant_trait_class]
        mock_plant_trait_class.is_a = [Thing]
        
        mock_ontology.search_one.return_value = mock_drought_tolerance_class
        
        # Act
        depth = get_class_hierarchy_depth(mock_ontology, "DroughtTolerance")
        
        # Assert
        assert depth == 2  # DroughtTolerance is 2 levels below Thing

    @parametrize("class_name,expected_category", [
        ("MolecularTrait", "go_molecular_function"),
        ("PlantTrait", "trait_ontology_classification"),  
        ("HumanTrait", "chemfont_classification"),
        ("UnknownTrait", "unknown_classification")
    ])
    def test_classify_functional_annotation_type(
        self, 
        class_name: str, 
        expected_category: str,
        mock_ontology: Mock
    ):
        """
        Test classification of functional annotation types based on class names.
        
        Args:
            class_name: Name of the class to classify
            expected_category: Expected classification category
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_functional import classify_functional_annotation_type
        
        # Mock class with appropriate name
        mock_class = Mock()
        mock_class.name = class_name
        mock_ontology.search_one.return_value = mock_class
        
        # Act
        category = classify_functional_annotation_type(mock_ontology, class_name)
        
        # Assert
        assert category == expected_category

    def test_create_multiple_functional_classes_batch(self, mock_ontology: Mock):
        """
        Test batch creation of multiple functional annotation classes.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_functional import create_functional_classes_batch
        
        class_specs = [
            {"name": "MolecularTrait", "type": "go_molecular_function", "parent": None},
            {"name": "PlantTrait", "type": "trait_ontology", "parent": None},
            {"name": "DroughtTolerance", "type": "trait_ontology", "parent": "PlantTrait"}
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
            results = create_functional_classes_batch(mock_ontology, class_specs)
            
            # Assert
            assert len(results) == 3
            assert all(result is not None for result in results)
            assert len(created_classes) == 3
            
            # Verify class names
            class_names = [cls.name for cls in created_classes]
            assert "MolecularTrait" in class_names
            assert "PlantTrait" in class_names
            assert "DroughtTolerance" in class_names

    def test_validate_functional_class_properties(
        self, 
        mock_ontology: Mock,
        mock_molecular_trait_class: Mock
    ):
        """
        Test validation of required properties for functional annotation classes.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_molecular_trait_class: Mock MolecularTrait fixture
        """
        from src.ontology.scheme_functional import validate_functional_class_properties
        
        # Configure mock class with required properties
        mock_molecular_trait_class.label = ["Molecular Function Trait"]
        mock_molecular_trait_class.comment = ["Base class for molecular-level functional traits"]
        mock_molecular_trait_class.is_a = [Thing]
        
        mock_ontology.search_one.return_value = mock_molecular_trait_class
        
        # Act
        is_valid = validate_functional_class_properties(mock_ontology, "MolecularTrait")
        
        # Assert
        assert is_valid is True

    def test_validate_functional_class_properties_missing_label(
        self, 
        mock_ontology: Mock,
        mock_molecular_trait_class: Mock
    ):
        """
        Test validation failure when functional class is missing required label.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_molecular_trait_class: Mock MolecularTrait fixture
        """
        from src.ontology.scheme_functional import validate_functional_class_properties
        
        # Configure mock class with missing label
        mock_molecular_trait_class.label = []  # Missing label
        mock_molecular_trait_class.comment = ["Base class for molecular-level functional traits"]
        mock_molecular_trait_class.is_a = [Thing]
        
        mock_ontology.search_one.return_value = mock_molecular_trait_class
        
        # Act
        is_valid = validate_functional_class_properties(mock_ontology, "MolecularTrait")
        
        # Assert
        assert is_valid is False

    def test_create_class_with_invalid_ontology(self):
        """
        Test error handling when trying to create class with invalid ontology.
        """
        from src.ontology.scheme_functional import create_molecular_trait_class, FunctionalClassError
        
        # Act & Assert
        with expect_exception(FunctionalClassError, match="Invalid ontology"):
            create_molecular_trait_class(None, "MolecularTrait")

    def test_create_class_with_empty_name(self, mock_ontology: Mock):
        """
        Test error handling when trying to create class with empty name.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_functional import create_molecular_trait_class, FunctionalClassError
        
        # Act & Assert
        with expect_exception(FunctionalClassError, match="Invalid class name"):
            create_molecular_trait_class(mock_ontology, "")

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
        from src.ontology.scheme_functional import create_plant_trait_class, FunctionalClassError
        
        # Act & Assert
        with expect_exception(FunctionalClassError, match="Invalid class name"):
            create_plant_trait_class(mock_ontology, invalid_name)

    def test_create_class_owlready_error_handling(self, mock_ontology: Mock):
        """
        Test error handling when Owlready2 operations fail.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_functional import create_human_trait_class, FunctionalClassError
        
        with patch('owlready2.types.new_class') as mock_new_class:
            # Mock Owlready2 error
            mock_new_class.side_effect = OwlReadyError("Owlready2 operation failed")
            
            # Act & Assert
            with expect_exception(FunctionalClassError, match="Owlready2 error"):
                create_human_trait_class(mock_ontology, "HumanTrait")

    def test_verify_thing_inheritance(
        self, 
        mock_ontology: Mock,
        mock_molecular_trait_class: Mock
    ):
        """
        Test verification that functional classes properly inherit from Thing.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_molecular_trait_class: Mock MolecularTrait fixture
        """
        from src.ontology.scheme_functional import verify_thing_inheritance
        
        # Configure class to inherit from Thing
        mock_molecular_trait_class.is_a = [Thing]
        mock_ontology.search_one.return_value = mock_molecular_trait_class
        
        # Act
        inherits_from_thing = verify_thing_inheritance(mock_ontology, "MolecularTrait")
        
        # Assert
        assert inherits_from_thing is True

    def test_verify_thing_inheritance_failure(
        self, 
        mock_ontology: Mock,
        mock_molecular_trait_class: Mock
    ):
        """
        Test verification failure when class doesn't inherit from Thing.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_molecular_trait_class: Mock MolecularTrait fixture
        """
        from src.ontology.scheme_functional import verify_thing_inheritance
        
        # Configure class to not inherit from Thing
        mock_other_class = Mock()
        mock_molecular_trait_class.is_a = [mock_other_class]  # Doesn't inherit from Thing
        mock_ontology.search_one.return_value = mock_molecular_trait_class
        
        # Act
        inherits_from_thing = verify_thing_inheritance(mock_ontology, "MolecularTrait")
        
        # Assert
        assert inherits_from_thing is False

    def test_namespace_integration(
        self, 
        mock_ontology: Mock, 
        mock_namespace: Mock,
        mock_molecular_trait_class: Mock
    ):
        """
        Test that classes are properly integrated with ontology namespace.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_namespace: Mock namespace fixture
            mock_molecular_trait_class: Mock MolecularTrait fixture
        """
        from src.ontology.scheme_functional import create_molecular_trait_class
        
        with patch('owlready2.types.new_class') as mock_new_class:
            mock_new_class.return_value = mock_molecular_trait_class
            
            # Act
            result = create_molecular_trait_class(mock_ontology, "MolecularTrait")
            
            # Assert
            assert result is not None
            
            # Verify namespace was accessed
            assert mock_ontology.get_namespace.call_count >= 1
            
            # Verify class creation used correct namespace
            mock_new_class.assert_called_once()
            args, kwargs = mock_new_class.call_args
            # Namespace parameter should be the third argument or in kwargs
            assert len(args) >= 3 or 'namespace' in kwargs

    def test_concurrent_class_creation_thread_safety(self, mock_ontology: Mock):
        """
        Test thread safety when creating multiple functional classes concurrently.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        import threading
        from src.ontology.scheme_functional import create_molecular_trait_class
        
        results = []
        errors = []
        
        def create_class_worker(class_name: str):
            try:
                with patch('owlready2.types.new_class') as mock_new_class:
                    mock_class = Mock()
                    mock_class.name = class_name
                    mock_new_class.return_value = mock_class
                    
                    result = create_molecular_trait_class(mock_ontology, class_name)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        class_names = ["MolecularTrait1", "MolecularTrait2", "MolecularTrait3"]
        
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

    def test_functional_class_error_custom_exception(self):
        """
        Test that custom FunctionalClassError exception works correctly.
        """
        from src.ontology.scheme_functional import FunctionalClassError
        
        # Test basic exception creation
        error_msg = "Test functional class error"
        exception = FunctionalClassError(error_msg)
        
        assert str(exception) == error_msg
        assert isinstance(exception, Exception)

    def test_functional_class_error_with_cause(self):
        """
        Test that FunctionalClassError properly handles exception chaining.
        """
        from src.ontology.scheme_functional import FunctionalClassError
        
        # Test exception chaining
        original_error = ValueError("Original error")
        try:
            raise FunctionalClassError("Wrapped functional error") from original_error
        except FunctionalClassError as chained_error:
            assert str(chained_error) == "Wrapped functional error"
            assert chained_error.__cause__ == original_error

    def test_get_all_functional_classes(self, mock_ontology: Mock):
        """
        Test retrieval of all functional annotation classes from ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_functional import get_all_functional_classes
        
        # Mock functional classes
        mock_molecular_trait = Mock()
        mock_molecular_trait.name = "MolecularTrait"
        mock_plant_trait = Mock()
        mock_plant_trait.name = "PlantTrait"
        mock_human_trait = Mock()
        mock_human_trait.name = "HumanTrait"
        
        # Mock ontology search to return functional classes
        mock_ontology.classes.return_value = [mock_molecular_trait, mock_plant_trait, mock_human_trait]
        
        # Act
        functional_classes = get_all_functional_classes(mock_ontology)
        
        # Assert
        assert len(functional_classes) == 3
        class_names = [cls.name for cls in functional_classes]
        assert "MolecularTrait" in class_names
        assert "PlantTrait" in class_names
        assert "HumanTrait" in class_names

    def test_functional_class_metadata_validation(
        self, 
        mock_ontology: Mock,
        mock_molecular_trait_class: Mock
    ):
        """
        Test validation of functional class metadata and annotations.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_molecular_trait_class: Mock MolecularTrait fixture
        """
        from src.ontology.scheme_functional import validate_class_metadata
        
        # Configure class with complete metadata
        mock_molecular_trait_class.label = ["Molecular Function Trait"]
        mock_molecular_trait_class.comment = ["Base class for molecular-level functional traits from GO"]
        mock_molecular_trait_class.iri = "http://test.example.org/ontology#MolecularTrait"
        
        # Mock custom annotations
        mock_molecular_trait_class.classification_system = ["Gene Ontology"]
        mock_molecular_trait_class.version = ["1.0"]
        
        mock_ontology.search_one.return_value = mock_molecular_trait_class
        
        # Act
        metadata_valid = validate_class_metadata(mock_ontology, "MolecularTrait")
        
        # Assert
        assert metadata_valid is True

    def test_cleanup_functional_classes(self, mock_ontology: Mock):
        """
        Test cleanup of functional annotation classes from ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_functional import cleanup_functional_classes
        
        # Mock functional classes to be cleaned up
        mock_classes = [Mock(), Mock(), Mock()]
        for i, mock_class in enumerate(mock_classes):
            mock_class.name = f"FunctionalClass{i}"
            mock_class.destroy = Mock()
        
        mock_ontology.classes.return_value = mock_classes
        
        # Act
        cleanup_count = cleanup_functional_classes(mock_ontology)
        
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
        from src.ontology.scheme_functional import create_molecular_trait_class
        
        with patch('owlready2.types.new_class') as mock_new_class:
            # Verify Thing is imported and used correctly
            mock_class = Mock()
            mock_class.is_a = [Thing]
            mock_new_class.return_value = mock_class
            
            # Act
            result = create_molecular_trait_class(mock_ontology, "MolecularTrait")
            
            # Assert
            assert result is not None
            assert Thing in result.is_a
            
            # Verify new_class was called with Thing as base
            args, kwargs = mock_new_class.call_args
            assert Thing in args[1]  # Base classes tuple

    def test_add_initial_key_terms_success(self, mock_ontology: Mock):
        """
        Test successful addition of initial key terms from GO, Trait Ontology, and ChemFont.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_functional import add_initial_key_terms
        
        # Mock the required classes
        mock_molecular_trait_class = Mock()
        mock_plant_trait_class = Mock()
        mock_human_trait_class = Mock()
        
        # Mock search_one to return the required classes
        def search_side_effect(iri):
            if "MolecularTrait" in iri:
                return mock_molecular_trait_class
            elif "PlantTrait" in iri:
                return mock_plant_trait_class
            elif "HumanTrait" in iri:
                return mock_human_trait_class
            return None
        
        mock_ontology.search_one.side_effect = search_side_effect
        
        # Mock the ontology context manager
        mock_ontology.__enter__ = Mock(return_value=mock_ontology)
        mock_ontology.__exit__ = Mock(return_value=None)
        
        # Mock instance creation
        mock_go_instances = []
        mock_trait_instances = []
        mock_chemfont_instances = []
        
        def create_go_instance(name):
            instance = Mock()
            instance.name = name
            instance.label = []
            instance.comment = []
            mock_go_instances.append(instance)
            return instance
        
        def create_trait_instance(name):
            instance = Mock()
            instance.name = name
            instance.label = []
            instance.comment = []
            mock_trait_instances.append(instance)
            return instance
        
        def create_chemfont_instance(name):
            instance = Mock()
            instance.name = name
            instance.label = []
            instance.comment = []
            mock_chemfont_instances.append(instance)
            return instance
        
        mock_molecular_trait_class.side_effect = create_go_instance
        mock_plant_trait_class.side_effect = create_trait_instance
        mock_human_trait_class.side_effect = create_chemfont_instance
        
        # Act
        result = add_initial_key_terms(mock_ontology)
        
        # Assert
        assert result is not None
        assert 'go_instances' in result
        assert 'trait_ontology_instances' in result
        assert 'chemfont_instances' in result
        
        # Verify expected number of instances were created
        assert len(result['go_instances']) == 8  # Expected GO instances
        assert len(result['trait_ontology_instances']) == 8  # Expected Trait Ontology instances
        assert len(result['chemfont_instances']) == 8  # Expected ChemFont instances
        
        # Verify instance properties were set
        for instance in result['go_instances']:
            assert instance.label is not None
            assert instance.comment is not None
        
        for instance in result['trait_ontology_instances']:
            assert instance.label is not None
            assert instance.comment is not None
        
        for instance in result['chemfont_instances']:
            assert instance.label is not None
            assert instance.comment is not None

    def test_add_initial_key_terms_missing_classes(self, mock_ontology: Mock):
        """
        Test add_initial_key_terms when required classes are missing.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_functional import add_initial_key_terms, FunctionalClassError
        
        # Mock search_one to return None (classes not found)
        mock_ontology.search_one.return_value = None
        
        # Act & Assert
        with expect_exception(FunctionalClassError, "Required functional classes not found"):
            add_initial_key_terms(mock_ontology)

    def test_add_initial_key_terms_specific_instances(self, mock_ontology: Mock):
        """
        Test creation of specific representative instances from GO, Trait Ontology, and ChemFont.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_functional import add_initial_key_terms
        
        # Mock the required classes
        mock_molecular_trait_class = Mock()
        mock_plant_trait_class = Mock()
        mock_human_trait_class = Mock()
        
        def search_side_effect(iri):
            if "MolecularTrait" in iri:
                return mock_molecular_trait_class
            elif "PlantTrait" in iri:
                return mock_plant_trait_class
            elif "HumanTrait" in iri:
                return mock_human_trait_class
            return None
        
        mock_ontology.search_one.side_effect = search_side_effect
        
        # Mock the ontology context manager
        mock_ontology.__enter__ = Mock(return_value=mock_ontology)
        mock_ontology.__exit__ = Mock(return_value=None)
        
        # Track created instances with their names
        created_instances = []
        
        def create_instance(name):
            instance = Mock()
            instance.name = name
            instance.label = []
            instance.comment = []
            created_instances.append(instance)
            return instance
        
        mock_molecular_trait_class.side_effect = create_instance
        mock_plant_trait_class.side_effect = create_instance
        mock_human_trait_class.side_effect = create_instance
        
        # Act
        result = add_initial_key_terms(mock_ontology)
        
        # Assert specific representative instances were created
        instance_names = [inst.name for inst in created_instances]
        
        # Check for expected GO instances
        assert "ATPase_activity" in instance_names
        assert "DNA_binding" in instance_names
        assert "Catalytic_activity" in instance_names
        assert "Transporter_activity" in instance_names
        
        # Check for expected Trait Ontology instances
        assert "DroughtTolerance" in instance_names
        assert "FloweringTime" in instance_names
        assert "PlantHeight" in instance_names
        assert "SeedWeight" in instance_names
        
        # Check for expected ChemFont instances
        assert "Toxicity" in instance_names
        assert "Bioavailability" in instance_names
        assert "MetabolicStability" in instance_names
        assert "DrugLikeness" in instance_names

    def test_add_initial_key_terms_with_owlready_error(self, mock_ontology: Mock):
        """
        Test add_initial_key_terms handling of Owlready2 errors.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_functional import add_initial_key_terms, FunctionalClassError
        
        # Mock search_one to raise OwlReadyError
        mock_ontology.search_one.side_effect = OwlReadyError("Owlready2 error")
        
        # Act & Assert
        with expect_exception(FunctionalClassError, "Owlready2 error creating initial key terms"):
            add_initial_key_terms(mock_ontology)

    def test_validate_initial_key_terms_success(self, mock_ontology: Mock):
        """
        Test successful validation of initial key terms.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_functional import validate_initial_key_terms
        
        # Mock the required classes
        mock_molecular_trait_class = Mock()
        mock_plant_trait_class = Mock()
        mock_human_trait_class = Mock()
        
        def search_side_effect(iri):
            if "MolecularTrait" in iri:
                return mock_molecular_trait_class
            elif "PlantTrait" in iri:
                return mock_plant_trait_class
            elif "HumanTrait" in iri:
                return mock_human_trait_class
            return None
        
        mock_ontology.search_one.side_effect = search_side_effect
        
        # Mock instances with proper labels and comments
        def create_mock_instances(count):
            instances = []
            for i in range(count):
                instance = Mock()
                instance.label = [f"Label {i}"]
                instance.comment = [f"Comment {i}"]
                instances.append(instance)
            return instances
        
        mock_molecular_trait_class.instances.return_value = create_mock_instances(8)
        mock_plant_trait_class.instances.return_value = create_mock_instances(8)
        mock_human_trait_class.instances.return_value = create_mock_instances(8)
        
        # Act
        result = validate_initial_key_terms(mock_ontology)
        
        # Assert
        assert result['go_count'] == 8
        assert result['trait_ontology_count'] == 8
        assert result['chemfont_count'] == 8
        assert result['total_count'] == 24

    def test_validate_initial_key_terms_missing_classes(self, mock_ontology: Mock):
        """
        Test validation when required classes are missing.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_functional import validate_initial_key_terms
        
        # Mock search_one to return None (classes not found)
        mock_ontology.search_one.return_value = None
        
        # Act
        result = validate_initial_key_terms(mock_ontology)
        
        # Assert
        assert result['go_count'] == 0
        assert result['trait_ontology_count'] == 0
        assert result['chemfont_count'] == 0
        assert result['total_count'] == 0

    def test_define_core_functional_classes_success(self):
        """
        Test successful definition of core functional annotation classes.
        
        This integration test verifies that define_core_functional_classes() meets all
        requirements for AIM2-ODIE-011-T3 with a real (temporary) ontology:
        - Defines MolecularTrait, PlantTrait, HumanTrait classes
        - All classes inherit from owlready2.Thing
        - All classes are associated with main ontology namespace
        - Classes are programmatically defined as code
        - Proper semantic annotations are included
        """
        import tempfile
        from pathlib import Path
        from owlready2 import get_ontology, Thing
        from src.ontology.scheme_functional import define_core_functional_classes
        
        # Create a temporary ontology file
        with tempfile.NamedTemporaryFile(suffix='.owl', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create a test ontology
            ontology = get_ontology(f"file://{temp_path}")
            
            # Act - Call the function under test
            result = define_core_functional_classes(ontology)
            
            # Assert - Requirement 1: Define core functional annotation concepts
            required_classes = ['MolecularTrait', 'PlantTrait', 'HumanTrait']
            assert all(class_name in result for class_name in required_classes), \
                f"Missing required classes. Expected: {required_classes}, Got: {list(result.keys())}"
            
            # Assert - Requirement 2: Classes inherit from owlready2.Thing
            for class_name, cls in result.items():
                assert issubclass(cls, Thing), f"{class_name} does not inherit from Thing"
            
            # Assert - Requirement 3: Associated with main ontology namespace
            for class_name, cls in result.items():
                assert cls.namespace == ontology, f"{class_name} not associated with main ontology namespace"
            
            # Assert - Requirement 4: Proper semantic annotations
            molecular_trait = result['MolecularTrait']
            plant_trait = result['PlantTrait']
            human_trait = result['HumanTrait']
            
            # Verify labels exist and are appropriate
            assert hasattr(molecular_trait, 'label') and molecular_trait.label, "MolecularTrait missing label"
            assert hasattr(plant_trait, 'label') and plant_trait.label, "PlantTrait missing label"
            assert hasattr(human_trait, 'label') and human_trait.label, "HumanTrait missing label"
            
            assert "Molecular Function Trait" in molecular_trait.label
            assert "Plant Functional Trait" in plant_trait.label
            assert "Human Functional Trait" in human_trait.label
            
            # Verify comments exist and provide context
            assert hasattr(molecular_trait, 'comment') and molecular_trait.comment, "MolecularTrait missing comment"
            assert hasattr(plant_trait, 'comment') and plant_trait.comment, "PlantTrait missing comment"
            assert hasattr(human_trait, 'comment') and human_trait.comment, "HumanTrait missing comment"
            
            assert "GO" in molecular_trait.comment[0]
            assert "Trait Ontology" in plant_trait.comment[0]
            assert "ChemFont" in human_trait.comment[0]
            
            # Verify function returns exactly the expected number of classes
            assert len(result) == 3, f"Expected 3 classes, got {len(result)}"
            
        finally:
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)

    def test_define_core_functional_classes_invalid_ontology(self):
        """
        Test that define_core_functional_classes raises FunctionalClassError for invalid ontology.
        """
        from src.ontology.scheme_functional import define_core_functional_classes, FunctionalClassError
        
        # Act & Assert
        with expect_exception(FunctionalClassError, "Invalid ontology: cannot be None"):
            define_core_functional_classes(None)

    def test_define_core_functional_classes_with_owlready_error(self, mock_ontology: Mock):
        """
        Test that define_core_functional_classes handles OwlReadyError properly.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_functional import define_core_functional_classes, FunctionalClassError
        
        # Setup mock ontology context manager to raise OwlReadyError
        mock_ontology.__enter__ = Mock(side_effect=OwlReadyError("Test error"))
        mock_ontology.__exit__ = Mock(return_value=None)
        
        # Act & Assert
        with expect_exception(FunctionalClassError, "Owlready2 error defining core functional classes: Test error"):
            define_core_functional_classes(mock_ontology)