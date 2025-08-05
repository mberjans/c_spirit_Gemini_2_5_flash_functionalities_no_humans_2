"""
Unit tests for the ontology scheme source module.

This module contains comprehensive tests for creating and managing source
annotation classes in OWL 2.0 ontologies using Owlready2. Tests cover the
creation of Plant Ontology, NCBI Taxonomy, and PECO (Plant Experimental
Conditions Ontology) categories within target ontologies, including hierarchical 
relationships and validation.

Test Categories:
- PlantAnatomy class creation and validation
- Species class creation and hierarchical relationships
- ExperimentalCondition class creation and categorization
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


class TestSchemeSource:
    """Test suite for source annotation class creation and management."""

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
    def mock_plant_anatomy_class(self) -> Mock:
        """
        Create a mock PlantAnatomy class for testing.
        
        Returns:
            Mock: Mock PlantAnatomy object
        """
        mock_class = Mock()
        mock_class.name = "PlantAnatomy"
        mock_class.label = ["Plant Anatomical Entity"]
        mock_class.comment = ["Base class for plant anatomical structures based on Plant Ontology"]
        mock_class.is_a = [Thing]
        mock_class.equivalent_to = []
        mock_class.instances = Mock(return_value=[])
        return mock_class

    @pytest.fixture
    def mock_species_class(self) -> Mock:
        """
        Create a mock Species class for testing.
        
        Returns:
            Mock: Mock Species object
        """
        mock_class = Mock()
        mock_class.name = "Species"
        mock_class.label = ["Taxonomic Species"]
        mock_class.comment = ["Base class for taxonomic species classification based on NCBI Taxonomy"]
        mock_class.is_a = [Thing]
        mock_class.equivalent_to = []
        mock_class.instances = Mock(return_value=[])
        return mock_class

    @pytest.fixture
    def mock_experimental_condition_class(self) -> Mock:
        """
        Create a mock ExperimentalCondition class for testing.
        
        Returns:
            Mock: Mock ExperimentalCondition object
        """
        mock_class = Mock()
        mock_class.name = "ExperimentalCondition"
        mock_class.label = ["Plant Experimental Condition"]
        mock_class.comment = ["Base class for plant experimental conditions based on PECO"]
        mock_class.is_a = [Thing]
        mock_class.equivalent_to = []
        mock_class.instances = Mock(return_value=[])
        return mock_class

    @pytest.fixture
    def mock_organism_class(self) -> Mock:
        """
        Create a mock Organism parent class for hierarchical testing.
        
        Returns:
            Mock: Mock Organism class object
        """
        mock_class = Mock()
        mock_class.name = "Organism"
        mock_class.label = ["Biological Organism"]
        mock_class.comment = ["General class for biological organisms"]
        mock_class.is_a = [Thing]
        return mock_class

    def test_create_plant_anatomy_class_success(
        self, 
        mock_ontology: Mock, 
        mock_namespace: Mock,
        mock_plant_anatomy_class: Mock
    ):
        """
        Test successful creation of a PlantAnatomy class in the target ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_namespace: Mock namespace fixture
            mock_plant_anatomy_class: Mock PlantAnatomy fixture
        """
        from src.ontology.scheme_source import create_plant_anatomy_class
        
        # Mock the types() function to return our mock class
        with patch('owlready2.types.new_class') as mock_new_class:
            mock_new_class.return_value = mock_plant_anatomy_class
            
            # Act
            result = create_plant_anatomy_class(mock_ontology, "PlantAnatomy")
            
            # Assert
            assert result is not None
            assert result == mock_plant_anatomy_class
            assert result.name == "PlantAnatomy"
            assert "Plant Anatomical Entity" in result.label
            
            # Verify new_class was called with correct parameters
            mock_new_class.assert_called_once()
            args, kwargs = mock_new_class.call_args
            assert args[0] == "PlantAnatomy"  # Class name
            assert Thing in args[1]  # Inherits from Thing

    def test_create_species_class_success(
        self, 
        mock_ontology: Mock, 
        mock_namespace: Mock,
        mock_species_class: Mock
    ):
        """
        Test successful creation of a Species class in the target ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_namespace: Mock namespace fixture
            mock_species_class: Mock Species fixture
        """
        from src.ontology.scheme_source import create_species_class
        
        # Mock the types() function to return our mock class
        with patch('owlready2.types.new_class') as mock_new_class:
            mock_new_class.return_value = mock_species_class
            
            # Act
            result = create_species_class(mock_ontology, "Species")
            
            # Assert
            assert result is not None
            assert result == mock_species_class
            assert result.name == "Species"
            assert "Taxonomic Species" in result.label
            
            # Verify new_class was called with correct parameters
            mock_new_class.assert_called_once()
            args, kwargs = mock_new_class.call_args
            assert args[0] == "Species"  # Class name
            assert Thing in args[1]  # Inherits from Thing

    def test_create_experimental_condition_class_success(
        self, 
        mock_ontology: Mock, 
        mock_namespace: Mock,
        mock_experimental_condition_class: Mock
    ):
        """
        Test successful creation of an ExperimentalCondition class in the target ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_namespace: Mock namespace fixture
            mock_experimental_condition_class: Mock ExperimentalCondition fixture
        """
        from src.ontology.scheme_source import create_experimental_condition_class
        
        # Mock the types() function to return our mock class
        with patch('owlready2.types.new_class') as mock_new_class:
            mock_new_class.return_value = mock_experimental_condition_class
            
            # Act
            result = create_experimental_condition_class(mock_ontology, "ExperimentalCondition")
            
            # Assert
            assert result is not None
            assert result == mock_experimental_condition_class
            assert result.name == "ExperimentalCondition"
            assert "Plant Experimental Condition" in result.label
            
            # Verify new_class was called with correct parameters
            mock_new_class.assert_called_once()
            args, kwargs = mock_new_class.call_args
            assert args[0] == "ExperimentalCondition"  # Class name
            assert Thing in args[1]  # Inherits from Thing

    def test_create_hierarchical_species_relationship(
        self, 
        mock_ontology: Mock, 
        mock_species_class: Mock,
        mock_organism_class: Mock
    ):
        """
        Test creation of Species class with hierarchical relationship to Organism.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_species_class: Mock Species fixture
            mock_organism_class: Mock Organism fixture
        """
        from src.ontology.scheme_source import create_species_class_with_parent
        
        # Mock ontology search to find parent class
        mock_ontology.search_one.return_value = mock_organism_class
        
        with patch('owlready2.types.new_class') as mock_new_class:
            # Configure mock to set is_a relationship
            def configure_hierarchy(name, bases, namespace):
                mock_species_class.is_a = list(bases)
                return mock_species_class
            
            mock_new_class.side_effect = configure_hierarchy
            
            # Act
            result = create_species_class_with_parent(
                mock_ontology, 
                "Species", 
                parent_class_name="Organism"
            )
            
            # Assert
            assert result is not None
            assert result == mock_species_class
            assert mock_organism_class in result.is_a
            
            # Verify parent class was searched for
            mock_ontology.search_one.assert_called_once_with(iri="*Organism")

    def test_verify_class_accessibility_success(
        self, 
        mock_ontology: Mock,
        mock_plant_anatomy_class: Mock
    ):
        """
        Test verification that created classes are accessible in the ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_plant_anatomy_class: Mock PlantAnatomy fixture
        """
        from src.ontology.scheme_source import verify_class_accessibility
        
        # Mock ontology search to find the class
        mock_ontology.search_one.return_value = mock_plant_anatomy_class
        
        # Act
        result = verify_class_accessibility(mock_ontology, "PlantAnatomy")
        
        # Assert
        assert result is True
        mock_ontology.search_one.assert_called_once_with(iri="*PlantAnatomy")

    def test_verify_class_accessibility_failure(self, mock_ontology: Mock):
        """
        Test verification failure when class is not accessible in the ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_source import verify_class_accessibility
        
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
        mock_species_class: Mock,
        mock_organism_class: Mock
    ):
        """
        Test calculation of class hierarchy depth for source classes.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_species_class: Mock Species fixture
            mock_organism_class: Mock Organism fixture
        """
        from src.ontology.scheme_source import get_class_hierarchy_depth
        
        # Set up hierarchy: Species -> Organism -> Thing
        mock_species_class.is_a = [mock_organism_class]
        mock_organism_class.is_a = [Thing]
        
        mock_ontology.search_one.return_value = mock_species_class
        
        # Act
        depth = get_class_hierarchy_depth(mock_ontology, "Species")
        
        # Assert
        assert depth == 2  # Species is 2 levels below Thing

    @parametrize("class_name,expected_category", [
        ("PlantAnatomy", "plant_ontology_classification"),
        ("Species", "ncbi_taxonomy_classification"),  
        ("ExperimentalCondition", "peco_classification"),
        ("UnknownClass", "unknown_classification")
    ])
    def test_classify_source_annotation_type(
        self, 
        class_name: str, 
        expected_category: str,
        mock_ontology: Mock
    ):
        """
        Test classification of source annotation types based on class names.
        
        Args:
            class_name: Name of the class to classify
            expected_category: Expected classification category
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_source import classify_source_annotation_type
        
        # Mock class with appropriate name
        mock_class = Mock()
        mock_class.name = class_name
        mock_ontology.search_one.return_value = mock_class
        
        # Act
        category = classify_source_annotation_type(mock_ontology, class_name)
        
        # Assert
        assert category == expected_category

    def test_create_multiple_source_classes_batch(self, mock_ontology: Mock):
        """
        Test batch creation of multiple source annotation classes.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_source import create_source_classes_batch
        
        class_specs = [
            {"name": "PlantAnatomy", "type": "plant_ontology", "parent": None},
            {"name": "Species", "type": "ncbi_taxonomy", "parent": "Organism"},
            {"name": "ExperimentalCondition", "type": "peco", "parent": "Condition"}
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
            results = create_source_classes_batch(mock_ontology, class_specs)
            
            # Assert
            assert len(results) == 3
            assert all(result is not None for result in results)
            assert len(created_classes) == 3
            
            # Verify class names
            class_names = [cls.name for cls in created_classes]
            assert "PlantAnatomy" in class_names
            assert "Species" in class_names
            assert "ExperimentalCondition" in class_names

    def test_validate_source_class_properties(
        self, 
        mock_ontology: Mock,
        mock_plant_anatomy_class: Mock
    ):
        """
        Test validation of required properties for source annotation classes.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_plant_anatomy_class: Mock PlantAnatomy fixture
        """
        from src.ontology.scheme_source import validate_source_class_properties
        
        # Configure mock class with required properties
        mock_plant_anatomy_class.label = ["Plant Anatomical Entity"]
        mock_plant_anatomy_class.comment = ["Base class for plant anatomical structures"]
        mock_plant_anatomy_class.is_a = [Thing]
        
        mock_ontology.search_one.return_value = mock_plant_anatomy_class
        
        # Act
        is_valid = validate_source_class_properties(mock_ontology, "PlantAnatomy")
        
        # Assert
        assert is_valid is True

    def test_validate_source_class_properties_missing_label(
        self, 
        mock_ontology: Mock,
        mock_plant_anatomy_class: Mock
    ):
        """
        Test validation failure when source class is missing required label.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_plant_anatomy_class: Mock PlantAnatomy fixture
        """
        from src.ontology.scheme_source import validate_source_class_properties
        
        # Configure mock class with missing label
        mock_plant_anatomy_class.label = []  # Missing label
        mock_plant_anatomy_class.comment = ["Base class for plant anatomical structures"]
        mock_plant_anatomy_class.is_a = [Thing]
        
        mock_ontology.search_one.return_value = mock_plant_anatomy_class
        
        # Act
        is_valid = validate_source_class_properties(mock_ontology, "PlantAnatomy")
        
        # Assert
        assert is_valid is False

    def test_create_class_with_invalid_ontology(self):
        """
        Test error handling when trying to create class with invalid ontology.
        """
        from src.ontology.scheme_source import create_plant_anatomy_class, SourceClassError
        
        # Act & Assert
        with expect_exception(SourceClassError, match="Invalid ontology"):
            create_plant_anatomy_class(None, "PlantAnatomy")

    def test_create_class_with_empty_name(self, mock_ontology: Mock):
        """
        Test error handling when trying to create class with empty name.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_source import create_plant_anatomy_class, SourceClassError
        
        # Act & Assert
        with expect_exception(SourceClassError, match="Invalid class name"):
            create_plant_anatomy_class(mock_ontology, "")

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
        from src.ontology.scheme_source import create_species_class, SourceClassError
        
        # Act & Assert
        with expect_exception(SourceClassError, match="Invalid class name"):
            create_species_class(mock_ontology, invalid_name)

    def test_create_class_owlready_error_handling(self, mock_ontology: Mock):
        """
        Test error handling when Owlready2 operations fail.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_source import create_experimental_condition_class, SourceClassError
        
        with patch('owlready2.types.new_class') as mock_new_class:
            # Mock Owlready2 error
            mock_new_class.side_effect = OwlReadyError("Owlready2 operation failed")
            
            # Act & Assert
            with expect_exception(SourceClassError, match="Owlready2 error"):
                create_experimental_condition_class(mock_ontology, "ExperimentalCondition")

    def test_verify_thing_inheritance(
        self, 
        mock_ontology: Mock,
        mock_plant_anatomy_class: Mock
    ):
        """
        Test verification that source classes properly inherit from Thing.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_plant_anatomy_class: Mock PlantAnatomy fixture
        """
        from src.ontology.scheme_source import verify_thing_inheritance
        
        # Configure class to inherit from Thing
        mock_plant_anatomy_class.is_a = [Thing]
        mock_ontology.search_one.return_value = mock_plant_anatomy_class
        
        # Act
        inherits_from_thing = verify_thing_inheritance(mock_ontology, "PlantAnatomy")
        
        # Assert
        assert inherits_from_thing is True

    def test_verify_thing_inheritance_failure(
        self, 
        mock_ontology: Mock,
        mock_plant_anatomy_class: Mock
    ):
        """
        Test verification failure when class doesn't inherit from Thing.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_plant_anatomy_class: Mock PlantAnatomy fixture
        """
        from src.ontology.scheme_source import verify_thing_inheritance
        
        # Configure class to not inherit from Thing
        mock_other_class = Mock()
        mock_plant_anatomy_class.is_a = [mock_other_class]  # Doesn't inherit from Thing
        mock_ontology.search_one.return_value = mock_plant_anatomy_class
        
        # Act
        inherits_from_thing = verify_thing_inheritance(mock_ontology, "PlantAnatomy")
        
        # Assert
        assert inherits_from_thing is False

    def test_namespace_integration(
        self, 
        mock_ontology: Mock, 
        mock_namespace: Mock,
        mock_plant_anatomy_class: Mock
    ):
        """
        Test that classes are properly integrated with ontology namespace.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_namespace: Mock namespace fixture
            mock_plant_anatomy_class: Mock PlantAnatomy fixture
        """
        from src.ontology.scheme_source import create_plant_anatomy_class
        
        with patch('owlready2.types.new_class') as mock_new_class:
            mock_new_class.return_value = mock_plant_anatomy_class
            
            # Act
            result = create_plant_anatomy_class(mock_ontology, "PlantAnatomy")
            
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
        Test thread safety when creating multiple source classes concurrently.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        import threading
        from src.ontology.scheme_source import create_plant_anatomy_class
        
        results = []
        errors = []
        
        def create_class_worker(class_name: str):
            try:
                with patch('owlready2.types.new_class') as mock_new_class:
                    mock_class = Mock()
                    mock_class.name = class_name
                    mock_new_class.return_value = mock_class
                    
                    result = create_plant_anatomy_class(mock_ontology, class_name)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        class_names = ["PlantAnatomy1", "PlantAnatomy2", "PlantAnatomy3"]
        
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

    def test_source_class_error_custom_exception(self):
        """
        Test that custom SourceClassError exception works correctly.
        """
        from src.ontology.scheme_source import SourceClassError
        
        # Test basic exception creation
        error_msg = "Test source class error"
        exception = SourceClassError(error_msg)
        
        assert str(exception) == error_msg
        assert isinstance(exception, Exception)

    def test_source_class_error_with_cause(self):
        """
        Test that SourceClassError properly handles exception chaining.
        """
        from src.ontology.scheme_source import SourceClassError
        
        # Test exception chaining
        original_error = ValueError("Original error")
        try:
            raise SourceClassError("Wrapped source error") from original_error
        except SourceClassError as chained_error:
            assert str(chained_error) == "Wrapped source error"
            assert chained_error.__cause__ == original_error

    def test_get_all_source_classes(self, mock_ontology: Mock):
        """
        Test retrieval of all source annotation classes from ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_source import get_all_source_classes
        
        # Mock source classes
        mock_plant_anatomy = Mock()
        mock_plant_anatomy.name = "PlantAnatomy"
        mock_species = Mock()
        mock_species.name = "Species"
        mock_experimental_condition = Mock()
        mock_experimental_condition.name = "ExperimentalCondition"
        
        # Mock ontology search to return source classes
        mock_ontology.classes.return_value = [mock_plant_anatomy, mock_species, mock_experimental_condition]
        
        # Act
        source_classes = get_all_source_classes(mock_ontology)
        
        # Assert
        assert len(source_classes) == 3
        class_names = [cls.name for cls in source_classes]
        assert "PlantAnatomy" in class_names
        assert "Species" in class_names
        assert "ExperimentalCondition" in class_names

    def test_source_class_metadata_validation(
        self, 
        mock_ontology: Mock,
        mock_plant_anatomy_class: Mock
    ):
        """
        Test validation of source class metadata and annotations.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_plant_anatomy_class: Mock PlantAnatomy fixture
        """
        from src.ontology.scheme_source import validate_class_metadata
        
        # Configure class with complete metadata
        mock_plant_anatomy_class.label = ["Plant Anatomical Entity"]
        mock_plant_anatomy_class.comment = ["Base class for plant anatomical structures based on Plant Ontology"]
        mock_plant_anatomy_class.iri = "http://test.example.org/ontology#PlantAnatomy"
        
        # Mock custom annotations
        mock_plant_anatomy_class.classification_system = ["Plant Ontology"]
        mock_plant_anatomy_class.version = ["1.0"]
        
        mock_ontology.search_one.return_value = mock_plant_anatomy_class
        
        # Act
        metadata_valid = validate_class_metadata(mock_ontology, "PlantAnatomy")
        
        # Assert
        assert metadata_valid is True

    def test_cleanup_source_classes(self, mock_ontology: Mock):
        """
        Test cleanup of source annotation classes from ontology.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_source import cleanup_source_classes
        
        # Mock source classes to be cleaned up
        mock_classes = [Mock(), Mock(), Mock()]
        for i, mock_class in enumerate(mock_classes):
            mock_class.name = f"SourceClass{i}"
            mock_class.destroy = Mock()
        
        mock_ontology.classes.return_value = mock_classes
        
        # Act
        cleanup_count = cleanup_source_classes(mock_ontology)
        
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
        from src.ontology.scheme_source import create_plant_anatomy_class
        
        with patch('owlready2.types.new_class') as mock_new_class:
            # Verify Thing is imported and used correctly
            mock_class = Mock()
            mock_class.is_a = [Thing]
            mock_new_class.return_value = mock_class
            
            # Act
            result = create_plant_anatomy_class(mock_ontology, "PlantAnatomy")
            
            # Assert
            assert result is not None
            assert Thing in result.is_a
            
            # Verify new_class was called with Thing as base
            args, kwargs = mock_new_class.call_args
            assert Thing in args[1]  # Base classes tuple

    def test_add_initial_key_terms_success(self, mock_ontology: Mock):
        """
        Test successful addition of initial key terms from all source systems.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_source import add_initial_key_terms
        
        # Mock the required classes
        mock_plant_anatomy_class = Mock()
        mock_species_class = Mock()
        mock_experimental_condition_class = Mock()
        
        # Mock search_one to return the required classes
        def search_side_effect(iri):
            if "PlantAnatomy" in iri:
                return mock_plant_anatomy_class
            elif "Species" in iri:
                return mock_species_class
            elif "ExperimentalCondition" in iri:
                return mock_experimental_condition_class
            return None
        
        mock_ontology.search_one.side_effect = search_side_effect
        
        # Mock the ontology context manager
        mock_ontology.__enter__ = Mock(return_value=mock_ontology)
        mock_ontology.__exit__ = Mock(return_value=None)
        
        # Mock instance creation
        mock_plant_anatomy_instances = []
        mock_species_instances = []
        mock_peco_instances = []
        
        def create_plant_anatomy_instance(name):
            instance = Mock()
            instance.name = name
            instance.label = []
            instance.comment = []
            mock_plant_anatomy_instances.append(instance)
            return instance
        
        def create_species_instance(name):
            instance = Mock()
            instance.name = name
            instance.label = []
            instance.comment = []
            mock_species_instances.append(instance)
            return instance
        
        def create_peco_instance(name):
            instance = Mock()
            instance.name = name
            instance.label = []
            instance.comment = []
            mock_peco_instances.append(instance)
            return instance
        
        mock_plant_anatomy_class.side_effect = create_plant_anatomy_instance
        mock_species_class.side_effect = create_species_instance
        mock_experimental_condition_class.side_effect = create_peco_instance
        
        # Act
        result = add_initial_key_terms(mock_ontology)
        
        # Assert
        assert result is not None
        assert 'plant_anatomy_instances' in result
        assert 'species_instances' in result
        assert 'peco_instances' in result
        
        # Verify expected number of instances were created
        assert len(result['plant_anatomy_instances']) == 8  # Expected Plant Ontology instances
        assert len(result['species_instances']) == 8        # Expected NCBI Taxonomy instances
        assert len(result['peco_instances']) == 8           # Expected PECO instances
        
        # Verify instance properties were set
        for instance in result['plant_anatomy_instances']:
            assert instance.label is not None
            assert instance.comment is not None
        
        for instance in result['species_instances']:
            assert instance.label is not None
            assert instance.comment is not None
        
        for instance in result['peco_instances']:
            assert instance.label is not None
            assert instance.comment is not None

    def test_add_initial_key_terms_missing_classes(self, mock_ontology: Mock):
        """
        Test add_initial_key_terms when required classes are missing.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_source import add_initial_key_terms, SourceClassError
        
        # Mock search_one to return None (classes not found)
        mock_ontology.search_one.return_value = None
        
        # Act & Assert
        with expect_exception(SourceClassError, "Required source classes not found"):
            add_initial_key_terms(mock_ontology)

    def test_add_initial_key_terms_specific_instances(self, mock_ontology: Mock):
        """
        Test creation of specific representative instances.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_source import add_initial_key_terms
        
        # Mock the required classes
        mock_plant_anatomy_class = Mock()
        mock_species_class = Mock()
        mock_experimental_condition_class = Mock()
        
        def search_side_effect(iri):
            if "PlantAnatomy" in iri:
                return mock_plant_anatomy_class
            elif "Species" in iri:
                return mock_species_class
            elif "ExperimentalCondition" in iri:
                return mock_experimental_condition_class
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
        
        mock_plant_anatomy_class.side_effect = create_instance
        mock_species_class.side_effect = create_instance
        mock_experimental_condition_class.side_effect = create_instance
        
        # Act
        result = add_initial_key_terms(mock_ontology)
        
        # Assert specific representative instances were created
        instance_names = [inst.name for inst in created_instances]
        
        # Check for expected Plant Ontology instances
        assert "Root" in instance_names
        assert "Leaf" in instance_names
        assert "Stem" in instance_names
        assert "Flower" in instance_names
        
        # Check for expected NCBI Taxonomy instances
        assert "Arabidopsis_thaliana" in instance_names
        assert "Oryza_sativa" in instance_names
        assert "Zea_mays" in instance_names
        assert "Solanum_lycopersicum" in instance_names
        
        # Check for expected PECO instances
        assert "Drought_stress" in instance_names
        assert "Salt_stress" in instance_names
        assert "Heat_stress" in instance_names
        assert "Cold_stress" in instance_names

    def test_add_initial_key_terms_with_owlready_error(self, mock_ontology: Mock):
        """
        Test add_initial_key_terms handling of Owlready2 errors.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_source import add_initial_key_terms, SourceClassError
        
        # Mock search_one to raise OwlReadyError
        mock_ontology.search_one.side_effect = OwlReadyError("Owlready2 error")
        
        # Act & Assert
        with expect_exception(SourceClassError, "Owlready2 error creating initial key terms"):
            add_initial_key_terms(mock_ontology)

    def test_validate_initial_key_terms_success(self, mock_ontology: Mock):
        """
        Test successful validation of initial key terms.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_source import validate_initial_key_terms
        
        # Mock the required classes
        mock_plant_anatomy_class = Mock()
        mock_species_class = Mock()
        mock_experimental_condition_class = Mock()
        
        def search_side_effect(iri):
            if "PlantAnatomy" in iri:
                return mock_plant_anatomy_class
            elif "Species" in iri:
                return mock_species_class
            elif "ExperimentalCondition" in iri:
                return mock_experimental_condition_class
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
        
        mock_plant_anatomy_class.instances.return_value = create_mock_instances(8)
        mock_species_class.instances.return_value = create_mock_instances(8)
        mock_experimental_condition_class.instances.return_value = create_mock_instances(8)
        
        # Act
        result = validate_initial_key_terms(mock_ontology)
        
        # Assert
        assert result['plant_anatomy_count'] == 8
        assert result['species_count'] == 8
        assert result['peco_count'] == 8
        assert result['total_count'] == 24

    def test_validate_initial_key_terms_missing_classes(self, mock_ontology: Mock):
        """
        Test validation when required classes are missing.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_source import validate_initial_key_terms
        
        # Mock search_one to return None (classes not found)
        mock_ontology.search_one.return_value = None
        
        # Act
        result = validate_initial_key_terms(mock_ontology)
        
        # Assert
        assert result['plant_anatomy_count'] == 0
        assert result['species_count'] == 0
        assert result['peco_count'] == 0
        assert result['total_count'] == 0

    def test_validate_initial_key_terms_invalid_instances(self, mock_ontology: Mock):
        """
        Test validation with instances missing required properties.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_source import validate_initial_key_terms
        
        # Mock the required classes
        mock_plant_anatomy_class = Mock()
        mock_species_class = Mock()
        mock_experimental_condition_class = Mock()
        
        def search_side_effect(iri):
            if "PlantAnatomy" in iri:
                return mock_plant_anatomy_class
            elif "Species" in iri:
                return mock_species_class
            elif "ExperimentalCondition" in iri:
                return mock_experimental_condition_class
            return None
        
        mock_ontology.search_one.side_effect = search_side_effect
        
        # Mock instances with missing or empty labels/comments
        def create_invalid_instances(count):
            instances = []
            for i in range(count):
                instance = Mock()
                # Some instances missing label, some missing comment, some empty
                if i % 3 == 0:
                    del instance.label  # Missing label attribute
                    instance.comment = [f"Comment {i}"]
                elif i % 3 == 1:
                    instance.label = [f"Label {i}"]
                    del instance.comment  # Missing comment attribute
                else:
                    instance.label = []  # Empty label
                    instance.comment = []  # Empty comment
                instances.append(instance)
            return instances
        
        mock_plant_anatomy_class.instances.return_value = create_invalid_instances(6)
        mock_species_class.instances.return_value = create_invalid_instances(6)
        mock_experimental_condition_class.instances.return_value = create_invalid_instances(6)
        
        # Act
        result = validate_initial_key_terms(mock_ontology)
        
        # Assert - no instances should be counted as valid
        assert result['plant_anatomy_count'] == 0
        assert result['species_count'] == 0
        assert result['peco_count'] == 0
        assert result['total_count'] == 0

    def test_hierarchical_root_subclass_of_plant_anatomy(
        self, 
        mock_ontology: Mock, 
        mock_plant_anatomy_class: Mock
    ):
        """
        Test that Root is created as a subclass of PlantAnatomy.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_plant_anatomy_class: Mock PlantAnatomy fixture
        """
        from src.ontology.scheme_source import create_root_class_with_parent
        
        # Mock ontology search to find parent class
        mock_ontology.search_one.return_value = mock_plant_anatomy_class
        
        with patch('owlready2.types.new_class') as mock_new_class:
            # Create mock Root class
            mock_root_class = Mock()
            mock_root_class.name = "Root"
            mock_root_class.label = ["Plant Root"]
            mock_root_class.comment = ["Plant root anatomical structure"]
            
            # Configure mock to set is_a relationship
            def configure_hierarchy(name, bases, namespace):
                mock_root_class.is_a = list(bases)
                return mock_root_class
            
            mock_new_class.side_effect = configure_hierarchy
            
            # Act
            result = create_root_class_with_parent(
                mock_ontology, 
                "Root", 
                parent_class_name="PlantAnatomy"
            )
            
            # Assert
            assert result is not None
            assert result == mock_root_class
            assert mock_plant_anatomy_class in result.is_a
            
            # Verify parent class was searched for
            mock_ontology.search_one.assert_called_once_with(iri="*PlantAnatomy")

    def test_peco_stress_condition_hierarchy(
        self, 
        mock_ontology: Mock, 
        mock_experimental_condition_class: Mock
    ):
        """
        Test creation of stress condition as subclass of ExperimentalCondition.
        
        Args:
            mock_ontology: Mock ontology fixture
            mock_experimental_condition_class: Mock ExperimentalCondition fixture
        """
        from src.ontology.scheme_source import create_stress_condition_with_parent
        
        # Mock ontology search to find parent class
        mock_ontology.search_one.return_value = mock_experimental_condition_class
        
        with patch('owlready2.types.new_class') as mock_new_class:
            # Create mock StressCondition class
            mock_stress_class = Mock()
            mock_stress_class.name = "StressCondition"
            mock_stress_class.label = ["Plant Stress Condition"]
            mock_stress_class.comment = ["Experimental condition involving plant stress"]
            
            # Configure mock to set is_a relationship
            def configure_hierarchy(name, bases, namespace):
                mock_stress_class.is_a = list(bases)
                return mock_stress_class
            
            mock_new_class.side_effect = configure_hierarchy
            
            # Act
            result = create_stress_condition_with_parent(
                mock_ontology, 
                "StressCondition", 
                parent_class_name="ExperimentalCondition"
            )
            
            # Assert
            assert result is not None
            assert result == mock_stress_class
            assert mock_experimental_condition_class in result.is_a
            
            # Verify parent class was searched for
            mock_ontology.search_one.assert_called_once_with(iri="*ExperimentalCondition")

    def test_define_core_source_classes_success(self):
        """
        Test successful definition of core source annotation classes.
        
        This integration test verifies that define_core_source_classes() meets all
        requirements for AIM2-ODIE-010-T3 with a real (temporary) ontology:
        - Defines PlantAnatomy, Species, ExperimentalCondition classes
        - All classes inherit from owlready2.Thing
        - All classes are associated with main ontology namespace
        - Classes are programmatically defined as code
        - Proper semantic annotations are included
        """
        import tempfile
        from pathlib import Path
        from owlready2 import get_ontology, Thing
        from src.ontology.scheme_source import define_core_source_classes
        
        # Create a temporary ontology file
        with tempfile.NamedTemporaryFile(suffix='.owl', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create a test ontology
            ontology = get_ontology(f"file://{temp_path}")
            
            # Act - Call the function under test
            result = define_core_source_classes(ontology)
            
            # Assert - Requirement 1: Define core source annotation concepts
            required_classes = ['PlantAnatomy', 'Species', 'ExperimentalCondition']
            assert all(class_name in result for class_name in required_classes), \
                f"Missing required classes. Expected: {required_classes}, Got: {list(result.keys())}"
            
            # Assert - Requirement 2: Classes inherit from owlready2.Thing
            for class_name, cls in result.items():
                assert issubclass(cls, Thing), f"{class_name} does not inherit from Thing"
            
            # Assert - Requirement 3: Associated with main ontology namespace
            for class_name, cls in result.items():
                assert cls.namespace == ontology, f"{class_name} not associated with main ontology namespace"
            
            # Assert - Requirement 4: Proper semantic annotations
            plant_anatomy = result['PlantAnatomy']
            species = result['Species']
            experimental_condition = result['ExperimentalCondition']
            
            # Verify labels exist and are appropriate
            assert hasattr(plant_anatomy, 'label') and plant_anatomy.label, "PlantAnatomy missing label"
            assert hasattr(species, 'label') and species.label, "Species missing label"
            assert hasattr(experimental_condition, 'label') and experimental_condition.label, "ExperimentalCondition missing label"
            
            assert "Plant Anatomical Entity" in plant_anatomy.label
            assert "Taxonomic Species" in species.label
            assert "Plant Experimental Condition" in experimental_condition.label
            
            # Verify comments exist and provide context
            assert hasattr(plant_anatomy, 'comment') and plant_anatomy.comment, "PlantAnatomy missing comment"
            assert hasattr(species, 'comment') and species.comment, "Species missing comment"
            assert hasattr(experimental_condition, 'comment') and experimental_condition.comment, "ExperimentalCondition missing comment"
            
            assert "Plant Ontology" in plant_anatomy.comment[0]
            assert "NCBI Taxonomy" in species.comment[0]
            assert "PECO" in experimental_condition.comment[0]
            
            # Verify function returns exactly the expected number of classes
            assert len(result) == 3, f"Expected 3 classes, got {len(result)}"
            
        finally:
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)

    def test_define_core_source_classes_invalid_ontology(self):
        """
        Test that define_core_source_classes raises SourceClassError for invalid ontology.
        """
        from src.ontology.scheme_source import define_core_source_classes, SourceClassError
        
        # Act & Assert
        with expect_exception(SourceClassError, "Invalid ontology: cannot be None"):
            define_core_source_classes(None)

    def test_define_core_source_classes_with_owlready_error(self, mock_ontology: Mock):
        """
        Test that define_core_source_classes handles OwlReadyError properly.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.scheme_source import define_core_source_classes, SourceClassError
        
        # Setup mock ontology context manager to raise OwlReadyError
        mock_ontology.__enter__ = Mock(side_effect=OwlReadyError("Test error"))
        mock_ontology.__exit__ = Mock(return_value=None)
        
        # Act & Assert
        with expect_exception(SourceClassError, "Owlready2 error defining core source classes: Test error"):
            define_core_source_classes(mock_ontology)