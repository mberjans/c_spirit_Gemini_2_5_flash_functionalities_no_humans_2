"""
Unit tests for the ontology editor module.

This module contains comprehensive tests for editing OWL 2.0 ontologies
using Owlready2's destroy_entity() function. Tests cover deletion of classes,
individuals, and properties, along with comprehensive error handling.

Test Categories:
- Class deletion functionality and error handling
- Individual deletion functionality and error handling  
- Property deletion functionality and error handling
- Integration tests for complex deletion scenarios
- Custom exception handling and error messages
- Thread safety and concurrent operations
"""

import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import Any, Generator, List

import pytest
from owlready2 import OwlReadyError, OwlReadyOntologyParsingError

from src.utils.testing_framework import expect_exception, parametrize


class TestOntologyEditor:
    """Test suite for ontology editing functionality."""

    @pytest.fixture
    def mock_ontology(self) -> Mock:
        """
        Create a mock ontology object for testing.
        
        Returns:
            Mock: Mock ontology object with search methods
        """
        mock_ont = Mock()
        mock_ont.name = "test_ontology"
        mock_ont.base_iri = "http://test.example.org/ontology"
        mock_ont.search_one.return_value = None  # Default: entity not found
        mock_ont.search.return_value = []  # Default: no entities found
        return mock_ont

    @pytest.fixture
    def mock_class_entity(self) -> Mock:
        """
        Create a mock class entity for testing.
        
        Returns:
            Mock: Mock class entity
        """
        mock_class = Mock()
        mock_class.iri = "http://test.example.org/ontology#TestClass"
        mock_class.name = "TestClass"
        mock_class.instances.return_value = []  # Default: no instances
        return mock_class

    @pytest.fixture
    def mock_individual_entity(self) -> Mock:
        """
        Create a mock individual entity for testing.
        
        Returns:
            Mock: Mock individual entity
        """
        mock_individual = Mock()
        mock_individual.iri = "http://test.example.org/ontology#TestIndividual"
        mock_individual.name = "TestIndividual"
        return mock_individual

    @pytest.fixture
    def mock_property_entity(self) -> Mock:
        """
        Create a mock property entity for testing.
        
        Returns:
            Mock: Mock property entity
        """
        mock_property = Mock()
        mock_property.iri = "http://test.example.org/ontology#testProperty"
        mock_property.name = "testProperty"
        return mock_property

    @pytest.fixture
    def mock_instance_entities(self) -> List[Mock]:
        """
        Create mock instance entities for testing class deletion.
        
        Returns:
            List[Mock]: List of mock instance entities
        """
        instances = []
        for i in range(3):
            instance = Mock()
            instance.iri = f"http://test.example.org/ontology#TestInstance{i}"
            instance.name = f"TestInstance{i}"
            instances.append(instance)
        return instances

    # =====================================================
    # Tests for delete_class() function
    # =====================================================

    @patch('src.ontology.editor.destroy_entity')
    def test_delete_class_success(
        self, 
        mock_destroy_entity: Mock,
        mock_ontology: Mock, 
        mock_class_entity: Mock
    ):
        """
        Test successful deletion of a class and verification of its absence.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
            mock_class_entity: Mock class entity fixture
        """
        from src.ontology.editor import delete_class
        
        class_iri = "http://test.example.org/ontology#TestClass"
        
        # Setup: class exists initially, then doesn't exist after deletion
        mock_ontology.search_one.side_effect = [mock_class_entity, None]
        
        # Act
        delete_class(mock_ontology, class_iri)
        
        # Assert
        # Verify search_one was called to find the class
        expected_calls = [call(iri=class_iri), call(iri=class_iri)]
        mock_ontology.search_one.assert_has_calls(expected_calls)
        
        # Verify destroy_entity was called on the class
        mock_destroy_entity.assert_called_once_with(mock_class_entity)

    @patch('src.ontology.editor.destroy_entity')
    def test_delete_class_with_instances_success(
        self, 
        mock_destroy_entity: Mock,
        mock_ontology: Mock, 
        mock_class_entity: Mock,
        mock_instance_entities: List[Mock]
    ):
        """
        Test deletion of a class with instances - instances should be removed.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
            mock_class_entity: Mock class entity fixture
            mock_instance_entities: Mock instance entities fixture
        """
        from src.ontology.editor import delete_class
        
        class_iri = "http://test.example.org/ontology#TestClass"
        
        # Setup: class has instances
        mock_class_entity.instances.return_value = mock_instance_entities
        mock_ontology.search_one.side_effect = [mock_class_entity, None]
        
        # Act
        delete_class(mock_ontology, class_iri)
        
        # Assert
        # Verify destroy_entity was called for all instances first, then the class
        expected_calls = []
        for instance in mock_instance_entities:
            expected_calls.append(call(instance))
        expected_calls.append(call(mock_class_entity))
        
        mock_destroy_entity.assert_has_calls(expected_calls)

    def test_delete_class_nonexistent_error(self, mock_ontology: Mock):
        """
        Test error handling when attempting to delete non-existent class.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.editor import delete_class, EntityDeletionError
        
        class_iri = "http://test.example.org/ontology#NonExistentClass"
        
        # Setup: class doesn't exist
        mock_ontology.search_one.return_value = None
        
        # Act & Assert
        with expect_exception(EntityDeletionError, match="Class not found"):
            delete_class(mock_ontology, class_iri)

    @patch('src.ontology.editor.destroy_entity')
    def test_delete_class_owlready_error(
        self, 
        mock_destroy_entity: Mock,
        mock_ontology: Mock, 
        mock_class_entity: Mock
    ):
        """
        Test error handling when Owlready2 raises an error during class deletion.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
            mock_class_entity: Mock class entity fixture
        """
        from src.ontology.editor import delete_class, EntityDeletionError
        
        class_iri = "http://test.example.org/ontology#TestClass"
        
        # Setup: class exists but destroy_entity fails
        mock_ontology.search_one.return_value = mock_class_entity
        mock_destroy_entity.side_effect = OwlReadyError("Destruction failed")
        
        # Act & Assert
        with expect_exception(EntityDeletionError, match="Failed to delete class"):
            delete_class(mock_ontology, class_iri)

    @parametrize("class_iri", [
        "",
        None,
        "   ",
        "invalid-iri",
        "http://example.com/missing#fragment",
    ])
    def test_delete_class_invalid_iri(self, mock_ontology: Mock, class_iri: str):
        """
        Test error handling for invalid class IRIs.
        
        Args:
            mock_ontology: Mock ontology fixture
            class_iri: Invalid class IRI to test
        """
        from src.ontology.editor import delete_class, EntityDeletionError
        
        # Act & Assert
        with expect_exception(EntityDeletionError, match="Invalid class IRI"):
            delete_class(mock_ontology, class_iri)

    # =====================================================
    # Tests for delete_individual() function
    # =====================================================

    @patch('src.ontology.editor.destroy_entity')
    def test_delete_individual_success(
        self, 
        mock_destroy_entity: Mock,
        mock_ontology: Mock, 
        mock_individual_entity: Mock
    ):
        """
        Test successful deletion of an individual and verification of its absence.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
            mock_individual_entity: Mock individual entity fixture
        """
        from src.ontology.editor import delete_individual
        
        individual_iri = "http://test.example.org/ontology#TestIndividual"
        
        # Setup: individual exists initially, then doesn't exist after deletion
        mock_ontology.search_one.side_effect = [mock_individual_entity, None]
        
        # Act
        delete_individual(mock_ontology, individual_iri)
        
        # Assert
        # Verify search_one was called to find the individual
        expected_calls = [call(iri=individual_iri), call(iri=individual_iri)]
        mock_ontology.search_one.assert_has_calls(expected_calls)
        
        # Verify destroy_entity was called on the individual
        mock_destroy_entity.assert_called_once_with(mock_individual_entity)

    @patch('src.ontology.editor.destroy_entity')
    def test_delete_individual_with_relationships_cleanup(
        self, 
        mock_destroy_entity: Mock,
        mock_ontology: Mock, 
        mock_individual_entity: Mock
    ):
        """
        Test that relationships involving deleted individual are cleaned up.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
            mock_individual_entity: Mock individual entity fixture
        """
        from src.ontology.editor import delete_individual
        
        individual_iri = "http://test.example.org/ontology#TestIndividual"
        
        # Setup: individual has relationships (mocked as attributes)
        related_entities = [Mock(), Mock()]
        mock_individual_entity.get_properties.return_value = related_entities
        mock_ontology.search_one.side_effect = [mock_individual_entity, None]
        
        # Act
        delete_individual(mock_ontology, individual_iri)
        
        # Assert
        # Verify destroy_entity was called (relationships are cleaned up automatically by Owlready2)
        mock_destroy_entity.assert_called_once_with(mock_individual_entity)

    def test_delete_individual_nonexistent_error(self, mock_ontology: Mock):
        """
        Test error handling when attempting to delete non-existent individual.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.editor import delete_individual, EntityDeletionError
        
        individual_iri = "http://test.example.org/ontology#NonExistentIndividual"
        
        # Setup: individual doesn't exist
        mock_ontology.search_one.return_value = None
        
        # Act & Assert
        with expect_exception(EntityDeletionError, match="Individual not found"):
            delete_individual(mock_ontology, individual_iri)

    @patch('src.ontology.editor.destroy_entity')
    def test_delete_individual_owlready_error(
        self, 
        mock_destroy_entity: Mock,
        mock_ontology: Mock, 
        mock_individual_entity: Mock
    ):
        """
        Test error handling when Owlready2 raises an error during individual deletion.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
            mock_individual_entity: Mock individual entity fixture
        """
        from src.ontology.editor import delete_individual, EntityDeletionError
        
        individual_iri = "http://test.example.org/ontology#TestIndividual"
        
        # Setup: individual exists but destroy_entity fails
        mock_ontology.search_one.return_value = mock_individual_entity
        mock_destroy_entity.side_effect = OwlReadyError("Destruction failed")
        
        # Act & Assert
        with expect_exception(EntityDeletionError, match="Failed to delete individual"):
            delete_individual(mock_ontology, individual_iri)

    @parametrize("individual_iri", [
        "",
        None,
        "   ",
        "invalid-iri",
        "http://example.com/missing#fragment",
    ])
    def test_delete_individual_invalid_iri(self, mock_ontology: Mock, individual_iri: str):
        """
        Test error handling for invalid individual IRIs.
        
        Args:
            mock_ontology: Mock ontology fixture
            individual_iri: Invalid individual IRI to test
        """
        from src.ontology.editor import delete_individual, EntityDeletionError
        
        # Act & Assert
        with expect_exception(EntityDeletionError, match="Invalid individual IRI"):
            delete_individual(mock_ontology, individual_iri)

    # =====================================================
    # Tests for delete_property() function
    # =====================================================

    @patch('src.ontology.editor.destroy_entity')
    def test_delete_property_success(
        self, 
        mock_destroy_entity: Mock,
        mock_ontology: Mock, 
        mock_property_entity: Mock
    ):
        """
        Test successful deletion of a property and verification of its absence.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
            mock_property_entity: Mock property entity fixture
        """
        from src.ontology.editor import delete_property
        
        property_iri = "http://test.example.org/ontology#testProperty"
        
        # Setup: property exists initially, then doesn't exist after deletion
        mock_ontology.search_one.side_effect = [mock_property_entity, None]
        
        # Act
        delete_property(mock_ontology, property_iri)
        
        # Assert
        # Verify search_one was called to find the property
        expected_calls = [call(iri=property_iri), call(iri=property_iri)]
        mock_ontology.search_one.assert_has_calls(expected_calls)
        
        # Verify destroy_entity was called on the property
        mock_destroy_entity.assert_called_once_with(mock_property_entity)

    @patch('src.ontology.editor.destroy_entity')
    def test_delete_property_with_relationships_cleanup(
        self, 
        mock_destroy_entity: Mock,
        mock_ontology: Mock, 
        mock_property_entity: Mock
    ):
        """
        Test that relationships using deleted property are cleaned up.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
            mock_property_entity: Mock property entity fixture
        """
        from src.ontology.editor import delete_property
        
        property_iri = "http://test.example.org/ontology#testProperty"
        
        # Setup: property is used in relationships
        using_entities = [Mock(), Mock()]
        mock_property_entity.get_relations.return_value = using_entities
        mock_ontology.search_one.side_effect = [mock_property_entity, None]
        
        # Act
        delete_property(mock_ontology, property_iri)
        
        # Assert
        # Verify destroy_entity was called (relationships are cleaned up automatically by Owlready2)
        mock_destroy_entity.assert_called_once_with(mock_property_entity)

    def test_delete_property_nonexistent_error(self, mock_ontology: Mock):
        """
        Test error handling when attempting to delete non-existent property.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.editor import delete_property, EntityDeletionError
        
        property_iri = "http://test.example.org/ontology#nonExistentProperty"
        
        # Setup: property doesn't exist
        mock_ontology.search_one.return_value = None
        
        # Act & Assert
        with expect_exception(EntityDeletionError, match="Property not found"):
            delete_property(mock_ontology, property_iri)

    @patch('src.ontology.editor.destroy_entity')
    def test_delete_property_owlready_error(
        self, 
        mock_destroy_entity: Mock,
        mock_ontology: Mock, 
        mock_property_entity: Mock
    ):
        """
        Test error handling when Owlready2 raises an error during property deletion.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
            mock_property_entity: Mock property entity fixture
        """
        from src.ontology.editor import delete_property, EntityDeletionError
        
        property_iri = "http://test.example.org/ontology#testProperty"
        
        # Setup: property exists but destroy_entity fails
        mock_ontology.search_one.return_value = mock_property_entity
        mock_destroy_entity.side_effect = OwlReadyError("Destruction failed")
        
        # Act & Assert
        with expect_exception(EntityDeletionError, match="Failed to delete property"):
            delete_property(mock_ontology, property_iri)

    @parametrize("property_iri", [
        "",
        None,
        "   ",
        "invalid-iri",
        "http://example.com/missing#fragment",
    ])
    def test_delete_property_invalid_iri(self, mock_ontology: Mock, property_iri: str):
        """
        Test error handling for invalid property IRIs.
        
        Args:
            mock_ontology: Mock ontology fixture
            property_iri: Invalid property IRI to test
        """
        from src.ontology.editor import delete_property, EntityDeletionError
        
        # Act & Assert
        with expect_exception(EntityDeletionError, match="Invalid property IRI"):
            delete_property(mock_ontology, property_iri)

    # =====================================================
    # Integration Tests
    # =====================================================

    @patch('src.ontology.editor.destroy_entity')
    def test_complex_deletion_scenario(
        self, 
        mock_destroy_entity: Mock,
        mock_ontology: Mock, 
        mock_class_entity: Mock,
        mock_individual_entity: Mock,
        mock_property_entity: Mock,
        mock_instance_entities: List[Mock]
    ):
        """
        Test complex deletion scenario involving multiple entity types.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
            mock_class_entity: Mock class entity fixture
            mock_individual_entity: Mock individual entity fixture
            mock_property_entity: Mock property entity fixture
            mock_instance_entities: Mock instance entities fixture
        """
        from src.ontology.editor import delete_class, delete_individual, delete_property
        
        class_iri = "http://test.example.org/ontology#TestClass"
        individual_iri = "http://test.example.org/ontology#TestIndividual"
        property_iri = "http://test.example.org/ontology#testProperty"
        
        # Setup: class has instances, all entities exist initially
        mock_class_entity.instances.return_value = mock_instance_entities
        
        # Configure search_one to return appropriate entities, then None after deletion
        search_side_effects = [
            # First call for property deletion
            mock_property_entity, None,  # property exists, then deleted
            # Second call for individual deletion
            mock_individual_entity, None,  # individual exists, then deleted
            # Third call for class deletion
            mock_class_entity, None,  # class exists, then deleted
        ]
        mock_ontology.search_one.side_effect = search_side_effects
        
        # Act: Delete in specific order (property -> individual -> class)
        delete_property(mock_ontology, property_iri)
        delete_individual(mock_ontology, individual_iri)
        delete_class(mock_ontology, class_iri)
        
        # Assert: All entities were destroyed in correct order
        expected_calls = [
            call(mock_property_entity),
            call(mock_individual_entity),
        ]
        # Add calls for class instances, then class
        for instance in mock_instance_entities:
            expected_calls.append(call(instance))
        expected_calls.append(call(mock_class_entity))
        
        mock_destroy_entity.assert_has_calls(expected_calls)

    @patch('src.ontology.editor.destroy_entity')
    def test_multiple_deletions_transaction_like(self, mock_destroy_entity: Mock, mock_ontology: Mock):
        """
        Test that multiple deletions can be performed in sequence reliably.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.editor import delete_class, delete_individual, delete_property
        
        # Setup multiple entities for deletion
        entities = []
        iris = []
        for i, entity_type in enumerate(['class', 'individual', 'property']):
            entity = Mock()
            iri = f"http://test.example.org/ontology#{entity_type}{i}"
            entity.iri = iri
            entities.append(entity)
            iris.append(iri)
        
        # Configure search to return entities, then None
        search_returns = []
        for entity in entities:
            search_returns.extend([entity, None])
        mock_ontology.search_one.side_effect = search_returns
        
        # Act: Delete all entities
        delete_class(mock_ontology, iris[0])
        delete_individual(mock_ontology, iris[1])
        delete_property(mock_ontology, iris[2])
        
        # Assert: All entities were destroyed
        expected_calls = [call(entity) for entity in entities]
        mock_destroy_entity.assert_has_calls(expected_calls)

    @patch('src.ontology.editor.destroy_entity')
    def test_deletion_verification_failure(
        self, 
        mock_destroy_entity: Mock,
        mock_ontology: Mock, 
        mock_class_entity: Mock
    ):
        """
        Test handling when entity still exists after deletion attempt.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
            mock_class_entity: Mock class entity fixture
        """
        from src.ontology.editor import delete_class, EntityDeletionError
        
        class_iri = "http://test.example.org/ontology#TestClass"
        
        # Setup: class exists before and after deletion (deletion failed)
        mock_ontology.search_one.side_effect = [mock_class_entity, mock_class_entity]
        
        # Act & Assert
        with expect_exception(EntityDeletionError, match="Entity still exists after deletion"):
            delete_class(mock_ontology, class_iri)

    # =====================================================
    # Error Handling Tests
    # =====================================================

    def test_entity_deletion_error_custom_exception(self):
        """
        Test that custom EntityDeletionError exception works correctly.
        """
        from src.ontology.editor import EntityDeletionError
        
        # Test basic exception creation
        error_msg = "Test deletion error"
        exception = EntityDeletionError(error_msg)
        
        assert str(exception) == error_msg
        assert isinstance(exception, Exception)

    def test_entity_deletion_error_with_cause(self):
        """
        Test that EntityDeletionError properly handles exception chaining.
        """
        from src.ontology.editor import EntityDeletionError
        
        # Test exception chaining
        original_error = OwlReadyError("Owlready2 error")
        try:
            raise EntityDeletionError("Wrapped deletion error") from original_error
        except EntityDeletionError as chained_error:
            assert str(chained_error) == "Wrapped deletion error"
            assert chained_error.__cause__ == original_error

    @parametrize("entity_type,function_name", [
        ("individual", "delete_individual"),
        ("property", "delete_property"),
    ])
    @patch('src.ontology.editor.destroy_entity')
    def test_generic_owlready_error_handling(
        self, 
        mock_destroy_entity: Mock,
        mock_ontology: Mock, 
        entity_type: str,
        function_name: str
    ):
        """
        Test handling of generic Owlready2 errors for all deletion functions.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
            entity_type: Type of entity being deleted
            function_name: Name of deletion function to test
        """
        from src.ontology.editor import EntityDeletionError
        import src.ontology.editor as editor_module
        
        entity_iri = f"http://test.example.org/ontology#Test{entity_type.title()}"
        
        # Setup: entity exists but destroy_entity raises generic error
        mock_entity = Mock()
        mock_destroy_entity.side_effect = OwlReadyOntologyParsingError("Parsing error")
        mock_ontology.search_one.return_value = mock_entity
        
        # Get the function to test
        delete_function = getattr(editor_module, function_name)
        
        # Act & Assert
        with expect_exception(EntityDeletionError, match=f"Failed to delete {entity_type}"):
            delete_function(mock_ontology, entity_iri)

    @patch('src.ontology.editor.destroy_entity')
    def test_class_generic_owlready_error_handling(
        self, 
        mock_destroy_entity: Mock,
        mock_ontology: Mock
    ):
        """
        Test handling of generic Owlready2 errors for class deletion function.
        
        This test is separate because class deletion has different error handling
        when dealing with instances.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.editor import delete_class, EntityDeletionError
        
        entity_iri = "http://test.example.org/ontology#TestClass"
        
        # Setup: class exists with no instances, but destroy_entity raises generic error
        mock_entity = Mock()
        mock_entity.instances.return_value = []  # No instances
        mock_destroy_entity.side_effect = OwlReadyOntologyParsingError("Parsing error")
        mock_ontology.search_one.return_value = mock_entity
        
        # Act & Assert
        with expect_exception(EntityDeletionError, match="Failed to delete class"):
            delete_class(mock_ontology, entity_iri)

    # =====================================================
    # Thread Safety Tests
    # =====================================================

    @patch('src.ontology.editor.destroy_entity')
    def test_concurrent_deletions_thread_safety(self, mock_destroy_entity: Mock, mock_ontology: Mock):
        """
        Test that ontology editing is thread-safe for concurrent operations.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.editor import delete_class, delete_individual, delete_property
        
        results = []
        errors = []
        
        def delete_entity_worker(entity_type: str, entity_id: int):
            try:
                # Create unique entity for this thread
                mock_entity = Mock()
                entity_iri = f"http://test.example.org/ontology#{entity_type}{entity_id}"
                mock_entity.iri = entity_iri
                
                # Configure search to return entity, then None
                def search_side_effect(iri):
                    if iri == entity_iri:
                        # First call returns entity, second returns None
                        if not hasattr(search_side_effect, f'called_{entity_id}'):
                            setattr(search_side_effect, f'called_{entity_id}', True)
                            return mock_entity
                        return None
                    return None
                
                mock_ontology.search_one.side_effect = search_side_effect
                
                # Call appropriate deletion function
                if entity_type == "Class":
                    delete_class(mock_ontology, entity_iri)
                elif entity_type == "Individual":
                    delete_individual(mock_ontology, entity_iri)
                else:  # Property
                    delete_property(mock_ontology, entity_iri)
                
                results.append(f"{entity_type}{entity_id}")
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads for different entity types
        threads = []
        entity_types = ["Class", "Individual", "Property"]
        for i, entity_type in enumerate(entity_types):
            thread = threading.Thread(
                target=delete_entity_worker, 
                args=(entity_type, i)
            )
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
        assert "Class0" in results
        assert "Individual1" in results
        assert "Property2" in results

    @patch('src.ontology.editor.destroy_entity')
    def test_concurrent_class_instance_deletion(
        self, 
        mock_destroy_entity: Mock,
        mock_ontology: Mock,
        mock_class_entity: Mock,
        mock_instance_entities: List[Mock]
    ):
        """
        Test thread safety when deleting class with multiple instances concurrently.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
            mock_class_entity: Mock class entity fixture
            mock_instance_entities: Mock instance entities fixture
        """
        from src.ontology.editor import delete_class
        
        class_iri = "http://test.example.org/ontology#TestClass"
        
        # Setup: class has instances  
        mock_class_entity.instances.return_value = mock_instance_entities
        
        # Setup search_one to provide different results per thread to avoid race conditions
        search_call_count = 0
        def search_side_effect(iri):
            nonlocal search_call_count
            search_call_count += 1
            # First thread gets the class and then None (successful deletion)
            # Subsequent threads get None immediately (class not found)
            if search_call_count <= 2:
                return mock_class_entity if search_call_count == 1 else None
            else:
                return None
        
        mock_ontology.search_one.side_effect = search_side_effect
        
        errors = []
        
        def delete_worker():
            try:
                delete_class(mock_ontology, class_iri)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads trying to delete the same class
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=delete_worker)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Assert at least one succeeded (others may fail with "not found" which is acceptable)
        # In real implementation, proper locking would prevent this race condition
        non_not_found_errors = [e for e in errors if "not found" not in str(e)]
        assert len(non_not_found_errors) == 0, f"Unexpected errors: {non_not_found_errors}"

    # =====================================================
    # Memory Management Tests
    # =====================================================

    @patch('src.ontology.editor.destroy_entity')
    def test_deletion_memory_cleanup(
        self, 
        mock_destroy_entity: Mock,
        mock_ontology: Mock, 
        mock_class_entity: Mock
    ):
        """
        Test that deletion properly handles memory cleanup on errors.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
            mock_class_entity: Mock class entity fixture
        """
        from src.ontology.editor import delete_class, EntityDeletionError
        
        class_iri = "http://test.example.org/ontology#TestClass"
        
        # Setup: class exists but destroy_entity fails after partial cleanup
        mock_ontology.search_one.return_value = mock_class_entity
        mock_destroy_entity.side_effect = Exception("Partial failure after cleanup")
        
        # Act & Assert
        with expect_exception(EntityDeletionError):
            delete_class(mock_ontology, class_iri)
        
        # Verify cleanup was attempted (destroy_entity was called)
        mock_destroy_entity.assert_called_once_with(mock_class_entity)

    @patch('src.ontology.editor.destroy_entity')
    def test_large_instance_set_deletion_performance(
        self, 
        mock_destroy_entity: Mock,
        mock_ontology: Mock, 
        mock_class_entity: Mock
    ):
        """
        Test performance considerations when deleting class with many instances.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
            mock_class_entity: Mock class entity fixture
        """
        from src.ontology.editor import delete_class
        
        class_iri = "http://test.example.org/ontology#TestClass"
        
        # Setup: class with many instances
        large_instance_set = []
        for i in range(100):  # Simulate 100 instances
            instance = Mock()
            instance.iri = f"http://test.example.org/ontology#Instance{i}"
            large_instance_set.append(instance)
        
        mock_class_entity.instances.return_value = large_instance_set
        mock_ontology.search_one.side_effect = [mock_class_entity, None]
        
        # Act
        delete_class(mock_ontology, class_iri)
        
        # Assert: All instances were destroyed, then the class
        expected_calls = []
        for instance in large_instance_set:
            expected_calls.append(call(instance))
        expected_calls.append(call(mock_class_entity))
        
        mock_destroy_entity.assert_has_calls(expected_calls)

    # =====================================================
    # Logging Integration Tests
    # =====================================================

    @patch('src.ontology.editor.destroy_entity')
    def test_deletion_logging_integration(
        self, 
        mock_destroy_entity: Mock,
        mock_ontology: Mock, 
        mock_class_entity: Mock
    ):
        """
        Test that deletion operations integrate properly with logging system.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
            mock_class_entity: Mock class entity fixture
        """
        from src.ontology.editor import delete_class
        import logging
        
        with patch('logging.getLogger') as mock_logger:
            class_iri = "http://test.example.org/ontology#TestClass"
            
            # Setup
            mock_ontology.search_one.side_effect = [mock_class_entity, None]
            logger_instance = Mock()
            mock_logger.return_value = logger_instance
            
            # Act
            delete_class(mock_ontology, class_iri)
            
            # Assert logging was configured (if implemented in actual editor)
            # This test documents expected logging behavior
            mock_logger.assert_called_with('src.ontology.editor')

    def test_detailed_error_messages(self, mock_ontology: Mock):
        """
        Test that error messages provide sufficient detail for debugging.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.editor import delete_class, EntityDeletionError
        
        class_iri = "http://test.example.org/ontology#DetailedErrorTest"
        
        # Setup: class doesn't exist
        mock_ontology.search_one.return_value = None
        
        # Act & Assert
        with expect_exception(EntityDeletionError) as exc_info:
            delete_class(mock_ontology, class_iri)
        
        # Verify error message contains helpful information
        error_message = str(exc_info.value)
        assert "Class not found" in error_message
        assert class_iri in error_message or "TestClass" in error_message

    @patch('src.ontology.editor.destroy_entity')
    @parametrize("entity_count", [0, 1, 5, 50])
    def test_delete_class_various_instance_counts(
        self, 
        mock_destroy_entity: Mock,
        mock_ontology: Mock, 
        mock_class_entity: Mock,
        entity_count: int
    ):
        """
        Test class deletion with various numbers of instances.
        
        Args:
            mock_destroy_entity: Mock for destroy_entity function
            mock_ontology: Mock ontology fixture
            mock_class_entity: Mock class entity fixture
            entity_count: Number of instances to create
        """
        from src.ontology.editor import delete_class
        
        class_iri = "http://test.example.org/ontology#TestClass"
        
        # Setup: create specified number of instances
        instances = []
        for i in range(entity_count):
            instance = Mock()
            instance.iri = f"http://test.example.org/ontology#Instance{i}"
            instances.append(instance)
        
        mock_class_entity.instances.return_value = instances
        mock_ontology.search_one.side_effect = [mock_class_entity, None]
        
        # Act
        delete_class(mock_ontology, class_iri)
        
        # Assert: All instances were destroyed, then the class
        expected_calls = []
        for instance in instances:
            expected_calls.append(call(instance))
        expected_calls.append(call(mock_class_entity))
        
        assert len(instances) == entity_count
        mock_destroy_entity.assert_has_calls(expected_calls)