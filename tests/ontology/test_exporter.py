"""
Unit tests for the ontology exporter module.

This module contains comprehensive tests for exporting OWL 2.0 ontologies
using Owlready2 to various formats. Tests cover successful export scenarios
as well as various error conditions.

Test Categories:
- Successful export to temporary file paths
- Verification of exported file content (OWL/RDF/XML tags)
- Loading exported files back into Owlready2 for validation
- Error handling for invalid file paths and write permissions
- Format validation and error handling
- Custom exception handling
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Generator
import xml.etree.ElementTree as ET

import pytest
from owlready2 import OwlReadyError, OwlReadyOntologyParsingError

from src.utils.testing_framework import expect_exception, parametrize


class TestOntologyExporter:
    """Test suite for ontology export functionality."""

    @pytest.fixture
    def mock_ontology(self) -> Mock:
        """
        Create a mock ontology object for testing export.
        
        Returns:
            Mock: Mock ontology object with save method
        """
        mock_ont = Mock()
        mock_ont.save = Mock()
        mock_ont.name = "test_ontology"
        mock_ont.base_iri = "http://test.example.org/ontology"
        mock_ont.classes = Mock(return_value=[])
        mock_ont.individuals = Mock(return_value=[])
        mock_ont.properties = Mock(return_value=[])
        return mock_ont

    @pytest.fixture
    def temp_export_file(self, temp_dir: Path) -> Generator[Path, None, None]:
        """
        Create a temporary file path for export testing.
        
        Args:
            temp_dir: Temporary directory fixture from conftest.py
            
        Yields:
            Path: Path to temporary export file
        """
        export_file = temp_dir / "exported_ontology.owl"
        yield export_file

    @pytest.fixture
    def valid_owl_content(self) -> str:
        """
        Create valid OWL/RDF/XML content for testing.
        
        Returns:
            str: Valid OWL content string
        """
        return '''<?xml version="1.0"?>
<rdf:RDF xmlns="http://test.example.org/ontology#"
         xml:base="http://test.example.org/ontology"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://test.example.org/ontology">
        <rdfs:label>Test Ontology</rdfs:label>
        <rdfs:comment>A test ontology exported for validation</rdfs:comment>
    </owl:Ontology>
    
    <owl:Class rdf:about="http://test.example.org/ontology#ExportedClass">
        <rdfs:label>Exported Class</rdfs:label>
        <rdfs:comment>A test class to verify export functionality</rdfs:comment>
    </owl:Class>
</rdf:RDF>'''

    def test_export_ontology_success_default_format(
        self, 
        mock_ontology: Mock,
        temp_export_file: Path,
        valid_owl_content: str
    ):
        """
        Test successful export of ontology using default RDF/XML format.
        
        Args:
            mock_ontology: Mock ontology fixture
            temp_export_file: Temporary export file path
            valid_owl_content: Valid OWL content for verification
        """
        from src.ontology.exporter import export_ontology
        
        # Mock the save method to actually create a file with content
        def mock_save(file=None, format=None):
            if file:
                Path(file).write_text(valid_owl_content, encoding="utf-8")
        
        mock_ontology.save.side_effect = mock_save
        
        # Act
        result = export_ontology(mock_ontology, str(temp_export_file))
        
        # Assert
        assert result is True or result is None  # Allow for void function
        
        # Verify save was called with correct parameters
        mock_ontology.save.assert_called_once_with(
            file=str(temp_export_file), 
            format='rdfxml'
        )
        
        # Verify file was created and has content
        assert temp_export_file.exists()
        assert temp_export_file.stat().st_size > 0

    @parametrize("export_format", [
        "rdfxml",
        "owlxml", 
        "ntriples",
        "turtle"
    ])
    def test_export_ontology_success_different_formats(
        self, 
        mock_ontology: Mock,
        temp_export_file: Path,
        valid_owl_content: str,
        export_format: str
    ):
        """
        Test successful export with different format options.
        
        Args:
            mock_ontology: Mock ontology fixture
            temp_export_file: Temporary export file path
            valid_owl_content: Valid OWL content for verification
            export_format: Format to test
        """
        from src.ontology.exporter import export_ontology
        
        # Mock the save method to create file with content
        def mock_save(file=None, format=None):
            if file:
                Path(file).write_text(valid_owl_content, encoding="utf-8")
        
        mock_ontology.save.side_effect = mock_save
        
        # Act
        export_ontology(mock_ontology, str(temp_export_file), format=export_format)
        
        # Assert
        mock_ontology.save.assert_called_once_with(
            file=str(temp_export_file), 
            format=export_format
        )
        
        # Verify file was created
        assert temp_export_file.exists()

    def test_export_ontology_file_content_validation(
        self, 
        mock_ontology: Mock,
        temp_export_file: Path,
        valid_owl_content: str
    ):
        """
        Test that exported file contains expected OWL/RDF/XML tags.
        
        Args:
            mock_ontology: Mock ontology fixture
            temp_export_file: Temporary export file path
            valid_owl_content: Valid OWL content for verification
        """
        from src.ontology.exporter import export_ontology
        
        # Mock the save method to create file with actual OWL content
        def mock_save(file=None, format=None):
            if file:
                Path(file).write_text(valid_owl_content, encoding="utf-8")
        
        mock_ontology.save.side_effect = mock_save
        
        # Act
        export_ontology(mock_ontology, str(temp_export_file))
        
        # Assert file content contains expected OWL/RDF/XML elements
        content = temp_export_file.read_text(encoding="utf-8")
        
        # Check for essential OWL/RDF elements
        assert '<?xml version="1.0"?>' in content
        assert 'xmlns:owl="http://www.w3.org/2002/07/owl#"' in content
        assert 'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"' in content
        assert 'xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"' in content
        assert '<owl:Ontology' in content
        assert '</rdf:RDF>' in content
        
        # Verify it's valid XML by parsing it
        try:
            ET.fromstring(content)
        except ET.ParseError as e:
            pytest.fail(f"Exported content is not valid XML: {e}")

    def test_export_ontology_reload_validation(
        self, 
        mock_ontology: Mock,
        temp_export_file: Path,
        valid_owl_content: str
    ):
        """
        Test loading exported file back into Owlready2 to confirm validity.
        
        Args:
            mock_ontology: Mock ontology fixture
            temp_export_file: Temporary export file path
            valid_owl_content: Valid OWL content for verification
        """
        from src.ontology.exporter import export_ontology
        from src.ontology.loader import load_ontology_from_file
        
        # Mock the save method to create valid OWL file
        def mock_save(file=None, format=None):
            if file:
                Path(file).write_text(valid_owl_content, encoding="utf-8")
        
        mock_ontology.save.side_effect = mock_save
        
        # Export ontology
        export_ontology(mock_ontology, str(temp_export_file))
        
        # Verify file was created
        assert temp_export_file.exists()
        
        # Test that the exported file can be loaded back
        # This will use mocked owlready2.get_ontology, so we verify the call pattern
        with patch('owlready2.get_ontology') as mock_get_ont:
            # Setup mock for successful loading
            mock_loaded_ont = Mock()
            mock_loaded_ont.load.return_value = mock_loaded_ont
            mock_get_ont.return_value = mock_loaded_ont
            
            # Act - attempt to load exported file
            loaded_ontology = load_ontology_from_file(str(temp_export_file))
            
            # Assert loading was attempted and succeeded
            assert loaded_ontology is not None
            mock_get_ont.assert_called_once()
            mock_loaded_ont.load.assert_called_once()

    def test_export_ontology_invalid_file_path(self, mock_ontology: Mock):
        """
        Test error handling for invalid file paths.
        
        Args:
            mock_ontology: Mock ontology fixture
        """
        from src.ontology.exporter import export_ontology, OntologyExportError
        
        invalid_path = "/invalid/nonexistent/directory/file.owl"
        
        # Mock save to raise permission error for invalid path
        mock_ontology.save.side_effect = PermissionError("Permission denied")
        
        # Act & Assert
        with expect_exception(OntologyExportError, match="Permission denied|write permissions"):
            export_ontology(mock_ontology, invalid_path)

    def test_export_ontology_permission_error(
        self, 
        mock_ontology: Mock,
        temp_dir: Path
    ):
        """
        Test error handling for write permission errors.
        
        Args:
            mock_ontology: Mock ontology fixture
            temp_dir: Temporary directory fixture
        """
        from src.ontology.exporter import export_ontology, OntologyExportError
        
        # Create a path in temp directory
        protected_file = temp_dir / "protected_file.owl"
        
        # Mock save to raise permission error
        mock_ontology.save.side_effect = PermissionError("Write permission denied")
        
        # Act & Assert
        with expect_exception(OntologyExportError, match="Permission denied|write permissions"):
            export_ontology(mock_ontology, str(protected_file))

    def test_export_ontology_disk_space_error(
        self, 
        mock_ontology: Mock,
        temp_export_file: Path
    ):
        """
        Test error handling for disk space issues.
        
        Args:
            mock_ontology: Mock ontology fixture
            temp_export_file: Temporary export file path
        """
        from src.ontology.exporter import export_ontology, OntologyExportError
        
        # Mock save to raise disk space error
        mock_ontology.save.side_effect = OSError("No space left on device")
        
        # Act & Assert
        with expect_exception(OntologyExportError, match="No space left on device|disk space"):
            export_ontology(mock_ontology, str(temp_export_file))

    def test_export_ontology_generic_owlready_error(
        self, 
        mock_ontology: Mock,
        temp_export_file: Path
    ):
        """
        Test error handling for generic Owlready2 errors during export.
        
        Args:
            mock_ontology: Mock ontology fixture
            temp_export_file: Temporary export file path
        """
        from src.ontology.exporter import export_ontology, OntologyExportError
        
        # Mock save to raise generic Owlready2 error
        mock_ontology.save.side_effect = OwlReadyError("Export serialization failed")
        
        # Act & Assert
        with expect_exception(OntologyExportError, match="Owlready2 error"):
            export_ontology(mock_ontology, str(temp_export_file))

    @parametrize("file_path", [
        "",
        None,
        "   ",
    ])
    def test_export_ontology_invalid_file_path_input(
        self, 
        mock_ontology: Mock,
        file_path: str
    ):
        """
        Test error handling for invalid file path inputs.
        
        Args:
            mock_ontology: Mock ontology fixture
            file_path: Invalid file path to test
        """
        from src.ontology.exporter import export_ontology, OntologyExportError
        
        # Act & Assert
        with expect_exception(OntologyExportError, match="Invalid file path"):
            export_ontology(mock_ontology, file_path)

    @parametrize("export_format", [
        "invalid_format",
        "pdf",
        "json",
        "",
        None
    ])
    def test_export_ontology_invalid_format(
        self, 
        mock_ontology: Mock,
        temp_export_file: Path,
        export_format: str
    ):
        """
        Test error handling for invalid export formats.
        
        Args:
            mock_ontology: Mock ontology fixture
            temp_export_file: Temporary export file path
            export_format: Invalid format to test
        """
        from src.ontology.exporter import export_ontology, OntologyExportError
        
        # Mock save to raise error for invalid format
        mock_ontology.save.side_effect = ValueError(f"Unsupported format: {export_format}")
        
        # Act & Assert
        with expect_exception(OntologyExportError, match="Unsupported format|Invalid format"):
            export_ontology(mock_ontology, str(temp_export_file), format=export_format)

    def test_export_ontology_none_ontology_input(self, temp_export_file: Path):
        """
        Test error handling when None ontology is passed.
        
        Args:
            temp_export_file: Temporary export file path
        """
        from src.ontology.exporter import export_ontology, OntologyExportError
        
        # Act & Assert
        with expect_exception(OntologyExportError, match="Invalid ontology|ontology cannot be None"):
            export_ontology(None, str(temp_export_file))

    def test_export_ontology_empty_ontology(
        self, 
        temp_export_file: Path,
        valid_owl_content: str
    ):
        """
        Test export of an empty ontology (no classes, individuals, properties).
        
        Args:
            temp_export_file: Temporary export file path
            valid_owl_content: Valid OWL content for verification
        """
        from src.ontology.exporter import export_ontology
        
        # Create mock empty ontology
        empty_ontology = Mock()
        empty_ontology.classes = Mock(return_value=[])
        empty_ontology.individuals = Mock(return_value=[])
        empty_ontology.properties = Mock(return_value=[])
        empty_ontology.name = "empty_ontology"
        
        # Mock save to create minimal valid OWL file
        def mock_save(file=None, format=None):
            if file:
                minimal_content = '''<?xml version="1.0"?>
<rdf:RDF xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <owl:Ontology rdf:about="http://example.org/empty"/>
</rdf:RDF>'''
                Path(file).write_text(minimal_content, encoding="utf-8")
        
        empty_ontology.save.side_effect = mock_save
        
        # Act
        export_ontology(empty_ontology, str(temp_export_file))
        
        # Assert file was created and is valid
        assert temp_export_file.exists()
        content = temp_export_file.read_text(encoding="utf-8")
        assert '<owl:Ontology' in content
        
        # Verify it's valid XML
        try:
            ET.fromstring(content)
        except ET.ParseError as e:
            pytest.fail(f"Exported empty ontology is not valid XML: {e}")

    def test_ontology_export_error_custom_exception(self):
        """
        Test that custom OntologyExportError exception works correctly.
        """
        from src.ontology.exporter import OntologyExportError
        
        # Test basic exception creation
        error_msg = "Test export error message"
        exception = OntologyExportError(error_msg)
        
        assert str(exception) == error_msg
        assert isinstance(exception, Exception)

    def test_ontology_export_error_with_cause(self):
        """
        Test that OntologyExportError properly handles exception chaining.
        """
        from src.ontology.exporter import OntologyExportError
        
        # Test exception chaining
        original_error = IOError("Original I/O error")
        try:
            raise OntologyExportError("Export failed") from original_error
        except OntologyExportError as chained_error:
            assert str(chained_error) == "Export failed"
            assert chained_error.__cause__ == original_error

    def test_export_ontology_file_overwrite(
        self, 
        mock_ontology: Mock,
        temp_export_file: Path,
        valid_owl_content: str
    ):
        """
        Test that export correctly overwrites existing files.
        
        Args:
            mock_ontology: Mock ontology fixture
            temp_export_file: Temporary export file path
            valid_owl_content: Valid OWL content for verification
        """
        from src.ontology.exporter import export_ontology
        
        # Create existing file with different content
        temp_export_file.write_text("existing content", encoding="utf-8")
        original_size = temp_export_file.stat().st_size
        
        # Mock save to overwrite with new content
        def mock_save(file=None, format=None):
            if file:
                Path(file).write_text(valid_owl_content, encoding="utf-8")
        
        mock_ontology.save.side_effect = mock_save
        
        # Act
        export_ontology(mock_ontology, str(temp_export_file))
        
        # Assert file was overwritten
        assert temp_export_file.exists()
        new_content = temp_export_file.read_text(encoding="utf-8")
        assert new_content == valid_owl_content
        assert temp_export_file.stat().st_size != original_size

    def test_export_ontology_concurrent_exports(
        self, 
        temp_dir: Path,
        valid_owl_content: str
    ):
        """
        Test that multiple concurrent exports work correctly.
        
        Args:
            temp_dir: Temporary directory fixture
            valid_owl_content: Valid OWL content for verification
        """
        import threading
        from src.ontology.exporter import export_ontology
        
        results = []
        errors = []
        
        def export_worker(worker_id: int):
            try:
                # Create mock ontology for this worker
                mock_ont = Mock()
                mock_ont.name = f"ontology_{worker_id}"
                
                # Create unique export file for this worker
                export_file = temp_dir / f"export_{worker_id}.owl"
                
                def mock_save(file=None, format=None):
                    if file:
                        Path(file).write_text(
                            valid_owl_content.replace("test_ontology", f"ontology_{worker_id}"), 
                            encoding="utf-8"
                        )
                
                mock_ont.save.side_effect = mock_save
                
                # Export ontology
                export_ontology(mock_ont, str(export_file))
                results.append(export_file)
                
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=export_worker, args=(i,))
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
        
        # Verify all files were created
        for export_file in results:
            assert export_file.exists()
            assert export_file.stat().st_size > 0

    def test_export_ontology_large_file_handling(
        self, 
        mock_ontology: Mock,
        temp_export_file: Path
    ):
        """
        Test export handling for large ontologies.
        
        Args:
            mock_ontology: Mock ontology fixture
            temp_export_file: Temporary export file path
        """
        from src.ontology.exporter import export_ontology
        
        # Create large content to simulate big ontology
        large_content = '''<?xml version="1.0"?>
<rdf:RDF xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <owl:Ontology rdf:about="http://example.org/large"/>
''' + '\n'.join([f'    <owl:Class rdf:about="http://example.org/class{i}"/>' 
                 for i in range(1000)]) + '\n</rdf:RDF>'
        
        def mock_save(file=None, format=None):
            if file:
                Path(file).write_text(large_content, encoding="utf-8")
        
        mock_ontology.save.side_effect = mock_save
        
        # Act
        export_ontology(mock_ontology, str(temp_export_file))
        
        # Assert large file was created successfully
        assert temp_export_file.exists()
        assert temp_export_file.stat().st_size > 10000  # Should be reasonably large
        
        # Verify content is still valid XML
        content = temp_export_file.read_text(encoding="utf-8")
        try:
            ET.fromstring(content)
        except ET.ParseError as e:
            pytest.fail(f"Large exported ontology is not valid XML: {e}")