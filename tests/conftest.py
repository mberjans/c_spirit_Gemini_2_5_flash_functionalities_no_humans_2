"""
Pytest configuration and shared fixtures for AIM2-ODIE project tests.

This module provides common fixtures and configuration for all test modules
in the AIM2-ODIE ontology development and information extraction system.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
import pytest


# Add src directory to Python path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def project_root_path() -> Path:
    """
    Fixture providing the absolute path to the project root directory.
    
    Returns:
        Path: Absolute path to the project root
    """
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root_path: Path) -> Path:
    """
    Fixture providing the path to test data directory.
    
    Args:
        project_root_path: Path to project root
        
    Returns:
        Path: Path to test data directory
    """
    return project_root_path / "data" / "test"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Fixture providing a temporary directory for test operations.
    
    Yields:
        Path: Temporary directory path that is automatically cleaned up
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_text() -> str:
    """
    Fixture providing sample text for text processing tests.
    
    Returns:
        str: Sample text about plant metabolomics
    """
    return """
    Plant metabolomics is the study of small molecules (metabolites) found in plants.
    These metabolites include primary metabolites like amino acids, sugars, and organic acids,
    as well as secondary metabolites such as flavonoids, alkaloids, and terpenoids.
    The analysis of plant metabolomes provides insights into plant physiology,
    stress responses, and biochemical pathways.
    """


@pytest.fixture
def sample_ontology_data() -> Dict[str, Any]:
    """
    Fixture providing sample ontology data for testing.
    
    Returns:
        Dict[str, Any]: Sample ontology structure
    """
    return {
        "entities": [
            {
                "id": "CHEBI:15756",
                "name": "hexose",
                "definition": "Any six-carbon monosaccharide",
                "synonyms": ["six-carbon sugar"]
            },
            {
                "id": "CHEBI:18059",
                "name": "lipid",
                "definition": "Any of a group of organic compounds",
                "synonyms": ["fat", "fatty substance"]
            }
        ],
        "relations": [
            {
                "subject": "CHEBI:15756",
                "predicate": "is_a",
                "object": "CHEBI:16646"
            }
        ]
    }


@pytest.fixture
def sample_extraction_result() -> Dict[str, Any]:
    """
    Fixture providing sample LLM extraction results for testing.
    
    Returns:
        Dict[str, Any]: Sample extraction result structure
    """
    return {
        "entities": [
            {
                "text": "flavonoids",
                "label": "COMPOUND",
                "start": 0,
                "end": 10,
                "confidence": 0.95
            },
            {
                "text": "Arabidopsis thaliana",
                "label": "ORGANISM",
                "start": 20,
                "end": 40,
                "confidence": 0.98
            }
        ],
        "relations": [
            {
                "subject": "flavonoids",
                "relation": "found_in",
                "object": "Arabidopsis thaliana",
                "confidence": 0.87
            }
        ]
    }


@pytest.fixture
def mock_pubmed_record() -> Dict[str, Any]:
    """
    Fixture providing mock PubMed record for data acquisition tests.
    
    Returns:
        Dict[str, Any]: Mock PubMed record structure
    """
    return {
        "pmid": "12345678",
        "title": "Metabolomic analysis of plant stress responses",
        "abstract": "This study investigates metabolomic changes in plants under stress conditions...",
        "authors": ["Smith J", "Johnson A", "Brown K"],
        "journal": "Plant Physiology",
        "year": 2023,
        "doi": "10.1104/pp.23.00123"
    }


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Session-scoped fixture to set up the test environment.
    Automatically runs for all tests.
    """
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    yield
    
    # Cleanup after all tests
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
    if "LOG_LEVEL" in os.environ:
        del os.environ["LOG_LEVEL"]


# Pytest configuration
def pytest_configure(config):
    """
    Configure pytest with custom markers and settings.
    
    Args:
        config: Pytest configuration object
    """
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "ontology: mark test as ontology-related"
    )
    config.addinivalue_line(
        "markers", "llm: mark test as LLM-related"
    )
    config.addinivalue_line(
        "markers", "data_acquisition: mark test as data acquisition-related"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify collected test items to add markers based on test location.
    
    Args:
        config: Pytest configuration object
        items: List of collected test items
    """
    for item in items:
        # Add markers based on test file location
        test_path = str(item.fspath)
        
        if "/tests/ontology/" in test_path:
            item.add_marker(pytest.mark.ontology)
        elif "/tests/llm_extraction/" in test_path:
            item.add_marker(pytest.mark.llm)
        elif "/tests/data_acquisition/" in test_path:
            item.add_marker(pytest.mark.data_acquisition)
        
        # Mark integration tests
        if "integration" in item.name.lower() or "test_integration" in test_path:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)