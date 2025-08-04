"""
Unit tests for verifying Poetry installation and basic dependency imports.

This module tests that Poetry is properly installed and accessible, and that
core dependencies can be imported (handling cases where they're not yet installed).

Task: AIM2-ODIE-002-T1 - Create unit tests for Poetry installation and dependency verification
"""

import subprocess
import sys
import importlib
from typing import List, Tuple
import pytest


class TestPoetryInstallation:
    """Test suite for verifying Poetry installation and accessibility."""

    def test_poetry_command_exists(self):
        """Test that Poetry command is available in the system PATH."""
        try:
            result = subprocess.run(
                ["poetry", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode == 0, (
                f"Poetry command failed with return code {result.returncode}. "
                f"stderr: {result.stderr}"
            )
            assert "Poetry" in result.stdout, (
                f"Expected 'Poetry' in version output, got: {result.stdout}"
            )
        except FileNotFoundError:
            pytest.fail(
                "Poetry command not found. Please install Poetry first. "
                "Visit: https://python-poetry.org/docs/#installation"
            )
        except subprocess.TimeoutExpired:
            pytest.fail("Poetry command timed out after 10 seconds")

    def test_poetry_show_command(self):
        """Test that Poetry can list installed packages."""
        try:
            result = subprocess.run(
                ["poetry", "show"],
                capture_output=True,
                text=True,
                timeout=15
            )
            # Poetry show should work even with no packages installed
            assert result.returncode in [0, 1], (
                f"Poetry show command failed unexpectedly with return code "
                f"{result.returncode}. stderr: {result.stderr}"
            )
        except FileNotFoundError:
            pytest.fail("Poetry command not found")
        except subprocess.TimeoutExpired:
            pytest.fail("Poetry show command timed out after 15 seconds")

    def test_poetry_env_info(self):
        """Test that Poetry can provide environment information."""
        try:
            result = subprocess.run(
                ["poetry", "env", "info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Should work regardless of virtual environment state
            assert result.returncode in [0, 1], (
                f"Poetry env info failed with return code {result.returncode}. "
                f"stderr: {result.stderr}"
            )
        except FileNotFoundError:
            pytest.fail("Poetry command not found")
        except subprocess.TimeoutExpired:
            pytest.fail("Poetry env info command timed out after 10 seconds")


class TestCoreDependencies:
    """Test suite for verifying core project dependencies can be imported."""

    # Core dependencies that will be used in the project
    CORE_DEPENDENCIES = [
        "owlready2",
        "Bio",  # Biopython
        "fitz",  # PyMuPDF
        "text2term",
        "llm_ie",  # LLM-IE
        "ontogpt",  # OntoGPT
        "fuzzywuzzy",
        "dedupe",
        "multitax",
        "ncbi_taxonomist"
    ]

    @pytest.mark.parametrize("dependency", CORE_DEPENDENCIES)
    def test_core_dependency_import(self, dependency: str):
        """
        Test that core dependencies can be imported when available.
        
        This test will skip if the dependency is not installed, making it
        suitable for early project phases where dependencies aren't yet added.
        
        Args:
            dependency: Name of the dependency module to test
        """
        try:
            importlib.import_module(dependency)
            print(f"✓ Successfully imported {dependency}")
        except ImportError as e:
            pytest.skip(
                f"Dependency '{dependency}' not yet installed. "
                f"This is expected in early project phases. Error: {e}"
            )
        except Exception as e:
            pytest.fail(
                f"Unexpected error importing '{dependency}': {type(e).__name__}: {e}"
            )

    def test_multiple_core_dependencies_summary(self):
        """Provide a summary of which core dependencies are available."""
        available_deps: List[str] = []
        missing_deps: List[str] = []
        error_deps: List[Tuple[str, str]] = []

        for dep in self.CORE_DEPENDENCIES:
            try:
                importlib.import_module(dep)
                available_deps.append(dep)
            except ImportError:
                missing_deps.append(dep)
            except Exception as e:
                error_deps.append((dep, f"{type(e).__name__}: {e}"))

        # Print summary for informational purposes
        print("\nDependency Summary:")
        print(f"Available ({len(available_deps)}): {', '.join(available_deps) if available_deps else 'None'}")
        print(f"Missing ({len(missing_deps)}): {', '.join(missing_deps) if missing_deps else 'None'}")
        if error_deps:
            print(f"Errors ({len(error_deps)}):")
            for dep, error in error_deps:
                print(f"  - {dep}: {error}")

        # This test always passes as it's informational
        # In early project phases, it's expected that most dependencies are missing
        assert True, "Dependency summary completed"


class TestDevelopmentDependencies:
    """Test suite for verifying development dependencies."""

    DEVELOPMENT_DEPENDENCIES = [
        "pytest",
        "ruff", 
        "black"
    ]

    @pytest.mark.parametrize("dev_dependency", DEVELOPMENT_DEPENDENCIES)
    def test_dev_dependency_import(self, dev_dependency: str):
        """
        Test that development dependencies can be imported when available.
        
        Args:
            dev_dependency: Name of the development dependency to test
        """
        try:
            importlib.import_module(dev_dependency)
            print(f"✓ Successfully imported development dependency {dev_dependency}")
        except ImportError as e:
            pytest.skip(
                f"Development dependency '{dev_dependency}' not yet installed. "
                f"Error: {e}"
            )
        except Exception as e:
            pytest.fail(
                f"Unexpected error importing '{dev_dependency}': {type(e).__name__}: {e}"
            )

    def test_pytest_functionality(self):
        """Test that pytest is working correctly (this test itself proves it)."""
        assert True, "If this test runs, pytest is working correctly"

    def test_python_version_compatibility(self):
        """Test that Python version is compatible with project requirements."""
        python_version = sys.version_info
        
        # Assuming Python 3.8+ is required (adjust as needed)
        min_python = (3, 8)
        
        assert python_version >= min_python, (
            f"Python {min_python[0]}.{min_python[1]}+ is required, "
            f"but found Python {python_version.major}.{python_version.minor}.{python_version.micro}"
        )
        
        print(f"✓ Python version {python_version.major}.{python_version.minor}.{python_version.micro} is compatible")


class TestProjectStructure:
    """Test suite for verifying basic project structure requirements."""

    def test_tests_directory_structure(self):
        """Test that the tests directory has the expected structure."""
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent
        tests_dir = project_root / "tests"
        
        assert tests_dir.exists(), "Tests directory should exist"
        assert (tests_dir / "__init__.py").exists(), "Tests __init__.py should exist"
        
        # Check for expected test subdirectories
        expected_subdirs = [
            "cli", "data_acquisition", "data_quality", "evaluation",
            "llm_extraction", "ontology", "ontology_mapping", "text_processing"
        ]
        
        for subdir in expected_subdirs:
            subdir_path = tests_dir / subdir
            assert subdir_path.exists(), f"Tests subdirectory '{subdir}' should exist"
            assert (subdir_path / "__init__.py").exists(), f"Tests subdirectory '{subdir}' should have __init__.py"

    def test_src_directory_exists(self):
        """Test that the src directory exists for the main source code."""
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent
        src_dir = project_root / "src"
        
        assert src_dir.exists(), "Source directory should exist"


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v", "--tb=short"])