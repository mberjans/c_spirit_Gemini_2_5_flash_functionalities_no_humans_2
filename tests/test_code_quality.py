"""
Test module to validate code quality tools (ruff, black, pytest) on the dummy test file.
This module demonstrates that code quality tools can detect intentional issues.
"""

import subprocess
import sys
import os
from pathlib import Path
import pytest


class TestCodeQualityTools:
    """Test class to validate code quality tools functionality."""
    
    @classmethod
    def setup_class(cls):
        """Set up the test class with file paths."""
        cls.project_root = Path(__file__).parent.parent
        cls.dummy_file = cls.project_root / "src" / "temp_test_file.py"
        
        # Ensure the dummy file exists
        if not cls.dummy_file.exists():
            pytest.skip("Dummy test file not found at src/temp_test_file.py")
    
    def test_ruff_detects_issues(self):
        """Test that ruff detects code quality issues in the dummy file."""
        try:
            # Run ruff check on the dummy file
            result = subprocess.run(
                ["ruff", "check", str(self.dummy_file)],
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            
            # Ruff should detect issues (non-zero exit code)
            assert result.returncode != 0, "Ruff should have detected issues in the dummy file"
            
            # Check that output contains expected issue types
            output = result.stdout + result.stderr
            
            # Expected issues in the dummy file:
            expected_issues = [
                "F401",  # Unused import
                "F841",  # Unused variable
                "E501",  # Line too long (if configured)
            ]
            
            detected_issues = []
            for issue_code in expected_issues:
                if issue_code in output:
                    detected_issues.append(issue_code)
            
            assert len(detected_issues) > 0, f"Expected to detect at least one of {expected_issues}, but got output: {output}"
            
            print(f"Ruff successfully detected {len(detected_issues)} types of issues: {detected_issues}")
            print(f"Ruff output: {output}")
            
        except FileNotFoundError:
            pytest.skip("Ruff is not installed or not available in PATH")
        except subprocess.SubprocessError as e:
            pytest.fail(f"Failed to run ruff: {e}")
    
    def test_black_detects_formatting_issues(self):
        """Test that black detects formatting issues in the dummy file."""
        try:
            # Run black in check mode (don't modify files)
            result = subprocess.run(
                ["black", "--check", "--diff", str(self.dummy_file)],
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            
            # Black should detect formatting issues (non-zero exit code)
            assert result.returncode != 0, "Black should have detected formatting issues in the dummy file"
            
            # Check that output indicates changes would be made
            output = result.stdout + result.stderr
            
            # Black typically outputs "would reformat" or shows diffs
            format_indicators = ["would reformat", "reformatted", "---", "+++"]
            has_format_issues = any(indicator in output.lower() for indicator in format_indicators)
            
            assert has_format_issues, f"Expected black to show formatting changes, but got output: {output}"
            
            print(f"Black successfully detected formatting issues")
            print(f"Black output: {output}")
            
        except FileNotFoundError:
            pytest.skip("Black is not installed or not available in PATH")
        except subprocess.SubprocessError as e:
            pytest.fail(f"Failed to run black: {e}")
    
    def test_pytest_runs_on_dummy_file(self):
        """Test that pytest can run the test functions in the dummy file."""
        try:
            # Run pytest on the dummy file specifically
            result = subprocess.run(
                ["python", "-m", "pytest", str(self.dummy_file), "-v"],
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            
            # Pytest should run successfully (the tests in dummy file should pass)
            assert result.returncode == 0, f"Pytest should run successfully on dummy file. Output: {result.stdout + result.stderr}"
            
            output = result.stdout + result.stderr
            
            # Check that pytest found and ran the test functions
            test_indicators = ["test_calculate_sum", "test_format_string", "test_data_processor"]
            found_tests = []
            
            for test_name in test_indicators:
                if test_name in output:
                    found_tests.append(test_name)
            
            assert len(found_tests) > 0, f"Expected pytest to find test functions, but got output: {output}"
            
            # Check for PASSED indicators
            assert "PASSED" in output or "passed" in output, f"Expected tests to pass, but got output: {output}"
            
            print(f"Pytest successfully ran {len(found_tests)} test functions: {found_tests}")
            print(f"Pytest output: {output}")
            
        except FileNotFoundError:
            pytest.skip("Python or pytest is not available")
        except subprocess.SubprocessError as e:
            pytest.fail(f"Failed to run pytest: {e}")
    
    def test_ruff_format_detects_issues(self):
        """Test that ruff format detects formatting issues in the dummy file."""
        try:
            # Run ruff format in check mode (don't modify files)
            result = subprocess.run(
                ["ruff", "format", "--check", str(self.dummy_file)],
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            
            # Ruff format should detect formatting issues (non-zero exit code)
            assert result.returncode != 0, "Ruff format should have detected formatting issues in the dummy file"
            
            output = result.stdout + result.stderr
            
            # Ruff format typically shows which files would be reformatted
            assert "would reformat" in output.lower() or str(self.dummy_file.name) in output, \
                f"Expected ruff format to indicate formatting changes, but got output: {output}"
            
            print(f"Ruff format successfully detected formatting issues")
            print(f"Ruff format output: {output}")
            
        except FileNotFoundError:
            pytest.skip("Ruff is not installed or not available in PATH")
        except subprocess.SubprocessError as e:
            pytest.fail(f"Failed to run ruff format: {e}")
    
    def test_code_quality_tools_integration(self):
        """Integration test to verify all code quality tools work together."""
        tools_results = {}
        
        # Test ruff check
        try:
            result = subprocess.run(
                ["ruff", "check", str(self.dummy_file)],
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            tools_results["ruff_check"] = {
                "available": True,
                "detected_issues": result.returncode != 0,
                "output_length": len(result.stdout + result.stderr)
            }
        except FileNotFoundError:
            tools_results["ruff_check"] = {"available": False, "detected_issues": False, "output_length": 0}
        
        # Test black check
        try:
            result = subprocess.run(
                ["black", "--check", str(self.dummy_file)],
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            tools_results["black"] = {
                "available": True,
                "detected_issues": result.returncode != 0,
                "output_length": len(result.stdout + result.stderr)
            }
        except FileNotFoundError:
            tools_results["black"] = {"available": False, "detected_issues": False, "output_length": 0}
        
        # Test pytest
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", str(self.dummy_file), "-v"],
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            tools_results["pytest"] = {
                "available": True,
                "tests_passed": result.returncode == 0,
                "output_length": len(result.stdout + result.stderr)
            }
        except FileNotFoundError:
            tools_results["pytest"] = {"available": False, "tests_passed": False, "output_length": 0}
        
        # Verify that at least some tools are available and working
        available_tools = [name for name, data in tools_results.items() if data["available"]]
        assert len(available_tools) > 0, f"No code quality tools are available. Results: {tools_results}"
        
        # Print summary
        print(f"\nCode Quality Tools Integration Test Results:")
        for tool_name, results in tools_results.items():
            if results["available"]:
                status = "✓ Available"
                if "detected_issues" in results:
                    status += f", Issues detected: {results['detected_issues']}"
                if "tests_passed" in results:
                    status += f", Tests passed: {results['tests_passed']}"
            else:
                status = "✗ Not available"
            print(f"  {tool_name}: {status}")
        
        print(f"\nSummary: {len(available_tools)}/{len(tools_results)} tools are available and working")