"""Utility modules for the C-Spirit project.

This package contains utility modules that provide common functionality
across the C-Spirit project, including testing frameworks and helper functions.
"""

# Version information
__version__ = "0.1.0"

# Import main utilities to make them available at package level
from .testing_framework import (
    expect_exception,
    freeze_time,
    parametrize,
)

__all__ = [
    "expect_exception",
    "freeze_time",
    "parametrize",
]
