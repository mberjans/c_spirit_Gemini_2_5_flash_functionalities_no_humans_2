"""Testing framework utilities for the C-Spirit project.

This module provides convenient wrapper functions and re-exports for common
pytest utilities and time-based testing tools. It encapsulates:

- pytest.raises for exception testing
- pytest.mark.parametrize for parameterized testing
- freezegun.freeze_time for time-based testing

The module is designed to provide a consistent interface for testing
utilities across the project while maintaining clean imports and
proper error handling.
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Union

# Import pytest utilities
try:
    import pytest
except ImportError as e:
    msg = (
        "pytest is required for testing framework utilities. "
        "Install with: pip install pytest"
    )
    raise ImportError(msg) from e

# Import freezegun for time-based testing
try:
    from freezegun import freeze_time as freezegun_freeze_time

    FREEZEGUN_AVAILABLE = True
except ImportError:
    freezegun_freeze_time = None
    FREEZEGUN_AVAILABLE = False


def expect_exception(
    exception_type: type[Exception], match: Union[str, None] = None
) -> Any:
    """Wrapper for pytest.raises with improved interface.

    This function provides a clean interface for testing that code raises
    expected exceptions. It's a wrapper around pytest.raises with better
    naming and optional regex matching for exception messages.

    Args:
        exception_type: The type of exception expected to be raised.
        match: Optional regex pattern to match against the exception message.
            If provided, the exception message must match this pattern.

    Returns:
        A context manager that can be used with 'with' statement to test
        for exceptions.

    Example:
        Basic usage:
        >>> with expect_exception(ValueError):
        ...     raise ValueError("Invalid input")

        With message matching:
        >>> with expect_exception(ValueError, match=r"Invalid.*input"):
        ...     raise ValueError("Invalid input provided")

        Accessing exception info:
        >>> with expect_exception(ValueError) as exc_info:
        ...     raise ValueError("Test message")
        >>> assert str(exc_info.value) == "Test message"
    """
    if match is not None:
        return pytest.raises(exception_type, match=match)
    return pytest.raises(exception_type)


def parametrize(argnames: str, argvalues: Any, **kwargs: Any) -> Callable:
    """Wrapper for pytest.mark.parametrize with improved interface.

    This function provides a clean interface for parameterized testing,
    wrapping pytest.mark.parametrize with consistent naming and additional
    validation.

    Args:
        argnames: A string containing comma-separated argument names,
            or a list/tuple of argument names.
        argvalues: The list of argument value tuples for the parameters.
        **kwargs: Additional keyword arguments passed to pytest.mark.parametrize.
            Common options include:
            - ids: List of test IDs for each parameter set
            - indirect: Mark parameters as indirect (fixture names)

    Returns:
        A decorator function that can be applied to test functions.

    Example:
        Basic parameterization:
        >>> @parametrize("input,expected", [
        ...     (1, 2),
        ...     (2, 4),
        ...     (3, 6)
        ... ])
        ... def test_double(input, expected):
        ...     assert input * 2 == expected

        With custom test IDs:
        >>> @parametrize("value", [1, 2, 3], ids=["one", "two", "three"])
        ... def test_positive(value):
        ...     assert value > 0
    """
    # Validate argnames
    if not argnames:
        msg = "argnames cannot be empty"
        raise ValueError(msg)

    # Validate argvalues
    if not argvalues:
        msg = "argvalues cannot be empty"
        raise ValueError(msg)

    return pytest.mark.parametrize(argnames, argvalues, **kwargs)


@contextmanager
def freeze_time(
    time_to_freeze: Union[str, datetime, None] = None, **kwargs: Any
) -> Iterator[Any]:
    """Wrapper for freezegun.freeze_time with improved interface.

    This function provides a clean interface for time-based testing,
    allowing you to freeze time at a specific moment for consistent
    testing of time-dependent code.

    Args:
        time_to_freeze: The time to freeze at. Can be:
            - A string in ISO format (e.g., "2023-01-01 12:00:00")
            - A datetime object
            - None to freeze at the current time
        **kwargs: Additional keyword arguments passed to freezegun.freeze_time.
            Common options include:
            - tz_offset: Timezone offset in hours
            - ignore: List of modules to ignore when freezing time
            - tick: Whether time should tick forward normally

    Yields:
        The frozen time object that can be used to manipulate time
        during the test.

    Raises:
        ImportError: If freezegun is not installed.

    Example:
        Basic time freezing:
        >>> with freeze_time("2023-01-01 12:00:00"):
        ...     from datetime import datetime
        ...     assert datetime.now().year == 2023

        Using the frozen time object:
        >>> with freeze_time("2023-01-01") as frozen_time:
        ...     # Test initial state
        ...     assert datetime.now().day == 1
        ...     # Move time forward
        ...     frozen_time.tick(delta=timedelta(days=1))
        ...     assert datetime.now().day == 2

        Freezing at current time:
        >>> with freeze_time() as frozen_time:
        ...     initial_time = datetime.now()
        ...     # Time is frozen, so this will be the same
        ...     later_time = datetime.now()
        ...     assert initial_time == later_time
    """
    if not FREEZEGUN_AVAILABLE:
        msg = (
            "freezegun is required for time-based testing utilities. "
            "Install with: pip install freezegun"
        )
        raise ImportError(msg)

    with freezegun_freeze_time(time_to_freeze, **kwargs) as frozen_time:
        yield frozen_time


# Convenience re-exports for direct access to underlying utilities
# This allows users to import the original functions if needed
pytest_parametrize = pytest.mark.parametrize
pytest_mark = pytest.mark
freezegun_freeze = freezegun_freeze_time if FREEZEGUN_AVAILABLE else None


def get_testing_framework_info() -> dict[str, str]:
    """Get information about the testing framework and its dependencies.

    Returns:
        A dictionary containing version information for the testing
        framework components.

    Example:
        >>> info = get_testing_framework_info()
        >>> print(f"pytest version: {info['pytest']}")
        >>> print(f"freezegun version: {info['freezegun']}")
    """
    info = {}

    # Get pytest version
    try:
        info["pytest"] = pytest.__version__
    except AttributeError:
        info["pytest"] = "unknown"

    # Get freezegun version
    if FREEZEGUN_AVAILABLE:
        try:
            import freezegun

            info["freezegun"] = freezegun.__version__
        except AttributeError:
            info["freezegun"] = "unknown"
    else:
        info["freezegun"] = "not installed"

    # Get Python version
    info["python"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    return info


# Export all public functions and utilities
__all__ = [
    "expect_exception",
    "freeze_time",
    "freezegun_freeze",
    "get_testing_framework_info",
    "parametrize",
    "pytest_mark",
    "pytest_parametrize",
]
