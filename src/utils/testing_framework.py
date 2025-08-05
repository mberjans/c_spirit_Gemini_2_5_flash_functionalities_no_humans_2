"""Testing framework utilities for the C-Spirit project.

This module provides convenient wrapper functions and re-exports for common
pytest utilities and time-based testing tools. It encapsulates:

- pytest.raises for exception testing
- pytest.mark.parametrize for parameterized testing
- freezegun.freeze_time for time-based testing
- faker for generating fake test data

The module is designed to provide a consistent interface for testing
utilities across the project while maintaining clean imports and
proper error handling.
"""

import random
import re
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

# Import faker for generating fake test data
try:
    from faker import Faker

    FAKER_AVAILABLE = True
    _faker_instance = Faker()
except ImportError:
    Faker = None
    FAKER_AVAILABLE = False
    _faker_instance = None


def expect_exception(
    exception_type: type[Exception], 
    message_or_match: Union[str, None] = None,
    match: Union[str, None] = None
) -> Any:
    """Wrapper for pytest.raises with improved interface.

    This function provides a clean interface for testing that code raises
    expected exceptions. It's a wrapper around pytest.raises with better
    naming and optional regex matching for exception messages.

    Args:
        exception_type: The type of exception expected to be raised.
        message_or_match: Optional string to match against the exception message.
            Can be passed as positional or keyword argument for backward compatibility.
        match: Optional regex pattern to match against the exception message.
            If provided, the exception message must match this pattern.

    Returns:
        A context manager that can be used with 'with' statement to test
        for exceptions.

    Example:
        Basic usage:
        >>> with expect_exception(ValueError):
        ...     raise ValueError("Invalid input")

        With message matching (positional):
        >>> with expect_exception(ValueError, "Invalid input"):
        ...     raise ValueError("Invalid input")

        With message matching (keyword):
        >>> with expect_exception(ValueError, match=r"Invalid.*input"):
        ...     raise ValueError("Invalid input provided")

        Accessing exception info:
        >>> with expect_exception(ValueError) as exc_info:
        ...     raise ValueError("Test message")
        >>> assert str(exc_info.value) == "Test message"
    """
    # Handle backward compatibility - if message_or_match is provided, use it
    # If both are provided, prefer the match parameter
    effective_match = match or message_or_match
    
    if effective_match is not None:
        return pytest.raises(exception_type, match=re.escape(effective_match))
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


def fake_text(max_nb_chars: int = 200, ext_word_list: list[str] | None = None) -> str:
    """Generate fake text for testing purposes.

    Args:
        max_nb_chars: Maximum number of characters in the generated text.
        ext_word_list: Optional list of words to use for text generation.

    Returns:
        A string of fake text.

    Raises:
        ImportError: If faker is not installed.

    Example:
        >>> text = fake_text(50)
        >>> assert len(text) <= 50
        >>> assert isinstance(text, str)

        With custom word list:
        >>> words = ["metabolite", "enzyme", "pathway", "compound"]
        >>> text = fake_text(100, ext_word_list=words)
        >>> assert any(word in text for word in words)
    """
    if not FAKER_AVAILABLE:
        msg = (
            "faker is required for fake data generation utilities. "
            "Install with: pip install faker"
        )
        raise ImportError(msg)

    return _faker_instance.text(max_nb_chars=max_nb_chars, ext_word_list=ext_word_list)


def fake_entity(entity_type: str = "compound") -> str:
    """Generate a fake entity name for testing purposes.

    Args:
        entity_type: Type of entity to generate. Options include:
            "compound", "enzyme", "pathway", "gene", "protein", "species".

    Returns:
        A fake entity name appropriate for the specified type.

    Raises:
        ImportError: If faker is not installed.

    Example:
        >>> compound = fake_entity("compound")
        >>> assert isinstance(compound, str)

        >>> species = fake_entity("species")
        >>> assert isinstance(species, str)
    """
    if not FAKER_AVAILABLE:
        msg = (
            "faker is required for fake data generation utilities. "
            "Install with: pip install faker"
        )
        raise ImportError(msg)

    # Define entity-specific patterns
    if entity_type == "compound":
        prefixes = ["methyl", "ethyl", "propyl", "butyl", "phenyl", "hydroxy", "amino"]
        suffixes = ["ene", "ane", "ol", "acid", "ester", "amine", "oxide"]
        return f"{random.choice(prefixes)}{random.choice(suffixes)}"
    if entity_type == "enzyme":
        prefixes = ["alpha", "beta", "gamma", "delta"]
        names = ["synthase", "reductase", "oxidase", "transferase", "hydrolase"]
        return f"{random.choice(prefixes)}-{_faker_instance.word()}-{random.choice(names)}"
    if entity_type == "pathway":
        processes = ["biosynthesis", "metabolism", "catabolism", "transport", "signaling"]
        compounds = ["glucose", "fatty acid", "amino acid", "nucleotide", "steroid"]
        return f"{random.choice(compounds)} {random.choice(processes)} pathway"
    if entity_type == "gene":
        return f"{_faker_instance.lexify('???').upper()}{random.randint(1, 999)}"
    if entity_type == "protein":
        domains = ["kinase", "receptor", "transporter", "channel", "binding protein"]
        return f"{_faker_instance.word()} {random.choice(domains)}"
    if entity_type == "species":
        return fake_species_name()
    return f"{entity_type}_{_faker_instance.word()}"


def fake_chemical_name() -> str:
    """Generate a fake chemical compound name for testing.

    Returns:
        A realistic-looking chemical compound name.

    Raises:
        ImportError: If faker is not installed.

    Example:
        >>> chemical = fake_chemical_name()
        >>> assert isinstance(chemical, str)
        >>> assert len(chemical) > 0
    """
    if not FAKER_AVAILABLE:
        msg = (
            "faker is required for fake data generation utilities. "
            "Install with: pip install faker"
        )
        raise ImportError(msg)

    # Common chemical prefixes and suffixes for metabolomics
    prefixes = [
        "acetyl", "methyl", "ethyl", "propyl", "butyl", "pentyl",
        "hexyl", "phenyl", "benzyl", "hydroxy", "amino", "nitro",
        "chloro", "fluoro", "bromo", "iodo", "cyano", "carboxy"
    ]

    suffixes = [
        "acid", "amine", "anol", "ene", "ane", "ester", "ether",
        "oxide", "aldehyde", "ketone", "phenol", "benzene", "pyridine",
        "furan", "thiophene", "imidazole", "pyrazole", "quinoline"
    ]

    # Generate compound name with 1-3 prefixes and 1 suffix
    num_prefixes = random.randint(1, 3)
    selected_prefixes = random.sample(prefixes, num_prefixes)
    suffix = random.choice(suffixes)

    if num_prefixes == 1:
        return f"{selected_prefixes[0]}{suffix}"
    prefix_part = "-".join(selected_prefixes)
    return f"{prefix_part}-{suffix}"


def fake_species_name() -> str:
    """Generate a fake species name in binomial nomenclature format.

    Returns:
        A fake species name in the format "Genus species".

    Raises:
        ImportError: If faker is not installed.

    Example:
        >>> species = fake_species_name()
        >>> assert isinstance(species, str)
        >>> parts = species.split()
        >>> assert len(parts) == 2
        >>> assert parts[0].istitle()  # Genus should be capitalized
        >>> assert parts[1].islower()  # species should be lowercase
    """
    if not FAKER_AVAILABLE:
        msg = (
            "faker is required for fake data generation utilities. "
            "Install with: pip install faker"
        )
        raise ImportError(msg)

    # Common plant genus patterns
    genus_patterns = [
        "Arabidopsis", "Solanum", "Nicotiana", "Medicago", "Glycine",
        "Oryza", "Zea", "Triticum", "Hordeum", "Populus", "Eucalyptus",
        "Brassica", "Lycopersicon", "Phaseolus", "Pisum", "Vicia"
    ]

    # Generate genus (use pattern or fake word)
    if random.choice([True, False]):
        genus = random.choice(genus_patterns)
    else:
        genus = _faker_instance.word().capitalize()

    # Generate species epithet
    species_suffixes = ["ensis", "iana", "icus", "osa", "ata", "alis", "oides", "ella"]
    species = _faker_instance.word().lower() + random.choice(species_suffixes)

    return f"{genus} {species}"


def fake_metabolite_id() -> str:
    """Generate a fake metabolite identifier for testing.

    Returns:
        A fake metabolite ID in a realistic format.

    Raises:
        ImportError: If faker is not installed.

    Example:
        >>> metabolite_id = fake_metabolite_id()
        >>> assert isinstance(metabolite_id, str)
        >>> assert len(metabolite_id) > 0
    """
    if not FAKER_AVAILABLE:
        msg = (
            "faker is required for fake data generation utilities. "
            "Install with: pip install faker"
        )
        raise ImportError(msg)

    # Common metabolite ID patterns
    patterns = [
        f"HMDB{random.randint(10000, 99999)}",  # Human Metabolome Database
        f"CHEBI:{random.randint(1000, 99999)}",  # Chemical Entities of Biological Interest
        f"KEGG:C{random.randint(10000, 99999):05d}",  # KEGG Compound
        f"CAS:{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1, 9)}",  # CAS Registry Number
        f"PUBCHEM:{random.randint(100000, 999999)}",  # PubChem CID
    ]

    return random.choice(patterns)


def fake_experimental_condition() -> dict[str, Any]:
    """Generate fake experimental condition data for testing.

    Returns:
        A dictionary containing fake experimental condition parameters.

    Raises:
        ImportError: If faker is not installed.

    Example:
        >>> condition = fake_experimental_condition()
        >>> assert isinstance(condition, dict)
        >>> assert "temperature" in condition
        >>> assert "ph" in condition
        >>> assert "treatment" in condition
    """
    if not FAKER_AVAILABLE:
        msg = (
            "faker is required for fake data generation utilities. "
            "Install with: pip install faker"
        )
        raise ImportError(msg)

    treatments = [
        "control", "drought_stress", "salt_stress", "heat_stress", "cold_stress",
        "light_stress", "nutrient_deficiency", "pathogen_infection", "hormone_treatment"
    ]

    return {
        "treatment": random.choice(treatments),
        "temperature": round(random.uniform(15.0, 35.0), 1),  # Celsius
        "ph": round(random.uniform(5.5, 8.5), 1),
        "humidity": round(random.uniform(40.0, 90.0), 1),  # Percentage
        "light_intensity": random.randint(100, 1000),  # µmol/m²/s
        "duration_hours": random.randint(1, 168),  # 1 hour to 1 week
        "replicate": random.randint(1, 10),
    }


def fake_plant_anatomy_term() -> str:
    """Generate a fake plant anatomical structure term for testing.

    Returns:
        A fake plant anatomy term.

    Raises:
        ImportError: If faker is not installed.

    Example:
        >>> anatomy_term = fake_plant_anatomy_term()
        >>> assert isinstance(anatomy_term, str)
        >>> assert len(anatomy_term) > 0
    """
    if not FAKER_AVAILABLE:
        msg = (
            "faker is required for fake data generation utilities. "
            "Install with: pip install faker"
        )
        raise ImportError(msg)

    anatomy_terms = [
        "leaf", "root", "stem", "flower", "seed", "fruit", "bark",
        "epidermis", "mesophyll", "xylem", "phloem", "cambium",
        "petal", "sepal", "stamen", "pistil", "ovary", "anther",
        "cotyledon", "endosperm", "pericarp", "trichome", "stomata",
        "guard cell", "palisade mesophyll", "spongy mesophyll",
        "root hair", "root cap", "apical meristem", "node", "internode"
    ]

    return random.choice(anatomy_terms)


def fake_molecular_trait() -> dict[str, Any]:
    """Generate fake molecular trait data for testing.

    Returns:
        A dictionary containing fake molecular trait information.

    Raises:
        ImportError: If faker is not installed.

    Example:
        >>> trait = fake_molecular_trait()
        >>> assert isinstance(trait, dict)
        >>> assert "trait_name" in trait
        >>> assert "value" in trait
        >>> assert "unit" in trait
    """
    if not FAKER_AVAILABLE:
        msg = (
            "faker is required for fake data generation utilities. "
            "Install with: pip install faker"
        )
        raise ImportError(msg)

    trait_types = [
        ("protein_concentration", "mg/g", (0.1, 50.0)),
        ("enzyme_activity", "units/mg", (0.01, 100.0)),
        ("gene_expression", "FPKM", (0.1, 1000.0)),
        ("metabolite_concentration", "µmol/g", (0.001, 10.0)),
        ("antioxidant_capacity", "µmol TE/g", (1.0, 100.0)),
        ("chlorophyll_content", "mg/g", (0.1, 5.0)),
        ("sugar_content", "% dry weight", (1.0, 25.0)),
    ]

    trait_name, unit, (min_val, max_val) = random.choice(trait_types)

    return {
        "trait_name": trait_name,
        "value": round(random.uniform(min_val, max_val), 3),
        "unit": unit,
        "measurement_method": _faker_instance.word(),
        "tissue_type": fake_plant_anatomy_term(),
        "developmental_stage": random.choice(["seedling", "vegetative", "flowering", "fruiting", "senescent"]),
    }


# Convenience re-exports for direct access to underlying utilities
# This allows users to import the original functions if needed
pytest_parametrize = pytest.mark.parametrize
pytest_mark = pytest.mark
freezegun_freeze = freezegun_freeze_time if FREEZEGUN_AVAILABLE else None
faker_instance = _faker_instance if FAKER_AVAILABLE else None


def get_testing_framework_info() -> dict[str, str]:
    """Get information about the testing framework and its dependencies.

    Returns:
        A dictionary containing version information for the testing
        framework components.

    Example:
        >>> info = get_testing_framework_info()
        >>> print(f"pytest version: {info['pytest']}")
        >>> print(f"freezegun version: {info['freezegun']}")
        >>> print(f"faker version: {info['faker']}")
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

    # Get faker version
    if FAKER_AVAILABLE:
        try:
            import faker

            info["faker"] = faker.__version__
        except AttributeError:
            info["faker"] = "unknown"
    else:
        info["faker"] = "not installed"

    # Get Python version
    info["python"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    return info


# Export all public functions and utilities
__all__ = [
    "expect_exception",
    "fake_chemical_name",
    "fake_entity",
    "fake_experimental_condition",
    "fake_metabolite_id",
    "fake_molecular_trait",
    "fake_plant_anatomy_term",
    "fake_species_name",
    "fake_text",
    "faker_instance",
    "freeze_time",
    "freezegun_freeze",
    "get_testing_framework_info",
    "parametrize",
    "pytest_mark",
    "pytest_parametrize",
]
