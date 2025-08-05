"""
Ontology Reasoning Integration Module

This module integrates Owlready2's reasoning capabilities (HermiT/Pellet) to infer
new facts and reclassify instances/classes based on defined relationships.

Key Features:
- Integration with HermiT/Pellet reasoners
- Inference of new class memberships
- Optional property value inference
- Handling of inconsistent ontologies
- Error handling and validation

Dependencies:
- Owlready2 for reasoning capabilities
- Java runtime for HermiT/Pellet execution
"""

import logging
from typing import Any, Optional
from owlready2 import sync_reasoner, OwlReadyInconsistentOntologyError, Ontology


# Set up logging
logger = logging.getLogger(__name__)


class ReasonerError(Exception):
    """Custom exception for reasoning-related errors."""
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        """
        Initialize ReasonerError.
        
        Args:
            message: Error message
            cause: Optional underlying exception that caused this error
        """
        super().__init__(message)
        self.cause = cause
        if cause:
            self.message = f"{message}. Caused by: {str(cause)}"
        else:
            self.message = message
    
    def __str__(self):
        return self.message


def run_reasoner(ontology: Any, infer_property_values: bool = False) -> bool:
    """
    Run reasoning on the given ontology using Owlready2's sync_reasoner.
    
    This function uses HermiT/Pellet reasoner to infer new facts and reclassify
    instances/classes based on defined relationships in the ontology.
    
    Args:
        ontology: Owlready2 ontology object to run reasoning on
        infer_property_values: Whether to infer property values during reasoning
    
    Returns:
        bool: True if reasoning completed successfully, False otherwise
    
    Raises:
        ReasonerError: If reasoning fails or ontology is invalid/inconsistent
    
    Example:
        >>> from owlready2 import get_ontology
        >>> onto = get_ontology("path/to/ontology.owl").load()
        >>> success = run_reasoner(onto, infer_property_values=True)
        >>> if success:
        ...     print("Reasoning completed successfully")
    """
    # Validate input parameters
    if ontology is None:
        raise ReasonerError("Ontology cannot be None")
    
    if not hasattr(ontology, 'classes') or not hasattr(ontology, 'individuals'):
        raise ReasonerError("Invalid ontology object: must be an Owlready2 ontology")
    
    try:
        logger.info(f"Starting reasoning on ontology: {ontology.base_iri}")
        logger.debug(f"Infer property values: {infer_property_values}")
        
        # Configure reasoner parameters
        reasoner_params = {}
        if infer_property_values:
            reasoner_params['infer_property_values'] = True
        
        # Run the reasoner
        with ontology:
            if infer_property_values:
                # Use sync_reasoner with property value inference
                sync_reasoner(infer_property_values=True)
            else:
                # Use sync_reasoner without property value inference
                sync_reasoner()
        
        logger.info("Reasoning completed successfully")
        return True
        
    except OwlReadyInconsistentOntologyError as e:
        error_msg = f"Ontology is inconsistent and cannot be reasoned over: {str(e)}"
        logger.error(error_msg)
        raise ReasonerError(error_msg, e)
    
    except Exception as e:
        error_msg = f"Reasoning failed due to unexpected error: {str(e)}"
        logger.error(error_msg)
        raise ReasonerError(error_msg, e)


def validate_java_configuration() -> bool:
    """
    Validate that Java is properly configured for HermiT/Pellet execution.
    
    Returns:
        bool: True if Java configuration is valid, False otherwise
    
    Note:
        This function checks if the Java executable path is correctly configured
        for Owlready2 to find HermiT/Pellet reasoners.
    """
    try:
        import subprocess
        import sys
        
        # Try to run java -version to check if Java is available
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        
        if result.returncode == 0:
            logger.info("Java is available for reasoning")
            return True
        else:
            logger.warning("Java is not available or not properly configured")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
        logger.warning(f"Could not validate Java configuration: {str(e)}")
        return False


def get_inferred_classes(ontology: Any, individual: Any) -> list:
    """
    Get all inferred classes for a given individual after reasoning.
    
    Args:
        ontology: Owlready2 ontology object
        individual: Individual instance to get inferred classes for
    
    Returns:
        list: List of inferred classes for the individual
    
    Raises:
        ReasonerError: If individual is invalid or reasoning hasn't been run
    """
    if ontology is None or individual is None:
        raise ReasonerError("Ontology and individual cannot be None")
    
    try:
        # Get all classes the individual belongs to (including inferred ones)
        inferred_classes = list(individual.is_a)
        logger.debug(f"Individual {individual} belongs to classes: {inferred_classes}")
        return inferred_classes
        
    except Exception as e:
        error_msg = f"Failed to get inferred classes for individual {individual}: {str(e)}"
        logger.error(error_msg)
        raise ReasonerError(error_msg, e)


def get_inferred_properties(ontology: Any, individual: Any) -> dict:
    """
    Get all inferred property values for a given individual after reasoning.
    
    Args:
        ontology: Owlready2 ontology object
        individual: Individual instance to get inferred properties for
    
    Returns:
        dict: Dictionary mapping property names to their values
    
    Raises:
        ReasonerError: If individual is invalid or reasoning hasn't been run
    """
    if ontology is None or individual is None:
        raise ReasonerError("Ontology and individual cannot be None")
    
    try:
        inferred_properties = {}
        
        # Get all properties defined in the ontology
        for prop in ontology.properties():
            prop_name = prop.name
            if hasattr(individual, prop_name):
                prop_value = getattr(individual, prop_name)
                if prop_value:  # Only include non-empty values
                    inferred_properties[prop_name] = prop_value
        
        logger.debug(f"Individual {individual} has properties: {inferred_properties}")
        return inferred_properties
        
    except Exception as e:
        error_msg = f"Failed to get inferred properties for individual {individual}: {str(e)}"
        logger.error(error_msg)
        raise ReasonerError(error_msg, e)


def check_ontology_consistency(ontology: Any) -> bool:
    """
    Check if an ontology is consistent by attempting to run reasoning.
    
    Args:
        ontology: Owlready2 ontology object to check
    
    Returns:
        bool: True if ontology is consistent, False if inconsistent
    
    Note:
        This function attempts reasoning and catches inconsistency errors.
        It's a non-destructive way to check consistency.
    """
    if ontology is None:
        logger.warning("Cannot check consistency of None ontology")
        return False
    
    try:
        # Attempt reasoning to check consistency
        run_reasoner(ontology, infer_property_values=False)
        logger.info("Ontology is consistent")
        return True
        
    except ReasonerError as e:
        if "inconsistent" in str(e).lower():
            logger.warning(f"Ontology is inconsistent: {str(e)}")
            return False
        else:
            # Other reasoning errors don't necessarily mean inconsistency
            logger.error(f"Could not determine consistency due to error: {str(e)}")
            raise e


def setup_reasoner_environment() -> bool:
    """
    Set up the reasoning environment and validate configuration.
    
    Returns:
        bool: True if environment is properly set up, False otherwise
    
    Note:
        This function should be called before running reasoning to ensure
        proper configuration of Java and reasoner paths.
    """
    try:
        # Validate Java configuration
        java_ok = validate_java_configuration()
        if not java_ok:
            logger.warning("Java configuration validation failed")
            return False
        
        # Additional setup could be added here (e.g., setting JAVA_HOME)
        logger.info("Reasoner environment setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to set up reasoner environment: {str(e)}")
        return False


# Module initialization
logger.info("Ontology reasoner module loaded successfully")