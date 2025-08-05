"""
Ontology scheme structural module for plant metabolomics.

This module defines and integrates terms for "Structural Annotation" including
Chemont classification, NP Classifier, and Plant Metabolic Network (PMN) into
a core ontology using Owlready2. It provides functions for creating, managing,
and validating structural annotation classes within OWL 2.0 ontologies.

The module supports:
- ChemontClass creation for chemical entity classification
- NPClass creation for natural product classification  
- PMNCompound creation for plant metabolic compounds
- Hierarchical relationships between classes
- Batch operations for multiple class creation
- Comprehensive validation and error handling
- Thread-safe operations

All created classes inherit from owlready2.Thing and include proper OWL/RDF
annotations (label, comment) for semantic interoperability.
"""

import logging
import re
import threading
from typing import Any, Dict, List, Optional, Set, Union

from owlready2 import Thing, OwlReadyError, types as owlready_types, get_ontology


# Configure logging
logger = logging.getLogger(__name__)

# Thread lock for thread-safe operations
_creation_lock = threading.Lock()


class StructuralClassError(Exception):
    """Custom exception for structural class operations.
    
    This exception is raised when errors occur during the creation,
    validation, or manipulation of structural annotation classes.
    
    Args:
        message: Error description
        
    Example:
        raise StructuralClassError("Invalid ontology provided")
    """
    
    def __init__(self, message: str) -> None:
        """Initialize the structural class error.
        
        Args:
            message: Error description
        """
        super().__init__(message)
        self.message = message


def _validate_class_name(class_name: str) -> None:
    """Validate that a class name follows OWL naming conventions.
    
    Args:
        class_name: Name of the class to validate
        
    Raises:
        StructuralClassError: If class name is invalid
    """
    if not class_name or not isinstance(class_name, str):
        raise StructuralClassError("Invalid class name: must be a non-empty string")
    
    # Remove leading/trailing whitespace
    class_name = class_name.strip()
    
    if not class_name:
        raise StructuralClassError("Invalid class name: cannot be empty or whitespace only")
    
    # Check for valid OWL class name pattern
    if not re.match(r'^[A-Za-z][A-Za-z0-9_]*$', class_name):
        raise StructuralClassError(
            f"Invalid class name '{class_name}': must start with letter and "
            "contain only letters, numbers, and underscores"
        )


def _validate_ontology(ontology: Any) -> None:
    """Validate that the provided ontology is valid.
    
    Args:
        ontology: Ontology object to validate
        
    Raises:
        StructuralClassError: If ontology is invalid
    """
    if ontology is None:
        raise StructuralClassError("Invalid ontology: cannot be None")


def create_chemont_class(ontology: Any, class_name: str) -> Any:
    """Create a ChemontClass for chemical entity classification.
    
    Creates a new OWL class for chemical entity classification based on the
    ChEMONT (Chemical Entities of Biological Interest) ontology. The class
    inherits from owlready2.Thing and includes appropriate semantic annotations.
    
    Args:
        ontology: Target ontology for class creation
        class_name: Name of the ChemontClass to create
        
    Returns:
        The created ChemontClass object
        
    Raises:
        StructuralClassError: If creation fails due to invalid parameters or Owlready2 errors
        
    Example:
        chemont_class = create_chemont_class(ontology, "ChemontClass")
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    
    try:
        with _creation_lock:
            # Get the ontology namespace
            namespace = ontology.get_namespace()
            
            # Create the ChemontClass
            chemont_class = owlready_types.new_class(
                class_name,
                (Thing,),
                namespace
            )
            
            # Add semantic annotations
            chemont_class.label = [f"Chemical Entity Class (Chemont)"]
            chemont_class.comment = [
                "Base class for chemical entity classification based on ChEMONT ontology"
            ]
            
            logger.info(f"Created ChemontClass: {class_name}")
            return chemont_class
            
    except OwlReadyError as e:
        raise StructuralClassError(f"Owlready2 error creating ChemontClass '{class_name}': {e}")
    except Exception as e:
        raise StructuralClassError(f"Failed to create ChemontClass '{class_name}': {e}")


def create_np_class(ontology: Any, class_name: str) -> Any:
    """Create an NPClass for natural product classification.
    
    Creates a new OWL class for natural product classification based on the
    NP Classifier system. The class inherits from owlready2.Thing and includes
    appropriate semantic annotations for natural product categorization.
    
    Args:
        ontology: Target ontology for class creation
        class_name: Name of the NPClass to create
        
    Returns:
        The created NPClass object
        
    Raises:
        StructuralClassError: If creation fails due to invalid parameters or Owlready2 errors
        
    Example:
        np_class = create_np_class(ontology, "NPClass")
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    
    try:
        with _creation_lock:
            # Get the ontology namespace
            namespace = ontology.get_namespace()
            
            # Create the NPClass
            np_class = owlready_types.new_class(
                class_name,
                (Thing,),
                namespace
            )
            
            # Add semantic annotations
            np_class.label = [f"Natural Product Class"]
            np_class.comment = [
                "Base class for natural product classification based on NP Classifier"
            ]
            
            logger.info(f"Created NPClass: {class_name}")
            return np_class
            
    except OwlReadyError as e:
        raise StructuralClassError(f"Owlready2 error creating NPClass '{class_name}': {e}")
    except Exception as e:
        raise StructuralClassError(f"Failed to create NPClass '{class_name}': {e}")


def create_pmn_compound(ontology: Any, class_name: str) -> Any:
    """Create a PMNCompound for plant metabolic compounds.
    
    Creates a new OWL class for plant metabolic compounds from the Plant
    Metabolic Network (PMN) database. The class inherits from owlready2.Thing
    and includes appropriate semantic annotations for plant metabolite classification.
    
    Args:
        ontology: Target ontology for class creation
        class_name: Name of the PMNCompound to create
        
    Returns:
        The created PMNCompound object
        
    Raises:
        StructuralClassError: If creation fails due to invalid parameters or Owlready2 errors
        
    Example:
        pmn_compound = create_pmn_compound(ontology, "PMNCompound")
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    
    try:
        with _creation_lock:
            # Get the ontology namespace
            namespace = ontology.get_namespace()
            
            # Create the PMNCompound
            pmn_compound = owlready_types.new_class(
                class_name,
                (Thing,),
                namespace
            )
            
            # Add semantic annotations
            pmn_compound.label = [f"Plant Metabolic Network Compound"]
            pmn_compound.comment = [
                "Base class for plant metabolic compounds from PMN database"
            ]
            
            logger.info(f"Created PMNCompound: {class_name}")
            return pmn_compound
            
    except OwlReadyError as e:
        raise StructuralClassError(f"Owlready2 error creating PMNCompound '{class_name}': {e}")
    except Exception as e:
        raise StructuralClassError(f"Failed to create PMNCompound '{class_name}': {e}")


def create_np_class_with_parent(ontology: Any, class_name: str, parent_class_name: str) -> Any:
    """Create an NPClass with hierarchical relationship to a parent class.
    
    Creates a new NPClass that inherits from both Thing and a specified parent class,
    establishing hierarchical relationships in the ontology structure.
    
    Args:
        ontology: Target ontology for class creation
        class_name: Name of the NPClass to create
        parent_class_name: Name of the parent class to inherit from
        
    Returns:
        The created NPClass object with parent relationship
        
    Raises:
        StructuralClassError: If creation fails or parent class not found
        
    Example:
        np_class = create_np_class_with_parent(ontology, "NPClass", "ChemicalClass")
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    _validate_class_name(parent_class_name)
    
    try:
        with _creation_lock:
            # Find the parent class
            parent_class = ontology.search_one(iri=f"*{parent_class_name}")
            if not parent_class:
                raise StructuralClassError(f"Parent class '{parent_class_name}' not found")
            
            # Get the ontology namespace
            namespace = ontology.get_namespace()
            
            # Create the NPClass with parent relationship
            np_class = owlready_types.new_class(
                class_name,
                (parent_class,),
                namespace
            )
            
            # Add semantic annotations
            np_class.label = [f"Natural Product Class"]
            np_class.comment = [
                "Base class for natural product classification based on NP Classifier"
            ]
            
            logger.info(f"Created NPClass with parent: {class_name} -> {parent_class_name}")
            return np_class
            
    except OwlReadyError as e:
        raise StructuralClassError(f"Owlready2 error creating NPClass with parent '{class_name}': {e}")
    except Exception as e:
        raise StructuralClassError(f"Failed to create NPClass with parent '{class_name}': {e}")


def verify_class_accessibility(ontology: Any, class_name: str) -> bool:
    """Verify that a class is accessible in the ontology.
    
    Checks if a class can be found and accessed within the ontology structure,
    ensuring proper integration and availability for further operations.
    
    Args:
        ontology: Ontology to search within
        class_name: Name of the class to verify
        
    Returns:
        True if class is accessible, False otherwise
        
    Example:
        is_accessible = verify_class_accessibility(ontology, "ChemontClass")
    """
    try:
        _validate_ontology(ontology)
        _validate_class_name(class_name)
        
        # Search for the class in the ontology
        found_class = ontology.search_one(iri=f"*{class_name}")
        return found_class is not None
        
    except Exception as e:
        logger.warning(f"Error verifying class accessibility for '{class_name}': {e}")
        return False


def get_class_hierarchy_depth(ontology: Any, class_name: str) -> int:
    """Calculate the hierarchy depth of a class from Thing.
    
    Traverses the class hierarchy to determine how many levels the specified
    class is below the root Thing class.
    
    Args:
        ontology: Ontology containing the class
        class_name: Name of the class to analyze
        
    Returns:
        Depth level (0 for Thing, 1 for direct Thing subclass, etc.)
        
    Raises:
        StructuralClassError: If class not found
        
    Example:
        depth = get_class_hierarchy_depth(ontology, "NPClass")  # Returns 2 if NPClass -> ChemicalClass -> Thing
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    
    try:
        # Find the class
        target_class = ontology.search_one(iri=f"*{class_name}")
        if not target_class:
            raise StructuralClassError(f"Class '{class_name}' not found in ontology")
        
        # Calculate depth by traversing parent classes
        depth = 0
        current_class = target_class
        
        while current_class and current_class is not Thing:
            if hasattr(current_class, 'is_a') and current_class.is_a:
                # Get the first parent class
                parent_classes = [cls for cls in current_class.is_a if cls is not Thing]
                if parent_classes:
                    current_class = parent_classes[0]
                    depth += 1
                else:
                    # Direct child of Thing
                    depth += 1
                    break
            else:
                break
        
        return depth
        
    except Exception as e:
        raise StructuralClassError(f"Failed to calculate hierarchy depth for '{class_name}': {e}")


def classify_structural_annotation_type(ontology: Any, class_name: str) -> str:
    """Classify the structural annotation type based on class name.
    
    Determines the type of structural annotation based on the class name patterns
    used in the ontology for different classification systems.
    
    Args:
        ontology: Ontology containing the class
        class_name: Name of the class to classify
        
    Returns:
        Classification category as string
        
    Example:
        category = classify_structural_annotation_type(ontology, "ChemontClass")
        # Returns "structural_classification"
    """
    try:
        _validate_ontology(ontology)
        _validate_class_name(class_name)
        
        # Find the class to verify it exists
        target_class = ontology.search_one(iri=f"*{class_name}")
        if not target_class:
            return "unknown_classification"
        
        # Classify based on name patterns
        name_lower = class_name.lower()
        
        if "chemont" in name_lower:
            return "structural_classification"
        elif "np" in name_lower and "class" in name_lower:
            return "natural_product_classification"
        elif "pmn" in name_lower:
            return "plant_metabolic_classification"
        else:
            return "unknown_classification"
            
    except Exception as e:
        logger.warning(f"Error classifying annotation type for '{class_name}': {e}")
        return "unknown_classification"


def create_structural_classes_batch(ontology: Any, class_specs: List[Dict[str, Any]]) -> List[Any]:
    """Create multiple structural annotation classes in batch.
    
    Efficiently creates multiple structural classes based on provided specifications,
    supporting different class types and parent relationships.
    
    Args:
        ontology: Target ontology for class creation
        class_specs: List of class specifications, each containing:
            - name: Class name
            - type: Class type ("chemont", "np_classifier", "pmn")
            - parent: Optional parent class name
        
    Returns:
        List of created class objects
        
    Raises:
        StructuralClassError: If batch creation fails
        
    Example:
        specs = [
            {"name": "ChemontClass", "type": "chemont", "parent": None},
            {"name": "NPClass", "type": "np_classifier", "parent": "ChemicalClass"}
        ]
        classes = create_structural_classes_batch(ontology, specs)
    """
    _validate_ontology(ontology)
    
    if not class_specs or not isinstance(class_specs, list):
        raise StructuralClassError("Invalid class specifications: must be a non-empty list")
    
    created_classes = []
    
    try:
        for spec in class_specs:
            if not isinstance(spec, dict):
                raise StructuralClassError("Invalid class specification: must be a dictionary")
            
            name = spec.get("name")
            class_type = spec.get("type")
            parent = spec.get("parent")
            
            if not name:
                raise StructuralClassError("Class specification missing 'name' field")
            
            # Create class based on type
            if class_type == "chemont":
                created_class = create_chemont_class(ontology, name)
            elif class_type == "np_classifier":
                if parent:
                    created_class = create_np_class_with_parent(ontology, name, parent)
                else:
                    created_class = create_np_class(ontology, name)
            elif class_type == "pmn":
                created_class = create_pmn_compound(ontology, name)
            else:
                # Default to creating as NPClass
                created_class = create_np_class(ontology, name)
            
            created_classes.append(created_class)
            
        logger.info(f"Successfully created {len(created_classes)} structural classes in batch")
        return created_classes
        
    except Exception as e:
        logger.error(f"Batch class creation failed: {e}")
        raise StructuralClassError(f"Failed to create structural classes in batch: {e}")


def validate_structural_class_properties(ontology: Any, class_name: str) -> bool:
    """Validate that a structural class has required properties.
    
    Checks if a structural annotation class has all required properties including
    proper labels, comments, and inheritance relationships.
    
    Args:
        ontology: Ontology containing the class
        class_name: Name of the class to validate
        
    Returns:
        True if class has all required properties, False otherwise
        
    Example:
        is_valid = validate_structural_class_properties(ontology, "ChemontClass")
    """
    try:
        _validate_ontology(ontology)
        _validate_class_name(class_name)
        
        # Find the class
        target_class = ontology.search_one(iri=f"*{class_name}")
        if not target_class:
            return False
        
        # Check for required properties
        required_properties = ['label', 'comment', 'is_a']
        
        for prop in required_properties:
            if not hasattr(target_class, prop):
                return False
            
            prop_value = getattr(target_class, prop)
            
            # For label and comment, ensure they are not empty
            if prop in ['label', 'comment']:
                if not prop_value or not any(prop_value):
                    return False
            
            # For is_a, ensure it includes Thing or a subclass
            if prop == 'is_a':
                if not prop_value:
                    return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Error validating properties for '{class_name}': {e}")
        return False


def verify_thing_inheritance(ontology: Any, class_name: str) -> bool:
    """Verify that a structural class properly inherits from Thing.
    
    Checks the inheritance chain to ensure the class ultimately inherits from
    owlready2.Thing, either directly or through parent classes.
    
    Args:
        ontology: Ontology containing the class
        class_name: Name of the class to verify
        
    Returns:
        True if class inherits from Thing, False otherwise
        
    Example:
        inherits_thing = verify_thing_inheritance(ontology, "ChemontClass")
    """
    try:
        _validate_ontology(ontology)
        _validate_class_name(class_name)
        
        # Find the class
        target_class = ontology.search_one(iri=f"*{class_name}")
        if not target_class:
            return False
        
        # Check inheritance chain
        def _check_thing_inheritance(cls) -> bool:
            if not hasattr(cls, 'is_a') or not cls.is_a:
                return False
            
            for parent in cls.is_a:
                if parent is Thing:
                    return True
                # Recursively check parent classes
                if _check_thing_inheritance(parent):
                    return True
            
            return False
        
        return _check_thing_inheritance(target_class)
        
    except Exception as e:
        logger.warning(f"Error verifying Thing inheritance for '{class_name}': {e}")
        return False


def get_all_structural_classes(ontology: Any) -> List[Any]:
    """Get all structural annotation classes from the ontology.
    
    Retrieves all classes that match structural annotation patterns,
    including ChemontClass, NPClass, and PMNCompound types.
    
    Args:
        ontology: Ontology to search
        
    Returns:
        List of structural annotation class objects
        
    Example:
        structural_classes = get_all_structural_classes(ontology)
    """
    try:
        _validate_ontology(ontology)
        
        # Get all classes from the ontology
        all_classes = list(ontology.classes())
        
        # Filter for structural annotation classes
        structural_classes = []
        structural_patterns = ['chemont', 'npclass', 'pmn', 'structural']
        
        for cls in all_classes:
            if hasattr(cls, 'name') and cls.name:
                name_lower = cls.name.lower()
                if any(pattern in name_lower for pattern in structural_patterns):
                    structural_classes.append(cls)
        
        return structural_classes
        
    except Exception as e:
        logger.error(f"Error retrieving structural classes: {e}")
        return []


def validate_class_metadata(ontology: Any, class_name: str) -> bool:
    """Validate class metadata and annotations.
    
    Performs comprehensive validation of class metadata including labels,
    comments, IRI structure, and custom annotations specific to structural
    annotation classes.
    
    Args:
        ontology: Ontology containing the class
        class_name: Name of the class to validate
        
    Returns:
        True if metadata is valid, False otherwise
        
    Example:
        metadata_valid = validate_class_metadata(ontology, "ChemontClass")
    """
    try:
        _validate_ontology(ontology)
        _validate_class_name(class_name)
        
        # Find the class
        target_class = ontology.search_one(iri=f"*{class_name}")
        if not target_class:
            return False
        
        # Validate basic metadata
        if not validate_structural_class_properties(ontology, class_name):
            return False
        
        # Validate IRI structure if present
        if hasattr(target_class, 'iri') and target_class.iri:
            iri = target_class.iri
            if not isinstance(iri, str) or not iri.startswith('http'):
                return False
        
        # Additional metadata checks can be added here
        # For now, basic validation is sufficient
        
        return True
        
    except Exception as e:
        logger.warning(f"Error validating metadata for '{class_name}': {e}")
        return False


def cleanup_structural_classes(ontology: Any) -> int:
    """Cleanup structural annotation classes from the ontology.
    
    Removes all structural annotation classes from the ontology, useful for
    cleanup operations or resetting the ontology state.
    
    Args:
        ontology: Ontology to clean up
        
    Returns:
        Number of classes removed
        
    Warning:
        This operation is destructive and cannot be undone
        
    Example:
        removed_count = cleanup_structural_classes(ontology)
    """
    try:
        _validate_ontology(ontology)
        
        # Get all structural classes
        structural_classes = get_all_structural_classes(ontology)
        
        cleanup_count = 0
        with _creation_lock:
            for cls in structural_classes:
                try:
                    if hasattr(cls, 'destroy'):
                        cls.destroy()
                        cleanup_count += 1
                except Exception as e:
                    logger.warning(f"Failed to destroy class {cls.name}: {e}")
        
        logger.info(f"Cleaned up {cleanup_count} structural classes")
        return cleanup_count
        
    except Exception as e:
        logger.error(f"Error during structural class cleanup: {e}")
        return 0


def define_core_structural_classes(ontology: Any) -> Dict[str, Any]:
    """Define core structural annotation classes in the main ontology namespace.
    
    Creates the fundamental structural annotation classes (ChemontClass, NPClass, 
    PMNCompound) that inherit from owlready2.Thing and associates them with the 
    main ontology namespace. This implements the core requirements for 
    AIM2-ODIE-009-T3.
    
    Args:
        ontology: Main ontology to define classes in
        
    Returns:
        Dictionary mapping class names to created class objects
        
    Raises:
        StructuralClassError: If class definition fails
        
    Example:
        classes = define_core_structural_classes(ontology)
        chemont_class = classes['ChemontClass']
        np_class = classes['NPClass']
        pmn_compound = classes['PMNCompound']
    """
    _validate_ontology(ontology)
    
    try:
        with _creation_lock:
            # Use the ontology context for class creation
            with ontology:
                # Define ChemontClass for chemical entity classification
                class ChemontClass(Thing):
                    namespace = ontology
                    
                ChemontClass.label = ["Chemical Entity Class (Chemont)"]
                ChemontClass.comment = [
                    "Base class for chemical entity classification based on ChEMONT (Chemical Entities of Biological Interest) ontology. "
                    "Provides structural annotation for metabolites using chemical classification hierarchies."
                ]
                
                # Define NPClass for natural product classification
                class NPClass(Thing):
                    namespace = ontology
                    
                NPClass.label = ["Natural Product Class"]
                NPClass.comment = [
                    "Base class for natural product classification based on NP Classifier system. "
                    "Provides structural annotation for natural products and secondary metabolites."
                ]
                
                # Define PMNCompound for plant metabolic compounds
                class PMNCompound(Thing):
                    namespace = ontology
                    
                PMNCompound.label = ["Plant Metabolic Network Compound"]
                PMNCompound.comment = [
                    "Base class for plant metabolic compounds from Plant Metabolic Network (PMN) database. "
                    "Provides structural annotation for compounds involved in plant metabolic pathways."
                ]
                
                # Create the class registry
                defined_classes = {
                    'ChemontClass': ChemontClass,
                    'NPClass': NPClass,
                    'PMNCompound': PMNCompound
                }
                
                logger.info(f"Successfully defined {len(defined_classes)} core structural classes")
                
                return defined_classes
            
    except OwlReadyError as e:
        raise StructuralClassError(f"Owlready2 error defining core structural classes: {e}")
    except Exception as e:
        raise StructuralClassError(f"Failed to define core structural classes: {e}")


def establish_structural_hierarchy(ontology: Any, classes: Dict[str, Any]) -> None:
    """Establish hierarchical relationships between structural annotation classes.
    
    Creates is_a relationships and other hierarchical connections between the
    defined structural annotation classes to represent classification hierarchies.
    
    Args:
        ontology: Main ontology containing the classes
        classes: Dictionary of class names to class objects
        
    Raises:
        StructuralClassError: If hierarchy establishment fails
        
    Example:
        classes = define_core_structural_classes(ontology)
        establish_structural_hierarchy(ontology, classes)
    """
    _validate_ontology(ontology)
    
    if not classes or not isinstance(classes, dict):
        raise StructuralClassError("Invalid classes dictionary")
    
    try:
        with _creation_lock:
            # Get required classes
            chemont_class = classes.get('ChemontClass')
            np_class = classes.get('NPClass')
            pmn_compound = classes.get('PMNCompound')
            
            if not all([chemont_class, np_class, pmn_compound]):
                raise StructuralClassError("Missing required structural classes")
            
            # Create a general StructuralAnnotation superclass within the ontology context
            with ontology:
                class StructuralAnnotation(Thing):
                    namespace = ontology
                    
                StructuralAnnotation.label = ["Structural Annotation"]
                StructuralAnnotation.comment = [
                    "Superclass for all structural annotation concepts including chemical classifications, "
                    "natural product categories, and metabolic compound classifications."
                ]
            
            # Establish hierarchical relationships
            # All structural classes inherit from StructuralAnnotation
            chemont_class.is_a.append(StructuralAnnotation)
            np_class.is_a.append(StructuralAnnotation)
            pmn_compound.is_a.append(StructuralAnnotation)
            
            # NPClass and PMNCompound are more specific chemical classifications
            # So they could also inherit from ChemontClass concepts if needed
            # For now, keep them as peer classes under StructuralAnnotation
            
            logger.info("Successfully established structural class hierarchy")
            
    except Exception as e:
        raise StructuralClassError(f"Failed to establish structural hierarchy: {e}")


def add_initial_key_terms(ontology: Any) -> Dict[str, List[Any]]:
    """Add initial key terms/instances from Chemont, NP Classifier, and PMN to the ontology.
    
    Creates representative instances of ChemontClass, NPClass, and PMNCompound classes
    to populate the ontology with initial key terms from each classification system.
    This function implements AIM2-ODIE-009-T4 by adding concrete examples from each
    structural annotation system.
    
    Args:
        ontology: Target ontology for instance creation
        
    Returns:
        Dictionary with keys "chemont_instances", "np_instances", "pmn_instances"
        containing lists of created instances
        
    Raises:
        StructuralClassError: If instance creation fails
        
    Example:
        instances = add_initial_key_terms(ontology)
        benzopyranoids = instances['chemont_instances'][0]
        alkaloids = instances['np_instances'][0]
        glucose = instances['pmn_instances'][0]
    """
    _validate_ontology(ontology)
    
    try:
        with _creation_lock:
            # Get the required classes
            chemont_class = ontology.search_one(iri="*ChemontClass")
            np_class = ontology.search_one(iri="*NPClass")
            pmn_compound = ontology.search_one(iri="*PMNCompound")
            
            if not all([chemont_class, np_class, pmn_compound]):
                raise StructuralClassError(
                    "Required structural classes not found. Please run define_core_structural_classes() first."
                )
            
            # Define representative Chemont chemical entity instances
            chemont_terms = [
                {
                    "name": "Benzopyranoids",
                    "label": "Benzopyranoids",
                    "comment": "Chemical entities containing a benzopyran moiety, including flavonoids and related compounds."
                },
                {
                    "name": "Flavonoids", 
                    "label": "Flavonoids",
                    "comment": "A class of polyphenolic secondary metabolites with antioxidant properties."
                },
                {
                    "name": "Phenolic_compounds",
                    "label": "Phenolic compounds", 
                    "comment": "Chemical compounds containing one or more hydroxyl groups attached to aromatic rings."
                },
                {
                    "name": "Alkaloids_chemont",
                    "label": "Alkaloids (Chemont)",
                    "comment": "Nitrogen-containing basic chemical compounds of plant origin with pharmacological activity."
                },
                {
                    "name": "Organic_acids",
                    "label": "Organic acids",
                    "comment": "Carbon-containing compounds with acidic properties, common in plant metabolism."
                },
                {
                    "name": "Lipids_and_lipid_molecules",
                    "label": "Lipids and lipid-like molecules",
                    "comment": "Hydrophobic or amphiphilic small molecules including fatty acids, glycerolipids, and sterols."
                },
                {
                    "name": "Carbohydrates",
                    "label": "Carbohydrates",
                    "comment": "Polyhydroxy aldehydes or ketones and their derivatives, primary energy storage molecules."
                },
                {
                    "name": "Amino_acids_and_derivatives", 
                    "label": "Amino acids and derivatives",
                    "comment": "Building blocks of proteins and their chemical derivatives involved in metabolism."
                }
            ]
            
            # Define representative NP Classifier natural product instances
            np_terms = [
                {
                    "name": "Alkaloids_np",
                    "label": "Alkaloids (NP)",
                    "comment": "Natural products containing basic nitrogen atoms, often with complex ring structures."
                },
                {
                    "name": "Terpenes",
                    "label": "Terpenes", 
                    "comment": "Large and diverse class of organic compounds derived from isoprene units."
                },
                {
                    "name": "Polyketides",
                    "label": "Polyketides",
                    "comment": "Secondary metabolites produced by polyketide synthases with diverse biological activities."
                },
                {
                    "name": "Phenylpropanoids",
                    "label": "Phenylpropanoids",
                    "comment": "Natural products derived from phenylalanine and tyrosine through the shikimate pathway."
                },
                {
                    "name": "Peptides_np",
                    "label": "Peptides (NP)",
                    "comment": "Short chains of amino acids with biological activity, often produced by non-ribosomal synthesis."
                },
                {
                    "name": "Saccharides_np", 
                    "label": "Saccharides (NP)",
                    "comment": "Natural product carbohydrates including complex polysaccharides and glycosides."
                },
                {
                    "name": "Fatty_acids_np",
                    "label": "Fatty acids (NP)",
                    "comment": "Natural product fatty acids with unique structural features and biological activities."
                },
                {
                    "name": "Shikimates",
                    "label": "Shikimates",
                    "comment": "Natural products derived from the shikimate pathway, precursors to aromatic amino acids."
                }
            ]
            
            # Define representative PMN plant metabolic compound instances
            pmn_terms = [
                {
                    "name": "Glucose_pmn",
                    "label": "Glucose",
                    "comment": "Primary sugar and energy source in plant metabolism, central to carbohydrate pathways."
                },
                {
                    "name": "Sucrose_pmn",
                    "label": "Sucrose", 
                    "comment": "Major transport sugar in plants, composed of glucose and fructose units."
                },
                {
                    "name": "Chlorophyll_a",
                    "label": "Chlorophyll a",
                    "comment": "Primary photosynthetic pigment in plants, essential for light harvesting."
                },
                {
                    "name": "Cellulose_pmn",
                    "label": "Cellulose",
                    "comment": "Structural polysaccharide forming the primary component of plant cell walls."
                },
                {
                    "name": "Starch_pmn", 
                    "label": "Starch",
                    "comment": "Energy storage polysaccharide in plants, composed of amylose and amylopectin."
                },
                {
                    "name": "ATP_pmn",
                    "label": "Adenosine triphosphate (ATP)",
                    "comment": "Universal energy currency in plant cells for metabolic processes."
                },
                {
                    "name": "Lignin_pmn",
                    "label": "Lignin",
                    "comment": "Complex aromatic polymer providing structural support in plant cell walls."
                },
                {
                    "name": "Anthocyanins_pmn",
                    "label": "Anthocyanins", 
                    "comment": "Water-soluble pigments responsible for red, purple, and blue colors in plants."
                }
            ]
            
            # Create instances within the ontology context
            with ontology:
                chemont_instances = []
                np_instances = []
                pmn_instances = []
                
                # Create Chemont instances
                for term_data in chemont_terms:
                    instance = chemont_class(term_data["name"])
                    instance.label = [term_data["label"]]
                    instance.comment = [term_data["comment"]]
                    chemont_instances.append(instance)
                    logger.debug(f"Created Chemont instance: {term_data['name']}")
                
                # Create NP Classifier instances  
                for term_data in np_terms:
                    instance = np_class(term_data["name"])
                    instance.label = [term_data["label"]]
                    instance.comment = [term_data["comment"]]
                    np_instances.append(instance)
                    logger.debug(f"Created NP instance: {term_data['name']}")
                
                # Create PMN instances
                for term_data in pmn_terms:
                    instance = pmn_compound(term_data["name"])
                    instance.label = [term_data["label"]]
                    instance.comment = [term_data["comment"]]
                    pmn_instances.append(instance)
                    logger.debug(f"Created PMN instance: {term_data['name']}")
                
                result = {
                    'chemont_instances': chemont_instances,
                    'np_instances': np_instances, 
                    'pmn_instances': pmn_instances
                }
                
                total_instances = len(chemont_instances) + len(np_instances) + len(pmn_instances)
                logger.info(f"Successfully created {total_instances} initial key term instances "
                           f"({len(chemont_instances)} Chemont, {len(np_instances)} NP, {len(pmn_instances)} PMN)")
                
                return result
                
    except OwlReadyError as e:
        raise StructuralClassError(f"Owlready2 error creating initial key terms: {e}")
    except Exception as e:
        raise StructuralClassError(f"Failed to create initial key terms: {e}")


def validate_initial_key_terms(ontology: Any) -> Dict[str, int]:
    """Validate that initial key terms/instances have been created successfully.
    
    Checks that instances of ChemontClass, NPClass, and PMNCompound have been
    properly created in the ontology with correct properties.
    
    Args:
        ontology: Ontology to validate
        
    Returns:
        Dictionary with counts of found instances for each class type
        
    Example:
        counts = validate_initial_key_terms(ontology)
        print(f"Found {counts['chemont_count']} Chemont instances")
    """
    try:
        _validate_ontology(ontology)
        
        # Get the required classes
        chemont_class = ontology.search_one(iri="*ChemontClass")
        np_class = ontology.search_one(iri="*NPClass")
        pmn_compound = ontology.search_one(iri="*PMNCompound")
        
        if not all([chemont_class, np_class, pmn_compound]):
            logger.warning("Required structural classes not found")
            return {"chemont_count": 0, "np_count": 0, "pmn_count": 0, "total_count": 0}
        
        # Count instances of each class
        chemont_instances = list(chemont_class.instances())
        np_instances = list(np_class.instances())
        pmn_instances = list(pmn_compound.instances())
        
        # Validate that instances have proper labels and comments
        valid_instances = {"chemont": 0, "np": 0, "pmn": 0}
        
        for instance in chemont_instances:
            if hasattr(instance, 'label') and hasattr(instance, 'comment'):
                if instance.label and instance.comment:
                    valid_instances["chemont"] += 1
        
        for instance in np_instances:
            if hasattr(instance, 'label') and hasattr(instance, 'comment'):
                if instance.label and instance.comment:
                    valid_instances["np"] += 1
        
        for instance in pmn_instances:
            if hasattr(instance, 'label') and hasattr(instance, 'comment'):
                if instance.label and instance.comment:
                    valid_instances["pmn"] += 1
        
        result = {
            "chemont_count": valid_instances["chemont"],
            "np_count": valid_instances["np"], 
            "pmn_count": valid_instances["pmn"],
            "total_count": sum(valid_instances.values())
        }
        
        logger.info(f"Validated key terms: {result['total_count']} total instances "
                   f"({result['chemont_count']} Chemont, {result['np_count']} NP, {result['pmn_count']} PMN)")
        
        return result
        
    except Exception as e:
        logger.error(f"Error validating initial key terms: {e}")
        return {"chemont_count": 0, "np_count": 0, "pmn_count": 0, "total_count": 0}


def validate_core_structural_classes(ontology: Any) -> bool:
    """Validate that core structural classes are properly defined.
    
    Checks that ChemontClass, NPClass, and PMNCompound are properly defined
    in the ontology with correct inheritance and properties.
    
    Args:
        ontology: Ontology to validate
        
    Returns:
        True if all core classes are properly defined, False otherwise
        
    Example:
        is_valid = validate_core_structural_classes(ontology)
    """
    try:
        _validate_ontology(ontology)
        
        required_classes = ['ChemontClass', 'NPClass', 'PMNCompound']
        
        for class_name in required_classes:
            # Check if class exists
            if not verify_class_accessibility(ontology, class_name):
                logger.warning(f"Required class not found: {class_name}")
                return False
            
            # Validate class properties
            if not validate_structural_class_properties(ontology, class_name):
                logger.warning(f"Invalid properties for class: {class_name}")
                return False
            
            # Verify Thing inheritance
            if not verify_thing_inheritance(ontology, class_name):
                logger.warning(f"Class does not inherit from Thing: {class_name}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating core structural classes: {e}")
        return False