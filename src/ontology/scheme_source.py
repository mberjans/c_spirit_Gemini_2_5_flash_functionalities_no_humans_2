"""
Ontology scheme source module for plant metabolomics.

This module defines and integrates terms for "Source Annotation" including
Plant Ontology, NCBI Taxonomy, and PECO (Plant Experimental Conditions Ontology)
into a core ontology using Owlready2. It provides functions for creating, managing,
and validating source annotation classes within OWL 2.0 ontologies.

The module supports:
- PlantAnatomy creation for plant anatomical structure classification
- Species creation for taxonomic species classification  
- ExperimentalCondition creation for plant experimental conditions
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


class SourceClassError(Exception):
    """Custom exception for source class operations.
    
    This exception is raised when errors occur during the creation,
    validation, or manipulation of source annotation classes.
    
    Args:
        message: Error description
        
    Example:
        raise SourceClassError("Invalid ontology provided")
    """
    
    def __init__(self, message: str) -> None:
        """Initialize the source class error.
        
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
        SourceClassError: If class name is invalid
    """
    if not class_name or not isinstance(class_name, str):
        raise SourceClassError("Invalid class name: must be a non-empty string")
    
    # Remove leading/trailing whitespace
    class_name = class_name.strip()
    
    if not class_name:
        raise SourceClassError("Invalid class name: cannot be empty or whitespace only")
    
    # Check for valid OWL class name pattern
    if not re.match(r'^[A-Za-z][A-Za-z0-9_]*$', class_name):
        raise SourceClassError(
            f"Invalid class name '{class_name}': must start with letter and "
            "contain only letters, numbers, and underscores"
        )


def _validate_ontology(ontology: Any) -> None:
    """Validate that the provided ontology is valid.
    
    Args:
        ontology: Ontology object to validate
        
    Raises:
        SourceClassError: If ontology is invalid
    """
    if ontology is None:
        raise SourceClassError("Invalid ontology: cannot be None")


def create_plant_anatomy_class(ontology: Any, class_name: str) -> Any:
    """Create a PlantAnatomy class for plant anatomical structure classification.
    
    Creates a new OWL class for plant anatomical structure classification based on the
    Plant Ontology. The class inherits from owlready2.Thing and includes appropriate 
    semantic annotations.
    
    Args:
        ontology: Target ontology for class creation
        class_name: Name of the PlantAnatomy class to create
        
    Returns:
        The created PlantAnatomy class object
        
    Raises:
        SourceClassError: If creation fails due to invalid parameters or Owlready2 errors
        
    Example:
        plant_anatomy_class = create_plant_anatomy_class(ontology, "PlantAnatomy")
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    
    try:
        with _creation_lock:
            # Get the ontology namespace
            namespace = ontology.get_namespace()
            
            # Create the PlantAnatomy class
            plant_anatomy_class = owlready_types.new_class(
                class_name,
                (Thing,),
                namespace
            )
            
            # Add semantic annotations
            plant_anatomy_class.label = [f"Plant Anatomical Entity"]
            plant_anatomy_class.comment = [
                "Base class for plant anatomical structures based on Plant Ontology"
            ]
            
            logger.info(f"Created PlantAnatomy class: {class_name}")
            return plant_anatomy_class
            
    except OwlReadyError as e:
        raise SourceClassError(f"Owlready2 error creating PlantAnatomy class '{class_name}': {e}")
    except Exception as e:
        raise SourceClassError(f"Failed to create PlantAnatomy class '{class_name}': {e}")


def create_species_class(ontology: Any, class_name: str) -> Any:
    """Create a Species class for taxonomic species classification.
    
    Creates a new OWL class for taxonomic species classification based on the
    NCBI Taxonomy. The class inherits from owlready2.Thing and includes
    appropriate semantic annotations for species categorization.
    
    Args:
        ontology: Target ontology for class creation
        class_name: Name of the Species class to create
        
    Returns:
        The created Species class object
        
    Raises:
        SourceClassError: If creation fails due to invalid parameters or Owlready2 errors
        
    Example:
        species_class = create_species_class(ontology, "Species")
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    
    try:
        with _creation_lock:
            # Get the ontology namespace
            namespace = ontology.get_namespace()
            
            # Create the Species class
            species_class = owlready_types.new_class(
                class_name,
                (Thing,),
                namespace
            )
            
            # Add semantic annotations
            species_class.label = [f"Taxonomic Species"]
            species_class.comment = [
                "Base class for taxonomic species classification based on NCBI Taxonomy"
            ]
            
            logger.info(f"Created Species class: {class_name}")
            return species_class
            
    except OwlReadyError as e:
        raise SourceClassError(f"Owlready2 error creating Species class '{class_name}': {e}")
    except Exception as e:
        raise SourceClassError(f"Failed to create Species class '{class_name}': {e}")


def create_experimental_condition_class(ontology: Any, class_name: str) -> Any:
    """Create an ExperimentalCondition class for plant experimental conditions.
    
    Creates a new OWL class for plant experimental conditions from the PECO
    (Plant Experimental Conditions Ontology). The class inherits from owlready2.Thing
    and includes appropriate semantic annotations for experimental condition classification.
    
    Args:
        ontology: Target ontology for class creation
        class_name: Name of the ExperimentalCondition class to create
        
    Returns:
        The created ExperimentalCondition class object
        
    Raises:
        SourceClassError: If creation fails due to invalid parameters or Owlready2 errors
        
    Example:
        exp_condition_class = create_experimental_condition_class(ontology, "ExperimentalCondition")
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    
    try:
        with _creation_lock:
            # Get the ontology namespace
            namespace = ontology.get_namespace()
            
            # Create the ExperimentalCondition class
            experimental_condition_class = owlready_types.new_class(
                class_name,
                (Thing,),
                namespace
            )
            
            # Add semantic annotations
            experimental_condition_class.label = [f"Plant Experimental Condition"]
            experimental_condition_class.comment = [
                "Base class for plant experimental conditions based on PECO"
            ]
            
            logger.info(f"Created ExperimentalCondition class: {class_name}")
            return experimental_condition_class
            
    except OwlReadyError as e:
        raise SourceClassError(f"Owlready2 error creating ExperimentalCondition class '{class_name}': {e}")
    except Exception as e:
        raise SourceClassError(f"Failed to create ExperimentalCondition class '{class_name}': {e}")


def create_species_class_with_parent(ontology: Any, class_name: str, parent_class_name: str) -> Any:
    """Create a Species class with hierarchical relationship to a parent class.
    
    Creates a new Species class that inherits from both Thing and a specified parent class,
    establishing hierarchical relationships in the ontology structure.
    
    Args:
        ontology: Target ontology for class creation
        class_name: Name of the Species class to create
        parent_class_name: Name of the parent class to inherit from
        
    Returns:
        The created Species class object with parent relationship
        
    Raises:
        SourceClassError: If creation fails or parent class not found
        
    Example:
        species_class = create_species_class_with_parent(ontology, "Species", "Organism")
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    _validate_class_name(parent_class_name)
    
    try:
        with _creation_lock:
            # Find the parent class
            parent_class = ontology.search_one(iri=f"*{parent_class_name}")
            if not parent_class:
                raise SourceClassError(f"Parent class '{parent_class_name}' not found")
            
            # Get the ontology namespace
            namespace = ontology.get_namespace()
            
            # Create the Species class with parent relationship
            species_class = owlready_types.new_class(
                class_name,
                (parent_class,),
                namespace
            )
            
            # Add semantic annotations
            species_class.label = [f"Taxonomic Species"]
            species_class.comment = [
                "Base class for taxonomic species classification based on NCBI Taxonomy"
            ]
            
            logger.info(f"Created Species class with parent: {class_name} -> {parent_class_name}")
            return species_class
            
    except OwlReadyError as e:
        raise SourceClassError(f"Owlready2 error creating Species class with parent '{class_name}': {e}")
    except Exception as e:
        raise SourceClassError(f"Failed to create Species class with parent '{class_name}': {e}")


def create_root_class_with_parent(ontology: Any, class_name: str, parent_class_name: str) -> Any:
    """Create a Root class with hierarchical relationship to a parent class.
    
    Creates a new Root class that inherits from a specified parent class,
    establishing hierarchical relationships in the plant anatomy ontology structure.
    
    Args:
        ontology: Target ontology for class creation
        class_name: Name of the Root class to create
        parent_class_name: Name of the parent class to inherit from
        
    Returns:
        The created Root class object with parent relationship
        
    Raises:
        SourceClassError: If creation fails or parent class not found
        
    Example:
        root_class = create_root_class_with_parent(ontology, "Root", "PlantAnatomy")
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    _validate_class_name(parent_class_name)
    
    try:
        with _creation_lock:
            # Find the parent class
            parent_class = ontology.search_one(iri=f"*{parent_class_name}")
            if not parent_class:
                raise SourceClassError(f"Parent class '{parent_class_name}' not found")
            
            # Get the ontology namespace
            namespace = ontology.get_namespace()
            
            # Create the Root class with parent relationship
            root_class = owlready_types.new_class(
                class_name,
                (parent_class,),
                namespace
            )
            
            # Add semantic annotations
            root_class.label = [f"Plant Root"]
            root_class.comment = [
                "Plant root anatomical structure based on Plant Ontology"
            ]
            
            logger.info(f"Created Root class with parent: {class_name} -> {parent_class_name}")
            return root_class
            
    except OwlReadyError as e:
        raise SourceClassError(f"Owlready2 error creating Root class with parent '{class_name}': {e}")
    except Exception as e:
        raise SourceClassError(f"Failed to create Root class with parent '{class_name}': {e}")


def create_stress_condition_with_parent(ontology: Any, class_name: str, parent_class_name: str) -> Any:
    """Create a StressCondition class with hierarchical relationship to a parent class.
    
    Creates a new StressCondition class that inherits from a specified parent class,
    establishing hierarchical relationships in the experimental condition ontology structure.
    
    Args:
        ontology: Target ontology for class creation
        class_name: Name of the StressCondition class to create
        parent_class_name: Name of the parent class to inherit from
        
    Returns:
        The created StressCondition class object with parent relationship
        
    Raises:
        SourceClassError: If creation fails or parent class not found
        
    Example:
        stress_class = create_stress_condition_with_parent(ontology, "StressCondition", "ExperimentalCondition")
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    _validate_class_name(parent_class_name)
    
    try:
        with _creation_lock:
            # Find the parent class
            parent_class = ontology.search_one(iri=f"*{parent_class_name}")
            if not parent_class:
                raise SourceClassError(f"Parent class '{parent_class_name}' not found")
            
            # Get the ontology namespace
            namespace = ontology.get_namespace()
            
            # Create the StressCondition class with parent relationship
            stress_condition_class = owlready_types.new_class(
                class_name,
                (parent_class,),
                namespace
            )
            
            # Add semantic annotations
            stress_condition_class.label = [f"Plant Stress Condition"]
            stress_condition_class.comment = [
                "Experimental condition involving plant stress based on PECO"
            ]
            
            logger.info(f"Created StressCondition class with parent: {class_name} -> {parent_class_name}")
            return stress_condition_class
            
    except OwlReadyError as e:
        raise SourceClassError(f"Owlready2 error creating StressCondition class with parent '{class_name}': {e}")
    except Exception as e:
        raise SourceClassError(f"Failed to create StressCondition class with parent '{class_name}': {e}")


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
        is_accessible = verify_class_accessibility(ontology, "PlantAnatomy")
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
        SourceClassError: If class not found
        
    Example:
        depth = get_class_hierarchy_depth(ontology, "Species")  # Returns 2 if Species -> Organism -> Thing
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    
    try:
        # Find the class
        target_class = ontology.search_one(iri=f"*{class_name}")
        if not target_class:
            raise SourceClassError(f"Class '{class_name}' not found in ontology")
        
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
        raise SourceClassError(f"Failed to calculate hierarchy depth for '{class_name}': {e}")


def classify_source_annotation_type(ontology: Any, class_name: str) -> str:
    """Classify the source annotation type based on class name.
    
    Determines the type of source annotation based on the class name patterns
    used in the ontology for different classification systems.
    
    Args:
        ontology: Ontology containing the class
        class_name: Name of the class to classify
        
    Returns:
        Classification category as string
        
    Example:
        category = classify_source_annotation_type(ontology, "PlantAnatomy")
        # Returns "plant_ontology_classification"
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
        
        if "plant" in name_lower and "anatomy" in name_lower:
            return "plant_ontology_classification"
        elif "species" in name_lower:
            return "ncbi_taxonomy_classification"
        elif "experimental" in name_lower and "condition" in name_lower:
            return "peco_classification"
        else:
            return "unknown_classification"
            
    except Exception as e:
        logger.warning(f"Error classifying annotation type for '{class_name}': {e}")
        return "unknown_classification"


def create_source_classes_batch(ontology: Any, class_specs: List[Dict[str, Any]]) -> List[Any]:
    """Create multiple source annotation classes in batch.
    
    Efficiently creates multiple source classes based on provided specifications,
    supporting different class types and parent relationships.
    
    Args:
        ontology: Target ontology for class creation
        class_specs: List of class specifications, each containing:
            - name: Class name
            - type: Class type ("plant_ontology", "ncbi_taxonomy", "peco")
            - parent: Optional parent class name
        
    Returns:
        List of created class objects
        
    Raises:
        SourceClassError: If batch creation fails
        
    Example:
        specs = [
            {"name": "PlantAnatomy", "type": "plant_ontology", "parent": None},
            {"name": "Species", "type": "ncbi_taxonomy", "parent": "Organism"}
        ]
        classes = create_source_classes_batch(ontology, specs)
    """
    _validate_ontology(ontology)
    
    if not class_specs or not isinstance(class_specs, list):
        raise SourceClassError("Invalid class specifications: must be a non-empty list")
    
    created_classes = []
    
    try:
        for spec in class_specs:
            if not isinstance(spec, dict):
                raise SourceClassError("Invalid class specification: must be a dictionary")
            
            name = spec.get("name")
            class_type = spec.get("type")
            parent = spec.get("parent")
            
            if not name:
                raise SourceClassError("Class specification missing 'name' field")
            
            # Create class based on type
            if class_type == "plant_ontology":
                created_class = create_plant_anatomy_class(ontology, name)
            elif class_type == "ncbi_taxonomy":
                if parent:
                    created_class = create_species_class_with_parent(ontology, name, parent)
                else:
                    created_class = create_species_class(ontology, name)
            elif class_type == "peco":
                created_class = create_experimental_condition_class(ontology, name)
            else:
                # Default to creating as Species class
                created_class = create_species_class(ontology, name)
            
            created_classes.append(created_class)
            
        logger.info(f"Successfully created {len(created_classes)} source classes in batch")
        return created_classes
        
    except Exception as e:
        logger.error(f"Batch class creation failed: {e}")
        raise SourceClassError(f"Failed to create source classes in batch: {e}")


def validate_source_class_properties(ontology: Any, class_name: str) -> bool:
    """Validate that a source class has required properties.
    
    Checks if a source annotation class has all required properties including
    proper labels, comments, and inheritance relationships.
    
    Args:
        ontology: Ontology containing the class
        class_name: Name of the class to validate
        
    Returns:
        True if class has all required properties, False otherwise
        
    Example:
        is_valid = validate_source_class_properties(ontology, "PlantAnatomy")
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
    """Verify that a source class properly inherits from Thing.
    
    Checks the inheritance chain to ensure the class ultimately inherits from
    owlready2.Thing, either directly or through parent classes.
    
    Args:
        ontology: Ontology containing the class
        class_name: Name of the class to verify
        
    Returns:
        True if class inherits from Thing, False otherwise
        
    Example:
        inherits_thing = verify_thing_inheritance(ontology, "PlantAnatomy")
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


def get_all_source_classes(ontology: Any) -> List[Any]:
    """Get all source annotation classes from the ontology.
    
    Retrieves all classes that match source annotation patterns,
    including PlantAnatomy, Species, and ExperimentalCondition types.
    
    Args:
        ontology: Ontology to search
        
    Returns:
        List of source annotation class objects
        
    Example:
        source_classes = get_all_source_classes(ontology)
    """
    try:
        _validate_ontology(ontology)
        
        # Get all classes from the ontology
        all_classes = list(ontology.classes())
        
        # Filter for source annotation classes
        source_classes = []
        source_patterns = ['plantanatomy', 'species', 'experimentalcondition', 'source']
        
        for cls in all_classes:
            if hasattr(cls, 'name') and cls.name:
                name_lower = cls.name.lower()
                if any(pattern in name_lower for pattern in source_patterns):
                    source_classes.append(cls)
        
        return source_classes
        
    except Exception as e:
        logger.error(f"Error retrieving source classes: {e}")
        return []


def validate_class_metadata(ontology: Any, class_name: str) -> bool:
    """Validate class metadata and annotations.
    
    Performs comprehensive validation of class metadata including labels,
    comments, IRI structure, and custom annotations specific to source
    annotation classes.
    
    Args:
        ontology: Ontology containing the class
        class_name: Name of the class to validate
        
    Returns:
        True if metadata is valid, False otherwise
        
    Example:
        metadata_valid = validate_class_metadata(ontology, "PlantAnatomy")
    """
    try:
        _validate_ontology(ontology)
        _validate_class_name(class_name)
        
        # Find the class
        target_class = ontology.search_one(iri=f"*{class_name}")
        if not target_class:
            return False
        
        # Validate basic metadata
        if not validate_source_class_properties(ontology, class_name):
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


def cleanup_source_classes(ontology: Any) -> int:
    """Cleanup source annotation classes from the ontology.
    
    Removes all source annotation classes from the ontology, useful for
    cleanup operations or resetting the ontology state.
    
    Args:
        ontology: Ontology to clean up
        
    Returns:
        Number of classes removed
        
    Warning:
        This operation is destructive and cannot be undone
        
    Example:
        removed_count = cleanup_source_classes(ontology)
    """
    try:
        _validate_ontology(ontology)
        
        # Get all source classes
        source_classes = get_all_source_classes(ontology)
        
        cleanup_count = 0
        with _creation_lock:
            for cls in source_classes:
                try:
                    if hasattr(cls, 'destroy'):
                        cls.destroy()
                        cleanup_count += 1
                except Exception as e:
                    logger.warning(f"Failed to destroy class {cls.name}: {e}")
        
        logger.info(f"Cleaned up {cleanup_count} source classes")
        return cleanup_count
        
    except Exception as e:
        logger.error(f"Error during source class cleanup: {e}")
        return 0


def define_core_source_classes(ontology: Any) -> Dict[str, Any]:
    """Define core source annotation classes in the main ontology namespace.
    
    Creates the fundamental source annotation classes (PlantAnatomy, Species, 
    ExperimentalCondition) that inherit from owlready2.Thing and associates them with the 
    main ontology namespace. This implements the core requirements for 
    AIM2-ODIE-010-T2.
    
    Args:
        ontology: Main ontology to define classes in
        
    Returns:
        Dictionary mapping class names to created class objects
        
    Raises:
        SourceClassError: If class definition fails
        
    Example:
        classes = define_core_source_classes(ontology)
        plant_anatomy_class = classes['PlantAnatomy']
        species_class = classes['Species']
        experimental_condition_class = classes['ExperimentalCondition']
    """
    _validate_ontology(ontology)
    
    try:
        with _creation_lock:
            # Use the ontology context for class creation
            with ontology:
                # Define PlantAnatomy for plant anatomical structure classification
                class PlantAnatomy(Thing):
                    namespace = ontology
                    
                PlantAnatomy.label = ["Plant Anatomical Entity"]
                PlantAnatomy.comment = [
                    "Base class for plant anatomical structures based on Plant Ontology. "
                    "Provides source annotation for metabolites using plant anatomy classifications."
                ]
                
                # Define Species for taxonomic species classification
                class Species(Thing):
                    namespace = ontology
                    
                Species.label = ["Taxonomic Species"]
                Species.comment = [
                    "Base class for taxonomic species classification based on NCBI Taxonomy. "
                    "Provides source annotation for metabolites using taxonomic hierarchies."
                ]
                
                # Define ExperimentalCondition for plant experimental conditions
                class ExperimentalCondition(Thing):
                    namespace = ontology
                    
                ExperimentalCondition.label = ["Plant Experimental Condition"]
                ExperimentalCondition.comment = [
                    "Base class for plant experimental conditions based on PECO (Plant Experimental Conditions Ontology). "
                    "Provides source annotation for metabolites using experimental condition classifications."
                ]
                
                # Create the class registry
                defined_classes = {
                    'PlantAnatomy': PlantAnatomy,
                    'Species': Species,
                    'ExperimentalCondition': ExperimentalCondition
                }
                
                logger.info(f"Successfully defined {len(defined_classes)} core source classes")
                
                return defined_classes
            
    except OwlReadyError as e:
        raise SourceClassError(f"Owlready2 error defining core source classes: {e}")
    except Exception as e:
        raise SourceClassError(f"Failed to define core source classes: {e}")


def establish_source_hierarchy(ontology: Any, classes: Dict[str, Any]) -> None:
    """Establish hierarchical relationships between source annotation classes.
    
    Creates is_a relationships and other hierarchical connections between the
    defined source annotation classes to represent classification hierarchies.
    
    Args:
        ontology: Main ontology containing the classes
        classes: Dictionary of class names to class objects
        
    Raises:
        SourceClassError: If hierarchy establishment fails
        
    Example:
        classes = define_core_source_classes(ontology)
        establish_source_hierarchy(ontology, classes)
    """
    _validate_ontology(ontology)
    
    if not classes or not isinstance(classes, dict):
        raise SourceClassError("Invalid classes dictionary")
    
    try:
        with _creation_lock:
            # Get required classes
            plant_anatomy_class = classes.get('PlantAnatomy')
            species_class = classes.get('Species')
            experimental_condition_class = classes.get('ExperimentalCondition')
            
            if not all([plant_anatomy_class, species_class, experimental_condition_class]):
                raise SourceClassError("Missing required source classes")
            
            # Create a general SourceAnnotation superclass within the ontology context
            with ontology:
                class SourceAnnotation(Thing):
                    namespace = ontology
                    
                SourceAnnotation.label = ["Source Annotation"]
                SourceAnnotation.comment = [
                    "Superclass for all source annotation concepts including plant anatomical structures, "
                    "taxonomic species, and experimental conditions."
                ]
            
            # Establish hierarchical relationships
            # All source classes inherit from SourceAnnotation
            plant_anatomy_class.is_a.append(SourceAnnotation)
            species_class.is_a.append(SourceAnnotation)
            experimental_condition_class.is_a.append(SourceAnnotation)
            
            # Keep them as peer classes under SourceAnnotation for now
            # More specific hierarchies can be added as needed
            
            logger.info("Successfully established source class hierarchy")
            
    except Exception as e:
        raise SourceClassError(f"Failed to establish source hierarchy: {e}")


def add_initial_key_terms(ontology: Any) -> Dict[str, List[Any]]:
    """Add initial key terms/instances from Plant Ontology, NCBI Taxonomy, and PECO to the ontology.
    
    Creates representative instances of PlantAnatomy, Species, and ExperimentalCondition classes
    to populate the ontology with initial key terms from each classification system.
    This function implements AIM2-ODIE-010-T2 by adding concrete examples from each
    source annotation system.
    
    Args:
        ontology: Target ontology for instance creation
        
    Returns:
        Dictionary with keys "plant_anatomy_instances", "species_instances", "peco_instances"
        containing lists of created instances
        
    Raises:
        SourceClassError: If instance creation fails
        
    Example:
        instances = add_initial_key_terms(ontology)
        root = instances['plant_anatomy_instances'][0]
        arabidopsis = instances['species_instances'][0]
        drought_stress = instances['peco_instances'][0]
    """
    _validate_ontology(ontology)
    
    try:
        with _creation_lock:
            # Get the required classes
            plant_anatomy_class = ontology.search_one(iri="*PlantAnatomy")
            species_class = ontology.search_one(iri="*Species")
            experimental_condition_class = ontology.search_one(iri="*ExperimentalCondition")
            
            if not all([plant_anatomy_class, species_class, experimental_condition_class]):
                raise SourceClassError(
                    "Required source classes not found. Please run define_core_source_classes() first."
                )
            
            # Define representative Plant Ontology anatomical structure instances
            plant_anatomy_terms = [
                {
                    "name": "Root",
                    "label": "Plant Root",
                    "comment": "Underground plant organ responsible for water and nutrient absorption."
                },
                {
                    "name": "Leaf", 
                    "label": "Plant Leaf",
                    "comment": "Photosynthetic organ of plants, typically flat and green."
                },
                {
                    "name": "Stem",
                    "label": "Plant Stem", 
                    "comment": "Main structural axis of a plant that supports leaves and reproductive structures."
                },
                {
                    "name": "Flower",
                    "label": "Plant Flower",
                    "comment": "Reproductive structure of flowering plants containing sexual organs."
                },
                {
                    "name": "Seed",
                    "label": "Plant Seed",
                    "comment": "Embryonic plant enclosed in a protective outer covering."
                },
                {
                    "name": "Fruit",
                    "label": "Plant Fruit",
                    "comment": "Seed-bearing structure in flowering plants formed from the flower after flowering."
                },
                {
                    "name": "Bark",
                    "label": "Plant Bark",
                    "comment": "Outermost layers of stems and roots of woody plants."
                },
                {
                    "name": "Pollen",
                    "label": "Plant Pollen",
                    "comment": "Fine powder containing male gametes of seed plants."
                }
            ]
            
            # Define representative NCBI Taxonomy species instances
            species_terms = [
                {
                    "name": "Arabidopsis_thaliana",
                    "label": "Arabidopsis thaliana",
                    "comment": "Model organism in plant biology, thale cress."
                },
                {
                    "name": "Oryza_sativa",
                    "label": "Oryza sativa", 
                    "comment": "Asian rice, staple food crop."
                },
                {
                    "name": "Zea_mays",
                    "label": "Zea mays",
                    "comment": "Maize or corn, major cereal grain."
                },
                {
                    "name": "Solanum_lycopersicum",
                    "label": "Solanum lycopersicum",
                    "comment": "Tomato, cultivated worldwide for food."
                },
                {
                    "name": "Glycine_max",
                    "label": "Glycine max",
                    "comment": "Soybean, legume grown for protein-rich beans."
                },
                {
                    "name": "Triticum_aestivum", 
                    "label": "Triticum aestivum",
                    "comment": "Common wheat, major food grain."
                },
                {
                    "name": "Medicago_truncatula",
                    "label": "Medicago truncatula",
                    "comment": "Barrel medic, model legume species."
                },
                {
                    "name": "Populus_trichocarpa",
                    "label": "Populus trichocarpa",
                    "comment": "Black cottonwood, model tree species."
                }
            ]
            
            # Define representative PECO experimental condition instances
            peco_terms = [
                {
                    "name": "Drought_stress",
                    "label": "Drought stress",
                    "comment": "Water deficit condition affecting plant growth and metabolism."
                },
                {
                    "name": "Salt_stress",
                    "label": "Salt stress",
                    "comment": "High salinity condition that disrupts plant ionic homeostasis."
                },
                {
                    "name": "Heat_stress",
                    "label": "Heat stress", 
                    "comment": "High temperature condition that affects plant cellular processes."
                },
                {
                    "name": "Cold_stress",
                    "label": "Cold stress",
                    "comment": "Low temperature condition that impairs plant metabolism."
                },
                {
                    "name": "Light_stress",
                    "label": "Light stress",
                    "comment": "Excessive light condition that can damage photosynthetic apparatus."
                },
                {
                    "name": "Nutrient_deficiency", 
                    "label": "Nutrient deficiency",
                    "comment": "Insufficient nutrient availability affecting plant growth."
                },
                {
                    "name": "Pathogen_infection",
                    "label": "Pathogen infection",
                    "comment": "Disease condition caused by pathogenic microorganisms."
                },
                {
                    "name": "Mechanical_stress",
                    "label": "Mechanical stress",
                    "comment": "Physical force or wounding that affects plant tissues."
                }
            ]
            
            # Create instances within the ontology context
            with ontology:
                plant_anatomy_instances = []
                species_instances = []
                peco_instances = []
                
                # Create Plant Anatomy instances
                for term_data in plant_anatomy_terms:
                    instance = plant_anatomy_class(term_data["name"])
                    instance.label = [term_data["label"]]
                    instance.comment = [term_data["comment"]]
                    plant_anatomy_instances.append(instance)
                    logger.debug(f"Created Plant Anatomy instance: {term_data['name']}")
                
                # Create Species instances  
                for term_data in species_terms:
                    instance = species_class(term_data["name"])
                    instance.label = [term_data["label"]]
                    instance.comment = [term_data["comment"]]
                    species_instances.append(instance)
                    logger.debug(f"Created Species instance: {term_data['name']}")
                
                # Create PECO instances
                for term_data in peco_terms:
                    instance = experimental_condition_class(term_data["name"])
                    instance.label = [term_data["label"]]
                    instance.comment = [term_data["comment"]]
                    peco_instances.append(instance)
                    logger.debug(f"Created PECO instance: {term_data['name']}")
                
                result = {
                    'plant_anatomy_instances': plant_anatomy_instances,
                    'species_instances': species_instances, 
                    'peco_instances': peco_instances
                }
                
                total_instances = len(plant_anatomy_instances) + len(species_instances) + len(peco_instances)
                logger.info(f"Successfully created {total_instances} initial key term instances "
                           f"({len(plant_anatomy_instances)} Plant Anatomy, {len(species_instances)} Species, {len(peco_instances)} PECO)")
                
                return result
                
    except OwlReadyError as e:
        raise SourceClassError(f"Owlready2 error creating initial key terms: {e}")
    except Exception as e:
        raise SourceClassError(f"Failed to create initial key terms: {e}")


def validate_initial_key_terms(ontology: Any) -> Dict[str, int]:
    """Validate that initial key terms/instances have been created successfully.
    
    Checks that instances of PlantAnatomy, Species, and ExperimentalCondition have been
    properly created in the ontology with correct properties.
    
    Args:
        ontology: Ontology to validate
        
    Returns:
        Dictionary with counts of found instances for each class type
        
    Example:
        counts = validate_initial_key_terms(ontology)
        print(f"Found {counts['plant_anatomy_count']} Plant Anatomy instances")
    """
    try:
        _validate_ontology(ontology)
        
        # Get the required classes
        plant_anatomy_class = ontology.search_one(iri="*PlantAnatomy")
        species_class = ontology.search_one(iri="*Species")
        experimental_condition_class = ontology.search_one(iri="*ExperimentalCondition")
        
        if not all([plant_anatomy_class, species_class, experimental_condition_class]):
            logger.warning("Required source classes not found")
            return {"plant_anatomy_count": 0, "species_count": 0, "peco_count": 0, "total_count": 0}
        
        # Count instances of each class
        plant_anatomy_instances = list(plant_anatomy_class.instances())
        species_instances = list(species_class.instances())
        peco_instances = list(experimental_condition_class.instances())
        
        # Validate that instances have proper labels and comments
        valid_instances = {"plant_anatomy": 0, "species": 0, "peco": 0}
        
        for instance in plant_anatomy_instances:
            if hasattr(instance, 'label') and hasattr(instance, 'comment'):
                if instance.label and instance.comment:
                    valid_instances["plant_anatomy"] += 1
        
        for instance in species_instances:
            if hasattr(instance, 'label') and hasattr(instance, 'comment'):
                if instance.label and instance.comment:
                    valid_instances["species"] += 1
        
        for instance in peco_instances:
            if hasattr(instance, 'label') and hasattr(instance, 'comment'):
                if instance.label and instance.comment:
                    valid_instances["peco"] += 1
        
        result = {
            "plant_anatomy_count": valid_instances["plant_anatomy"],
            "species_count": valid_instances["species"], 
            "peco_count": valid_instances["peco"],
            "total_count": sum(valid_instances.values())
        }
        
        logger.info(f"Validated key terms: {result['total_count']} total instances "
                   f"({result['plant_anatomy_count']} Plant Anatomy, {result['species_count']} Species, {result['peco_count']} PECO)")
        
        return result
        
    except Exception as e:
        logger.error(f"Error validating initial key terms: {e}")
        return {"plant_anatomy_count": 0, "species_count": 0, "peco_count": 0, "total_count": 0}


def validate_core_source_classes(ontology: Any) -> bool:
    """Validate that core source classes are properly defined.
    
    Checks that PlantAnatomy, Species, and ExperimentalCondition are properly defined
    in the ontology with correct inheritance and properties.
    
    Args:
        ontology: Ontology to validate
        
    Returns:
        True if all core classes are properly defined, False otherwise
        
    Example:
        is_valid = validate_core_source_classes(ontology)
    """
    try:
        _validate_ontology(ontology)
        
        required_classes = ['PlantAnatomy', 'Species', 'ExperimentalCondition']
        
        for class_name in required_classes:
            # Check if class exists
            if not verify_class_accessibility(ontology, class_name):
                logger.warning(f"Required class not found: {class_name}")
                return False
            
            # Validate class properties
            if not validate_source_class_properties(ontology, class_name):
                logger.warning(f"Invalid properties for class: {class_name}")
                return False
            
            # Verify Thing inheritance
            if not verify_thing_inheritance(ontology, class_name):
                logger.warning(f"Class does not inherit from Thing: {class_name}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating core source classes: {e}")
        return False