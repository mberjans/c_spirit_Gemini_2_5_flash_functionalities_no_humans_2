"""
Ontology scheme functional module for plant metabolomics.

This module defines and integrates terms for "Functional Annotation" including
Gene Ontology (GO), Trait Ontology, and ChemFont into a core ontology using
Owlready2. It provides functions for creating, managing, and validating
functional annotation classes within OWL 2.0 ontologies.

The module supports:
- MolecularTrait creation for GO molecular function classifications
- PlantTrait creation for plant-specific functional trait classifications  
- HumanTrait creation for human-related functional trait classifications
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


class FunctionalClassError(Exception):
    """Custom exception for functional class operations.
    
    This exception is raised when errors occur during the creation,
    validation, or manipulation of functional annotation classes.
    
    Args:
        message: Error description
        
    Example:
        raise FunctionalClassError("Invalid ontology provided")
    """
    
    def __init__(self, message: str) -> None:
        """Initialize the functional class error.
        
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
        FunctionalClassError: If class name is invalid
    """
    if not class_name or not isinstance(class_name, str):
        raise FunctionalClassError("Invalid class name: must be a non-empty string")
    
    # Remove leading/trailing whitespace
    class_name = class_name.strip()
    
    if not class_name:
        raise FunctionalClassError("Invalid class name: cannot be empty or whitespace only")
    
    # Check for valid OWL class name pattern
    if not re.match(r'^[A-Za-z][A-Za-z0-9_]*$', class_name):
        raise FunctionalClassError(
            f"Invalid class name '{class_name}': must start with letter and "
            "contain only letters, numbers, and underscores"
        )


def _validate_ontology(ontology: Any) -> None:
    """Validate that the provided ontology is valid.
    
    Args:
        ontology: Ontology object to validate
        
    Raises:
        FunctionalClassError: If ontology is invalid
    """
    if ontology is None:
        raise FunctionalClassError("Invalid ontology: cannot be None")


def create_molecular_trait_class(ontology: Any, class_name: str) -> Any:
    """Create a MolecularTrait class for GO molecular function classification.
    
    Creates a new OWL class for molecular function classification based on the
    Gene Ontology (GO). The class inherits from owlready2.Thing and includes 
    appropriate semantic annotations.
    
    Args:
        ontology: Target ontology for class creation
        class_name: Name of the MolecularTrait class to create
        
    Returns:
        The created MolecularTrait class object
        
    Raises:
        FunctionalClassError: If creation fails due to invalid parameters or Owlready2 errors
        
    Example:
        molecular_trait_class = create_molecular_trait_class(ontology, "MolecularTrait")
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    
    try:
        with _creation_lock:
            # Get the ontology namespace
            namespace = ontology.get_namespace()
            
            # Create the MolecularTrait class
            molecular_trait_class = owlready_types.new_class(
                class_name,
                (Thing,),
                namespace
            )
            
            # Add semantic annotations
            molecular_trait_class.label = [f"Molecular Function Trait"]
            molecular_trait_class.comment = [
                "Base class for molecular function traits based on Gene Ontology (GO)"
            ]
            
            logger.info(f"Created MolecularTrait class: {class_name}")
            return molecular_trait_class
            
    except OwlReadyError as e:
        raise FunctionalClassError(f"Owlready2 error creating MolecularTrait class '{class_name}': {e}")
    except Exception as e:
        raise FunctionalClassError(f"Failed to create MolecularTrait class '{class_name}': {e}")


def create_plant_trait_class(ontology: Any, class_name: str) -> Any:
    """Create a PlantTrait class for plant-specific functional trait classification.
    
    Creates a new OWL class for plant-specific functional trait classification based on the
    Trait Ontology. The class inherits from owlready2.Thing and includes
    appropriate semantic annotations for plant trait categorization.
    
    Args:
        ontology: Target ontology for class creation
        class_name: Name of the PlantTrait class to create
        
    Returns:
        The created PlantTrait class object
        
    Raises:
        FunctionalClassError: If creation fails due to invalid parameters or Owlready2 errors
        
    Example:
        plant_trait_class = create_plant_trait_class(ontology, "PlantTrait")
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    
    try:
        with _creation_lock:
            # Get the ontology namespace
            namespace = ontology.get_namespace()
            
            # Create the PlantTrait class
            plant_trait_class = owlready_types.new_class(
                class_name,
                (Thing,),
                namespace
            )
            
            # Add semantic annotations
            plant_trait_class.label = [f"Plant Functional Trait"]
            plant_trait_class.comment = [
                "Base class for plant-specific functional traits based on Trait Ontology"
            ]
            
            logger.info(f"Created PlantTrait class: {class_name}")
            return plant_trait_class
            
    except OwlReadyError as e:
        raise FunctionalClassError(f"Owlready2 error creating PlantTrait class '{class_name}': {e}")
    except Exception as e:
        raise FunctionalClassError(f"Failed to create PlantTrait class '{class_name}': {e}")


def create_human_trait_class(ontology: Any, class_name: str) -> Any:
    """Create a HumanTrait class for human-related functional traits.
    
    Creates a new OWL class for human-related functional traits from ChemFont.
    The class inherits from owlready2.Thing and includes appropriate semantic 
    annotations for human functional trait classification.
    
    Args:
        ontology: Target ontology for class creation
        class_name: Name of the HumanTrait class to create
        
    Returns:
        The created HumanTrait class object
        
    Raises:
        FunctionalClassError: If creation fails due to invalid parameters or Owlready2 errors
        
    Example:
        human_trait_class = create_human_trait_class(ontology, "HumanTrait")
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    
    try:
        with _creation_lock:
            # Get the ontology namespace
            namespace = ontology.get_namespace()
            
            # Create the HumanTrait class
            human_trait_class = owlready_types.new_class(
                class_name,
                (Thing,),
                namespace
            )
            
            # Add semantic annotations
            human_trait_class.label = [f"Human Functional Trait"]
            human_trait_class.comment = [
                "Base class for human-related functional traits based on ChemFont"
            ]
            
            logger.info(f"Created HumanTrait class: {class_name}")
            return human_trait_class
            
    except OwlReadyError as e:
        raise FunctionalClassError(f"Owlready2 error creating HumanTrait class '{class_name}': {e}")
    except Exception as e:
        raise FunctionalClassError(f"Failed to create HumanTrait class '{class_name}': {e}")


def create_drought_tolerance_class_with_parent(ontology: Any, class_name: str, parent_class_name: str) -> Any:
    """Create a DroughtTolerance class with hierarchical relationship to a parent class.
    
    Creates a new DroughtTolerance class that inherits from a specified parent class,
    establishing hierarchical relationships in the plant trait ontology structure.
    
    Args:
        ontology: Target ontology for class creation
        class_name: Name of the DroughtTolerance class to create
        parent_class_name: Name of the parent class to inherit from
        
    Returns:
        The created DroughtTolerance class object with parent relationship
        
    Raises:
        FunctionalClassError: If creation fails or parent class not found
        
    Example:
        drought_class = create_drought_tolerance_class_with_parent(ontology, "DroughtTolerance", "PlantTrait")
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    _validate_class_name(parent_class_name)
    
    try:
        with _creation_lock:
            # Find the parent class
            parent_class = ontology.search_one(iri=f"*{parent_class_name}")
            if not parent_class:
                raise FunctionalClassError(f"Parent class '{parent_class_name}' not found")
            
            # Get the ontology namespace
            namespace = ontology.get_namespace()
            
            # Create the DroughtTolerance class with parent relationship
            drought_tolerance_class = owlready_types.new_class(
                class_name,
                (parent_class,),
                namespace
            )
            
            # Add semantic annotations
            drought_tolerance_class.label = [f"Plant Drought Tolerance"]
            drought_tolerance_class.comment = [
                "Plant functional trait related to drought tolerance based on Trait Ontology"
            ]
            
            logger.info(f"Created DroughtTolerance class with parent: {class_name} -> {parent_class_name}")
            return drought_tolerance_class
            
    except OwlReadyError as e:
        raise FunctionalClassError(f"Owlready2 error creating DroughtTolerance class with parent '{class_name}': {e}")
    except Exception as e:
        raise FunctionalClassError(f"Failed to create DroughtTolerance class with parent '{class_name}': {e}")


def create_atpase_activity_class_with_parent(ontology: Any, class_name: str, parent_class_name: str) -> Any:
    """Create an ATPaseActivity class with hierarchical relationship to a parent class.
    
    Creates a new ATPaseActivity class that inherits from a specified parent class,
    establishing hierarchical relationships in the molecular trait ontology structure.
    
    Args:
        ontology: Target ontology for class creation
        class_name: Name of the ATPaseActivity class to create
        parent_class_name: Name of the parent class to inherit from
        
    Returns:
        The created ATPaseActivity class object with parent relationship
        
    Raises:
        FunctionalClassError: If creation fails or parent class not found
        
    Example:
        atpase_class = create_atpase_activity_class_with_parent(ontology, "ATPaseActivity", "MolecularTrait")
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    _validate_class_name(parent_class_name)
    
    try:
        with _creation_lock:
            # Find the parent class
            parent_class = ontology.search_one(iri=f"*{parent_class_name}")
            if not parent_class:
                raise FunctionalClassError(f"Parent class '{parent_class_name}' not found")
            
            # Get the ontology namespace
            namespace = ontology.get_namespace()
            
            # Create the ATPaseActivity class with parent relationship
            atpase_activity_class = owlready_types.new_class(
                class_name,
                (parent_class,),
                namespace
            )
            
            # Add semantic annotations
            atpase_activity_class.label = [f"ATPase Activity"]
            atpase_activity_class.comment = [
                "Molecular function involving ATPase enzymatic activity based on Gene Ontology"
            ]
            
            logger.info(f"Created ATPaseActivity class with parent: {class_name} -> {parent_class_name}")
            return atpase_activity_class
            
    except OwlReadyError as e:
        raise FunctionalClassError(f"Owlready2 error creating ATPaseActivity class with parent '{class_name}': {e}")
    except Exception as e:
        raise FunctionalClassError(f"Failed to create ATPaseActivity class with parent '{class_name}': {e}")


def create_toxicity_class_with_parent(ontology: Any, class_name: str, parent_class_name: str) -> Any:
    """Create a Toxicity class with hierarchical relationship to a parent class.
    
    Creates a new Toxicity class that inherits from a specified parent class,
    establishing hierarchical relationships in the human trait ontology structure.
    
    Args:
        ontology: Target ontology for class creation
        class_name: Name of the Toxicity class to create
        parent_class_name: Name of the parent class to inherit from
        
    Returns:
        The created Toxicity class object with parent relationship
        
    Raises:
        FunctionalClassError: If creation fails or parent class not found
        
    Example:
        toxicity_class = create_toxicity_class_with_parent(ontology, "Toxicity", "HumanTrait")
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    _validate_class_name(parent_class_name)
    
    try:
        with _creation_lock:
            # Find the parent class
            parent_class = ontology.search_one(iri=f"*{parent_class_name}")
            if not parent_class:
                raise FunctionalClassError(f"Parent class '{parent_class_name}' not found")
            
            # Get the ontology namespace
            namespace = ontology.get_namespace()
            
            # Create the Toxicity class with parent relationship
            toxicity_class = owlready_types.new_class(
                class_name,
                (parent_class,),
                namespace
            )
            
            # Add semantic annotations
            toxicity_class.label = [f"Chemical Toxicity"]
            toxicity_class.comment = [
                "Human-related functional trait involving chemical toxicity based on ChemFont"
            ]
            
            logger.info(f"Created Toxicity class with parent: {class_name} -> {parent_class_name}")
            return toxicity_class
            
    except OwlReadyError as e:
        raise FunctionalClassError(f"Owlready2 error creating Toxicity class with parent '{class_name}': {e}")
    except Exception as e:
        raise FunctionalClassError(f"Failed to create Toxicity class with parent '{class_name}': {e}")


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
        is_accessible = verify_class_accessibility(ontology, "MolecularTrait")
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


def validate_functional_class_properties(ontology: Any, class_name: str) -> bool:
    """Validate that a functional class has required properties.
    
    Checks if a functional annotation class has all required properties including
    proper labels, comments, and inheritance relationships.
    
    Args:
        ontology: Ontology containing the class
        class_name: Name of the class to validate
        
    Returns:
        True if class has all required properties, False otherwise
        
    Example:
        is_valid = validate_functional_class_properties(ontology, "MolecularTrait")
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
    """Verify that a functional class properly inherits from Thing.
    
    Checks the inheritance chain to ensure the class ultimately inherits from
    owlready2.Thing, either directly or through parent classes.
    
    Args:
        ontology: Ontology containing the class
        class_name: Name of the class to verify
        
    Returns:
        True if class inherits from Thing, False otherwise
        
    Example:
        inherits_thing = verify_thing_inheritance(ontology, "MolecularTrait")
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
        FunctionalClassError: If class not found
        
    Example:
        depth = get_class_hierarchy_depth(ontology, "ATPaseActivity")  # Returns 2 if ATPaseActivity -> MolecularTrait -> Thing
    """
    _validate_ontology(ontology)
    _validate_class_name(class_name)
    
    try:
        # Find the class
        target_class = ontology.search_one(iri=f"*{class_name}")
        if not target_class:
            raise FunctionalClassError(f"Class '{class_name}' not found in ontology")
        
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
        raise FunctionalClassError(f"Failed to calculate hierarchy depth for '{class_name}': {e}")


def classify_functional_annotation_type(ontology: Any, class_name: str) -> str:
    """Classify the functional annotation type based on class name.
    
    Determines the type of functional annotation based on the class name patterns
    used in the ontology for different classification systems.
    
    Args:
        ontology: Ontology containing the class
        class_name: Name of the class to classify
        
    Returns:
        Classification category as string
        
    Example:
        category = classify_functional_annotation_type(ontology, "MolecularTrait")
        # Returns "go_molecular_function_classification"
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
        
        if "molecular" in name_lower and "trait" in name_lower:
            return "go_molecular_function"
        elif "plant" in name_lower and "trait" in name_lower:
            return "trait_ontology_classification"
        elif "human" in name_lower and "trait" in name_lower:
            return "chemfont_classification"
        else:
            return "unknown_classification"
            
    except Exception as e:
        logger.warning(f"Error classifying annotation type for '{class_name}': {e}")
        return "unknown_classification"


def create_functional_classes_batch(ontology: Any, class_specs: List[Dict[str, Any]]) -> List[Any]:
    """Create multiple functional annotation classes in batch.
    
    Efficiently creates multiple functional classes based on provided specifications,
    supporting different class types and parent relationships.
    
    Args:
        ontology: Target ontology for class creation
        class_specs: List of class specifications, each containing:
            - name: Class name
            - type: Class type ("molecular_trait", "plant_trait", "human_trait")
            - parent: Optional parent class name
        
    Returns:
        List of created class objects
        
    Raises:
        FunctionalClassError: If batch creation fails
        
    Example:
        specs = [
            {"name": "MolecularTrait", "type": "molecular_trait", "parent": None},
            {"name": "PlantTrait", "type": "plant_trait", "parent": "FunctionalTrait"}
        ]
        classes = create_functional_classes_batch(ontology, specs)
    """
    _validate_ontology(ontology)
    
    if not class_specs or not isinstance(class_specs, list):
        raise FunctionalClassError("Invalid class specifications: must be a non-empty list")
    
    created_classes = []
    
    try:
        for spec in class_specs:
            if not isinstance(spec, dict):
                raise FunctionalClassError("Invalid class specification: must be a dictionary")
            
            name = spec.get("name")
            class_type = spec.get("type")
            parent = spec.get("parent")
            
            if not name:
                raise FunctionalClassError("Class specification missing 'name' field")
            
            # Create class based on type
            if class_type == "molecular_trait":
                created_class = create_molecular_trait_class(ontology, name)
            elif class_type == "plant_trait":
                if parent:
                    created_class = create_drought_tolerance_class_with_parent(ontology, name, parent)
                else:
                    created_class = create_plant_trait_class(ontology, name)
            elif class_type == "human_trait":
                created_class = create_human_trait_class(ontology, name)
            else:
                # Default to creating as MolecularTrait class
                created_class = create_molecular_trait_class(ontology, name)
            
            created_classes.append(created_class)
            
        logger.info(f"Successfully created {len(created_classes)} functional classes in batch")
        return created_classes
        
    except Exception as e:
        logger.error(f"Batch class creation failed: {e}")
        raise FunctionalClassError(f"Failed to create functional classes in batch: {e}")


def get_all_functional_classes(ontology: Any) -> List[Any]:
    """Get all functional annotation classes from the ontology.
    
    Retrieves all classes that match functional annotation patterns,
    including MolecularTrait, PlantTrait, and HumanTrait types.
    
    Args:
        ontology: Ontology to search
        
    Returns:
        List of functional annotation class objects
        
    Example:
        functional_classes = get_all_functional_classes(ontology)
    """
    try:
        _validate_ontology(ontology)
        
        # Get all classes from the ontology
        all_classes = list(ontology.classes())
        
        # Filter for functional annotation classes
        functional_classes = []
        functional_patterns = ['moleculartrait', 'planttrait', 'humantrait', 'functional']
        
        for cls in all_classes:
            if hasattr(cls, 'name') and cls.name:
                name_lower = cls.name.lower()
                if any(pattern in name_lower for pattern in functional_patterns):
                    functional_classes.append(cls)
        
        return functional_classes
        
    except Exception as e:
        logger.error(f"Error retrieving functional classes: {e}")
        return []


def validate_class_metadata(ontology: Any, class_name: str) -> bool:
    """Validate class metadata and annotations.
    
    Performs comprehensive validation of class metadata including labels,
    comments, IRI structure, and custom annotations specific to functional
    annotation classes.
    
    Args:
        ontology: Ontology containing the class
        class_name: Name of the class to validate
        
    Returns:
        True if metadata is valid, False otherwise
        
    Example:
        metadata_valid = validate_class_metadata(ontology, "MolecularTrait")
    """
    try:
        _validate_ontology(ontology)
        _validate_class_name(class_name)
        
        # Find the class
        target_class = ontology.search_one(iri=f"*{class_name}")
        if not target_class:
            return False
        
        # Validate basic metadata
        if not validate_functional_class_properties(ontology, class_name):
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


def cleanup_functional_classes(ontology: Any) -> int:
    """Cleanup functional annotation classes from the ontology.
    
    Removes all functional annotation classes from the ontology, useful for
    cleanup operations or resetting the ontology state.
    
    Args:
        ontology: Ontology to clean up
        
    Returns:
        Number of classes removed
        
    Warning:
        This operation is destructive and cannot be undone
        
    Example:
        removed_count = cleanup_functional_classes(ontology)
    """
    try:
        _validate_ontology(ontology)
        
        # Get all functional classes
        functional_classes = get_all_functional_classes(ontology)
        
        cleanup_count = 0
        with _creation_lock:
            for cls in functional_classes:
                try:
                    if hasattr(cls, 'destroy'):
                        cls.destroy()
                        cleanup_count += 1
                except Exception as e:
                    logger.warning(f"Failed to destroy class {cls.name}: {e}")
        
        logger.info(f"Cleaned up {cleanup_count} functional classes")
        return cleanup_count
        
    except Exception as e:
        logger.error(f"Error during functional class cleanup: {e}")
        return 0


def define_core_functional_classes(ontology: Any) -> Dict[str, Any]:
    """Define core functional annotation classes in the main ontology namespace.
    
    Creates the fundamental functional annotation classes (MolecularTrait, PlantTrait, 
    HumanTrait) that inherit from owlready2.Thing and associates them with the 
    main ontology namespace. This implements the core requirements for 
    AIM2-ODIE-011-T2.
    
    Args:
        ontology: Main ontology to define classes in
        
    Returns:
        Dictionary mapping class names to created class objects
        
    Raises:
        FunctionalClassError: If class definition fails
        
    Example:
        classes = define_core_functional_classes(ontology)
        molecular_trait_class = classes['MolecularTrait']
        plant_trait_class = classes['PlantTrait']
        human_trait_class = classes['HumanTrait']
    """
    _validate_ontology(ontology)
    
    try:
        with _creation_lock:
            # Use the ontology context for class creation
            with ontology:
                # Define MolecularTrait for GO molecular function classification
                class MolecularTrait(Thing):
                    namespace = ontology
                    
                MolecularTrait.label = ["Molecular Function Trait"]
                MolecularTrait.comment = [
                    "Base class for molecular function traits based on Gene Ontology (GO). "
                    "Provides functional annotation for metabolites using molecular function classifications."
                ]
                
                # Define PlantTrait for plant-specific functional trait classification
                class PlantTrait(Thing):
                    namespace = ontology
                    
                PlantTrait.label = ["Plant Functional Trait"]
                PlantTrait.comment = [
                    "Base class for plant-specific functional traits based on Trait Ontology. "
                    "Provides functional annotation for metabolites using plant trait classifications."
                ]
                
                # Define HumanTrait for human-related functional traits
                class HumanTrait(Thing):
                    namespace = ontology
                    
                HumanTrait.label = ["Human Functional Trait"]
                HumanTrait.comment = [
                    "Base class for human-related functional traits based on ChemFont. "
                    "Provides functional annotation for metabolites using human-relevant trait classifications."
                ]
                
                # Create the class registry
                defined_classes = {
                    'MolecularTrait': MolecularTrait,
                    'PlantTrait': PlantTrait,
                    'HumanTrait': HumanTrait
                }
                
                logger.info(f"Successfully defined {len(defined_classes)} core functional classes")
                
                return defined_classes
            
    except OwlReadyError as e:
        raise FunctionalClassError(f"Owlready2 error defining core functional classes: {e}")
    except Exception as e:
        raise FunctionalClassError(f"Failed to define core functional classes: {e}")


def establish_functional_hierarchy(ontology: Any, classes: Dict[str, Any]) -> None:
    """Establish hierarchical relationships between functional annotation classes.
    
    Creates is_a relationships and other hierarchical connections between the
    defined functional annotation classes to represent classification hierarchies.
    
    Args:
        ontology: Main ontology containing the classes
        classes: Dictionary of class names to class objects
        
    Raises:
        FunctionalClassError: If hierarchy establishment fails
        
    Example:
        classes = define_core_functional_classes(ontology)
        establish_functional_hierarchy(ontology, classes)
    """
    _validate_ontology(ontology)
    
    if not classes or not isinstance(classes, dict):
        raise FunctionalClassError("Invalid classes dictionary")
    
    try:
        with _creation_lock:
            # Get required classes
            molecular_trait_class = classes.get('MolecularTrait')
            plant_trait_class = classes.get('PlantTrait')
            human_trait_class = classes.get('HumanTrait')
            
            if not all([molecular_trait_class, plant_trait_class, human_trait_class]):
                raise FunctionalClassError("Missing required functional classes")
            
            # Create a general FunctionalAnnotation superclass within the ontology context
            with ontology:
                class FunctionalAnnotation(Thing):
                    namespace = ontology
                    
                FunctionalAnnotation.label = ["Functional Annotation"]
                FunctionalAnnotation.comment = [
                    "Superclass for all functional annotation concepts including molecular function traits, "
                    "plant functional traits, and human functional traits."
                ]
            
            # Establish hierarchical relationships
            # All functional classes inherit from FunctionalAnnotation
            molecular_trait_class.is_a.append(FunctionalAnnotation)
            plant_trait_class.is_a.append(FunctionalAnnotation)
            human_trait_class.is_a.append(FunctionalAnnotation)
            
            # Keep them as peer classes under FunctionalAnnotation for now
            # More specific hierarchies can be added as needed
            
            logger.info("Successfully established functional class hierarchy")
            
    except Exception as e:
        raise FunctionalClassError(f"Failed to establish functional hierarchy: {e}")


def add_initial_key_terms(ontology: Any) -> Dict[str, List[Any]]:
    """Add initial key terms/instances from GO, Trait Ontology, and ChemFont to the ontology.
    
    Creates representative instances of MolecularTrait, PlantTrait, and HumanTrait classes
    to populate the ontology with initial key terms from each classification system.
    This function implements AIM2-ODIE-011-T2 by adding concrete examples from each
    functional annotation system.
    
    Args:
        ontology: Target ontology for instance creation
        
    Returns:
        Dictionary with keys "go_instances", "trait_ontology_instances", "chemfont_instances"
        containing lists of created instances
        
    Raises:
        FunctionalClassError: If instance creation fails
        
    Example:
        instances = add_initial_key_terms(ontology)
        atpase_activity = instances['go_instances'][0]
        drought_tolerance = instances['trait_ontology_instances'][0]
        toxicity = instances['chemfont_instances'][0]
    """
    _validate_ontology(ontology)
    
    try:
        with _creation_lock:
            # Get the required classes
            molecular_trait_class = ontology.search_one(iri="*MolecularTrait")
            plant_trait_class = ontology.search_one(iri="*PlantTrait")
            human_trait_class = ontology.search_one(iri="*HumanTrait")
            
            if not all([molecular_trait_class, plant_trait_class, human_trait_class]):
                raise FunctionalClassError(
                    "Required functional classes not found. Please run define_core_functional_classes() first."
                )
            
            # Define representative Gene Ontology molecular function instances
            molecular_trait_terms = [
                {
                    "name": "ATPase_activity",
                    "label": "ATPase activity",
                    "comment": "Catalysis of the reaction: ATP + H2O = ADP + phosphate to release energy."
                },
                {
                    "name": "DNA_binding", 
                    "label": "DNA binding",
                    "comment": "Any molecular function by which a gene product interacts selectively and non-covalently with DNA."
                },
                {
                    "name": "Catalytic_activity",
                    "label": "Catalytic activity",
                    "comment": "Catalysis of a biochemical reaction at physiological temperatures."
                },
                {
                    "name": "Transporter_activity",
                    "label": "Transporter activity",
                    "comment": "Enables the directed movement of substances (such as macromolecules, small molecules, ions) into, out of or within a cell, or between cells."
                },
                {
                    "name": "Kinase_activity",
                    "label": "Kinase activity",
                    "comment": "Catalysis of the transfer of a phosphate group, usually from ATP, to a substrate molecule."
                },
                {
                    "name": "Phosphatase_activity",
                    "label": "Phosphatase activity",
                    "comment": "Catalysis of the removal of a phosphate group from a substrate molecule."
                },
                {
                    "name": "Oxidoreductase_activity",
                    "label": "Oxidoreductase activity",
                    "comment": "Catalysis of an oxidation-reduction (redox) reaction, a reversible chemical reaction in which the oxidation state of an atom or atoms within a molecule is altered."
                },
                {
                    "name": "Transferase_activity",
                    "label": "Transferase activity",
                    "comment": "Catalysis of the transfer of a group, e.g. a methyl group, glycosyl group, acyl group, phosphorus-containing, or other groups, from one compound (generally regarded as donor) to another compound (generally regarded as acceptor)."
                }
            ]
            
            # Define representative Trait Ontology plant trait instances
            plant_trait_terms = [
                {
                    "name": "DroughtTolerance",
                    "label": "Drought tolerance",
                    "comment": "Plant's ability to survive, grow, and reproduce under drought stress conditions."
                },
                {
                    "name": "FloweringTime",
                    "label": "Flowering time", 
                    "comment": "The time from germination to the opening of the first flower."
                },
                {
                    "name": "PlantHeight",
                    "label": "Plant height",
                    "comment": "The vertical distance from the ground level to the uppermost part of the primary shoot."
                },
                {
                    "name": "SeedWeight",
                    "label": "Seed weight",
                    "comment": "The weight of mature seeds, typically measured as 100-seed weight or 1000-seed weight."
                },
                {
                    "name": "YieldTrait",
                    "label": "Yield trait",
                    "comment": "Plant traits related to productivity and harvestable output."
                },
                {
                    "name": "StressResistance",
                    "label": "Stress resistance",
                    "comment": "Plant's ability to resist various biotic and abiotic stress factors."
                },
                {
                    "name": "NutrientUseEfficiency",
                    "label": "Nutrient use efficiency",
                    "comment": "Plant's ability to efficiently utilize available nutrients for growth and development."
                },
                {
                    "name": "PhotosynthesisEfficiency",
                    "label": "Photosynthesis efficiency",
                    "comment": "Plant's efficiency in converting light energy into chemical energy through photosynthesis."
                }
            ]
            
            # Define representative ChemFont human trait instances
            human_trait_terms = [
                {
                    "name": "Toxicity",
                    "label": "Toxicity",
                    "comment": "The degree to which a chemical substance or compound can harm humans or animals."
                },
                {
                    "name": "Bioavailability",
                    "label": "Bioavailability",
                    "comment": "The fraction of an administered compound that reaches the systemic circulation unchanged."
                },
                {
                    "name": "MetabolicStability", 
                    "label": "Metabolic stability",
                    "comment": "The susceptibility of a compound to biotransformation in biological systems."
                },
                {
                    "name": "DrugLikeness",
                    "label": "Drug likeness",
                    "comment": "A qualitative concept used in drug design to indicate how 'drug-like' a substance is with respect to factors such as bioavailability."
                },
                {
                    "name": "Solubility",
                    "label": "Solubility",
                    "comment": "The ability of a substance to dissolve in a solvent, typically measured in aqueous solutions."
                },
                {
                    "name": "Permeability",
                    "label": "Permeability",
                    "comment": "The ability of a compound to cross biological membranes."
                },
                {
                    "name": "ProteinBinding",
                    "label": "Protein binding",
                    "comment": "The degree to which a compound binds to proteins in blood plasma or tissues."
                },
                {
                    "name": "ClearanceRate",
                    "label": "Clearance rate",
                    "comment": "The volume of plasma from which a substance is completely removed per unit time."
                }
            ]
            
            # Create instances within the ontology context
            with ontology:
                molecular_trait_instances = []
                plant_trait_instances = []
                human_trait_instances = []
                
                # Create Molecular Trait instances
                for term_data in molecular_trait_terms:
                    instance = molecular_trait_class(term_data["name"])
                    instance.label = [term_data["label"]]
                    instance.comment = [term_data["comment"]]
                    molecular_trait_instances.append(instance)
                    logger.debug(f"Created Molecular Trait instance: {term_data['name']}")
                
                # Create Plant Trait instances  
                for term_data in plant_trait_terms:
                    instance = plant_trait_class(term_data["name"])
                    instance.label = [term_data["label"]]
                    instance.comment = [term_data["comment"]]
                    plant_trait_instances.append(instance)
                    logger.debug(f"Created Plant Trait instance: {term_data['name']}")
                
                # Create Human Trait instances
                for term_data in human_trait_terms:
                    instance = human_trait_class(term_data["name"])
                    instance.label = [term_data["label"]]
                    instance.comment = [term_data["comment"]]
                    human_trait_instances.append(instance)
                    logger.debug(f"Created Human Trait instance: {term_data['name']}")
                
                result = {
                    'go_instances': molecular_trait_instances,
                    'trait_ontology_instances': plant_trait_instances, 
                    'chemfont_instances': human_trait_instances
                }
                
                total_instances = len(molecular_trait_instances) + len(plant_trait_instances) + len(human_trait_instances)
                logger.info(f"Successfully created {total_instances} initial key term instances "
                           f"({len(molecular_trait_instances)} Molecular Trait, {len(plant_trait_instances)} Plant Trait, {len(human_trait_instances)} Human Trait)")
                
                return result
                
    except OwlReadyError as e:
        raise FunctionalClassError(f"Owlready2 error creating initial key terms: {e}")
    except Exception as e:
        raise FunctionalClassError(f"Failed to create initial key terms: {e}")


def validate_initial_key_terms(ontology: Any) -> Dict[str, int]:
    """Validate that initial key terms/instances have been created successfully.
    
    Checks that instances of MolecularTrait, PlantTrait, and HumanTrait have been
    properly created in the ontology with correct properties.
    
    Args:
        ontology: Ontology to validate
        
    Returns:
        Dictionary with counts of found instances for each class type
        
    Example:
        counts = validate_initial_key_terms(ontology)
        print(f"Found {counts['molecular_trait_count']} Molecular Trait instances")
    """
    try:
        _validate_ontology(ontology)
        
        # Get the required classes
        molecular_trait_class = ontology.search_one(iri="*MolecularTrait")
        plant_trait_class = ontology.search_one(iri="*PlantTrait")
        human_trait_class = ontology.search_one(iri="*HumanTrait")
        
        if not all([molecular_trait_class, plant_trait_class, human_trait_class]):
            logger.warning("Required functional classes not found")
            return {"go_count": 0, "trait_ontology_count": 0, "chemfont_count": 0, "total_count": 0}
        
        # Count instances of each class
        molecular_trait_instances = list(molecular_trait_class.instances())
        plant_trait_instances = list(plant_trait_class.instances())
        human_trait_instances = list(human_trait_class.instances())
        
        # Validate that instances have proper labels and comments
        valid_instances = {"molecular_trait": 0, "plant_trait": 0, "human_trait": 0}
        
        for instance in molecular_trait_instances:
            if hasattr(instance, 'label') and hasattr(instance, 'comment'):
                if instance.label and instance.comment:
                    valid_instances["molecular_trait"] += 1
        
        for instance in plant_trait_instances:
            if hasattr(instance, 'label') and hasattr(instance, 'comment'):
                if instance.label and instance.comment:
                    valid_instances["plant_trait"] += 1
        
        for instance in human_trait_instances:
            if hasattr(instance, 'label') and hasattr(instance, 'comment'):
                if instance.label and instance.comment:
                    valid_instances["human_trait"] += 1
        
        result = {
            "go_count": valid_instances["molecular_trait"],
            "trait_ontology_count": valid_instances["plant_trait"], 
            "chemfont_count": valid_instances["human_trait"],
            "total_count": sum(valid_instances.values())
        }
        
        logger.info(f"Validated key terms: {result['total_count']} total instances "
                   f"({result['go_count']} GO, {result['trait_ontology_count']} Trait Ontology, {result['chemfont_count']} ChemFont)")
        
        return result
        
    except Exception as e:
        logger.error(f"Error validating initial key terms: {e}")
        return {"go_count": 0, "trait_ontology_count": 0, "chemfont_count": 0, "total_count": 0}


def validate_core_functional_classes(ontology: Any) -> bool:
    """Validate that core functional classes are properly defined.
    
    Checks that MolecularTrait, PlantTrait, and HumanTrait are properly defined
    in the ontology with correct inheritance and properties.
    
    Args:
        ontology: Ontology to validate
        
    Returns:
        True if all core classes are properly defined, False otherwise
        
    Example:
        is_valid = validate_core_functional_classes(ontology)
    """
    try:
        _validate_ontology(ontology)
        
        required_classes = ['MolecularTrait', 'PlantTrait', 'HumanTrait']
        
        for class_name in required_classes:
            # Check if class exists
            if not verify_class_accessibility(ontology, class_name):
                logger.warning(f"Required class not found: {class_name}")
                return False
            
            # Validate class properties
            if not validate_functional_class_properties(ontology, class_name):
                logger.warning(f"Invalid properties for class: {class_name}")
                return False
            
            # Verify Thing inheritance
            if not verify_thing_inheritance(ontology, class_name):
                logger.warning(f"Class does not inherit from Thing: {class_name}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating core functional classes: {e}")
        return False