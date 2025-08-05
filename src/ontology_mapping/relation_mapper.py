"""
Relationship-to-ontology mapping functionality using text2term.

This module provides functionality for mapping extracted relationship triples to ontology 
properties using the text2term library. It supports various mapping methods, minimum 
similarity scores, and semantic consistency validation for comprehensive ontology integration.

Key Features:
- Multiple mapping methods (TFIDF, Levenshtein, Jaro-Winkler, Jaccard, Fuzzy)
- Minimum score filtering for high-confidence mappings
- Support for different term types (property, objectProperty, dataProperty)
- Semantic consistency validation through domain/range checking
- Comprehensive relationship context preservation
- Integration with Owlready2 ontology objects
- Robust error handling and input validation

Usage:
    from src.ontology_mapping.relation_mapper import map_relationships_to_ontology
    
    relationships = [
        ("glucose", "metabolized_by", "enzyme"),
        ("arabidopsis", "has_part", "leaf"),
        ("ATP", "produced_by", "respiration")
    ]
    
    # Using Owlready2 ontology object
    import owlready2
    onto = owlready2.get_ontology("http://purl.obolibrary.org/obo/ro.owl").load()
    results = map_relationships_to_ontology(
        relationships=relationships,
        ontology_obj=onto,
        mapping_method='tfidf',
        min_score=0.8,
        validate_semantics=True
    )
"""

import pandas as pd
import re
from typing import List, Tuple, Optional, Union, Any
from urllib.parse import urlparse

try:
    import text2term
except ImportError:
    # For testing purposes, we'll define a mock text2term module structure
    # The actual import error will be raised at runtime if text2term is needed
    class MockText2Term:
        class Mapper:
            TFIDF = "TFIDF"
            LEVENSHTEIN = "LEVENSHTEIN" 
            JARO_WINKLER = "JARO_WINKLER"
            JACCARD = "JACCARD"
            FUZZY = "FUZZY"
        
        @staticmethod
        def map_terms(**kwargs):
            raise ImportError(
                "text2term is required for relationship mapping functionality. "
                "Install it with: pip install text2term"
            )
    
    text2term = MockText2Term()

# Conditional import of owlready2 to avoid hard dependency
try:
    import owlready2
    OWLREADY2_AVAILABLE = True
except ImportError:
    owlready2 = None
    OWLREADY2_AVAILABLE = False


# Custom Exception Classes
class RelationMapperError(Exception):
    """Base exception for relation mapper errors."""
    pass


class OntologyNotFoundError(RelationMapperError):
    """Exception raised when specified ontology cannot be found or accessed."""
    pass


class MappingError(RelationMapperError):
    """Exception raised when the mapping process fails."""
    pass


class SemanticValidationError(RelationMapperError):
    """Exception raised when semantic validation fails."""
    pass


# Helper Functions for Owlready2 Integration
def _is_owlready2_ontology(obj: Any) -> bool:
    """
    Check if an object is an Owlready2 ontology.
    
    Args:
        obj: Object to check
        
    Returns:
        bool: True if object is an Owlready2 ontology, False otherwise
    """
    if not OWLREADY2_AVAILABLE:
        return False
    
    # Check if object is an instance of owlready2.Ontology
    try:
        return isinstance(obj, owlready2.Ontology)
    except Exception:
        return False


def _extract_iri_from_owlready2_ontology(ontology: Any) -> str:
    """
    Extract IRI from an Owlready2 ontology object.
    
    Args:
        ontology: Owlready2 ontology object
        
    Returns:
        str: IRI of the ontology
        
    Raises:
        ValueError: If ontology object is invalid or has no IRI
    """
    if not OWLREADY2_AVAILABLE:
        raise ValueError(
            "Owlready2 is not available. Install it with: pip install owlready2"
        )
    
    if not _is_owlready2_ontology(ontology):
        raise ValueError(
            "Object is not a valid Owlready2 ontology. Expected owlready2.Ontology instance."
        )
    
    try:
        # Get the ontology IRI
        iri = ontology.base_iri
        
        if not iri:
            raise ValueError(
                "Owlready2 ontology does not have a valid base IRI. "
                "Ensure the ontology is properly loaded and has an IRI."
            )
        
        # Remove trailing slash if present for consistency
        if iri.endswith('/'):
            iri = iri[:-1]
        
        return iri
        
    except AttributeError:
        raise ValueError(
            "Unable to extract IRI from Owlready2 ontology. "
            "The ontology object may be corrupted or improperly loaded."
        )
    except Exception as e:
        raise ValueError(
            f"Error extracting IRI from Owlready2 ontology: {str(e)}"
        )


# Validation Functions
def _validate_relationships(relationships: List[Tuple[str, str, str]]) -> None:
    """
    Validate relationships list input.
    
    Args:
        relationships: List of relationship tuples to validate
        
    Raises:
        ValueError: If relationships list is invalid
    """
    if relationships is None:
        raise ValueError("Relationships list cannot be None")
    
    if not isinstance(relationships, list):
        raise ValueError("Relationships must be a list")
    
    if len(relationships) == 0:
        raise ValueError("Relationships list cannot be empty")
    
    for i, relationship in enumerate(relationships):
        if not isinstance(relationship, tuple):
            raise ValueError(f"Invalid relationship format at index {i}: must be a tuple, got {type(relationship)}")
        
        if len(relationship) != 3:
            raise ValueError(f"Invalid relationship format at index {i}: must have exactly 3 elements (subject, relation, object), got {len(relationship)}")
        
        subject, relation, obj = relationship
        
        for j, element in enumerate([subject, relation, obj]):
            element_names = ["subject", "relation", "object"]
            if not isinstance(element, str):
                raise ValueError(f"Invalid relationship format at index {i}: {element_names[j]} must be a string, got {type(element)}")
            
            if element.strip() == "":
                raise ValueError(f"Invalid relationship format at index {i}: {element_names[j]} cannot be empty or whitespace only")


def _validate_mapping_method(method: str) -> None:
    """
    Validate mapping method parameter.
    
    Args:
        method: Mapping method string to validate
        
    Raises:
        ValueError: If mapping method is invalid
    """
    valid_methods = {'tfidf', 'levenshtein', 'jaro_winkler', 'jaccard', 'fuzzy'}
    
    if not isinstance(method, str):
        raise ValueError("Invalid mapping method: must be a string")
    
    if method not in valid_methods:
        raise ValueError(
            f"Invalid mapping method: '{method}'. "
            f"Valid methods are: {', '.join(sorted(valid_methods))}"
        )


def _validate_ontology_object(ontology_obj: Any) -> str:
    """
    Validate and process ontology object parameter.
    
    Args:
        ontology_obj: Ontology object to validate (expected to be Owlready2 ontology)
        
    Returns:
        str: Extracted ontology IRI
        
    Raises:
        ValueError: If the ontology object is invalid
    """
    if ontology_obj is None:
        raise ValueError("Invalid ontology object: cannot be None")
    
    # Handle Owlready2 ontology object input
    if _is_owlready2_ontology(ontology_obj):
        return _extract_iri_from_owlready2_ontology(ontology_obj)
    
    # Check if it's a mock object with base_iri attribute (for testing)
    if hasattr(ontology_obj, 'base_iri') and isinstance(ontology_obj.base_iri, str):
        iri = ontology_obj.base_iri.strip()
        if not iri:
            raise ValueError("Invalid ontology object: base_iri cannot be empty")
        return iri
    
    # Invalid input type
    raise ValueError(
        f"Invalid ontology object: must be an Owlready2 ontology object, got {type(ontology_obj)}. "
        "If using Owlready2, ensure it's installed with: pip install owlready2"
    )


def _validate_min_score(min_score: float) -> None:
    """
    Validate minimum score parameter.
    
    Args:
        min_score: Minimum score value to validate
        
    Raises:
        ValueError: If minimum score is invalid
    """
    if not isinstance(min_score, (int, float)):
        raise ValueError("Minimum score must be between 0.0 and 1.0")
    
    if not (0.0 <= min_score <= 1.0):
        raise ValueError("Minimum score must be between 0.0 and 1.0")


def _validate_term_type(term_type: str) -> None:
    """
    Validate term type parameter.
    
    Args:
        term_type: Term type string to validate
        
    Raises:
        ValueError: If term type is invalid
    """
    valid_term_types = {'property', 'objectProperty', 'dataProperty'}
    
    if not isinstance(term_type, str):
        raise ValueError("Invalid term type: must be a string")
    
    if term_type not in valid_term_types:
        raise ValueError(
            f"Invalid term type: '{term_type}'. "
            f"Valid term types are: {', '.join(sorted(valid_term_types))}"
        )


# Utility Functions
def _clean_relationships(relationships: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    Clean relationship tuples by removing leading/trailing whitespace.
    
    Args:
        relationships: List of relationship tuples to clean
        
    Returns:
        List of cleaned relationship tuples
    """
    cleaned = []
    for subject, relation, obj in relationships:
        cleaned.append((subject.strip(), relation.strip(), obj.strip()))
    return cleaned


def _extract_relation_terms(relationships: List[Tuple[str, str, str]]) -> List[str]:
    """
    Extract relation terms from relationship tuples.
    
    Args:
        relationships: List of relationship tuples
        
    Returns:
        List of relation terms (middle element of each tuple)
    """
    return [relation for _, relation, _ in relationships]


def _get_text2term_mapper(mapping_method: str):
    """
    Get the appropriate text2term Mapper enum value.
    
    Args:
        mapping_method: String name of the mapping method
        
    Returns:
        text2term.Mapper enum value
    """
    mapper_mapping = {
        'tfidf': text2term.Mapper.TFIDF,
        'levenshtein': text2term.Mapper.LEVENSHTEIN,
        'jaro_winkler': text2term.Mapper.JARO_WINKLER,
        'jaccard': text2term.Mapper.JACCARD,
        'fuzzy': text2term.Mapper.FUZZY
    }
    
    return mapper_mapping[mapping_method]


def _process_mapping_results(
    relationships: List[Tuple[str, str, str]], 
    mapping_df: pd.DataFrame,
    validate_semantics: Optional[bool] = None,
    ontology_obj: Any = None
) -> pd.DataFrame:
    """
    Process and combine mapping results with relationship context.
    
    Args:
        relationships: List of original relationship tuples
        mapping_df: DataFrame with text2term mapping results
        validate_semantics: Whether to perform semantic validation
        ontology_obj: Ontology object for semantic validation
        
    Returns:
        Processed DataFrame with relationship context and mappings
    """
    if mapping_df.empty:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'Subject', 'Relation', 'Object', 'Mapped_Relation_Label',
            'Mapped_Relation_IRI', 'Mapping_Score', 'Term_Type', 'Semantic_Valid'
        ])
    
    # Create a copy to avoid modifying the original
    processed_df = mapping_df.copy()
    
    # Remove rows with null values in critical columns
    critical_columns = ['Source Term', 'Mapped Term IRI', 'Mapping Score']
    for col in critical_columns:
        if col in processed_df.columns:
            processed_df = processed_df.dropna(subset=[col])
    
    # Remove rows with empty string IRIs
    if 'Mapped Term IRI' in processed_df.columns:
        processed_df = processed_df[processed_df['Mapped Term IRI'].str.strip() != '']
    
    # Create mapping from relation terms to mapped properties
    relation_mappings = {}
    for _, row in processed_df.iterrows():
        source_term = row['Source Term']
        relation_mappings[source_term] = {
            'Mapped_Relation_Label': row.get('Mapped Term Label', ''),
            'Mapped_Relation_IRI': row['Mapped Term IRI'],
            'Mapping_Score': row['Mapping Score'],
            'Term_Type': row.get('Term Type', '')
        }
    
    # Build result rows by matching relationships with mappings
    result_rows = []
    for subject, relation, obj in relationships:
        if relation in relation_mappings:
            mapping_info = relation_mappings[relation]
            
            # Perform semantic validation based on the validation flag
            semantic_valid = None
            if validate_semantics is True:
                # Explicitly enabled - always validate
                try:
                    semantic_valid = _validate_semantic_consistency(
                        subject, relation, obj, 
                        mapping_info['Mapped_Relation_IRI'], ontology_obj
                    )
                except SemanticValidationError:
                    # Re-raise semantic validation errors
                    raise
                except Exception:
                    semantic_valid = False
            elif validate_semantics is None and ontology_obj is not None:
                # Default behavior - validate when ontology is available
                try:
                    semantic_valid = _validate_semantic_consistency(
                        subject, relation, obj, 
                        mapping_info['Mapped_Relation_IRI'], ontology_obj
                    )
                except SemanticValidationError:
                    # Re-raise semantic validation errors
                    raise
                except Exception:
                    semantic_valid = False
            # If validate_semantics is False, semantic_valid stays None
            
            result_row = {
                'Subject': subject,
                'Relation': relation,
                'Object': obj,
                'Mapped_Relation_Label': mapping_info['Mapped_Relation_Label'],
                'Mapped_Relation_IRI': mapping_info['Mapped_Relation_IRI'],
                'Mapping_Score': mapping_info['Mapping_Score'],
                'Term_Type': mapping_info['Term_Type'],
                'Semantic_Valid': semantic_valid
            }
            result_rows.append(result_row)
    
    result_df = pd.DataFrame(result_rows)
    
    # Reset index after processing
    result_df = result_df.reset_index(drop=True)
    
    return result_df


def _filter_by_score(df: pd.DataFrame, min_score: float) -> pd.DataFrame:
    """
    Filter mapping results by minimum score threshold.
    
    Args:
        df: DataFrame with mapping results
        min_score: Minimum score threshold
        
    Returns:
        Filtered DataFrame with mappings above threshold
    """
    if df.empty or 'Mapping_Score' not in df.columns:
        return df
    
    # Filter by minimum score
    filtered_df = df[df['Mapping_Score'] >= min_score]
    
    # Reset index after filtering
    filtered_df = filtered_df.reset_index(drop=True)
    
    return filtered_df


def _get_domain_range_constraints(ontology_obj: Any, property_iri: str) -> Tuple[List[str], List[str]]:
    """
    Extract domain and range constraints from ontology property.
    
    Args:
        ontology_obj: Owlready2 ontology object
        property_iri: IRI of the property to analyze
        
    Returns:
        Tuple of (domain_classes, range_classes) as lists of class names
    """
    if not OWLREADY2_AVAILABLE or not _is_owlready2_ontology(ontology_obj):
        return ([], [])
    
    try:
        # Extract property name from IRI (handle both / and # separators)
        if '#' in property_iri:
            property_name = property_iri.split('#')[-1]
        else:
            property_name = property_iri.split('/')[-1]
        
        # Search for property in ontology by IRI first (most reliable)
        properties = ontology_obj.search(iri=property_iri)
        
        # If not found by IRI, try searching by name
        if not properties:
            properties = ontology_obj.search(name=property_name)
        
        # If still not found, try searching with underscore-to-space conversion
        if not properties and '_' in property_name:
            space_name = property_name.replace('_', ' ')
            properties = ontology_obj.search(name=space_name)
        
        # If still not found, try searching with spaces-to-underscore conversion
        if not properties and ' ' in property_name:
            underscore_name = property_name.replace(' ', '_')
            properties = ontology_obj.search(name=underscore_name)
        
        if not properties:
            return ([], [])
        
        property_obj = properties[0]
        
        # Extract domain constraints
        domain_classes = []
        if hasattr(property_obj, 'domain') and property_obj.domain:
            for domain_class in property_obj.domain:
                if hasattr(domain_class, 'name') and domain_class.name:
                    domain_classes.append(domain_class.name)
                elif hasattr(domain_class, 'iri') and domain_class.iri:
                    # Extract class name from IRI if name is not available
                    class_name = domain_class.iri.split('#')[-1].split('/')[-1]
                    if class_name:
                        domain_classes.append(class_name)
        
        # Extract range constraints
        range_classes = []
        if hasattr(property_obj, 'range') and property_obj.range:
            for range_class in property_obj.range:
                if hasattr(range_class, 'name') and range_class.name:
                    range_classes.append(range_class.name)
                elif hasattr(range_class, 'iri') and range_class.iri:
                    # Extract class name from IRI if name is not available
                    class_name = range_class.iri.split('#')[-1].split('/')[-1]
                    if class_name:
                        range_classes.append(class_name)
        
        return (domain_classes, range_classes)
        
    except Exception:
        return ([], [])


def _classify_entity_type(entity: str, ontology_obj: Any) -> List[str]:
    """
    Classify an entity by determining its possible ontological types.
    
    This function attempts to classify an entity by searching for it in the ontology
    and determining what class(es) it might belong to based on various heuristics.
    
    Args:
        entity: Entity name to classify
        ontology_obj: Owlready2 ontology object
        
    Returns:
        List of possible class names that the entity could belong to
    """
    if not OWLREADY2_AVAILABLE or not _is_owlready2_ontology(ontology_obj):
        return []
    
    try:
        possible_types = []
        
        # First, try to find the entity directly in the ontology
        entity_results = ontology_obj.search(name=entity)
        if entity_results:
            for result in entity_results:
                if hasattr(result, 'is_a') and result.is_a:
                    for parent_class in result.is_a:
                        if hasattr(parent_class, 'name') and parent_class.name:
                            possible_types.append(parent_class.name)
        
        # If not found directly, try variations of the entity name
        if not possible_types:
            # Try with underscores replaced by spaces
            if '_' in entity:
                space_entity = entity.replace('_', ' ')
                entity_results = ontology_obj.search(name=space_entity)
                if entity_results:
                    for result in entity_results:
                        if hasattr(result, 'is_a') and result.is_a:
                            for parent_class in result.is_a:
                                if hasattr(parent_class, 'name') and parent_class.name:
                                    possible_types.append(parent_class.name)
            
            # Try with spaces replaced by underscores
            if ' ' in entity:
                underscore_entity = entity.replace(' ', '_')
                entity_results = ontology_obj.search(name=underscore_entity)
                if entity_results:
                    for result in entity_results:
                        if hasattr(result, 'is_a') and result.is_a:
                            for parent_class in result.is_a:
                                if hasattr(parent_class, 'name') and parent_class.name:
                                    possible_types.append(parent_class.name)
        
        # If still no direct matches, use heuristic classification based on entity name patterns
        if not possible_types:
            possible_types = _heuristic_entity_classification(entity)
        
        # Remove duplicates and return
        return list(set(possible_types))
        
    except Exception:
        # Fall back to heuristic classification if ontology search fails
        return _heuristic_entity_classification(entity)


def _heuristic_entity_classification(entity: str) -> List[str]:
    """
    Perform heuristic entity classification based on naming patterns.
    
    This function uses common naming conventions and patterns to classify entities
    when they cannot be found directly in the ontology.
    
    Args:
        entity: Entity name to classify
        
    Returns:
        List of possible class names based on heuristics
    """
    entity_lower = entity.lower()
    possible_types = []
    
    # Chemical entity patterns
    chemical_patterns = [
        'glucose', 'atp', 'nadh', 'nadph', 'acetyl', 'pyruvate', 'lactate',
        'amino_acid', 'fatty_acid', 'protein', 'enzyme', 'hormone', 'drug',
        'compound', 'metabolite', 'cofactor', 'substrate', 'product',
        'inhibitor', 'activator', 'ligand', 'neurotransmitter', 'vitamin',
        'mineral', 'ion', 'salt', 'acid', 'base', 'alcohol', 'ester',
        'aldehyde', 'ketone', 'lipid', 'carbohydrate', 'nucleotide'
    ]
    
    # Biological entity patterns  
    biological_patterns = [
        'cell', 'tissue', 'organ', 'organism', 'bacteria', 'virus',
        'gene', 'chromosome', 'dna', 'rna', 'mrna', 'protein', 'enzyme',
        'receptor', 'antibody', 'antigen', 'membrane', 'organelle',
        'mitochondria', 'nucleus', 'ribosome', 'chloroplast',
        'arabidopsis', 'plant', 'animal', 'human', 'mouse', 'rat'
    ]
    
    # Process patterns
    process_patterns = [
        'photosynthesis', 'respiration', 'glycolysis', 'metabolism',
        'transcription', 'translation', 'replication', 'repair',
        'synthesis', 'degradation', 'transport', 'signaling',
        'regulation', 'development', 'differentiation', 'apoptosis',
        'cell_cycle', 'mitosis', 'meiosis', 'fermentation'
    ]
    
    # Function patterns
    function_patterns = [
        'catalysis', 'binding', 'transport', 'regulation', 'signaling',
        'recognition', 'activation', 'inhibition', 'modulation',
        'protection', 'repair', 'maintenance', 'homeostasis'
    ]
    
    # Check patterns and assign types
    for pattern in chemical_patterns:
        if pattern in entity_lower:
            possible_types.extend(['ChemicalEntity', 'Molecule', 'Compound'])
            break
    
    for pattern in biological_patterns:
        if pattern in entity_lower:
            possible_types.extend(['BiologicalEntity', 'LivingThing'])
            break
    
    for pattern in process_patterns:
        if pattern in entity_lower:
            possible_types.extend(['BiologicalProcess', 'Process'])
            break
    
    for pattern in function_patterns:
        if pattern in entity_lower:
            possible_types.extend(['MolecularFunction', 'Function'])
            break
    
    # Default fallback classifications
    if not possible_types:
        # If entity contains certain keywords, make educated guesses
        if any(keyword in entity_lower for keyword in ['gene', 'protein', 'enzyme']):
            possible_types = ['BiologicalEntity', 'Macromolecule']
        elif any(keyword in entity_lower for keyword in ['cell', 'tissue', 'organ']):
            possible_types = ['AnatomicalEntity', 'BiologicalEntity']
        elif entity_lower.endswith('ase') or entity_lower.endswith('in'):
            # Likely enzyme or protein
            possible_types = ['Protein', 'Enzyme', 'Macromolecule']
        else:
            # Very general fallback
            possible_types = ['Entity', 'Thing']
    
    return list(set(possible_types))


def _check_class_inheritance(entity_types: List[str], constraint_classes: List[str], ontology_obj: Any) -> bool:
    """
    Check if any of the entity types match the constraint classes, considering inheritance.
    
    This function checks if an entity's inferred types are compatible with the domain/range
    constraints of a property, taking into account class inheritance hierarchies.
    
    Args:
        entity_types: List of possible types for the entity
        constraint_classes: List of constraint classes (domain or range)
        ontology_obj: Owlready2 ontology object
        
    Returns:
        bool: True if entity types are compatible with constraints, False otherwise
    """
    if not entity_types or not constraint_classes:
        return True  # Permissive approach when no constraints
    
    if not OWLREADY2_AVAILABLE or not _is_owlready2_ontology(ontology_obj):
        # Fall back to simple string matching if Owlready2 not available
        return bool(set(entity_types) & set(constraint_classes))
    
    try:
        # Direct match check first
        if set(entity_types) & set(constraint_classes):
            return True
        
        # Check inheritance relationships using ontology
        for entity_type in entity_types:
            entity_class_results = ontology_obj.search(name=entity_type)
            if entity_class_results:
                entity_class = entity_class_results[0]
                
                # Get all superclasses (ancestors) of the entity class
                if hasattr(entity_class, 'ancestors'):
                    ancestors = entity_class.ancestors()
                    ancestor_names = []
                    for ancestor in ancestors:
                        if hasattr(ancestor, 'name') and ancestor.name:
                            ancestor_names.append(ancestor.name)
                    
                    # Check if any ancestor matches constraint classes
                    if set(ancestor_names) & set(constraint_classes):
                        return True
                
                # Alternative approach: check is_a relationships recursively
                if hasattr(entity_class, 'is_a'):
                    if _check_is_a_hierarchy(entity_class, constraint_classes, set()):
                        return True
        
        # Check the reverse: if constraint classes are subclasses of entity types
        for constraint_class in constraint_classes:
            constraint_class_results = ontology_obj.search(name=constraint_class)
            if constraint_class_results:
                constraint_class_obj = constraint_class_results[0]
                
                if hasattr(constraint_class_obj, 'ancestors'):
                    ancestors = constraint_class_obj.ancestors()
                    ancestor_names = []
                    for ancestor in ancestors:
                        if hasattr(ancestor, 'name') and ancestor.name:
                            ancestor_names.append(ancestor.name)
                    
                    # Check if any ancestor matches entity types
                    if set(ancestor_names) & set(entity_types):
                        return True
        
        return False
        
    except Exception:
        # Fall back to simple string matching on error
        return bool(set(entity_types) & set(constraint_classes))


def _check_is_a_hierarchy(class_obj: Any, target_classes: List[str], visited: set) -> bool:
    """
    Recursively check is_a hierarchy to find target classes.
    
    Args:
        class_obj: Owlready2 class object to check
        target_classes: List of target class names to find
        visited: Set of already visited classes to avoid cycles
        
    Returns:
        bool: True if any target class is found in hierarchy, False otherwise
    """
    try:
        # Avoid infinite recursion
        if hasattr(class_obj, 'name') and class_obj.name in visited:
            return False
        
        if hasattr(class_obj, 'name') and class_obj.name:
            visited.add(class_obj.name)
            
            # Check if current class matches target
            if class_obj.name in target_classes:
                return True
        
        # Recursively check parent classes
        if hasattr(class_obj, 'is_a') and class_obj.is_a:
            for parent in class_obj.is_a:
                if _check_is_a_hierarchy(parent, target_classes, visited):
                    return True
        
        return False
        
    except Exception:
        return False


def _validate_semantic_consistency(
    subject: str, 
    relation: str, 
    obj: str, 
    property_iri: str, 
    ontology_obj: Any
) -> bool:
    """
    Validate semantic consistency of a relationship against ontology constraints.
    
    This function performs comprehensive domain and range validation by:
    1. Extracting domain/range constraints from the ontology property
    2. Classifying the subject and object entities to determine their types
    3. Checking if entity types conform to domain/range constraints
    4. Handling inheritance hierarchies in the validation process
    
    Args:
        subject: Subject entity of the relationship
        relation: Relation predicate
        obj: Object entity of the relationship
        property_iri: IRI of the mapped ontology property
        ontology_obj: Owlready2 ontology object
        
    Returns:
        bool: True if semantically consistent, False otherwise
        
    Raises:
        SemanticValidationError: If validation process fails
    """
    try:
        # Get domain and range constraints from the ontology property
        domain_classes, range_classes = _get_domain_range_constraints(ontology_obj, property_iri)
        
        # If no constraints are found, consider it valid (permissive approach)
        if not domain_classes and not range_classes:
            return True
        
        # Classify subject and object entities to determine their possible types
        subject_types = _classify_entity_type(subject, ontology_obj)
        object_types = _classify_entity_type(obj, ontology_obj)
        
        # Domain validation: check if subject conforms to domain constraints
        domain_valid = True
        if domain_classes:
            domain_valid = _check_class_inheritance(subject_types, domain_classes, ontology_obj)
        
        # Range validation: check if object conforms to range constraints  
        range_valid = True
        if range_classes:
            range_valid = _check_class_inheritance(object_types, range_classes, ontology_obj)
        
        # Both domain and range must be valid for the relationship to be semantically consistent
        return domain_valid and range_valid
        
    except Exception as e:
        raise SemanticValidationError(f"Semantic validation failed: {str(e)}")


# Main Function
def map_relationships_to_ontology(
    relationships: List[Tuple[str, str, str]],
    ontology_obj: Any,
    mapping_method: str = 'tfidf',
    min_score: float = 0.3,
    term_type: str = 'property',
    validate_semantics: Optional[bool] = None,
    incl_unmapped: bool = False
) -> pd.DataFrame:
    """
    Map relationship triples to ontology properties using text2term.
    
    This function takes a list of relationship triples (subject-relation-object) and maps 
    the relation components to properties in a specified ontology using the text2term library. 
    It supports various mapping methods, filtering options, and semantic validation for 
    high-quality ontology integration.
    
    Args:
        relationships: List of relationship tuples (subject, relation, object) to map
        ontology_obj: Owlready2 ontology object containing target properties
        mapping_method: Method to use for mapping ('tfidf', 'levenshtein', 
                       'jaro_winkler', 'jaccard', 'fuzzy'). Defaults to 'tfidf'.
        min_score: Minimum similarity score threshold (0.0-1.0). Defaults to 0.3.
        term_type: Type of ontology terms to map to ('property', 'objectProperty', 
                  'dataProperty'). Defaults to 'property'.
        validate_semantics: Whether to perform semantic consistency validation
                           through domain/range checking. None (default) enables validation
                           when ontology is available, True always enables, False disables.
        incl_unmapped: Whether to include unmapped relationships in results.
                      Defaults to False.
    
    Returns:
        pandas.DataFrame: DataFrame with columns:
            - 'Subject': Subject entity from the original relationship
            - 'Relation': Relation predicate from the original relationship
            - 'Object': Object entity from the original relationship
            - 'Mapped_Relation_Label': Label of the mapped ontology property
            - 'Mapped_Relation_IRI': IRI of the mapped ontology property
            - 'Mapping_Score': Similarity score (0.0-1.0)
            - 'Term_Type': Type of the ontology property
            - 'Semantic_Valid': Boolean indicating semantic consistency (if validated)
    
    Raises:
        ValueError: If input parameters are invalid
        OntologyNotFoundError: If the specified ontology cannot be found
        MappingError: If the mapping process fails
        SemanticValidationError: If semantic validation fails
        
    Examples:
        Basic relationship mapping:
        >>> relationships = [
        ...     ("glucose", "metabolized_by", "enzyme"),
        ...     ("arabidopsis", "has_part", "leaf"),
        ...     ("ATP", "produced_by", "respiration")
        ... ]
        >>> import owlready2
        >>> onto = owlready2.get_ontology("http://purl.obolibrary.org/obo/ro.owl").load()
        >>> results = map_relationships_to_ontology(
        ...     relationships=relationships,
        ...     ontology_obj=onto,
        ...     mapping_method='tfidf',
        ...     min_score=0.8
        ... )
        >>> print(results)
        
        With semantic validation:
        >>> results = map_relationships_to_ontology(
        ...     relationships=relationships,
        ...     ontology_obj=onto,
        ...     mapping_method='levenshtein',
        ...     min_score=0.7,
        ...     validate_semantics=True
        ... )
        >>> print(results)
    """
    # Input validation
    _validate_relationships(relationships)
    ontology_iri = _validate_ontology_object(ontology_obj)
    _validate_mapping_method(mapping_method)
    _validate_min_score(min_score)
    _validate_term_type(term_type)
    
    # Clean relationships
    cleaned_relationships = _clean_relationships(relationships)
    
    # Extract relation terms for mapping
    relation_terms = _extract_relation_terms(cleaned_relationships)
    
    # Get text2term mapper
    mapper = _get_text2term_mapper(mapping_method)
    
    try:
        # Call text2term mapping function
        mapping_results = text2term.map_terms(
            source_terms=relation_terms,
            target_ontology=ontology_iri,
            mapper=mapper,
            min_score=min_score,
            term_type=term_type,
            incl_unmapped=incl_unmapped
        )
        
    except FileNotFoundError as e:
        raise OntologyNotFoundError(f"Ontology not found: {str(e)}")
    except Exception as e:
        raise MappingError(f"Failed to map relationships: {str(e)}")
    
    # Process and filter results
    try:
        processed_results = _process_mapping_results(
            cleaned_relationships, 
            mapping_results, 
            validate_semantics=validate_semantics,
            ontology_obj=ontology_obj
        )
        filtered_results = _filter_by_score(processed_results, min_score)
        
        return filtered_results
        
    except SemanticValidationError:
        # Re-raise semantic validation errors as-is
        raise
    except Exception as e:
        raise MappingError(f"Failed to process mapping results: {str(e)}")