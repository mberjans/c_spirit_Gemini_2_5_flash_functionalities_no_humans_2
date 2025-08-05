"""
Relationship schemas for LLM-based relationship extraction in plant metabolomics research.

This module defines comprehensive relationship schemas that specify valid relationship patterns
between entities in plant metabolomics, biology, and related scientific domains. Each schema
provides structured definitions of subject-predicate-object patterns with domain and range
constraints to ensure biologically meaningful relationships.

The schemas are designed to support the AIM2-ODIE ontology development and information
extraction system by defining:
1. Valid relationship types between different entity categories
2. Domain and range constraints (which entity types can participate)
3. Specific biological relationships like "Compound-Affects-Trait"
4. Semantic validation and type checking

Functions:
    get_plant_metabolomics_relationship_schema: Get comprehensive relationship schema
    get_basic_relationship_schema: Get basic relationship schema with core patterns
    validate_relationship_pattern: Validate subject-predicate-object patterns
    get_compatible_relationships: Get relationships compatible with entity types
    filter_relationships_by_domain: Filter relationships by domain constraints

Constants:
    PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA: Complete relationship schema
    BASIC_RELATIONSHIP_SCHEMA: Simplified schema with essential relationships
    DOMAIN_RANGE_CONSTRAINTS: Entity type compatibility rules
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass


@dataclass
class RelationshipPattern:
    """
    Structured definition of a relationship pattern with constraints.
    
    Attributes:
        relation_type: The relationship type/predicate
        description: Human-readable description of the relationship
        domain: Set of valid subject entity types
        range: Set of valid object entity types
        inverse: Optional inverse relationship type
        symmetric: Whether the relationship is symmetric
        transitive: Whether the relationship is transitive
        examples: Example relationships of this type
    """
    relation_type: str
    description: str
    domain: Set[str]
    range: Set[str]
    inverse: Optional[str] = None
    symmetric: bool = False
    transitive: bool = False
    examples: Optional[List[Tuple[str, str]]] = None


# Comprehensive relationship schema for plant metabolomics research
PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA = {
    # Metabolite-related relationships
    "synthesized_by": RelationshipPattern(
        relation_type="synthesized_by",
        description="Metabolite is synthesized/produced by an organism, enzyme, or biological system",
        domain={"METABOLITE", "COMPOUND", "PHENOLIC_COMPOUND", "FLAVONOID", "ALKALOID", "TERPENOID", "LIPID", "CARBOHYDRATE", "AMINO_ACID", "ORGANIC_ACID"},
        range={"SPECIES", "PLANT_SPECIES", "ORGANISM", "ENZYME", "PROTEIN", "PLANT_PART", "PLANT_ORGAN", "PLANT_TISSUE", "CELL_TYPE"},
        inverse="synthesizes",
        examples=[("anthocyanins", "grape berries"), ("quercetin", "chalcone synthase")]
    ),
    
    "found_in": RelationshipPattern(
        relation_type="found_in",
        description="Metabolite is found/detected in a specific plant part, species, or biological sample",
        domain={"METABOLITE", "COMPOUND", "PHENOLIC_COMPOUND", "FLAVONOID", "ALKALOID", "TERPENOID", "LIPID", "CARBOHYDRATE", "AMINO_ACID", "ORGANIC_ACID"},
        range={"SPECIES", "PLANT_SPECIES", "PLANT_PART", "PLANT_ORGAN", "PLANT_TISSUE", "CELL_TYPE", "CELLULAR_COMPONENT", "ROOT", "LEAF", "STEM", "FLOWER", "FRUIT", "SEED"},
        inverse="contains",
        examples=[("resveratrol", "grape skin"), ("caffeine", "coffee beans")]
    ),
    
    "accumulates_in": RelationshipPattern(
        relation_type="accumulates_in",
        description="Metabolite accumulates or concentrates in a specific plant part, tissue, or cellular location",
        domain={"METABOLITE", "COMPOUND", "PHENOLIC_COMPOUND", "FLAVONOID", "ALKALOID", "TERPENOID", "LIPID", "CARBOHYDRATE", "AMINO_ACID", "ORGANIC_ACID"},
        range={"PLANT_PART", "PLANT_ORGAN", "PLANT_TISSUE", "CELL_TYPE", "CELLULAR_COMPONENT", "ROOT", "LEAF", "STEM", "FLOWER", "FRUIT", "SEED"},
        examples=[("starch", "root tubers"), ("anthocyanins", "flower petals")]
    ),
    
    "derived_from": RelationshipPattern(
        relation_type="derived_from",
        description="Metabolite is derived from another compound or precursor through biochemical transformation",
        domain={"METABOLITE", "COMPOUND", "PHENOLIC_COMPOUND", "FLAVONOID", "ALKALOID", "TERPENOID", "LIPID", "CARBOHYDRATE", "AMINO_ACID", "ORGANIC_ACID"},
        range={"METABOLITE", "COMPOUND", "PHENOLIC_COMPOUND", "FLAVONOID", "ALKALOID", "TERPENOID", "LIPID", "CARBOHYDRATE", "AMINO_ACID", "ORGANIC_ACID"},
        inverse="precursor_of",
        transitive=True,
        examples=[("quercetin", "naringenin"), ("anthocyanins", "flavonols")]
    ),
    
    "converted_to": RelationshipPattern(
        relation_type="converted_to",
        description="Metabolite is converted to another compound through enzymatic or chemical transformation",
        domain={"METABOLITE", "COMPOUND", "PHENOLIC_COMPOUND", "FLAVONOID", "ALKALOID", "TERPENOID", "LIPID", "CARBOHYDRATE", "AMINO_ACID", "ORGANIC_ACID"},
        range={"METABOLITE", "COMPOUND", "PHENOLIC_COMPOUND", "FLAVONOID", "ALKALOID", "TERPENOID", "LIPID", "CARBOHYDRATE", "AMINO_ACID", "ORGANIC_ACID"},
        examples=[("phenylalanine", "cinnamate"), ("glucose", "fructose")]
    ),
    
    "precursor_of": RelationshipPattern(
        relation_type="precursor_of",
        description="Metabolite is a precursor of another compound in biosynthetic pathway",
        domain={"METABOLITE", "COMPOUND", "PHENOLIC_COMPOUND", "FLAVONOID", "ALKALOID", "TERPENOID", "LIPID", "CARBOHYDRATE", "AMINO_ACID", "ORGANIC_ACID"},
        range={"METABOLITE", "COMPOUND", "PHENOLIC_COMPOUND", "FLAVONOID", "ALKALOID", "TERPENOID", "LIPID", "CARBOHYDRATE", "AMINO_ACID", "ORGANIC_ACID"},
        inverse="derived_from",
        transitive=True,
        examples=[("naringenin", "quercetin"), ("flavonols", "anthocyanins")]
    ),
    
    "made_via": RelationshipPattern(
        relation_type="made_via",
        description="Metabolite is produced via a specific metabolic pathway or biosynthetic process",
        domain={"METABOLITE", "COMPOUND", "PHENOLIC_COMPOUND", "FLAVONOID", "ALKALOID", "TERPENOID", "LIPID", "CARBOHYDRATE", "AMINO_ACID", "ORGANIC_ACID"},
        range={"METABOLIC_PATHWAY", "BIOSYNTHESIS", "ENZYME_ACTIVITY"},
        examples=[("flavonoids", "phenylpropanoid pathway"), ("terpenoids", "mevalonate pathway")]
    ),
    
    # Gene/Protein-related relationships
    "encodes": RelationshipPattern(
        relation_type="encodes",
        description="Gene encodes a specific protein, enzyme, or functional RNA",
        domain={"GENE"},
        range={"PROTEIN", "ENZYME", "TRANSCRIPTION_FACTOR"},
        inverse="encoded_by",
        examples=[("CHS gene", "chalcone synthase"), ("PAL gene", "phenylalanine ammonia-lyase")]
    ),
    
    "expressed_in": RelationshipPattern(
        relation_type="expressed_in",
        description="Gene is expressed in a specific tissue, organ, or developmental stage",
        domain={"GENE", "PROTEIN", "ENZYME", "TRANSCRIPTION_FACTOR"},
        range={"PLANT_PART", "PLANT_ORGAN", "PLANT_TISSUE", "CELL_TYPE", "DEVELOPMENTAL_STAGE", "SPECIES", "PLANT_SPECIES"},
        examples=[("anthocyanin biosynthesis genes", "flower petals"), ("root-specific genes", "root tissue")]
    ),
    
    "regulated_by": RelationshipPattern(
        relation_type="regulated_by",
        description="Gene, protein, or process is regulated by another molecular factor",
        domain={"GENE", "PROTEIN", "ENZYME", "TRANSCRIPTION_FACTOR", "GENE_EXPRESSION", "ENZYME_ACTIVITY", "METABOLIC_PATHWAY"},
        range={"TRANSCRIPTION_FACTOR", "PROTEIN", "ENZYME", "EXPERIMENTAL_CONDITION", "STRESS_CONDITION", "ENVIRONMENTAL_FACTOR"},
        inverse="regulates",
        examples=[("flavonoid genes", "MYB transcription factors"), ("enzyme activity", "temperature stress")]
    ),
    
    "upregulates": RelationshipPattern(
        relation_type="upregulates",
        description="Factor increases the expression, activity, or abundance of target",
        domain={"TRANSCRIPTION_FACTOR", "PROTEIN", "ENZYME", "EXPERIMENTAL_CONDITION", "STRESS_CONDITION", "ENVIRONMENTAL_FACTOR", "TREATMENT"},
        range={"GENE", "PROTEIN", "ENZYME", "GENE_EXPRESSION", "ENZYME_ACTIVITY", "METABOLITE_LEVEL", "METABOLIC_PATHWAY"},
        inverse="upregulated_by",
        examples=[("drought stress", "proline biosynthesis"), ("light", "anthocyanin production")]
    ),
    
    "downregulates": RelationshipPattern(
        relation_type="downregulates",
        description="Factor decreases the expression, activity, or abundance of target",
        domain={"TRANSCRIPTION_FACTOR", "PROTEIN", "ENZYME", "EXPERIMENTAL_CONDITION", "STRESS_CONDITION", "ENVIRONMENTAL_FACTOR", "TREATMENT"},
        range={"GENE", "PROTEIN", "ENZYME", "GENE_EXPRESSION", "ENZYME_ACTIVITY", "METABOLITE_LEVEL", "METABOLIC_PATHWAY"},
        inverse="downregulated_by",
        examples=[("cold stress", "photosynthesis"), ("darkness", "chlorophyll biosynthesis")]
    ),
    
    "catalyzes": RelationshipPattern(
        relation_type="catalyzes",
        description="Enzyme catalyzes a specific biochemical reaction or process",
        domain={"ENZYME", "PROTEIN"},
        range={"BIOSYNTHESIS", "METABOLIC_PATHWAY", "ENZYME_ACTIVITY"},
        inverse="catalyzed_by",
        examples=[("chalcone synthase", "flavonoid biosynthesis"), ("rubisco", "carbon fixation")]
    ),
    
    # Pathway and process relationships
    "involved_in": RelationshipPattern(
        relation_type="involved_in",
        description="Entity participates in or is part of a metabolic pathway or biological process",
        domain={"METABOLITE", "COMPOUND", "GENE", "PROTEIN", "ENZYME", "TRANSCRIPTION_FACTOR"},
        range={"METABOLIC_PATHWAY", "BIOSYNTHESIS", "BIOLOGICAL_ACTIVITY", "SIGNALING", "REGULATION"},
        examples=[("phenolic compounds", "plant defense"), ("cytochrome P450", "xenobiotic metabolism")]
    ),
    
    "part_of": RelationshipPattern(
        relation_type="part_of",
        description="Entity is a structural or functional component of a larger system",
        domain={"METABOLITE", "COMPOUND", "GENE", "PROTEIN", "ENZYME", "PLANT_PART", "PLANT_TISSUE", "CELL_TYPE"},
        range={"METABOLIC_PATHWAY", "BIOSYNTHESIS", "PLANT_ORGAN", "PLANT_PART", "CELLULAR_COMPONENT", "SPECIES"},
        transitive=True,
        examples=[("stomata", "leaf epidermis"), ("chloroplasts", "mesophyll cells")]
    ),
    
    "upstream_of": RelationshipPattern(
        relation_type="upstream_of",
        description="Entity acts upstream in a pathway or process relative to another entity",
        domain={"METABOLITE", "COMPOUND", "GENE", "PROTEIN", "ENZYME", "METABOLIC_PATHWAY"},
        range={"METABOLITE", "COMPOUND", "GENE", "PROTEIN", "ENZYME", "METABOLIC_PATHWAY"},
        inverse="downstream_of",
        transitive=True,
        examples=[("shikimate pathway", "phenylpropanoid pathway"), ("PAL", "CHS")]
    ),
    
    "downstream_of": RelationshipPattern(
        relation_type="downstream_of",
        description="Entity acts downstream in a pathway or process relative to another entity",
        domain={"METABOLITE", "COMPOUND", "GENE", "PROTEIN", "ENZYME", "METABOLIC_PATHWAY"},
        range={"METABOLITE", "COMPOUND", "GENE", "PROTEIN", "ENZYME", "METABOLIC_PATHWAY"},
        inverse="upstream_of",
        transitive=True,
        examples=[("anthocyanin biosynthesis", "phenylpropanoid pathway"), ("flavonol synthase", "chalcone synthase")]
    ),
    
    # Experimental and condition relationships
    "responds_to": RelationshipPattern(
        relation_type="responds_to",
        description="Entity responds to experimental treatment, stress, or environmental condition",
        domain={"METABOLITE", "COMPOUND", "GENE", "PROTEIN", "ENZYME", "GENE_EXPRESSION", "ENZYME_ACTIVITY", "METABOLITE_LEVEL", "PLANT_TRAIT", "MOLECULAR_TRAIT"},
        range={"EXPERIMENTAL_CONDITION", "STRESS_CONDITION", "ABIOTIC_STRESS", "BIOTIC_STRESS", "TREATMENT", "ENVIRONMENTAL_FACTOR"},
        examples=[("heat shock proteins", "temperature stress"), ("osmolytes", "drought stress")]
    ),
    
    "affected_by": RelationshipPattern(
        relation_type="affected_by",
        description="Entity is affected by experimental treatment, stress, or environmental factor",
        domain={"METABOLITE", "COMPOUND", "GENE", "PROTEIN", "ENZYME", "GENE_EXPRESSION", "ENZYME_ACTIVITY", "METABOLITE_LEVEL", "PLANT_TRAIT", "MOLECULAR_TRAIT"},
        range={"EXPERIMENTAL_CONDITION", "STRESS_CONDITION", "ABIOTIC_STRESS", "BIOTIC_STRESS", "TREATMENT", "ENVIRONMENTAL_FACTOR", "GROWTH_CONDITION", "DEVELOPMENTAL_STAGE"},
        inverse="affects",
        examples=[("anthocyanin content", "light intensity"), ("root growth", "salt stress")]
    ),
    
    "increases_under": RelationshipPattern(
        relation_type="increases_under",
        description="Entity increases in abundance, activity, or expression under specific conditions",
        domain={"METABOLITE", "COMPOUND", "GENE_EXPRESSION", "ENZYME_ACTIVITY", "METABOLITE_LEVEL", "PROTEIN_ABUNDANCE"},
        range={"EXPERIMENTAL_CONDITION", "STRESS_CONDITION", "ABIOTIC_STRESS", "BIOTIC_STRESS", "TREATMENT", "ENVIRONMENTAL_FACTOR", "GROWTH_CONDITION"},
        examples=[("proline", "drought stress"), ("heat shock proteins", "high temperature")]
    ),
    
    "decreases_under": RelationshipPattern(
        relation_type="decreases_under",
        description="Entity decreases in abundance, activity, or expression under specific conditions",
        domain={"METABOLITE", "COMPOUND", "GENE_EXPRESSION", "ENZYME_ACTIVITY", "METABOLITE_LEVEL", "PROTEIN_ABUNDANCE"},
        range={"EXPERIMENTAL_CONDITION", "STRESS_CONDITION", "ABIOTIC_STRESS", "BIOTIC_STRESS", "TREATMENT", "ENVIRONMENTAL_FACTOR", "GROWTH_CONDITION"},
        examples=[("chlorophyll", "darkness"), ("photosynthesis", "cold stress")]
    ),
    
    # Structural and localization relationships
    "located_in": RelationshipPattern(
        relation_type="located_in",
        description="Entity is located in a specific cellular, tissue, or organ location",
        domain={"METABOLITE", "COMPOUND", "PROTEIN", "ENZYME", "GENE_EXPRESSION", "ENZYME_ACTIVITY"},
        range={"CELLULAR_COMPONENT", "CELL_TYPE", "PLANT_TISSUE", "PLANT_ORGAN", "PLANT_PART", "ROOT", "LEAF", "STEM", "FLOWER", "FRUIT", "SEED"},
        examples=[("chlorophyll", "chloroplasts"), ("starch", "amyloplasts")]
    ),
    
    "binds_to": RelationshipPattern(
        relation_type="binds_to",
        description="Molecule binds to another molecule, protein, or cellular target",
        domain={"METABOLITE", "COMPOUND", "PROTEIN", "ENZYME", "TRANSCRIPTION_FACTOR"},
        range={"METABOLITE", "COMPOUND", "PROTEIN", "ENZYME", "GENE", "CELLULAR_COMPONENT"},
        symmetric=True,
        examples=[("transcription factor", "promoter region"), ("substrate", "enzyme active site")]
    ),
    
    "interacts_with": RelationshipPattern(
        relation_type="interacts_with",
        description="Entity interacts with another entity through physical, chemical, or regulatory mechanisms",
        domain={"METABOLITE", "COMPOUND", "PROTEIN", "ENZYME", "GENE", "TRANSCRIPTION_FACTOR"},
        range={"METABOLITE", "COMPOUND", "PROTEIN", "ENZYME", "GENE", "TRANSCRIPTION_FACTOR", "CELLULAR_COMPONENT"},
        symmetric=True,
        examples=[("protein-protein interaction", "transcriptional complex"), ("metabolite-enzyme interaction", "allosteric regulation")]
    ),
    
    # Phenotypic and trait relationships
    "associated_with": RelationshipPattern(
        relation_type="associated_with",
        description="Entity is statistically or functionally associated with a trait, phenotype, or condition",
        domain={"METABOLITE", "COMPOUND", "GENE", "PROTEIN", "ENZYME", "MOLECULAR_TRAIT", "GENE_EXPRESSION", "ENZYME_ACTIVITY", "METABOLITE_LEVEL"},
        range={"PLANT_TRAIT", "MORPHOLOGICAL_TRAIT", "PHYSIOLOGICAL_TRAIT", "BIOCHEMICAL_TRAIT", "GROWTH_TRAIT", "REPRODUCTIVE_TRAIT", "STRESS_TOLERANCE", "QUALITY_TRAIT", "YIELD_TRAIT", "HUMAN_TRAIT", "DISEASE", "HEALTH_BENEFIT"},
        symmetric=True,
        examples=[("anthocyanins", "flower color"), ("drought tolerance genes", "water use efficiency")]
    ),
    
    "contributes_to": RelationshipPattern(
        relation_type="contributes_to",
        description="Entity contributes to or influences a specific trait, function, or phenotype",
        domain={"METABOLITE", "COMPOUND", "GENE", "PROTEIN", "ENZYME", "MOLECULAR_TRAIT", "GENE_EXPRESSION", "ENZYME_ACTIVITY", "METABOLITE_LEVEL"},
        range={"PLANT_TRAIT", "MORPHOLOGICAL_TRAIT", "PHYSIOLOGICAL_TRAIT", "BIOCHEMICAL_TRAIT", "GROWTH_TRAIT", "REPRODUCTIVE_TRAIT", "STRESS_TOLERANCE", "QUALITY_TRAIT", "YIELD_TRAIT", "BIOLOGICAL_ACTIVITY", "PHARMACOLOGICAL_ACTIVITY"},
        examples=[("lignin", "mechanical strength"), ("antioxidants", "stress tolerance")]
    ),
    
    "required_for": RelationshipPattern(
        relation_type="required_for",
        description="Entity is essential or required for a specific process, trait, or function",
        domain={"METABOLITE", "COMPOUND", "GENE", "PROTEIN", "ENZYME", "MOLECULAR_TRAIT"},
        range={"PLANT_TRAIT", "MORPHOLOGICAL_TRAIT", "PHYSIOLOGICAL_TRAIT", "BIOCHEMICAL_TRAIT", "GROWTH_TRAIT", "REPRODUCTIVE_TRAIT", "METABOLIC_PATHWAY", "BIOSYNTHESIS", "BIOLOGICAL_ACTIVITY"},
        examples=[("chlorophyll", "photosynthesis"), ("auxin", "root development")]
    ),
    
    # Analytical and measurement relationships
    "detected_by": RelationshipPattern(
        relation_type="detected_by",
        description="Entity is detected, identified, or analyzed using a specific analytical method",
        domain={"METABOLITE", "COMPOUND", "PROTEIN", "GENE_EXPRESSION", "ENZYME_ACTIVITY", "METABOLITE_LEVEL"},
        range={"ANALYTICAL_METHOD", "CHROMATOGRAPHY", "MASS_SPECTROMETRY", "SPECTROSCOPY"},
        inverse="detects",
        examples=[("phenolic compounds", "HPLC"), ("volatile compounds", "GC-MS")]
    ),
    
    "measured_with": RelationshipPattern(
        relation_type="measured_with",
        description="Entity is quantified or measured using a specific analytical technique or instrument",
        domain={"METABOLITE", "COMPOUND", "PROTEIN", "GENE_EXPRESSION", "ENZYME_ACTIVITY", "METABOLITE_LEVEL", "PLANT_TRAIT"},
        range={"ANALYTICAL_METHOD", "CHROMATOGRAPHY", "MASS_SPECTROMETRY", "SPECTROSCOPY"},
        examples=[("metabolite concentration", "LC-MS/MS"), ("gene expression", "qRT-PCR")]
    ),
    
    "characterized_by": RelationshipPattern(
        relation_type="characterized_by",
        description="Entity is characterized or described by specific analytical approaches or properties",
        domain={"METABOLITE", "COMPOUND", "PROTEIN", "ENZYME", "PLANT_TRAIT", "MOLECULAR_TRAIT"},
        range={"ANALYTICAL_METHOD", "CHROMATOGRAPHY", "MASS_SPECTROMETRY", "SPECTROSCOPY", "BIOLOGICAL_ACTIVITY", "PHARMACOLOGICAL_ACTIVITY"},
        examples=[("bioactive compounds", "antioxidant assays"), ("protein structure", "X-ray crystallography")]
    ),
    
    # Key requested relationships
    "affects": RelationshipPattern(
        relation_type="affects",
        description="Compound affects or influences a plant trait, phenotype, or biological process",
        domain={"COMPOUND", "METABOLITE", "PHENOLIC_COMPOUND", "FLAVONOID", "ALKALOID", "TERPENOID", "LIPID", "CARBOHYDRATE", "AMINO_ACID", "ORGANIC_ACID"},
        range={"PLANT_TRAIT", "MORPHOLOGICAL_TRAIT", "PHYSIOLOGICAL_TRAIT", "BIOCHEMICAL_TRAIT", "GROWTH_TRAIT", "REPRODUCTIVE_TRAIT", "STRESS_TOLERANCE", "QUALITY_TRAIT", "YIELD_TRAIT", "MOLECULAR_TRAIT", "BIOLOGICAL_ACTIVITY"},
        inverse="affected_by",
        examples=[("salicylic acid", "disease resistance"), ("cytokinins", "cell division")]
    ),
    
    "involved_in_biological_process": RelationshipPattern(
        relation_type="involved_in_biological_process",
        description="Metabolite participates in or is involved in a specific biological process or pathway",
        domain={"METABOLITE", "COMPOUND", "PHENOLIC_COMPOUND", "FLAVONOID", "ALKALOID", "TERPENOID", "LIPID", "CARBOHYDRATE", "AMINO_ACID", "ORGANIC_ACID"},
        range={"METABOLIC_PATHWAY", "BIOSYNTHESIS", "BIOLOGICAL_ACTIVITY", "SIGNALING", "REGULATION", "STRESS_TOLERANCE", "PLANT_TRAIT", "PHYSIOLOGICAL_TRAIT"},
        examples=[("auxin", "root development"), ("abscisic acid", "stomatal closure")]
    )
}


# Basic relationship schema with essential patterns
BASIC_RELATIONSHIP_SCHEMA = {
    "synthesized_by": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["synthesized_by"],
    "found_in": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["found_in"],
    "derived_from": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["derived_from"],
    "encodes": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["encodes"],
    "expressed_in": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["expressed_in"],
    "involved_in": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["involved_in"],
    "responds_to": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["responds_to"],
    "associated_with": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["associated_with"],
    "affects": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["affects"],
    "involved_in_biological_process": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["involved_in_biological_process"]
}


# Domain-range constraint mappings for validation
DOMAIN_RANGE_CONSTRAINTS = {}
for rel_type, pattern in PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA.items():
    DOMAIN_RANGE_CONSTRAINTS[rel_type] = {
        "domain": pattern.domain,
        "range": pattern.range,
        "inverse": pattern.inverse,
        "symmetric": pattern.symmetric,
        "transitive": pattern.transitive
    }


def get_plant_metabolomics_relationship_schema() -> Dict[str, RelationshipPattern]:
    """
    Get the comprehensive plant metabolomics relationship schema.
    
    Returns:
        Dictionary mapping relationship types to their pattern definitions
    """
    return PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA.copy()


def get_basic_relationship_schema() -> Dict[str, RelationshipPattern]:
    """
    Get the basic relationship schema with essential patterns.
    
    Returns:
        Dictionary mapping basic relationship types to their pattern definitions
    """
    return BASIC_RELATIONSHIP_SCHEMA.copy()


def validate_relationship_pattern(
    subject_entity_type: str,
    relation_type: str,
    object_entity_type: str,
    schema: Optional[Dict[str, RelationshipPattern]] = None
) -> bool:
    """
    Validate if a subject-predicate-object pattern is valid according to schema constraints.
    
    Args:
        subject_entity_type: Type of the subject entity
        relation_type: Type of the relationship
        object_entity_type: Type of the object entity
        schema: Relationship schema to validate against (defaults to full schema)
        
    Returns:
        True if the pattern is valid according to domain/range constraints
        
    Raises:
        ValueError: For invalid input parameters
    """
    if schema is None:
        schema = PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA
    
    if relation_type not in schema:
        return False
    
    pattern = schema[relation_type]
    
    # Check domain constraint (subject entity type)
    if subject_entity_type not in pattern.domain:
        return False
    
    # Check range constraint (object entity type)
    if object_entity_type not in pattern.range:
        return False
    
    return True


def get_compatible_relationships(
    subject_entity_type: str,
    object_entity_type: str,
    schema: Optional[Dict[str, RelationshipPattern]] = None
) -> List[str]:
    """
    Get all relationship types that are compatible with given subject and object entity types.
    
    Args:
        subject_entity_type: Type of the subject entity
        object_entity_type: Type of the object entity
        schema: Relationship schema to check against (defaults to full schema)
        
    Returns:
        List of compatible relationship types
    """
    if schema is None:
        schema = PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA
    
    compatible_relations = []
    
    for relation_type, pattern in schema.items():
        if (subject_entity_type in pattern.domain and 
            object_entity_type in pattern.range):
            compatible_relations.append(relation_type)
    
    return compatible_relations


def filter_relationships_by_domain(
    domain_entities: Set[str],
    schema: Optional[Dict[str, RelationshipPattern]] = None
) -> Dict[str, RelationshipPattern]:
    """
    Filter relationship schema to include only relationships compatible with given entity types.
    
    Args:
        domain_entities: Set of entity types to filter by
        schema: Relationship schema to filter (defaults to full schema)
        
    Returns:
        Filtered relationship schema
    """
    if schema is None:
        schema = PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA
    
    filtered_schema = {}
    
    for relation_type, pattern in schema.items():
        # Include relationship if any domain or range entities match
        if (pattern.domain.intersection(domain_entities) or 
            pattern.range.intersection(domain_entities)):
            filtered_schema[relation_type] = pattern
    
    return filtered_schema


def get_relationship_schema_by_category() -> Dict[str, List[str]]:
    """
    Get relationship types organized by category.
    
    Returns:
        Dictionary mapping categories to lists of relationship types
    """
    return {
        "metabolite_relationships": [
            "synthesized_by", "found_in", "accumulates_in", "derived_from", 
            "converted_to", "made_via", "affects", "involved_in_biological_process"
        ],
        "gene_protein_relationships": [
            "encodes", "expressed_in", "regulated_by", "upregulates", 
            "downregulates", "catalyzes"
        ],
        "pathway_relationships": [
            "involved_in", "part_of", "upstream_of", "downstream_of"
        ],
        "experimental_relationships": [
            "responds_to", "affected_by", "increases_under", "decreases_under"
        ],
        "structural_relationships": [
            "located_in", "binds_to", "interacts_with"
        ],
        "phenotypic_relationships": [
            "associated_with", "contributes_to", "required_for"
        ],
        "analytical_relationships": [
            "detected_by", "measured_with", "characterized_by"
        ]
    }


def validate_relationship_schema(schema: Dict[str, RelationshipPattern]) -> bool:
    """
    Validate relationship schema format and consistency.
    
    Args:
        schema: Relationship schema to validate
        
    Returns:
        True if schema is valid
        
    Raises:
        ValueError: For invalid schema format
    """
    if schema is None:
        raise ValueError("Schema cannot be None")
    
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a dictionary")
    
    if not schema:
        raise ValueError("Schema cannot be empty")
    
    for relation_type, pattern in schema.items():
        if not isinstance(relation_type, str):
            raise ValueError(f"Relation type must be string, got {type(relation_type)}")
        
        if not relation_type.strip():
            raise ValueError("Relation type cannot be empty")
        
        if not isinstance(pattern, RelationshipPattern):
            raise ValueError(f"Pattern for '{relation_type}' must be RelationshipPattern instance")
        
        if pattern.relation_type != relation_type:
            raise ValueError(f"Pattern relation_type '{pattern.relation_type}' doesn't match key '{relation_type}'")
        
        if not pattern.description.strip():
            raise ValueError(f"Pattern description for '{relation_type}' cannot be empty")
        
        if not pattern.domain:
            raise ValueError(f"Pattern domain for '{relation_type}' cannot be empty")
        
        if not pattern.range:
            raise ValueError(f"Pattern range for '{relation_type}' cannot be empty")
        
        # Validate inverse relationships
        if pattern.inverse:
            if pattern.inverse in schema:
                inverse_pattern = schema[pattern.inverse]
                if inverse_pattern.inverse != relation_type:
                    raise ValueError(f"Inverse relationship inconsistency: '{relation_type}' <-> '{pattern.inverse}'")
    
    return True


def get_relationship_statistics(schema: Dict[str, RelationshipPattern]) -> Dict[str, Any]:
    """
    Get statistics about a relationship schema.
    
    Args:
        schema: Relationship schema to analyze
        
    Returns:
        Dictionary with schema statistics
    """
    validate_relationship_schema(schema)
    
    all_domain_types = set()
    all_range_types = set()
    symmetric_count = 0
    transitive_count = 0
    inverse_pairs = 0
    
    for pattern in schema.values():
        all_domain_types.update(pattern.domain)
        all_range_types.update(pattern.range)
        if pattern.symmetric:
            symmetric_count += 1
        if pattern.transitive:
            transitive_count += 1
        if pattern.inverse:
            inverse_pairs += 1
    
    return {
        "total_relationships": len(schema),
        "unique_domain_types": len(all_domain_types),
        "unique_range_types": len(all_range_types),
        "symmetric_relationships": symmetric_count,
        "transitive_relationships": transitive_count,
        "relationships_with_inverse": inverse_pairs,
        "domain_types": sorted(all_domain_types),
        "range_types": sorted(all_range_types)
    }


def convert_schema_to_simple_dict(schema: Dict[str, RelationshipPattern]) -> Dict[str, str]:
    """
    Convert relationship schema to simple dictionary format for compatibility.
    
    Args:
        schema: Relationship schema with RelationshipPattern objects
        
    Returns:
        Dictionary mapping relationship types to descriptions
    """
    return {
        relation_type: pattern.description 
        for relation_type, pattern in schema.items()
    }


def get_domain_specific_schema(domain: str) -> Dict[str, RelationshipPattern]:
    """
    Get domain-specific relationship schema.
    
    Args:
        domain: Domain name (metabolomics, genetics, biochemistry, etc.)
        
    Returns:
        Dictionary mapping domain-specific relationship types to patterns
        
    Raises:
        ValueError: For unsupported domain names
    """
    domain = domain.lower().strip()
    
    if domain in ["metabolomics", "plant_metabolomics", "metabolite"]:
        return {
            "synthesized_by": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["synthesized_by"],
            "found_in": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["found_in"],
            "accumulates_in": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["accumulates_in"],
            "derived_from": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["derived_from"],
            "made_via": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["made_via"],
            "affects": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["affects"],
            "involved_in_biological_process": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["involved_in_biological_process"],
            "detected_by": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["detected_by"]
        }
    
    elif domain in ["genetics", "genomics", "molecular_biology"]:
        return {
            "encodes": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["encodes"],
            "expressed_in": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["expressed_in"],
            "regulated_by": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["regulated_by"],
            "upregulates": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["upregulates"],
            "downregulates": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["downregulates"],
            "catalyzes": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["catalyzes"],
            "involved_in": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["involved_in"]
        }
    
    elif domain in ["biochemistry", "enzymology"]:
        return {
            "catalyzes": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["catalyzes"],
            "involved_in": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["involved_in"],
            "upstream_of": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["upstream_of"],
            "downstream_of": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["downstream_of"],
            "part_of": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["part_of"],
            "binds_to": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["binds_to"]
        }
    
    elif domain in ["plant_biology", "botany", "plant_science"]:
        return {
            "found_in": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["found_in"],
            "accumulates_in": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["accumulates_in"],
            "expressed_in": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["expressed_in"],
            "located_in": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["located_in"],
            "part_of": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["part_of"],
            "contributes_to": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["contributes_to"],
            "required_for": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["required_for"]
        }
    
    elif domain in ["stress", "plant_stress", "environmental_stress"]:
        return {
            "responds_to": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["responds_to"],
            "affected_by": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["affected_by"],
            "increases_under": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["increases_under"],
            "decreases_under": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["decreases_under"],
            "upregulates": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["upregulates"],
            "downregulates": PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA["downregulates"]
        }
    
    else:
        raise ValueError(f"Unsupported domain: {domain}. Supported domains: metabolomics, genetics, biochemistry, plant_biology, stress")


# Predefined domain-specific schemas for common use cases
METABOLOMICS_RELATIONSHIP_SCHEMA = get_domain_specific_schema("metabolomics")
GENETICS_RELATIONSHIP_SCHEMA = get_domain_specific_schema("genetics")
PLANT_BIOLOGY_RELATIONSHIP_SCHEMA = get_domain_specific_schema("plant_biology")
STRESS_RELATIONSHIP_SCHEMA = get_domain_specific_schema("stress")
BIOCHEMISTRY_RELATIONSHIP_SCHEMA = get_domain_specific_schema("biochemistry")