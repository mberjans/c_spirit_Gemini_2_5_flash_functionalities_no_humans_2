"""
Entity schemas for Named Entity Recognition in plant metabolomics research.

This module defines comprehensive entity type schemas for extracting domain-specific
entities from scientific literature in plant metabolomics, biology, and related fields.
Each schema provides uppercase entity type keys with descriptive definitions to guide
LLM-based entity extraction.

The schemas are designed to cover the six core entity types required for the AIM2-ODIE
ontology development and information extraction system:
1. Plant metabolites (primary and secondary metabolites)
2. Species (plant and organism names)
3. Plant anatomical structures (roots, leaves, stems, etc.)
4. Experimental conditions (treatments, environmental conditions)
5. Molecular traits (molecular characteristics and properties)
6. Plant traits (phenotypic traits and characteristics)

Functions:
    get_plant_metabolomics_schema: Get comprehensive plant metabolomics entity schema
    get_basic_schema: Get basic entity schema with core types
    get_extended_schema: Get extended schema with additional specialized types
    get_schema_by_domain: Get domain-specific schema (metabolomics, genetics, etc.)
    validate_schema: Validate entity schema format
    merge_schemas: Merge multiple schemas into one

Constants:
    PLANT_METABOLOMICS_SCHEMA: Complete schema for plant metabolomics research
    BASIC_SCHEMA: Simplified schema with essential entity types
    EXTENDED_SCHEMA: Comprehensive schema with all entity types
"""

from typing import Dict, List, Optional, Set


# Core entity schema for plant metabolomics research
PLANT_METABOLOMICS_SCHEMA = {
    # Plant Metabolites - Primary and secondary metabolites
    "METABOLITE": "Primary and secondary metabolites including sugars, amino acids, organic acids, phenolic compounds, alkaloids, and terpenoids",
    "COMPOUND": "Chemical compounds including metabolites, drugs, molecular entities, and bioactive substances",
    "PHENOLIC_COMPOUND": "Phenolic compounds and derivatives including flavonoids, phenolic acids, tannins, and lignans",
    "FLAVONOID": "Flavonoid compounds including anthocyanins, flavonols, flavones, flavanones, and isoflavones",
    "ALKALOID": "Alkaloid compounds including nitrogen-containing plant metabolites and their derivatives",
    "TERPENOID": "Terpenoid compounds including monoterpenes, sesquiterpenes, diterpenes, and triterpenes",
    "LIPID": "Lipid compounds including fatty acids, phospholipids, glycolipids, and waxes",
    "CARBOHYDRATE": "Carbohydrate compounds including sugars, oligosaccharides, polysaccharides, and glycosides",
    "AMINO_ACID": "Amino acids including proteinogenic and non-proteinogenic amino acids and their derivatives",
    "ORGANIC_ACID": "Organic acids including citric acid, malic acid, and other carboxylic acids",
    
    # Species - Plant and organism names
    "SPECIES": "Organism species names including binomial nomenclature and common names",
    "PLANT_SPECIES": "Plant species names including crops, wild plants, and model organisms",
    "ORGANISM": "Organism names including plants, bacteria, fungi, and other microorganisms",
    "CULTIVAR": "Plant cultivars, varieties, and breeding lines",
    "ECOTYPE": "Plant ecotypes and natural variants",
    
    # Plant Anatomical Structures - Organs, tissues, and cellular components
    "PLANT_PART": "Plant anatomical structures and tissues including organs, tissues, and cellular components",
    "PLANT_ORGAN": "Plant organs including roots, stems, leaves, flowers, fruits, and seeds",
    "PLANT_TISSUE": "Plant tissues including vascular, dermal, and ground tissues",
    "CELL_TYPE": "Plant cell types including mesophyll, epidermal, guard, and parenchyma cells",
    "CELLULAR_COMPONENT": "Cellular components including organelles, membranes, and subcellular structures",
    "ROOT": "Root structures including primary roots, lateral roots, and root hairs",
    "LEAF": "Leaf structures including blade, petiole, and leaf modifications",
    "STEM": "Stem structures including shoots, nodes, internodes, and stem modifications",
    "FLOWER": "Flower structures including petals, sepals, stamens, and pistils",
    "FRUIT": "Fruit structures including pericarp, seeds, and fruit types",
    "SEED": "Seed structures including embryo, endosperm, and seed coat",
    
    # Experimental Conditions - Treatments and environmental factors
    "EXPERIMENTAL_CONDITION": "Experimental treatments and environmental conditions applied to plants",
    "STRESS_CONDITION": "Stress conditions including abiotic and biotic stresses",
    "ABIOTIC_STRESS": "Abiotic stress conditions including drought, salt, temperature, and light stress",
    "BIOTIC_STRESS": "Biotic stress conditions including pathogen attack, herbivory, and competition",
    "TREATMENT": "Experimental treatments including chemical applications, physical manipulations, and interventions",
    "ENVIRONMENTAL_FACTOR": "Environmental factors including temperature, humidity, light, and soil conditions",
    "GROWTH_CONDITION": "Plant growth conditions including media, nutrients, and culture conditions",
    "DEVELOPMENTAL_STAGE": "Plant developmental stages including germination, vegetative growth, flowering, and senescence",
    "TIME_POINT": "Temporal aspects including sampling times, treatment durations, and developmental timing",
    
    # Molecular Traits - Molecular characteristics and properties
    "MOLECULAR_TRAIT": "Molecular characteristics and properties including gene expression, enzyme activity, and metabolite levels",
    "GENE_EXPRESSION": "Gene expression levels, patterns, and regulation",
    "ENZYME_ACTIVITY": "Enzyme activity levels, kinetics, and functional properties",
    "METABOLITE_LEVEL": "Metabolite concentrations, abundance, and accumulation patterns",
    "PROTEIN_ABUNDANCE": "Protein expression levels and abundance",
    "METABOLIC_PATHWAY": "Biochemical and metabolic pathways including biosynthesis and degradation pathways",
    "BIOSYNTHESIS": "Biosynthetic processes and pathways for metabolite production",
    "REGULATION": "Regulatory mechanisms including transcriptional, post-transcriptional, and metabolic regulation",
    "SIGNALING": "Cell signaling pathways and signal transduction mechanisms",
    
    # Plant Traits - Phenotypic traits and characteristics
    "PLANT_TRAIT": "Plant phenotypic traits and characteristics including morphological, physiological, and biochemical traits",
    "MORPHOLOGICAL_TRAIT": "Morphological traits including size, shape, color, and structural features",
    "PHYSIOLOGICAL_TRAIT": "Physiological traits including growth rate, photosynthesis, and metabolic processes",
    "BIOCHEMICAL_TRAIT": "Biochemical traits including metabolite profiles, enzyme activities, and chemical compositions",
    "GROWTH_TRAIT": "Growth-related traits including height, biomass, and developmental timing",
    "REPRODUCTIVE_TRAIT": "Reproductive traits including flowering time, seed production, and fertility",
    "STRESS_TOLERANCE": "Stress tolerance traits including drought tolerance, salt tolerance, and disease resistance",
    "QUALITY_TRAIT": "Quality traits including nutritional content, taste, and processing characteristics",
    "YIELD_TRAIT": "Yield-related traits including productivity, harvest index, and economic yield",
    
    # Additional supporting entity types
    "GENE": "Gene names and genetic elements including protein-coding genes, regulatory genes, and genetic markers",
    "PROTEIN": "Protein names and enzyme identifiers including structural proteins, enzymes, and regulatory proteins",
    "ENZYME": "Enzyme names and classifications including EC numbers and enzyme families",
    "TRANSCRIPTION_FACTOR": "Transcription factors and regulatory proteins controlling gene expression",
    "ANALYTICAL_METHOD": "Analytical techniques and instruments for metabolite analysis and characterization",
    "CHROMATOGRAPHY": "Chromatographic methods including LC, GC, and related separation techniques",
    "MASS_SPECTROMETRY": "Mass spectrometry methods including MS/MS, LC-MS, and GC-MS techniques",
    "SPECTROSCOPY": "Spectroscopic methods including NMR, IR, and UV-Vis spectroscopy",
    "BIOLOGICAL_ACTIVITY": "Biological activities and functions including antioxidant, antimicrobial, and therapeutic activities",
    "PHARMACOLOGICAL_ACTIVITY": "Pharmacological activities including drug-like properties and therapeutic effects",
    "HUMAN_TRAIT": "Human health-related traits and conditions relevant to plant metabolite research",
    "DISEASE": "Human diseases and health conditions that may be affected by plant metabolites",
    "HEALTH_BENEFIT": "Health benefits and therapeutic effects of plant metabolites",
    "BIOMARKER": "Biomarkers including metabolic biomarkers and diagnostic indicators"
}


# Basic schema with essential entity types
BASIC_SCHEMA = {
    "METABOLITE": "Primary and secondary metabolites found in plants",
    "SPECIES": "Plant and organism species names",
    "PLANT_PART": "Plant anatomical structures and tissues", 
    "EXPERIMENTAL_CONDITION": "Experimental treatments and environmental conditions",
    "MOLECULAR_TRAIT": "Molecular characteristics and gene expression patterns",
    "PLANT_TRAIT": "Plant phenotypic traits and characteristics",
    "GENE": "Gene names and genetic elements",
    "COMPOUND": "Chemical compounds and molecular entities"
}


# Extended schema with all specialized types
EXTENDED_SCHEMA = PLANT_METABOLOMICS_SCHEMA.copy()


def get_plant_metabolomics_schema() -> Dict[str, str]:
    """
    Get the comprehensive plant metabolomics entity schema.
    
    Returns:
        Dictionary mapping entity types to their descriptions
    """
    return PLANT_METABOLOMICS_SCHEMA.copy()


def get_basic_schema() -> Dict[str, str]:
    """
    Get the basic entity schema with core types.
    
    Returns:
        Dictionary mapping basic entity types to their descriptions
    """
    return BASIC_SCHEMA.copy()


def get_extended_schema() -> Dict[str, str]:
    """
    Get the extended entity schema with all types.
    
    Returns:
        Dictionary mapping all entity types to their descriptions  
    """
    return EXTENDED_SCHEMA.copy()


def get_schema_by_domain(domain: str) -> Dict[str, str]:
    """
    Get domain-specific entity schema.
    
    Args:
        domain: Domain name (metabolomics, genetics, biochemistry, etc.)
        
    Returns:
        Dictionary mapping domain-specific entity types to descriptions
        
    Raises:
        ValueError: For unsupported domain names
    """
    domain = domain.lower().strip()
    
    if domain in ["metabolomics", "plant_metabolomics", "metabolite"]:
        return {
            "METABOLITE": PLANT_METABOLOMICS_SCHEMA["METABOLITE"],
            "COMPOUND": PLANT_METABOLOMICS_SCHEMA["COMPOUND"],
            "PHENOLIC_COMPOUND": PLANT_METABOLOMICS_SCHEMA["PHENOLIC_COMPOUND"],
            "FLAVONOID": PLANT_METABOLOMICS_SCHEMA["FLAVONOID"],
            "ALKALOID": PLANT_METABOLOMICS_SCHEMA["ALKALOID"],
            "TERPENOID": PLANT_METABOLOMICS_SCHEMA["TERPENOID"],
            "LIPID": PLANT_METABOLOMICS_SCHEMA["LIPID"],
            "CARBOHYDRATE": PLANT_METABOLOMICS_SCHEMA["CARBOHYDRATE"],
            "AMINO_ACID": PLANT_METABOLOMICS_SCHEMA["AMINO_ACID"],
            "ORGANIC_ACID": PLANT_METABOLOMICS_SCHEMA["ORGANIC_ACID"],
            "SPECIES": PLANT_METABOLOMICS_SCHEMA["SPECIES"],
            "PLANT_PART": PLANT_METABOLOMICS_SCHEMA["PLANT_PART"],
            "ANALYTICAL_METHOD": PLANT_METABOLOMICS_SCHEMA["ANALYTICAL_METHOD"]
        }
    
    elif domain in ["genetics", "genomics", "molecular_biology"]:
        return {
            "GENE": PLANT_METABOLOMICS_SCHEMA["GENE"],
            "PROTEIN": PLANT_METABOLOMICS_SCHEMA["PROTEIN"],
            "ENZYME": PLANT_METABOLOMICS_SCHEMA["ENZYME"],
            "TRANSCRIPTION_FACTOR": PLANT_METABOLOMICS_SCHEMA["TRANSCRIPTION_FACTOR"],
            "GENE_EXPRESSION": PLANT_METABOLOMICS_SCHEMA["GENE_EXPRESSION"],
            "MOLECULAR_TRAIT": PLANT_METABOLOMICS_SCHEMA["MOLECULAR_TRAIT"],
            "METABOLIC_PATHWAY": PLANT_METABOLOMICS_SCHEMA["METABOLIC_PATHWAY"],
            "REGULATION": PLANT_METABOLOMICS_SCHEMA["REGULATION"],
            "SIGNALING": PLANT_METABOLOMICS_SCHEMA["SIGNALING"],
            "SPECIES": PLANT_METABOLOMICS_SCHEMA["SPECIES"],
            "PLANT_PART": PLANT_METABOLOMICS_SCHEMA["PLANT_PART"]
        }
    
    elif domain in ["biochemistry", "enzymology"]:
        return {
            "ENZYME": PLANT_METABOLOMICS_SCHEMA["ENZYME"],
            "PROTEIN": PLANT_METABOLOMICS_SCHEMA["PROTEIN"],
            "ENZYME_ACTIVITY": PLANT_METABOLOMICS_SCHEMA["ENZYME_ACTIVITY"],
            "METABOLIC_PATHWAY": PLANT_METABOLOMICS_SCHEMA["METABOLIC_PATHWAY"],
            "BIOSYNTHESIS": PLANT_METABOLOMICS_SCHEMA["BIOSYNTHESIS"],
            "COMPOUND": PLANT_METABOLOMICS_SCHEMA["COMPOUND"],
            "METABOLITE": PLANT_METABOLOMICS_SCHEMA["METABOLITE"],
            "BIOCHEMICAL_TRAIT": PLANT_METABOLOMICS_SCHEMA["BIOCHEMICAL_TRAIT"]
        }
    
    elif domain in ["plant_biology", "botany", "plant_science"]:
        return {
            "SPECIES": PLANT_METABOLOMICS_SCHEMA["SPECIES"],
            "PLANT_SPECIES": PLANT_METABOLOMICS_SCHEMA["PLANT_SPECIES"],
            "CULTIVAR": PLANT_METABOLOMICS_SCHEMA["CULTIVAR"],
            "PLANT_PART": PLANT_METABOLOMICS_SCHEMA["PLANT_PART"],
            "PLANT_ORGAN": PLANT_METABOLOMICS_SCHEMA["PLANT_ORGAN"],
            "PLANT_TISSUE": PLANT_METABOLOMICS_SCHEMA["PLANT_TISSUE"],
            "PLANT_TRAIT": PLANT_METABOLOMICS_SCHEMA["PLANT_TRAIT"],
            "MORPHOLOGICAL_TRAIT": PLANT_METABOLOMICS_SCHEMA["MORPHOLOGICAL_TRAIT"],
            "PHYSIOLOGICAL_TRAIT": PLANT_METABOLOMICS_SCHEMA["PHYSIOLOGICAL_TRAIT"],
            "DEVELOPMENTAL_STAGE": PLANT_METABOLOMICS_SCHEMA["DEVELOPMENTAL_STAGE"],
            "GROWTH_CONDITION": PLANT_METABOLOMICS_SCHEMA["GROWTH_CONDITION"]
        }
    
    elif domain in ["stress", "plant_stress", "environmental_stress"]:
        return {
            "STRESS_CONDITION": PLANT_METABOLOMICS_SCHEMA["STRESS_CONDITION"],
            "ABIOTIC_STRESS": PLANT_METABOLOMICS_SCHEMA["ABIOTIC_STRESS"],
            "BIOTIC_STRESS": PLANT_METABOLOMICS_SCHEMA["BIOTIC_STRESS"],
            "ENVIRONMENTAL_FACTOR": PLANT_METABOLOMICS_SCHEMA["ENVIRONMENTAL_FACTOR"],
            "STRESS_TOLERANCE": PLANT_METABOLOMICS_SCHEMA["STRESS_TOLERANCE"],
            "EXPERIMENTAL_CONDITION": PLANT_METABOLOMICS_SCHEMA["EXPERIMENTAL_CONDITION"],
            "TREATMENT": PLANT_METABOLOMICS_SCHEMA["TREATMENT"],
            "SPECIES": PLANT_METABOLOMICS_SCHEMA["SPECIES"],
            "PLANT_PART": PLANT_METABOLOMICS_SCHEMA["PLANT_PART"]
        }
    
    elif domain in ["analytical", "analytical_chemistry", "instrumentation"]:
        return {
            "ANALYTICAL_METHOD": PLANT_METABOLOMICS_SCHEMA["ANALYTICAL_METHOD"],
            "CHROMATOGRAPHY": PLANT_METABOLOMICS_SCHEMA["CHROMATOGRAPHY"],
            "MASS_SPECTROMETRY": PLANT_METABOLOMICS_SCHEMA["MASS_SPECTROMETRY"],
            "SPECTROSCOPY": PLANT_METABOLOMICS_SCHEMA["SPECTROSCOPY"],
            "COMPOUND": PLANT_METABOLOMICS_SCHEMA["COMPOUND"],
            "METABOLITE": PLANT_METABOLOMICS_SCHEMA["METABOLITE"]
        }
    
    elif domain in ["pharmacology", "bioactivity", "health"]:
        return {
            "BIOLOGICAL_ACTIVITY": PLANT_METABOLOMICS_SCHEMA["BIOLOGICAL_ACTIVITY"],
            "PHARMACOLOGICAL_ACTIVITY": PLANT_METABOLOMICS_SCHEMA["PHARMACOLOGICAL_ACTIVITY"],
            "HEALTH_BENEFIT": PLANT_METABOLOMICS_SCHEMA["HEALTH_BENEFIT"],
            "DISEASE": PLANT_METABOLOMICS_SCHEMA["DISEASE"],
            "HUMAN_TRAIT": PLANT_METABOLOMICS_SCHEMA["HUMAN_TRAIT"],
            "BIOMARKER": PLANT_METABOLOMICS_SCHEMA["BIOMARKER"],
            "COMPOUND": PLANT_METABOLOMICS_SCHEMA["COMPOUND"],
            "METABOLITE": PLANT_METABOLOMICS_SCHEMA["METABOLITE"]
        }
    
    else:
        raise ValueError(f"Unsupported domain: {domain}. Supported domains: metabolomics, genetics, biochemistry, plant_biology, stress, analytical, pharmacology")


def get_entity_types_by_category() -> Dict[str, List[str]]:
    """
    Get entity types organized by category.
    
    Returns:
        Dictionary mapping categories to lists of entity types
    """
    return {
        "metabolites": [
            "METABOLITE", "COMPOUND", "PHENOLIC_COMPOUND", "FLAVONOID", 
            "ALKALOID", "TERPENOID", "LIPID", "CARBOHYDRATE", "AMINO_ACID", "ORGANIC_ACID"
        ],
        "species": [
            "SPECIES", "PLANT_SPECIES", "ORGANISM", "CULTIVAR", "ECOTYPE"
        ],
        "plant_anatomy": [
            "PLANT_PART", "PLANT_ORGAN", "PLANT_TISSUE", "CELL_TYPE", "CELLULAR_COMPONENT",
            "ROOT", "LEAF", "STEM", "FLOWER", "FRUIT", "SEED"
        ],
        "experimental_conditions": [
            "EXPERIMENTAL_CONDITION", "STRESS_CONDITION", "ABIOTIC_STRESS", "BIOTIC_STRESS",
            "TREATMENT", "ENVIRONMENTAL_FACTOR", "GROWTH_CONDITION", "DEVELOPMENTAL_STAGE", "TIME_POINT"
        ],
        "molecular_traits": [
            "MOLECULAR_TRAIT", "GENE_EXPRESSION", "ENZYME_ACTIVITY", "METABOLITE_LEVEL",
            "PROTEIN_ABUNDANCE", "METABOLIC_PATHWAY", "BIOSYNTHESIS", "REGULATION", "SIGNALING"
        ],
        "plant_traits": [
            "PLANT_TRAIT", "MORPHOLOGICAL_TRAIT", "PHYSIOLOGICAL_TRAIT", "BIOCHEMICAL_TRAIT",
            "GROWTH_TRAIT", "REPRODUCTIVE_TRAIT", "STRESS_TOLERANCE", "QUALITY_TRAIT", "YIELD_TRAIT"
        ],
        "genetics": [
            "GENE", "PROTEIN", "ENZYME", "TRANSCRIPTION_FACTOR"
        ],
        "analytical": [
            "ANALYTICAL_METHOD", "CHROMATOGRAPHY", "MASS_SPECTROMETRY", "SPECTROSCOPY"  
        ],
        "bioactivity": [
            "BIOLOGICAL_ACTIVITY", "PHARMACOLOGICAL_ACTIVITY", "HUMAN_TRAIT", "DISEASE", 
            "HEALTH_BENEFIT", "BIOMARKER"
        ]
    }


def validate_schema(schema: Dict[str, str]) -> bool:
    """
    Validate entity schema format and requirements.
    
    Args:
        schema: Entity schema dictionary to validate
        
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
    
    for key, value in schema.items():
        if not isinstance(key, str):
            raise ValueError(f"Schema key must be string, got {type(key)}")
        
        if not key.strip():
            raise ValueError("Schema key cannot be empty")
        
        if not key.isupper():
            raise ValueError(f"Schema key '{key}' should be uppercase")
        
        if not isinstance(value, str):
            raise ValueError(f"Schema value for '{key}' must be string, got {type(value)}")
        
        if not value.strip():
            raise ValueError(f"Schema description for '{key}' cannot be empty")
        
        # Check for reasonable description length
        if len(value.strip()) < 10:
            raise ValueError(f"Schema description for '{key}' should be more descriptive")
    
    return True


def merge_schemas(*schemas: Dict[str, str]) -> Dict[str, str]:
    """
    Merge multiple entity schemas into one.
    
    Args:
        *schemas: Variable number of schema dictionaries to merge
        
    Returns:
        Merged schema dictionary
        
    Raises:
        ValueError: For invalid schemas or conflicting entity types
    """
    if not schemas:
        raise ValueError("At least one schema must be provided")
    
    merged = {}
    conflicts = set()
    
    for schema in schemas:
        validate_schema(schema)
        
        for key, value in schema.items():
            if key in merged and merged[key] != value:
                conflicts.add(key)
            merged[key] = value
    
    if conflicts:
        raise ValueError(f"Conflicting entity type definitions found: {', '.join(sorted(conflicts))}")
    
    return merged


def get_schema_statistics(schema: Dict[str, str]) -> Dict[str, int]:
    """
    Get statistics about an entity schema.
    
    Args:
        schema: Entity schema dictionary
        
    Returns:
        Dictionary with schema statistics
    """
    validate_schema(schema)
    
    return {
        "total_entities": len(schema),
        "avg_description_length": sum(len(desc) for desc in schema.values()) // len(schema),
        "min_description_length": min(len(desc) for desc in schema.values()),
        "max_description_length": max(len(desc) for desc in schema.values()),
        "entity_types": sorted(schema.keys())
    }


def filter_schema_by_keywords(schema: Dict[str, str], keywords: List[str], 
                            case_sensitive: bool = False) -> Dict[str, str]:
    """
    Filter entity schema by keywords in descriptions.
    
    Args:
        schema: Entity schema to filter
        keywords: List of keywords to search for
        case_sensitive: Whether to perform case-sensitive matching
        
    Returns:
        Filtered schema containing only entities with matching keywords
    """
    validate_schema(schema)
    
    if not keywords:
        return schema.copy()
    
    filtered = {}
    
    for entity_type, description in schema.items():
        search_text = description if case_sensitive else description.lower()
        search_keywords = keywords if case_sensitive else [kw.lower() for kw in keywords]
        
        if any(keyword in search_text for keyword in search_keywords):
            filtered[entity_type] = description
    
    return filtered


# Predefined domain-specific schemas for common use cases
METABOLOMICS_SCHEMA = get_schema_by_domain("metabolomics")
GENETICS_SCHEMA = get_schema_by_domain("genetics")  
PLANT_BIOLOGY_SCHEMA = get_schema_by_domain("plant_biology")
STRESS_SCHEMA = get_schema_by_domain("stress")
ANALYTICAL_SCHEMA = get_schema_by_domain("analytical")
PHARMACOLOGY_SCHEMA = get_schema_by_domain("pharmacology")