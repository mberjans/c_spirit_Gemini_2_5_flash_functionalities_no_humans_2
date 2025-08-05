"""
Zero-shot and few-shot prompt templates for plant metabolomics NER and relationship extraction.

This module provides comprehensive prompt templates designed for extracting entities and
relationships from scientific literature in plant metabolomics research. The templates 
support Named Entity Recognition (NER) for 117 entity types across 6 main categories,
and sophisticated relationship extraction with hierarchical differentiation and 
contextual understanding.

The templates are optimized for:
- Clear, unambiguous instructions for LLMs
- Structured JSON output format
- Context-aware prompts for scientific literature
- Robust handling of overlapping entities and edge cases
- Integration with existing schema formatting
- Few-shot learning with synthetic examples
- Hierarchical relationship differentiation (specific vs broad)
- Contextual relationship understanding (conditional, temporal, spatial)
- Dynamic example generation and selection

NER Template Categories:
- Basic zero-shot templates for general use
- Detailed zero-shot templates with explicit instructions
- Domain-specific templates for different research contexts
- Precision-focused templates (minimize false positives)
- Recall-focused templates (capture more entities)
- Few-shot templates with synthetic examples
- Adaptive templates with context-aware example selection

Relationship Extraction Template Categories:
- Basic relationship extraction templates
- Hierarchical templates for specificity differentiation
- Contextual templates for conditional relationship understanding
- Multi-type templates for comprehensive relationship coverage
- Few-shot templates with domain-specific relationship examples
- Template management functions for optimal template selection

Functions:
    # NER Functions
    get_basic_zero_shot_template: Get basic zero-shot template
    get_detailed_zero_shot_template: Get detailed template with explicit instructions
    get_precision_focused_template: Get template optimized for precision
    get_recall_focused_template: Get template optimized for recall
    get_domain_specific_template: Get template for specific research domains
    get_scientific_literature_template: Get template optimized for scientific papers
    get_few_shot_template: Get few-shot template with examples
    get_few_shot_basic_template: Get basic few-shot template
    get_few_shot_detailed_template: Get detailed few-shot template
    get_few_shot_precision_template: Get precision-focused few-shot template
    get_few_shot_recall_template: Get recall-focused few-shot template
    get_few_shot_domain_template: Get domain-specific few-shot template
    generate_synthetic_examples: Generate synthetic examples for entity types
    select_examples: Select optimal examples for given context
    validate_template: Validate template format and placeholders
    get_template_by_name: Get template by name with validation
    list_available_templates: List all available template names
    
    # Relationship Extraction Functions
    get_relationship_template: Get relationship extraction template by type
    get_relationship_template_metadata: Get metadata for relationship templates
    generate_relationship_examples: Generate synthetic relationship examples
    format_relationship_schema_for_template: Format schema for template use
    get_compatible_relationships_for_entities: Get compatible relationships for entity types
    select_optimal_relationship_template: Select best template based on text characteristics
    validate_relationship_template_inputs: Validate inputs for relationship extraction
    list_available_relationship_templates: List all relationship templates

Classes:
    TemplateError: Base exception for template-related errors
    InvalidTemplateError: Exception for invalid template format
    TemplateNotFoundError: Exception for missing templates
    TemplateType: Enumeration of available template types
"""

import re
import json
import random
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from .entity_schemas import get_entity_types_by_category, PLANT_METABOLOMICS_SCHEMA


class TemplateError(Exception):
    """Base exception class for template-related errors."""
    pass


class InvalidTemplateError(TemplateError):
    """Exception raised for invalid template format."""
    pass


class TemplateNotFoundError(TemplateError):
    """Exception raised when template is not found."""
    pass


class TemplateType(Enum):
    """Enumeration of available template types."""
    BASIC = "basic"
    DETAILED = "detailed"
    PRECISION = "precision"
    RECALL = "recall"
    SCIENTIFIC = "scientific"
    METABOLOMICS = "metabolomics"
    GENETICS = "genetics"
    PLANT_BIOLOGY = "plant_biology"
    BIOCHEMISTRY = "biochemistry"
    STRESS = "stress"
    ANALYTICAL = "analytical"
    FEW_SHOT_BASIC = "few_shot_basic"
    FEW_SHOT_DETAILED = "few_shot_detailed"
    FEW_SHOT_PRECISION = "few_shot_precision"
    FEW_SHOT_RECALL = "few_shot_recall"
    FEW_SHOT_SCIENTIFIC = "few_shot_scientific"
    FEW_SHOT_METABOLOMICS = "few_shot_metabolomics"
    FEW_SHOT_GENETICS = "few_shot_genetics"
    FEW_SHOT_PLANT_BIOLOGY = "few_shot_plant_biology"
    # Relationship extraction template types
    RELATIONSHIP_BASIC = "relationship_basic"
    RELATIONSHIP_DETAILED = "relationship_detailed"
    RELATIONSHIP_SCIENTIFIC = "relationship_scientific"
    RELATIONSHIP_METABOLOMICS = "relationship_metabolomics"
    RELATIONSHIP_HIERARCHICAL = "relationship_hierarchical"
    RELATIONSHIP_CONTEXTUAL = "relationship_contextual"
    RELATIONSHIP_MULTI_TYPE = "relationship_multi_type"
    RELATIONSHIP_FEW_SHOT_METABOLOMICS = "relationship_few_shot_metabolomics"
    RELATIONSHIP_FEW_SHOT_HIERARCHICAL = "relationship_few_shot_hierarchical"
    RELATIONSHIP_FEW_SHOT_CONTEXTUAL = "relationship_few_shot_contextual"


# Comprehensive synthetic examples database for all 117 entity types
SYNTHETIC_EXAMPLES_DATABASE = {
    # Plant Metabolites (10 types)
    "METABOLITE": [
        {
            "text": "The leaves accumulated high levels of quercetin and kaempferol after UV stress treatment.",
            "entities": [
                {"text": "quercetin", "label": "METABOLITE", "start": 40, "end": 49, "confidence": 0.95},
                {"text": "kaempferol", "label": "METABOLITE", "start": 54, "end": 64, "confidence": 0.95}
            ]
        },
        {
            "text": "Chlorophyll a content decreased while anthocyanin concentrations increased during senescence.",
            "entities": [
                {"text": "Chlorophyll a", "label": "METABOLITE", "start": 0, "end": 13, "confidence": 0.98},
                {"text": "anthocyanin", "label": "METABOLITE", "start": 34, "end": 45, "confidence": 0.96}
            ]
        },
        {
            "text": "HPLC analysis revealed the presence of caffeic acid, rutin, and hesperidin in the extract.",
            "entities": [
                {"text": "caffeic acid", "label": "METABOLITE", "start": 47, "end": 59, "confidence": 0.97},
                {"text": "rutin", "label": "METABOLITE", "start": 61, "end": 66, "confidence": 0.95},
                {"text": "hesperidin", "label": "METABOLITE", "start": 72, "end": 82, "confidence": 0.96}
            ]
        }
    ],
    
    "COMPOUND": [
        {
            "text": "The bioactive compounds include ascorbic acid and tocopherol with antioxidant properties.",
            "entities": [
                {"text": "ascorbic acid", "label": "COMPOUND", "start": 32, "end": 45, "confidence": 0.96},
                {"text": "tocopherol", "label": "COMPOUND", "start": 50, "end": 60, "confidence": 0.94}
            ]
        },
        {
            "text": "Salicylic acid application enhanced the production of secondary metabolites.",
            "entities": [
                {"text": "Salicylic acid", "label": "COMPOUND", "start": 0, "end": 14, "confidence": 0.98}
            ]
        }
    ],
    
    "PHENOLIC_COMPOUND": [
        {
            "text": "The polyphenolic profile showed high levels of gallic acid and ellagic acid.",
            "entities": [
                {"text": "gallic acid", "label": "PHENOLIC_COMPOUND", "start": 47, "end": 58, "confidence": 0.97},
                {"text": "ellagic acid", "label": "PHENOLIC_COMPOUND", "start": 63, "end": 75, "confidence": 0.97}
            ]
        },
        {
            "text": "Ferulic acid and p-coumaric acid are major phenolic compounds in cell walls.",
            "entities": [
                {"text": "Ferulic acid", "label": "PHENOLIC_COMPOUND", "start": 0, "end": 12, "confidence": 0.98},
                {"text": "p-coumaric acid", "label": "PHENOLIC_COMPOUND", "start": 17, "end": 32, "confidence": 0.98}
            ]
        }
    ],
    
    "FLAVONOID": [
        {
            "text": "Flavonoid biosynthesis genes were upregulated, leading to increased apigenin and luteolin production.",
            "entities": [
                {"text": "apigenin", "label": "FLAVONOID", "start": 77, "end": 85, "confidence": 0.96},
                {"text": "luteolin", "label": "FLAVONOID", "start": 90, "end": 98, "confidence": 0.96}
            ]
        },
        {
            "text": "The flowers contained cyanidin-3-glucoside and delphinidin derivatives.",
            "entities": [
                {"text": "cyanidin-3-glucoside", "label": "FLAVONOID", "start": 22, "end": 42, "confidence": 0.98},
                {"text": "delphinidin", "label": "FLAVONOID", "start": 47, "end": 58, "confidence": 0.95}
            ]
        }
    ],
    
    "ALKALOID": [
        {
            "text": "Caffeine and theobromine levels were measured using LC-MS analysis.",
            "entities": [
                {"text": "Caffeine", "label": "ALKALOID", "start": 0, "end": 8, "confidence": 0.98},
                {"text": "theobromine", "label": "ALKALOID", "start": 13, "end": 24, "confidence": 0.97}
            ]
        },
        {
            "text": "The tropane alkaloids atropine and scopolamine were detected in roots.",
            "entities": [
                {"text": "atropine", "label": "ALKALOID", "start": 22, "end": 30, "confidence": 0.96},
                {"text": "scopolamine", "label": "ALKALOID", "start": 35, "end": 46, "confidence": 0.96}
            ]
        }
    ],
    
    "TERPENOID": [
        {
            "text": "Essential oil analysis revealed limonene, pinene, and camphene as major components.",
            "entities": [
                {"text": "limonene", "label": "TERPENOID", "start": 32, "end": 40, "confidence": 0.97},
                {"text": "pinene", "label": "TERPENOID", "start": 42, "end": 48, "confidence": 0.96},
                {"text": "camphene", "label": "TERPENOID", "start": 54, "end": 62, "confidence": 0.95}
            ]
        },
        {
            "text": "The diterpene gibberellic acid regulates stem elongation and flowering.",
            "entities": [
                {"text": "gibberellic acid", "label": "TERPENOID", "start": 14, "end": 30, "confidence": 0.98}
            ]
        }
    ],
    
    "LIPID": [
        {
            "text": "Fatty acid composition showed high oleic acid and linoleic acid content.",
            "entities": [
                {"text": "oleic acid", "label": "LIPID", "start": 39, "end": 49, "confidence": 0.97},
                {"text": "linoleic acid", "label": "LIPID", "start": 54, "end": 67, "confidence": 0.97}
            ]
        },
        {
            "text": "Phosphatidylcholine and phosphatidylethanolamine are major membrane lipids.",
            "entities": [
                {"text": "Phosphatidylcholine", "label": "LIPID", "start": 0, "end": 19, "confidence": 0.98},
                {"text": "phosphatidylethanolamine", "label": "LIPID", "start": 24, "end": 48, "confidence": 0.98}
            ]
        }
    ],
    
    "CARBOHYDRATE": [
        {
            "text": "Starch granules and cellulose fibers were observed in the stem cross-section.",
            "entities": [
                {"text": "Starch", "label": "CARBOHYDRATE", "start": 0, "end": 6, "confidence": 0.96},
                {"text": "cellulose", "label": "CARBOHYDRATE", "start": 20, "end": 29, "confidence": 0.97}
            ]
        },
        {
            "text": "Sucrose and glucose concentrations varied with developmental stage.",
            "entities": [
                {"text": "Sucrose", "label": "CARBOHYDRATE", "start": 0, "end": 7, "confidence": 0.98},
                {"text": "glucose", "label": "CARBOHYDRATE", "start": 12, "end": 19, "confidence": 0.98}
            ]
        }
    ],
    
    "AMINO_ACID": [
        {
            "text": "Free amino acid analysis detected proline, glycine, and tryptophan in roots.",
            "entities": [
                {"text": "proline", "label": "AMINO_ACID", "start": 35, "end": 42, "confidence": 0.97},
                {"text": "glycine", "label": "AMINO_ACID", "start": 44, "end": 51, "confidence": 0.97},
                {"text": "tryptophan", "label": "AMINO_ACID", "start": 57, "end": 67, "confidence": 0.97}
            ]
        },
        {
            "text": "Arginine and lysine content increased under nitrogen-rich conditions.",
            "entities": [
                {"text": "Arginine", "label": "AMINO_ACID", "start": 0, "end": 8, "confidence": 0.98},
                {"text": "lysine", "label": "AMINO_ACID", "start": 13, "end": 19, "confidence": 0.98}
            ]
        }
    ],
    
    "ORGANIC_ACID": [
        {
            "text": "Citric acid and malic acid are the predominant organic acids in fruit tissue.",
            "entities": [
                {"text": "Citric acid", "label": "ORGANIC_ACID", "start": 0, "end": 11, "confidence": 0.98},
                {"text": "malic acid", "label": "ORGANIC_ACID", "start": 16, "end": 26, "confidence": 0.98}
            ]
        },
        {
            "text": "Oxalic acid accumulation was observed in leaves under stress conditions.",
            "entities": [
                {"text": "Oxalic acid", "label": "ORGANIC_ACID", "start": 0, "end": 11, "confidence": 0.97}
            ]
        }
    ],
    
    # Species (5 types)
    "SPECIES": [
        {
            "text": "Comparative analysis between Arabidopsis thaliana and Oryza sativa revealed differences.",
            "entities": [
                {"text": "Arabidopsis thaliana", "label": "SPECIES", "start": 29, "end": 49, "confidence": 0.99},
                {"text": "Oryza sativa", "label": "SPECIES", "start": 54, "end": 66, "confidence": 0.99}
            ]
        },
        {
            "text": "Escherichia coli was used as the bacterial host for transformation experiments.",
            "entities": [
                {"text": "Escherichia coli", "label": "SPECIES", "start": 0, "end": 16, "confidence": 0.99}
            ]
        }
    ],
    
    "PLANT_SPECIES": [
        {
            "text": "Tomato (Solanum lycopersicum) and potato (Solanum tuberosum) were analyzed.",
            "entities": [
                {"text": "Solanum lycopersicum", "label": "PLANT_SPECIES", "start": 8, "end": 28, "confidence": 0.99},
                {"text": "Solanum tuberosum", "label": "PLANT_SPECIES", "start": 43, "end": 60, "confidence": 0.99}
            ]
        },
        {
            "text": "Wild-type Nicotiana benthamiana plants were used for transient expression.",
            "entities": [
                {"text": "Nicotiana benthamiana", "label": "PLANT_SPECIES", "start": 10, "end": 31, "confidence": 0.99}
            ]
        }
    ],
    
    "ORGANISM": [
        {
            "text": "The pathogen Fusarium oxysporum caused wilting symptoms in infected plants.",
            "entities": [
                {"text": "Fusarium oxysporum", "label": "ORGANISM", "start": 13, "end": 31, "confidence": 0.98}
            ]
        },
        {
            "text": "Agrobacterium tumefaciens-mediated transformation was successful.",
            "entities": [
                {"text": "Agrobacterium tumefaciens", "label": "ORGANISM", "start": 0, "end": 25, "confidence": 0.99}
            ]
        }
    ],
    
    "CULTIVAR": [
        {
            "text": "The cultivar 'Golden Delicious' showed higher sugar content than 'Granny Smith'.",
            "entities": [
                {"text": "Golden Delicious", "label": "CULTIVAR", "start": 14, "end": 30, "confidence": 0.96},
                {"text": "Granny Smith", "label": "CULTIVAR", "start": 67, "end": 79, "confidence": 0.96}
            ]
        },
        {
            "text": "Rice variety IR64 was more drought-tolerant than variety Nipponbare.",
            "entities": [
                {"text": "IR64", "label": "CULTIVAR", "start": 13, "end": 17, "confidence": 0.95},
                {"text": "Nipponbare", "label": "CULTIVAR", "start": 58, "end": 68, "confidence": 0.95}
            ]
        }
    ],
    
    "ECOTYPE": [
        {
            "text": "The Columbia ecotype of Arabidopsis showed different flowering time than Landsberg.",
            "entities": [
                {"text": "Columbia", "label": "ECOTYPE", "start": 4, "end": 12, "confidence": 0.94},
                {"text": "Landsberg", "label": "ECOTYPE", "start": 74, "end": 83, "confidence": 0.94}
            ]
        }
    ],
    
    # Plant Anatomical Structures (11 types)
    "PLANT_PART": [
        {
            "text": "Root and shoot biomass were measured separately after harvest.",
            "entities": [
                {"text": "Root", "label": "PLANT_PART", "start": 0, "end": 4, "confidence": 0.96},
                {"text": "shoot", "label": "PLANT_PART", "start": 9, "end": 14, "confidence": 0.96}
            ]
        },
        {
            "text": "Leaf epidermis and mesophyll tissues showed different expression patterns.",
            "entities": [
                {"text": "Leaf", "label": "PLANT_PART", "start": 0, "end": 4, "confidence": 0.95},
                {"text": "epidermis", "label": "PLANT_PART", "start": 5, "end": 14, "confidence": 0.94},
                {"text": "mesophyll", "label": "PLANT_PART", "start": 19, "end": 28, "confidence": 0.94}
            ]
        }
    ],
    
    "PLANT_ORGAN": [
        {
            "text": "Flowers, fruits, and seeds were collected at different developmental stages.",
            "entities": [
                {"text": "Flowers", "label": "PLANT_ORGAN", "start": 0, "end": 7, "confidence": 0.98},
                {"text": "fruits", "label": "PLANT_ORGAN", "start": 9, "end": 15, "confidence": 0.98},
                {"text": "seeds", "label": "PLANT_ORGAN", "start": 21, "end": 26, "confidence": 0.98}
            ]
        }
    ],
    
    "PLANT_TISSUE": [
        {
            "text": "Vascular tissue and cortex showed distinct metabolite profiles.",
            "entities": [
                {"text": "Vascular tissue", "label": "PLANT_TISSUE", "start": 0, "end": 15, "confidence": 0.97},
                {"text": "cortex", "label": "PLANT_TISSUE", "start": 20, "end": 26, "confidence": 0.96}
            ]
        }
    ],
    
    "CELL_TYPE": [
        {
            "text": "Guard cells and epidermal cells regulate gas exchange.",
            "entities": [
                {"text": "Guard cells", "label": "CELL_TYPE", "start": 0, "end": 11, "confidence": 0.97},
                {"text": "epidermal cells", "label": "CELL_TYPE", "start": 16, "end": 31, "confidence": 0.97}
            ]
        }
    ],
    
    "CELLULAR_COMPONENT": [
        {
            "text": "Chloroplasts and mitochondria were isolated for proteomic analysis.",
            "entities": [
                {"text": "Chloroplasts", "label": "CELLULAR_COMPONENT", "start": 0, "end": 12, "confidence": 0.98},
                {"text": "mitochondria", "label": "CELLULAR_COMPONENT", "start": 17, "end": 29, "confidence": 0.98}
            ]
        }
    ],
    
    "ROOT": [
        {
            "text": "Primary root elongation was inhibited by salt stress.",
            "entities": [
                {"text": "Primary root", "label": "ROOT", "start": 0, "end": 12, "confidence": 0.96}
            ]
        }
    ],
    
    "LEAF": [
        {
            "text": "Leaf blade and petiole samples were analyzed separately.",
            "entities": [
                {"text": "Leaf blade", "label": "LEAF", "start": 0, "end": 10, "confidence": 0.96},
                {"text": "petiole", "label": "LEAF", "start": 15, "end": 22, "confidence": 0.95}
            ]
        }
    ],
    
    "STEM": [
        {
            "text": "Stem internode length increased under low light conditions.",
            "entities": [
                {"text": "Stem", "label": "STEM", "start": 0, "end": 4, "confidence": 0.96},
                {"text": "internode", "label": "STEM", "start": 5, "end": 14, "confidence": 0.94}
            ]
        }
    ],
    
    "FLOWER": [
        {
            "text": "Petal color and sepal morphology varied among genotypes.",
            "entities": [
                {"text": "Petal", "label": "FLOWER", "start": 0, "end": 5, "confidence": 0.95},
                {"text": "sepal", "label": "FLOWER", "start": 16, "end": 21, "confidence": 0.95}
            ]
        }
    ],
    
    "FRUIT": [
        {
            "text": "Fruit ripening was associated with changes in pericarp thickness.",
            "entities": [
                {"text": "Fruit", "label": "FRUIT", "start": 0, "end": 5, "confidence": 0.96},
                {"text": "pericarp", "label": "FRUIT", "start": 47, "end": 55, "confidence": 0.94}
            ]
        }
    ],
    
    "SEED": [
        {
            "text": "Seed coat permeability affected germination rates.",
            "entities": [
                {"text": "Seed coat", "label": "SEED", "start": 0, "end": 9, "confidence": 0.96}
            ]
        }
    ],
    
    # Experimental Conditions (9 types)
    "EXPERIMENTAL_CONDITION": [
        {
            "text": "Plants were grown under controlled temperature and humidity conditions.",
            "entities": [
                {"text": "controlled temperature", "label": "EXPERIMENTAL_CONDITION", "start": 22, "end": 45, "confidence": 0.94},
                {"text": "humidity conditions", "label": "EXPERIMENTAL_CONDITION", "start": 50, "end": 69, "confidence": 0.93}
            ]
        }
    ],
    
    "STRESS_CONDITION": [
        {
            "text": "Drought stress and heat stress were applied for 48 hours.",
            "entities": [
                {"text": "Drought stress", "label": "STRESS_CONDITION", "start": 0, "end": 14, "confidence": 0.97},
                {"text": "heat stress", "label": "STRESS_CONDITION", "start": 19, "end": 30, "confidence": 0.97}
            ]
        }
    ],
    
    "ABIOTIC_STRESS": [
        {
            "text": "Salt treatment at 150 mM NaCl induced osmotic stress.",
            "entities": [
                {"text": "Salt treatment", "label": "ABIOTIC_STRESS", "start": 0, "end": 14, "confidence": 0.96},
                {"text": "osmotic stress", "label": "ABIOTIC_STRESS", "start": 39, "end": 53, "confidence": 0.95}
            ]
        }
    ],
    
    "BIOTIC_STRESS": [
        {
            "text": "Pathogen infection and herbivore damage triggered defense responses.",
            "entities": [
                {"text": "Pathogen infection", "label": "BIOTIC_STRESS", "start": 0, "end": 18, "confidence": 0.97},
                {"text": "herbivore damage", "label": "BIOTIC_STRESS", "start": 23, "end": 39, "confidence": 0.96}
            ]
        }
    ],
    
    "TREATMENT": [
        {
            "text": "Chemical treatment with ABA enhanced stress tolerance.",
            "entities": [
                {"text": "Chemical treatment with ABA", "label": "TREATMENT", "start": 0, "end": 27, "confidence": 0.95}
            ]
        }
    ],
    
    "ENVIRONMENTAL_FACTOR": [
        {
            "text": "Light intensity and photoperiod affected flowering time.",
            "entities": [
                {"text": "Light intensity", "label": "ENVIRONMENTAL_FACTOR", "start": 0, "end": 15, "confidence": 0.96},
                {"text": "photoperiod", "label": "ENVIRONMENTAL_FACTOR", "start": 20, "end": 31, "confidence": 0.95}
            ]
        }
    ],
    
    "GROWTH_CONDITION": [
        {
            "text": "Hydroponic culture with modified Hoagland solution was used.",
            "entities": [
                {"text": "Hydroponic culture", "label": "GROWTH_CONDITION", "start": 0, "end": 18, "confidence": 0.96}
            ]
        }
    ],
    
    "DEVELOPMENTAL_STAGE": [
        {
            "text": "Samples were collected at vegetative and reproductive stages.",
            "entities": [
                {"text": "vegetative", "label": "DEVELOPMENTAL_STAGE", "start": 26, "end": 36, "confidence": 0.95},
                {"text": "reproductive stages", "label": "DEVELOPMENTAL_STAGE", "start": 41, "end": 60, "confidence": 0.95}
            ]
        }
    ],
    
    "TIME_POINT": [
        {
            "text": "Gene expression was measured at 6, 12, and 24 hours after treatment.",
            "entities": [
                {"text": "6, 12, and 24 hours", "label": "TIME_POINT", "start": 32, "end": 51, "confidence": 0.94}
            ]
        }
    ],
    
    # Molecular Traits (9 types)
    "MOLECULAR_TRAIT": [
        {
            "text": "Transcript abundance and protein levels varied between treatments.",
            "entities": [
                {"text": "Transcript abundance", "label": "MOLECULAR_TRAIT", "start": 0, "end": 20, "confidence": 0.95},
                {"text": "protein levels", "label": "MOLECULAR_TRAIT", "start": 25, "end": 39, "confidence": 0.94}
            ]
        }
    ],
    
    "GENE_EXPRESSION": [
        {
            "text": "Upregulation of defense genes was observed after pathogen treatment.",
            "entities": [
                {"text": "Upregulation", "label": "GENE_EXPRESSION", "start": 0, "end": 12, "confidence": 0.95}
            ]
        }
    ],
    
    "ENZYME_ACTIVITY": [
        {
            "text": "Catalase activity increased threefold under oxidative stress.",
            "entities": [
                {"text": "Catalase activity", "label": "ENZYME_ACTIVITY", "start": 0, "end": 17, "confidence": 0.97}
            ]
        }
    ],
    
    "METABOLITE_LEVEL": [
        {
            "text": "Flavonoid concentration was higher in stressed plants.",
            "entities": [
                {"text": "Flavonoid concentration", "label": "METABOLITE_LEVEL", "start": 0, "end": 23, "confidence": 0.96}
            ]
        }
    ],
    
    "PROTEIN_ABUNDANCE": [
        {
            "text": "Heat shock protein expression increased under temperature stress.",
            "entities": [
                {"text": "Heat shock protein expression", "label": "PROTEIN_ABUNDANCE", "start": 0, "end": 30, "confidence": 0.96}
            ]
        }
    ],
    
    "METABOLIC_PATHWAY": [
        {
            "text": "The phenylpropanoid pathway was activated during stress response.",
            "entities": [
                {"text": "phenylpropanoid pathway", "label": "METABOLIC_PATHWAY", "start": 4, "end": 27, "confidence": 0.97}
            ]
        }
    ],
    
    "BIOSYNTHESIS": [
        {
            "text": "Flavonoid biosynthesis genes were coordinately regulated.",
            "entities": [
                {"text": "Flavonoid biosynthesis", "label": "BIOSYNTHESIS", "start": 0, "end": 22, "confidence": 0.97}
            ]
        }
    ],
    
    "REGULATION": [
        {
            "text": "Transcriptional regulation of stress-responsive genes was complex.",
            "entities": [
                {"text": "Transcriptional regulation", "label": "REGULATION", "start": 0, "end": 26, "confidence": 0.96}
            ]
        }
    ],
    
    "SIGNALING": [
        {
            "text": "Calcium signaling mediated the stress response pathway.",
            "entities": [
                {"text": "Calcium signaling", "label": "SIGNALING", "start": 0, "end": 17, "confidence": 0.96}
            ]
        }
    ],
    
    # Plant Traits (9 types)
    "PLANT_TRAIT": [
        {
            "text": "Plant height and leaf area were measured weekly.",
            "entities": [
                {"text": "Plant height", "label": "PLANT_TRAIT", "start": 0, "end": 12, "confidence": 0.96},
                {"text": "leaf area", "label": "PLANT_TRAIT", "start": 17, "end": 26, "confidence": 0.95}
            ]
        }
    ],
    
    "MORPHOLOGICAL_TRAIT": [
        {
            "text": "Leaf shape and flower color varied among cultivars.",
            "entities": [
                {"text": "Leaf shape", "label": "MORPHOLOGICAL_TRAIT", "start": 0, "end": 10, "confidence": 0.96},
                {"text": "flower color", "label": "MORPHOLOGICAL_TRAIT", "start": 15, "end": 27, "confidence": 0.96}
            ]
        }
    ],
    
    "PHYSIOLOGICAL_TRAIT": [
        {
            "text": "Photosynthetic rate and water use efficiency were measured.",
            "entities": [
                {"text": "Photosynthetic rate", "label": "PHYSIOLOGICAL_TRAIT", "start": 0, "end": 19, "confidence": 0.97},
                {"text": "water use efficiency", "label": "PHYSIOLOGICAL_TRAIT", "start": 24, "end": 44, "confidence": 0.96}
            ]
        }
    ],
    
    "BIOCHEMICAL_TRAIT": [
        {
            "text": "Total phenolic content and antioxidant capacity were analyzed.",
            "entities": [
                {"text": "Total phenolic content", "label": "BIOCHEMICAL_TRAIT", "start": 0, "end": 22, "confidence": 0.96},
                {"text": "antioxidant capacity", "label": "BIOCHEMICAL_TRAIT", "start": 27, "end": 47, "confidence": 0.95}
            ]
        }
    ],
    
    "GROWTH_TRAIT": [
        {
            "text": "Biomass accumulation and growth rate differed between genotypes.",
            "entities": [
                {"text": "Biomass accumulation", "label": "GROWTH_TRAIT", "start": 0, "end": 20, "confidence": 0.96},
                {"text": "growth rate", "label": "GROWTH_TRAIT", "start": 25, "end": 36, "confidence": 0.95}
            ]
        }
    ],
    
    "REPRODUCTIVE_TRAIT": [
        {
            "text": "Flowering time and seed production were recorded.",
            "entities": [
                {"text": "Flowering time", "label": "REPRODUCTIVE_TRAIT", "start": 0, "end": 14, "confidence": 0.96},
                {"text": "seed production", "label": "REPRODUCTIVE_TRAIT", "start": 19, "end": 34, "confidence": 0.95}
            ]
        }
    ],
    
    "STRESS_TOLERANCE": [
        {
            "text": "Drought tolerance and salt tolerance were evaluated in field conditions.",
            "entities": [
                {"text": "Drought tolerance", "label": "STRESS_TOLERANCE", "start": 0, "end": 17, "confidence": 0.97},
                {"text": "salt tolerance", "label": "STRESS_TOLERANCE", "start": 22, "end": 36, "confidence": 0.97}
            ]
        }
    ],
    
    "QUALITY_TRAIT": [
        {
            "text": "Nutritional value and taste quality were assessed by sensory panel.",
            "entities": [
                {"text": "Nutritional value", "label": "QUALITY_TRAIT", "start": 0, "end": 17, "confidence": 0.95},
                {"text": "taste quality", "label": "QUALITY_TRAIT", "start": 22, "end": 35, "confidence": 0.94}
            ]
        }
    ],
    
    "YIELD_TRAIT": [
        {
            "text": "Grain yield and harvest index were higher in improved varieties.",
            "entities": [
                {"text": "Grain yield", "label": "YIELD_TRAIT", "start": 0, "end": 11, "confidence": 0.96},
                {"text": "harvest index", "label": "YIELD_TRAIT", "start": 16, "end": 29, "confidence": 0.95}
            ]
        }
    ],
    
    # Additional supporting entity types
    "GENE": [
        {
            "text": "The CHS gene encodes chalcone synthase, a key enzyme in flavonoid biosynthesis.",
            "entities": [
                {"text": "CHS", "label": "GENE", "start": 4, "end": 7, "confidence": 0.97}
            ]
        }
    ],
    
    "PROTEIN": [
        {
            "text": "Rubisco protein levels decreased under drought stress conditions.",
            "entities": [
                {"text": "Rubisco", "label": "PROTEIN", "start": 0, "end": 7, "confidence": 0.98}
            ]
        }
    ],
    
    "ENZYME": [
        {
            "text": "Peroxidase and catalase showed increased activity during oxidative stress.",
            "entities": [
                {"text": "Peroxidase", "label": "ENZYME", "start": 0, "end": 10, "confidence": 0.97},
                {"text": "catalase", "label": "ENZYME", "start": 15, "end": 23, "confidence": 0.97}
            ]
        }
    ],
    
    "TRANSCRIPTION_FACTOR": [
        {
            "text": "The MYB transcription factor regulates anthocyanin biosynthesis genes.",
            "entities": [
                {"text": "MYB", "label": "TRANSCRIPTION_FACTOR", "start": 4, "end": 7, "confidence": 0.96}
            ]
        }
    ],
    
    "ANALYTICAL_METHOD": [
        {
            "text": "LC-MS/MS analysis was performed for metabolite identification.",
            "entities": [
                {"text": "LC-MS/MS", "label": "ANALYTICAL_METHOD", "start": 0, "end": 8, "confidence": 0.98}
            ]
        }
    ],
    
    "CHROMATOGRAPHY": [
        {
            "text": "HPLC separation was followed by mass spectrometric detection.",
            "entities": [
                {"text": "HPLC", "label": "CHROMATOGRAPHY", "start": 0, "end": 4, "confidence": 0.98}
            ]
        }
    ],
    
    "MASS_SPECTROMETRY": [
        {
            "text": "ESI-MS analysis revealed the molecular ion peaks of flavonoids.",
            "entities": [
                {"text": "ESI-MS", "label": "MASS_SPECTROMETRY", "start": 0, "end": 6, "confidence": 0.97}
            ]
        }
    ],
    
    "SPECTROSCOPY": [
        {
            "text": "1H-NMR spectroscopy confirmed the structure of the isolated compound.",
            "entities": [
                {"text": "1H-NMR", "label": "SPECTROSCOPY", "start": 0, "end": 6, "confidence": 0.98}
            ]
        }
    ],
    
    "BIOLOGICAL_ACTIVITY": [
        {
            "text": "The extract showed strong antioxidant activity and antimicrobial properties.",
            "entities": [
                {"text": "antioxidant activity", "label": "BIOLOGICAL_ACTIVITY", "start": 27, "end": 47, "confidence": 0.96},
                {"text": "antimicrobial properties", "label": "BIOLOGICAL_ACTIVITY", "start": 52, "end": 76, "confidence": 0.95}
            ]
        }
    ],
    
    "PHARMACOLOGICAL_ACTIVITY": [
        {
            "text": "Anti-inflammatory effects were observed in treated cells.",
            "entities": [
                {"text": "Anti-inflammatory effects", "label": "PHARMACOLOGICAL_ACTIVITY", "start": 0, "end": 26, "confidence": 0.96}
            ]
        }
    ],
    
    "HUMAN_TRAIT": [
        {
            "text": "Blood pressure and cholesterol levels were monitored in the study.",
            "entities": [
                {"text": "Blood pressure", "label": "HUMAN_TRAIT", "start": 0, "end": 14, "confidence": 0.96},
                {"text": "cholesterol levels", "label": "HUMAN_TRAIT", "start": 19, "end": 37, "confidence": 0.95}
            ]
        }
    ],
    
    "DISEASE": [
        {
            "text": "The compound showed protective effects against diabetes and cardiovascular disease.",
            "entities": [
                {"text": "diabetes", "label": "DISEASE", "start": 48, "end": 56, "confidence": 0.97},
                {"text": "cardiovascular disease", "label": "DISEASE", "start": 61, "end": 83, "confidence": 0.97}
            ]
        }
    ],
    
    "HEALTH_BENEFIT": [
        {
            "text": "Cardioprotective effects and neuroprotection were demonstrated in vivo.",
            "entities": [
                {"text": "Cardioprotective effects", "label": "HEALTH_BENEFIT", "start": 0, "end": 25, "confidence": 0.96},
                {"text": "neuroprotection", "label": "HEALTH_BENEFIT", "start": 30, "end": 45, "confidence": 0.95}
            ]
        }
    ],
    
    "BIOMARKER": [
        {
            "text": "Serum metabolites served as biomarkers for disease progression.",
            "entities": [
                {"text": "Serum metabolites", "label": "BIOMARKER", "start": 0, "end": 17, "confidence": 0.95}
            ]
        }
    ]
}


# Core zero-shot prompt templates
BASIC_ZERO_SHOT_TEMPLATE = """You are an expert in plant metabolomics and scientific literature analysis. Your task is to extract named entities from the provided text.

**ENTITY TYPES TO EXTRACT:**
{schema}

**INPUT TEXT:**
{text}

**INSTRUCTIONS:**
1. Identify all mentions of the specified entity types in the text
2. Extract the exact text spans as they appear in the input
3. Assign the most appropriate entity label from the schema
4. Provide confidence scores between 0.0 and 1.0

**OUTPUT FORMAT:**
Return a JSON object with an "entities" array. Each entity must include:
- "text": exact text span from the input
- "label": entity type from the schema (uppercase)
- "start": character start position
- "end": character end position
- "confidence": confidence score (0.0-1.0)

**EXAMPLE OUTPUT:**
Return JSON like: {"entities": [{"text": "quercetin", "label": "METABOLITE", "start": 15, "end": 24, "confidence": 0.95}]}

Extract all relevant entities now:{examples}"""


DETAILED_ZERO_SHOT_TEMPLATE = """You are a specialized NER system for plant metabolomics research. Extract named entities from scientific text with high accuracy and precision.

**TASK OVERVIEW:**
Extract all named entities that match the provided entity schema from the input text. Focus on scientific terminology, chemical compounds, biological entities, and research-related concepts.

**ENTITY CATEGORIES:**
{schema}

**TEXT TO ANALYZE:**
{text}

**DETAILED EXTRACTION GUIDELINES:**
1. **Entity Identification**: Scan the text systematically for mentions of each entity type
2. **Exact Spans**: Extract the precise text as it appears, maintaining original formatting
3. **Scientific Context**: Consider the scientific domain when disambiguating entities
4. **Hierarchical Types**: When multiple labels could apply, choose the most specific one
5. **Confidence Assessment**: Base confidence on:
   - Clarity of the match (0.9-1.0 for obvious matches)
   - Context appropriateness (0.7-0.9 for likely matches)
   - Ambiguity level (0.5-0.7 for uncertain matches)

**HANDLING SPECIAL CASES:**
- Chemical formulas and systematic names: Extract complete names
- Species names: Include both common and scientific names when present
- Gene/protein names: Maintain original formatting and capitalization
- Overlapping entities: Include all valid interpretations
- Abbreviations: Extract both abbreviation and full form if present

**OUTPUT REQUIREMENTS:**
Must return valid JSON with "entities" array containing objects with required fields:
- "text": exact substring from input text
- "label": uppercase entity type from schema
- "start": zero-indexed character start position
- "end": zero-indexed character end position (exclusive)
- "confidence": float between 0.0 and 1.0

**QUALITY CRITERIA:**
- Completeness: Extract all relevant entities
- Accuracy: Ensure correct entity type assignment
- Precision: Avoid false positives and over-extraction
- Consistency: Apply the same criteria throughout the text

Begin extraction:{examples}"""


PRECISION_FOCUSED_TEMPLATE = """You are a high-precision Named Entity Recognition system for plant metabolomics. Prioritize accuracy over completeness to minimize false positives.

**PRECISION GUIDELINES:**
- Only extract entities you are highly confident about (confidence ≥ 0.8)
- When in doubt, exclude rather than include
- Prefer specific entity types over general ones
- Require clear scientific context for ambiguous terms

**ENTITY TYPES:**
{schema}

**INPUT TEXT:**
{text}

**EXTRACTION CRITERIA:**
1. **High Confidence Only**: Extract only entities with strong contextual support
2. **Scientific Terminology**: Focus on established scientific terms and nomenclature
3. **Context Validation**: Ensure entity fits the scientific domain and context
4. **Avoid Ambiguity**: Skip terms that could have multiple interpretations
5. **Systematic Names**: Prefer systematic chemical names over common names when available

**OUTPUT FORMAT:**
Return JSON with "entities" array. Only include entities meeting high precision criteria.

Return JSON like: {"entities": [{"text": "anthocyanin", "label": "PHENOLIC_COMPOUND", "start": 23, "end": 34, "confidence": 0.98}]}

Extract high-precision entities:{examples}"""


RECALL_FOCUSED_TEMPLATE = """You are a comprehensive Named Entity Recognition system for plant metabolomics. Maximize recall to capture all potentially relevant entities.

**RECALL OPTIMIZATION:**
- Extract all possible entity mentions, even with lower confidence
- Include borderline cases that might be relevant
- Consider multiple interpretations for ambiguous terms
- Capture both formal and informal terminology

**ENTITY TYPES:**
{schema}

**INPUT TEXT:**
{text}

**COMPREHENSIVE EXTRACTION APPROACH:**
1. **Exhaustive Search**: Identify all potential entity mentions
2. **Inclusive Criteria**: Include entities with moderate confidence (≥ 0.5)
3. **Multiple Labels**: Consider if entities could fit multiple categories
4. **Contextual Clues**: Use surrounding text to identify implied entities
5. **Variant Forms**: Include abbreviations, synonyms, and alternative names
6. **Partial Matches**: Consider substring matches for compound terms

**ENTITY DISCOVERY STRATEGY:**
- Scan for chemical compound patterns (-ine, -ose, -ol endings)
- Look for species indicators (italicized text, binomial nomenclature)
- Identify gene/protein markers (capitalization patterns, nomenclature)
- Find experimental indicators (stress, treatment, condition keywords)
- Detect analytical method mentions (abbreviations, instrument names)

**OUTPUT FORMAT:**
JSON with comprehensive "entities" array including all potential matches:

Return JSON like: {"entities": [{"text": "flavonoid", "label": "PHENOLIC_COMPOUND", "start": 10, "end": 19, "confidence": 0.85}, {"text": "flavonoid compound", "label": "COMPOUND", "start": 10, "end": 28, "confidence": 0.75}]}

Perform comprehensive extraction:{examples}"""


SCIENTIFIC_LITERATURE_TEMPLATE = """You are analyzing scientific literature in plant metabolomics. Extract entities following academic writing conventions and scientific nomenclature standards.

**SCIENTIFIC CONTEXT:**
This text is from peer-reviewed research literature. Apply domain expertise in:
- Chemical nomenclature and systematic naming
- Biological taxonomy and classification
- Experimental methodology and instrumentation
- Statistical and analytical terminology

**ENTITY SCHEMA:**
{schema}

**RESEARCH TEXT:**
{text}

**LITERATURE-SPECIFIC GUIDELINES:**
1. **Nomenclature Standards**: Follow IUPAC, IUBMB, and taxonomic naming conventions
2. **Abbreviation Handling**: Link abbreviations to full forms when defined
3. **Statistical Terms**: Recognize experimental design and analysis terminology
4. **Methodological Terms**: Identify analytical techniques and procedures
5. **Citation Context**: Consider entities mentioned in comparative contexts
6. **Figure/Table References**: Include entities referenced in captions or legends

**ACADEMIC WRITING PATTERNS:**
- Species names: Often italicized or in binomial form
- Gene names: Following organism-specific conventions
- Chemical names: Systematic IUPAC names or common research names
- Methods: Standard analytical procedure names
- Statistics: Recognize p-values, significance tests, effect sizes

**CONFIDENCE CALIBRATION FOR LITERATURE:**
- 0.95-1.0: Standard scientific terminology with clear context
- 0.85-0.95: Domain-specific terms with appropriate usage
- 0.75-0.85: Technical terms requiring domain knowledge
- 0.65-0.75: Contextually appropriate but potentially ambiguous
- 0.50-0.65: Uncertain but scientifically plausible

**JSON OUTPUT:**
Return JSON like: {"entities": [{"text": "Arabidopsis thaliana", "label": "PLANT_SPECIES", "start": 45, "end": 65, "confidence": 0.99}]}

Extract scientific entities:{examples}"""


# Domain-specific templates
METABOLOMICS_TEMPLATE = """Extract metabolomics-specific entities from plant research text. Focus on metabolites, analytical methods, and biochemical processes.

**METABOLOMICS FOCUS AREAS:**
- Primary and secondary metabolites
- Analytical instrumentation and methods
- Metabolic pathways and processes
- Chemical compound classifications
- Bioactivity and function

**ENTITY TYPES:**
{schema}

**TEXT:**
{text}

**METABOLOMICS-SPECIFIC PATTERNS:**
1. **Metabolite Names**: Look for chemical compound names, especially those ending in -ine, -ose, -ol, -acid
2. **Analytical Methods**: LC-MS, GC-MS, NMR, HPLC abbreviations and full forms
3. **Chemical Classes**: Flavonoids, alkaloids, terpenoids, phenolic compounds
4. **Pathway Terms**: Biosynthesis, metabolism, accumulation, regulation
5. **Quantitative Terms**: Concentration, levels, content, abundance

Extract metabolomics entities:{examples}"""


GENETICS_TEMPLATE = """Extract genetics and molecular biology entities from plant research. Focus on genes, proteins, and molecular processes.

**GENETICS FOCUS:**
{schema}

**TEXT:**
{text}

**GENETIC ENTITY PATTERNS:**
1. **Gene Names**: Often italicized, specific nomenclature conventions
2. **Protein Names**: Enzyme names, transcription factors, structural proteins
3. **Molecular Processes**: Expression, regulation, transcription, translation
4. **Genetic Elements**: Promoters, enhancers, binding sites, motifs

Extract genetic entities:{examples}"""


PLANT_BIOLOGY_TEMPLATE = """Extract plant biology entities focusing on anatomy, physiology, and development.

**PLANT BIOLOGY ENTITIES:**
{schema}

**TEXT:**
{text}

**PLANT-SPECIFIC PATTERNS:**
1. **Anatomical Structures**: Organs, tissues, cell types
2. **Developmental Stages**: Growth phases, life cycle stages
3. **Physiological Processes**: Photosynthesis, respiration, transport
4. **Morphological Features**: Size, shape, color, structure descriptors

Extract plant biology entities:{examples}"""


# Few-shot prompt templates with examples
FEW_SHOT_BASIC_TEMPLATE = """You are an expert in plant metabolomics and scientific literature analysis. Your task is to extract named entities from the provided text using the examples below as guidance.

**ENTITY TYPES TO EXTRACT:**
{schema}

**EXAMPLES:**
{examples}

**INPUT TEXT:**
{text}

**INSTRUCTIONS:**
1. Study the examples above to understand the entity extraction patterns
2. Identify all mentions of the specified entity types in the input text
3. Extract the exact text spans as they appear in the input
4. Assign the most appropriate entity label from the schema
5. Follow the same format and confidence scoring as shown in examples

**OUTPUT FORMAT:**
Return a JSON object with an "entities" array. Each entity must include:
- "text": exact text span from the input
- "label": entity type from the schema (uppercase)
- "start": character start position
- "end": character end position
- "confidence": confidence score (0.0-1.0)

Extract all relevant entities following the example patterns:"""


FEW_SHOT_DETAILED_TEMPLATE = """You are a specialized NER system for plant metabolomics research. Use the provided examples to guide accurate entity extraction from scientific text.

**TASK OVERVIEW:**
Extract all named entities that match the provided entity schema from the input text. Use the examples below to understand extraction patterns, entity boundaries, and confidence scoring.

**ENTITY CATEGORIES:**
{schema}

**LEARNING EXAMPLES:**
{examples}

**TEXT TO ANALYZE:**
{text}

**DETAILED EXTRACTION GUIDELINES:**
1. **Pattern Recognition**: Study the examples to identify common patterns for each entity type
2. **Boundary Detection**: Follow the example patterns for determining entity start/end positions
3. **Context Analysis**: Use scientific context clues as demonstrated in the examples
4. **Confidence Calibration**: Match confidence levels to similar examples
5. **Multi-entity Handling**: Extract overlapping entities when appropriate, as shown in examples

**ADVANCED STRATEGIES:**
- Chemical nomenclature: Follow IUPAC naming patterns from examples
- Species identification: Use binomial nomenclature patterns
- Analytical methods: Recognize instrument abbreviations and techniques
- Experimental conditions: Identify treatment and environmental factors
- Morphological terms: Extract anatomical and structural descriptors

**QUALITY ASSURANCE:**
- Verify each extraction against similar examples
- Ensure consistent formatting and field structure
- Validate confidence scores against example patterns
- Cross-check entity types for accuracy

**JSON OUTPUT:**
Return JSON with "entities" array containing objects with required fields:
- "text": exact substring from input text
- "label": uppercase entity type from schema
- "start": zero-indexed character start position
- "end": zero-indexed character end position (exclusive)
- "confidence": float between 0.0 and 1.0

Begin comprehensive extraction following example patterns:"""


FEW_SHOT_PRECISION_TEMPLATE = """You are a high-precision Named Entity Recognition system for plant metabolomics. Use the provided high-confidence examples to guide precise entity extraction while minimizing false positives.

**PRECISION STRATEGY:**
- Focus on examples with confidence scores ≥ 0.90 for pattern matching
- Only extract entities with strong contextual support similar to examples
- When in doubt, follow the conservative approach shown in precision examples
- Prefer specific entity types over general ones as demonstrated

**ENTITY TYPES:**
{schema}

**HIGH-PRECISION EXAMPLES:**
{examples}

**INPUT TEXT:**
{text}

**PRECISION-FOCUSED EXTRACTION:**
1. **Strict Pattern Matching**: Only extract entities that closely match example patterns
2. **Context Validation**: Require strong scientific context as shown in examples
3. **Confidence Thresholding**: Assign confidence scores ≥ 0.80 only for clear matches
4. **Entity Verification**: Cross-reference against example entity boundaries
5. **Conservative Approach**: When uncertain, exclude rather than include

**PRECISION CRITERIA:**
- Established scientific terminology matching examples
- Clear entity boundaries following example patterns
- Unambiguous context supporting entity type assignment
- High similarity to provided examples

Return JSON with high-confidence entities only:"""


FEW_SHOT_RECALL_TEMPLATE = """You are a comprehensive Named Entity Recognition system for plant metabolomics. Use the provided diverse examples to maximize entity recall and capture all potentially relevant mentions.

**RECALL OPTIMIZATION:**
- Study all example patterns including lower-confidence extractions
- Extract entities with moderate confidence (≥ 0.50) following example guidance
- Consider multiple interpretations as shown in diverse examples
- Include borderline cases that match any example patterns

**ENTITY TYPES:**
{schema}

**COMPREHENSIVE EXAMPLES:**
{examples}

**INPUT TEXT:**
{text}

**RECALL-MAXIMIZING APPROACH:**
1. **Exhaustive Pattern Matching**: Use all example patterns for entity identification
2. **Inclusive Extraction**: Include entities matching any example confidence level
3. **Variant Recognition**: Extract synonyms and alternative forms shown in examples
4. **Context Flexibility**: Use broad contextual clues from examples
5. **Multi-interpretation**: Consider overlapping entity types as demonstrated

**COVERAGE STRATEGY:**
- Apply all entity type patterns from examples
- Extract both formal and informal terminology
- Include partial matches following example precedents
- Capture compound and nested entities as shown
- Consider abbreviations and full forms from examples

Perform comprehensive extraction using all example patterns:"""


FEW_SHOT_SCIENTIFIC_TEMPLATE = """You are analyzing scientific literature in plant metabolomics. Use the provided academic examples to guide precise entity extraction following scientific writing conventions.

**SCIENTIFIC CONTEXT:**
This text is from peer-reviewed research literature. The examples demonstrate proper scientific nomenclature and terminology extraction patterns specific to academic writing.

**ENTITY SCHEMA:**
{schema}

**SCIENTIFIC EXAMPLES:**
{examples}

**RESEARCH TEXT:**
{text}

**LITERATURE-SPECIFIC GUIDELINES:**
1. **Nomenclature Standards**: Follow IUPAC, IUBMB, and taxonomic naming patterns from examples
2. **Citation Conventions**: Extract entities as they appear in scientific references
3. **Methodology Terms**: Identify analytical techniques following example patterns
4. **Statistical Context**: Recognize experimental terminology as demonstrated
5. **Formal Language**: Apply scientific writing conventions from examples

**ACADEMIC EXTRACTION PATTERNS:**
- Species names: Italicized binomial nomenclature following examples
- Chemical names: Systematic IUPAC names and common research terms
- Gene nomenclature: Organism-specific naming conventions
- Analytical methods: Standard procedure names and abbreviations
- Statistical terms: Experimental design and analysis terminology

**CONFIDENCE CALIBRATION FOR LITERATURE:**
Base confidence scores on example patterns:
- 0.95-1.0: Standard terminology matching examples exactly
- 0.85-0.95: Domain-specific terms following example patterns
- 0.75-0.85: Technical terms requiring domain knowledge
- 0.65-0.75: Contextually appropriate following examples

Extract scientific entities following academic example patterns:"""


# Domain-specific few-shot templates
FEW_SHOT_METABOLOMICS_TEMPLATE = """Extract metabolomics-specific entities from plant research text using the provided domain examples as guidance.

**METABOLOMICS FOCUS:**
Use examples to identify metabolites, analytical methods, and biochemical processes in plant metabolomics research.

**ENTITY TYPES:**
{schema}

**METABOLOMICS EXAMPLES:**
{examples}

**TEXT:**
{text}

**DOMAIN-SPECIFIC PATTERNS FROM EXAMPLES:**
1. **Metabolite Recognition**: Chemical compound names, especially -ine, -ose, -ol, -acid endings
2. **Analytical Methods**: LC-MS, GC-MS, NMR abbreviations and full forms
3. **Chemical Classes**: Flavonoids, alkaloids, terpenoids, phenolic compounds
4. **Pathway Terms**: Biosynthesis, metabolism, accumulation, regulation
5. **Quantitative Context**: Concentration, levels, content, abundance

Follow metabolomics example patterns for extraction:"""


FEW_SHOT_GENETICS_TEMPLATE = """Extract genetics and molecular biology entities using the provided genetic research examples.

**GENETICS FOCUS:**
{schema}

**GENETIC EXAMPLES:**
{examples}

**TEXT:**
{text}

**GENETIC PATTERNS FROM EXAMPLES:**
1. **Gene Nomenclature**: Organism-specific naming conventions from examples
2. **Protein Names**: Enzyme names, transcription factors, structural proteins
3. **Molecular Processes**: Expression, regulation, transcription patterns
4. **Genetic Elements**: Promoters, enhancers, binding sites from examples

Extract genetic entities following example patterns:"""


FEW_SHOT_PLANT_BIOLOGY_TEMPLATE = """Extract plant biology entities focusing on anatomy, physiology, and development using botanical examples.

**PLANT BIOLOGY ENTITIES:**
{schema}

**BOTANICAL EXAMPLES:**
{examples}

**TEXT:**
{text}

**PLANT-SPECIFIC PATTERNS FROM EXAMPLES:**
1. **Anatomical Terms**: Organs, tissues, cell types from examples
2. **Developmental Stages**: Growth phases, life cycle stages
3. **Physiological Processes**: Photosynthesis, respiration, transport
4. **Morphological Features**: Size, shape, color, structure descriptors

Extract plant biology entities following botanical example patterns:"""


# Template registry
TEMPLATE_REGISTRY = {
    TemplateType.BASIC.value: BASIC_ZERO_SHOT_TEMPLATE,
    TemplateType.DETAILED.value: DETAILED_ZERO_SHOT_TEMPLATE,
    TemplateType.PRECISION.value: PRECISION_FOCUSED_TEMPLATE,
    TemplateType.RECALL.value: RECALL_FOCUSED_TEMPLATE,
    TemplateType.SCIENTIFIC.value: SCIENTIFIC_LITERATURE_TEMPLATE,
    TemplateType.METABOLOMICS.value: METABOLOMICS_TEMPLATE,
    TemplateType.GENETICS.value: GENETICS_TEMPLATE,
    TemplateType.PLANT_BIOLOGY.value: PLANT_BIOLOGY_TEMPLATE,
    TemplateType.FEW_SHOT_BASIC.value: FEW_SHOT_BASIC_TEMPLATE,
    TemplateType.FEW_SHOT_DETAILED.value: FEW_SHOT_DETAILED_TEMPLATE,
    TemplateType.FEW_SHOT_PRECISION.value: FEW_SHOT_PRECISION_TEMPLATE,
    TemplateType.FEW_SHOT_RECALL.value: FEW_SHOT_RECALL_TEMPLATE,
    TemplateType.FEW_SHOT_SCIENTIFIC.value: FEW_SHOT_SCIENTIFIC_TEMPLATE,
    TemplateType.FEW_SHOT_METABOLOMICS.value: FEW_SHOT_METABOLOMICS_TEMPLATE,
    TemplateType.FEW_SHOT_GENETICS.value: FEW_SHOT_GENETICS_TEMPLATE,
    TemplateType.FEW_SHOT_PLANT_BIOLOGY.value: FEW_SHOT_PLANT_BIOLOGY_TEMPLATE,
}


# Example generation and selection functions

def generate_synthetic_examples(
    entity_types: List[str], 
    num_examples: int = 3,
    difficulty_level: str = "mixed",
    domain_focus: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate synthetic examples for given entity types.
    
    Args:
        entity_types: List of entity type labels to generate examples for
        num_examples: Number of examples to generate per entity type
        difficulty_level: "simple", "complex", or "mixed"
        domain_focus: Optional domain to focus examples on
        
    Returns:
        List of synthetic examples in the format expected by extract_entities
    """
    examples = []
    available_types = set(SYNTHETIC_EXAMPLES_DATABASE.keys())
    
    for entity_type in entity_types:
        if entity_type not in available_types:
            continue
            
        type_examples = SYNTHETIC_EXAMPLES_DATABASE[entity_type]
        
        # Filter by difficulty if specified
        if difficulty_level == "simple":
            # Prefer examples with single entities and high confidence
            filtered = [ex for ex in type_examples if len(ex["entities"]) <= 2 and 
                       all(ent["confidence"] >= 0.90 for ent in ex["entities"])]
        elif difficulty_level == "complex":
            # Prefer examples with multiple entities or lower confidence
            filtered = [ex for ex in type_examples if len(ex["entities"]) > 2 or 
                       any(ent["confidence"] < 0.90 for ent in ex["entities"])]
        else:
            filtered = type_examples
            
        if not filtered:
            filtered = type_examples
            
        # Select examples randomly
        selected = random.sample(filtered, min(num_examples, len(filtered)))
        examples.extend(selected)
    
    return examples


def select_examples(
    target_entity_types: List[str],
    strategy: str = "balanced",
    max_examples: int = 10,
    confidence_filter: Optional[Tuple[float, float]] = None,
    domain_context: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Select optimal examples for given entity types using various strategies.
    
    Args:
        target_entity_types: Entity types to find examples for
        strategy: Selection strategy ("balanced", "random", "high_confidence", "diverse")
        max_examples: Maximum number of examples to return
        confidence_filter: Optional (min, max) confidence range filter
        domain_context: Optional domain context for selection
        
    Returns:
        List of selected examples
    """
    available_examples = []
    
    # Collect relevant examples
    for entity_type in target_entity_types:
        if entity_type in SYNTHETIC_EXAMPLES_DATABASE:
            type_examples = SYNTHETIC_EXAMPLES_DATABASE[entity_type]
            
            # Apply confidence filter if specified
            if confidence_filter:
                min_conf, max_conf = confidence_filter
                filtered_examples = []
                for example in type_examples:
                    valid_entities = [ent for ent in example["entities"] 
                                    if min_conf <= ent["confidence"] <= max_conf]
                    if valid_entities:
                        filtered_example = example.copy()
                        filtered_example["entities"] = valid_entities
                        filtered_examples.append(filtered_example)
                type_examples = filtered_examples
            
            available_examples.extend(type_examples)
    
    if not available_examples:
        return []
    
    # Apply selection strategy
    if strategy == "random":
        selected = random.sample(available_examples, min(max_examples, len(available_examples)))
    
    elif strategy == "high_confidence":
        # Sort by average confidence and take top examples
        def avg_confidence(example):
            return sum(ent["confidence"] for ent in example["entities"]) / len(example["entities"])
        
        sorted_examples = sorted(available_examples, key=avg_confidence, reverse=True)
        selected = sorted_examples[:max_examples]
    
    elif strategy == "diverse":
        # Select examples with diverse entity types
        selected = []
        seen_types = set()
        
        for example in available_examples:
            example_types = {ent["label"] for ent in example["entities"]}
            if not seen_types.intersection(example_types) or len(selected) < max_examples // 2:
                selected.append(example)
                seen_types.update(example_types)
                if len(selected) >= max_examples:
                    break
    
    elif strategy == "balanced":
        # Balance examples across entity types
        examples_by_type = {}
        for example in available_examples:
            for entity in example["entities"]:
                entity_type = entity["label"]
                if entity_type not in examples_by_type:
                    examples_by_type[entity_type] = []
                examples_by_type[entity_type].append(example)
        
        selected = []
        examples_per_type = max(1, max_examples // len(examples_by_type))
        
        for entity_type, type_examples in examples_by_type.items():
            type_selected = random.sample(type_examples, min(examples_per_type, len(type_examples)))
            selected.extend(type_selected[:examples_per_type])
            if len(selected) >= max_examples:
                break
        
        selected = selected[:max_examples]
    
    else:
        selected = available_examples[:max_examples]
    
    return selected


def get_examples_by_domain(domain: str, max_examples: int = 8) -> List[Dict[str, Any]]:
    """
    Get examples filtered by domain categories.
    
    Args:
        domain: Domain name (metabolomics, genetics, plant_biology, etc.)
        max_examples: Maximum number of examples to return
        
    Returns:
        List of domain-specific examples
    """
    try:
        entity_categories = get_entity_types_by_category()
        domain_mapping = {
            "metabolomics": entity_categories["metabolites"],
            "genetics": entity_categories["genetics"],
            "plant_biology": entity_categories["plant_anatomy"] + entity_categories["plant_traits"],
            "biochemistry": entity_categories["metabolites"] + entity_categories["genetics"],
            "stress": entity_categories["experimental_conditions"],
            "analytical": entity_categories["analytical"],
            "pharmacology": entity_categories["bioactivity"]
        }
        
        if domain.lower() in domain_mapping:
            relevant_types = domain_mapping[domain.lower()]
            return select_examples(relevant_types, strategy="balanced", max_examples=max_examples)
        else:
            # Default to mixed examples
            all_types = list(SYNTHETIC_EXAMPLES_DATABASE.keys())[:10]  # First 10 types
            return select_examples(all_types, strategy="balanced", max_examples=max_examples)
            
    except Exception:
        # Fallback to random selection
        all_examples = []
        for examples in SYNTHETIC_EXAMPLES_DATABASE.values():
            all_examples.extend(examples)
        return random.sample(all_examples, min(max_examples, len(all_examples)))


def format_examples_for_prompt(examples: List[Dict[str, Any]]) -> str:
    """
    Format examples for inclusion in prompts.
    
    Args:
        examples: List of example dictionaries
        
    Returns:
        Formatted string for prompt inclusion
    """
    if not examples:
        return ""
    
    formatted_examples = []
    for i, example in enumerate(examples, 1):
        example_text = example["text"]
        entities_json = json.dumps({"entities": example["entities"]}, indent=2)
        formatted_examples.append(f"Example {i}:\nText: {example_text}\nExpected Output:\n{entities_json}")
    
    return "\n\n".join(formatted_examples)


def get_context_aware_examples(
    input_text: str,
    entity_schema: Dict[str, str],
    max_examples: int = 6
) -> List[Dict[str, Any]]:
    """
    Select examples based on input text context and similarity.
    
    Args:
        input_text: Input text to analyze for context
        entity_schema: Entity schema being used
        max_examples: Maximum number of examples to return
        
    Returns:
        List of contextually relevant examples
    """
    # Simple keyword-based context matching
    input_lower = input_text.lower()
    
    # Identify potential domains based on keywords
    domain_keywords = {
        "metabolomics": ["metabolite", "compound", "concentration", "hplc", "ms", "nmr", "flavonoid", "phenolic"],
        "genetics": ["gene", "expression", "protein", "enzyme", "transcription", "regulation", "dna", "rna"],
        "plant_biology": ["leaf", "root", "stem", "flower", "plant", "tissue", "cell", "organ"],
        "stress": ["stress", "drought", "salt", "heat", "cold", "treatment", "condition"],
        "analytical": ["analysis", "chromatography", "spectroscopy", "detection", "identification"]
    }
    
    # Score domains based on keyword matches
    domain_scores = {}
    for domain, keywords in domain_keywords.items():
        score = sum(1 for keyword in keywords if keyword in input_lower)
        if score > 0:
            domain_scores[domain] = score
    
    # Select examples from top-scoring domains
    if domain_scores:
        top_domain = max(domain_scores, key=domain_scores.get)
        return get_examples_by_domain(top_domain, max_examples)
    else:
        # Fallback to schema-based selection
        schema_types = list(entity_schema.keys())
        return select_examples(schema_types, strategy="balanced", max_examples=max_examples)


# Few-shot template getter functions

def get_few_shot_template(
    template_type: str = "basic",
    examples: Optional[List[Dict[str, Any]]] = None,
    entity_types: Optional[List[str]] = None,
    auto_generate_examples: bool = True
) -> str:
    """
    Get few-shot template with examples.
    
    Args:
        template_type: Type of few-shot template
        examples: Pre-selected examples (optional)
        entity_types: Entity types to generate examples for (if auto_generate_examples=True)
        auto_generate_examples: Whether to auto-generate examples if not provided
        
    Returns:
        Few-shot template string
    """
    template_mapping = {
        "basic": FEW_SHOT_BASIC_TEMPLATE,
        "detailed": FEW_SHOT_DETAILED_TEMPLATE,
        "precision": FEW_SHOT_PRECISION_TEMPLATE,
        "recall": FEW_SHOT_RECALL_TEMPLATE,
        "scientific": FEW_SHOT_SCIENTIFIC_TEMPLATE,
        "metabolomics": FEW_SHOT_METABOLOMICS_TEMPLATE,
        "genetics": FEW_SHOT_GENETICS_TEMPLATE,
        "plant_biology": FEW_SHOT_PLANT_BIOLOGY_TEMPLATE
    }
    
    if template_type not in template_mapping:
        template_type = "basic"
    
    template = template_mapping[template_type]
    
    # Auto-generate examples if needed
    if examples is None and auto_generate_examples and entity_types:
        examples = generate_synthetic_examples(entity_types, num_examples=2)
    
    return template


def get_few_shot_basic_template() -> str:
    """Get basic few-shot template."""
    return FEW_SHOT_BASIC_TEMPLATE


def get_few_shot_detailed_template() -> str:
    """Get detailed few-shot template."""
    return FEW_SHOT_DETAILED_TEMPLATE


def get_few_shot_precision_template() -> str:
    """Get precision-focused few-shot template."""
    return FEW_SHOT_PRECISION_TEMPLATE


def get_few_shot_recall_template() -> str:
    """Get recall-focused few-shot template."""
    return FEW_SHOT_RECALL_TEMPLATE


def get_few_shot_scientific_template() -> str:
    """Get scientific literature few-shot template."""
    return FEW_SHOT_SCIENTIFIC_TEMPLATE


def get_few_shot_domain_template(domain: str) -> str:
    """
    Get domain-specific few-shot template.
    
    Args:
        domain: Domain name (metabolomics, genetics, plant_biology)
        
    Returns:
        Domain-specific few-shot template
        
    Raises:
        TemplateNotFoundError: If domain template not found
    """
    domain = domain.lower().strip()
    
    domain_templates = {
        "metabolomics": FEW_SHOT_METABOLOMICS_TEMPLATE,
        "plant_metabolomics": FEW_SHOT_METABOLOMICS_TEMPLATE,
        "genetics": FEW_SHOT_GENETICS_TEMPLATE,
        "genomics": FEW_SHOT_GENETICS_TEMPLATE,
        "molecular_biology": FEW_SHOT_GENETICS_TEMPLATE,
        "plant_biology": FEW_SHOT_PLANT_BIOLOGY_TEMPLATE,
        "botany": FEW_SHOT_PLANT_BIOLOGY_TEMPLATE,
        "plant_science": FEW_SHOT_PLANT_BIOLOGY_TEMPLATE,
    }
    
    if domain not in domain_templates:
        available_domains = ", ".join(domain_templates.keys())
        raise TemplateNotFoundError(
            f"Few-shot domain '{domain}' not supported. Available domains: {available_domains}"
        )
    
    return domain_templates[domain]


def get_basic_zero_shot_template() -> str:
    """
    Get the basic zero-shot template for general NER tasks.

    Returns:
        Basic template string with placeholders for text, schema, and examples
    """
    return BASIC_ZERO_SHOT_TEMPLATE


def get_detailed_zero_shot_template() -> str:
    """
    Get the detailed zero-shot template with explicit instructions.

    Returns:
        Detailed template string with comprehensive guidelines
    """
    return DETAILED_ZERO_SHOT_TEMPLATE


def get_precision_focused_template() -> str:
    """
    Get template optimized for high precision (minimize false positives).

    Returns:
        Precision-focused template string
    """
    return PRECISION_FOCUSED_TEMPLATE


def get_recall_focused_template() -> str:
    """
    Get template optimized for high recall (capture more entities).

    Returns:
        Recall-focused template string
    """
    return RECALL_FOCUSED_TEMPLATE


def get_scientific_literature_template() -> str:
    """
    Get template optimized for scientific literature analysis.

    Returns:
        Scientific literature template string
    """
    return SCIENTIFIC_LITERATURE_TEMPLATE


def get_domain_specific_template(domain: str) -> str:
    """
    Get domain-specific template for specialized research areas.

    Args:
        domain: Research domain (metabolomics, genetics, plant_biology, etc.)

    Returns:
        Domain-specific template string

    Raises:
        TemplateNotFoundError: If domain template is not available
    """
    domain = domain.lower().strip()

    domain_mapping = {
        "metabolomics": TemplateType.METABOLOMICS.value,
        "plant_metabolomics": TemplateType.METABOLOMICS.value,
        "genetics": TemplateType.GENETICS.value,
        "genomics": TemplateType.GENETICS.value,
        "molecular_biology": TemplateType.GENETICS.value,
        "plant_biology": TemplateType.PLANT_BIOLOGY.value,
        "botany": TemplateType.PLANT_BIOLOGY.value,
        "plant_science": TemplateType.PLANT_BIOLOGY.value,
    }

    if domain not in domain_mapping:
        available_domains = ", ".join(domain_mapping.keys())
        raise TemplateNotFoundError(
            f"Domain '{domain}' not supported. Available domains: {available_domains}"
        )

    template_key = domain_mapping[domain]
    return TEMPLATE_REGISTRY[template_key]


def get_template_by_name(template_name: str) -> str:
    """
    Get template by name with validation.

    Args:
        template_name: Name of the template to retrieve

    Returns:
        Template string

    Raises:
        TemplateNotFoundError: If template name is not found
    """
    template_name = template_name.lower().strip()

    if template_name not in TEMPLATE_REGISTRY:
        available_templates = ", ".join(TEMPLATE_REGISTRY.keys())
        raise TemplateNotFoundError(
            f"Template '{template_name}' not found. Available templates: {available_templates}"
        )

    return TEMPLATE_REGISTRY[template_name]


def list_available_templates() -> List[str]:
    """
    List all available template names.

    Returns:
        List of template names
    """
    return list(TEMPLATE_REGISTRY.keys())


def validate_template(template: str) -> bool:
    """
    Validate template format and required placeholders.

    Args:
        template: Template string to validate

    Returns:
        True if template is valid

    Raises:
        InvalidTemplateError: If template format is invalid
    """
    if not isinstance(template, str):
        raise InvalidTemplateError("Template must be a string")

    if not template.strip():
        raise InvalidTemplateError("Template cannot be empty")

    # Check for required placeholders
    required_placeholders = {"{text}", "{schema}"}
    optional_placeholders = {"{examples}"}
    all_placeholders = required_placeholders | optional_placeholders

    # Find all placeholders in template, but ignore JSON-like structures
    # Remove JSON examples from template temporarily for placeholder validation
    temp_template = re.sub(r'\{[^}]*"[^"]*"[^}]*\}', '', template)
    found_placeholders = set(re.findall(r'\{[^}]+\}', temp_template))

    # Check for required placeholders
    missing_required = required_placeholders - found_placeholders
    if missing_required:
        raise InvalidTemplateError(
            f"Template missing required placeholders: {', '.join(missing_required)}"
        )

    # Check for unknown placeholders
    unknown_placeholders = found_placeholders - all_placeholders
    if unknown_placeholders:
        raise InvalidTemplateError(
            f"Template contains unknown placeholders: {', '.join(unknown_placeholders)}"
        )

    # Validate JSON output format mentions
    if "json" not in template.lower():
        raise InvalidTemplateError("Template should specify JSON output format")

    # Validate entity structure mentions
    required_fields = ["text", "label", "start", "end", "confidence"]
    for field in required_fields:
        if field not in template.lower():
            raise InvalidTemplateError(f"Template should mention required field: {field}")

    return True


def get_template_for_use_case(
    use_case: str,
    domain: Optional[str] = None,
    precision_recall_balance: str = "balanced"
) -> str:
    """
    Get the most appropriate template for a specific use case.

    Args:
        use_case: Use case description (e.g., "research_paper", "quick_analysis")
        domain: Optional domain specification
        precision_recall_balance: "precision", "recall", or "balanced"

    Returns:
        Most appropriate template string

    Raises:
        TemplateNotFoundError: If no suitable template is found
    """
    use_case = use_case.lower().strip()
    precision_recall_balance = precision_recall_balance.lower().strip()

    # Use case mapping
    if use_case in ["research_paper", "scientific_literature", "publication"]:
        return get_scientific_literature_template()
    elif use_case in ["quick_analysis", "basic_extraction", "simple"]:
        return get_basic_zero_shot_template()
    elif use_case in ["comprehensive", "detailed_analysis", "thorough"]:
        return get_detailed_zero_shot_template()
    elif domain:
        return get_domain_specific_template(domain)
    else:
        # Choose based on precision/recall preference
        if precision_recall_balance == "precision":
            return get_precision_focused_template()
        elif precision_recall_balance == "recall":
            return get_recall_focused_template()
        else:
            return get_detailed_zero_shot_template()


def customize_template(
    base_template: str,
    custom_instructions: Optional[str] = None,
    additional_examples: Optional[List[str]] = None,
    confidence_threshold: Optional[float] = None
) -> str:
    """
    Customize a base template with additional instructions and parameters.

    Args:
        base_template: Base template to customize
        custom_instructions: Additional instructions to append
        additional_examples: Extra examples to include
        confidence_threshold: Minimum confidence threshold to specify

    Returns:
        Customized template string

    Raises:
        InvalidTemplateError: If base template is invalid
    """
    # Validate base template
    validate_template(base_template)

    customized = base_template

    # Add custom instructions
    if custom_instructions:
        instruction_section = f"\n\n**CUSTOM INSTRUCTIONS:**\n{custom_instructions.strip()}"
        # Insert before the final extraction command
        if "extract" in customized.lower():
            # Find the last occurrence of extract/begin/perform
            extraction_commands = ["extract", "begin", "perform"]
            last_command_pos = -1
            for command in extraction_commands:
                pos = customized.lower().rfind(command)
                if pos > last_command_pos:
                    last_command_pos = pos

            if last_command_pos != -1:
                customized = (
                    customized[:last_command_pos] +
                    instruction_section +
                    "\n\n" +
                    customized[last_command_pos:]
                )

    # Add confidence threshold
    if confidence_threshold is not None:
        if not (0.0 <= confidence_threshold <= 1.0):
            raise InvalidTemplateError("Confidence threshold must be between 0.0 and 1.0")

        threshold_instruction = f"\n\n**CONFIDENCE THRESHOLD:**\nOnly extract entities with confidence >= {confidence_threshold:.2f}"
        customized += threshold_instruction

    # Add additional examples (placeholder for now, would need integration with examples parameter)
    if additional_examples:
        example_instruction = f"\n\n**ADDITIONAL CONTEXT:**\n" + "\n".join(additional_examples)
        customized += example_instruction

    return customized


def get_template_statistics(template: str) -> Dict[str, Any]:
    """
    Get statistics and information about a template.

    Args:
        template: Template string to analyze

    Returns:
        Dictionary with template statistics
    """
    validate_template(template)

    # Count words and characters
    word_count = len(template.split())
    char_count = len(template)

    # Find placeholders
    placeholders = set(re.findall(r'\{[^}]+\}', template))

    # Count sections (marked by **SECTION:** patterns)
    sections = re.findall(r'\*\*([^*]+)\*\*', template)

    # Analyze instruction density
    instruction_keywords = [
        "extract", "identify", "recognize", "find", "locate", "analyze",
        "must", "should", "require", "ensure", "include", "focus"
    ]
    instruction_count = sum(
        template.lower().count(keyword) for keyword in instruction_keywords
    )

    return {
        "word_count": word_count,
        "character_count": char_count,
        "placeholders": list(placeholders),
        "placeholder_count": len(placeholders),
        "sections": sections,
        "section_count": len(sections),
        "instruction_density": instruction_count / word_count if word_count > 0 else 0,
        "estimated_complexity": "high" if word_count > 300 else "medium" if word_count > 150 else "low"
    }


# Template validation patterns
ENTITY_FIELD_PATTERNS = {
    "text": r'"text":\s*"[^"]*"',
    "label": r'"label":\s*"[A-Z_]+"',
    "start": r'"start":\s*\d+',
    "end": r'"end":\s*\d+',
    "confidence": r'"confidence":\s*0?\.\d+'
}


# Advanced Template Validation and Utility Functions

def validate_template_structure(template: str) -> bool:
    """
    Validate the overall structure of a prompt template.
    
    This function performs comprehensive validation of template structure,
    checking for required placeholders, proper formatting, instruction clarity,
    and output format specifications.
    
    Args:
        template: Template string to validate
        
    Returns:
        True if template structure is valid
        
    Raises:
        InvalidTemplateError: If template structure is invalid
        
    Example:
        >>> template = "Extract entities from {text} using {schema}. Return JSON with entities array."
        >>> validate_template_structure(template)
        True
    """
    if not isinstance(template, str):
        raise InvalidTemplateError("Template must be a string")
        
    if not template.strip():
        raise InvalidTemplateError("Template cannot be empty")
        
    # Length validation - templates should be substantial but not excessive
    if len(template) < 50:
        raise InvalidTemplateError("Template too short - should provide clear instructions")
        
    if len(template) > 10000:
        raise InvalidTemplateError("Template too long - may cause processing issues")
    
    # Required placeholder validation
    required_placeholders = {"{text}", "{schema}"}
    optional_placeholders = {"{examples}"}
    all_valid_placeholders = required_placeholders | optional_placeholders
    
    # Extract placeholders while avoiding JSON structures
    temp_template = re.sub(r'\{[^}]*"[^"]*"[^}]*\}', '', template)
    found_placeholders = set(re.findall(r'\{[^}]+\}', temp_template))
    
    # Check for required placeholders
    missing_required = required_placeholders - found_placeholders
    if missing_required:
        raise InvalidTemplateError(
            f"Template missing required placeholders: {', '.join(missing_required)}"
        )
    
    # Check for invalid placeholders
    invalid_placeholders = found_placeholders - all_valid_placeholders
    if invalid_placeholders:
        raise InvalidTemplateError(
            f"Template contains invalid placeholders: {', '.join(invalid_placeholders)}"
        )
    
    # Instruction quality validation
    template_lower = template.lower()
    
    # Must mention JSON output
    if "json" not in template_lower:
        raise InvalidTemplateError("Template must specify JSON output format")
        
    # Must mention entities array or similar structure
    if not any(term in template_lower for term in ["entities", "array", "list"]):
        raise InvalidTemplateError("Template must specify entities array structure")
        
    # Required field mentions
    required_fields = ["text", "label", "start", "end", "confidence"]
    missing_fields = [field for field in required_fields if field not in template_lower]
    if missing_fields:
        raise InvalidTemplateError(
            f"Template must mention required fields: {', '.join(missing_fields)}"
        )
    
    # Instruction clarity - should contain action verbs
    action_verbs = ["extract", "identify", "find", "analyze", "recognize", "detect"]
    if not any(verb in template_lower for verb in action_verbs):
        raise InvalidTemplateError("Template should contain clear action instructions")
        
    # Example validation
    if "example" in template_lower:
        # If template mentions examples, it should show proper JSON structure
        json_patterns = [r'\{[^}]*"[^"]*"[^}]*\}', r'\[[^\]]*\]']
        has_json_example = any(re.search(pattern, template) for pattern in json_patterns)
        if not has_json_example:
            raise InvalidTemplateError("Template mentions examples but lacks proper JSON example")
    
    return True


def validate_examples_format(examples: List[Dict]) -> bool:
    """
    Validate the format and structure of training examples.
    
    Ensures examples conform to the expected format with proper entity
    annotations, character positions, and confidence scores.
    
    Args:
        examples: List of example dictionaries to validate
        
    Returns:
        True if all examples are properly formatted
        
    Raises:
        InvalidTemplateError: If examples format is invalid
        
    Example:
        >>> examples = [{
        ...     "text": "Quercetin is a flavonoid compound.",
        ...     "entities": [{
        ...         "text": "Quercetin", "label": "METABOLITE",
        ...         "start": 0, "end": 9, "confidence": 0.95
        ...     }]
        ... }]
        >>> validate_examples_format(examples)
        True
    """
    if not isinstance(examples, list):
        raise InvalidTemplateError("Examples must be a list")
        
    if not examples:
        raise InvalidTemplateError("Examples list cannot be empty")
        
    if len(examples) > 50:
        raise InvalidTemplateError("Too many examples (max 50)")
    
    for i, example in enumerate(examples):
        if not isinstance(example, dict):
            raise InvalidTemplateError(f"Example {i+1} must be a dictionary")
            
        # Required fields for examples
        required_fields = ["text", "entities"]
        for field in required_fields:
            if field not in example:
                raise InvalidTemplateError(f"Example {i+1} missing required field: {field}")
                
        # Validate text field
        text = example["text"]
        if not isinstance(text, str) or not text.strip():
            raise InvalidTemplateError(f"Example {i+1} text must be non-empty string")
            
        if len(text) > 2000:
            raise InvalidTemplateError(f"Example {i+1} text too long (max 2000 chars)")
            
        # Validate entities field
        entities = example["entities"]
        if not isinstance(entities, list):
            raise InvalidTemplateError(f"Example {i+1} entities must be a list")
            
        # Validate each entity
        for j, entity in enumerate(entities):
            if not isinstance(entity, dict):
                raise InvalidTemplateError(f"Example {i+1}, entity {j+1} must be a dictionary")
                
            # Required entity fields
            entity_fields = ["text", "label", "start", "end", "confidence"]
            for field in entity_fields:
                if field not in entity:
                    raise InvalidTemplateError(
                        f"Example {i+1}, entity {j+1} missing field: {field}"
                    )
            
            # Validate entity text
            entity_text = entity["text"]
            if not isinstance(entity_text, str) or not entity_text.strip():
                raise InvalidTemplateError(
                    f"Example {i+1}, entity {j+1} text must be non-empty string"
                )
                
            # Validate entity label
            label = entity["label"]
            if not isinstance(label, str) or not label.isupper():
                raise InvalidTemplateError(
                    f"Example {i+1}, entity {j+1} label must be uppercase string"
                )
                
            # Validate positions
            start, end = entity["start"], entity["end"]
            if not isinstance(start, int) or not isinstance(end, int):
                raise InvalidTemplateError(
                    f"Example {i+1}, entity {j+1} start/end must be integers"
                )
                
            if start < 0 or end < 0 or start >= end:
                raise InvalidTemplateError(
                    f"Example {i+1}, entity {j+1} invalid positions: start={start}, end={end}"
                )
                
            if end > len(text):
                raise InvalidTemplateError(
                    f"Example {i+1}, entity {j+1} end position beyond text length"
                )
                
            # Validate text span matches
            actual_span = text[start:end]
            if actual_span != entity_text:
                raise InvalidTemplateError(
                    f"Example {i+1}, entity {j+1} text span mismatch: "
                    f"expected '{entity_text}', got '{actual_span}'"
                )
                
            # Validate confidence
            confidence = entity["confidence"]
            if not isinstance(confidence, (int, float)):
                raise InvalidTemplateError(
                    f"Example {i+1}, entity {j+1} confidence must be numeric"
                )
                
            if not (0.0 <= confidence <= 1.0):
                raise InvalidTemplateError(
                    f"Example {i+1}, entity {j+1} confidence must be between 0.0 and 1.0"
                )
    
    return True


def optimize_prompt_for_model(prompt: str, model: str) -> str:
    """
    Optimize prompt template for specific language models.
    
    Adjusts prompt structure, length, and formatting based on known
    characteristics and preferences of different language models.
    
    Args:
        prompt: Original prompt template
        model: Model identifier (e.g., 'gpt-4', 'claude-3', 'gemini-pro')
        
    Returns:
        Optimized prompt template
        
    Example:
        >>> prompt = "Extract entities from {text}"
        >>> optimized = optimize_prompt_for_model(prompt, "gpt-4")
        >>> len(optimized) > len(prompt)
        True
    """
    model = model.lower().strip()
    
    # Model-specific optimizations
    if "gpt" in model:
        # GPT models prefer structured formatting with clear sections
        optimized = _optimize_for_gpt(prompt, model)
    elif "claude" in model:
        # Claude models work well with detailed instructions and examples
        optimized = _optimize_for_claude(prompt, model)
    elif "gemini" in model:
        # Gemini models prefer concise but complete instructions
        optimized = _optimize_for_gemini(prompt, model)
    elif "llama" in model:
        # Llama models benefit from explicit formatting instructions
        optimized = _optimize_for_llama(prompt, model)
    else:
        # Generic optimization for unknown models
        optimized = _optimize_generic(prompt)
    
    # Universal optimizations
    optimized = _apply_universal_optimizations(optimized)
    
    return optimized


def _optimize_for_gpt(prompt: str, model: str) -> str:
    """Optimize prompt for GPT models."""
    # GPT models prefer clear structure with headers
    if "**TASK:**" not in prompt:
        prompt = "**TASK:**\nNamed Entity Recognition for Plant Metabolomics\n\n" + prompt
    
    # Add system-style instruction for newer GPT models
    if "gpt-4" in model:
        system_instruction = "You are a specialized NER system for scientific literature. Follow instructions precisely.\n\n"
        if not prompt.startswith("You are"):
            prompt = system_instruction + prompt
    
    # Emphasize JSON format for GPT models
    if "JSON" not in prompt:
        prompt += "\n\n**CRITICAL: Return valid JSON only. No additional text.**"
    
    return prompt


def _optimize_for_claude(prompt: str, model: str) -> str:
    """Optimize prompt for Claude models."""
    # Claude works well with detailed explanations
    if len(prompt) < 500:
        prompt += "\n\n**DETAILED APPROACH:**\n"
        prompt += "1. Read the input text carefully\n"
        prompt += "2. Identify entities matching the schema\n"
        prompt += "3. Extract exact text spans with precise boundaries\n"
        prompt += "4. Assign appropriate confidence scores\n"
        prompt += "5. Format as valid JSON structure"
    
    # Claude benefits from explicit reasoning steps
    if "reasoning" not in prompt.lower():
        prompt += "\n\n**NOTE:** Apply systematic reasoning for each entity identification."
    
    return prompt


def _optimize_for_gemini(prompt: str, model: str) -> str:
    """Optimize prompt for Gemini models."""
    # Gemini prefers concise but complete instructions
    if len(prompt) > 1500:
        # Simplify verbose prompts for Gemini
        prompt = re.sub(r'\*\*[^*]+\*\*\s*\n', '', prompt)  # Remove section headers
        prompt = re.sub(r'\n\s*\n', '\n', prompt)  # Remove extra newlines
    
    # Add clear output specification
    if "JSON format" not in prompt:
        prompt += "\n\nReturn results in JSON format with entities array."
    
    return prompt


def _optimize_for_llama(prompt: str, model: str) -> str:
    """Optimize prompt for Llama models."""
    # Llama models need explicit format instructions
    format_instruction = (
        "\n\n**OUTPUT FORMAT:**\n"
        "Return only valid JSON. Structure: {\"entities\": [...]}\n"
        "Do not include explanations or additional text."
    )
    
    if "OUTPUT FORMAT" not in prompt:
        prompt += format_instruction
    
    # Add clear task definition
    if not prompt.startswith("Task:"):
        prompt = "Task: Extract named entities from scientific text.\n\n" + prompt
    
    return prompt


def _optimize_generic(prompt: str) -> str:
    """Generic optimization for unknown models."""
    # Add clear instruction structure
    if "INSTRUCTIONS:" not in prompt:
        prompt += "\n\n**INSTRUCTIONS:**\n1. Extract all entities\n2. Use exact text spans\n3. Return JSON format"
    
    return prompt


def _apply_universal_optimizations(prompt: str) -> str:
    """Apply optimizations that work for all models."""
    # Ensure consistent formatting
    prompt = re.sub(r'\n{3,}', '\n\n', prompt)  # Max 2 consecutive newlines
    
    # Add final JSON reminder if not present
    if not prompt.strip().endswith("}"):
        prompt += "\n\nReturn valid JSON only."
    
    return prompt.strip()


def calculate_template_metrics(template: str) -> Dict:
    """
    Calculate comprehensive metrics for a prompt template.
    
    Analyzes template complexity, instruction clarity, example coverage,
    and other quality indicators to help evaluate template effectiveness.
    
    Args:
        template: Template string to analyze
        
    Returns:
        Dictionary containing various template metrics
        
    Example:
        >>> template = "Extract entities from {text} using {schema}. Return JSON."
        >>> metrics = calculate_template_metrics(template)
        >>> metrics['word_count'] > 0
        True
    """
    # Basic text metrics
    word_count = len(template.split())
    char_count = len(template)
    line_count = len(template.split('\n'))
    
    # Placeholder analysis
    placeholders = re.findall(r'\{[^}]+\}', template)
    unique_placeholders = set(placeholders)
    
    # Instruction analysis
    template_lower = template.lower()
    
    # Count instruction keywords
    instruction_keywords = [
        'extract', 'identify', 'find', 'analyze', 'recognize', 'detect',
        'must', 'should', 'require', 'ensure', 'include', 'exclude'
    ]
    instruction_count = sum(template_lower.count(keyword) for keyword in instruction_keywords)
    
    # Quality indicators
    quality_indicators = {
        'has_json_spec': 'json' in template_lower,
        'has_examples': 'example' in template_lower,
        'has_confidence': 'confidence' in template_lower,
        'has_entity_fields': all(field in template_lower for field in ['text', 'label', 'start', 'end']),
        'has_clear_output': any(term in template_lower for term in ['return', 'output', 'format']),
        'has_error_handling': any(term in template_lower for term in ['error', 'invalid', 'fail'])
    }
    
    # Section analysis
    sections = re.findall(r'\*\*([^*]+)\*\*', template)
    section_types = {
        'task': any('task' in s.lower() for s in sections),
        'instructions': any('instruction' in s.lower() for s in sections),
        'examples': any('example' in s.lower() for s in sections),
        'output': any('output' in s.lower() or 'format' in s.lower() for s in sections)
    }
    
    # Complexity assessment
    complexity_score = (
        min(word_count / 100, 1.0) * 0.3 +
        min(len(unique_placeholders) / 5, 1.0) * 0.2 +
        min(instruction_count / 10, 1.0) * 0.3 +
        min(len(sections) / 8, 1.0) * 0.2
    )
    
    complexity_level = (
        'high' if complexity_score > 0.7 else
        'medium' if complexity_score > 0.4 else
        'low'
    )
    
    # Readability metrics (simplified)
    avg_sentence_length = word_count / max(template.count('.') + template.count('?') + template.count('!'), 1)
    readability_score = max(0, min(1, 1 - (avg_sentence_length - 15) / 20))  # Optimal around 15 words/sentence
    
    return {
        # Basic metrics
        'word_count': word_count,
        'character_count': char_count,
        'line_count': line_count,
        
        # Placeholder metrics
        'placeholder_count': len(placeholders),
        'unique_placeholders': len(unique_placeholders),
        'placeholders': list(unique_placeholders),
        
        # Instruction metrics
        'instruction_keyword_count': instruction_count,
        'instruction_density': instruction_count / word_count if word_count > 0 else 0,
        
        # Structure metrics
        'section_count': len(sections),
        'sections': sections,
        'section_types': section_types,
        
        # Quality metrics
        'quality_indicators': quality_indicators,
        'quality_score': sum(quality_indicators.values()) / len(quality_indicators),
        
        # Complexity metrics
        'complexity_score': complexity_score,
        'complexity_level': complexity_level,
        
        # Readability metrics
        'avg_sentence_length': avg_sentence_length,
        'readability_score': readability_score,
        
        # Overall assessment
        'estimated_effectiveness': _calculate_effectiveness_score(
            complexity_score, sum(quality_indicators.values()) / len(quality_indicators), readability_score
        )
    }


def _calculate_effectiveness_score(complexity: float, quality: float, readability: float) -> str:
    """Calculate overall template effectiveness."""
    overall_score = (complexity * 0.3 + quality * 0.5 + readability * 0.2)
    
    if overall_score > 0.8:
        return 'excellent'
    elif overall_score > 0.6:
        return 'good'
    elif overall_score > 0.4:
        return 'fair'
    else:
        return 'poor'


def suggest_template_improvements(template: str) -> List[str]:
    """
    Analyze template and suggest specific improvements.
    
    Provides actionable recommendations to enhance template effectiveness,
    clarity, and performance based on best practices and common issues.
    
    Args:
        template: Template string to analyze
        
    Returns:
        List of improvement suggestions
        
    Example:
        >>> template = "Extract entities from {text}"
        >>> suggestions = suggest_template_improvements(template)
        >>> len(suggestions) > 0
        True
    """
    suggestions = []
    metrics = calculate_template_metrics(template)
    template_lower = template.lower()
    
    # Check basic requirements
    if '{schema}' not in template:
        suggestions.append(
            "Add {schema} placeholder to include entity type definitions"
        )
    
    if 'json' not in template_lower:
        suggestions.append(
            "Explicitly specify JSON output format requirement"
        )
    
    # Check required fields specification
    required_fields = ['text', 'label', 'start', 'end', 'confidence']
    missing_fields = [field for field in required_fields if field not in template_lower]
    if missing_fields:
        suggestions.append(
            f"Specify required entity fields: {', '.join(missing_fields)}"
        )
    
    # Check for examples
    if 'example' not in template_lower:
        suggestions.append(
            "Add concrete examples to demonstrate expected output format"
        )
    
    # Length and complexity checks
    if metrics['word_count'] < 50:
        suggestions.append(
            "Expand template with more detailed instructions (currently too brief)"
        )
    elif metrics['word_count'] > 800:
        suggestions.append(
            "Consider simplifying template to improve clarity (currently very long)"
        )
    
    # Instruction clarity
    if metrics['instruction_density'] < 0.1:
        suggestions.append(
            "Add more specific action instructions (extract, identify, analyze)"
        )
    
    # Quality indicators
    quality = metrics['quality_indicators']
    
    if not quality['has_clear_output']:
        suggestions.append(
            "Add clear output format specification (Return JSON like: {...})"
        )
    
    if not quality['has_confidence']:
        suggestions.append(
            "Include confidence score requirements and guidelines"
        )
    
    # Structure suggestions
    if metrics['section_count'] < 3:
        suggestions.append(
            "Add section headers to improve template organization (e.g., **TASK:**, **INSTRUCTIONS:**)"
        )
    
    # Error handling
    if not quality['has_error_handling']:
        suggestions.append(
            "Add guidance for handling ambiguous or unclear entities"
        )
    
    # Domain-specific suggestions
    if 'scientific' in template_lower and 'nomenclature' not in template_lower:
        suggestions.append(
            "Include scientific nomenclature guidelines for accuracy"
        )
    
    if 'metabolomics' in template_lower and 'chemical' not in template_lower:
        suggestions.append(
            "Add chemical compound identification guidelines"
        )
    
    # Performance suggestions
    if metrics['complexity_level'] == 'high' and metrics['readability_score'] < 0.5:
        suggestions.append(
            "Simplify sentence structure to improve readability"
        )
    
    # Context-aware suggestions
    if '{examples}' in template and 'few-shot' not in template_lower:
        suggestions.append(
            "Specify how examples should be used for few-shot learning"
        )
    
    # Final validation
    try:
        validate_template_structure(template)
    except InvalidTemplateError as e:
        suggestions.append(f"Fix template structure: {str(e)}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_suggestions = []
    for suggestion in suggestions:
        if suggestion not in seen:
            seen.add(suggestion)
            unique_suggestions.append(suggestion)
    
    return unique_suggestions


# Template Registry and Management Functions

def register_custom_template(name: str, template: str, description: str = "") -> bool:
    """
    Register a custom template in the template registry.
    
    Args:
        name: Unique name for the template
        template: Template string
        description: Optional description of the template
        
    Returns:
        True if registration successful
        
    Raises:
        InvalidTemplateError: If template is invalid
        ValueError: If name already exists
    """
    name = name.lower().strip()
    
    if name in TEMPLATE_REGISTRY:
        raise ValueError(f"Template name '{name}' already exists in registry")
    
    # Validate template before registration
    validate_template_structure(template)
    
    # Add to registry
    TEMPLATE_REGISTRY[name] = template
    
    # Store metadata if needed (could extend to include descriptions)
    return True


def get_template_metadata(template_name: str) -> Dict[str, Any]:
    """
    Get metadata and metrics for a registered template.
    
    Args:
        template_name: Name of the template
        
    Returns:
        Dictionary with template metadata and metrics
        
    Raises:
        TemplateNotFoundError: If template not found
    """
    if template_name not in TEMPLATE_REGISTRY:
        raise TemplateNotFoundError(f"Template '{template_name}' not found in registry")
    
    template = TEMPLATE_REGISTRY[template_name]
    metrics = calculate_template_metrics(template)
    
    return {
        'name': template_name,
        'template_type': _determine_template_type(template),
        'domain_focus': _determine_domain_focus(template),
        'use_case': _determine_use_case(template),
        'metrics': metrics,
        'suggestions': suggest_template_improvements(template)
    }


def _determine_template_type(template: str) -> str:
    """Determine the type of template based on content analysis."""
    template_lower = template.lower()
    
    if 'few-shot' in template_lower or '{examples}' in template:
        return 'few-shot'
    elif 'precision' in template_lower:
        return 'precision-focused'
    elif 'recall' in template_lower:
        return 'recall-focused'
    elif 'scientific' in template_lower:
        return 'scientific-literature'
    else:
        return 'zero-shot'


def _determine_domain_focus(template: str) -> str:
    """Determine the domain focus of a template."""
    template_lower = template.lower()
    
    domain_keywords = {
        'metabolomics': ['metabolite', 'compound', 'chemical', 'flavonoid'],
        'genetics': ['gene', 'protein', 'expression', 'transcription'],
        'plant_biology': ['plant', 'leaf', 'root', 'tissue', 'anatomy'],
        'analytical': ['analysis', 'chromatography', 'spectroscopy', 'hplc']
    }
    
    for domain, keywords in domain_keywords.items():
        if any(keyword in template_lower for keyword in keywords):
            return domain
    
    return 'general'


def _determine_use_case(template: str) -> str:
    """Determine the intended use case of a template."""
    template_lower = template.lower()
    
    if 'research' in template_lower or 'literature' in template_lower:
        return 'research'
    elif 'quick' in template_lower or 'basic' in template_lower:
        return 'quick_analysis'
    elif 'comprehensive' in template_lower or 'detailed' in template_lower:
        return 'detailed_analysis'
    else:
        return 'general_purpose'


def compare_templates(template1: str, template2: str) -> Dict[str, Any]:
    """
    Compare two templates and provide detailed analysis.
    
    Args:
        template1: First template to compare
        template2: Second template to compare
        
    Returns:
        Dictionary with comparison results
    """
    metrics1 = calculate_template_metrics(template1)
    metrics2 = calculate_template_metrics(template2)
    
    comparison = {
        'template1_metrics': metrics1,
        'template2_metrics': metrics2,
        'differences': {
            'word_count_diff': metrics2['word_count'] - metrics1['word_count'],
            'complexity_diff': metrics2['complexity_score'] - metrics1['complexity_score'],
            'quality_diff': metrics2['quality_score'] - metrics1['quality_score'],
            'readability_diff': metrics2['readability_score'] - metrics1['readability_score']
        },
        'recommendation': _get_template_recommendation(metrics1, metrics2)
    }
    
    return comparison


def _get_template_recommendation(metrics1: Dict, metrics2: Dict) -> str:
    """Get recommendation on which template is better."""
    score1 = (
        metrics1['complexity_score'] * 0.3 +
        metrics1['quality_score'] * 0.5 +
        metrics1['readability_score'] * 0.2
    )
    
    score2 = (
        metrics2['complexity_score'] * 0.3 +
        metrics2['quality_score'] * 0.5 +
        metrics2['readability_score'] * 0.2
    )
    
    if abs(score1 - score2) < 0.1:
        return "Templates are roughly equivalent in quality"
    elif score1 > score2:
        return "Template 1 appears to be better overall"
    else:
        return "Template 2 appears to be better overall"


def get_template_recommendations(requirements: Dict[str, Any]) -> List[str]:
    """
    Get template recommendations based on specific requirements.
    
    Args:
        requirements: Dictionary with requirements like domain, accuracy_priority, etc.
        
    Returns:
        List of recommended template names sorted by suitability
    """
    domain = requirements.get('domain', 'general')
    accuracy_priority = requirements.get('accuracy_priority', 'balanced')
    complexity_preference = requirements.get('complexity', 'medium')
    use_case = requirements.get('use_case', 'general')
    
    # Score all templates
    template_scores = []
    
    for template_name in TEMPLATE_REGISTRY:
        try:
            metadata = get_template_metadata(template_name)
            score = _calculate_template_suitability_score(
                metadata, domain, accuracy_priority, complexity_preference, use_case
            )
            template_scores.append((template_name, score))
        except Exception:
            continue  # Skip templates that can't be analyzed
    
    # Sort by score and return top recommendations
    template_scores.sort(key=lambda x: x[1], reverse=True)
    return [name for name, score in template_scores[:10]]  # Top 10 recommendations


def _calculate_template_suitability_score(
    metadata: Dict[str, Any],
    domain: str,
    accuracy_priority: str,
    complexity_preference: str,
    use_case: str
) -> float:
    """Calculate how suitable a template is for given requirements."""
    score = 0.0
    
    # Domain match
    if metadata['domain_focus'] == domain:
        score += 0.4
    elif metadata['domain_focus'] == 'general':
        score += 0.2
    
    # Use case match
    if metadata['use_case'] == use_case:
        score += 0.3
    elif metadata['use_case'] == 'general_purpose':
        score += 0.15
    
    # Accuracy priority match
    template_type = metadata['template_type']
    if accuracy_priority == 'precision' and 'precision' in template_type:
        score += 0.2
    elif accuracy_priority == 'recall' and 'recall' in template_type:
        score += 0.2
    elif accuracy_priority == 'balanced' and template_type in ['zero-shot', 'few-shot']:
        score += 0.15
    
    # Complexity preference
    complexity = metadata['metrics']['complexity_level']
    if complexity_preference == complexity:
        score += 0.1
    
    return score


def validate_template_output_format(template: str) -> bool:
    """
    Validate that template includes proper output format specifications.

    Args:
        template: Template to validate

    Returns:
        True if output format is properly specified

    Raises:
        InvalidTemplateError: If output format specification is inadequate
    """
    template_lower = template.lower()

    # Check for JSON specification
    if not any(term in template_lower for term in ["json", "entities", "array"]):
        raise InvalidTemplateError("Template must specify JSON output with entities array")

    # Check for required field specifications
    required_fields = ["text", "label", "start", "end", "confidence"]
    for field in required_fields:
        if field not in template_lower:
            raise InvalidTemplateError(f"Template must specify '{field}' field requirement")

    # Check for example output
    if not any(pattern in template for pattern in ["{", "}", "["]) or "example" not in template_lower:
        raise InvalidTemplateError("Template should include example JSON output")

    return True


def get_recommended_template(
    text_length: int,
    entity_count_estimate: int,
    domain: Optional[str] = None,
    accuracy_priority: str = "balanced"
) -> str:
    """
    Get recommended template based on text characteristics and requirements.

    Args:
        text_length: Length of text to process (in characters)
        entity_count_estimate: Estimated number of entities in text
        domain: Optional domain specification
        accuracy_priority: "precision", "recall", or "balanced"

    Returns:
        Recommended template string
    """
    accuracy_priority = accuracy_priority.lower().strip()

    # For short texts with few entities, use basic template
    if text_length < 500 and entity_count_estimate < 10:
        return get_basic_zero_shot_template()

    # For domain-specific content, use domain template
    if domain:
        try:
            return get_domain_specific_template(domain)
        except TemplateNotFoundError:
            pass  # Fall through to other recommendations

    # For scientific papers, use scientific template
    if text_length > 2000:
        return get_scientific_literature_template()

    # Choose based on accuracy priority
    if accuracy_priority == "precision":
        return get_precision_focused_template()
    elif accuracy_priority == "recall":
        return get_recall_focused_template()
    else:
        return get_detailed_zero_shot_template()


# Relationship extraction templates
RELATIONSHIP_BASIC_TEMPLATE = """Extract relationships between entities from the following text.

Text: {text}

Entities:
{entities}

Available relationship types:
{schema}

Instructions:
1. Identify meaningful relationships between the provided entities
2. Use only the relationship types from the schema above
3. Return results in the specified JSON format
4. Each relationship should have a confidence score between 0.0 and 1.0
5. Include supporting context where possible

{examples}

Output the relationships in this JSON format:
{{
    "relationships": [
        {{
            "subject_entity": {{"text": "entity1", "label": "LABEL1"}},
            "relation_type": "relationship_type",
            "object_entity": {{"text": "entity2", "label": "LABEL2"}},
            "confidence": 0.95,
            "context": "supporting context",
            "evidence": "text span supporting relationship"
        }}
    ]
}}"""

RELATIONSHIP_DETAILED_TEMPLATE = """You are an expert in extracting semantic relationships from scientific text. Your task is to identify meaningful relationships between the provided entities.

Text: {text}

Entities to consider:
{entities}

Relationship Schema:
{schema}

Guidelines:
1. ENTITY MATCHING: Only create relationships between entities that are explicitly provided in the entities list
2. RELATIONSHIP TYPES: Use only the relationship types defined in the schema above
3. DIRECTION: Pay attention to the direction of relationships (subject -> relation -> object)
4. CONFIDENCE: Assign confidence scores based on:
   - 0.9-1.0: Explicitly stated in text with clear evidence
   - 0.7-0.9: Strongly implied with good contextual support
   - 0.5-0.7: Reasonably inferred from context
   - Below 0.5: Uncertain or speculative (avoid these)
5. CONTEXT: Include relevant context that supports the relationship
6. EVIDENCE: Provide the text span that most directly supports the relationship

{examples}

Output Format:
Return a JSON object with the following structure:
{{
    "relationships": [
        {{
            "subject_entity": {{"text": "subject_text", "label": "SUBJECT_LABEL"}},
            "relation_type": "valid_relationship_type",
            "object_entity": {{"text": "object_text", "label": "OBJECT_LABEL"}},
            "confidence": 0.95,
            "context": "relevant surrounding context",
            "evidence": "specific text span supporting this relationship"
        }}
    ]
}}

If no relationships are found, return: {{"relationships": []}}"""

RELATIONSHIP_SCIENTIFIC_TEMPLATE = """Extract semantic relationships from scientific text with high precision and domain expertise.

Scientific Text: {text}

Identified Entities:
{entities}

Relationship Taxonomy:
{schema}

Scientific Relationship Extraction Protocol:
1. PRECISION FIRST: Only extract relationships with strong textual evidence
2. NO SPECULATION: Avoid inferring relationships not clearly supported by the text
3. DOMAIN EXPERTISE: Apply scientific domain knowledge for accurate interpretation
4. RELATIONSHIP HIERARCHY: Choose the most specific applicable relationship type
5. BIDIRECTIONAL ANALYSIS: Consider if relationships should be bidirectional
6. TEMPORAL CONTEXT: Account for temporal or conditional relationships when mentioned

Quality Criteria:
- Confidence ≥ 0.8 for scientific literature
- Clear evidence in original text
- Scientifically meaningful relationships
- Proper entity pairing based on scientific logic

{examples}

Expected JSON Output:
{{
    "relationships": [
        {{
            "subject_entity": {{"text": "entity_text", "label": "ENTITY_TYPE"}},
            "relation_type": "precise_relationship",
            "object_entity": {{"text": "target_text", "label": "TARGET_TYPE"}},
            "confidence": 0.92,
            "context": "scientific context supporting relationship",
            "evidence": "exact text evidence"
        }}
    ]
}}"""

RELATIONSHIP_METABOLOMICS_TEMPLATE = """Extract metabolomics-specific relationships from plant science literature.

Text: {text}

Entities:
{entities}

Metabolomics Relationship Types:
{schema}

Domain-Specific Guidelines:
1. METABOLIC PATHWAYS: Focus on biosynthetic and catabolic relationships
2. ENZYME-SUBSTRATE: Identify catalytic relationships and conversions  
3. LOCALIZATION: Extract spatial relationships (tissue, cellular, subcellular)
4. REGULATION: Capture regulatory relationships (up/downregulation)
5. ANALYTICAL: Include detection and measurement relationships
6. STRESS RESPONSES: Identify stress-related metabolic changes

Metabolomics Context:
- Consider pathway hierarchies and metabolic networks
- Account for tissue-specific expression and accumulation
- Recognize analytical method associations
- Include environmental and stress response relationships

{examples}

JSON Output Format:
{{
    "relationships": [
        {{
            "subject_entity": {{"text": "metabolite", "label": "METABOLITE"}},
            "relation_type": "synthesized_by",
            "object_entity": {{"text": "enzyme", "label": "ENZYME"}},
            "confidence": 0.94,
            "context": "in specific tissue under conditions",
            "evidence": "direct text support"
        }}
    ]
}}"""


# Advanced relationship extraction templates with hierarchical differentiation

RELATIONSHIP_HIERARCHICAL_TEMPLATE = """Extract relationships between entities with hierarchical differentiation between broad and specific relationship types.

Text: {text}

Entities:
{entities}

Relationship Hierarchy Schema:
{schema}

HIERARCHICAL DIFFERENTIATION GUIDELINES:
This is critical - always choose the MOST SPECIFIC relationship type that applies:

1. BROAD vs SPECIFIC Relationships:
   - Use BROAD relationships (like "involved_in", "affects", "associated_with") only when more specific types don't apply
   - Always prefer SPECIFIC relationships (like "upregulates", "synthesized_by", "accumulates_in") when there's evidence
   
2. Hierarchy Examples:
   - Instead of "involved_in" → use "catalyzes", "synthesized_by", "regulated_by"
   - Instead of "affects" → use "upregulates", "downregulates", "increases_under", "decreases_under"
   - Instead of "found_in" → use "accumulates_in", "expressed_in", "localized_in"
   - Instead of "related_to" → use "derived_from", "converted_to", "precursor_of"

3. Context-Dependent Selection:
   - "Gene X affects metabolite Y" → Use "upregulates" or "downregulates" if direction is clear
   - "Metabolite found in tissue" → Use "accumulates_in" if concentration/storage is implied
   - "Enzyme produces compound" → Use "synthesized_by" (compound synthesized_by enzyme)
   - "Stress increases compound" → Use "increases_under" for environmental effects

4. Evidence Requirements by Specificity:
   - SPECIFIC relationships (0.8+ confidence): Need clear textual evidence
   - BROAD relationships (0.6+ confidence): Use only when specific types don't fit

{examples}

OUTPUT FORMAT:
Return JSON with relationships ranked by specificity (most specific first):
{{
    "relationships": [
        {{
            "subject_entity": {{"text": "entity1", "label": "LABEL1"}},
            "relation_type": "specific_relationship_type",
            "object_entity": {{"text": "entity2", "label": "LABEL2"}},
            "confidence": 0.92,
            "specificity_level": "high",
            "alternative_broader_types": ["broader_type1", "broader_type2"],
            "context": "supporting context",
            "evidence": "text span supporting relationship",
            "reasoning": "why this specific type was chosen over broader ones"
        }}
    ]
}}"""


RELATIONSHIP_CONTEXTUAL_TEMPLATE = """Extract relationships with sophisticated contextual understanding and conditional relationship selection.

Text: {text}

Entities:
{entities}

Relationship Schema:
{schema}

CONTEXTUAL UNDERSTANDING PROTOCOL:

1. CONDITIONAL RELATIONSHIPS:
   - Consider environmental conditions (stress, treatment, developmental stage)
   - Account for tissue/organ specificity
   - Recognize temporal dependencies (before/after, during/after treatment)

2. CONTEXT-DEPENDENT RELATIONSHIP SELECTION:
   - Same entity pairs may have different relationships in different contexts
   - Environmental modifiers: "increases_under_stress", "decreases_under_drought"
   - Temporal modifiers: "expressed_during_flowering", "accumulated_after_treatment"
   - Spatial modifiers: "localized_in_root", "transported_to_leaf"

3. MULTI-LAYERED RELATIONSHIP ANALYSIS:
   - Direct relationships: "A synthesized_by B"
   - Conditional relationships: "A increases_under C when B is present"
   - Regulatory relationships: "A upregulates B in response to C"

4. EVIDENCE CONTEXTUALIZATION:
   - Quote exact conditions mentioned in text
   - Include temporal markers ("after 24h", "during stress", "in flowering stage")
   - Capture quantitative context ("2-fold increase", "significantly reduced")

{examples}

CONDITIONAL RELATIONSHIP EXAMPLES:
- "Under drought stress, proline accumulates in leaf tissues" → proline "accumulates_in" leaf AND proline "increases_under" drought
- "ABA upregulates stress genes during water deficit" → ABA "upregulates" genes AND ABA "active_during" water_deficit
- "Anthocyanins are synthesized by CHS in response to UV light" → anthocyanins "synthesized_by" CHS AND CHS "activated_by" UV_light

JSON OUTPUT:
{{
    "relationships": [
        {{
            "subject_entity": {{"text": "entity1", "label": "LABEL1"}},
            "relation_type": "contextual_relationship",
            "object_entity": {{"text": "entity2", "label": "LABEL2"}},
            "confidence": 0.89,
            "context_conditions": ["condition1", "condition2"],
            "temporal_context": "temporal_info",
            "spatial_context": "spatial_info",
            "quantitative_context": "quantitative_info",
            "evidence": "supporting text with context",
            "conditional_modifiers": ["modifier1", "modifier2"]
        }}
    ]
}}"""


RELATIONSHIP_MULTI_TYPE_TEMPLATE = """Extract multiple relationship types simultaneously with comprehensive coverage and cross-validation.

Text: {text}

Entities:
{entities}

Complete Relationship Schema:
{schema}

MULTI-TYPE EXTRACTION PROTOCOL:

1. COMPREHENSIVE COVERAGE:
   - Extract ALL valid relationships, not just the most obvious ones
   - Consider direct, indirect, and conditional relationships
   - Look for pathway-level connections and regulatory networks

2. RELATIONSHIP CATEGORIES TO CONSIDER:
   a) BIOSYNTHETIC: synthesis, conversion, derivation pathways
   b) REGULATORY: up/downregulation, activation, inhibition
   c) LOCALIZATION: spatial distribution, accumulation, transport
   d) FUNCTIONAL: catalysis, binding, interaction
   e) ENVIRONMENTAL: stress responses, condition-dependent changes
   f) ANALYTICAL: detection, measurement, quantification methods

3. CROSS-VALIDATION CHECKS:
   - Ensure relationship consistency (no contradictory relationships)
   - Validate entity-relationship compatibility using domain knowledge
   - Check for missing inverse relationships when applicable

4. CONFIDENCE CALIBRATION:
   - Primary relationships (directly stated): 0.85-1.0
   - Secondary relationships (strongly implied): 0.70-0.85  
   - Supporting relationships (contextually inferred): 0.55-0.70

{examples}

MULTI-TYPE EXTRACTION EXAMPLE:
Text: "Chalcone synthase catalyzes the first step of flavonoid biosynthesis, converting coumaroyl-CoA to naringenin chalcone, which accumulates in flower petals under UV stress."

Expected relationships:
1. chalcone_synthase "catalyzes" flavonoid_biosynthesis
2. coumaroyl-CoA "converted_to" naringenin_chalcone  
3. chalcone_synthase "synthesizes" naringenin_chalcone
4. naringenin_chalcone "accumulates_in" flower_petals
5. naringenin_chalcone "increases_under" UV_stress
6. flavonoid_biosynthesis "starts_with" chalcone_synthase

JSON OUTPUT:
{{
    "relationships": [
        {{
            "subject_entity": {{"text": "entity1", "label": "LABEL1"}},
            "relation_type": "relationship_type",
            "object_entity": {{"text": "entity2", "label": "LABEL2"}},
            "confidence": 0.92,
            "category": "biosynthetic|regulatory|localization|functional|environmental|analytical",
            "evidence": "supporting text",
            "cross_references": ["related_relationship_ids"]
        }}
    ],
    "relationship_summary": {{
        "total_relationships": 6,
        "categories": {{"biosynthetic": 3, "localization": 2, "environmental": 1}},
        "confidence_distribution": {{"high": 4, "medium": 2, "low": 0}}
    }}
}}"""


# Few-shot relationship extraction templates with domain-specific examples

RELATIONSHIP_FEW_SHOT_METABOLOMICS_TEMPLATE = """Extract relationships from plant metabolomics text using few-shot learning with domain-specific examples.

LEARNING EXAMPLES:

Example 1:
Text: "Anthocyanins are synthesized by anthocyanidin synthase and accumulate in grape berry skin during ripening."
Entities: [("anthocyanins", "METABOLITE"), ("anthocyanidin synthase", "ENZYME"), ("grape berry skin", "PLANT_TISSUE"), ("ripening", "DEVELOPMENTAL_STAGE")]
Relationships: [
    {{
        "subject_entity": {{"text": "anthocyanins", "label": "METABOLITE"}},
        "relation_type": "synthesized_by",
        "object_entity": {{"text": "anthocyanidin synthase", "label": "ENZYME"}},
        "confidence": 0.95,
        "evidence": "anthocyanins are synthesized by anthocyanidin synthase"
    }},
    {{
        "subject_entity": {{"text": "anthocyanins", "label": "METABOLITE"}},
        "relation_type": "accumulates_in",
        "object_entity": {{"text": "grape berry skin", "label": "PLANT_TISSUE"}},
        "confidence": 0.92,
        "evidence": "accumulate in grape berry skin"
    }},
    {{
        "subject_entity": {{"text": "anthocyanins", "label": "METABOLITE"}},
        "relation_type": "increases_during",
        "object_entity": {{"text": "ripening", "label": "DEVELOPMENTAL_STAGE"}},
        "confidence": 0.88,
        "evidence": "accumulate...during ripening"
    }}
]

Example 2:
Text: "Under drought stress, proline levels increased 3-fold in root tissues, while being regulated by P5CS enzyme."
Entities: [("proline", "AMINO_ACID"), ("drought stress", "STRESS_CONDITION"), ("root tissues", "PLANT_TISSUE"), ("P5CS", "ENZYME")]
Relationships: [
    {{
        "subject_entity": {{"text": "proline", "label": "AMINO_ACID"}},
        "relation_type": "increases_under",
        "object_entity": {{"text": "drought stress", "label": "STRESS_CONDITION"}},
        "confidence": 0.94,
        "evidence": "Under drought stress, proline levels increased 3-fold"
    }},
    {{
        "subject_entity": {{"text": "proline", "label": "AMINO_ACID"}},
        "relation_type": "accumulates_in",
        "object_entity": {{"text": "root tissues", "label": "PLANT_TISSUE"}},
        "confidence": 0.91,
        "evidence": "proline levels increased 3-fold in root tissues"
    }},
    {{
        "subject_entity": {{"text": "proline", "label": "AMINO_ACID"}},
        "relation_type": "regulated_by",
        "object_entity": {{"text": "P5CS", "label": "ENZYME"}},
        "confidence": 0.89,
        "evidence": "being regulated by P5CS enzyme"
    }}
]

Example 3:
Text: "Quercetin, derived from kaempferol, exhibits antioxidant activity and is detected by HPLC analysis."
Entities: [("quercetin", "FLAVONOID"), ("kaempferol", "FLAVONOID"), ("antioxidant activity", "BIOLOGICAL_ACTIVITY"), ("HPLC", "ANALYTICAL_METHOD")]
Relationships: [
    {{
        "subject_entity": {{"text": "quercetin", "label": "FLAVONOID"}},
        "relation_type": "derived_from",
        "object_entity": {{"text": "kaempferol", "label": "FLAVONOID"}},
        "confidence": 0.93,
        "evidence": "Quercetin, derived from kaempferol"
    }},
    {{
        "subject_entity": {{"text": "quercetin", "label": "FLAVONOID"}},
        "relation_type": "exhibits",
        "object_entity": {{"text": "antioxidant activity", "label": "BIOLOGICAL_ACTIVITY"}},
        "confidence": 0.96,
        "evidence": "exhibits antioxidant activity"
    }},
    {{
        "subject_entity": {{"text": "quercetin", "label": "FLAVONOID"}},
        "relation_type": "detected_by",
        "object_entity": {{"text": "HPLC", "label": "ANALYTICAL_METHOD"}},
        "confidence": 0.92,
        "evidence": "is detected by HPLC analysis"
    }}
]

NOW EXTRACT RELATIONSHIPS FROM THE FOLLOWING TEXT:

Text: {text}

Entities:
{entities}

Relationship Schema:
{schema}

Apply the patterns learned from the examples above. Focus on:
1. Specific relationship types over broad ones
2. Multiple relationships per entity when appropriate  
3. Confidence based on evidence strength
4. Domain-specific biological relationships

{examples}

JSON Output:
{{
    "relationships": [
        {{
            "subject_entity": {{"text": "entity_text", "label": "ENTITY_LABEL"}},
            "relation_type": "specific_relationship",
            "object_entity": {{"text": "target_text", "label": "TARGET_LABEL"}},
            "confidence": 0.92,
            "evidence": "supporting text evidence"
        }}
    ]
}}"""


RELATIONSHIP_FEW_SHOT_HIERARCHICAL_TEMPLATE = """Extract relationships with hierarchical differentiation using few-shot examples that demonstrate broad vs specific relationship selection.

HIERARCHICAL DIFFERENTIATION EXAMPLES:

Example 1 - SPECIFIC over BROAD:
Text: "CYP75A upregulates anthocyanin biosynthesis genes in response to cold stress."
❌ BROAD: CYP75A "affects" anthocyanin_biosynthesis (too general)
✅ SPECIFIC: CYP75A "upregulates" anthocyanin_biosynthesis (preferred - more precise)

Example 2 - ACCUMULATION vs PRESENCE:
Text: "High concentrations of resveratrol were found in grape skins after UV treatment."
❌ BROAD: resveratrol "found_in" grape_skins (too general)
✅ SPECIFIC: resveratrol "accumulates_in" grape_skins (preferred - implies concentration)

Example 3 - ENZYMATIC SPECIFICITY:
Text: "Phenylalanine ammonia-lyase catalyzes the first step of phenylpropanoid metabolism."
❌ BROAD: PAL "involved_in" phenylpropanoid_metabolism (too general)  
✅ SPECIFIC: PAL "catalyzes" phenylpropanoid_metabolism (preferred - shows enzyme function)

Example 4 - REGULATORY PRECISION:
Text: "ABA treatment significantly increased proline synthesis in stressed plants."
❌ BROAD: ABA "affects" proline_synthesis (too general)
✅ SPECIFIC: ABA "upregulates" proline_synthesis (preferred - shows direction)

Example 5 - ENVIRONMENTAL RELATIONSHIPS:
Text: "Drought conditions led to elevated trehalose levels in root tissues."
❌ BROAD: trehalose "associated_with" drought (too vague)
✅ SPECIFIC: trehalose "increases_under" drought (preferred - shows environmental response)

HIERARCHICAL DECISION TREE:
1. Is there enzymatic activity mentioned? → Use "catalyzes", "synthesized_by"
2. Is there directional regulation? → Use "upregulates", "downregulates"  
3. Is there accumulation/concentration? → Use "accumulates_in", not "found_in"
4. Is there environmental response? → Use "increases_under", "decreases_under"
5. Is there derivation/conversion? → Use "derived_from", "converted_to"
6. Only use broad terms ("affects", "involved_in") when specific ones don't apply

NOW EXTRACT RELATIONSHIPS:

Text: {text}

Entities:
{entities}

Relationship Schema:
{schema}

For each potential relationship, ask:
- Is there a more specific relationship type that applies?
- What is the strongest evidence for this specific relationship?
- Can I justify choosing this over a broader alternative?

{examples}

JSON Output:
{{
    "relationships": [
        {{
            "subject_entity": {{"text": "entity", "label": "LABEL"}},
            "relation_type": "most_specific_applicable_type",
            "object_entity": {{"text": "target", "label": "TARGET_LABEL"}},
            "confidence": 0.88,
            "specificity_justification": "chose 'synthesized_by' over 'produced_by' because enzymatic synthesis is explicitly mentioned",
            "alternative_broader_types": ["produced_by", "made_by", "involved_in"],
            "evidence": "exact text supporting the specific relationship"
        }}
    ]
}}"""


RELATIONSHIP_FEW_SHOT_CONTEXTUAL_TEMPLATE = """Extract relationships with sophisticated contextual understanding using examples that show context-dependent relationship selection.

CONTEXTUAL RELATIONSHIP EXAMPLES:

Example 1 - CONDITIONAL CONTEXT:
Text: "Salicylic acid activates pathogenesis-related genes only under pathogen attack conditions."
Context Analysis: The relationship is conditional on pathogen presence
Relationships:
- salicylic_acid "activates" pathogenesis_related_genes (base relationship)  
- salicylic_acid "active_under" pathogen_attack (conditional context)
- pathogenesis_related_genes "expressed_during" pathogen_attack (temporal context)

Example 2 - TISSUE-SPECIFIC CONTEXT:
Text: "Lignin biosynthesis genes are highly expressed in stem tissues but downregulated in leaf tissues during development."
Context Analysis: Same genes have opposite expression patterns in different tissues
Relationships:
- lignin_biosynthesis_genes "highly_expressed_in" stem_tissues
- lignin_biosynthesis_genes "downregulated_in" leaf_tissues  
- lignin_biosynthesis_genes "active_during" development

Example 3 - TEMPORAL CONTEXT:
Text: "ABA levels peaked at 6 hours post-stress and then gradually decreased, correlating with stress gene expression."
Context Analysis: Temporal pattern with correlation
Relationships:
- ABA "peaks_at" 6_hours_post_stress
- ABA "decreases_after" 6_hours
- ABA "correlates_with" stress_gene_expression
- stress_gene_expression "follows_pattern" ABA_levels

Example 4 - QUANTITATIVE CONTEXT:
Text: "Moderate drought (30% soil moisture) increased proline 2-fold, while severe drought (10% soil moisture) increased it 5-fold."
Context Analysis: Dose-dependent relationship with quantitative thresholds
Relationships:
- proline "increases_under" moderate_drought (context: "2-fold at 30% soil moisture")
- proline "highly_increases_under" severe_drought (context: "5-fold at 10% soil moisture")  
- moderate_drought "precedes" severe_drought (intensity relationship)

Example 5 - MULTI-FACTOR CONTEXT:
Text: "Cold-induced anthocyanin accumulation requires both low temperature and high light intensity in Arabidopsis leaves."
Context Analysis: Multiple required conditions
Relationships:
- anthocyanin "accumulates_under" cold_temperature (requires high_light)
- anthocyanin "requires" low_temperature (for cold_induction)
- anthocyanin "requires" high_light_intensity (for cold_induction)
- cold_induction "occurs_in" Arabidopsis_leaves (spatial context)

CONTEXTUAL EXTRACTION PROTOCOL:
1. Identify all contextual modifiers (time, space, conditions, quantities)
2. Determine if relationships are conditional, tissue-specific, or temporal
3. Extract both direct relationships and contextual relationships
4. Capture quantitative and qualitative context details

NOW EXTRACT RELATIONSHIPS:

Text: {text}

Entities:
{entities}

Relationship Schema:
{schema}

Look for:
- Environmental conditions and their effects
- Tissue/organ/developmental stage specificity  
- Temporal sequences and dependencies
- Quantitative relationships and thresholds
- Multi-factor requirements and interactions

{examples}

JSON Output:
{{
    "relationships": [
        {{
            "subject_entity": {{"text": "entity", "label": "LABEL"}},
            "relation_type": "context_aware_relationship",
            "object_entity": {{"text": "target", "label": "TARGET_LABEL"}},
            "confidence": 0.91,
            "context_type": "temporal|spatial|conditional|quantitative|multi_factor",
            "context_details": {{
                "conditions": ["condition1", "condition2"],
                "temporal_info": "timing information",
                "spatial_info": "location/tissue information",
                "quantitative_info": "measurements/fold changes",
                "dependencies": ["required_factor1", "required_factor2"]
            }},
            "evidence": "text with full context"
        }}
    ]
}}"""


def get_relationship_template(template_type: str = "basic") -> str:
    """
    Get relationship extraction template by type.
    
    Args:
        template_type: Type of template ("basic", "detailed", "scientific", "metabolomics")
        
    Returns:
        Relationship extraction template string
        
    Raises:
        TemplateNotFoundError: If template type is not found
    """
    templates = {
        "basic": RELATIONSHIP_BASIC_TEMPLATE,
        "detailed": RELATIONSHIP_DETAILED_TEMPLATE,
        "scientific": RELATIONSHIP_SCIENTIFIC_TEMPLATE,
        "metabolomics": RELATIONSHIP_METABOLOMICS_TEMPLATE,
        "hierarchical": RELATIONSHIP_HIERARCHICAL_TEMPLATE,
        "contextual": RELATIONSHIP_CONTEXTUAL_TEMPLATE,
        "multi_type": RELATIONSHIP_MULTI_TYPE_TEMPLATE,
        "few_shot_metabolomics": RELATIONSHIP_FEW_SHOT_METABOLOMICS_TEMPLATE,
        "few_shot_hierarchical": RELATIONSHIP_FEW_SHOT_HIERARCHICAL_TEMPLATE,
        "few_shot_contextual": RELATIONSHIP_FEW_SHOT_CONTEXTUAL_TEMPLATE,
        # Legacy aliases
        "relationship_basic": RELATIONSHIP_BASIC_TEMPLATE,
        "relationship_detailed": RELATIONSHIP_DETAILED_TEMPLATE,
        "relationship_scientific": RELATIONSHIP_SCIENTIFIC_TEMPLATE,
        "relationship_metabolomics": RELATIONSHIP_METABOLOMICS_TEMPLATE,
        "relationship_hierarchical": RELATIONSHIP_HIERARCHICAL_TEMPLATE,
        "relationship_contextual": RELATIONSHIP_CONTEXTUAL_TEMPLATE,
        "relationship_multi_type": RELATIONSHIP_MULTI_TYPE_TEMPLATE,
        "relationship_few_shot_metabolomics": RELATIONSHIP_FEW_SHOT_METABOLOMICS_TEMPLATE,
        "relationship_few_shot_hierarchical": RELATIONSHIP_FEW_SHOT_HIERARCHICAL_TEMPLATE,
        "relationship_few_shot_contextual": RELATIONSHIP_FEW_SHOT_CONTEXTUAL_TEMPLATE
    }
    
    if template_type not in templates:
        raise TemplateNotFoundError(f"Relationship template '{template_type}' not found. Available: {list(templates.keys())}")
    
    return templates[template_type]


def get_relationship_template_metadata(template_type: str) -> Dict[str, Any]:
    """
    Get metadata for relationship extraction templates.
    
    Args:
        template_type: Type of template
        
    Returns:
        Dictionary with template metadata
    """
    metadata = {
        "basic": {
            "description": "Basic relationship extraction template",
            "use_cases": ["general relationship extraction", "simple texts"],
            "strengths": ["easy to use", "fast processing"],
            "limitations": ["may miss complex relationships"]
        },
        "detailed": {
            "description": "Detailed relationship extraction with comprehensive guidelines",
            "use_cases": ["complex texts", "high-precision extraction"],
            "strengths": ["detailed instructions", "high precision"],
            "limitations": ["may be slower", "more verbose"]
        },
        "scientific": {
            "description": "Scientific literature relationship extraction",
            "use_cases": ["research papers", "scientific articles"],
            "strengths": ["domain expertise", "high precision"],
            "limitations": ["specialized for scientific text"]
        },
        "metabolomics": {
            "description": "Plant metabolomics relationship extraction",
            "use_cases": ["metabolomics papers", "plant biology"],
            "strengths": ["domain-specific", "pathway-aware"],
            "limitations": ["specialized domain only"]
        },
        "hierarchical": {
            "description": "Hierarchical relationship extraction with specificity differentiation",
            "use_cases": ["complex biological texts", "pathway analysis"],
            "strengths": ["specific over broad relationships", "clear decision tree"],
            "limitations": ["requires domain knowledge"]
        },
        "contextual": {
            "description": "Context-aware relationship extraction with conditional understanding",
            "use_cases": ["complex experimental texts", "multi-condition studies"],
            "strengths": ["captures context dependencies", "temporal/spatial awareness"],
            "limitations": ["may extract more complex structures"]
        },
        "multi_type": {
            "description": "Comprehensive multi-type relationship extraction",
            "use_cases": ["pathway reconstruction", "comprehensive analysis"],
            "strengths": ["complete coverage", "cross-validation"],
            "limitations": ["may be slower", "more complex output"]
        },
        "few_shot_metabolomics": {
            "description": "Few-shot metabolomics relationship extraction with examples",
            "use_cases": ["metabolomics literature", "biosynthetic pathways"],
            "strengths": ["learns from examples", "domain-specific patterns"],
            "limitations": ["requires compatible text types"]
        },
        "few_shot_hierarchical": {
            "description": "Few-shot hierarchical relationship extraction with specificity examples",
            "use_cases": ["complex biological relationships", "regulatory networks"],
            "strengths": ["learns specificity patterns", "clear examples"],
            "limitations": ["specific to biological domains"]
        },
        "few_shot_contextual": {
            "description": "Few-shot contextual relationship extraction with conditional examples",
            "use_cases": ["experimental studies", "condition-dependent relationships"],
            "strengths": ["context-aware learning", "multi-factor analysis"],
            "limitations": ["complex output structure"]
        }
    }
    
    return metadata.get(template_type, {})


# Template management and utility functions for relationship extraction

def generate_relationship_examples(relationship_type: str, entity_types: List[str], count: int = 3) -> List[Dict[str, Any]]:
    """
    Generate synthetic examples for relationship extraction templates.
    
    Args:
        relationship_type: Type of relationship to generate examples for
        entity_types: List of entity types involved in relationships  
        count: Number of examples to generate
        
    Returns:
        List of example relationship dictionaries
    """
    # Domain-specific example patterns for different relationship types
    relationship_examples = {
        "synthesized_by": [
            {
                "text": "Anthocyanins are synthesized by anthocyanidin synthase in grape berries",
                "subject": ("anthocyanins", "METABOLITE"),
                "object": ("anthocyanidin synthase", "ENZYME"),
                "evidence": "anthocyanins are synthesized by anthocyanidin synthase"
            },
            {
                "text": "Resveratrol is produced by stilbene synthase under UV stress",
                "subject": ("resveratrol", "PHENOLIC_COMPOUND"),  
                "object": ("stilbene synthase", "ENZYME"),
                "evidence": "resveratrol is produced by stilbene synthase"
            },
            {
                "text": "Caffeine synthesis is catalyzed by N-methyltransferase in coffee plants",
                "subject": ("caffeine", "ALKALOID"),
                "object": ("N-methyltransferase", "ENZYME"), 
                "evidence": "caffeine synthesis is catalyzed by N-methyltransferase"
            }
        ],
        "accumulates_in": [
            {
                "text": "Proline accumulates in root tissues during drought stress",
                "subject": ("proline", "AMINO_ACID"),
                "object": ("root tissues", "PLANT_TISSUE"),
                "evidence": "proline accumulates in root tissues"
            },
            {
                "text": "Starch granules concentrate in tuber parenchyma cells",
                "subject": ("starch granules", "CARBOHYDRATE"),
                "object": ("tuber parenchyma cells", "CELL_TYPE"),
                "evidence": "starch granules concentrate in tuber parenchyma cells"
            },
            {
                "text": "Essential oils are stored in secretory cavities of citrus peel",
                "subject": ("essential oils", "TERPENOID"),
                "object": ("secretory cavities", "CELLULAR_COMPONENT"),
                "evidence": "essential oils are stored in secretory cavities"
            }
        ],
        "upregulates": [
            {
                "text": "ABA upregulates stress-responsive genes during water deficit",
                "subject": ("ABA", "HORMONE"),
                "object": ("stress-responsive genes", "GENE"),
                "evidence": "ABA upregulates stress-responsive genes"
            },
            {
                "text": "Cold treatment enhances anthocyanin biosynthesis pathway",
                "subject": ("cold treatment", "STRESS_CONDITION"),
                "object": ("anthocyanin biosynthesis", "PATHWAY"),
                "evidence": "cold treatment enhances anthocyanin biosynthesis pathway"
            },
            {
                "text": "MYB transcription factors activate flavonoid gene expression",
                "subject": ("MYB transcription factors", "TRANSCRIPTION_FACTOR"),
                "object": ("flavonoid genes", "GENE"),
                "evidence": "MYB transcription factors activate flavonoid gene expression"
            }
        ],
        "derived_from": [
            {
                "text": "Quercetin is derived from the precursor naringenin via enzyme action",
                "subject": ("quercetin", "FLAVONOID"),
                "object": ("naringenin", "FLAVONOID"),
                "evidence": "quercetin is derived from the precursor naringenin"
            },
            {  
                "text": "Lignin monomers originate from phenylalanine through phenylpropanoid pathway",
                "subject": ("lignin monomers", "PHENOLIC_COMPOUND"),
                "object": ("phenylalanine", "AMINO_ACID"),
                "evidence": "lignin monomers originate from phenylalanine"
            },
            {
                "text": "Terpenoids are biosynthesized from isoprene units in specialized pathways",
                "subject": ("terpenoids", "TERPENOID"),
                "object": ("isoprene units", "COMPOUND"),
                "evidence": "terpenoids are biosynthesized from isoprene units"
            }
        ],
        "increases_under": [
            {
                "text": "Proline levels increase significantly under drought stress conditions", 
                "subject": ("proline", "AMINO_ACID"),
                "object": ("drought stress", "STRESS_CONDITION"),
                "evidence": "proline levels increase significantly under drought stress"
            },
            {
                "text": "Anthocyanin content rises during cold acclimation in leaves",
                "subject": ("anthocyanin", "METABOLITE"), 
                "object": ("cold acclimation", "STRESS_CONDITION"),
                "evidence": "anthocyanin content rises during cold acclimation"
            },
            {
                "text": "Sugar accumulation peaks under water deficit in storage organs",
                "subject": ("sugar", "CARBOHYDRATE"),
                "object": ("water deficit", "STRESS_CONDITION"), 
                "evidence": "sugar accumulation peaks under water deficit"
            }
        ]
    }
    
    # Get examples for the specific relationship type
    examples = relationship_examples.get(relationship_type, [])
    
    # Return requested number of examples (cycling if needed)
    result = []
    for i in range(min(count, len(examples))):
        example = examples[i]
        result.append({
            "subject_entity": {"text": example["subject"][0], "label": example["subject"][1]},
            "relation_type": relationship_type,
            "object_entity": {"text": example["object"][0], "label": example["object"][1]},
            "confidence": 0.90 + (i * 0.02),  # Slight variation in confidence
            "evidence": example["evidence"],
            "example_text": example["text"]
        })
    
    return result


def format_relationship_schema_for_template(schema_dict: Dict[str, Any]) -> str:
    """
    Format relationship schema dictionary for use in templates.
    
    Args:
        schema_dict: Dictionary containing relationship schema definitions
        
    Returns:
        Formatted string representation of schema for template use
    """
    formatted_lines = []
    
    for relation_type, pattern in schema_dict.items():
        if hasattr(pattern, 'description'):
            # Handle RelationshipPattern objects
            description = pattern.description
            domain = list(pattern.domain) if pattern.domain else []
            range_types = list(pattern.range) if pattern.range else []
            examples = pattern.examples if pattern.examples else []
        else:
            # Handle dictionary format
            description = pattern.get('description', '')
            domain = pattern.get('domain', [])
            range_types = pattern.get('range', [])
            examples = pattern.get('examples', [])
        
        # Format the relationship entry
        formatted_lines.append(f"- {relation_type}: {description}")
        
        if domain:
            formatted_lines.append(f"  Domain: {', '.join(domain[:5])}{'...' if len(domain) > 5 else ''}")
        
        if range_types:
            formatted_lines.append(f"  Range: {', '.join(range_types[:5])}{'...' if len(range_types) > 5 else ''}")
            
        if examples:
            example_str = ', '.join([f'"{ex[0]} → {ex[1]}"' for ex in examples[:2]])
            formatted_lines.append(f"  Examples: {example_str}")
        
        formatted_lines.append("")  # Empty line between relationships
    
    return "\n".join(formatted_lines)


def get_compatible_relationships_for_entities(entity_types: List[str], schema_dict: Dict[str, Any]) -> List[str]:
    """
    Get list of relationship types compatible with given entity types.
    
    Args:
        entity_types: List of entity type labels
        schema_dict: Relationship schema dictionary
        
    Returns:
        List of compatible relationship types
    """
    compatible_relations = []
    entity_set = set(entity_types)
    
    for relation_type, pattern in schema_dict.items():
        if hasattr(pattern, 'domain') and hasattr(pattern, 'range'):
            # Handle RelationshipPattern objects
            domain = pattern.domain
            range_types = pattern.range
        else:
            # Handle dictionary format
            domain = set(pattern.get('domain', []))
            range_types = set(pattern.get('range', []))
        
        # Check if any entity types match domain or range
        if domain.intersection(entity_set) and range_types.intersection(entity_set):
            compatible_relations.append(relation_type)
    
    return compatible_relations


def select_optimal_relationship_template(text_characteristics: Dict[str, Any]) -> str:
    """
    Select the most appropriate relationship template based on text characteristics.
    
    Args:
        text_characteristics: Dictionary with text analysis results including:
            - complexity: low/medium/high
            - domain: metabolomics/genetics/general
            - context_dependency: boolean
            - relationship_density: low/medium/high
            - temporal_markers: boolean
            - conditional_statements: boolean
            
    Returns:
        Recommended template type name
    """
    complexity = text_characteristics.get('complexity', 'medium')
    domain = text_characteristics.get('domain', 'general')
    context_dependent = text_characteristics.get('context_dependency', False)
    temporal_markers = text_characteristics.get('temporal_markers', False)
    conditional_statements = text_characteristics.get('conditional_statements', False)
    relationship_density = text_characteristics.get('relationship_density', 'medium')
    
    # Decision logic for template selection
    if context_dependent or temporal_markers or conditional_statements:
        if domain == 'metabolomics':
            return 'few_shot_contextual'
        else:
            return 'contextual'
    
    elif relationship_density == 'high' or complexity == 'high':
        return 'multi_type'
    
    elif domain == 'metabolomics':
        if complexity == 'high':
            return 'few_shot_metabolomics'
        else:
            return 'metabolomics'
    
    elif complexity == 'high':
        return 'hierarchical'
    
    elif complexity == 'low':
        return 'basic'
    
    else:
        return 'detailed'


def validate_relationship_template_inputs(template: str, text: str, entities: List[Dict], schema: Dict) -> Dict[str, Any]:
    """
    Validate inputs for relationship extraction templates.
    
    Args:
        template: Template string with placeholders
        text: Input text to analyze
        entities: List of entity dictionaries
        schema: Relationship schema dictionary
        
    Returns:
        Dictionary with validation results and recommendations
    """
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }
    
    # Check text length
    if len(text.strip()) < 10:
        validation_results['errors'].append("Text too short for meaningful relationship extraction")
        validation_results['valid'] = False
    elif len(text) > 5000:
        validation_results['warnings'].append("Text is very long, consider splitting into smaller chunks")
    
    # Check entities
    if not entities:
        validation_results['errors'].append("No entities provided for relationship extraction")
        validation_results['valid'] = False
    elif len(entities) < 2:
        validation_results['warnings'].append("Only one entity found, relationships require at least two entities")
    
    # Validate entity format
    for i, entity in enumerate(entities):
        if not isinstance(entity, dict) or 'text' not in entity or 'label' not in entity:
            validation_results['errors'].append(f"Entity {i} missing required 'text' or 'label' fields")
            validation_results['valid'] = False
    
    # Check schema
    if not schema:
        validation_results['warnings'].append("No relationship schema provided, using default relationships")
    
    # Check template placeholders
    required_placeholders = ['{text}', '{entities}', '{schema}']
    for placeholder in required_placeholders:
        if placeholder not in template:
            validation_results['errors'].append(f"Template missing required placeholder: {placeholder}")
            validation_results['valid'] = False
    
    # Performance recommendations
    if len(entities) > 20:
        validation_results['recommendations'].append("Large number of entities may impact performance, consider entity filtering")
    
    if len(schema) > 50:
        validation_results['recommendations'].append("Large schema may impact performance, consider schema filtering")
    
    return validation_results


def list_available_relationship_templates() -> List[Dict[str, str]]:
    """
    List all available relationship extraction templates with descriptions.
    
    Returns:
        List of dictionaries with template information
    """
    templates = [
        {
            "name": "basic",
            "type": "zero-shot",
            "description": "Basic relationship extraction with simple instructions",
            "use_case": "General texts, simple relationships"
        },
        {
            "name": "detailed", 
            "type": "zero-shot",
            "description": "Detailed relationship extraction with comprehensive guidelines",
            "use_case": "Complex texts requiring precision"
        },
        {
            "name": "scientific",
            "type": "zero-shot", 
            "description": "Scientific literature relationship extraction with domain expertise",
            "use_case": "Research papers, scientific articles"
        },
        {
            "name": "metabolomics",
            "type": "zero-shot",
            "description": "Plant metabolomics relationship extraction",
            "use_case": "Metabolomics papers, plant biology"
        },
        {
            "name": "hierarchical",
            "type": "zero-shot",
            "description": "Hierarchical differentiation between broad and specific relationships", 
            "use_case": "Complex biological texts requiring relationship specificity"
        },
        {
            "name": "contextual",
            "type": "zero-shot",
            "description": "Context-aware relationship extraction with conditional understanding",
            "use_case": "Experimental texts with environmental/temporal conditions"
        },
        {
            "name": "multi_type",
            "type": "zero-shot",
            "description": "Comprehensive multi-type relationship extraction with cross-validation",
            "use_case": "Pathway reconstruction, comprehensive analysis"
        },
        {
            "name": "few_shot_metabolomics",
            "type": "few-shot",
            "description": "Few-shot metabolomics relationship extraction with domain examples",
            "use_case": "Metabolomics literature, biosynthetic pathways"
        },
        {
            "name": "few_shot_hierarchical", 
            "type": "few-shot",
            "description": "Few-shot hierarchical relationship extraction with specificity examples",
            "use_case": "Complex biological relationships, regulatory networks"
        },
        {
            "name": "few_shot_contextual",
            "type": "few-shot", 
            "description": "Few-shot contextual relationship extraction with conditional examples",
            "use_case": "Experimental studies, condition-dependent relationships"
        }
    ]
    
    return templates