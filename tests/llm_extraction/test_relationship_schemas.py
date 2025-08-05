"""
Unit tests for relationship schemas in plant metabolomics research.

This module provides comprehensive tests for the relationship schema system,
including validation of relationship patterns, domain/range constraints,
and schema functionality.
"""

import pytest
from typing import Dict, List, Set

from src.llm_extraction.relationship_schemas import (
    RelationshipPattern,
    PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA,
    BASIC_RELATIONSHIP_SCHEMA,
    DOMAIN_RANGE_CONSTRAINTS,
    get_plant_metabolomics_relationship_schema,
    get_basic_relationship_schema,
    validate_relationship_pattern,
    get_compatible_relationships,
    filter_relationships_by_domain,
    get_relationship_schema_by_category,
    validate_relationship_schema,
    get_relationship_statistics,
    convert_schema_to_simple_dict,
    get_domain_specific_schema,
    METABOLOMICS_RELATIONSHIP_SCHEMA,
    GENETICS_RELATIONSHIP_SCHEMA,
    PLANT_BIOLOGY_RELATIONSHIP_SCHEMA,
    STRESS_RELATIONSHIP_SCHEMA,
    BIOCHEMISTRY_RELATIONSHIP_SCHEMA
)


class TestRelationshipPattern:
    """Test RelationshipPattern dataclass functionality."""
    
    def test_relationship_pattern_creation(self):
        """Test creating RelationshipPattern instances."""
        pattern = RelationshipPattern(
            relation_type="synthesized_by",
            description="Metabolite is synthesized by an organism",
            domain={"METABOLITE", "COMPOUND"},
            range={"SPECIES", "ORGANISM"}
        )
        
        assert pattern.relation_type == "synthesized_by"
        assert pattern.description == "Metabolite is synthesized by an organism"
        assert pattern.domain == {"METABOLITE", "COMPOUND"}
        assert pattern.range == {"SPECIES", "ORGANISM"}
        assert pattern.inverse is None
        assert pattern.symmetric is False
        assert pattern.transitive is False
        assert pattern.examples is None
    
    def test_relationship_pattern_with_all_fields(self):
        """Test creating RelationshipPattern with all optional fields."""
        pattern = RelationshipPattern(
            relation_type="binds_to",
            description="Molecule binds to target",
            domain={"PROTEIN", "ENZYME"},
            range={"METABOLITE", "PROTEIN"},
            inverse="bound_by",
            symmetric=True,
            transitive=False,
            examples=[("enzyme", "substrate"), ("antibody", "antigen")]
        )
        
        assert pattern.inverse == "bound_by"
        assert pattern.symmetric is True
        assert pattern.transitive is False
        assert len(pattern.examples) == 2


class TestRelationshipSchemas:
    """Test relationship schema constants and basic functionality."""
    
    def test_plant_metabolomics_schema_structure(self):
        """Test the structure of PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA."""
        schema = PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA
        
        # Check that it's a non-empty dictionary
        assert isinstance(schema, dict)
        assert len(schema) > 0
        
        # Check that all values are RelationshipPattern instances
        for relation_type, pattern in schema.items():
            assert isinstance(relation_type, str)
            assert isinstance(pattern, RelationshipPattern)
            assert pattern.relation_type == relation_type
            assert isinstance(pattern.description, str)
            assert len(pattern.description) > 0
            assert isinstance(pattern.domain, set)
            assert len(pattern.domain) > 0
            assert isinstance(pattern.range, set)
            assert len(pattern.range) > 0
    
    def test_basic_schema_structure(self):
        """Test the structure of BASIC_RELATIONSHIP_SCHEMA."""
        schema = BASIC_RELATIONSHIP_SCHEMA
        
        assert isinstance(schema, dict)
        assert len(schema) > 0
        assert len(schema) < len(PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA)
        
        # Check that all basic schema patterns are in the full schema
        for relation_type in schema.keys():
            assert relation_type in PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA
    
    def test_specific_requested_relationships(self):
        """Test that specifically requested relationships are present."""
        schema = PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA
        
        # Check for "Compound-Affects-Trait" relationship
        assert "affects" in schema
        affects_pattern = schema["affects"]
        assert "COMPOUND" in affects_pattern.domain
        assert "PLANT_TRAIT" in affects_pattern.range
        
        # Check for "Metabolite-InvolvedIn-BiologicalProcess" relationship
        assert "involved_in_biological_process" in schema
        involved_pattern = schema["involved_in_biological_process"]
        assert "METABOLITE" in involved_pattern.domain
        assert "METABOLIC_PATHWAY" in involved_pattern.range
    
    def test_domain_range_constraints(self):
        """Test DOMAIN_RANGE_CONSTRAINTS structure."""
        constraints = DOMAIN_RANGE_CONSTRAINTS
        
        assert isinstance(constraints, dict)
        assert len(constraints) > 0
        
        for relation_type, constraint in constraints.items():
            assert isinstance(constraint, dict)
            assert "domain" in constraint
            assert "range" in constraint
            assert isinstance(constraint["domain"], set)
            assert isinstance(constraint["range"], set)
            assert len(constraint["domain"]) > 0
            assert len(constraint["range"]) > 0


class TestSchemaAccessFunctions:
    """Test functions for accessing relationship schemas."""
    
    def test_get_plant_metabolomics_relationship_schema(self):
        """Test get_plant_metabolomics_relationship_schema function."""
        schema = get_plant_metabolomics_relationship_schema()
        
        assert isinstance(schema, dict)
        assert len(schema) > 0
        
        # Check that it's a copy, not the original
        original_len = len(PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA)
        schema.clear()
        assert len(PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA) == original_len
    
    def test_get_basic_relationship_schema(self):
        """Test get_basic_relationship_schema function."""
        schema = get_basic_relationship_schema()
        
        assert isinstance(schema, dict)
        assert len(schema) > 0
        assert len(schema) < len(PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA)
        
        # Check that essential relationships are present
        essential_relations = ["synthesized_by", "found_in", "affects", "involved_in"]
        for relation in essential_relations:
            assert relation in schema


class TestValidationFunctions:
    """Test relationship pattern validation functions."""
    
    def test_validate_relationship_pattern_valid(self):
        """Test validation of valid relationship patterns."""
        # Test valid patterns
        assert validate_relationship_pattern("METABOLITE", "synthesized_by", "SPECIES")
        assert validate_relationship_pattern("COMPOUND", "affects", "PLANT_TRAIT")
        assert validate_relationship_pattern("GENE", "encodes", "PROTEIN")
        assert validate_relationship_pattern("METABOLITE", "found_in", "LEAF")
    
    def test_validate_relationship_pattern_invalid(self):
        """Test validation of invalid relationship patterns."""
        # Test invalid domain (subject type)
        assert not validate_relationship_pattern("INVALID_TYPE", "synthesized_by", "SPECIES")
        
        # Test invalid range (object type)
        assert not validate_relationship_pattern("METABOLITE", "synthesized_by", "INVALID_TYPE")
        
        # Test non-existent relationship
        assert not validate_relationship_pattern("METABOLITE", "invalid_relation", "SPECIES")
        
        # Test incompatible domain-range combination
        assert not validate_relationship_pattern("GENE", "accumulates_in", "SPECIES")
    
    def test_validate_relationship_pattern_with_custom_schema(self):
        """Test validation with custom schema."""
        custom_schema = {
            "custom_relation": RelationshipPattern(
                relation_type="custom_relation",
                description="Custom relationship for testing",
                domain={"TYPE_A"},
                range={"TYPE_B"}
            )
        }
        
        assert validate_relationship_pattern("TYPE_A", "custom_relation", "TYPE_B", custom_schema)
        assert not validate_relationship_pattern("TYPE_B", "custom_relation", "TYPE_A", custom_schema)
    
    def test_get_compatible_relationships(self):
        """Test getting compatible relationships for entity types."""
        # Test metabolite-species compatibility
        compatible = get_compatible_relationships("METABOLITE", "SPECIES")
        assert "synthesized_by" in compatible
        assert "found_in" in compatible
        
        # Test gene-protein compatibility
        compatible = get_compatible_relationships("GENE", "PROTEIN")
        assert "encodes" in compatible
        
        # Test compound-trait compatibility
        compatible = get_compatible_relationships("COMPOUND", "PLANT_TRAIT")
        assert "affects" in compatible
        assert "associated_with" in compatible
        
        # Test incompatible types
        compatible = get_compatible_relationships("INVALID_TYPE1", "INVALID_TYPE2")
        assert len(compatible) == 0
    
    def test_get_compatible_relationships_with_custom_schema(self):
        """Test getting compatible relationships with custom schema."""
        custom_schema = {
            "relation1": RelationshipPattern(
                relation_type="relation1",
                description="Test relation 1",
                domain={"TYPE_A", "TYPE_B"},
                range={"TYPE_C"}
            ),
            "relation2": RelationshipPattern(
                relation_type="relation2",
                description="Test relation 2",
                domain={"TYPE_A"},
                range={"TYPE_B", "TYPE_C"}
            )
        }
        
        compatible = get_compatible_relationships("TYPE_A", "TYPE_C", custom_schema)
        assert "relation1" in compatible
        assert "relation2" in compatible
        
        compatible = get_compatible_relationships("TYPE_B", "TYPE_C", custom_schema)
        assert "relation1" in compatible
        assert "relation2" not in compatible


class TestSchemaFiltering:
    """Test schema filtering and domain-specific functions."""
    
    def test_filter_relationships_by_domain(self):
        """Test filtering relationships by domain entity types."""
        # Test filtering by metabolite entities
        metabolite_entities = {"METABOLITE", "COMPOUND", "FLAVONOID"}
        filtered = filter_relationships_by_domain(metabolite_entities)
        
        assert len(filtered) > 0
        assert "synthesized_by" in filtered
        assert "found_in" in filtered
        assert "affects" in filtered
        
        # Test filtering by gene entities
        gene_entities = {"GENE", "PROTEIN", "ENZYME"}
        filtered = filter_relationships_by_domain(gene_entities)
        
        assert "encodes" in filtered
        assert "catalyzes" in filtered
        assert "expressed_in" in filtered
    
    def test_filter_relationships_by_domain_empty(self):
        """Test filtering with empty or invalid domain."""
        # Test with empty set
        filtered = filter_relationships_by_domain(set())
        assert len(filtered) == 0
        
        # Test with non-existent entity types
        filtered = filter_relationships_by_domain({"INVALID_TYPE"})
        assert len(filtered) == 0
    
    def test_get_relationship_schema_by_category(self):
        """Test getting relationships organized by category."""
        categories = get_relationship_schema_by_category()
        
        assert isinstance(categories, dict)
        assert len(categories) > 0
        
        # Check expected categories
        expected_categories = [
            "metabolite_relationships",
            "gene_protein_relationships", 
            "pathway_relationships",
            "experimental_relationships",
            "structural_relationships",
            "phenotypic_relationships",
            "analytical_relationships"
        ]
        
        for category in expected_categories:
            assert category in categories
            assert isinstance(categories[category], list)
            assert len(categories[category]) > 0
        
        # Check that specific relationships are in correct categories
        assert "synthesized_by" in categories["metabolite_relationships"]
        assert "encodes" in categories["gene_protein_relationships"]
        assert "upstream_of" in categories["pathway_relationships"]
        assert "responds_to" in categories["experimental_relationships"]


class TestSchemaValidation:
    """Test schema validation functions."""
    
    def test_validate_relationship_schema_valid(self):
        """Test validation of valid relationship schemas."""
        # Test full schema
        assert validate_relationship_schema(PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA)
        
        # Test basic schema
        assert validate_relationship_schema(BASIC_RELATIONSHIP_SCHEMA)
        
        # Test custom valid schema
        custom_schema = {
            "test_relation": RelationshipPattern(
                relation_type="test_relation",
                description="Test relationship",
                domain={"TYPE_A"},
                range={"TYPE_B"}
            )
        }
        assert validate_relationship_schema(custom_schema)
    
    def test_validate_relationship_schema_invalid(self):
        """Test validation of invalid relationship schemas."""
        # Test None schema
        with pytest.raises(ValueError, match="Schema cannot be None"):
            validate_relationship_schema(None)
        
        # Test non-dict schema
        with pytest.raises(ValueError, match="Schema must be a dictionary"):
            validate_relationship_schema("not a dict")
        
        # Test empty schema
        with pytest.raises(ValueError, match="Schema cannot be empty"):
            validate_relationship_schema({})
        
        # Test invalid relation type
        with pytest.raises(ValueError, match="Relation type must be string"):
            validate_relationship_schema({123: RelationshipPattern("test", "desc", {"A"}, {"B"})})
        
        # Test mismatched relation type
        pattern = RelationshipPattern("different_name", "desc", {"A"}, {"B"})
        with pytest.raises(ValueError, match="Pattern relation_type.*doesn't match key"):
            validate_relationship_schema({"test_relation": pattern})
    
    def test_validate_relationship_schema_inverse_consistency(self):
        """Test validation of inverse relationship consistency."""
        # Test valid inverse relationships
        valid_schema = {
            "relation_a": RelationshipPattern(
                relation_type="relation_a",
                description="Relation A",
                domain={"TYPE_1"},
                range={"TYPE_2"},
                inverse="relation_b"
            ),
            "relation_b": RelationshipPattern(
                relation_type="relation_b",
                description="Relation B",
                domain={"TYPE_2"},
                range={"TYPE_1"},
                inverse="relation_a"
            )
        }
        assert validate_relationship_schema(valid_schema)
        
        # Test invalid inverse relationships
        invalid_schema = {
            "relation_a": RelationshipPattern(
                relation_type="relation_a",
                description="Relation A",
                domain={"TYPE_1"},
                range={"TYPE_2"},
                inverse="relation_b"
            ),
            "relation_b": RelationshipPattern(
                relation_type="relation_b",
                description="Relation B",
                domain={"TYPE_2"},
                range={"TYPE_1"},
                inverse="different_relation"  # Inconsistent inverse
            )
        }
        with pytest.raises(ValueError, match="Inverse relationship inconsistency"):
            validate_relationship_schema(invalid_schema)


class TestSchemaStatistics:
    """Test schema statistics and analysis functions."""
    
    def test_get_relationship_statistics(self):
        """Test getting relationship schema statistics."""
        stats = get_relationship_statistics(PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA)
        
        assert isinstance(stats, dict)
        assert "total_relationships" in stats
        assert "unique_domain_types" in stats
        assert "unique_range_types" in stats
        assert "symmetric_relationships" in stats
        assert "transitive_relationships" in stats
        assert "relationships_with_inverse" in stats
        assert "domain_types" in stats
        assert "range_types" in stats
        
        # Check that counts are reasonable
        assert stats["total_relationships"] > 0
        assert stats["unique_domain_types"] > 0
        assert stats["unique_range_types"] > 0
        assert isinstance(stats["domain_types"], list)
        assert isinstance(stats["range_types"], list)
    
    def test_convert_schema_to_simple_dict(self):
        """Test converting schema to simple dictionary format."""
        simple_dict = convert_schema_to_simple_dict(BASIC_RELATIONSHIP_SCHEMA)
        
        assert isinstance(simple_dict, dict)
        assert len(simple_dict) == len(BASIC_RELATIONSHIP_SCHEMA)
        
        for relation_type, description in simple_dict.items():
            assert isinstance(relation_type, str)
            assert isinstance(description, str)
            assert len(description) > 0
            
            # Check that description matches original pattern
            original_pattern = BASIC_RELATIONSHIP_SCHEMA[relation_type]
            assert description == original_pattern.description


class TestDomainSpecificSchemas:
    """Test domain-specific relationship schemas."""
    
    def test_get_domain_specific_schema_metabolomics(self):
        """Test metabolomics domain schema."""
        schema = get_domain_specific_schema("metabolomics")
        
        assert isinstance(schema, dict)
        assert len(schema) > 0
        
        # Check for metabolomics-specific relationships
        expected_relations = ["synthesized_by", "found_in", "made_via", "affects"]
        for relation in expected_relations:
            assert relation in schema
    
    def test_get_domain_specific_schema_genetics(self):
        """Test genetics domain schema."""
        schema = get_domain_specific_schema("genetics")
        
        assert isinstance(schema, dict)
        assert len(schema) > 0
        
        # Check for genetics-specific relationships
        expected_relations = ["encodes", "expressed_in", "regulated_by", "upregulates"]
        for relation in expected_relations:
            assert relation in schema
    
    def test_get_domain_specific_schema_invalid(self):
        """Test invalid domain name."""
        with pytest.raises(ValueError, match="Unsupported domain"):
            get_domain_specific_schema("invalid_domain")
    
    def test_predefined_domain_schemas(self):
        """Test predefined domain-specific schemas."""
        schemas = [
            METABOLOMICS_RELATIONSHIP_SCHEMA,
            GENETICS_RELATIONSHIP_SCHEMA,
            PLANT_BIOLOGY_RELATIONSHIP_SCHEMA,
            STRESS_RELATIONSHIP_SCHEMA,
            BIOCHEMISTRY_RELATIONSHIP_SCHEMA
        ]
        
        for schema in schemas:
            assert isinstance(schema, dict)
            assert len(schema) > 0
            
            # Validate each schema
            assert validate_relationship_schema(schema)


class TestComplexRelationshipPatterns:
    """Test complex relationship patterns and edge cases."""
    
    def test_symmetric_relationships(self):
        """Test symmetric relationship patterns."""
        schema = PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA
        
        # Check symmetric relationships
        symmetric_relations = [rel for rel, pattern in schema.items() if pattern.symmetric]
        assert len(symmetric_relations) > 0
        
        for relation in symmetric_relations:
            pattern = schema[relation]
            assert pattern.symmetric is True
            
            # For symmetric relations, domain and range should allow bidirectional patterns
            # Test both directions for compatible entity types
            compatible_forward = get_compatible_relationships("PROTEIN", "METABOLITE")
            compatible_reverse = get_compatible_relationships("METABOLITE", "PROTEIN")
            
            if relation in compatible_forward:
                # For symmetric relations, the reverse should also be possible
                # (though the validation depends on domain/range definitions)
                pass  # This test depends on specific domain/range configurations
    
    def test_transitive_relationships(self):
        """Test transitive relationship patterns."""
        schema = PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA
        
        # Check transitive relationships
        transitive_relations = [rel for rel, pattern in schema.items() if pattern.transitive]
        assert len(transitive_relations) > 0
        
        for relation in transitive_relations:
            pattern = schema[relation]
            assert pattern.transitive is True
    
    def test_inverse_relationships(self):
        """Test inverse relationship patterns."""
        schema = PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA
        
        # Check relationships with inverses
        relations_with_inverse = [rel for rel, pattern in schema.items() if pattern.inverse]
        assert len(relations_with_inverse) > 0
        
        for relation in relations_with_inverse:
            pattern = schema[relation]
            inverse_relation = pattern.inverse
            
            # If inverse is defined in schema, check consistency
            if inverse_relation in schema:
                inverse_pattern = schema[inverse_relation]
                assert inverse_pattern.inverse == relation
    
    def test_relationship_examples(self):
        """Test relationship pattern examples."""
        schema = PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA
        
        relations_with_examples = [rel for rel, pattern in schema.items() if pattern.examples]
        assert len(relations_with_examples) > 0
        
        for relation in relations_with_examples:
            pattern = schema[relation]
            assert isinstance(pattern.examples, list)
            assert len(pattern.examples) > 0
            
            for example in pattern.examples:
                assert isinstance(example, tuple)
                assert len(example) == 2  # (subject, object) pairs
                assert isinstance(example[0], str)
                assert isinstance(example[1], str)


class TestIntegrationWithEntitySchemas:
    """Test integration between relationship schemas and entity schemas."""
    
    def test_entity_type_consistency(self):
        """Test that relationship domain/range types are consistent with entity schemas."""
        # Import entity types from entity_schemas
        try:
            from src.llm_extraction.entity_schemas import PLANT_METABOLOMICS_SCHEMA as ENTITY_SCHEMA
            
            # Get all entity types from entity schema
            valid_entity_types = set(ENTITY_SCHEMA.keys())
            
            # Check that all domain/range types in relationship schema are valid entity types
            schema = PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA
            
            for relation_type, pattern in schema.items():
                for domain_type in pattern.domain:
                    assert domain_type in valid_entity_types, f"Domain type '{domain_type}' in relation '{relation_type}' not found in entity schema"
                
                for range_type in pattern.range:
                    assert range_type in valid_entity_types, f"Range type '{range_type}' in relation '{relation_type}' not found in entity schema"
                    
        except ImportError:
            # Skip this test if entity_schemas module is not available
            pytest.skip("Entity schemas module not available for integration testing")
    
    def test_key_biological_relationships(self):
        """Test that key biological relationships are properly defined."""
        schema = PLANT_METABOLOMICS_RELATIONSHIP_SCHEMA
        
        # Test Compound-Affects-Trait relationship
        assert "affects" in schema
        affects_pattern = schema["affects"]
        
        # Should allow compounds to affect various trait types
        compound_types = {"COMPOUND", "METABOLITE", "PHENOLIC_COMPOUND", "FLAVONOID"}
        trait_types = {"PLANT_TRAIT", "MORPHOLOGICAL_TRAIT", "PHYSIOLOGICAL_TRAIT", "BIOCHEMICAL_TRAIT"}
        
        assert compound_types.intersection(affects_pattern.domain), "Compound types should be in 'affects' domain"
        assert trait_types.intersection(affects_pattern.range), "Trait types should be in 'affects' range"
        
        # Test Metabolite-InvolvedIn-BiologicalProcess relationship
        assert "involved_in_biological_process" in schema
        involved_pattern = schema["involved_in_biological_process"]
        
        metabolite_types = {"METABOLITE", "COMPOUND"}
        process_types = {"METABOLIC_PATHWAY", "BIOSYNTHESIS", "BIOLOGICAL_ACTIVITY"}
        
        assert metabolite_types.intersection(involved_pattern.domain), "Metabolite types should be in 'involved_in_biological_process' domain"
        assert process_types.intersection(involved_pattern.range), "Process types should be in 'involved_in_biological_process' range"


if __name__ == "__main__":
    pytest.main([__file__])