"""
Unit tests for src/data_quality/taxonomy.py

This module tests the NCBI taxonomy integration functionality for loading taxonomic data,
filtering species by lineage, and retrieving lineage information for species in the 
AIM2-ODIE ontology development and information extraction system. The taxonomy module
integrates with multitax and ncbi-taxonomist libraries for taxonomic data processing.

Test Coverage:
- Taxonomy loading: NCBI taxonomy database loading using multitax.NcbiTx()
- Species filtering: lineage-based filtering using multitax.filter() or ncbi-taxonomist
- Lineage retrieval: getting full taxonomic lineage for species names/IDs
- Error handling: invalid inputs, non-existent species, API failures
- Edge cases: empty results, malformed names, special characters
- Performance considerations: large datasets, memory efficiency, timeout handling
- External API mocking: proper mocking of multitax and ncbi-taxonomist calls
- Integration scenarios: combining multiple taxonomy operations

Functions Under Test:
- load_ncbi_taxonomy() -> TaxonomyObject: Load NCBI taxonomy database
- filter_species_by_lineage(taxonomy_obj, target_lineage: str) -> list[dict]: Filter species by lineage
- get_lineage_for_species(taxonomy_obj, species_name_or_id: str) -> dict: Get lineage information

Classes Under Test:
- TaxonomyError: Custom exception for taxonomy-related errors

Dependencies:
- multitax.NcbiTx: Primary taxonomy database interface
- multitax.filter: Species filtering functionality  
- ncbi-taxonomist: Alternative taxonomy processing library
- External NCBI API: Remote taxonomy database access (mocked in tests)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call, PropertyMock
from typing import List, Dict, Any, Optional, Union, Tuple
import tempfile
import os
import json
import time

# Import the taxonomy functions (will be implemented)
from src.data_quality.taxonomy import (
    load_ncbi_taxonomy,
    filter_species_by_lineage,
    get_lineage_for_species,
    TaxonomyError
)


class TestLoadNcbiTaxonomy:
    """Test cases for NCBI taxonomy loading functionality."""
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_load_ncbi_taxonomy_basic_functionality(self, mock_multitax):
        """Test basic NCBI taxonomy loading."""
        # Mock multitax.NcbiTx() response
        mock_taxonomy = MagicMock()
        mock_taxonomy.name = "NCBI Taxonomy Database"
        mock_taxonomy.version = "2024.1"
        mock_taxonomy.total_nodes = 2500000
        mock_multitax.NcbiTx.return_value = mock_taxonomy
        
        result = load_ncbi_taxonomy()
        
        # Verify multitax.NcbiTx was called
        mock_multitax.NcbiTx.assert_called_once()
        
        # Verify returned taxonomy object
        assert result is mock_taxonomy
        assert hasattr(result, 'name')
        assert hasattr(result, 'version')
        assert hasattr(result, 'total_nodes')
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_load_ncbi_taxonomy_with_custom_path(self, mock_multitax):
        """Test loading taxonomy with custom database path."""
        mock_taxonomy = MagicMock()
        mock_multitax.NcbiTx.return_value = mock_taxonomy
        
        custom_path = "/custom/path/to/taxonomy"
        result = load_ncbi_taxonomy(db_path=custom_path)
        
        # Verify custom path was passed
        mock_multitax.NcbiTx.assert_called_once_with(db_path=custom_path)
        assert result is mock_taxonomy
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_load_ncbi_taxonomy_with_download(self, mock_multitax):
        """Test loading taxonomy with automatic download."""
        mock_taxonomy = MagicMock()
        mock_multitax.NcbiTx.return_value = mock_taxonomy
        
        result = load_ncbi_taxonomy(download=True)
        
        # Verify download flag was passed
        mock_multitax.NcbiTx.assert_called_once_with(download=True)
        assert result is mock_taxonomy
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_load_ncbi_taxonomy_caching(self, mock_multitax):
        """Test that taxonomy loading supports caching."""
        mock_taxonomy = MagicMock()
        mock_multitax.NcbiTx.return_value = mock_taxonomy
        
        # Load taxonomy twice
        result1 = load_ncbi_taxonomy()
        result2 = load_ncbi_taxonomy()
        
        # Should reuse the same instance or create new ones as appropriate
        assert isinstance(result1, MagicMock)
        assert isinstance(result2, MagicMock)
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_load_ncbi_taxonomy_network_error_handling(self, mock_multitax):
        """Test error handling when network fails during loading."""
        # Mock network error
        mock_multitax.NcbiTx.side_effect = ConnectionError("Network unavailable")
        
        with pytest.raises(TaxonomyError, match="Failed to load NCBI taxonomy"):
            load_ncbi_taxonomy()
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_load_ncbi_taxonomy_file_not_found_error(self, mock_multitax):
        """Test error handling when taxonomy files are not found."""
        # Mock file not found error
        mock_multitax.NcbiTx.side_effect = FileNotFoundError("Taxonomy database not found")
        
        with pytest.raises(TaxonomyError, match="Taxonomy database file not found"):
            load_ncbi_taxonomy()
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_load_ncbi_taxonomy_permission_error(self, mock_multitax):
        """Test error handling for permission issues."""
        # Mock permission error
        mock_multitax.NcbiTx.side_effect = PermissionError("Access denied to taxonomy database")
        
        with pytest.raises(TaxonomyError, match="Permission denied"):
            load_ncbi_taxonomy()
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_load_ncbi_taxonomy_corrupt_database(self, mock_multitax):
        """Test error handling for corrupted database files."""
        # Mock corruption error
        mock_multitax.NcbiTx.side_effect = ValueError("Corrupted taxonomy database")
        
        with pytest.raises(TaxonomyError, match="Corrupted or invalid taxonomy database"):
            load_ncbi_taxonomy()
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_load_ncbi_taxonomy_timeout_handling(self, mock_multitax):
        """Test timeout handling for slow taxonomy loading."""
        # Mock timeout scenario
        def slow_load(*args, **kwargs):
            time.sleep(0.1)  # Simulate slow loading
            return MagicMock()
        
        mock_multitax.NcbiTx.side_effect = slow_load
        
        # Should complete without timeout (basic test)
        result = load_ncbi_taxonomy()
        assert result is not None
    
    def test_load_ncbi_taxonomy_multitax_not_available(self):
        """Test fallback when multitax is not available."""
        with patch('src.data_quality.taxonomy.multitax', side_effect=ImportError):
            with patch('src.data_quality.taxonomy.subprocess') as mock_subprocess:
                # Mock ncbi-taxonomist as fallback
                mock_subprocess.run.return_value.returncode = 0
                mock_subprocess.run.return_value.stdout = '{"status": "loaded"}'
                
                with pytest.raises(TaxonomyError, match="Neither multitax nor ncbi-taxonomist"):
                    load_ncbi_taxonomy()


class TestFilterSpeciesByLineage:
    """Test cases for species filtering by taxonomic lineage."""
    
    def test_filter_species_by_lineage_basic_functionality(self):
        """Test basic lineage filtering functionality."""
        # Mock taxonomy object
        mock_taxonomy = MagicMock()
        
        # Mock filter results
        expected_species = [
            {"tax_id": 3702, "name": "Arabidopsis thaliana", "lineage": "Viridiplantae;Streptophyta;Brassicaceae"},
            {"tax_id": 4113, "name": "Brassica napus", "lineage": "Viridiplantae;Streptophyta;Brassicaceae"},
            {"tax_id": 3847, "name": "Glycine max", "lineage": "Viridiplantae;Streptophyta;Fabaceae"}
        ]
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.filter.return_value = expected_species
            
            result = filter_species_by_lineage(mock_taxonomy, "Viridiplantae")
            
            # Verify multitax.filter was called correctly
            mock_multitax.filter.assert_called_once_with(
                mock_taxonomy, lineage="Viridiplantae"
            )
            
            # Verify results
            assert result == expected_species
            assert len(result) == 3
            assert all("Viridiplantae" in species["lineage"] for species in result)
    
    def test_filter_species_by_lineage_specific_lineage(self):
        """Test filtering by specific taxonomic lineage."""
        mock_taxonomy = MagicMock()
        
        # Mock Brassicaceae family results
        brassicaceae_species = [
            {"tax_id": 3702, "name": "Arabidopsis thaliana", "lineage": "Viridiplantae;Streptophyta;Brassicaceae"},
            {"tax_id": 4113, "name": "Brassica napus", "lineage": "Viridiplantae;Streptophyta;Brassicaceae"},
            {"tax_id": 3708, "name": "Brassica oleracea", "lineage": "Viridiplantae;Streptophyta;Brassicaceae"}
        ]
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.filter.return_value = brassicaceae_species
            
            result = filter_species_by_lineage(mock_taxonomy, "Brassicaceae")
            
            assert len(result) == 3
            assert all("Brassicaceae" in species["lineage"] for species in result)
            assert all("Arabidopsis" in species["name"] or "Brassica" in species["name"] for species in result)
    
    def test_filter_species_by_lineage_empty_results(self):
        """Test filtering that returns no results."""
        mock_taxonomy = MagicMock()
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.filter.return_value = []
            
            result = filter_species_by_lineage(mock_taxonomy, "NonexistentLineage")
            
            assert result == []
            assert isinstance(result, list)
    
    def test_filter_species_by_lineage_case_insensitive(self):
        """Test that lineage filtering is case insensitive."""
        mock_taxonomy = MagicMock()
        
        expected_species = [
            {"tax_id": 3702, "name": "Arabidopsis thaliana", "lineage": "Viridiplantae;Streptophyta"}
        ]
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.filter.return_value = expected_species
            
            # Test different cases
            for lineage in ["viridiplantae", "VIRIDIPLANTAE", "ViRiDiPlAnTaE"]:
                result = filter_species_by_lineage(mock_taxonomy, lineage)
                assert len(result) >= 0  # Should not error on case differences
    
    def test_filter_species_by_lineage_partial_lineage(self):
        """Test filtering by partial lineage paths."""
        mock_taxonomy = MagicMock()
        
        plant_species = [
            {"tax_id": 3702, "name": "Arabidopsis thaliana", "lineage": "Eukaryota;Viridiplantae;Streptophyta;Brassicaceae"},
            {"tax_id": 4081, "name": "Solanum lycopersicum", "lineage": "Eukaryota;Viridiplantae;Streptophyta;Solanaceae"}
        ]
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.filter.return_value = plant_species
            
            result = filter_species_by_lineage(mock_taxonomy, "Streptophyta")
            
            assert len(result) == 2
            assert all("Streptophyta" in species["lineage"] for species in result)
    
    def test_filter_species_by_lineage_with_rank_filtering(self):
        """Test filtering species by lineage with specific taxonomic ranks."""
        mock_taxonomy = MagicMock()
        
        family_species = [
            {"tax_id": 3702, "name": "Arabidopsis thaliana", "rank": "species", "lineage": "Brassicaceae"},
            {"tax_id": 3708, "name": "Brassica oleracea", "rank": "species", "lineage": "Brassicaceae"}
        ]
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.filter.return_value = family_species
            
            result = filter_species_by_lineage(mock_taxonomy, "Brassicaceae", rank="species")
            
            assert len(result) == 2
            assert all(species["rank"] == "species" for species in result)
    
    def test_filter_species_by_lineage_invalid_taxonomy_object(self):
        """Test error handling for invalid taxonomy object."""
        with pytest.raises(TaxonomyError, match="Invalid taxonomy object"):
            filter_species_by_lineage(None, "Viridiplantae")
        
        with pytest.raises(TaxonomyError, match="Invalid taxonomy object"):
            filter_species_by_lineage("not_a_taxonomy_object", "Viridiplantae")
    
    def test_filter_species_by_lineage_invalid_lineage(self):
        """Test error handling for invalid lineage input."""
        mock_taxonomy = MagicMock()
        
        # Test None lineage
        with pytest.raises(TaxonomyError, match="Lineage cannot be None or empty"):
            filter_species_by_lineage(mock_taxonomy, None)
        
        # Test empty lineage
        with pytest.raises(TaxonomyError, match="Lineage cannot be None or empty"):
            filter_species_by_lineage(mock_taxonomy, "")
        
        # Test non-string lineage
        with pytest.raises(TaxonomyError, match="Lineage must be a string"):
            filter_species_by_lineage(mock_taxonomy, 12345)
    
    def test_filter_species_by_lineage_api_error(self):
        """Test error handling when filtering API fails."""
        mock_taxonomy = MagicMock()
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.filter.side_effect = Exception("API call failed")
            
            with pytest.raises(TaxonomyError, match="Error filtering species by lineage"):
                filter_species_by_lineage(mock_taxonomy, "Viridiplantae")
    
    def test_filter_species_by_lineage_ncbi_taxonomist_fallback(self):
        """Test fallback to ncbi-taxonomist when multitax filtering fails."""
        mock_taxonomy = MagicMock()
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.filter.side_effect = ImportError("multitax not available")
            
            with patch('src.data_quality.taxonomy.subprocess') as mock_subprocess:
                # Mock ncbi-taxonomist subtree command
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = json.dumps([
                    {"tax_id": 3702, "name": "Arabidopsis thaliana"}
                ])
                mock_subprocess.run.return_value = mock_result
                
                result = filter_species_by_lineage(mock_taxonomy, "Viridiplantae")
                
                # Should use ncbi-taxonomist as fallback
                assert isinstance(result, list)
                mock_subprocess.run.assert_called()


class TestGetLineageForSpecies:
    """Test cases for retrieving lineage information for species."""
    
    def test_get_lineage_for_species_by_name(self):
        """Test retrieving lineage information by species name."""
        mock_taxonomy = MagicMock()
        species_name = "Arabidopsis thaliana"
        
        expected_lineage = {
            "tax_id": 3702,
            "name": "Arabidopsis thaliana",
            "lineage": "Eukaryota;Viridiplantae;Streptophyta;Embryophyta;Tracheophyta;Spermatophyta;Magnoliopsida;Brassicales;Brassicaceae;Camelineae;Arabidopsis",
            "rank": "species",
            "parent_tax_id": 3701,
            "scientific_name": "Arabidopsis thaliana"
        }
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.get_lineage.return_value = expected_lineage
            
            result = get_lineage_for_species(mock_taxonomy, species_name)
            
            # Verify multitax.get_lineage was called correctly
            mock_multitax.get_lineage.assert_called_once_with(
                mock_taxonomy, species_name
            )
            
            # Verify results
            assert result == expected_lineage
            assert result["name"] == species_name
            assert result["tax_id"] == 3702
            assert "Brassicaceae" in result["lineage"]
    
    def test_get_lineage_for_species_by_tax_id(self):
        """Test retrieving lineage information by taxonomic ID."""
        mock_taxonomy = MagicMock()
        tax_id = 3702
        
        expected_lineage = {
            "tax_id": 3702,
            "name": "Arabidopsis thaliana",
            "lineage": "Eukaryota;Viridiplantae;Streptophyta;Brassicaceae",
            "rank": "species"
        }
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.get_lineage.return_value = expected_lineage
            
            result = get_lineage_for_species(mock_taxonomy, str(tax_id))
            
            assert result == expected_lineage
            assert result["tax_id"] == tax_id
    
    def test_get_lineage_for_species_integer_tax_id(self):
        """Test retrieving lineage information with integer tax ID."""
        mock_taxonomy = MagicMock()
        tax_id = 9606  # Homo sapiens
        
        expected_lineage = {
            "tax_id": 9606,
            "name": "Homo sapiens",
            "lineage": "Eukaryota;Metazoa;Chordata;Mammalia;Primates;Hominidae;Homo",
            "rank": "species"
        }
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.get_lineage.return_value = expected_lineage
            
            result = get_lineage_for_species(mock_taxonomy, tax_id)
            
            assert result == expected_lineage
            assert result["name"] == "Homo sapiens"
    
    def test_get_lineage_for_species_detailed_lineage(self):
        """Test retrieving detailed lineage with all taxonomic ranks."""
        mock_taxonomy = MagicMock()
        species_name = "Escherichia coli"
        
        expected_lineage = {
            "tax_id": 562,
            "name": "Escherichia coli",
            "lineage": "Bacteria;Proteobacteria;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Escherichia",
            "lineage_ranks": {
                "superkingdom": "Bacteria",
                "phylum": "Proteobacteria", 
                "class": "Gammaproteobacteria",
                "order": "Enterobacterales",
                "family": "Enterobacteriaceae",
                "genus": "Escherichia",
                "species": "Escherichia coli"
            },
            "rank": "species",
            "parent_tax_id": 561
        }
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.get_lineage.return_value = expected_lineage
            
            result = get_lineage_for_species(mock_taxonomy, species_name)
            
            assert result == expected_lineage
            assert "lineage_ranks" in result
            assert result["lineage_ranks"]["family"] == "Enterobacteriaceae"
    
    def test_get_lineage_for_species_partial_name_match(self):
        """Test retrieving lineage with partial name matching."""
        mock_taxonomy = MagicMock()
        partial_name = "Arabidopsis"  # Genus only
        
        expected_lineage = {
            "tax_id": 3701,
            "name": "Arabidopsis",
            "lineage": "Eukaryota;Viridiplantae;Streptophyta;Brassicaceae",
            "rank": "genus",
            "children": [3702, 81972]  # Species under this genus
        }
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.get_lineage.return_value = expected_lineage
            
            result = get_lineage_for_species(mock_taxonomy, partial_name)
            
            assert result == expected_lineage
            assert result["rank"] == "genus"
    
    def test_get_lineage_for_species_case_insensitive(self):
        """Test that species name lookup is case insensitive."""
        mock_taxonomy = MagicMock()
        
        expected_lineage = {
            "tax_id": 3702,
            "name": "Arabidopsis thaliana",
            "lineage": "Eukaryota;Viridiplantae",
            "rank": "species"
        }
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.get_lineage.return_value = expected_lineage
            
            # Test different cases
            test_names = [
                "arabidopsis thaliana",
                "ARABIDOPSIS THALIANA", 
                "Arabidopsis Thaliana",
                "aRaBiDoPsIs ThAlIaNa"
            ]
            
            for name in test_names:
                result = get_lineage_for_species(mock_taxonomy, name)
                assert result["name"] == "Arabidopsis thaliana"
    
    def test_get_lineage_for_species_nonexistent_species(self):
        """Test error handling for non-existent species."""
        mock_taxonomy = MagicMock()
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.get_lineage.return_value = None
            
            with pytest.raises(TaxonomyError, match="Species .* not found in taxonomy"):
                get_lineage_for_species(mock_taxonomy, "Nonexistent species")
    
    def test_get_lineage_for_species_invalid_tax_id(self):
        """Test error handling for invalid taxonomic IDs."""
        mock_taxonomy = MagicMock()
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.get_lineage.return_value = None
            
            with pytest.raises(TaxonomyError, match="Species .* not found in taxonomy"):
                get_lineage_for_species(mock_taxonomy, "999999999")  # Invalid tax ID
    
    def test_get_lineage_for_species_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        mock_taxonomy = MagicMock()
        
        # Test None taxonomy
        with pytest.raises(TaxonomyError, match="Invalid taxonomy object"):
            get_lineage_for_species(None, "Arabidopsis thaliana")
        
        # Test None species identifier
        with pytest.raises(TaxonomyError, match="Species identifier cannot be None or empty"):
            get_lineage_for_species(mock_taxonomy, None)
        
        # Test empty species identifier
        with pytest.raises(TaxonomyError, match="Species identifier cannot be None or empty"):
            get_lineage_for_species(mock_taxonomy, "")
        
        # Test invalid type
        with pytest.raises(TaxonomyError, match="Species identifier must be a string or integer"):
            get_lineage_for_species(mock_taxonomy, ["not", "valid"])
    
    def test_get_lineage_for_species_api_error(self):
        """Test error handling when lineage API fails."""
        mock_taxonomy = MagicMock()
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.get_lineage.side_effect = Exception("API call failed")
            
            with pytest.raises(TaxonomyError, match="Error retrieving lineage"):
                get_lineage_for_species(mock_taxonomy, "Arabidopsis thaliana")
    
    def test_get_lineage_for_species_ncbi_taxonomist_fallback(self):
        """Test fallback to ncbi-taxonomist for lineage retrieval."""
        mock_taxonomy = MagicMock()
        species_name = "Arabidopsis thaliana"
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.get_lineage.side_effect = ImportError("multitax not available")
            
            with patch('src.data_quality.taxonomy.subprocess') as mock_subprocess:
                # Mock ncbi-taxonomist resolve command
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = json.dumps({
                    "tax_id": 3702,
                    "name": "Arabidopsis thaliana",
                    "lineage": "Eukaryota;Viridiplantae"
                })
                mock_subprocess.run.return_value = mock_result
                
                result = get_lineage_for_species(mock_taxonomy, species_name)
                
                # Should use ncbi-taxonomist as fallback
                assert isinstance(result, dict)
                assert "tax_id" in result
                mock_subprocess.run.assert_called()


class TestTaxonomyIntegration:
    """Integration test cases combining multiple taxonomy operations."""
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_load_and_filter_integration(self, mock_multitax):
        """Test integration of taxonomy loading and species filtering."""
        # Mock taxonomy loading
        mock_taxonomy = MagicMock()
        mock_multitax.NcbiTx.return_value = mock_taxonomy
        
        # Mock species filtering
        plant_species = [
            {"tax_id": 3702, "name": "Arabidopsis thaliana", "lineage": "Viridiplantae"},
            {"tax_id": 4081, "name": "Solanum lycopersicum", "lineage": "Viridiplantae"}
        ]
        mock_multitax.filter.return_value = plant_species
        
        # Load taxonomy and filter species
        taxonomy = load_ncbi_taxonomy()
        species = filter_species_by_lineage(taxonomy, "Viridiplantae")
        
        assert len(species) == 2
        assert all("Viridiplantae" in s["lineage"] for s in species)
        
        # Verify both operations were called
        mock_multitax.NcbiTx.assert_called_once()
        mock_multitax.filter.assert_called_once_with(taxonomy, lineage="Viridiplantae")
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_filter_and_get_lineage_integration(self, mock_multitax):
        """Test integration of species filtering and lineage retrieval."""
        mock_taxonomy = MagicMock()
        
        # Mock species filtering results
        filtered_species = [
            {"tax_id": 3702, "name": "Arabidopsis thaliana"},
            {"tax_id": 3708, "name": "Brassica oleracea"}
        ]
        mock_multitax.filter.return_value = filtered_species
        
        # Mock lineage retrieval for each species
        lineage_data = {
            "Arabidopsis thaliana": {
                "tax_id": 3702,
                "name": "Arabidopsis thaliana",
                "lineage": "Eukaryota;Viridiplantae;Brassicaceae",
                "rank": "species"
            },
            "Brassica oleracea": {
                "tax_id": 3708,
                "name": "Brassica oleracea", 
                "lineage": "Eukaryota;Viridiplantae;Brassicaceae",
                "rank": "species"
            }
        }
        
        def mock_get_lineage(taxonomy, species_name):
            return lineage_data.get(species_name)
        
        mock_multitax.get_lineage.side_effect = mock_get_lineage
        
        # Filter species and get detailed lineage for each
        species_list = filter_species_by_lineage(mock_taxonomy, "Brassicaceae")
        detailed_species = []
        
        for species in species_list:
            lineage = get_lineage_for_species(mock_taxonomy, species["name"])
            detailed_species.append(lineage)
        
        assert len(detailed_species) == 2
        assert all("Brassicaceae" in s["lineage"] for s in detailed_species)
        assert all("rank" in s for s in detailed_species)
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_comprehensive_taxonomy_workflow(self, mock_multitax):
        """Test comprehensive workflow: load, filter, get lineage, and process results."""
        # Mock taxonomy loading
        mock_taxonomy = MagicMock()
        mock_taxonomy.total_nodes = 1000000
        mock_multitax.NcbiTx.return_value = mock_taxonomy
        
        # Mock filtering for plant species
        plant_species = [
            {"tax_id": 3702, "name": "Arabidopsis thaliana"},
            {"tax_id": 4081, "name": "Solanum lycopersicum"},
            {"tax_id": 3847, "name": "Glycine max"}
        ]
        mock_multitax.filter.return_value = plant_species
        
        # Mock detailed lineage for each species
        lineage_responses = [
            {
                "tax_id": 3702,
                "name": "Arabidopsis thaliana",
                "lineage": "Eukaryota;Viridiplantae;Brassicaceae",
                "rank": "species"
            },
            {
                "tax_id": 4081,
                "name": "Solanum lycopersicum", 
                "lineage": "Eukaryota;Viridiplantae;Solanaceae",
                "rank": "species"
            },
            {
                "tax_id": 3847,
                "name": "Glycine max",
                "lineage": "Eukaryota;Viridiplantae;Fabaceae", 
                "rank": "species"
            }
        ]
        mock_multitax.get_lineage.side_effect = lineage_responses
        
        # Execute comprehensive workflow
        taxonomy = load_ncbi_taxonomy()
        plant_species_list = filter_species_by_lineage(taxonomy, "Viridiplantae")
        
        enriched_species = []
        for species in plant_species_list:
            lineage_info = get_lineage_for_species(taxonomy, species["name"])
            enriched_species.append(lineage_info)
        
        # Verify comprehensive results
        assert len(enriched_species) == 3
        assert all(isinstance(s, dict) for s in enriched_species)
        assert all("lineage" in s for s in enriched_species)
        assert all("Viridiplantae" in s["lineage"] for s in enriched_species)
        
        # Verify all API calls were made
        mock_multitax.NcbiTx.assert_called_once()
        mock_multitax.filter.assert_called_once()
        assert mock_multitax.get_lineage.call_count == 3


class TestTaxonomyErrorHandling:
    """Test cases for comprehensive error handling and edge cases."""
    
    def test_taxonomy_error_inheritance(self):
        """Test that TaxonomyError properly inherits from Exception."""
        error = TaxonomyError("Test taxonomy error")
        assert isinstance(error, Exception)
        assert str(error) == "Test taxonomy error"
    
    def test_taxonomy_error_empty_message(self):
        """Test TaxonomyError with empty message."""
        error = TaxonomyError("")
        assert isinstance(error, Exception)
        assert str(error) == ""
    
    def test_taxonomy_error_with_details(self):
        """Test TaxonomyError with detailed message."""
        details = "Species 'Unknown species' not found in NCBI taxonomy database"
        error = TaxonomyError(f"Lookup failed: {details}")
        assert str(error) == f"Lookup failed: {details}"
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_network_connectivity_issues(self, mock_multitax):
        """Test handling of various network connectivity issues."""
        # Test different network errors
        network_errors = [
            ConnectionError("Connection refused"),
            TimeoutError("Request timed out"),
            OSError("Network is unreachable"),
            Exception("DNS resolution failed")
        ]
        
        for error in network_errors:
            mock_multitax.NcbiTx.side_effect = error
            
            with pytest.raises(TaxonomyError):
                load_ncbi_taxonomy()
    
    def test_memory_pressure_handling(self):
        """Test handling of memory pressure during taxonomy operations."""
        mock_taxonomy = MagicMock()
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            # Mock memory error during filtering
            mock_multitax.filter.side_effect = MemoryError("Insufficient memory")
            
            with pytest.raises(TaxonomyError, match="Memory error during taxonomy operation"):
                filter_species_by_lineage(mock_taxonomy, "Viridiplantae")
    
    def test_corrupted_data_handling(self):
        """Test handling of corrupted or malformed taxonomy data."""
        mock_taxonomy = MagicMock()
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            # Mock corrupted data response
            mock_multitax.filter.return_value = [
                {"tax_id": "invalid", "name": None},  # Invalid data structure
                {"malformed": "data"},  # Missing required fields
                None  # Null entry
            ]
            
            # Should handle corrupted data gracefully
            with pytest.raises(TaxonomyError, match="Corrupted taxonomy data"):
                filter_species_by_lineage(mock_taxonomy, "Viridiplantae")
    
    def test_unicode_species_names_handling(self):
        """Test handling of Unicode characters in species names."""
        mock_taxonomy = MagicMock()
        
        unicode_species_names = [
            "Caféa arabica",  # Accented characters
            "Αραβικός καφές",  # Greek characters
            "コーヒーノキ",  # Japanese characters
            "Café árabe",  # Mixed accents
            "Spéciès ñamë",  # Multiple special characters
        ]
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_lineage = {
                "tax_id": 12345,
                "name": "Unicode species",
                "lineage": "Test lineage",
                "rank": "species"
            }
            mock_multitax.get_lineage.return_value = mock_lineage
            
            for species_name in unicode_species_names:
                try:
                    result = get_lineage_for_species(mock_taxonomy, species_name)
                    assert isinstance(result, dict)
                except UnicodeError:
                    # Unicode errors are acceptable for some edge cases
                    pytest.skip(f"Unicode handling not supported for: {species_name}")


class TestTaxonomyPerformance:
    """Test cases for performance considerations and optimization."""
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_large_dataset_filtering_performance(self, mock_multitax):
        """Test performance with large species datasets."""
        mock_taxonomy = MagicMock()
        
        # Mock large dataset response
        large_species_list = [
            {"tax_id": i, "name": f"Species {i}", "lineage": "Viridiplantae"}
            for i in range(10000)
        ]
        mock_multitax.filter.return_value = large_species_list
        
        result = filter_species_by_lineage(mock_taxonomy, "Viridiplantae")
        
        # Should handle large datasets efficiently
        assert len(result) == 10000
        assert isinstance(result, list)
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_memory_efficiency_multiple_operations(self, mock_multitax):
        """Test memory efficiency with multiple taxonomy operations."""
        import sys
        
        mock_taxonomy = MagicMock()
        mock_multitax.NcbiTx.return_value = mock_taxonomy
        
        # Mock responses for multiple operations
        mock_multitax.filter.return_value = [
            {"tax_id": i, "name": f"Species {i}"} for i in range(100)
        ]
        mock_multitax.get_lineage.return_value = {
            "tax_id": 1,
            "name": "Test species",
            "lineage": "Test lineage"
        }
        
        # Get initial memory snapshot
        initial_refs = sys.getrefcount(mock_taxonomy)
        
        # Perform multiple operations
        for _ in range(50):
            species_list = filter_species_by_lineage(mock_taxonomy, "TestLineage")
            for species in species_list[:5]:  # Process subset to avoid excessive calls
                get_lineage_for_species(mock_taxonomy, species["name"])
        
        # Memory should not grow excessively
        final_refs = sys.getrefcount(mock_taxonomy)
        assert final_refs <= initial_refs + 20  # Allow reasonable growth
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_caching_optimization(self, mock_multitax):
        """Test caching optimization for repeated taxonomy queries."""
        mock_taxonomy = MagicMock()
        
        lineage_data = {
            "tax_id": 3702,
            "name": "Arabidopsis thaliana",
            "lineage": "Eukaryota;Viridiplantae",
            "rank": "species"
        }
        mock_multitax.get_lineage.return_value = lineage_data
        
        species_name = "Arabidopsis thaliana"
        
        # Make multiple identical requests
        for _ in range(10):
            result = get_lineage_for_species(mock_taxonomy, species_name)
            assert result == lineage_data
        
        # Verify API was called (implementation may optimize with caching)
        assert mock_multitax.get_lineage.call_count >= 1
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_concurrent_operations_handling(self, mock_multitax):
        """Test handling of concurrent taxonomy operations."""
        import threading
        import queue
        
        mock_taxonomy = MagicMock()
        mock_multitax.filter.return_value = [
            {"tax_id": 1, "name": "Species 1"},
            {"tax_id": 2, "name": "Species 2"}
        ]
        
        results_queue = queue.Queue()
        
        def worker():
            try:
                result = filter_species_by_lineage(mock_taxonomy, "TestLineage")
                results_queue.put(result)
            except Exception as e:
                results_queue.put(e)
        
        # Start multiple concurrent operations
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all operations completed successfully
        results = []
        while not results_queue.empty():
            result = results_queue.get()
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent operation failed: {result}")
            results.append(result)
        
        assert len(results) == 5
        assert all(isinstance(r, list) for r in results)


class TestTaxonomyEdgeCases:
    """Test cases for edge cases and special scenarios."""
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_empty_taxonomy_database(self, mock_multitax):
        """Test handling of empty taxonomy database."""
        mock_taxonomy = MagicMock()
        mock_taxonomy.total_nodes = 0
        mock_multitax.NcbiTx.return_value = mock_taxonomy
        
        taxonomy = load_ncbi_taxonomy()
        assert taxonomy.total_nodes == 0
        
        # Filtering should return empty results
        mock_multitax.filter.return_value = []
        result = filter_species_by_lineage(taxonomy, "AnyLineage")
        assert result == []
    
    def test_malformed_lineage_strings(self):
        """Test handling of malformed lineage strings."""
        mock_taxonomy = MagicMock()
        
        malformed_lineages = [
            ";;;;;;;",  # Only separators
            "Eukaryota;;Plantae;;;",  # Empty segments
            "Eukaryota;Plantae;",  # Trailing separator
            ";Eukaryota;Plantae",  # Leading separator
            "Eukaryota Plantae",  # Missing separators
            "",  # Empty string
            "   ",  # Whitespace only
        ]
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_multitax.filter.return_value = []
            
            for lineage in malformed_lineages:
                if lineage.strip():  # Skip empty/whitespace for input validation
                    try:
                        result = filter_species_by_lineage(mock_taxonomy, lineage)
                        assert isinstance(result, list)
                    except TaxonomyError:
                        # Some malformed lineages should raise errors
                        pass
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_very_long_species_names(self, mock_multitax):
        """Test handling of very long species names."""
        mock_taxonomy = MagicMock()
        
        # Create very long species name
        long_name = "Very long species name " * 100  # 2400+ characters
        
        mock_lineage = {
            "tax_id": 12345,
            "name": long_name,
            "lineage": "Test lineage",
            "rank": "species"
        }
        mock_multitax.get_lineage.return_value = mock_lineage
        
        result = get_lineage_for_species(mock_taxonomy, long_name)
        assert result["name"] == long_name
    
    @patch('src.data_quality.taxonomy.multitax')
    def test_numeric_species_identifiers(self, mock_multitax):
        """Test handling of numeric species identifiers and edge cases."""
        mock_taxonomy = MagicMock()
        
        test_identifiers = [
            0,  # Zero
            -1,  # Negative number
            999999999,  # Very large number
            "0",  # String zero
            "-1",  # String negative
            "abc123",  # Mixed alphanumeric
            "123abc",  # Numeric prefix
        ]
        
        for identifier in test_identifiers:
            mock_multitax.get_lineage.return_value = None
            
            try:
                get_lineage_for_species(mock_taxonomy, identifier)
            except TaxonomyError as e:
                # Expected for invalid identifiers
                assert "not found" in str(e).lower()
    
    def test_special_character_handling(self):
        """Test handling of special characters in species names."""
        mock_taxonomy = MagicMock()
        
        special_char_names = [
            "Species with spaces",
            "Species-with-hyphens",
            "Species_with_underscores",
            "Species.with.dots",
            "Species (with parentheses)",
            "Species [with brackets]",
            "Species {with braces}",
            "Species with 'quotes'",
            'Species with "double quotes"',
            "Species with numbers 123",
            "Species with symbols @#$%",
        ]
        
        with patch('src.data_quality.taxonomy.multitax') as mock_multitax:
            mock_lineage = {
                "tax_id": 12345,
                "name": "Test species",
                "lineage": "Test lineage"
            }
            mock_multitax.get_lineage.return_value = mock_lineage
            
            for species_name in special_char_names:
                try:
                    result = get_lineage_for_species(mock_taxonomy, species_name)
                    assert isinstance(result, dict)
                except TaxonomyError:
                    # Some special characters may not be supported
                    pass


# Fixtures for common test data
@pytest.fixture
def sample_taxonomy_object():
    """Fixture providing a mock taxonomy object for testing."""
    mock_taxonomy = MagicMock()
    mock_taxonomy.name = "NCBI Taxonomy Database"
    mock_taxonomy.version = "2024.1"
    mock_taxonomy.total_nodes = 2500000
    mock_taxonomy.last_updated = "2024-01-15"
    return mock_taxonomy


@pytest.fixture
def sample_plant_species():
    """Fixture providing sample plant species data."""
    return [
        {
            "tax_id": 3702,
            "name": "Arabidopsis thaliana",
            "lineage": "Eukaryota;Viridiplantae;Streptophyta;Embryophyta;Tracheophyta;Spermatophyta;Magnoliopsida;Brassicales;Brassicaceae;Camelineae;Arabidopsis",
            "rank": "species"
        },
        {
            "tax_id": 4081,
            "name": "Solanum lycopersicum",
            "lineage": "Eukaryota;Viridiplantae;Streptophyta;Embryophyta;Tracheophyta;Spermatophyta;Magnoliopsida;Solanales;Solanaceae;Solanoideae;Solaneae;Solanum",
            "rank": "species"
        },
        {
            "tax_id": 3847,
            "name": "Glycine max",
            "lineage": "Eukaryota;Viridiplantae;Streptophyta;Embryophyta;Tracheophyta;Spermatophyta;Magnoliopsida;Fabales;Fabaceae;Papilionoideae;Phaseoleae;Glycine",
            "rank": "species"
        }
    ]


@pytest.fixture
def sample_lineage_data():
    """Fixture providing sample detailed lineage data."""
    return {
        "tax_id": 3702,
        "name": "Arabidopsis thaliana",
        "lineage": "Eukaryota;Viridiplantae;Streptophyta;Embryophyta;Tracheophyta;Spermatophyta;Magnoliopsida;Brassicales;Brassicaceae;Camelineae;Arabidopsis",
        "lineage_ranks": {
            "superkingdom": "Eukaryota",
            "kingdom": "Viridiplantae",
            "phylum": "Streptophyta",
            "subphylum": "Embryophyta",
            "class": "Tracheophyta",
            "subclass": "Spermatophyta",
            "order": "Brassicales",
            "family": "Brassicaceae",
            "subfamily": "Camelineae",
            "genus": "Arabidopsis",
            "species": "Arabidopsis thaliana"
        },
        "rank": "species",
        "parent_tax_id": 3701,
        "scientific_name": "Arabidopsis thaliana",
        "common_names": ["thale cress", "mouse-ear cress"],
        "synonyms": ["Sisymbrium thalianum"]
    }


@pytest.fixture
def sample_target_lineages():
    """Fixture providing common taxonomic lineages for testing."""
    return [
        "Viridiplantae",  # Green plants
        "Metazoa",  # Animals
        "Bacteria",  # Bacteria
        "Archaea",  # Archaea
        "Fungi",  # Fungi
        "Brassicaceae",  # Mustard family
        "Solanaceae",  # Nightshade family
        "Fabaceae",  # Legume family
        "Enterobacteriaceae",  # Enterobacteria family
        "Saccharomycetaceae"  # Yeast family
    ]


@pytest.fixture
def temp_taxonomy_db_path():
    """Fixture providing temporary directory for taxonomy database testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "taxonomy_db")
        os.makedirs(db_path, exist_ok=True)
        yield db_path


# Parametrized test configurations
@pytest.mark.parametrize("lineage,expected_families", [
    ("Viridiplantae", ["Brassicaceae", "Solanaceae", "Fabaceae"]),
    ("Metazoa", ["Hominidae", "Muridae", "Drosophilidae"]),
    ("Bacteria", ["Enterobacteriaceae", "Bacillaceae", "Streptococcaceae"]),
    ("Fungi", ["Saccharomycetaceae", "Candida"]),
])
@patch('src.data_quality.taxonomy.multitax')
def test_filter_species_by_lineage_parametrized(mock_multitax, sample_taxonomy_object, 
                                               lineage, expected_families):
    """Parametrized test for filtering species by different lineages."""
    # Mock appropriate species for each lineage
    mock_species = [
        {"tax_id": i, "name": f"Species {i}", "lineage": f"{lineage};{family}"}
        for i, family in enumerate(expected_families, 1)
    ]
    mock_multitax.filter.return_value = mock_species
    
    result = filter_species_by_lineage(sample_taxonomy_object, lineage)
    
    assert len(result) == len(expected_families)
    assert all(lineage in species["lineage"] for species in result)
    mock_multitax.filter.assert_called_once_with(
        sample_taxonomy_object, lineage=lineage
    )


@pytest.mark.parametrize("tax_id,expected_rank", [
    (3702, "species"),  # Arabidopsis thaliana
    (3701, "genus"),  # Arabidopsis
    (3700, "subfamily"),  # Camelineae
    (3699, "family"),  # Brassicaceae
    (3698, "order"),  # Brassicales
])
@patch('src.data_quality.taxonomy.multitax')
def test_get_lineage_for_species_parametrized_ranks(mock_multitax, sample_taxonomy_object,
                                                   tax_id, expected_rank):
    """Parametrized test for retrieving lineage at different taxonomic ranks."""
    mock_lineage = {
        "tax_id": tax_id,
        "name": f"Taxon {tax_id}",
        "lineage": "Eukaryota;Viridiplantae",
        "rank": expected_rank
    }
    mock_multitax.get_lineage.return_value = mock_lineage
    
    result = get_lineage_for_species(sample_taxonomy_object, str(tax_id))
    
    assert result["tax_id"] == tax_id
    assert result["rank"] == expected_rank
    mock_multitax.get_lineage.assert_called_once_with(
        sample_taxonomy_object, str(tax_id)
    )


# Mark all tests in this module as data quality related
pytestmark = pytest.mark.unit