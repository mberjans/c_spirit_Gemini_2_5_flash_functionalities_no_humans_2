"""
Unit tests for src/data_quality/normalizer.py

This module tests the data quality normalization functionality for cleaning and
standardizing entity names, and performing fuzzy string matching to identify
similar entities in the AIM2-ODIE ontology development and information extraction system.

Test Coverage:
- Name normalization: case conversion, whitespace handling, specific word processing
- Fuzzy matching: FuzzyWuzzy integration with configurable thresholds and multiple algorithms
- Edge cases: empty strings, None values, empty lists, special characters
- Error handling: invalid inputs, type mismatches, threshold validation
- Performance considerations: large datasets, memory efficiency
- Integration scenarios: combining normalization with fuzzy matching

Functions Under Test:
- normalize_name(name: str) -> str: Basic name cleaning and normalization
- find_fuzzy_matches(query: str, candidates: List[str], threshold: int = 80) -> List[Tuple[str, int]]: Fuzzy string matching
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple, Dict, Any, Optional
import re

# Import the data quality normalizer functions (will be implemented)
from src.data_quality.normalizer import (
    normalize_name,
    find_fuzzy_matches,
    NormalizationError
)


class TestNormalizeName:
    """Test cases for name normalization functionality."""
    
    def test_normalize_name_basic_case_conversion(self):
        """Test basic case conversion from various formats."""
        # Test all uppercase to title case
        assert normalize_name("KING ARTHUR") == "King Arthur"
        
        # Test all lowercase to title case
        assert normalize_name("king arthur") == "King Arthur"
        
        # Test mixed case normalization
        assert normalize_name("kInG aRtHuR") == "King Arthur"
        
        # Test single word
        assert normalize_name("GLUCOSE") == "Glucose"
        assert normalize_name("glucose") == "Glucose"
    
    def test_normalize_name_whitespace_handling(self):
        """Test extra whitespace removal and normalization."""
        # Test leading and trailing whitespace
        assert normalize_name("  King Arthur  ") == "King Arthur"
        
        # Test multiple spaces between words
        assert normalize_name("King    Arthur") == "King Arthur"
        
        # Test mixed whitespace characters
        assert normalize_name("King\t\tArthur") == "King Arthur"
        assert normalize_name("King\n\nArthur") == "King Arthur"
        assert normalize_name("King\r\nArthur") == "King Arthur"
        
        # Test combination of various whitespace
        assert normalize_name("  \t King   \n  Arthur \r  ") == "King Arthur"
    
    def test_normalize_name_specific_word_handling(self):
        """Test handling of specific words like 'the', articles, and prepositions."""
        # Test 'the' at the beginning (should remain lowercase except when first word)
        assert normalize_name("THE KING") == "The King"
        assert normalize_name("king THE arthur") == "King the Arthur"
        
        # Test other articles and prepositions
        assert normalize_name("A TALE OF TWO CITIES") == "A Tale of Two Cities"
        assert normalize_name("THE LORD OF THE RINGS") == "The Lord of the Rings"
        
        # Test prepositions in middle positions
        assert normalize_name("JOURNEY TO THE CENTER") == "Journey to the Center"
        assert normalize_name("BATTLE FOR THE THRONE") == "Battle for the Throne"
        
        # Test conjunctions
        assert normalize_name("KING AND QUEEN") == "King and Queen"
        assert normalize_name("FAST AND FURIOUS") == "Fast and Furious"
    
    def test_normalize_name_scientific_names(self):
        """Test normalization of scientific and biological names."""
        # Test genus species format
        assert normalize_name("ARABIDOPSIS THALIANA") == "Arabidopsis Thaliana"
        assert normalize_name("homo sapiens") == "Homo Sapiens"
        
        # Test chemical compound names
        assert normalize_name("ASCORBIC ACID") == "Ascorbic Acid"
        assert normalize_name("beta-carotene") == "Beta-Carotene"
        
        # Test names with hyphens and special characters
        assert normalize_name("alpha-D-glucose") == "Alpha-D-Glucose"
        assert normalize_name("N-acetyl-L-cysteine") == "N-Acetyl-L-Cysteine"
    
    def test_normalize_name_special_characters(self):
        """Test handling of names with special characters and punctuation."""
        # Test names with hyphens
        assert normalize_name("alpha-amylase") == "Alpha-Amylase"
        assert normalize_name("beta-carotene") == "Beta-Carotene"
        
        # Test names with apostrophes
        assert normalize_name("o'malley") == "O'Malley"
        assert normalize_name("mcdonald's") == "Mcdonald's"
        
        # Test names with numbers
        assert normalize_name("vitamin b12") == "Vitamin B12"
        assert normalize_name("coenzyme q10") == "Coenzyme Q10"
        
        # Test names with parentheses
        assert normalize_name("calcium (ca2+)") == "Calcium (Ca2+)"
        assert normalize_name("adenosine triphosphate (atp)") == "Adenosine Triphosphate (Atp)"
    
    def test_normalize_name_unicode_characters(self):
        """Test handling of Unicode characters in names."""
        # Test accented characters
        assert normalize_name("café") == "Café"
        assert normalize_name("naïve") == "Naïve"
        assert normalize_name("résumé") == "Résumé"
        
        # Test Greek letters (common in scientific names)
        assert normalize_name("α-glucose") == "Α-Glucose"
        assert normalize_name("β-carotene") == "Β-Carotene"
    
    def test_normalize_name_empty_and_edge_cases(self):
        """Test edge cases including empty strings and whitespace-only strings."""
        # Test empty string
        assert normalize_name("") == ""
        
        # Test whitespace-only string
        assert normalize_name("   ") == ""
        assert normalize_name("\t\n\r") == ""
        
        # Test single character
        assert normalize_name("a") == "A"
        
        # Test single word with various cases
        assert normalize_name("WORD") == "Word"
        assert normalize_name("word") == "Word"
        assert normalize_name("WoRd") == "Word"
    
    def test_normalize_name_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test None input
        with pytest.raises(NormalizationError, match="Input name cannot be None"):
            normalize_name(None)
        
        # Test non-string input
        with pytest.raises(NormalizationError, match="Input must be a string"):
            normalize_name(12345)
        
        with pytest.raises(NormalizationError, match="Input must be a string"):
            normalize_name(["not", "a", "string"])
        
        with pytest.raises(NormalizationError, match="Input must be a string"):
            normalize_name({"not": "a string"})
    
    @pytest.mark.parametrize("input_name,expected", [
        ("SIMPLE NAME", "Simple Name"),
        ("complex name with MIXED case", "Complex Name with Mixed Case"),
        ("THE quick BROWN fox", "The Quick Brown Fox"),
        ("", ""),
        ("single", "Single"),
        ("alpha-BETA-gamma", "Alpha-Beta-Gamma"),
        ("vitamin C", "Vitamin C"),
        ("COVID-19", "Covid-19"),
        ("pH value", "Ph Value"),
        ("mRNA expression", "Mrna Expression"),
    ])
    def test_normalize_name_parametrized(self, input_name, expected):
        """Parametrized test for various normalization scenarios."""
        result = normalize_name(input_name)
        assert result == expected


class TestFindFuzzyMatches:
    """Test cases for fuzzy string matching functionality."""
    
    @patch('fuzzywuzzy.process.extract')
    def test_find_fuzzy_matches_basic_functionality(self, mock_extract):
        """Test basic fuzzy matching with default threshold."""
        # Mock FuzzyWuzzy response
        mock_extract.return_value = [
            ("Arabidopsis thaliana", 95),
            ("Arabidopsis lyrata", 85),
            ("Brassica napus", 60)  # Below default threshold
        ]
        
        query = "Arabidopsis"
        candidates = ["Arabidopsis thaliana", "Arabidopsis lyrata", "Brassica napus", "Solanum lycopersicum"]
        
        result = find_fuzzy_matches(query, candidates)
        
        # Should return matches above default threshold (80)
        expected = [("Arabidopsis thaliana", 95), ("Arabidopsis lyrata", 85)]
        assert result == expected
        
        # Verify FuzzyWuzzy was called correctly
        mock_extract.assert_called_once_with(
            query, candidates, limit=len(candidates)
        )
    
    @patch('fuzzywuzzy.process.extract')
    def test_find_fuzzy_matches_custom_threshold(self, mock_extract):
        """Test fuzzy matching with custom threshold."""
        mock_extract.return_value = [
            ("glucose", 98),
            ("glucose-6-phosphate", 70),
            ("fructose", 45),
            ("sucrose", 40)
        ]
        
        query = "glucose"
        candidates = ["glucose", "glucose-6-phosphate", "fructose", "sucrose"]
        
        # Test with higher threshold
        result = find_fuzzy_matches(query, candidates, threshold=90)
        expected = [("glucose", 98)]
        assert result == expected
        
        # Test with lower threshold
        result = find_fuzzy_matches(query, candidates, threshold=60)
        expected = [("glucose", 98), ("glucose-6-phosphate", 70)]
        assert result == expected
    
    @patch('fuzzywuzzy.fuzz.ratio')
    @patch('fuzzywuzzy.fuzz.partial_ratio')
    @patch('fuzzywuzzy.fuzz.token_sort_ratio')
    @patch('fuzzywuzzy.fuzz.token_set_ratio')
    def test_find_fuzzy_matches_multiple_algorithms(self, mock_token_set, mock_token_sort, 
                                                   mock_partial, mock_ratio):
        """Test that fuzzy matching uses multiple FuzzyWuzzy algorithms."""
        # Mock different algorithm scores
        mock_ratio.return_value = 85
        mock_partial.return_value = 90
        mock_token_sort.return_value = 88
        mock_token_set.return_value = 92
        
        query = "King Arthur"
        candidates = ["King Arthur of Camelot"]
        
        with patch('fuzzywuzzy.process.extract') as mock_extract:
            # Mock extract to return the highest score from our algorithms
            mock_extract.return_value = [("King Arthur of Camelot", 92)]
            
            result = find_fuzzy_matches(query, candidates, threshold=80)
            
            assert len(result) == 1
            assert result[0] == ("King Arthur of Camelot", 92)
    
    def test_find_fuzzy_matches_empty_candidates(self):
        """Test fuzzy matching with empty candidates list."""
        query = "test query"
        candidates = []
        
        result = find_fuzzy_matches(query, candidates)
        assert result == []
    
    def test_find_fuzzy_matches_no_matches_above_threshold(self):
        """Test fuzzy matching when no candidates meet the threshold."""
        with patch('fuzzywuzzy.process.extract') as mock_extract:
            # All matches below threshold
            mock_extract.return_value = [
                ("completely different", 30),
                ("totally unrelated", 25),
                ("nothing similar", 20)
            ]
            
            query = "specific query"
            candidates = ["completely different", "totally unrelated", "nothing similar"]
            
            result = find_fuzzy_matches(query, candidates, threshold=80)
            assert result == []
    
    def test_find_fuzzy_matches_perfect_match(self):
        """Test fuzzy matching with perfect exact matches."""
        with patch('fuzzywuzzy.process.extract') as mock_extract:
            mock_extract.return_value = [
                ("exact match", 100),
                ("close match", 85),
                ("distant match", 60)
            ]
            
            query = "exact match"
            candidates = ["exact match", "close match", "distant match"]
            
            result = find_fuzzy_matches(query, candidates, threshold=80)
            expected = [("exact match", 100), ("close match", 85)]
            assert result == expected
    
    def test_find_fuzzy_matches_case_sensitivity(self):
        """Test fuzzy matching handles case differences appropriately."""
        with patch('fuzzywuzzy.process.extract') as mock_extract:
            mock_extract.return_value = [
                ("King Arthur", 95),
                ("KING ARTHUR", 95),
                ("king arthur", 95)
            ]
            
            query = "King Arthur"
            candidates = ["King Arthur", "KING ARTHUR", "king arthur"]
            
            result = find_fuzzy_matches(query, candidates, threshold=90)
            assert len(result) == 3
            assert all(score >= 90 for _, score in result)
    
    def test_find_fuzzy_matches_special_characters(self):
        """Test fuzzy matching with special characters and punctuation."""
        with patch('fuzzywuzzy.process.extract') as mock_extract:
            mock_extract.return_value = [
                ("α-D-glucose", 88),
                ("alpha-D-glucose", 92),
                ("a-D-glucose", 85)
            ]
            
            query = "alpha-D-glucose"
            candidates = ["α-D-glucose", "alpha-D-glucose", "a-D-glucose"]
            
            result = find_fuzzy_matches(query, candidates, threshold=80)
            assert len(result) == 3
    
    def test_find_fuzzy_matches_scientific_names(self):
        """Test fuzzy matching with scientific nomenclature."""
        with patch('fuzzywuzzy.process.extract') as mock_extract:
            mock_extract.return_value = [
                ("Arabidopsis thaliana", 85),
                ("Arabidopsis lyrata", 82),
                ("Brassica oleracea", 45)
            ]
            
            query = "Arabidopsis"
            candidates = ["Arabidopsis thaliana", "Arabidopsis lyrata", "Brassica oleracea"]
            
            result = find_fuzzy_matches(query, candidates, threshold=80)
            expected = [("Arabidopsis thaliana", 85), ("Arabidopsis lyrata", 82)]
            assert result == expected
    
    def test_find_fuzzy_matches_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test None query
        with pytest.raises(NormalizationError, match="Query string cannot be None"):
            find_fuzzy_matches(None, ["candidate1", "candidate2"])
        
        # Test non-string query
        with pytest.raises(NormalizationError, match="Query must be a string"):
            find_fuzzy_matches(12345, ["candidate1", "candidate2"])
        
        # Test None candidates
        with pytest.raises(NormalizationError, match="Candidates list cannot be None"):
            find_fuzzy_matches("query", None)
        
        # Test non-list candidates
        with pytest.raises(NormalizationError, match="Candidates must be a list"):
            find_fuzzy_matches("query", "not a list")
        
        # Test invalid threshold values
        with pytest.raises(NormalizationError, match="Threshold must be between 0 and 100"):
            find_fuzzy_matches("query", ["candidate"], threshold=150)
        
        with pytest.raises(NormalizationError, match="Threshold must be between 0 and 100"):
            find_fuzzy_matches("query", ["candidate"], threshold=-10)
        
        # Test non-integer threshold
        with pytest.raises(NormalizationError, match="Threshold must be an integer"):
            find_fuzzy_matches("query", ["candidate"], threshold=80.5)
    
    def test_find_fuzzy_matches_empty_strings_in_candidates(self):
        """Test handling of empty strings in candidates list."""
        with patch('fuzzywuzzy.process.extract') as mock_extract:
            mock_extract.return_value = [
                ("valid candidate", 85),
                ("", 0)  # Empty string should have low score
            ]
            
            query = "test query"
            candidates = ["valid candidate", "", "another candidate"]
            
            result = find_fuzzy_matches(query, candidates, threshold=80)
            expected = [("valid candidate", 85)]
            assert result == expected
    
    def test_find_fuzzy_matches_non_string_candidates(self):
        """Test error handling when candidates contain non-string items."""
        candidates = ["valid string", 12345, "another string"]
        
        with pytest.raises(NormalizationError, match="All candidates must be strings"):
            find_fuzzy_matches("query", candidates)
    
    @pytest.mark.parametrize("threshold,expected_count", [
        (95, 1),  # Only highest matches
        (80, 3),  # Medium threshold
        (60, 4),  # Lower threshold
        (40, 5),  # Very low threshold
        (0, 5),   # All matches
    ])
    @patch('fuzzywuzzy.process.extract')
    def test_find_fuzzy_matches_threshold_variations(self, mock_extract, threshold, expected_count):
        """Parametrized test for different threshold values."""
        mock_extract.return_value = [
            ("exact match", 100),
            ("very close", 95),
            ("pretty close", 85),
            ("somewhat close", 70),
            ("distant", 50),
            ("very distant", 30)
        ]
        
        query = "test"
        candidates = ["exact match", "very close", "pretty close", "somewhat close", "distant", "very distant"]
        
        result = find_fuzzy_matches(query, candidates, threshold=threshold)
        
        # Count how many results should be above threshold
        mock_results = mock_extract.return_value
        expected_results = [item for item in mock_results if item[1] >= threshold]
        
        assert len(result) == len(expected_results)
        assert result == expected_results


class TestNormalizationIntegration:
    """Integration test cases combining normalization with fuzzy matching."""
    
    @patch('fuzzywuzzy.process.extract')
    def test_normalize_then_fuzzy_match_pipeline(self, mock_extract):
        """Test combining name normalization with fuzzy matching."""
        # First normalize the query
        raw_query = "KING arthur"
        normalized_query = normalize_name(raw_query)
        assert normalized_query == "King Arthur"
        
        # Then use normalized query for fuzzy matching
        mock_extract.return_value = [
            ("King Arthur of Camelot", 90),
            ("King Arthur Pendragon", 88),
            ("Arthur King", 75)
        ]
        
        candidates = ["King Arthur of Camelot", "King Arthur Pendragon", "Arthur King", "Merlin"]
        result = find_fuzzy_matches(normalized_query, candidates, threshold=80)
        
        expected = [("King Arthur of Camelot", 90), ("King Arthur Pendragon", 88)]
        assert result == expected
    
    @patch('fuzzywuzzy.process.extract')
    def test_normalize_candidates_before_matching(self, mock_extract):
        """Test normalizing both query and candidates before fuzzy matching."""
        # Normalize query
        raw_query = "glucose"
        normalized_query = normalize_name(raw_query)
        
        # Normalize candidates
        raw_candidates = ["GLUCOSE-6-PHOSPHATE", "beta-D-glucose", "FRUCTOSE"]
        normalized_candidates = [normalize_name(candidate) for candidate in raw_candidates]
        expected_normalized = ["Glucose-6-Phosphate", "Beta-D-Glucose", "Fructose"]
        assert normalized_candidates == expected_normalized
        
        # Mock fuzzy matching on normalized data
        mock_extract.return_value = [
            ("Beta-D-Glucose", 85),
            ("Glucose-6-Phosphate", 82),
            ("Fructose", 45)
        ]
        
        result = find_fuzzy_matches(normalized_query, normalized_candidates, threshold=80)
        expected = [("Beta-D-Glucose", 85), ("Glucose-6-Phosphate", 82)]
        assert result == expected
    
    def test_scientific_name_normalization_and_matching(self):
        """Test normalization and fuzzy matching for scientific names."""
        # Test scientific name normalization
        raw_names = [
            "arabidopsis THALIANA",
            "HOMO sapiens",
            "escherichia COLI"
        ]
        
        normalized_names = [normalize_name(name) for name in raw_names]
        expected = [
            "Arabidopsis Thaliana",
            "Homo Sapiens", 
            "Escherichia Coli"
        ]
        assert normalized_names == expected
        
        # Test fuzzy matching with normalized scientific names
        with patch('fuzzywuzzy.process.extract') as mock_extract:
            mock_extract.return_value = [
                ("Arabidopsis Thaliana", 100),
                ("Arabidopsis Lyrata", 85),
                ("Brassica Napus", 40)
            ]
            
            query = normalize_name("arabidopsis thaliana")
            candidates = normalized_names + ["Arabidopsis Lyrata", "Brassica Napus"]
            
            result = find_fuzzy_matches(query, candidates, threshold=80)
            assert len(result) >= 1
            assert result[0][1] >= 80  # At least one high-confidence match


class TestPerformanceAndEdgeCases:
    """Test cases for performance considerations and edge cases."""
    
    def test_normalize_name_performance_large_input(self):
        """Test normalization performance with large input strings."""
        # Create a large string
        large_name = "very long chemical compound name " * 1000
        
        # Should handle large inputs without errors
        result = normalize_name(large_name)
        assert len(result) > 0
        assert result.startswith("Very Long Chemical")
    
    @patch('fuzzywuzzy.process.extract')
    def test_fuzzy_matching_performance_large_candidates(self, mock_extract):
        """Test fuzzy matching performance with large candidate lists."""
        # Create a large candidates list
        large_candidates = [f"candidate_{i}" for i in range(1000)]
        
        # Mock response with subset of matches
        mock_extract.return_value = [(f"candidate_{i}", 80 + (i % 20)) for i in range(50)]
        
        query = "test_query"
        result = find_fuzzy_matches(query, large_candidates, threshold=85)
        
        # Should handle large lists efficiently
        assert isinstance(result, list)
        assert all(score >= 85 for _, score in result)
        
        # Verify FuzzyWuzzy was called with full candidate list
        mock_extract.assert_called_once_with(query, large_candidates, limit=len(large_candidates))
    
    def test_normalize_name_memory_efficiency(self):
        """Test memory efficiency of name normalization."""
        import sys
        
        # Process multiple large strings to test memory usage
        large_strings = [f"test string number {i} with extra content" * 100 for i in range(100)]
        
        # Get initial memory snapshot
        initial_refs = sys.getrefcount(large_strings)
        
        # Process all strings
        results = [normalize_name(s) for s in large_strings]
        
        # Verify processing completed
        assert len(results) == 100
        assert all(isinstance(result, str) for result in results)
        
        # Memory should not grow excessively (basic check)
        final_refs = sys.getrefcount(large_strings)
        assert final_refs <= initial_refs + 10  # Allow some reasonable growth
    
    def test_unicode_edge_cases(self):
        """Test edge cases with Unicode characters."""
        # Test various Unicode categories
        unicode_test_cases = [
            "café résumé naïve",  # Accented characters
            "α-D-glucose β-carotene",  # Greek letters
            "中文名称",  # Chinese characters
            "العربية",  # Arabic text
            "🧬 DNA 🧪",  # Emoji characters
            "test\u200Btext",  # Zero-width space
            "test\u00A0text",  # Non-breaking space
        ]
        
        for test_case in unicode_test_cases:
            try:
                result = normalize_name(test_case)
                # Should handle Unicode without crashing
                assert isinstance(result, str)
                assert len(result) >= 0
            except UnicodeError:
                # Unicode errors are acceptable for some edge cases
                pytest.skip(f"Unicode handling not supported for: {test_case}")
    
    def test_extreme_threshold_values(self):
        """Test fuzzy matching with extreme threshold values."""
        candidates = ["test1", "test2", "test3"]
        
        with patch('fuzzywuzzy.process.extract') as mock_extract:
            # Test minimum threshold
            mock_extract.return_value = [("test1", 1), ("test2", 0), ("test3", 50)]
            result = find_fuzzy_matches("query", candidates, threshold=0)
            assert len(result) == 3  # All matches included
            
            # Test maximum threshold
            mock_extract.return_value = [("test1", 100), ("test2", 99), ("test3", 98)]
            result = find_fuzzy_matches("query", candidates, threshold=100)
            assert len(result) == 1  # Only perfect matches
            assert result[0][1] == 100


class TestErrorHandlingAndValidation:
    """Comprehensive error handling and input validation tests."""
    
    def test_normalization_error_inheritance(self):
        """Test that NormalizationError properly inherits from Exception."""
        error = NormalizationError("Test error message")
        assert isinstance(error, Exception)
        assert str(error) == "Test error message"
    
    def test_normalize_name_type_validation(self):
        """Test comprehensive type validation for normalize_name."""
        invalid_inputs = [
            (None, "Input name cannot be None"),
            (123, "Input must be a string"),
            (12.34, "Input must be a string"),
            ([], "Input must be a string"),
            ({}, "Input must be a string"),
            (set(), "Input must be a string"),
            (True, "Input must be a string"),
        ]
        
        for invalid_input, expected_message in invalid_inputs:
            with pytest.raises(NormalizationError, match=expected_message):
                normalize_name(invalid_input)
    
    def test_fuzzy_matches_comprehensive_validation(self):
        """Test comprehensive input validation for find_fuzzy_matches."""
        valid_candidates = ["candidate1", "candidate2"]
        
        # Test query validation
        query_invalid_inputs = [
            (None, "Query string cannot be None"),
            (123, "Query must be a string"),
            ([], "Query must be a string"),
        ]
        
        for invalid_query, expected_message in query_invalid_inputs:
            with pytest.raises(NormalizationError, match=expected_message):
                find_fuzzy_matches(invalid_query, valid_candidates)
        
        # Test candidates validation
        candidates_invalid_inputs = [
            (None, "Candidates list cannot be None"),
            ("string", "Candidates must be a list"),
            (123, "Candidates must be a list"),
            ({}, "Candidates must be a list"),
        ]
        
        for invalid_candidates, expected_message in candidates_invalid_inputs:
            with pytest.raises(NormalizationError, match=expected_message):
                find_fuzzy_matches("query", invalid_candidates)
    
    def test_threshold_validation_edge_cases(self):
        """Test threshold validation with various edge cases."""
        query = "test"
        candidates = ["candidate"]
        
        # Test invalid threshold types
        invalid_thresholds = [
            (80.5, "Threshold must be an integer"),
            ("80", "Threshold must be an integer"),
            (None, "Threshold must be an integer"),
            ([], "Threshold must be an integer"),
        ]
        
        for invalid_threshold, expected_message in invalid_thresholds:
            with pytest.raises(NormalizationError, match=expected_message):
                find_fuzzy_matches(query, candidates, threshold=invalid_threshold)
        
        # Test out-of-range thresholds
        out_of_range_thresholds = [
            (-1, "Threshold must be between 0 and 100"),
            (101, "Threshold must be between 0 and 100"),
            (-100, "Threshold must be between 0 and 100"),
            (1000, "Threshold must be between 0 and 100"),
        ]
        
        for invalid_threshold, expected_message in out_of_range_thresholds:
            with pytest.raises(NormalizationError, match=expected_message):
                find_fuzzy_matches(query, candidates, threshold=invalid_threshold)
    
    def test_candidates_content_validation(self):
        """Test validation of candidates list content."""
        query = "test"
        
        # Test candidates with non-string items
        invalid_candidate_lists = [
            (["valid", 123, "also valid"], "All candidates must be strings"),
            (["valid", None, "also valid"], "All candidates must be strings"),
            (["valid", [], "also valid"], "All candidates must be strings"),
            ([123, 456, 789], "All candidates must be strings"),
        ]
        
        for invalid_candidates, expected_message in invalid_candidate_lists:
            with pytest.raises(NormalizationError, match=expected_message):
                find_fuzzy_matches(query, invalid_candidates)


# Fixtures for common test data
@pytest.fixture
def sample_scientific_names():
    """Fixture providing sample scientific names for testing."""
    return [
        "arabidopsis thaliana",
        "HOMO SAPIENS", 
        "escherichia COLI",
        "saccharomyces cerevisiae",
        "drosophila MELANOGASTER"
    ]


@pytest.fixture
def sample_chemical_compounds():
    """Fixture providing sample chemical compound names for testing."""
    return [
        "GLUCOSE",
        "alpha-D-glucose",
        "beta-carotene", 
        "ASCORBIC ACID",
        "adenosine triphosphate",
        "N-acetyl-L-cysteine"
    ]


@pytest.fixture
def sample_entity_variations():
    """Fixture providing entity name variations for fuzzy matching tests."""
    return {
        "glucose": [
            "glucose",
            "D-glucose", 
            "dextrose",
            "grape sugar",
            "glucose-6-phosphate"
        ],
        "arabidopsis": [
            "Arabidopsis thaliana",
            "Arabidopsis lyrata",
            "arabidopsis halleri",
            "thale cress",
            "mouse-ear cress"
        ]
    }


# Mark all tests in this module as data quality related
pytestmark = pytest.mark.unit