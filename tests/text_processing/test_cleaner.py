"""
Unit tests for src/text_processing/cleaner.py

This module tests the text cleaning and preprocessing functionality for normalizing,
tokenizing, deduplicating, filtering, and encoding text data in the AIM2-ODIE
ontology development and information extraction system.

Test Coverage:
- Text normalization: case conversion, whitespace handling, HTML tag removal
- Text tokenization: word and sentence segmentation using spaCy/NLTK
- Duplicate removal: exact and fuzzy matching with configurable thresholds
- Stopword filtering: English and biomedical stopwords with custom lists
- Encoding standardization: handling various input encodings to UTF-8
- Error handling for malformed inputs, encoding issues, and invalid parameters
- Edge cases: empty strings, None values, special characters, large texts
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import the text processing cleaner functions (will be implemented)
from src.text_processing.cleaner import (
    normalize_text,
    tokenize_text,
    remove_duplicates,
    filter_stopwords,
    standardize_encoding,
    TextCleaningError
)


class TestNormalizeText:
    """Test cases for text normalization functionality."""
    
    def test_normalize_text_basic_case_conversion(self):
        """Test basic case conversion to lowercase."""
        input_text = "PLANT METABOLOMICS Research"
        result = normalize_text(input_text)
        assert result == "plant metabolomics research"
    
    def test_normalize_text_whitespace_stripping(self):
        """Test whitespace stripping from beginning and end."""
        input_text = "  \t\n  plant metabolomics  \r\n  "
        result = normalize_text(input_text)
        assert result == "plant metabolomics"
    
    def test_normalize_text_multiple_whitespace_normalization(self):
        """Test normalization of multiple consecutive whitespaces."""
        input_text = "plant    metabolomics\t\tresearch\n\nstudy"
        result = normalize_text(input_text)
        assert result == "plant metabolomics research study"
    
    def test_normalize_text_html_tag_removal(self):
        """Test removal of HTML tags from text."""
        input_text = "<p>Plant <strong>metabolomics</strong> is the study of <em>metabolites</em>.</p>"
        result = normalize_text(input_text)
        assert result == "plant metabolomics is the study of metabolites."
    
    def test_normalize_text_complex_html_removal(self):
        """Test removal of complex HTML structures."""
        input_text = """
        <div class="abstract">
            <h2>Abstract</h2>
            <p>Plant <a href="#ref1">metabolomics</a> analysis of <span style="color:red">secondary metabolites</span>.</p>
            <!-- This is a comment -->
            <ul>
                <li>Flavonoids</li>
                <li>Alkaloids</li>
            </ul>
        </div>
        """
        result = normalize_text(input_text)
        expected = "abstract plant metabolomics analysis of secondary metabolites. flavonoids alkaloids"
        assert result == expected
    
    def test_normalize_text_empty_string(self):
        """Test normalization of empty string."""
        result = normalize_text("")
        assert result == ""
    
    def test_normalize_text_whitespace_only(self):
        """Test normalization of whitespace-only string."""
        result = normalize_text("   \t\n\r   ")
        assert result == ""
    
    def test_normalize_text_none_input(self):
        """Test error handling for None input."""
        with pytest.raises(TextCleaningError, match="Input text cannot be None"):
            normalize_text(None)
    
    def test_normalize_text_non_string_input(self):
        """Test error handling for non-string input."""
        with pytest.raises(TextCleaningError, match="Input must be a string"):
            normalize_text(12345)
    
    def test_normalize_text_unicode_characters(self):
        """Test handling of Unicode characters."""
        input_text = "Café metabolomics résearch naïve approach"
        result = normalize_text(input_text)
        assert result == "café metabolomics résearch naïve approach"
    
    def test_normalize_text_special_characters(self):
        """Test handling of special characters and punctuation."""
        input_text = "Plant metabolomics: analysis & research (2023)!"
        result = normalize_text(input_text)
        assert result == "plant metabolomics: analysis & research (2023)!"
    
    @pytest.mark.parametrize("input_text,expected", [
        ("Single", "single"),
        ("MULTIPLE WORDS", "multiple words"),
        ("Mixed   Case  \t Text", "mixed case text"),
        ("<tag>Content</tag>", "content"),
        ("", ""),
    ])
    def test_normalize_text_parametrized(self, input_text, expected):
        """Parametrized test for various normalization scenarios."""
        result = normalize_text(input_text)
        assert result == expected


class TestTokenizeText:
    """Test cases for text tokenization functionality."""
    
    @patch('spacy.load')
    def test_tokenize_text_basic_word_tokenization(self, mock_spacy_load):
        """Test basic word tokenization using spaCy."""
        # Mock spaCy model
        mock_doc = Mock()
        mock_token1 = Mock()
        mock_token1.text = "Plant"
        mock_token1.is_alpha = True
        mock_token1.is_space = False
        mock_token2 = Mock()
        mock_token2.text = "metabolomics"
        mock_token2.is_alpha = True
        mock_token2.is_space = False
        mock_doc.__iter__ = Mock(return_value=iter([mock_token1, mock_token2]))
        
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        result = tokenize_text("Plant metabolomics")
        
        assert result == ["Plant", "metabolomics"]
        mock_spacy_load.assert_called_once_with("en_core_web_sm")
    
    @patch('spacy.load')
    def test_tokenize_text_sentence_segmentation(self, mock_spacy_load):
        """Test sentence segmentation functionality."""
        # Mock spaCy model for sentence segmentation
        mock_sent1 = Mock()
        mock_sent1.text = "Plant metabolomics is important."
        mock_sent2 = Mock()
        mock_sent2.text = "It studies small molecules."
        
        mock_doc = Mock()
        mock_doc.sents = [mock_sent1, mock_sent2]
        
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        result = tokenize_text("Plant metabolomics is important. It studies small molecules.", mode="sentences")
        
        assert result == ["Plant metabolomics is important.", "It studies small molecules."]
    
    @patch('nltk.word_tokenize')
    @patch('nltk.download')
    def test_tokenize_text_nltk_fallback(self, mock_nltk_download, mock_word_tokenize):
        """Test NLTK fallback when spaCy is not available."""
        mock_word_tokenize.return_value = ["Plant", "metabolomics", "research"]
        
        with patch('spacy.load', side_effect=OSError("spaCy model not found")):
            result = tokenize_text("Plant metabolomics research", use_nltk=True)
        
        assert result == ["Plant", "metabolomics", "research"]
        mock_nltk_download.assert_called_with('punkt', quiet=True)
        mock_word_tokenize.assert_called_once_with("Plant metabolomics research")
    
    def test_tokenize_text_empty_string(self):
        """Test tokenization of empty string."""
        result = tokenize_text("")
        assert result == []
    
    def test_tokenize_text_none_input(self):
        """Test error handling for None input."""
        with pytest.raises(TextCleaningError, match="Input text cannot be None"):
            tokenize_text(None)
    
    def test_tokenize_text_non_string_input(self):
        """Test error handling for non-string input."""
        with pytest.raises(TextCleaningError, match="Input must be a string"):
            tokenize_text(12345)
    
    @patch('spacy.load')
    def test_tokenize_text_punctuation_filtering(self, mock_spacy_load):
        """Test filtering of punctuation tokens."""
        # Mock spaCy tokens with punctuation
        mock_tokens = []
        for text, is_alpha, is_punct in [("Plant", True, False), (",", False, True), ("metabolomics", True, False)]:
            token = Mock()
            token.text = text
            token.is_alpha = is_alpha
            token.is_punct = is_punct
            token.is_space = False
            mock_tokens.append(token)
        
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter(mock_tokens))
        
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        result = tokenize_text("Plant, metabolomics", filter_punct=True)
        
        assert result == ["Plant", "metabolomics"]
    
    @pytest.mark.parametrize("text,expected_length", [
        ("Single word", 2),
        ("Multiple words in sentence", 4),
        ("", 0),
        ("Word!", 1),  # Assuming punctuation is filtered
    ])
    @patch('spacy.load')
    def test_tokenize_text_parametrized(self, mock_spacy_load, text, expected_length):
        """Parametrized test for tokenization scenarios."""
        # Mock basic tokenization
        tokens = text.split() if text else []
        mock_tokens = []
        for token_text in tokens:
            token = Mock()
            token.text = token_text.rstrip('!')
            token.is_alpha = token_text.rstrip('!').isalpha()
            token.is_space = False
            token.is_punct = not token_text.rstrip('!').isalpha()
            mock_tokens.append(token)
        
        mock_doc = Mock()
        mock_doc.__iter__ = Mock(return_value=iter(mock_tokens))
        
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        result = tokenize_text(text, filter_punct=True)
        assert len(result) == expected_length


class TestRemoveDuplicates:
    """Test cases for duplicate removal functionality."""
    
    def test_remove_duplicates_exact_matches(self):
        """Test removal of exact duplicate strings."""
        text_list = [
            "plant metabolomics",
            "secondary metabolites",
            "plant metabolomics",
            "flavonoids analysis",
            "secondary metabolites"
        ]
        result = remove_duplicates(text_list)
        expected = ["plant metabolomics", "secondary metabolites", "flavonoids analysis"]
        assert set(result) == set(expected)
        assert len(result) == 3
    
    @patch('fuzzywuzzy.fuzz.ratio')
    def test_remove_duplicates_fuzzy_matching(self, mock_fuzz_ratio):
        """Test fuzzy duplicate removal with custom threshold."""
        # Mock fuzzy matching scores
        def mock_ratio(s1, s2):
            if (s1, s2) in [("plant metabolomics", "plant metabolomic"), ("plant metabolomic", "plant metabolomics")]:
                return 95  # Above threshold
            return 50  # Below threshold
        
        mock_fuzz_ratio.side_effect = mock_ratio
        
        text_list = [
            "plant metabolomics",
            "plant metabolomic",  # Similar enough to be considered duplicate
            "secondary metabolites",
            "alkaloid compounds"
        ]
        
        result = remove_duplicates(text_list, fuzzy_threshold=90)
        
        # Should keep only one of the similar strings
        assert len(result) == 3
        assert "secondary metabolites" in result
        assert "alkaloid compounds" in result
        # Either "plant metabolomics" or "plant metabolomic" should be present, but not both
        plant_variants = [s for s in result if "plant metabol" in s]
        assert len(plant_variants) == 1
    
    def test_remove_duplicates_empty_list(self):
        """Test duplicate removal on empty list."""
        result = remove_duplicates([])
        assert result == []
    
    def test_remove_duplicates_single_item(self):
        """Test duplicate removal on single-item list."""
        text_list = ["plant metabolomics"]
        result = remove_duplicates(text_list)
        assert result == ["plant metabolomics"]
    
    def test_remove_duplicates_no_duplicates(self):
        """Test duplicate removal when no duplicates exist."""
        text_list = [
            "plant metabolomics",
            "secondary metabolites",
            "flavonoids analysis",
            "alkaloid compounds"
        ]
        result = remove_duplicates(text_list)
        assert set(result) == set(text_list)
        assert len(result) == len(text_list)
    
    def test_remove_duplicates_case_sensitivity(self):
        """Test case-sensitive duplicate detection."""
        text_list = [
            "Plant Metabolomics",
            "plant metabolomics",
            "PLANT METABOLOMICS"
        ]
        result = remove_duplicates(text_list, case_sensitive=False)
        assert len(result) == 1
        
        result_case_sensitive = remove_duplicates(text_list, case_sensitive=True)
        assert len(result_case_sensitive) == 3
    
    def test_remove_duplicates_none_input(self):
        """Test error handling for None input."""
        with pytest.raises(TextCleaningError, match="Input text_list cannot be None"):
            remove_duplicates(None)
    
    def test_remove_duplicates_non_list_input(self):
        """Test error handling for non-list input."""
        with pytest.raises(TextCleaningError, match="Input must be a list"):
            remove_duplicates("not a list")
    
    def test_remove_duplicates_invalid_threshold(self):
        """Test error handling for invalid fuzzy threshold."""
        text_list = ["text1", "text2"]
        with pytest.raises(TextCleaningError, match="Fuzzy threshold must be between 0 and 100"):
            remove_duplicates(text_list, fuzzy_threshold=150)
        
        with pytest.raises(TextCleaningError, match="Fuzzy threshold must be between 0 and 100"):
            remove_duplicates(text_list, fuzzy_threshold=-10)
    
    @pytest.mark.parametrize("threshold,expected_count", [
        (100, 4),  # Only exact matches
        (90, 3),   # High similarity threshold
        (70, 2),   # Medium similarity threshold
        (50, 2),   # Low similarity threshold
    ])
    @patch('fuzzywuzzy.fuzz.ratio')
    def test_remove_duplicates_threshold_variations(self, mock_fuzz_ratio, threshold, expected_count):
        """Parametrized test for different fuzzy thresholds."""
        def mock_ratio(s1, s2):
            similarities = {
                ("plant metabolomics", "plant metabolomic"): 95,
                ("metabolomics analysis", "metabolomic analysis"): 85,
                ("secondary metabolites", "secondary compounds"): 75,
            }
            return similarities.get((s1, s2), similarities.get((s2, s1), 40))
        
        mock_fuzz_ratio.side_effect = mock_ratio
        
        text_list = [
            "plant metabolomics",
            "plant metabolomic",
            "metabolomics analysis", 
            "metabolomic analysis",
            "secondary metabolites",
            "secondary compounds"
        ]
        
        result = remove_duplicates(text_list, fuzzy_threshold=threshold)
        assert len(result) == expected_count


class TestFilterStopwords:
    """Test cases for stopword filtering functionality."""
    
    @patch('nltk.corpus.stopwords.words')
    def test_filter_stopwords_english_default(self, mock_stopwords):
        """Test filtering of default English stopwords."""
        mock_stopwords.return_value = ['the', 'is', 'and', 'of', 'in', 'a', 'to']
        
        tokens = ["the", "plant", "metabolomics", "is", "a", "study", "of", "metabolites"]
        result = filter_stopwords(tokens)
        
        expected = ["plant", "metabolomics", "study", "metabolites"]
        assert result == expected
        mock_stopwords.assert_called_once_with('english')
    
    @patch('nltk.corpus.stopwords.words')
    def test_filter_stopwords_custom_list(self, mock_stopwords):
        """Test filtering with custom stopwords list."""
        mock_stopwords.return_value = ['the', 'is', 'and']
        
        tokens = ["plant", "metabolomics", "analysis", "study", "research"]
        custom_stopwords = ["analysis", "study"]
        
        result = filter_stopwords(tokens, custom_stopwords_list=custom_stopwords)
        
        expected = ["plant", "metabolomics", "research"]
        assert result == expected
    
    @patch('nltk.corpus.stopwords.words')
    def test_filter_stopwords_biomedical_terms(self, mock_stopwords):
        """Test filtering of biomedical stopwords."""
        mock_stopwords.return_value = ['the', 'is', 'and']
        
        tokens = ["gene", "expression", "protein", "cell", "tissue", "metabolomics"]
        biomedical_stopwords = ["gene", "expression", "cell", "tissue"]
        
        result = filter_stopwords(tokens, custom_stopwords_list=biomedical_stopwords)
        
        expected = ["protein", "metabolomics"]
        assert result == expected
    
    def test_filter_stopwords_empty_tokens(self):
        """Test filtering empty token list."""
        result = filter_stopwords([])
        assert result == []
    
    def test_filter_stopwords_no_stopwords_found(self):
        """Test filtering when no stopwords are found in tokens."""
        tokens = ["metabolomics", "flavonoids", "alkaloids", "terpenoids"]
        result = filter_stopwords(tokens, custom_stopwords_list=["protein", "gene"])
        
        assert result == tokens
    
    def test_filter_stopwords_all_stopwords(self):
        """Test filtering when all tokens are stopwords."""
        tokens = ["the", "is", "and", "of", "in"]
        result = filter_stopwords(tokens, custom_stopwords_list=tokens)
        
        assert result == []
    
    def test_filter_stopwords_case_insensitive(self):
        """Test case-insensitive stopword filtering."""
        tokens = ["The", "Plant", "IS", "metabolomics"]
        custom_stopwords = ["the", "is", "and"]
        
        result = filter_stopwords(tokens, custom_stopwords_list=custom_stopwords, case_sensitive=False)
        
        expected = ["Plant", "metabolomics"]
        assert result == expected
    
    def test_filter_stopwords_case_sensitive(self):
        """Test case-sensitive stopword filtering."""
        tokens = ["The", "plant", "Is", "metabolomics"]
        custom_stopwords = ["the", "is", "and"]
        
        result = filter_stopwords(tokens, custom_stopwords_list=custom_stopwords, case_sensitive=True)
        
        expected = ["The", "plant", "Is", "metabolomics"]  # None match due to case
        assert result == expected
    
    def test_filter_stopwords_none_input(self):
        """Test error handling for None input."""
        with pytest.raises(TextCleaningError, match="Input tokens cannot be None"):
            filter_stopwords(None)
    
    def test_filter_stopwords_non_list_input(self):
        """Test error handling for non-list input."""
        with pytest.raises(TextCleaningError, match="Input must be a list"):
            filter_stopwords("not a list")
    
    @pytest.mark.parametrize("tokens,stopwords,expected", [
        (["plant", "the", "metabolomics"], ["the"], ["plant", "metabolomics"]),
        (["all", "stop", "words"], ["all", "stop", "words"], []),
        (["no", "stopwords", "here"], ["other", "words"], ["no", "stopwords", "here"]),
        ([], ["the", "is"], []),
    ])
    def test_filter_stopwords_parametrized(self, tokens, stopwords, expected):
        """Parametrized test for stopword filtering scenarios."""
        result = filter_stopwords(tokens, custom_stopwords_list=stopwords)
        assert result == expected


class TestStandardizeEncoding:
    """Test cases for text encoding standardization functionality."""
    
    def test_standardize_encoding_utf8_input(self):
        """Test standardization of UTF-8 encoded bytes."""
        input_bytes = "Plant metabolomics research".encode('utf-8')
        result = standardize_encoding(input_bytes)
        
        assert result == "Plant metabolomics research"
        assert isinstance(result, str)
    
    def test_standardize_encoding_latin1_input(self):
        """Test standardization of Latin-1 encoded bytes."""
        input_text = "Café metabolomics résearch"
        input_bytes = input_text.encode('latin-1')
        result = standardize_encoding(input_bytes, source_encoding='latin-1')
        
        assert result == input_text
    
    def test_standardize_encoding_cp1252_input(self):
        """Test standardization of CP1252 (Windows) encoded bytes."""
        input_text = "Plant metabolomics—analysis"
        input_bytes = input_text.encode('cp1252')
        result = standardize_encoding(input_bytes, source_encoding='cp1252')
        
        assert result == input_text
    
    def test_standardize_encoding_auto_detection(self):
        """Test automatic encoding detection."""
        input_text = "Plant metabolomics research"
        input_bytes = input_text.encode('utf-8')
        
        with patch('chardet.detect', return_value={'encoding': 'utf-8', 'confidence': 0.95}):
            result = standardize_encoding(input_bytes, auto_detect=True)
        
        assert result == input_text
    
    def test_standardize_encoding_custom_target(self):
        """Test encoding to custom target encoding."""
        input_bytes = "Plant metabolomics".encode('utf-8')
        result = standardize_encoding(input_bytes, target_encoding='ascii')
        
        # Should work for ASCII-compatible text
        assert result == "Plant metabolomics"
    
    def test_standardize_encoding_invalid_source(self):
        """Test error handling for invalid source encoding."""
        input_bytes = "Plant metabolomics".encode('utf-8')
        
        with pytest.raises(TextCleaningError, match="Failed to decode"):
            standardize_encoding(input_bytes, source_encoding='invalid-encoding')
    
    def test_standardize_encoding_decode_error(self):
        """Test handling of decode errors with error strategies."""
        # Create bytes that can't be decoded as ASCII
        input_bytes = "Café".encode('utf-8')
        
        # Should handle errors gracefully
        result = standardize_encoding(input_bytes, source_encoding='ascii', errors='ignore')
        assert "Caf" in result  # Non-ASCII characters ignored
    
    def test_standardize_encoding_none_input(self):
        """Test error handling for None input."""
        with pytest.raises(TextCleaningError, match="Input bytes cannot be None"):
            standardize_encoding(None)
    
    def test_standardize_encoding_non_bytes_input(self):
        """Test error handling for non-bytes input."""
        with pytest.raises(TextCleaningError, match="Input must be bytes"):
            standardize_encoding("string input")
    
    def test_standardize_encoding_empty_bytes(self):
        """Test standardization of empty bytes."""
        result = standardize_encoding(b"")
        assert result == ""
    
    @patch('chardet.detect')
    def test_standardize_encoding_detection_failure(self, mock_detect):
        """Test fallback when encoding detection fails."""
        mock_detect.return_value = {'encoding': None, 'confidence': 0.0}
        
        input_bytes = "Plant metabolomics".encode('utf-8')
        result = standardize_encoding(input_bytes, auto_detect=True, fallback_encoding='utf-8')
        
        assert result == "Plant metabolomics"
    
    @pytest.mark.parametrize("encoding,text", [
        ('utf-8', "Plant metabolomics"),
        ('ascii', "Simple text"),
        ('latin-1', "Café research"),
        ('utf-16', "Unicode text"),
    ])
    def test_standardize_encoding_parametrized(self, encoding, text):
        """Parametrized test for different encoding scenarios."""
        try:
            input_bytes = text.encode(encoding)
            result = standardize_encoding(input_bytes, source_encoding=encoding)
            assert result == text
        except UnicodeEncodeError:
            # Skip if text can't be encoded in the specified encoding
            pytest.skip(f"Text cannot be encoded as {encoding}")


class TestTextCleaningErrorHandling:
    """Test cases for error handling and edge cases across all functions."""
    
    def test_text_cleaning_error_inheritance(self):
        """Test that TextCleaningError properly inherits from Exception."""
        error = TextCleaningError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
    
    def test_memory_efficiency_large_text(self):
        """Test memory efficiency with large text inputs."""
        # Create a large text string
        large_text = "Plant metabolomics research. " * 10000
        
        # These should not raise memory errors
        normalized = normalize_text(large_text)
        assert len(normalized) > 0
        
        tokens = tokenize_text(large_text[:1000])  # Limit for tokenization test
        assert len(tokens) > 0
    
    def test_unicode_handling_across_functions(self):
        """Test Unicode handling across all text processing functions."""
        unicode_text = "Plant metabolömics résearch naïve approach café"
        
        # Should handle Unicode in normalization
        normalized = normalize_text(unicode_text)
        assert "metabolömics" in normalized
        
        # Should handle Unicode in tokenization
        tokens = tokenize_text(unicode_text)
        unicode_tokens = [t for t in tokens if any(ord(c) > 127 for c in t)]
        assert len(unicode_tokens) > 0
        
        # Should handle Unicode in deduplication
        duplicates = [unicode_text, unicode_text.upper()]
        result = remove_duplicates(duplicates, case_sensitive=False)
        assert len(result) == 1
    
    def test_concurrent_processing_safety(self):
        """Test thread safety of text processing functions."""
        import threading
        
        results = []
        errors = []
        
        def process_text(text):
            try:
                normalized = normalize_text(f"Test {text}")
                tokens = tokenize_text(normalized)
                filtered = filter_stopwords(tokens, custom_stopwords_list=["test"])
                results.append((normalized, tokens, filtered))
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=process_text, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have no errors and expected number of results
        assert len(errors) == 0
        assert len(results) == 10


class TestIntegrationScenarios:
    """Integration test cases combining multiple text processing functions."""
    
    def test_full_text_processing_pipeline(self):
        """Test complete text processing pipeline."""
        raw_text = """
        <p>PLANT METABOLOMICS is the study of small molecules.</p>
        <p>Plant metabolomics involves   analysis of metabolites.</p>
        <div>Secondary metabolites include flavonoids and alkaloids.</div>
        """
        
        # Step 1: Normalize text
        normalized = normalize_text(raw_text)
        expected_normalized = "plant metabolomics is the study of small molecules. plant metabolomics involves analysis of metabolites. secondary metabolites include flavonoids and alkaloids."
        assert normalized == expected_normalized
        
        # Step 2: Tokenize
        tokens = tokenize_text(normalized)
        assert "plant" in tokens
        assert "metabolomics" in tokens
        assert "flavonoids" in tokens
        
        # Step 3: Remove stopwords
        filtered_tokens = filter_stopwords(tokens, custom_stopwords_list=["the", "of", "is", "and"])
        assert "the" not in filtered_tokens
        assert "plant" in filtered_tokens
        
        # Step 4: Remove duplicates from processed texts
        text_variants = [
            "plant metabolomics study",
            "plant metabolomic study",  # Similar
            "secondary metabolites analysis"
        ]
        unique_texts = remove_duplicates(text_variants, fuzzy_threshold=90)
        assert len(unique_texts) <= len(text_variants)
    
    @patch('chardet.detect')
    def test_encoding_normalization_pipeline(self, mock_detect):
        """Test encoding standardization with text processing."""
        mock_detect.return_value = {'encoding': 'utf-8', 'confidence': 0.95}
        
        # Simulate text from different encoding
        original_text = "Plant metabolömics research café"
        encoded_bytes = original_text.encode('utf-8')
        
        # Step 1: Standardize encoding
        decoded_text = standardize_encoding(encoded_bytes, auto_detect=True)
        assert decoded_text == original_text
        
        # Step 2: Normalize
        normalized = normalize_text(decoded_text)
        assert normalized == "plant metabolömics research café"
        
        # Step 3: Tokenize
        tokens = tokenize_text(normalized)
        assert len(tokens) == 4


# Fixture for common test data
@pytest.fixture
def sample_biomedical_text():
    """Fixture providing sample biomedical text for testing."""
    return """
    <div class="abstract">
        <h2>Plant Metabolomics Analysis</h2>
        <p>Plant metabolomics is the comprehensive study of small molecules in plants.</p>
        <p>Secondary metabolites such as flavonoids, alkaloids, and terpenoids play crucial roles.</p>
        <ul>
            <li>Flavonoids: antioxidant compounds</li>
            <li>Alkaloids: nitrogen-containing compounds</li>
            <li>Terpenoids: diverse structural compounds</li>
        </ul>
    </div>
    """


@pytest.fixture
def sample_duplicate_texts():
    """Fixture providing sample texts with duplicates for testing."""
    return [
        "plant metabolomics analysis",
        "Plant Metabolomics Analysis",  # Case variation
        "plant metabolomic analysis",   # Fuzzy match
        "secondary metabolite study",
        "secondary metabolites study",  # Fuzzy match
        "flavonoid compound research",
        "plant metabolomics analysis"   # Exact duplicate
    ]


@pytest.fixture
def sample_encoded_bytes():
    """Fixture providing sample text in various encodings."""
    text = "Plant metabolömics café research"
    return {
        'utf-8': text.encode('utf-8'),
        'latin-1': text.encode('latin-1', errors='ignore'),
        'ascii': text.encode('ascii', errors='ignore'),
        'cp1252': text.encode('cp1252', errors='ignore')
    }


# Mark all tests in this module as text processing related
pytestmark = pytest.mark.unit