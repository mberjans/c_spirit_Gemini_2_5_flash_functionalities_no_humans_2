"""
Unit tests for src/text_processing/chunker.py

This module tests the text chunking functionality for preparing text for LLM processing
in the AIM2-ODIE ontology development and information extraction system. The chunking
functions split large texts into manageable pieces while maintaining semantic coherence.

Test Coverage:
- Fixed-size chunking: chunk_fixed_size with various sizes and overlaps
- Sentence-based chunking: chunk_by_sentences using NLTK/spaCy tokenizers
- Recursive character chunking: chunk_recursive_char using LangChain's RecursiveCharacterTextSplitter
- Edge cases: empty texts, very short texts, boundary conditions
- Error handling: invalid parameters, missing dependencies
- Semantic coherence: ensuring chunks don't split mid-word inappropriately
- Performance: handling large texts efficiently
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import the text processing chunker functions (will be implemented)
from src.text_processing.chunker import (
    chunk_fixed_size,
    chunk_by_sentences,
    chunk_recursive_char,
    ChunkingError
)


class TestChunkFixedSize:
    """Test cases for fixed-size text chunking functionality."""
    
    def test_chunk_fixed_size_basic_chunking(self):
        """Test basic fixed-size chunking with no overlap."""
        text = "Plant metabolomics is the comprehensive study of small molecules in plants. " \
               "These metabolites include primary and secondary compounds that are essential for plant function."
        
        chunks = chunk_fixed_size(text, chunk_size=50, chunk_overlap=0)
        
        assert len(chunks) >= 2
        assert all(len(chunk) <= 50 for chunk in chunks)
        
        # Check that all text is preserved
        reconstructed = "".join(chunks)
        assert len(reconstructed) == len(text)
    
    def test_chunk_fixed_size_with_overlap(self):
        """Test fixed-size chunking with overlap between chunks."""
        text = "Plant metabolomics research involves analyzing metabolites. " \
               "Secondary metabolites like flavonoids are particularly important."
        
        chunks = chunk_fixed_size(text, chunk_size=40, chunk_overlap=10)
        
        assert len(chunks) >= 2
        assert all(len(chunk) <= 40 for chunk in chunks)
        
        # Check overlap exists between consecutive chunks
        if len(chunks) > 1:
            overlap_found = False
            for i in range(len(chunks) - 1):
                # Check if any substring from end of current chunk appears in next chunk
                current_end = chunks[i][-10:]  # Last 10 chars of current chunk
                if any(current_end[j:] in chunks[i + 1] for j in range(len(current_end))):
                    overlap_found = True
                    break
            assert overlap_found, "No overlap found between chunks"
    
    def test_chunk_fixed_size_token_counting(self):
        """Test that chunk sizes are correctly measured in characters."""
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        
        # Test character-based chunking
        chunks = chunk_fixed_size(text, chunk_size=20, chunk_overlap=0)
        assert all(len(chunk) <= 20 for chunk in chunks)
        
        # Test with different chunk size to verify character counting
        chunks_small = chunk_fixed_size(text, chunk_size=15, chunk_overlap=0)
        assert all(len(chunk) <= 15 for chunk in chunks_small)
    
    def test_chunk_fixed_size_no_mid_word_splitting(self):
        """Test that chunks don't split words inappropriately."""
        text = "metabolomics flavonoids alkaloids terpenoids phenolics"
        
        chunks = chunk_fixed_size(text, chunk_size=20, chunk_overlap=0)
        
        for chunk in chunks:
            # Chunks should not start or end with partial words (except at text boundaries)
            if chunk != chunks[0]:  # Not first chunk
                assert chunk[0] == ' ' or chunk.split()[0] != text.split()[0], \
                    f"Chunk starts with partial word: '{chunk}'"
            if chunk != chunks[-1]:  # Not last chunk
                assert chunk[-1] == ' ' or chunk.split()[-1] in text, \
                    f"Chunk ends with partial word: '{chunk}'"
    
    def test_chunk_fixed_size_empty_text(self):
        """Test chunking of empty text."""
        chunks = chunk_fixed_size("", chunk_size=100, chunk_overlap=0)
        assert chunks == []
    
    def test_chunk_fixed_size_very_short_text(self):
        """Test chunking of text shorter than chunk size."""
        text = "Short text"
        chunks = chunk_fixed_size(text, chunk_size=100, chunk_overlap=0)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_fixed_size_single_word_larger_than_chunk(self):
        """Test handling of single words larger than chunk size."""
        text = "supercalifragilisticexpialidocious"
        chunks = chunk_fixed_size(text, chunk_size=10, chunk_overlap=0)
        
        # Should handle gracefully, either split the word or keep it whole
        assert len(chunks) >= 1
        assert "".join(chunks) == text or chunks[0] == text
    
    def test_chunk_fixed_size_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        text = "Sample text for testing"
        
        # Test negative chunk size
        with pytest.raises(ChunkingError, match="Chunk size must be positive"):
            chunk_fixed_size(text, chunk_size=-1, chunk_overlap=0)
        
        # Test zero chunk size
        with pytest.raises(ChunkingError, match="Chunk size must be positive"):
            chunk_fixed_size(text, chunk_size=0, chunk_overlap=0)
        
        # Test negative overlap
        with pytest.raises(ChunkingError, match="Chunk overlap cannot be negative"):
            chunk_fixed_size(text, chunk_size=10, chunk_overlap=-1)
        
        # Test overlap larger than chunk size
        with pytest.raises(ChunkingError, match="Chunk overlap cannot be larger than chunk size"):
            chunk_fixed_size(text, chunk_size=10, chunk_overlap=15)
    
    def test_chunk_fixed_size_none_input(self):
        """Test error handling for None input."""
        with pytest.raises(ChunkingError, match="Input text cannot be None"):
            chunk_fixed_size(None, chunk_size=10, chunk_overlap=0)
    
    def test_chunk_fixed_size_non_string_input(self):
        """Test error handling for non-string input."""
        with pytest.raises(ChunkingError, match="Input must be a string"):
            chunk_fixed_size(12345, chunk_size=10, chunk_overlap=0)
    
    @pytest.mark.parametrize("chunk_size,overlap,expected_min_chunks", [
        (50, 0, 2),   # No overlap
        (50, 10, 2),  # Small overlap
        (30, 5, 3),   # Multiple chunks with overlap
        (200, 0, 1),  # Single chunk for short text
    ])
    def test_chunk_fixed_size_parametrized(self, chunk_size, overlap, expected_min_chunks, 
                                          sample_scientific_text):
        """Parametrized test for various chunk size and overlap combinations."""
        chunks = chunk_fixed_size(sample_scientific_text, chunk_size, overlap)
        
        assert len(chunks) >= expected_min_chunks
        assert all(len(chunk) <= chunk_size for chunk in chunks)
        
        # Verify text preservation
        if overlap == 0:
            reconstructed = "".join(chunks)
            assert len(reconstructed) == len(sample_scientific_text)


class TestChunkBySentences:
    """Test cases for sentence-based text chunking functionality."""
    
    @patch('nltk.tokenize.sent_tokenize')
    @patch('src.text_processing.chunker.nltk.download')
    def test_chunk_by_sentences_nltk_basic(self, mock_download, mock_sent_tokenize):
        """Test basic sentence chunking using NLTK."""
        text = "Plant metabolomics is important. It studies small molecules. These are crucial for plant function."
        expected_sentences = [
            "Plant metabolomics is important.",
            "It studies small molecules.",
            "These are crucial for plant function."
        ]
        mock_sent_tokenize.return_value = expected_sentences
        
        chunks = chunk_by_sentences(text, tokenizer='nltk')
        
        assert chunks == expected_sentences
        mock_sent_tokenize.assert_called_once_with(text)
        mock_download.assert_called_with('punkt', quiet=True)
    
    @patch('spacy.load')
    def test_chunk_by_sentences_spacy_basic(self, mock_spacy_load):
        """Test basic sentence chunking using spaCy."""
        text = "Plant metabolomics research is advancing. New techniques are being developed."
        
        # Mock spaCy sentence segmentation
        mock_sent1 = Mock()
        mock_sent1.text = "Plant metabolomics research is advancing."
        mock_sent2 = Mock()
        mock_sent2.text = "New techniques are being developed."
        
        mock_doc = Mock()
        mock_doc.sents = [mock_sent1, mock_sent2]
        
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        chunks = chunk_by_sentences(text, tokenizer='spacy')
        
        expected = ["Plant metabolomics research is advancing.", "New techniques are being developed."]
        assert chunks == expected
        mock_spacy_load.assert_called_once_with("en_core_web_sm")
    
    def test_chunk_by_sentences_complex_punctuation(self):
        """Test sentence chunking with complex punctuation."""
        text = "Dr. Smith's research on plant metabolomics (published in 2023) shows interesting results. " \
               "The study analyzed flavonoids, e.g., quercetin and kaempferol. " \
               "Results indicate significant variations!"
        
        chunks = chunk_by_sentences(text)
        
        # Should correctly identify sentence boundaries despite complex punctuation
        assert len(chunks) == 3
        assert all(chunk.strip().endswith(('.', '!', '?')) for chunk in chunks)
    
    def test_chunk_by_sentences_scientific_abbreviations(self):
        """Test sentence chunking with scientific abbreviations."""
        text = "The concentration was 5 mg/L in H2O. Analysis via LC-MS/MS revealed metabolites. " \
               "Compounds like ATP, ADP, and NADH were detected."
        
        chunks = chunk_by_sentences(text)
        
        # Should not split on abbreviations like "mg/L", "LC-MS/MS"
        assert len(chunks) == 3
        assert any("mg/L" in chunk for chunk in chunks)
        assert any("LC-MS/MS" in chunk for chunk in chunks)
    
    def test_chunk_by_sentences_empty_text(self):
        """Test sentence chunking of empty text."""
        chunks = chunk_by_sentences("")
        assert chunks == []
    
    def test_chunk_by_sentences_single_sentence(self):
        """Test chunking of text with single sentence."""
        text = "Plant metabolomics is the study of small molecules in plants."
        chunks = chunk_by_sentences(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_by_sentences_no_punctuation(self):
        """Test chunking of text without sentence-ending punctuation."""
        text = "Plant metabolomics research alkaloids flavonoids"
        chunks = chunk_by_sentences(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_by_sentences_newlines_and_whitespace(self):
        """Test sentence chunking with newlines and extra whitespace."""
        text = "Plant metabolomics is important.\n\nIt studies metabolites.   \n\nResults are promising."
        
        chunks = chunk_by_sentences(text)
        
        assert len(chunks) == 3
        # Should clean up whitespace but preserve content
        assert all(chunk.strip() for chunk in chunks)
        assert "Plant metabolomics is important." in chunks[0]
    
    @patch('spacy.load', side_effect=OSError("Model not found"))
    @patch('nltk.tokenize.sent_tokenize')
    @patch('src.text_processing.chunker.nltk.download')
    def test_chunk_by_sentences_fallback_to_nltk(self, mock_download, mock_sent_tokenize, mock_spacy_load):
        """Test fallback to NLTK when spaCy is not available."""
        text = "First sentence. Second sentence."
        expected = ["First sentence.", "Second sentence."]
        mock_sent_tokenize.return_value = expected
        
        chunks = chunk_by_sentences(text, tokenizer='spacy')
        
        # Should fallback to NLTK
        assert chunks == expected
        mock_sent_tokenize.assert_called_once_with(text)
    
    def test_chunk_by_sentences_invalid_tokenizer(self):
        """Test error handling for invalid tokenizer."""
        text = "Sample text."
        
        with pytest.raises(ChunkingError, match="Unsupported tokenizer"):
            chunk_by_sentences(text, tokenizer='invalid_tokenizer')
    
    def test_chunk_by_sentences_none_input(self):
        """Test error handling for None input."""
        with pytest.raises(ChunkingError, match="Input text cannot be None"):
            chunk_by_sentences(None)
    
    @pytest.mark.parametrize("text,expected_count", [
        ("Single sentence.", 1),
        ("First sentence. Second sentence.", 2),
        ("One. Two. Three.", 3),
        ("No punctuation", 1),
        ("", 0),
    ])
    def test_chunk_by_sentences_parametrized(self, text, expected_count):
        """Parametrized test for sentence chunking scenarios."""
        chunks = chunk_by_sentences(text)
        assert len(chunks) == expected_count


class TestChunkRecursiveChar:
    """Test cases for recursive character text chunking using LangChain."""
    
    @patch('langchain.text_splitter.RecursiveCharacterTextSplitter')
    def test_chunk_recursive_char_basic(self, mock_text_splitter_class):
        """Test basic recursive character chunking."""
        text = "Plant metabolomics research involves analyzing small molecules. " \
               "These molecules include primary and secondary metabolites."
        
        expected_chunks = [
            "Plant metabolomics research involves analyzing small molecules.",
            "These molecules include primary and secondary metabolites."
        ]
        
        # Mock the RecursiveCharacterTextSplitter
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = expected_chunks
        mock_text_splitter_class.return_value = mock_splitter
        
        chunks = chunk_recursive_char(text, chunk_size=80, chunk_overlap=10)
        
        assert chunks == expected_chunks
        mock_text_splitter_class.assert_called_once_with(
            chunk_size=80,
            chunk_overlap=10,
            separators=["\n\n", "\n", " ", ""]
        )
        mock_splitter.split_text.assert_called_once_with(text)
    
    @patch('langchain.text_splitter.RecursiveCharacterTextSplitter')
    def test_chunk_recursive_char_custom_separators(self, mock_text_splitter_class):
        """Test recursive chunking with custom separators."""
        text = "Section1\n\nSection2\n\nSection3"
        custom_separators = ["\n\n", ".", "!", "?"]
        
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = ["Section1", "Section2", "Section3"]
        mock_text_splitter_class.return_value = mock_splitter
        
        chunks = chunk_recursive_char(text, chunk_size=50, chunk_overlap=5, 
                                    separators=custom_separators)
        
        mock_text_splitter_class.assert_called_once_with(
            chunk_size=50,
            chunk_overlap=5,
            separators=custom_separators
        )
    
    @patch('langchain.text_splitter.RecursiveCharacterTextSplitter')
    def test_chunk_recursive_char_default_separators(self, mock_text_splitter_class):
        """Test that default separators are used when none provided."""
        text = "Sample text for testing"
        
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = ["Sample text for testing"]
        mock_text_splitter_class.return_value = mock_splitter
        
        chunk_recursive_char(text, chunk_size=100, chunk_overlap=0)
        
        # Check that default separators were used
        call_args = mock_text_splitter_class.call_args
        assert call_args[1]['separators'] == ["\n\n", "\n", " ", ""]
    
    @patch('langchain.text_splitter.RecursiveCharacterTextSplitter')
    def test_chunk_recursive_char_semantic_coherence(self, mock_text_splitter_class):
        """Test that recursive chunking maintains semantic coherence."""
        text = "Plant metabolomics studies small molecules.\n\n" \
               "Primary metabolites include amino acids and sugars.\n\n" \
               "Secondary metabolites include flavonoids and alkaloids."
        
        expected_chunks = [
            "Plant metabolomics studies small molecules.",
            "Primary metabolites include amino acids and sugars.",
            "Secondary metabolites include flavonoids and alkaloids."
        ]
        
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = expected_chunks
        mock_text_splitter_class.return_value = mock_splitter
        
        chunks = chunk_recursive_char(text, chunk_size=60, chunk_overlap=0)
        
        # Each chunk should be semantically coherent (complete sentences/paragraphs)
        assert len(chunks) == 3
        assert all(chunk.strip().endswith('.') for chunk in chunks)
    
    @patch('langchain.text_splitter.RecursiveCharacterTextSplitter')
    def test_chunk_recursive_char_large_text(self, mock_text_splitter_class):
        """Test recursive chunking with large text."""
        # Simulate a large scientific abstract
        large_text = "Plant metabolomics research. " * 100
        
        # Mock returning multiple chunks
        expected_chunks = [large_text[i:i+200] for i in range(0, len(large_text), 200)]
        
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = expected_chunks
        mock_text_splitter_class.return_value = mock_splitter
        
        chunks = chunk_recursive_char(large_text, chunk_size=200, chunk_overlap=20)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 200 for chunk in chunks)
    
    def test_chunk_recursive_char_empty_text(self):
        """Test recursive chunking of empty text."""
        chunks = chunk_recursive_char("", chunk_size=100, chunk_overlap=0)
        assert chunks == []
    
    def test_chunk_recursive_char_very_short_text(self):
        """Test recursive chunking of very short text."""
        text = "Short"
        
        with patch('langchain.text_splitter.RecursiveCharacterTextSplitter') as mock_class:
            mock_splitter = Mock()
            mock_splitter.split_text.return_value = [text]
            mock_class.return_value = mock_splitter
            
            chunks = chunk_recursive_char(text, chunk_size=100, chunk_overlap=0)
            
            assert chunks == [text]
    
    def test_chunk_recursive_char_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        text = "Sample text"
        
        # Test negative chunk size
        with pytest.raises(ChunkingError, match="Chunk size must be positive"):
            chunk_recursive_char(text, chunk_size=-1, chunk_overlap=0)
        
        # Test negative overlap
        with pytest.raises(ChunkingError, match="Chunk overlap cannot be negative"):
            chunk_recursive_char(text, chunk_size=10, chunk_overlap=-1)
        
        # Test overlap larger than chunk size
        with pytest.raises(ChunkingError, match="Chunk overlap cannot be larger than chunk size"):
            chunk_recursive_char(text, chunk_size=10, chunk_overlap=15)
    
    def test_chunk_recursive_char_none_input(self):
        """Test error handling for None input."""
        with pytest.raises(ChunkingError, match="Input text cannot be None"):
            chunk_recursive_char(None, chunk_size=10, chunk_overlap=0)
    
    def test_chunk_recursive_char_invalid_separators(self):
        """Test error handling for invalid separators."""
        text = "Sample text"
        
        with pytest.raises(ChunkingError, match="Separators must be a list"):
            chunk_recursive_char(text, chunk_size=10, chunk_overlap=0, separators="invalid")
    
    @patch('langchain.text_splitter.RecursiveCharacterTextSplitter', side_effect=ImportError("LangChain not available"))
    def test_chunk_recursive_char_missing_dependency(self, mock_text_splitter_class):
        """Test error handling when LangChain is not available."""
        text = "Sample text"
        
        with pytest.raises(ChunkingError, match="LangChain library is required"):
            chunk_recursive_char(text, chunk_size=10, chunk_overlap=0)
    
    @pytest.mark.parametrize("chunk_size,overlap,separator_count", [
        (100, 0, 4),   # Default separators
        (50, 10, 4),   # With overlap
        (200, 20, 3),  # Custom separator count (will be mocked)
    ])
    @patch('langchain.text_splitter.RecursiveCharacterTextSplitter')
    def test_chunk_recursive_char_parametrized(self, mock_text_splitter_class, 
                                             chunk_size, overlap, separator_count,
                                             sample_scientific_text):
        """Parametrized test for recursive chunking scenarios."""
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = [sample_scientific_text[:chunk_size]]
        mock_text_splitter_class.return_value = mock_splitter
        
        chunks = chunk_recursive_char(sample_scientific_text, chunk_size, overlap)
        
        assert len(chunks) >= 1
        mock_text_splitter_class.assert_called_once()
        call_args = mock_text_splitter_class.call_args[1]
        assert call_args['chunk_size'] == chunk_size
        assert call_args['chunk_overlap'] == overlap


class TestChunkingErrorHandling:
    """Test cases for error handling and edge cases across all chunking functions."""
    
    def test_chunking_error_inheritance(self):
        """Test that ChunkingError properly inherits from Exception."""
        error = ChunkingError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
    
    def test_unicode_handling_across_functions(self):
        """Test Unicode handling across all chunking functions."""
        unicode_text = "Plant metabolömics résearch naïve approach café"
        
        # Test fixed-size chunking with Unicode
        chunks_fixed = chunk_fixed_size(unicode_text, chunk_size=20, chunk_overlap=0)
        assert any("metabolömics" in chunk for chunk in chunks_fixed)
        
        # Test sentence chunking with Unicode
        unicode_sentences = "Metabolömics research is important. Café studies are ongoing."
        chunks_sentences = chunk_by_sentences(unicode_sentences)
        assert len(chunks_sentences) == 2
        assert any("Metabolömics" in chunk for chunk in chunks_sentences)
    
    def test_memory_efficiency_large_text(self):
        """Test memory efficiency with large text inputs."""
        # Create a large text string
        large_text = "Plant metabolomics research analyzes small molecules. " * 1000
        
        # These should not raise memory errors
        chunks_fixed = chunk_fixed_size(large_text, chunk_size=200, chunk_overlap=10)
        assert len(chunks_fixed) > 0
        
        chunks_sentences = chunk_by_sentences(large_text[:1000])  # Limit for sentence test
        assert len(chunks_sentences) > 0
    
    def test_concurrent_processing_safety(self):
        """Test thread safety of chunking functions."""
        import threading
        
        results = []
        errors = []
        
        def process_text(text_id):
            try:
                text = f"Plant metabolomics research {text_id}. Analysis of metabolites is important."
                fixed_chunks = chunk_fixed_size(text, chunk_size=30, chunk_overlap=5)
                sentence_chunks = chunk_by_sentences(text)
                results.append((fixed_chunks, sentence_chunks))
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


class TestChunkingIntegrationScenarios:
    """Integration test cases combining multiple chunking approaches."""
    
    def test_chunking_comparison_consistency(self):
        """Test consistency between different chunking methods."""
        text = "Plant metabolomics is the study of small molecules. " \
               "Primary metabolites include amino acids and sugars. " \
               "Secondary metabolites include flavonoids and alkaloids."
        
        # Get chunks from different methods
        fixed_chunks = chunk_fixed_size(text, chunk_size=60, chunk_overlap=0)
        sentence_chunks = chunk_by_sentences(text)
        
        # All methods should preserve the original text content
        fixed_text = "".join(fixed_chunks)
        sentence_text = " ".join(sentence_chunks)
        
        # Content should be preserved (allowing for spacing differences)
        original_words = set(text.split())
        fixed_words = set(fixed_text.split())
        sentence_words = set(sentence_text.split())
        
        assert original_words == fixed_words
        assert original_words == sentence_words
    
    def test_chunking_for_llm_processing(self):
        """Test chunking optimized for LLM processing with realistic scientific text."""
        scientific_text = """
        Plant metabolomics is a rapidly advancing field that focuses on the comprehensive 
        analysis of small molecules (metabolites) in plant systems. These metabolites, 
        typically with molecular weights less than 1,500 Da, include both primary 
        metabolites such as amino acids, organic acids, and sugars, as well as secondary 
        metabolites like flavonoids, alkaloids, and terpenoids.
        
        The application of metabolomics in plant science has revolutionized our 
        understanding of plant physiology, stress responses, and biochemical pathways. 
        Advanced analytical techniques including liquid chromatography-mass spectrometry 
        (LC-MS) and gas chromatography-mass spectrometry (GC-MS) enable researchers 
        to identify and quantify thousands of metabolites simultaneously.
        """
        
        # Test different chunking strategies for LLM processing
        
        # Strategy 1: Fixed-size chunks for consistent processing
        llm_chunks_fixed = chunk_fixed_size(scientific_text, chunk_size=300, chunk_overlap=50)
        assert all(len(chunk) <= 300 for chunk in llm_chunks_fixed)
        assert len(llm_chunks_fixed) >= 2
        
        # Strategy 2: Sentence-based chunks for semantic coherence
        llm_chunks_sentences = chunk_by_sentences(scientific_text)
        assert all(chunk.strip().endswith('.') for chunk in llm_chunks_sentences if chunk.strip())
        
        # Both strategies should capture key scientific terms
        all_chunks_text = " ".join(llm_chunks_fixed + llm_chunks_sentences)
        key_terms = ["metabolomics", "metabolites", "flavonoids", "LC-MS", "GC-MS"]
        assert all(term in all_chunks_text for term in key_terms)
    
    @patch('langchain.text_splitter.RecursiveCharacterTextSplitter')
    def test_chunking_strategy_selection(self, mock_text_splitter_class):
        """Test selection of appropriate chunking strategy based on text characteristics."""
        mock_splitter = Mock()
        mock_text_splitter_class.return_value = mock_splitter
        
        # Short text - should prefer sentence chunking
        short_text = "Plant research is important. Metabolites are studied."
        sentence_chunks = chunk_by_sentences(short_text)
        assert len(sentence_chunks) == 2
        
        # Long text with clear structure - should prefer recursive chunking
        structured_text = "Section 1\n\nPlant metabolomics overview.\n\nSection 2\n\nAnalytical methods."
        mock_splitter.split_text.return_value = ["Section 1\n\nPlant metabolomics overview.", "Section 2\n\nAnalytical methods."]
        
        recursive_chunks = chunk_recursive_char(structured_text, chunk_size=50, chunk_overlap=10)
        mock_splitter.split_text.assert_called_once_with(structured_text)
        
        # Very long text - should prefer fixed-size chunking
        very_long_text = "Plant metabolomics research. " * 100
        fixed_chunks = chunk_fixed_size(very_long_text, chunk_size=200, chunk_overlap=20)
        assert len(fixed_chunks) > 5  # Should create multiple chunks


# Fixtures for test data
@pytest.fixture
def sample_scientific_text():
    """Fixture providing sample scientific text for testing."""
    return """
    Plant metabolomics is the comprehensive study of small molecules (metabolites) in plant systems.
    These studies involve the analysis of primary metabolites such as amino acids, organic acids,
    and sugars, as well as secondary metabolites including flavonoids, alkaloids, and terpenoids.
    
    Modern analytical techniques like liquid chromatography-mass spectrometry (LC-MS) and 
    gas chromatography-mass spectrometry (GC-MS) enable researchers to identify and quantify
    thousands of metabolites simultaneously. This comprehensive approach provides insights
    into plant physiology, stress responses, and biochemical pathways.
    
    The application of metabolomics in plant breeding and crop improvement has shown
    promising results in developing stress-resistant varieties and enhancing nutritional content.
    """


@pytest.fixture
def sample_biomedical_sentences():
    """Fixture providing sample biomedical sentences for testing."""
    return [
        "Plant metabolomics investigates small molecule profiles in plant tissues.",
        "Secondary metabolites like flavonoids provide antioxidant properties.",
        "Analytical platforms including LC-MS enable comprehensive metabolite detection.",
        "Stress responses in plants involve complex metabolic pathway alterations.",
        "Biomarker discovery through metabolomics supports crop improvement programs."
    ]


@pytest.fixture
def sample_structured_text():
    """Fixture providing sample structured text with clear separators."""
    return """Chapter 1: Introduction to Plant Metabolomics

Plant metabolomics represents a systems biology approach to understanding plant biochemistry.
This field has emerged as a powerful tool for investigating plant responses to environmental stress.

Chapter 2: Analytical Methods

Liquid chromatography-mass spectrometry (LC-MS) is the most widely used platform.
Gas chromatography-mass spectrometry (GC-MS) complements LC-MS for volatile compounds.

Chapter 3: Data Analysis

Multivariate statistical analysis helps identify significant metabolic changes.
Machine learning approaches are increasingly applied to metabolomics data interpretation."""


@pytest.fixture
def sample_large_text():
    """Fixture providing large text for performance testing."""
    base_text = """
    Plant metabolomics research focuses on the comprehensive analysis of metabolites.
    This interdisciplinary field combines analytical chemistry, bioinformatics, and plant biology.
    Advanced mass spectrometry techniques enable identification of thousands of compounds.
    Statistical analysis reveals metabolic patterns associated with specific conditions.
    """
    return (base_text * 50).strip()  # Create a large text by repetition


# Mark all tests in this module as text processing related
pytestmark = pytest.mark.unit