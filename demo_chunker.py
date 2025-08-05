#!/usr/bin/env python3
"""
Demonstration script for the AIM2-ODIE Text Processing Chunker Module.

This script demonstrates the text chunking functionality implemented in
src/text_processing/chunker.py for the AIM2-ODIE project.
"""

from src.text_processing import chunk_fixed_size, chunk_by_sentences, chunk_recursive_char, ChunkingError

def main():
    """Demonstrate text chunking functionality."""
    
    # Sample scientific text for demonstration
    scientific_text = """
    Plant metabolomics is the comprehensive study of small molecules (metabolites) in plant systems.
    These studies involve the analysis of primary metabolites such as amino acids, organic acids,
    and sugars, as well as secondary metabolites including flavonoids, alkaloids, and terpenoids.
    
    Modern analytical techniques like liquid chromatography-mass spectrometry (LC-MS) and 
    gas chromatography-mass spectrometry (GC-MS) enable researchers to identify and quantify
    thousands of metabolites simultaneously. This comprehensive approach provides insights
    into plant physiology, stress responses, and biochemical pathways.
    """
    
    print("=" * 80)
    print("AIM2-ODIE Text Processing Chunker Demonstration")
    print("=" * 80)
    
    # Demonstration 1: Fixed-size chunking with characters
    print("\n1. Fixed-Size Chunking (Character-based)")
    print("-" * 50)
    chunks = chunk_fixed_size(scientific_text.strip(), chunk_size=150, chunk_overlap=20)
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i} ({len(chunk)} chars): {chunk[:60]}...")
    
    # Demonstration 2: Fixed-size chunking with words
    print("\n2. Fixed-Size Chunking (Word-based)")
    print("-" * 50)
    chunks = chunk_fixed_size(scientific_text.strip(), chunk_size=25, chunk_overlap=5, unit='words')
    for i, chunk in enumerate(chunks, 1):
        word_count = len(chunk.split())
        print(f"Chunk {i} ({word_count} words): {chunk[:60]}...")
    
    # Demonstration 3: Sentence-based chunking
    print("\n3. Sentence-Based Chunking")
    print("-" * 50)
    sentences = chunk_by_sentences(scientific_text.strip())
    for i, sentence in enumerate(sentences, 1):
        print(f"Sentence {i}: {sentence}")
    
    # Demonstration 4: Recursive character chunking
    print("\n4. Recursive Character Chunking (LangChain)")
    print("-" * 50)
    try:
        structured_text = """Section 1: Introduction

Plant metabolomics research focuses on small molecules.

Section 2: Methods

Advanced analytical techniques are used.

Section 3: Results

Thousands of metabolites can be identified."""
        
        chunks = chunk_recursive_char(structured_text, chunk_size=80, chunk_overlap=10)
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i}: {chunk.strip()}")
    except ChunkingError as e:
        print(f"Note: {e}")
    
    # Demonstration 5: Error handling
    print("\n5. Error Handling Demonstration")
    print("-" * 50)
    try:
        chunk_fixed_size(None, 10, 0)
    except ChunkingError as e:
        print(f"✓ Correctly caught error: {e}")
    
    try:
        chunk_fixed_size("test", -5, 0)
    except ChunkingError as e:
        print(f"✓ Correctly caught error: {e}")
    
    print("\n" + "=" * 80)
    print("All chunking functionality demonstrated successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()