#!/usr/bin/env python3
"""
Test script for the refactored Phase 1 pipeline.

This script tests the new hybrid extraction approach and compares it with the original method.
"""

import os
import sys
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from values_model.src.phase1.hybrid_extract import extract_values_hybrid, compare_extraction_methods
from values_model.src.phase1.extract import extract_values as extract_values_original

def test_hybrid_extraction():
    """Test the hybrid extraction on sample sentences."""
    
    # Sample sentences from Epictetus for testing
    test_sentences = [
        "Some things are in our control and others not.",
        "The things in our control are by nature free, unrestrained, unhindered.",
        "But those not in our control are weak, slavish, restrained, belonging to others.",
        "Remember, then, that if you suppose that things which are slavish by nature are also free, then you will be hindered.",
        "You will lament, you will be disturbed, and you will find fault both with gods and men.",
        "But if you suppose that only to be your own which is your own, then no one will ever compel you or restrain you.",
        "You will do nothing against your will. No one will hurt you, you will have no enemies, and you will not be harmed.",
        "Aiming therefore at such great things, remember that you must not allow yourself to be carried, even with a slight tendency, towards the attainment of lesser things.",
        "But if you would both have these great things, along with power and riches, then you will not gain even the latter.",
        "You will absolutely fail of the former, by which alone happiness and freedom are achieved.",
        "An uninstructed person will lay the fault of his own bad condition upon others.",
        "Someone else will hinder or hurt you, when he acts according to his own proper character.",
        "But if it is his pleasure you should act a poor man, a cripple, a governor, or a private person, see that you act it naturally.",
        "When, therefore, you see anyone eminent in honors, or power, or in high esteem on any other account, take heed not to be hurried away with the appearance, and to pronounce him happy.",
        "For, if the essence of good consists in things in our own control, there will be no room for envy or emulation."
    ]
    
    print("=== Testing Hybrid Value Extraction ===\n")
    
    # Test 1: Original method
    print("1. Testing original keyword-based extraction...")
    original_results = extract_values_original(test_sentences)
    print(f"   Found {len(original_results)} statements")
    
    # Test 2: Hybrid without sentiment
    print("\n2. Testing hybrid extraction without sentiment...")
    hybrid_no_sentiment = extract_values_hybrid(test_sentences, use_sentiment=False, use_llm=False)
    print(f"   Found {len(hybrid_no_sentiment)} statements")
    
    # Test 3: Hybrid with sentiment
    print("\n3. Testing hybrid extraction with sentiment...")
    hybrid_with_sentiment = extract_values_hybrid(test_sentences, use_sentiment=True, use_llm=False)
    print(f"   Found {len(hybrid_with_sentiment)} statements")
    
    # Test 4: Full comparison
    print("\n4. Running full comparison...")
    comparison = compare_extraction_methods(test_sentences)
    
    # Display results
    print("\n=== EXTRACTION COMPARISON RESULTS ===")
    print(f"Original keyword method: {comparison['original_keyword_method']['count']} statements")
    print(f"Hybrid no sentiment: {comparison['hybrid_no_sentiment']['count']} statements")
    print(f"Hybrid with sentiment: {comparison['hybrid_with_sentiment']['count']} statements")
    
    improvement = comparison['comparison_summary']['improvement_with_sentiment']
    print(f"Improvement with sentiment: {improvement:+d} statements")
    
    # Show some example extractions
    print("\n=== SAMPLE EXTRACTIONS ===")
    if hybrid_with_sentiment:
        print("Sample hybrid extractions:")
        for i, stmt in enumerate(hybrid_with_sentiment[:3]):
            print(f"\n{i+1}. Sentence: {stmt['sentence'][:80]}...")
            print(f"   Value Object: {stmt['value_object']}")
            print(f"   Evaluation: {stmt['evaluation']}")
            print(f"   Value Type: {stmt['value_type']}")
            print(f"   Confidence: {stmt['confidence']:.2f}")
            print(f"   Method: {stmt['extraction_method']}")
    
    return comparison

def test_deterministic_behavior():
    """Test that the extraction is deterministic (same input = same output)."""
    print("\n=== Testing Deterministic Behavior ===")
    
    test_sentence = "You should be honest and virtuous in all your actions."
    
    # Run extraction multiple times
    results1 = extract_values_hybrid([test_sentence], use_sentiment=True, use_llm=False)
    results2 = extract_values_hybrid([test_sentence], use_sentiment=True, use_llm=False)
    results3 = extract_values_hybrid([test_sentence], use_sentiment=True, use_llm=False)
    
    # Check if results are identical
    is_deterministic = (results1 == results2 == results3)
    
    print(f"Test sentence: {test_sentence}")
    print(f"Run 1 results: {len(results1)} statements")
    print(f"Run 2 results: {len(results2)} statements")
    print(f"Run 3 results: {len(results3)} statements")
    print(f"Deterministic: {'✓ YES' if is_deterministic else '✗ NO'}")
    
    if not is_deterministic:
        print("WARNING: Extraction is not deterministic!")
    
    return is_deterministic

def save_test_results(comparison_results):
    """Save test results to a file."""
    output_file = project_root / "values_model" / "data" / "extracted" / "test_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=4)
    
    print(f"\nTest results saved to: {output_file}")

def main():
    """Run all tests."""
    print("Phase 1 Refactored Pipeline Test Suite")
    print("=" * 50)
    
    try:
        # Test hybrid extraction
        comparison_results = test_hybrid_extraction()
        
        # Test deterministic behavior
        is_deterministic = test_deterministic_behavior()
        
        # Save results
        save_test_results(comparison_results)
        
        # Summary
        print("\n=== TEST SUMMARY ===")
        print(f"✓ Hybrid extraction working")
        print(f"{'✓' if is_deterministic else '✗'} Deterministic behavior")
        print(f"✓ Results saved")
        
        if not is_deterministic:
            print("\n⚠️  WARNING: Extraction is not deterministic!")
            print("   This may indicate issues with the implementation.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
