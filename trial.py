from values_model.src.phase1.ingest import ingest_corpus
from values_model.src.phase1.clean import clean_text
from values_model.src.phase1.segment import segment_text
from values_model.src.phase1.hybrid_extract import extract_values_hybrid, compare_extraction_methods
from values_model.src.phase1.resolve import resolve_concepts_with_llm
import os
import argparse
import json

def main():
    """
    Runs the improved Phase 1 pipeline with hybrid value extraction.
    
    This pipeline implements:
    1. Deterministic NLP extraction with enhanced keyword detection
    2. Sentiment analysis for implicit value statements  
    3. Optional LLM validation (with deterministic fallback)
    4. Confidence scoring and quality assessment
    """
    parser = argparse.ArgumentParser(description="Extract and resolve value statements from a corpus file using hybrid approach.")
    parser.add_argument("filename", help="The name of the file in the data/raw/ directory.")
    parser.add_argument("--use-sentiment", action="store_true", default=True, 
                       help="Use sentiment analysis for implicit value detection (default: True)")
    parser.add_argument("--use-llm", action="store_true", default=False,
                       help="Use LLM validation (requires OPENAI_API_KEY)")
    parser.add_argument("--compare-methods", action="store_true", default=False,
                       help="Compare different extraction methods and save comparison results")
    args = parser.parse_args()

    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_path = os.path.join(base_dir, "values_model", "data", "raw", args.filename)
    extracted_data_dir = os.path.join(base_dir, "values_model", "data", "extracted")
    os.makedirs(extracted_data_dir, exist_ok=True)
    
    # Create output filenames
    base_filename = os.path.splitext(args.filename)[0]
    output_filename = f"{base_filename}_values_hybrid.json"
    output_file_path = os.path.join(extracted_data_dir, output_filename)
    
    if args.compare_methods:
        comparison_filename = f"{base_filename}_extraction_comparison.json"
        comparison_file_path = os.path.join(extracted_data_dir, comparison_filename)

    print(f"--- Starting Improved Phase 1 Pipeline for {args.filename} ---")
    print(f"Configuration: sentiment={args.use_sentiment}, llm={args.use_llm}, compare={args.compare_methods}")

    try:
        # Step 1: Ingest, Clean, Segment
        print("\n1. Processing text...")
        raw_text = ingest_corpus(input_file_path)
        cleaned_text = clean_text(raw_text)
        sentences = segment_text(cleaned_text)
        print(f"   ...processed {len(sentences)} sentences.")

        # Step 2: Hybrid Value Extraction
        print("\n2. Performing hybrid value extraction...")
        if args.compare_methods:
            print("   ...comparing extraction methods...")
            comparison_results = compare_extraction_methods(sentences)
            
            # Save comparison results
            with open(comparison_file_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, indent=4)
            print(f"   ...comparison results saved to {comparison_file_path}")
            
            # Use the best method (hybrid with sentiment) for main results
            extracted_statements = comparison_results['hybrid_with_sentiment']['statements']
        else:
            extracted_statements = extract_values_hybrid(
                sentences, 
                use_sentiment=args.use_sentiment, 
                use_llm=args.use_llm
            )
        
        print(f"   ...extracted {len(extracted_statements)} value statements.")

        # Step 3: Optional Legacy LLM Resolution (for compatibility)
        if args.use_llm:
            print("\n3. Applying legacy LLM resolution...")
            # Convert to legacy format for compatibility
            legacy_statements = []
            for stmt in extracted_statements:
                legacy_stmt = {
                    'sentence_index': stmt['sentence_index'],
                    'concept': stmt['value_object'],
                    'evaluation': stmt['evaluation'],
                    'keyword': stmt.get('keyword', ''),
                    'sentence': stmt['sentence']
                }
                legacy_statements.append(legacy_stmt)
            
            resolved_statements = resolve_concepts_with_llm(legacy_statements, sentences)
            print("   ...legacy LLM resolution completed.")
        else:
            resolved_statements = extracted_statements
            print("\n3. Skipping legacy LLM resolution (using hybrid results directly).")

        # Step 4: Save final results
        print(f"\n4. Saving results to {output_file_path}...")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(resolved_statements, f, indent=4)
        
        # Step 5: Summary and quality assessment
        print("\n--- Pipeline Finished ---")
        print(f"Final Results Summary:")
        print(f"  - Total sentences processed: {len(sentences)}")
        print(f"  - Value statements extracted: {len(resolved_statements)}")
        print(f"  - Extraction rate: {len(resolved_statements)/len(sentences)*100:.1f}%")
        
        if resolved_statements:
            # Calculate confidence statistics
            confidences = [stmt.get('confidence', 0.5) for stmt in resolved_statements]
            avg_confidence = sum(confidences) / len(confidences)
            print(f"  - Average confidence: {avg_confidence:.2f}")
            
            # Count by value type
            value_types = {}
            for stmt in resolved_statements:
                vtype = stmt.get('value_type', 'unknown')
                value_types[vtype] = value_types.get(vtype, 0) + 1
            
            print(f"  - Value types found: {dict(value_types)}")
        
        print(f"\nResults saved to: {output_file_path}")
        if args.compare_methods:
            print(f"Comparison saved to: {comparison_file_path}")

    except FileNotFoundError:
        print(f"ERROR: The file '{input_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()