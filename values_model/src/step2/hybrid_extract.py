"""
Hybrid Value Extraction Module

This module implements a robust, deterministic approach to value statement extraction
that combines traditional NLP techniques with optional LLM validation.

Phase 1 Implementation:
1. Deterministic NLP extraction (always consistent)
2. Sentiment analysis for implicit values
3. Optional LLM validation (with deterministic fallback)
4. Confidence scoring and quality assessment
"""

import spacy
import json
import os
from typing import List, Dict, Set, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model en_core_web_sm...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Enhanced keyword sets for value detection
VALUE_KEYWORDS = {
    'moral': {
        'positive': ['good', 'virtue', 'virtuous', 'right', 'moral', 'ethical', 'noble', 'honorable'],
        'negative': ['bad', 'vice', 'vicious', 'wrong', 'immoral', 'unethical', 'base', 'dishonorable']
    },
    'practical': {
        'positive': ['should', 'ought', 'must', 'wise', 'prudent', 'sensible', 'advisable'],
        'negative': ['shouldn\'t', 'oughtn\'t', 'mustn\'t', 'unwise', 'imprudent', 'foolish', 'inadvisable']
    },
    'character': {
        'positive': ['brave', 'courageous', 'honest', 'truthful', 'strong', 'resilient', 'temperate'],
        'negative': ['cowardly', 'dishonest', 'deceitful', 'weak', 'fragile', 'intemperate', 'wretched']
    },
    'aesthetic': {
        'positive': ['beautiful', 'harmonious', 'elegant', 'graceful', 'noble'],
        'negative': ['ugly', 'disharmonious', 'clumsy', 'graceless', 'base']
    },
    'social': {
        'positive': ['proper', 'appropriate', 'fitting', 'decent', 'respectable'],
        'negative': ['improper', 'inappropriate', 'unfitting', 'indecent', 'disrespectable']
    }
}

# Expanded ignored concepts
IGNORED_CONCEPTS = {
    "you", "he", "she", "it", "this", "these", "i", "we", "they", "that", "which",
    "him", "her", "them", "himself", "herself", "themselves", "yourself", "myself"
}

class HybridValueExtractor:
    """
    A hybrid value extraction system that combines deterministic NLP with optional LLM validation.
    """
    
    def __init__(self, use_sentiment: bool = True, use_llm: bool = False):
        self.use_sentiment = use_sentiment
        self.use_llm = use_llm
        
        # Initialize sentiment analyzers if requested
        self.vader_analyzer = None
        if use_sentiment:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.vader_analyzer = SentimentIntensityAnalyzer()
            except ImportError:
                print("Warning: VADER sentiment not available. Install with: pip install vaderSentiment")
        
        # Initialize TextBlob if available
        self.textblob_available = False
        if use_sentiment:
            try:
                from textblob import TextBlob
                self.textblob_available = True
            except ImportError:
                print("Warning: TextBlob not available. Install with: pip install textblob")
    
    def extract_values(self, sentences: List[str]) -> List[Dict]:
        """
        Main extraction method that combines multiple approaches.
        
        Args:
            sentences: List of sentences to analyze
            
        Returns:
            List of extracted value statements with metadata
        """
        print(f"Extracting values from {len(sentences)} sentences...")
        
        # Stage 1: Keyword-based extraction (deterministic)
        keyword_results = self._extract_keyword_values(sentences)
        print(f"Keyword extraction found {len(keyword_results)} statements")
        
        # Stage 2: Sentiment-based extraction (deterministic)
        sentiment_results = []
        if self.use_sentiment:
            sentiment_results = self._extract_sentiment_values(sentences)
            print(f"Sentiment extraction found {len(sentiment_results)} statements")
        
        # Stage 3: Combine and deduplicate results
        combined_results = self._combine_results(keyword_results, sentiment_results, sentences)
        print(f"Combined extraction found {len(combined_results)} unique statements")
        
        # Stage 4: Optional LLM validation
        if self.use_llm:
            try:
                validated_results = self._validate_with_llm(combined_results, sentences)
                print(f"LLM validation produced {len(validated_results)} validated statements")
                return validated_results
            except Exception as e:
                print(f"LLM validation failed: {e}")
                print("Falling back to deterministic validation...")
        
        # Stage 5: Deterministic validation and confidence scoring
        final_results = self._deterministic_validation(combined_results)
        print(f"Final results: {len(final_results)} high-quality statements")
        
        return final_results
    
    def _extract_keyword_values(self, sentences: List[str]) -> List[Dict]:
        """
        Extract values using enhanced keyword detection and NLP analysis.
        """
        value_statements = []
        
        for i, sentence in enumerate(sentences):
            # Check for any value keywords
            has_keywords, keyword_info = self._detect_keywords(sentence)
            if not has_keywords:
                continue
            
            # Analyze sentence structure
            doc = nlp(sentence)
            
            # Extract concepts using multiple strategies
            concepts = self._extract_concepts_advanced(doc, keyword_info)
            
            if concepts:
                for concept_info in concepts:
                    value_statements.append({
                        'sentence_index': i,
                        'sentence': sentence,
                        'value_object': concept_info['concept'],
                        'evaluation': concept_info['evaluation'],
                        'value_type': concept_info['value_type'],
                        'keyword': keyword_info['keyword'],
                        'confidence': concept_info['confidence'],
                        'extraction_method': 'keyword_nlp',
                        'reasoning': concept_info['reasoning']
                    })
        
        return value_statements
    
    def _extract_sentiment_values(self, sentences: List[str]) -> List[Dict]:
        """
        Extract values using sentiment analysis for implicit value statements.
        """
        value_statements = []
        
        for i, sentence in enumerate(sentences):
            sentiment_scores = self._analyze_sentiment(sentence)
            
            # Only consider sentences with strong sentiment
            if abs(sentiment_scores['compound']) > 0.5:
                doc = nlp(sentence)
                concepts = self._extract_sentiment_concepts(doc, sentiment_scores)
                
                for concept_info in concepts:
                    value_statements.append({
                        'sentence_index': i,
                        'sentence': sentence,
                        'value_object': concept_info['concept'],
                        'evaluation': concept_info['evaluation'],
                        'value_type': 'sentiment_implicit',
                        'sentiment_scores': sentiment_scores,
                        'confidence': concept_info['confidence'],
                        'extraction_method': 'sentiment_analysis',
                        'reasoning': concept_info['reasoning']
                    })
        
        return value_statements
    
    def _detect_keywords(self, sentence: str) -> Tuple[bool, Dict]:
        """
        Detect if sentence contains value keywords and return metadata.
        """
        sentence_lower = sentence.lower()
        
        for category, polarity_dict in VALUE_KEYWORDS.items():
            for polarity, keywords in polarity_dict.items():
                for keyword in keywords:
                    if f' {keyword} ' in f' {sentence_lower} ':
                        return True, {
                            'keyword': keyword,
                            'category': category,
                            'polarity': polarity,
                            'evaluation': 'positive' if polarity == 'positive' else 'negative'
                        }
        
        return False, {}
    
    def _extract_concepts_advanced(self, doc, keyword_info: Dict) -> List[Dict]:
        """
        Advanced concept extraction using multiple NLP strategies.
        """
        concepts = []
        
        # Strategy 1: Extract subjects of value statements
        subjects = self._extract_subjects(doc, keyword_info)
        for subject in subjects:
            concepts.append({
                'concept': subject,
                'evaluation': keyword_info['evaluation'],
                'value_type': keyword_info['category'],
                'confidence': 0.8,
                'reasoning': f"Subject of {keyword_info['keyword']} statement"
            })
        
        # Strategy 2: Extract verb phrases for "should" statements
        if keyword_info['keyword'] == 'should':
            verb_phrases = self._extract_verb_phrases(doc)
            for phrase in verb_phrases:
                concepts.append({
                    'concept': phrase,
                    'evaluation': 'positive',
                    'value_type': 'practical',
                    'confidence': 0.9,
                    'reasoning': "Verb phrase following 'should'"
                })
        
        # Strategy 3: Extract objects and complements
        objects = self._extract_objects(doc, keyword_info)
        for obj in objects:
            concepts.append({
                'concept': obj,
                'evaluation': keyword_info['evaluation'],
                'value_type': keyword_info['category'],
                'confidence': 0.7,
                'reasoning': f"Object of {keyword_info['keyword']} statement"
            })
        
        # Filter out ignored concepts
        concepts = [c for c in concepts if c['concept'].lower().strip() not in IGNORED_CONCEPTS]
        
        return concepts
    
    def _extract_subjects(self, doc, keyword_info: Dict) -> List[str]:
        """Extract grammatical subjects from the sentence."""
        subjects = []
        
        for token in doc:
            if token.dep_ in ['nsubj', 'nsubjpass', 'csubj']:
                subject_text = ' '.join([t.text for t in token.subtree])
                subjects.append(subject_text.strip())
        
        return subjects
    
    def _extract_verb_phrases(self, doc) -> List[str]:
        """Extract verb phrases, especially for 'should' statements."""
        verb_phrases = []
        
        for token in doc:
            if token.text.lower() == 'should' and token.dep_ == 'aux':
                # Get the main verb and its dependents
                main_verb = token.head
                phrase_tokens = [main_verb]
                
                # Add verb dependents
                for child in main_verb.children:
                    if child.dep_ in ['dobj', 'pobj', 'attr', 'acomp']:
                        phrase_tokens.extend([t for t in child.subtree])
                
                verb_phrase = ' '.join([t.text for t in sorted(phrase_tokens, key=lambda x: x.i)])
                verb_phrases.append(verb_phrase.strip())
        
        return verb_phrases
    
    def _extract_objects(self, doc, keyword_info: Dict) -> List[str]:
        """Extract objects and complements from the sentence."""
        objects = []
        
        for token in doc:
            if token.dep_ in ['dobj', 'pobj', 'attr', 'acomp']:
                obj_text = ' '.join([t.text for t in token.subtree])
                objects.append(obj_text.strip())
        
        return objects
    
    def _analyze_sentiment(self, sentence: str) -> Dict:
        """
        Analyze sentiment using multiple tools for robust scoring.
        """
        sentiment_scores = {}
        
        # VADER sentiment
        if self.vader_analyzer:
            sentiment_scores['vader'] = self.vader_analyzer.polarity_scores(sentence)
        
        # TextBlob sentiment
        if self.textblob_available:
            from textblob import TextBlob
            blob = TextBlob(sentence)
            sentiment_scores['textblob'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        
        # Calculate compound score
        compound_score = 0.0
        if 'vader' in sentiment_scores:
            compound_score = sentiment_scores['vader']['compound']
        elif 'textblob' in sentiment_scores:
            compound_score = sentiment_scores['textblob']['polarity']
        
        sentiment_scores['compound'] = compound_score
        sentiment_scores['evaluation'] = 'positive' if compound_score > 0 else 'negative' if compound_score < 0 else 'neutral'
        
        return sentiment_scores
    
    def _extract_sentiment_concepts(self, doc, sentiment_scores: Dict) -> List[Dict]:
        """
        Extract concepts from sentences identified through sentiment analysis.
        """
        concepts = []
        
        # Extract main subjects and objects
        for token in doc:
            if token.dep_ in ['nsubj', 'nsubjpass', 'dobj', 'pobj']:
                concept_text = ' '.join([t.text for t in token.subtree])
                if concept_text.lower().strip() not in IGNORED_CONCEPTS:
                    concepts.append({
                        'concept': concept_text.strip(),
                        'evaluation': sentiment_scores['evaluation'],
                        'confidence': min(abs(sentiment_scores['compound']) * 2, 0.9),
                        'reasoning': f"Concept identified through sentiment analysis (compound: {sentiment_scores['compound']:.2f})"
                    })
        
        return concepts
    
    def _combine_results(self, keyword_results: List[Dict], sentiment_results: List[Dict], sentences: List[str]) -> List[Dict]:
        """
        Combine results from different extraction methods and deduplicate.
        """
        all_results = keyword_results + sentiment_results
        
        # Deduplicate based on sentence index and similar concepts
        unique_results = []
        seen_combinations = set()
        
        for result in all_results:
            # Create a key for deduplication
            key = (result['sentence_index'], result['value_object'].lower().strip())
            
            if key not in seen_combinations:
                seen_combinations.add(key)
                unique_results.append(result)
        
        return unique_results
    
    def _validate_with_llm(self, results: List[Dict], sentences: List[str]) -> List[Dict]:
        """
        Optional LLM validation with deterministic fallback.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("No OpenAI API key found. Skipping LLM validation.")
            return self._deterministic_validation(results)
        
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            
            # Create validation prompt
            prompt = self._create_validation_prompt(results, sentences)
            
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Maximum determinism
                response_format={"type": "json_object"},
                max_tokens=3000
            )
            
            validated_data = json.loads(response.choices[0].message.content)
            return validated_data.get('validated_statements', results)
            
        except Exception as e:
            print(f"LLM validation failed: {e}")
            return self._deterministic_validation(results)
    
    def _create_validation_prompt(self, results: List[Dict], sentences: List[str]) -> str:
        """
        Create a prompt for LLM validation of extracted value statements.
        """
        prompt = f"""You are an expert in moral philosophy. Validate and refine these extracted value statements.

For each statement, assess:
1. Is this truly a value statement?
2. Is the value_object accurately extracted?
3. Is the evaluation correct?
4. What is the appropriate value_type?

Return a JSON object with this structure:
{{
    "validated_statements": [
        {{
            "sentence_index": [number],
            "sentence": "[original sentence]",
            "value_object": "[refined value object]",
            "evaluation": "positive|negative|neutral",
            "value_type": "moral|practical|character|aesthetic|social",
            "confidence": [0.0-1.0],
            "validation_status": "validated|refined|rejected",
            "reasoning": "[explanation]"
        }}
    ]
}}

EXTRACTED STATEMENTS TO VALIDATE:
{json.dumps(results, indent=2)}

ORIGINAL SENTENCES:
{chr(10).join([f"{i}: {sent}" for i, sent in enumerate(sentences)])}

Return ONLY the JSON object:"""
        
        return prompt
    
    def _deterministic_validation(self, results: List[Dict]) -> List[Dict]:
        """
        Deterministic validation using rule-based methods.
        """
        validated = []
        
        for result in results:
            # Apply validation rules
            confidence = result.get('confidence', 0.5)
            value_object = result.get('value_object', '')
            
            # Rule 1: Minimum confidence threshold
            if confidence < 0.3:
                result['validation_status'] = 'rejected'
                result['reasoning'] = f"Confidence too low: {confidence:.2f}"
                continue
            
            # Rule 2: Value object quality check
            if len(value_object.strip()) < 3:
                result['validation_status'] = 'rejected'
                result['reasoning'] = "Value object too short"
                continue
            
            # Rule 3: Check for meaningful content
            if value_object.lower().strip() in IGNORED_CONCEPTS:
                result['validation_status'] = 'rejected'
                result['reasoning'] = "Value object is ignored concept"
                continue
            
            # Accept the result
            result['validation_status'] = 'validated'
            result['reasoning'] = f"Passed validation checks (confidence: {confidence:.2f})"
            validated.append(result)
        
        return validated


def extract_values_hybrid(sentences: List[str], use_sentiment: bool = True, use_llm: bool = False) -> List[Dict]:
    """
    Convenience function for hybrid value extraction.
    
    Args:
        sentences: List of sentences to analyze
        use_sentiment: Whether to use sentiment analysis
        use_llm: Whether to use LLM validation (requires API key)
        
    Returns:
        List of extracted and validated value statements
    """
    extractor = HybridValueExtractor(use_sentiment=use_sentiment, use_llm=use_llm)
    return extractor.extract_values(sentences)


def compare_extraction_methods(sentences: List[str]) -> Dict:
    """
    Compare different extraction methods for evaluation purposes.
    """
    # Original keyword-based extraction
    from .extract import extract_values as extract_values_original
    
    original_results = extract_values_original(sentences)
    
    # Hybrid extraction without sentiment
    hybrid_no_sentiment = extract_values_hybrid(sentences, use_sentiment=False, use_llm=False)
    
    # Hybrid extraction with sentiment
    hybrid_with_sentiment = extract_values_hybrid(sentences, use_sentiment=True, use_llm=False)
    
    return {
        'original_keyword_method': {
            'count': len(original_results),
            'statements': original_results
        },
        'hybrid_no_sentiment': {
            'count': len(hybrid_no_sentiment),
            'statements': hybrid_no_sentiment
        },
        'hybrid_with_sentiment': {
            'count': len(hybrid_with_sentiment),
            'statements': hybrid_with_sentiment
        },
        'comparison_summary': {
            'original_count': len(original_results),
            'hybrid_no_sentiment_count': len(hybrid_no_sentiment),
            'hybrid_with_sentiment_count': len(hybrid_with_sentiment),
            'improvement_with_sentiment': len(hybrid_with_sentiment) - len(original_results)
        }
    }
