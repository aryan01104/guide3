#!/usr/bin/env python3
"""
Test script to see if enhanced NLP can capture complex ethical statements
like the Epictetus example.
"""

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load models
nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()

def analyze_epictetus_sentence():
    """Analyze the problematic Epictetus sentence with enhanced NLP."""
    
    sentence = "aiming therefore at such great things, remember that you must not allow yourself to be carried, even with a slight tendency, towards the attainment of lesser things."
    
    print("=== ANALYZING EPICTETUS SENTENCE ===")
    print(f"Sentence: {sentence}")
    print()
    
    # Parse with spaCy
    doc = nlp(sentence)
    
    print("=== SPACY ANALYSIS ===")
    print("Tokens and dependencies:")
    for token in doc:
        print(f"  {token.text:15} | {token.dep_:15} | {token.head.text:15} | {token.pos_:10}")
    
    print("\n=== NEGATION DETECTION ===")
    negations = []
    for token in doc:
        if token.dep_ == 'neg' or token.text.lower() in ['not', 'never', 'no']:
            negations.append({
                'token': token.text,
                'head': token.head.text,
                'scope': [t.text for t in token.subtree]
            })
    
    print("Negations found:")
    for neg in negations:
        print(f"  - '{neg['token']}' modifies '{neg['head']}'")
        print(f"    Scope: {' '.join(neg['scope'])}")
    
    print("\n=== SENTIMENT ANALYSIS ===")
    sentiment = vader.polarity_scores(sentence)
    print(f"VADER scores: {sentiment}")
    
    print("\n=== VALUE OBJECT IDENTIFICATION ===")
    
    # Find the main verb
    main_verb = None
    for token in doc:
        if token.dep_ == 'ROOT':
            main_verb = token
            break
    
    if main_verb:
        print(f"Main verb: '{main_verb.text}'")
        
        # Find what the main verb is about
        verb_objects = []
        for child in main_verb.children:
            if child.dep_ in ['dobj', 'pobj', 'prep']:
                obj_text = ' '.join([t.text for t in child.subtree])
                verb_objects.append(obj_text)
        
        print(f"Verb objects: {verb_objects}")
        
        # Look for "towards" preposition
        towards_phrase = None
        for token in doc:
            if token.text.lower() == 'towards':
                towards_phrase = ' '.join([t.text for t in token.subtree])
                break
        
        if towards_phrase:
            print(f"Towards phrase: '{towards_phrase}'")
    
    print("\n=== ENHANCED VALUE EXTRACTION ATTEMPT ===")
    
    # Try to extract the actual value statement
    value_objects = []
    
    # Method 1: Look for negated actions
    for token in doc:
        if token.dep_ == 'ROOT' and token.text == 'allow':
            # This is about not allowing something
            for child in token.children:
                if child.dep_ == 'pobj' and child.text == 'yourself':
                    # Find what you shouldn't allow yourself to do
                    for grandchild in child.children:
                        if grandchild.dep_ == 'acl':
                            action = ' '.join([t.text for t in grandchild.subtree])
                            value_objects.append({
                                'value_object': f'yourself {action}',
                                'evaluation': 'negative',
                                'reasoning': 'Negated action in main clause'
                            })
    
    # Method 2: Look for "towards" phrases that are negated
    for token in doc:
        if token.text.lower() == 'towards':
            towards_obj = ' '.join([t.text for t in token.subtree])
            # Check if this is in a negated context
            for ancestor in token.ancestors:
                if ancestor.text.lower() == 'not':
                    value_objects.append({
                        'value_object': f'being carried {towards_obj}',
                        'evaluation': 'negative', 
                        'reasoning': 'Negated prepositional phrase'
                    })
                    break
    
    print("Extracted value objects:")
    for i, obj in enumerate(value_objects, 1):
        print(f"  {i}. {obj['value_object']}")
        print(f"     Evaluation: {obj['evaluation']}")
        print(f"     Reasoning: {obj['reasoning']}")
    
    return value_objects

def test_other_ethical_sentences():
    """Test other complex ethical sentences."""
    
    test_sentences = [
        "You should not be carried away by the attainment of lesser things.",
        "It is better to die with hunger, exempt from grief and fear, than to live in affluence with perturbation.",
        "An uninstructed person will lay the fault of his own bad condition upon others.",
        "You must not allow yourself to be carried towards the attainment of lesser things."
    ]
    
    print("\n=== TESTING OTHER ETHICAL SENTENCES ===")
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i}. {sentence}")
        
        doc = nlp(sentence)
        sentiment = vader.polarity_scores(sentence)
        
        print(f"   Sentiment: {sentiment['compound']:.2f}")
        
        # Simple value extraction attempt
        negations = [token for token in doc if token.dep_ == 'neg' or token.text.lower() in ['not', 'never']]
        if negations:
            print(f"   Negations: {[n.text for n in negations]}")
        
        # Look for value keywords
        value_keywords = ['should', 'must', 'better', 'good', 'bad', 'virtue', 'vice']
        found_keywords = [token.text for token in doc if token.text.lower() in value_keywords]
        if found_keywords:
            print(f"   Value keywords: {found_keywords}")

if __name__ == "__main__":
    value_objects = analyze_epictetus_sentence()
    test_other_ethical_sentences()
    
    print("\n=== CONCLUSION ===")
    if value_objects:
        print("✅ Enhanced NLP CAN extract the correct value statement!")
        print("   The real value is about avoiding being carried towards lesser things.")
    else:
        print("❌ Even enhanced NLP struggles with complex ethical reasoning.")
        print("   This supports the need for LLM-based semantic understanding.")
