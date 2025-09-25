import spacy
from typing import List, Dict, Set

nlp = spacy.load("en_core_web_sm")

POSITIVE_KEYWORDS: Set[str] = {
    "good", "virtue", "desirable", "right", "should",
    "free", "unrestrained", "unhindered", "happiness", "tranquility", "attainment"
}

NEGATIVE_KEYWORDS: Set[str] = {
    "bad", "vice", "harmful", "wrong", "blameworthy",
    "hindered", "lament", "disturbed", "weak", "slavish", "wretched", "fail", "harmed"
}

SHOULD_KEYWORD = "should"

IGNORED_CONCEPTS: Set[str] = {
    "you", "he", "it", "this", "these", "i", "we", "they", "that", "which"
}

ALL_KEYWORDS = POSITIVE_KEYWORDS | NEGATIVE_KEYWORDS | {SHOULD_KEYWORD}

def get_verb_phrase(verb_token):
    tokens = []
    for child in verb_token.subtree:
        if child.dep_ == 'mark' and child.i > verb_token.i:
            break
        tokens.append(child.text)
    return ' '.join(tokens)

def is_in_conditional_clause(token):
    for ancestor in token.ancestors:
        for child in ancestor.children:
            if child.dep_ == 'mark' and child.text.lower() == 'if':
                return True
    return False

def extract_values(sentences: List[str]) -> List[Dict]:
    value_statements = []
    for i, sentence in enumerate(sentences):
        if not any(f' {kw} ' in f' {sentence} ' for kw in ALL_KEYWORDS):
            continue

        doc = nlp(sentence)
        root = next((token for token in doc if token.dep_ == 'ROOT'), None)
        if not root:
            continue

        keyword_token = None
        for token in doc:
            if token.text in ALL_KEYWORDS:
                keyword_token = token
                break
        if not keyword_token:
            continue

        if keyword_token.text == SHOULD_KEYWORD and is_in_conditional_clause(keyword_token):
            continue

        evaluation = "positive" if keyword_token.text in (POSITIVE_KEYWORDS | {SHOULD_KEYWORD}) else "negative"

        concept = None
        if keyword_token.text == SHOULD_KEYWORD and keyword_token.dep_ == 'aux':
            concept = get_verb_phrase(keyword_token.head)
        else:
            causal_clause = next((c for c in root.children if c.dep_ == 'advcl'), None)
            if causal_clause and keyword_token in root.subtree and keyword_token not in causal_clause.subtree:
                concept = ' '.join(t.text for t in causal_clause.subtree)
            else:
                subject = next((t for t in root.children if t.dep_ in ('nsubj', 'nsubjpass', 'csubj')), None)
                if subject:
                    concept = ' '.join(t.text for t in subject.subtree)

        if concept and concept.lower().strip() not in IGNORED_CONCEPTS:
            value_statements.append({
                "sentence_index": i, # Add the index of the sentence
                "concept": concept.strip(),
                "evaluation": evaluation,
                "keyword": keyword_token.text,
                "sentence": sentence
            })

    return value_statements