import spacy
from typing import List

# Load the spaCy model once when the module is imported
# This is more efficient than loading it inside the function.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model en_core_web_sm...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def segment_text(text: str) -> List[str]:
    """
    Segments the cleaned text into a list of sentences.

    Args:
        text: The cleaned text string.

    Returns:
        A list of strings, where each string is a sentence.
    """
    # Process the text with spaCy
    doc = nlp(text)

    # Extract sentences, strip whitespace, and filter out empty sentences
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    return sentences
