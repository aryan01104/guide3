import re
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    """
    Cleans the raw text by parsing HTML, lowercasing, and normalizing whitespace.

    Args:
        text: The raw text string, which may contain HTML.

    Returns:
        A cleaned text string.
    """
    # 1. Parse HTML and extract the text content
    soup = BeautifulSoup(text, 'html.parser')
    pre_tag = soup.find('pre')
    if pre_tag:
        text_content = pre_tag.get_text()
    else:
        text_content = soup.get_text()

    # 2. Convert to lowercase and normalize whitespace first
    text_content = text_content.lower()
    text_content = re.sub(r'[\n\t]+', ' ', text_content)
    text_content = re.sub(r'\s{2,}', ' ', text_content)

    # 3. Remove the boilerplate header now that the text is clean
    # This pattern looks for the start of the text and removes everything up to the first sentence.
    text_content = re.sub(r'^.*elizabeth carter 1\.', '', text_content, count=1)

    # 4. Remove spaces before punctuation
    text_content = re.sub(r'\s+([.,?!])', r'\1', text_content)

    return text_content.strip()
