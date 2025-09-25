from pathlib import Path
from pypdf import PdfReader
from striprtf.striprtf import rtf_to_text

def ingest_corpus(file_path: str) -> str:
    """
    Ingests a corpus file and returns its text content.

    Supports .txt, .pdf, and .rtf files.

    Args:

        file_path: The absolute path to the corpus file.

    Returns:
        The text content of the file.
        
    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"No file found at {file_path}")

    suffix = path.suffix.lower()
    
    if suffix == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    elif suffix == '.pdf':
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    elif suffix == '.rtf':
        with open(path, 'r') as f:
            rtf_content = f.read()
        return rtf_to_text(rtf_content)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

