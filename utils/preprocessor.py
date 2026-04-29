"""
Text Preprocessing Module
Handles text cleaning and normalization for document classification.
"""

import re
import string
import logging

logger = logging.getLogger(__name__)

# Fallback stopwords (common English stopwords) in case NLTK is unavailable
FALLBACK_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'this', 'that',
    'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he',
    'his', 'she', 'her', 'it', 'its', 'they', 'their', 'them', 'what',
    'which', 'who', 'whom', 'when', 'where', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'not', 'only', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just',
    'don', 'now', 're', 've', 'll', 'also', 'as', 'if', 'up', 'out', 'about',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between',
    'there', 'here', 'then', 'once', 'any', 'own', 'while'
}

try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
    logger.info("Using NLTK stopwords")
except Exception:
    STOP_WORDS = FALLBACK_STOPWORDS
    logger.info("Using fallback stopwords")


def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Clean and preprocess raw text.

    Args:
        text: Raw input text
        remove_stopwords: Whether to remove stopwords

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # Remove numbers (keep alphabetic)
    text = re.sub(r'\d+', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    if remove_stopwords:
        tokens = text.split()
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
        text = ' '.join(tokens)

    return text


def extract_text_from_file(file_path: str) -> str:
    """
    Extract text content from .txt, .pdf, or .docx files.

    Args:
        file_path: Path to the document file

    Returns:
        Extracted text as string
    """
    import os
    ext = os.path.splitext(file_path)[-1].lower()

    try:
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

        elif ext == '.pdf':
            try:
                import fitz  # PyMuPDF
                text = []
                with fitz.open(file_path) as doc:
                    for page in doc:
                        text.append(page.get_text())
                return '\n'.join(text)
            except ImportError:
                try:
                    import PyPDF2
                    text = []
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text.append(page_text)
                    return '\n'.join(text)
                except Exception as e:
                    raise ValueError(f"PDF extraction failed: {e}")
            except Exception as e:
                raise ValueError(f"PDF extraction failed: {e}")

        elif ext == '.docx':
            try:
                from docx import Document
                doc = Document(file_path)
                return '\n'.join([para.text for para in doc.paragraphs])
            except Exception as e:
                raise ValueError(f"DOCX extraction failed: {e}")

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    except Exception as e:
        logger.error(f"File extraction error for {file_path}: {e}")
        raise
