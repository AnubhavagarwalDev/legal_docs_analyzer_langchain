import re
from typing import List


def normalize_newlines(text: str) -> str:
    """Normalize different newline formats."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def remove_page_headers_footers(text: str) -> str:
    """
    Remove common headers/footers like page numbers.
    Conservative heuristic to avoid legal meaning loss.
    """
    cleaned_lines: List[str] = []

    for line in text.split("\n"):
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            cleaned_lines.append("")
            continue

        # Remove page numbering patterns
        if re.match(r"^page\s+\d+(\s+of\s+\d+)?$", stripped.lower()):
            continue

        # Remove standalone numbers (page numbers)
        if re.match(r"^\d+$", stripped):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def fix_hyphenated_words(text: str) -> str:
    """
    Fix words broken across lines by hyphenation.
    Example: 'agree-\nment' -> 'agreement'
    """
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def collapse_excessive_newlines(text: str) -> str:
    """Collapse more than two newlines into two."""
    return re.sub(r"\n{3,}", "\n\n", text)


def normalize_whitespace(text: str) -> str:
    """Normalize spaces and tabs."""
    text = text.replace("\t", " ")
    return re.sub(r"[ ]{2,}", " ", text)


def clean_text(raw_text: str) -> str:
    """
    Main cleaning pipeline.
    Input: raw extracted text (OCR or text-based)
    Output: normalized, LLM-friendly text
    """
    text = normalize_newlines(raw_text)
    text = fix_hyphenated_words(text)
    text = remove_page_headers_footers(text)
    text = normalize_whitespace(text)
    text = collapse_excessive_newlines(text)

    return text.strip()
