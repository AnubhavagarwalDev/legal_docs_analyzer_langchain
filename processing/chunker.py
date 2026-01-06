import re
from typing import List, Dict, Optional


# ---------------------------------------------------------
# Regex Patterns
# ---------------------------------------------------------

SECTION_NUMBER_PATTERN = re.compile(r"^\d+(\.\d+)*\.?\s+")
ALL_CAPS_HEADING_PATTERN = re.compile(r"^[A-Z][A-Z\s]{3,}$")
BULLET_PATTERN = re.compile(r"^(\(?[a-zA-Z0-9]+\)|[-•])\s*")


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def is_section_heading(line: str) -> bool:
    stripped = line.strip()

    # Numbered headings (1., 1.1 etc.)
    if SECTION_NUMBER_PATTERN.match(stripped):
        return True

    # ALL CAPS headings
    if ALL_CAPS_HEADING_PATTERN.match(stripped):
        return True

    # Safe Title-Case heading detection
    words = stripped.split()
    if (
        1 <= len(words) <= 6
        and stripped.istitle()
        and not re.search(r"[:.,;]", stripped)
    ):
        return True

    return False


def is_subclause(line: str) -> bool:
    return bool(BULLET_PATTERN.match(line.strip()))


def extract_heading(line: str) -> str:
    line = line.strip()
    line = SECTION_NUMBER_PATTERN.sub("", line)
    return line.strip()


# ---------------------------------------------------------
# Main Chunking Function
# ---------------------------------------------------------

def chunk_document(
    cleaned_text: str,
    max_words: int = 1000
) -> List[Dict[str, Optional[str]]]:
    """
    Clause-aware chunker for legal documents.

    - Preserves legal structure
    - Handles multi-line bullets
    - Prevents premature flushing
    """

    lines = cleaned_text.split("\n")
    chunks: List[Dict[str, Optional[str]]] = []

    current_heading: Optional[str] = None
    current_chunk_lines: List[str] = []
    inside_list = False
    chunk_id = 1

    def flush_chunk():
        nonlocal chunk_id, current_chunk_lines
        if not current_chunk_lines:
            return

        text = "\n".join(current_chunk_lines).strip()
        if not text:
            return

        chunks.append({
            "chunk_id": str(chunk_id),
            "heading": current_heading,
            "text": text
        })
        chunk_id += 1
        current_chunk_lines = []

    for line in lines:
        stripped = line.strip()

        # Blank lines
        if not stripped:
            current_chunk_lines.append("")
            continue

        # Section heading
        if is_section_heading(stripped):
            flush_chunk()
            inside_list = False
            current_heading = extract_heading(stripped)
            current_chunk_lines.append(stripped)
            continue

        # Bullet start
        if is_subclause(stripped):
            inside_list = True
            current_chunk_lines.append(stripped)
            continue

        # Bullet continuation
        if inside_list:
            current_chunk_lines.append(stripped)
            continue

        # Normal content
        current_chunk_lines.append(stripped)

        # Soft size guard (never split lists)
        word_count = sum(len(l.split()) for l in current_chunk_lines)
        if word_count > max_words and not inside_list:
            flush_chunk()

    flush_chunk()
    return chunks
