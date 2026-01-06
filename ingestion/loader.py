import pdfplumber
from typing import Dict, Any, List
from ingestion.ocr import ocr_pdf


MIN_TEXT_CHARS_PER_PAGE = 50
OCR_CONFIDENCE_THRESHOLD = 60.0


def load_pdf(
    pdf_path: str,
    source: str = "user"
) -> Dict[str, Any]:
    """
    Load a PDF file and extract text using:
    - Direct text extraction (preferred)
    - OCR fallback (page-level)

    Returns:
        {
            "text": str,
            "metadata": dict
        }
    """

    extracted_pages: List[str] = []
    ocr_pages_used = 0
    low_confidence_pages = 0

    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)

        # First pass: try text extraction page-by-page
        page_texts = []
        for page in pdf.pages:
            text = page.extract_text() or ""
            page_texts.append(text.strip())

    # Decide OCR per page
    final_pages = []
    ocr_required_pages = []

    for idx, text in enumerate(page_texts):
        if len(text) < MIN_TEXT_CHARS_PER_PAGE:
            ocr_required_pages.append(idx)
            final_pages.append(None)
        else:
            final_pages.append(text)

    # OCR only required pages
    if ocr_required_pages:
        ocr_texts, ocr_confs = ocr_pdf(pdf_path)

        for idx in ocr_required_pages:
            ocr_pages_used += 1
            ocr_text = ocr_texts[idx]
            conf = ocr_confs[idx]

            if conf < OCR_CONFIDENCE_THRESHOLD:
                low_confidence_pages += 1

            final_pages[idx] = ocr_text.strip()

    # Merge pages in correct order
    full_text = "\n\n".join(page for page in final_pages if page)

    metadata = {
        "source": source,
        "filename": pdf_path.split("/")[-1],
        "num_pages": num_pages,
        "used_ocr": ocr_pages_used > 0,
        "ocr_pages": ocr_pages_used,
        "low_confidence_ocr_pages": low_confidence_pages
    }

    return {
        "text": full_text,
        "metadata": metadata
    }
