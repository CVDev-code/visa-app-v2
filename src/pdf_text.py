import fitz  # PyMuPDF


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        parts = []
        for page in doc:
            parts.append(page.get_text("text"))
        return "\n".join(parts)
    finally:
        doc.close()


def extract_first_page_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extracts text from the first page only.
    Used for metadata (URL/date/money) detection and first-page callouts.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if doc.page_count == 0:
            return ""
        page = doc.load_page(0)
        return page.get_text("text")
    finally:
        doc.close()
