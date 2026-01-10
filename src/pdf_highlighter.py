import fitz  # PyMuPDF

def highlight_terms_in_pdf_bytes(
    pdf_bytes: bytes,
    terms: list[str],
    stroke_rgb=(1, 0, 0),
    border_width: float = 1.5,
):
    """
    Draw red rectangle annotations around exact matches of each term.
    Returns (output_pdf_bytes, report_dict).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    report = {
        "total_hits": 0,
        "hits_by_term": {},
        "hits_by_page": {},
    }

    if not terms:
        out = doc.write()
        doc.close()
        return out, report

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)

        for term in terms:
            if not term:
                continue

            rects = page.search_for(term)
            if not rects:
                continue

            report["total_hits"] += len(rects)
            report["hits_by_term"][term] = (
                report["hits_by_term"].get(term, 0) + len(rects)
            )
            report["hits_by_page"][page_index + 1] = (
                report["hits_by_page"].get(page_index + 1, 0) + len(rects)
            )

            for r in rects:
                annot = page.add_rect_annot(r)
                annot.set_colors(stroke=stroke_rgb)  # red
                annot.set_border(width=border_width)
                annot.update()

    out = doc.write()
    doc.close()
    return out, report
