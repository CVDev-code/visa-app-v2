import io
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF


RED = (1, 0, 0)


def _center(rect: fitz.Rect) -> fitz.Point:
    return fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)


def _draw_red_box(page: fitz.Page, rect: fitz.Rect, width: float = 1.5):
    page.draw_rect(rect, color=RED, width=width)


def _draw_callout(
    page: fitz.Page,
    callout_rect: fitz.Rect,
    text: str,
    target_rects: List[fitz.Rect],
    fontname: str = "Times-Roman",
    fontsize: int = 10,
):
    # Optional: outline the callout area lightly (comment out if you don't want a box)
    page.draw_rect(callout_rect, color=RED, width=1.0)

    page.insert_textbox(
        callout_rect,
        text,
        fontname=fontname,
        fontsize=fontsize,
        color=RED,
        align=fitz.TEXT_ALIGN_LEFT,
    )

    # Draw connector line(s) from callout to each target
    start = _center(callout_rect)
    for tr in target_rects:
        end = _center(tr)
        page.draw_line(start, end, color=RED, width=1.0)


def _find_all_instances(page: fitz.Page, needles: List[str]) -> List[fitz.Rect]:
    hits: List[fitz.Rect] = []
    for s in needles:
        s = (s or "").strip()
        if not s:
            continue
        try:
            rects = page.search_for(s)
            hits.extend(rects)
        except Exception:
            # search_for can fail on weird inputs; ignore and continue
            pass
    return hits


def _first_page_callout_layout(page: fitz.Page) -> Tuple[fitz.Rect, float]:
    """
    Returns a starting callout rectangle (top-right area) and a vertical step.
    This is a simple layout strategy to avoid overlapping callouts.
    """
    r = page.rect
    margin = 36  # half-inch-ish
    x0 = r.width * 0.55
    x1 = r.width - margin
    y0 = margin
    y1 = y0 + 48
    return fitz.Rect(x0, y0, x1, y1), 56  # step


def _criterion_fixed_callouts(criterion_id: str) -> List[str]:
    # Page-1 fixed annotations requested
    if criterion_id == "1":
        return ["Significant award in the music industry."]
    if criterion_id == "3":
        return ["Beneficiary’s performance is recognised as receiving critical acclaim."]
    if criterion_id == "5":
        return ["Publication demonstrates the beneficiary’s success is critically acclaimed."]
    if criterion_id == "6":
        return ["Publication demonstrates the beneficiary’s successes and achievements have gained significant recognition."]
    return []


def annotate_pdf_bytes(
    pdf_bytes: bytes,
    quote_terms: List[str],
    criterion_id: str,
    meta: Dict,
) -> Tuple[bytes, Dict]:
    """
    Draws red boxes around:
      - approved AI quote terms (all pages)
      - manual metadata fields (page 1 only), plus callouts + connector lines (page 1 only)
      - fixed criterion callouts (page 1 only)
    Returns (output_pdf_bytes, report)
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    total_quote_hits = 0
    total_meta_hits = 0

    # ---- A) Highlight approved quote terms across ALL pages
    quote_terms = [t.strip() for t in (quote_terms or []) if t and t.strip()]
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        for t in quote_terms:
            rects = page.search_for(t)
            if rects:
                total_quote_hits += len(rects)
                for r in rects:
                    _draw_red_box(page, r, width=1.5)

    # ---- B) Page 1 metadata + criterion callouts
    if len(doc) > 0:
        page1 = doc.load_page(0)

        callout_rect, step = _first_page_callout_layout(page1)
        callout_index = 0

        def next_callout_rect():
            nonlocal callout_index
            r = fitz.Rect(callout_rect)
            r.y0 += callout_index * step
            r.y1 += callout_index * step
            callout_index += 1
            return r

        # Helper: box + callout
        def box_and_callout(label: str, needles: List[str]):
            nonlocal total_meta_hits
            rects = _find_all_instances(page1, needles)
            if rects:
                total_meta_hits += len(rects)
                for rr in rects:
                    _draw_red_box(page1, rr, width=1.5)
                _draw_callout(page1, next_callout_rect(), label, rects)
            else:
                # Still draw the callout box (no lines) so reviewer sees intent
                _draw_callout(page1, next_callout_rect(), f"{label} (not found on page 1)", [])

        # (1) URL boxing on page 1 only
        source_url = (meta.get("source_url") or "").strip()
        if source_url:
            # try exact, then a few softened variants
            url_candidates = [source_url]
            if source_url.startswith("https://"):
                url_candidates.append(source_url.replace("https://", "http://"))
                url_candidates.append(source_url.replace("https://", ""))
            if source_url.startswith("http://"):
                url_candidates.append(source_url.replace("http://", "https://"))
                url_candidates.append(source_url.replace("http://", ""))
            box_and_callout("Original source of publication.", url_candidates)

        # Fixed criterion callouts on page 1 (no target unless you want one later)
        for msg in _criterion_fixed_callouts(criterion_id):
            _draw_callout(page1, next_callout_rect(), msg, [])

        # Shared manual metadata
        beneficiary_name = (meta.get("beneficiary_name") or "").strip()
        beneficiary_variants = meta.get("beneficiary_variants") or []
        venue_name = (meta.get("venue_name") or "").strip()
        org_name = (meta.get("org_name") or "").strip()
        performance_date = (meta.get("performance_date") or "").strip()
        salary_amount = (meta.get("salary_amount") or "").strip()

        # Criteria-specific behaviour:
        # Criterion 2 (past/future): beneficiary + date + optional venue
        if criterion_id in {"2_past", "2_future"}:
            if beneficiary_name:
                box_and_callout(
                    "Beneficiary named as a lead role in the distinguished performance.",
                    [beneficiary_name] + beneficiary_variants,
                )
            if performance_date:
                label = "past performance date" if criterion_id == "2_past" else "future performance date"
                box_and_callout(label, [performance_date])
            if venue_name:
                box_and_callout("The distinguished organization.", [venue_name])

        # Criterion 4 (past/future): org + beneficiary + date
        if criterion_id in {"4_past", "4_future"}:
            if venue_name:
                box_and_callout("The distinguished organization.", [venue_name])
            if beneficiary_name:
                box_and_callout(
                    "Beneficiary named as a lead role in performance at the distinguished organization.",
                    [beneficiary_name] + beneficiary_variants,
                )
            if performance_date:
                label = "past performance date" if criterion_id == "4_past" else "future performance date"
                box_and_callout(label, [performance_date])

        # Criterion 6: org name boxing (if provided)
        if criterion_id == "6":
            if org_name:
                box_and_callout(
                    "Publication demonstrates the beneficiary’s successes and achievements have gained significant recognition.",
                    [org_name],
                )

        # Criterion 7: salary amount boxing (if provided)
        if criterion_id == "7":
            if salary_amount:
                box_and_callout(
                    "Beneficiary’s salary is significant higher than others in the field.",
                    [salary_amount],
                )

    # Save to bytes
    out = io.BytesIO()
    doc.save(out)
    doc.close()
    out.seek(0)

    return out.getvalue(), {
        "total_quote_hits": total_quote_hits,
        "total_meta_hits": total_meta_hits,
        "criterion_id": criterion_id,
    }


# Backwards compatible wrapper (if other parts call the old function)
def highlight_terms_in_pdf_bytes(pdf_bytes: bytes, terms: List[str]):
    # legacy behaviour: quotes only, no metadata/callouts
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_hits = 0
    terms = [t.strip() for t in (terms or []) if t and t.strip()]

    for pno in range(len(doc)):
        page = doc.load_page(pno)
        for t in terms:
            rects = page.search_for(t)
            if rects:
                total_hits += len(rects)
                for r in rects:
                    _draw_red_box(page, r, width=1.5)

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    out.seek(0)
    return out.getvalue(), {"total_hits": total_hits}
