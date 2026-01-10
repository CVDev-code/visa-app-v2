import io
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF

RED = (1, 0, 0)


def _center(rect: fitz.Rect) -> fitz.Point:
    return fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)


def _draw_red_box(page: fitz.Page, rect: fitz.Rect, width: float = 1.5):
    page.draw_rect(rect, color=RED, width=width)


def _union_rect(rects: List[fitz.Rect]) -> Optional[fitz.Rect]:
    if not rects:
        return None
    r = fitz.Rect(rects[0])
    for x in rects[1:]:
        r |= x
    return r


def _page_text_rects(page: fitz.Page) -> List[fitz.Rect]:
    """
    Returns rectangles of text blocks for collision checks.
    """
    rects: List[fitz.Rect] = []
    try:
        blocks = page.get_text("blocks")  # (x0,y0,x1,y1,"text", block_no, block_type)
        for b in blocks:
            if len(b) >= 4:
                rects.append(fitz.Rect(b[0], b[1], b[2], b[3]))
    except Exception:
        pass
    return rects


def _intersects_any(r: fitz.Rect, others: List[fitz.Rect], pad: float = 2.0) -> bool:
    rr = fitz.Rect(r)
    rr.x0 -= pad
    rr.y0 -= pad
    rr.x1 += pad
    rr.y1 += pad
    for o in others:
        if rr.intersects(o):
            return True
    return False


def _find_free_callout_rect(page: fitz.Page, w: float = 220, h: float = 40) -> fitz.Rect:
    """
    Heuristic: choose a legible location by trying a set of candidate areas
    and picking the first that doesn't overlap text blocks.
    """
    r = page.rect
    margin = 24
    text_blocks = _page_text_rects(page)

    candidates: List[fitz.Rect] = [
        # top-right margin-ish
        fitz.Rect(r.width - margin - w, margin, r.width - margin, margin + h),
        fitz.Rect(r.width - margin - w, margin + 50, r.width - margin, margin + 50 + h),
        fitz.Rect(r.width - margin - w, margin + 100, r.width - margin, margin + 100 + h),
        # top-left
        fitz.Rect(margin, margin, margin + w, margin + h),
        fitz.Rect(margin, margin + 50, margin + w, margin + 50 + h),
        # bottom-right
        fitz.Rect(r.width - margin - w, r.height - margin - h, r.width - margin, r.height - margin),
        fitz.Rect(r.width - margin - w, r.height - margin - h - 50, r.width - margin, r.height - margin - 50),
        # bottom-left
        fitz.Rect(margin, r.height - margin - h, margin + w, r.height - margin),
        fitz.Rect(margin, r.height - margin - h - 50, margin + w, r.height - margin - 50),
    ]

    for c in candidates:
        if not _intersects_any(c, text_blocks, pad=3.0):
            return c

    # If we couldn't find a clear spot, still return top-right as fallback
    return candidates[0]


def _draw_callout_text(
    page: fitz.Page,
    callout_rect: fitz.Rect,
    text: str,
    fontname: str = "Times-Roman",
    fontsize: int = 10,
):
    """
    Draw text only (NO red box around the annotation).
    """
    page.insert_textbox(
        callout_rect,
        text,
        fontname=fontname,
        fontsize=fontsize,
        color=RED,
        align=fitz.TEXT_ALIGN_LEFT,
    )


def _draw_connectors(
    page: fitz.Page,
    callout_rect: fitz.Rect,
    targets: List[fitz.Rect],
    *,
    mode: str = "multi",
):
    """
    mode:
      - "single": draw one line to ONE target (or a union target)
      - "multi": draw one line to EACH target
    """
    if not targets:
        return

    start = _center(callout_rect)

    if mode == "single":
        # One line to the "best" target (prefer the first / closest)
        # If caller passed a union rect, that's fine too.
        target = targets[0]
        end = _center(target)
        page.draw_line(start, end, color=RED, width=1.0)
        return

    # Multi lines (one per target)
    for tr in targets:
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
            pass
    return hits


def _criterion_fixed_callouts(criterion_id: str) -> List[str]:
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
    Draws:
      - Red boxes around approved AI quote terms (all pages).
      - On page 1:
          * URL box + callout + ONE connector line
          * criterion fixed callouts (text only)
          * criterion-specific metadata boxes + callouts + connectors
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

        # We'll place each callout in a non-overlapping free spot by re-checking blocks each time.
        def place_callout(text: str) -> fitz.Rect:
            # Find a good position each time (so if earlier callouts filled a spot, we adapt)
            r = _find_free_callout_rect(page1, w=240, h=42)
            _draw_callout_text(page1, r, text)
            return r

        # Helper: box + callout + connectors
        def box_and_callout(
            label: str,
            needles: List[str],
            *,
            connector_mode: str = "multi",
            collapse_targets_to_one: bool = False,
        ):
            nonlocal total_meta_hits

            rects = _find_all_instances(page1, needles)
            if rects:
                total_meta_hits += len(rects)
                for rr in rects:
                    _draw_red_box(page1, rr, width=1.5)

                # For cases like URL where it appears twice (header/footer),
                # collapse to one union rect and draw ONE line.
                targets = rects
                if collapse_targets_to_one:
                    u = _union_rect(rects)
                    targets = [u] if u else rects

                callout_rect = place_callout(label)
                _draw_connectors(page1, callout_rect, targets, mode=connector_mode)
            else:
                # Draw the callout text anyway (no lines)
                place_callout(f"{label} (not found on page 1)")

        # (1) URL boxing on page 1 only
        source_url = (meta.get("source_url") or "").strip()
        if source_url:
            url_candidates = [source_url]
            if source_url.startswith("https://"):
                url_candidates.append(source_url.replace("https://", "http://"))
                url_candidates.append(source_url.replace("https://", ""))
            if source_url.startswith("http://"):
                url_candidates.append(source_url.replace("http://", "https://"))
                url_candidates.append(source_url.replace("http://", ""))

            # IMPORTANT: union targets + SINGLE connector line
            box_and_callout(
                "Original source of publication.",
                url_candidates,
                connector_mode="single",
                collapse_targets_to_one=True,
            )

        # Fixed criterion callouts (text only; no boxes; no connectors)
        for msg in _criterion_fixed_callouts(criterion_id):
            place_callout(msg)

        # Shared manual metadata
        beneficiary_name = (meta.get("beneficiary_name") or "").strip()
        beneficiary_variants = meta.get("beneficiary_variants") or []
        venue_name = (meta.get("venue_name") or "").strip()
        org_name = (meta.get("org_name") or "").strip()
        performance_date = (meta.get("performance_date") or "").strip()
        salary_amount = (meta.get("salary_amount") or "").strip()

        # Criteria-specific behaviour:
        if criterion_id in {"2_past", "2_future"}:
            if beneficiary_name:
                box_and_callout(
                    "Beneficiary named as a lead role in the distinguished performance.",
                    [beneficiary_name] + beneficiary_variants,
                    connector_mode="multi",  # one line to EACH match
                )
            if performance_date:
                label = "past performance date" if criterion_id == "2_past" else "future performance date"
                box_and_callout(label, [performance_date], connector_mode="single", collapse_targets_to_one=True)
            if venue_name:
                box_and_callout("The distinguished organization.", [venue_name], connector_mode="single", collapse_targets_to_one=True)

        if criterion_id in {"4_past", "4_future"}:
            if venue_name:
                box_and_callout("The distinguished organization.", [venue_name], connector_mode="single", collapse_targets_to_one=True)
            if beneficiary_name:
                box_and_callout(
                    "Beneficiary named as a lead role in performance at the distinguished organization.",
                    [beneficiary_name] + beneficiary_variants,
                    connector_mode="multi",  # connect to ALL beneficiary occurrences
                )
            if performance_date:
                label = "past performance date" if criterion_id == "4_past" else "future performance date"
                box_and_callout(label, [performance_date], connector_mode="single", collapse_targets_to_one=True)

        if criterion_id == "6" and org_name:
            box_and_callout(
                "Publication demonstrates the beneficiary’s successes and achievements have gained significant recognition.",
                [org_name],
                connector_mode="single",
                collapse_targets_to_one=True,
            )

        if criterion_id == "7" and salary_amount:
            box_and_callout(
                "Beneficiary’s salary is significant higher than others in the field.",
                [salary_amount],
                connector_mode="single",
                collapse_targets_to_one=True,
            )

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    out.seek(0)

    return out.getvalue(), {
        "total_quote_hits": total_quote_hits,
        "total_meta_hits": total_meta_hits,
        "criterion_id": criterion_id,
    }


# Legacy wrapper (quotes only)
def highlight_terms_in_pdf_bytes(pdf_bytes: bytes, terms: List[str]):
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
