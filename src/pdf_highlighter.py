import io
import math
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF

RED = (1, 0, 0)


# -----------------------------
# Geometry helpers
# -----------------------------
def _center(rect: fitz.Rect) -> fitz.Point:
    return fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)


def _dist(a: fitz.Point, b: fitz.Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _draw_red_box(page: fitz.Page, rect: fitz.Rect, width: float = 1.5):
    page.draw_rect(rect, color=RED, width=width)


def _union_rect(rects: List[fitz.Rect]) -> Optional[fitz.Rect]:
    if not rects:
        return None
    r = fitz.Rect(rects[0])
    for x in rects[1:]:
        r |= x
    return r


def _inflate(r: fitz.Rect, pad: float) -> fitz.Rect:
    rr = fitz.Rect(r)
    rr.x0 -= pad
    rr.y0 -= pad
    rr.x1 += pad
    rr.y1 += pad
    return rr


def _intersects_any(r: fitz.Rect, others: List[fitz.Rect], pad: float = 2.0) -> bool:
    rr = _inflate(r, pad)
    for o in others:
        if rr.intersects(o):
            return True
    return False


# -----------------------------
# "White space" blockers
# -----------------------------
def _page_text_rects(page: fitz.Page) -> List[fitz.Rect]:
    rects: List[fitz.Rect] = []
    try:
        blocks = page.get_text("blocks")  # (x0,y0,x1,y1,"text", block_no, block_type)
        for b in blocks:
            if len(b) >= 4:
                rects.append(fitz.Rect(b[0], b[1], b[2], b[3]))
    except Exception:
        pass
    return rects


def _page_drawing_rects(page: fitz.Page) -> List[fitz.Rect]:
    """
    Attempts to collect rectangles for images + vector drawings so callouts avoid them.
    """
    rects: List[fitz.Rect] = []

    # Images (most reliable)
    try:
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                irs = page.get_image_rects(xref)
                rects.extend(list(irs))
            except Exception:
                continue
    except Exception:
        pass

    # Vector drawings
    try:
        drawings = page.get_drawings()
        for d in drawings:
            r = d.get("rect")
            if r:
                rects.append(fitz.Rect(r))
    except Exception:
        pass

    return rects


def _blockers(page: fitz.Page, occupied_callouts: List[fitz.Rect]) -> List[fitz.Rect]:
    return _page_text_rects(page) + _page_drawing_rects(page) + list(occupied_callouts)


# -----------------------------
# Text wrapping + font selection (10–12)
# -----------------------------
def _wrap_text(text: str, fontname: str, fontsize: int, max_width: float) -> List[str]:
    """
    Greedy wrap based on text width. Uses fitz.get_text_length.
    """
    words = (text or "").split()
    if not words:
        return [""]

    lines: List[str] = []
    cur = words[0]
    for w in words[1:]:
        trial = f"{cur} {w}"
        if fitz.get_text_length(trial, fontname=fontname, fontsize=fontsize) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _best_textbox_layout(
    text: str,
    *,
    fontname: str = "Times-Roman",
    min_size: int = 10,
    max_size: int = 12,
    box_width: float = 260,
    max_lines: int = 6,
) -> Tuple[int, str, float]:
    """
    Choose fontsize (10..12) and line breaks to fit neatly.
    Returns (fontsize, final_text_with_newlines, box_height).
    """
    # Try bigger font first (nicer), fall back to smaller.
    for fontsize in range(max_size, min_size - 1, -1):
        usable_width = box_width  # textbox width; wrap to this
        lines = _wrap_text(text, fontname, fontsize, usable_width)
        if len(lines) <= max_lines:
            # line height ~ fontsize * 1.25 (looks decent)
            line_h = fontsize * 1.25
            height = max(38.0, line_h * len(lines) + 6)  # minimum for readability
            return fontsize, "\n".join(lines), height

    # If still too long, allow more lines but keep fontsize at min_size
    fontsize = min_size
    lines = _wrap_text(text, fontname, fontsize, box_width)
    # cap to something reasonable; remaining words become extra lines if needed
    # (still readable, avoids extreme "one word per line")
    if len(lines) > 10:
        # rewrap by increasing width slightly via allowing longer lines
        # but since width is fixed, just accept up to 10 lines.
        lines = lines[:10] + ["…"]
    line_h = fontsize * 1.25
    height = max(42.0, line_h * len(lines) + 6)
    return fontsize, "\n".join(lines), height


# -----------------------------
# Candidate placement near targets
# -----------------------------
def _candidate_callout_rects_near(
    page: fitz.Page,
    target_union: fitz.Rect,
    *,
    w: float,
    h: float,
    margin: float = 18.0,
) -> List[fitz.Rect]:
    """
    Generate candidate rectangles near the target, then some safe fallbacks.
    """
    pr = page.rect
    t = fitz.Rect(target_union)

    # Basic offsets around target
    gaps = [10, 18, 26, 34]
    candidates: List[fitz.Rect] = []

    for g in gaps:
        # above target
        candidates.append(fitz.Rect(t.x0, t.y0 - g - h, t.x0 + w, t.y0 - g))
        candidates.append(fitz.Rect(t.x1 - w, t.y0 - g - h, t.x1, t.y0 - g))
        # below target
        candidates.append(fitz.Rect(t.x0, t.y1 + g, t.x0 + w, t.y1 + g + h))
        candidates.append(fitz.Rect(t.x1 - w, t.y1 + g, t.x1, t.y1 + g + h))
        # left of target
        candidates.append(fitz.Rect(t.x0 - g - w, t.y0, t.x0 - g, t.y0 + h))
        candidates.append(fitz.Rect(t.x0 - g - w, t.y1 - h, t.x0 - g, t.y1))
        # right of target
        candidates.append(fitz.Rect(t.x1 + g, t.y0, t.x1 + g + w, t.y0 + h))
        candidates.append(fitz.Rect(t.x1 + g, t.y1 - h, t.x1 + g + w, t.y1))

    # Clamp candidates to page bounds (and discard those way off-page)
    clamped: List[fitz.Rect] = []
    for c in candidates:
        if c.x1 < 0 or c.y1 < 0 or c.x0 > pr.width or c.y0 > pr.height:
            continue
        cc = fitz.Rect(
            max(margin, c.x0),
            max(margin, c.y0),
            min(pr.width - margin, c.x1),
            min(pr.height - margin, c.y1),
        )
        # Maintain width/height if clamping crushed it too much
        if cc.width < w * 0.75 or cc.height < h * 0.75:
            continue
        clamped.append(cc)

    # Fallback corners (in case around-target spaces are busy)
    fallback = [
        fitz.Rect(pr.width - margin - w, margin, pr.width - margin, margin + h),
        fitz.Rect(margin, margin, margin + w, margin + h),
        fitz.Rect(pr.width - margin - w, pr.height - margin - h, pr.width - margin, pr.height - margin),
        fitz.Rect(margin, pr.height - margin - h, margin + w, pr.height - margin),
    ]
    clamped.extend(fallback)
    return clamped


def _score_candidate(
    cand: fitz.Rect,
    target_union: fitz.Rect,
    blockers: List[fitz.Rect],
) -> float:
    """
    Lower is better.
    Primary: distance from callout center to target center (short lines).
    Reject overlaps strongly.
    """
    # Hard reject if it overlaps blockers
    if _intersects_any(cand, blockers, pad=3.0):
        return 1e9

    # Distance term
    d = _dist(_center(cand), _center(target_union))

    # Slight preference for staying inside page margins (already clamped)
    # Slight preference for being above/below rather than far left/right
    # (purely aesthetic; keep small)
    dx = abs(_center(cand).x - _center(target_union).x)
    dy = abs(_center(cand).y - _center(target_union).y)
    axis_bias = 0.15 * dx + 0.05 * dy

    return d + axis_bias


def _choose_callout_rect(
    page: fitz.Page,
    targets: List[fitz.Rect],
    occupied_callouts: List[fitz.Rect],
    *,
    box_width: float = 260,
    box_height: float = 44,
) -> fitz.Rect:
    pr = page.rect
    margin = 18.0
    union = _union_rect(targets) or fitz.Rect(margin, margin, pr.width - margin, pr.height - margin)

    blockers = _blockers(page, occupied_callouts)
    candidates = _candidate_callout_rects_near(page, union, w=box_width, h=box_height, margin=margin)

    best = candidates[0]
    best_score = 1e18
    for c in candidates:
        s = _score_candidate(c, union, blockers)
        if s < best_score:
            best_score = s
            best = c

    return best


# -----------------------------
# Finding instances
# -----------------------------
def _find_all_instances(page: fitz.Page, needles: List[str]) -> List[fitz.Rect]:
    hits: List[fitz.Rect] = []
    for s in needles:
        s = (s or "").strip()
        if not s:
            continue
        try:
            hits.extend(page.search_for(s))
        except Exception:
            pass
    return hits


def _draw_connectors(page: fitz.Page, callout_rect: fitz.Rect, targets: List[fitz.Rect], *, mode: str):
    if not targets:
        return
    start = _center(callout_rect)

    if mode == "single":
        end = _center(targets[0])
        page.draw_line(start, end, color=RED, width=1.0)
        return

    for tr in targets:
        end = _center(tr)
        page.draw_line(start, end, color=RED, width=1.0)


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


# -----------------------------
# Main annotator
# -----------------------------
def annotate_pdf_bytes(
    pdf_bytes: bytes,
    quote_terms: List[str],
    criterion_id: str,
    meta: Dict,
) -> Tuple[bytes, Dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    total_quote_hits = 0
    total_meta_hits = 0

    # A) Quote boxes across all pages
    quote_terms = [t.strip() for t in (quote_terms or []) if t and t.strip()]
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        for t in quote_terms:
            rects = page.search_for(t)
            if rects:
                total_quote_hits += len(rects)
                for r in rects:
                    _draw_red_box(page, r, width=1.5)

    # B) Page 1 callouts + metadata boxes
    if len(doc) > 0:
        page1 = doc.load_page(0)
        occupied_callouts: List[fitz.Rect] = []

        def place_callout_near_targets(text: str, targets_for_placement: List[fitz.Rect]) -> fitz.Rect:
            # Choose font size + wrapping (10–12)
            fontsize, wrapped_text, box_h = _best_textbox_layout(
                text,
                fontname="Times-Roman",
                min_size=10,
                max_size=12,
                box_width=260,
                max_lines=6,
            )

            callout_rect = _choose_callout_rect(
                page1,
                targets_for_placement if targets_for_placement else [fitz.Rect(20, 20, 200, 60)],
                occupied_callouts,
                box_width=260,
                box_height=box_h,
            )

            # Text only (no box)
            page1.insert_textbox(
                callout_rect,
                wrapped_text,
                fontname="Times-Roman",
                fontsize=fontsize,
                color=RED,
                align=fitz.TEXT_ALIGN_LEFT,
            )

            occupied_callouts.append(callout_rect)
            return callout_rect

        def box_and_callout(
            label: str,
            needles: List[str],
            *,
            connector_mode: str = "multi",
            collapse_targets_to_one: bool = False,
            draw_callout_even_if_not_found: bool = True,
        ):
            nonlocal total_meta_hits

            rects = _find_all_instances(page1, needles)
            if rects:
                total_meta_hits += len(rects)
                for rr in rects:
                    _draw_red_box(page1, rr, width=1.5)

                targets = rects
                if collapse_targets_to_one:
                    u = _union_rect(rects)
                    targets = [u] if u else rects

                callout_rect = place_callout_near_targets(label, targets)
                _draw_connectors(page1, callout_rect, targets, mode=connector_mode)
                return

            if draw_callout_even_if_not_found:
                # Place it near top-right-ish, but still avoid collisions
                callout_rect = place_callout_near_targets(f"{label} (not found on page 1)", [])
                # no connectors

        # URL: prefer ONE connector line even if URL occurs multiple times
        source_url = (meta.get("source_url") or "").strip()
        if source_url:
            url_candidates = [source_url]
            if source_url.startswith("https://"):
                url_candidates += [source_url.replace("https://", "http://"), source_url.replace("https://", "")]
            elif source_url.startswith("http://"):
                url_candidates += [source_url.replace("http://", "https://"), source_url.replace("http://", "")]

            box_and_callout(
                "Original source of publication.",
                url_candidates,
                connector_mode="single",
                collapse_targets_to_one=True,
            )

        # Fixed criterion callouts (text only, placed near the page header area by using empty target)
        for msg in _criterion_fixed_callouts(criterion_id):
            place_callout_near_targets(msg, [])

        beneficiary_name = (meta.get("beneficiary_name") or "").strip()
        beneficiary_variants = meta.get("beneficiary_variants") or []
        venue_name = (meta.get("venue_name") or "").strip()
        org_name = (meta.get("org_name") or "").strip()
        performance_date = (meta.get("performance_date") or "").strip()
        salary_amount = (meta.get("salary_amount") or "").strip()

        if criterion_id in {"2_past", "2_future"}:
            if beneficiary_name:
                box_and_callout(
                    "Beneficiary named as a lead role in the distinguished performance.",
                    [beneficiary_name] + beneficiary_variants,
                    connector_mode="multi",
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
                    connector_mode="multi",
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


# (Optional) legacy helper retained for compatibility
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
