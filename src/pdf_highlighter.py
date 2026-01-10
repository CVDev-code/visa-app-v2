import io
import math
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF

RED = (1, 0, 0)
WHITE = (1, 1, 1)


# ============================================================
# Geometry helpers
# ============================================================

def _union_rect(rects: List[fitz.Rect]) -> fitz.Rect:
    r = fitz.Rect(rects[0])
    for x in rects[1:]:
        r |= x
    return r


def _center(rect: fitz.Rect) -> fitz.Point:
    return fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)


def _get_closest_point(rect: fitz.Rect, target: fitz.Point) -> fitz.Point:
    x = max(rect.x0, min(target.x, rect.x1))
    y = max(rect.y0, min(target.y, rect.y1))
    return fitz.Point(x, y)


def _draw_neat_connector(page: fitz.Page, callout_rect: fitz.Rect, target_rect: fitz.Rect):
    """
    Draws a visible connector line even when the callout is very close to the target.
    If too close, uses a 2-segment elbow to avoid a 'squashed' tiny line.
    """
    target = _center(target_rect)
    start = _get_closest_point(callout_rect, target)
    d = math.hypot(target.x - start.x, target.y - start.y)

    if d >= 24:
        page.draw_line(start, target, color=RED, width=1.0)
        return

    # elbow: push outward from the callout by a fixed amount, away from target
    dx = -1 if target.x > start.x else 1
    dy = -1 if target.y > start.y else 1
    mid = fitz.Point(start.x + dx * 22, start.y + dy * 12)

    page.draw_line(start, mid, color=RED, width=1.0)
    page.draw_line(mid, target, color=RED, width=1.0)


# ============================================================
# Blockers (to avoid covering text/images)
# ============================================================

def _page_blockers(page: fitz.Page, pad: float = 4.0) -> List[fitz.Rect]:
    """
    Collect rectangles that represent 'do not cover' areas:
    - text blocks
    - images
    - vector drawings
    Adds padding around each.
    """
    blockers: List[fitz.Rect] = []

    # text blocks
    try:
        for b in page.get_text("blocks"):
            r = fitz.Rect(b[:4])
            r.x0 -= pad
            r.y0 -= pad
            r.x1 += pad
            r.y1 += pad
            blockers.append(r)
    except Exception:
        pass

    # image rectangles
    try:
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                for r in page.get_image_rects(xref):
                    rr = fitz.Rect(r)
                    rr.x0 -= pad
                    rr.y0 -= pad
                    rr.x1 += pad
                    rr.y1 += pad
                    blockers.append(rr)
            except Exception:
                pass
    except Exception:
        pass

    # vector drawings
    try:
        for d in page.get_drawings():
            r = d.get("rect")
            if r:
                rr = fitz.Rect(r)
                rr.x0 -= pad
                rr.y0 -= pad
                rr.x1 += pad
                rr.y1 += pad
                blockers.append(rr)
    except Exception:
        pass

    return blockers


def _intersects_any(r: fitz.Rect, others: List[fitz.Rect]) -> bool:
    for o in others:
        if r.intersects(o):
            return True
    return False


# ============================================================
# Margin layout / wrapping
# ============================================================

def _optimize_layout_for_margin(
    text: str,
    target_margin_width: float,
    fontname: str = "Times-Roman",
) -> Tuple[int, str, float, float]:
    """
    Wrap text into a FIXED width textbox (target_margin_width), choosing fontsize 12/11/10.
    Returns: (fontsize, wrapped_text, box_width, box_height)
    """
    best = (10, text or "", target_margin_width, 40.0)
    best_height = float("inf")

    for fs in [12, 11, 10]:
        words = (text or "").split()
        if not words:
            return fs, "", target_margin_width, 30.0

        lines: List[str] = []
        cur: List[str] = []

        # wrap to width (leave small internal padding)
        max_w = max(10.0, target_margin_width - 6.0)

        for w in words:
            trial = " ".join(cur + [w])
            if fitz.get_text_length(trial, fontname=fontname, fontsize=fs) <= max_w:
                cur.append(w)
            else:
                if cur:
                    lines.append(" ".join(cur))
                    cur = [w]
                else:
                    # one long word; force it on its own line
                    lines.append(w)
                    cur = []

        if cur:
            lines.append(" ".join(cur))

        # height estimate: line height ~ fs*1.2 + padding
        h = (len(lines) * fs * 1.2) + 8.0

        if h < best_height:
            best_height = h
            best = (fs, "\n".join(lines), target_margin_width, h)

    return best


def _choose_best_margin_spot(
    page: fitz.Page,
    targets: List[fitz.Rect],
    occupied: List[fitz.Rect],
    label: str,
) -> Tuple[fitz.Rect, str, int, bool]:
    """
    Pick a callout rectangle in the LEFT or RIGHT margin near the target's Y position.
    Hard-avoids blockers (text/images/drawings) and occupied callouts when possible.
    Returns (rect, wrapped_text, fontsize, safe_to_paint_white_bg)
    """
    pr = page.rect
    target_union = _union_rect(targets)
    target_y = (target_union.y0 + target_union.y1) / 2

    # widen margin area a bit (helps avoid tall wraps)
    margin_w = 120.0
    side_pad = 16.0
    top_bot_pad = 16.0

    left_x = side_pad
    right_x = pr.width - side_pad - margin_w

    blockers = _page_blockers(page, pad=4.0)

    def make_candidate(x_start: float) -> Tuple[float, fitz.Rect, str, int, bool]:
        fs, wrapped, w, h = _optimize_layout_for_margin(label, margin_w, fontname="Times-Roman")

        # fixed width box in margin
        cand = fitz.Rect(x_start, target_y - h / 2, x_start + w, target_y + h / 2)

        # clamp vertically
        if cand.y0 < top_bot_pad:
            cand.y1 += (top_bot_pad - cand.y0)
            cand.y0 = top_bot_pad
        if cand.y1 > pr.height - top_bot_pad:
            cand.y0 -= (cand.y1 - (pr.height - top_bot_pad))
            cand.y1 = pr.height - top_bot_pad

        # "safe" means doesn't overlap blockers/occupied
        safe = not _intersects_any(cand, occupied) and not _intersects_any(cand, blockers)

        # score: prefer being close in y + safe; unsafe gets huge penalty
        score = abs(target_y - (cand.y0 + cand.y1) / 2)
        if not safe:
            score += 1e9

        return score, cand, wrapped, fs, safe

    candidates = [
        make_candidate(left_x),
        make_candidate(right_x),
    ]
    candidates.sort(key=lambda x: x[0])

    best_score, best_rect, best_text, best_fs, best_safe = candidates[0]

    # If both candidates are unsafe, pick the "less bad" one and mark unsafe;
    # we will avoid painting a white background in that case.
    if best_score >= 1e9:
        # re-score without the hard penalty to choose less collision
        soft_candidates = []
        for x_start in [left_x, right_x]:
            fs, wrapped, w, h = _optimize_layout_for_margin(label, margin_w, fontname="Times-Roman")
            cand = fitz.Rect(x_start, target_y - h / 2, x_start + w, target_y + h / 2)
            if cand.y0 < top_bot_pad:
                cand.y1 += (top_bot_pad - cand.y0)
                cand.y0 = top_bot_pad
            if cand.y1 > pr.height - top_bot_pad:
                cand.y0 -= (cand.y1 - (pr.height - top_bot_pad))
                cand.y1 = pr.height - top_bot_pad

            overlap_count = 0
            for o in occupied:
                if cand.intersects(o):
                    overlap_count += 3
            for b in blockers:
                if cand.intersects(b):
                    overlap_count += 1

            score = abs(target_y - (cand.y0 + cand.y1) / 2) + overlap_count * 5000
            soft_candidates.append((score, cand, wrapped, fs))

        soft_candidates.sort(key=lambda x: x[0])
        _, best_rect, best_text, best_fs = soft_candidates[0]
        best_safe = False

    return best_rect, best_text, best_fs, best_safe


# ============================================================
# Main annotation entrypoint
# ============================================================

def annotate_pdf_bytes(
    pdf_bytes: bytes,
    quote_terms: List[str],
    criterion_id: str,
    meta: Dict,
) -> Tuple[bytes, Dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(doc) == 0:
        return pdf_bytes, {}

    page1 = doc.load_page(0)
    occupied_callouts: List[fitz.Rect] = []
    total_quote_hits = 0
    total_meta_hits = 0

    # A) Global quote highlights (all pages)
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        for term in (quote_terms or []):
            t = (term or "").strip()
            if not t:
                continue
            try:
                found = page.search_for(t)
            except Exception:
                found = []
            for r in found:
                page.draw_rect(r, color=RED, width=1.5)
                total_quote_hits += 1

    # B) Page-1 metadata callouts (margin priority)
    def _do_job(
        label: str,
        value: Optional[str],
        *,
        connector_policy: str = "union",  # "single" | "union" | "all"
        highlight_all: bool = True,
    ):
        nonlocal total_meta_hits

        if not value or not str(value).strip():
            return

        val = str(value).strip()

        try:
            targets = page1.search_for(val)
        except Exception:
            targets = []

        if not targets:
            return

        # highlight targets
        if highlight_all:
            for t in targets:
                page1.draw_rect(t, color=RED, width=1.5)
        else:
            page1.draw_rect(targets[0], color=RED, width=1.5)

        total_meta_hits += len(targets) if highlight_all else 1

        # choose callout spot near targets (margin)
        callout_rect, wrapped_text, fs, safe = _choose_best_margin_spot(
            page1, targets, occupied_callouts, label
        )

        # only paint background if safe (prevents covering underlying text/images)
        if safe:
            page1.draw_rect(callout_rect, color=WHITE, fill=WHITE, overlay=True)

        # annotation text (no red box around it)
        page1.insert_textbox(
            callout_rect,
            wrapped_text,
            fontname="Times-Roman",
            fontsize=fs,
            color=RED,
            align=fitz.TEXT_ALIGN_LEFT,
        )

        # connectors
        if connector_policy == "single":
            _draw_neat_connector(page1, callout_rect, targets[0])

        elif connector_policy == "union":
            u = _union_rect(targets)
            _draw_neat_connector(page1, callout_rect, u)

        elif connector_policy == "all":
            for t in targets:
                _draw_neat_connector(page1, callout_rect, t)

        occupied_callouts.append(callout_rect)

    # Jobs (you can refine labels per criterion later)
    # - URL: ONE line (union/single), even if multiple matches
    _do_job(
        "Original source of publication.",
        meta.get("source_url"),
        connector_policy="union",
        highlight_all=True,
    )

    # - Venue/Org: usually one mention, keep single/union
    _do_job(
        "The distinguished organization.",
        meta.get("venue_name") or meta.get("org_name"),
        connector_policy="union",
        highlight_all=True,
    )

    # - Performance date: single line
    _do_job(
        "Performance date.",
        meta.get("performance_date"),
        connector_policy="union",
        highlight_all=True,
    )

    # - Beneficiary: connect to ALL occurrences (often multiple)
    _do_job(
        "Beneficiary lead role evidence.",
        meta.get("beneficiary_name"),
        connector_policy="all",
        highlight_all=True,
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
