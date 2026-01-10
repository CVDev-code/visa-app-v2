import io
import math
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# ---- spacing knobs (tweakable) ----
GAP_FROM_TEXT_BLOCKS = 10.0   # min gap from existing text/images
GAP_FROM_HIGHLIGHTS = 14.0    # min gap from red highlight boxes
GAP_BETWEEN_CALLOUTS = 10.0   # min gap between annotations
EDGE_PAD = 18.0              # margin from physical page edge
MIN_CONNECTOR_LEN = 22.0     # if shorter, draw elbow
ENDPOINT_PULLBACK = 1.5      # pull connector end slightly away from box interior


# ============================================================
# Geometry helpers
# ============================================================

def inflate_rect(r: fitz.Rect, pad: float) -> fitz.Rect:
    rr = fitz.Rect(r)
    rr.x0 -= pad
    rr.y0 -= pad
    rr.x1 += pad
    rr.y1 += pad
    return rr


def _union_rect(rects: List[fitz.Rect]) -> fitz.Rect:
    if not rects:
        return fitz.Rect(0, 0, 0, 0)
    r = fitz.Rect(rects[0])
    for x in rects[1:]:
        r |= x
    return r


def _center(rect: fitz.Rect) -> fitz.Point:
    return fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)


def _closest_point_on_rect_edge(rect: fitz.Rect, toward: fitz.Point) -> fitz.Point:
    """
    Returns the point on rect boundary closest to 'toward'.
    """
    x = max(rect.x0, min(toward.x, rect.x1))
    y = max(rect.y0, min(toward.y, rect.y1))
    # Now ensure it's on the *edge* not inside:
    # snap to nearest edge if point is strictly inside
    if rect.x0 < x < rect.x1 and rect.y0 < y < rect.y1:
        dx_left = abs(x - rect.x0)
        dx_right = abs(rect.x1 - x)
        dy_top = abs(y - rect.y0)
        dy_bot = abs(rect.y1 - y)
        m = min(dx_left, dx_right, dy_top, dy_bot)
        if m == dx_left:
            x = rect.x0
        elif m == dx_right:
            x = rect.x1
        elif m == dy_top:
            y = rect.y0
        else:
            y = rect.y1
    return fitz.Point(x, y)


def _pull_back_point(from_pt: fitz.Point, to_pt: fitz.Point, dist: float) -> fitz.Point:
    """
    Move point 'to_pt' slightly toward 'from_pt' by 'dist'.
    Used so connector doesn't visibly enter the target box.
    """
    vx = from_pt.x - to_pt.x
    vy = from_pt.y - to_pt.y
    d = math.hypot(vx, vy)
    if d == 0:
        return to_pt
    ux, uy = vx / d, vy / d
    return fitz.Point(to_pt.x + ux * dist, to_pt.y + uy * dist)


def _draw_connector_edge_to_edge(page: fitz.Page, callout_rect: fitz.Rect, target_rect: fitz.Rect):
    """
    Draw connector from edge of callout to edge of target box.
    Ensures line does NOT pass over text inside the highlighted box.
    Uses elbow if very short.
    """
    callout_center = _center(callout_rect)
    target_center = _center(target_rect)

    start = _closest_point_on_rect_edge(callout_rect, target_center)
    end = _closest_point_on_rect_edge(target_rect, callout_center)

    # Pull end point slightly OUT of the target to avoid entering the box visually
    end = _pull_back_point(start, end, ENDPOINT_PULLBACK)

    d = math.hypot(end.x - start.x, end.y - start.y)
    if d >= MIN_CONNECTOR_LEN:
        page.draw_line(start, end, color=RED, width=1.0)
        return

    # elbow: route a short visible line segment first, away from the target direction
    # choose a mid point offset from start
    dx = -1 if target_center.x > start.x else 1
    dy = -1 if target_center.y > start.y else 1
    mid = fitz.Point(start.x + dx * 24, start.y + dy * 14)

    page.draw_line(start, mid, color=RED, width=1.0)
    page.draw_line(mid, end, color=RED, width=1.0)


# ============================================================
# Blockers (avoid covering existing content)
# ============================================================

def _page_blockers(page: fitz.Page, pad: float = GAP_FROM_TEXT_BLOCKS) -> List[fitz.Rect]:
    blockers: List[fitz.Rect] = []

    # Text blocks
    try:
        for b in page.get_text("blocks"):
            r = fitz.Rect(b[:4])
            blockers.append(inflate_rect(r, pad))
    except Exception:
        pass

    # Images
    try:
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                for r in page.get_image_rects(xref):
                    blockers.append(inflate_rect(fitz.Rect(r), pad))
            except Exception:
                pass
    except Exception:
        pass

    # Vector drawings
    try:
        for d in page.get_drawings():
            r = d.get("rect")
            if r:
                blockers.append(inflate_rect(fitz.Rect(r), pad))
    except Exception:
        pass

    return blockers


def _intersects_any(r: fitz.Rect, others: List[fitz.Rect]) -> bool:
    for o in others:
        if r.intersects(o):
            return True
    return False


# ============================================================
# Margin wrapping
# ============================================================

def _optimize_layout_for_margin(
    text: str,
    box_width: float,
    fontname: str = "Times-Roman",
) -> Tuple[int, str, float, float]:
    """
    Fixed-width wrap into box_width with fontsize 12/11/10.
    Returns (fontsize, wrapped_text, box_width, box_height)
    """
    best = (10, text or "", box_width, 40.0)
    best_h = float("inf")

    for fs in [12, 11, 10]:
        words = (text or "").split()
        if not words:
            return fs, "", box_width, 30.0

        lines: List[str] = []
        cur: List[str] = []

        usable_w = max(10.0, box_width - 10.0)

        for w in words:
            trial = " ".join(cur + [w])
            if fitz.get_text_length(trial, fontname=fontname, fontsize=fs) <= usable_w:
                cur.append(w)
            else:
                if cur:
                    lines.append(" ".join(cur))
                    cur = [w]
                else:
                    lines.append(w)
                    cur = []
        if cur:
            lines.append(" ".join(cur))

        h = (len(lines) * fs * 1.2) + 10.0

        if h < best_h:
            best_h = h
            best = (fs, "\n".join(lines), box_width, h)

    return best


def _choose_best_margin_spot(
    page: fitz.Page,
    targets: List[fitz.Rect],
    occupied: List[fitz.Rect],
    label: str,
) -> Tuple[fitz.Rect, str, int, bool]:
    """
    Prefer margin placement (left/right) near target Y.
    Enforces minimum gaps by expanding blockers and expanded occupied rectangles.
    Returns: (rect, wrapped_text, fontsize, safe_for_white_bg)
    """
    pr = page.rect
    target_union = _union_rect(targets)
    target_y = (target_union.y0 + target_union.y1) / 2

    # margin geometry
    margin_w = 120.0
    left_x = EDGE_PAD
    right_x = pr.width - EDGE_PAD - margin_w

    blockers = _page_blockers(page, pad=GAP_FROM_TEXT_BLOCKS)

    # also keep away from red highlight boxes (targets) with a bigger buffer
    for t in targets:
        blockers.append(inflate_rect(t, GAP_FROM_HIGHLIGHTS))

    # also keep away from other callouts
    occupied_buf = [inflate_rect(o, GAP_BETWEEN_CALLOUTS) for o in occupied]

    def clamp_rect(r: fitz.Rect) -> fitz.Rect:
        rr = fitz.Rect(r)
        if rr.y0 < EDGE_PAD:
            rr.y1 += (EDGE_PAD - rr.y0)
            rr.y0 = EDGE_PAD
        if rr.y1 > pr.height - EDGE_PAD:
            rr.y0 -= (rr.y1 - (pr.height - EDGE_PAD))
            rr.y1 = pr.height - EDGE_PAD
        return rr

    candidates = []
    for x_start in [left_x, right_x]:
        fs, wrapped, w, h = _optimize_layout_for_margin(label, margin_w, fontname="Times-Roman")
        cand = fitz.Rect(x_start, target_y - h / 2, x_start + w, target_y + h / 2)
        cand = clamp_rect(cand)

        safe = (not _intersects_any(cand, blockers)) and (not _intersects_any(cand, occupied_buf))

        score = abs(target_y - (cand.y0 + cand.y1) / 2)
        if not safe:
            score += 1e9

        candidates.append((score, cand, wrapped, fs, safe))

    candidates.sort(key=lambda x: x[0])
    best = candidates[0]

    # If both unsafe: choose lesser-overlap option but mark unsafe (so we won't paint white bg)
    if best[0] >= 1e9:
        soft = []
        for x_start in [left_x, right_x]:
            fs, wrapped, w, h = _optimize_layout_for_margin(label, margin_w, fontname="Times-Roman")
            cand = fitz.Rect(x_start, target_y - h / 2, x_start + w, target_y + h / 2)
            cand = clamp_rect(cand)

            overlaps = 0
            for b in blockers:
                if cand.intersects(b):
                    overlaps += 1
            for o in occupied_buf:
                if cand.intersects(o):
                    overlaps += 2

            score = abs(target_y - (cand.y0 + cand.y1) / 2) + overlaps * 5000
            soft.append((score, cand, wrapped, fs))

        soft.sort(key=lambda x: x[0])
        _, cand, wrapped, fs = soft[0]
        return cand, wrapped, fs, False

    _, cand, wrapped, fs, safe = best
    return cand, wrapped, fs, safe


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

    # A) Quote highlights across all pages
    for page in doc:
        for term in (quote_terms or []):
            t = (term or "").strip()
            if not t:
                continue
            try:
                rects = page.search_for(t)
            except Exception:
                rects = []
            for r in rects:
                page.draw_rect(r, color=RED, width=1.5)
                total_quote_hits += 1

    # B) Metadata callouts (page 1)
    def _do_job(
        label: str,
        value: Optional[str],
        *,
        connector_policy: str = "union",  # "single" | "union" | "all"
    ):
        nonlocal total_meta_hits

        if not value or not str(value).strip():
            return

        needle = str(value).strip()
        try:
            targets = page1.search_for(needle)
        except Exception:
            targets = []
        if not targets:
            return

        # draw red boxes first
        for t in targets:
            page1.draw_rect(t, color=RED, width=1.5)
        total_meta_hits += len(targets)

        # choose callout location
        callout_rect, wrapped_text, fs, safe = _choose_best_margin_spot(
            page1, targets, occupied_callouts, label
        )

        # only paint background if truly safe
        if safe:
            page1.draw_rect(callout_rect, color=WHITE, fill=WHITE, overlay=True)

        # insert annotation text (no red outline box)
        page1.insert_textbox(
            callout_rect,
            wrapped_text,
            fontname="Times-Roman",
            fontsize=fs,
            color=RED,
            align=fitz.TEXT_ALIGN_LEFT,
        )

        # connect lines (edge-to-edge)
        if connector_policy == "single":
            _draw_connector_edge_to_edge(page1, callout_rect, targets[0])

        elif connector_policy == "union":
            u = _union_rect(targets)
            _draw_connector_edge_to_edge(page1, callout_rect, u)

        elif connector_policy == "all":
            for t in targets:
                _draw_connector_edge_to_edge(page1, callout_rect, t)

        occupied_callouts.append(callout_rect)

    jobs = [
        ("Original source of publication.", meta.get("source_url"), "union"),
        ("The distinguished organization.", meta.get("venue_name") or meta.get("org_name"), "union"),
        ("Performance date.", meta.get("performance_date"), "union"),
        ("Beneficiary lead role evidence.", meta.get("beneficiary_name"), "all"),
    ]

    for label, value, policy in jobs:
        _do_job(label, value, connector_policy=policy)

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    out.seek(0)

    return out.getvalue(), {
        "total_quote_hits": total_quote_hits,
        "total_meta_hits": total_meta_hits,
        "criterion_id": criterion_id,
    }
