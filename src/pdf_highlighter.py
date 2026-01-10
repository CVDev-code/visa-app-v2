import io
import math
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# ---- spacing knobs (tweakable) ----
GAP_FROM_TEXT_BLOCKS = 10.0   # min gap from existing text/images for callout placement
GAP_FROM_HIGHLIGHTS = 14.0    # min gap from red highlight boxes for callout placement
GAP_BETWEEN_CALLOUTS = 10.0   # min gap between callouts
EDGE_PAD = 18.0              # padding from page edge
ENDPOINT_PULLBACK = 1.5      # pull connector end slightly away from target edge
MIN_CONNECTOR_LEN = 22.0     # if shorter, draw elbow


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


def _segment_intersects_rect(p1: fitz.Point, p2: fitz.Point, r: fitz.Rect) -> bool:
    """
    Cheap intersection test: sample points along the segment and see if any land inside r.
    Good enough for routing callout connectors.
    """
    steps = 18
    for i in range(steps + 1):
        t = i / steps
        x = p1.x + (p2.x - p1.x) * t
        y = p1.y + (p2.y - p1.y) * t
        if r.contains(fitz.Point(x, y)):
            return True
    return False


def _pull_back_point(from_pt: fitz.Point, to_pt: fitz.Point, dist: float) -> fitz.Point:
    vx = from_pt.x - to_pt.x
    vy = from_pt.y - to_pt.y
    d = math.hypot(vx, vy)
    if d == 0:
        return to_pt
    ux, uy = vx / d, vy / d
    return fitz.Point(to_pt.x + ux * dist, to_pt.y + uy * dist)


def _callout_mid_edge_anchor(callout: fitz.Rect, target_center: fitz.Point) -> fitz.Point:
    """
    FIX #1: Anchor on the mid-height of the callout box (not text center).
    Choose left or right edge depending on where the target lies.
    """
    y = callout.y0 + (callout.height / 2.0)
    if target_center.x >= (callout.x0 + callout.width / 2.0):
        # target to the right -> anchor on right edge
        return fitz.Point(callout.x1, y)
    else:
        # target to the left -> anchor on left edge
        return fitz.Point(callout.x0, y)


def _target_edge_candidates(target: fitz.Rect) -> List[fitz.Point]:
    """
    Candidate attachment points on the target edge (midpoints + corners-ish).
    """
    cx = target.x0 + target.width / 2.0
    cy = target.y0 + target.height / 2.0
    return [
        fitz.Point(target.x0, cy),  # left mid
        fitz.Point(target.x1, cy),  # right mid
        fitz.Point(cx, target.y0),  # top mid
        fitz.Point(cx, target.y1),  # bottom mid
        # slight corner bias points (helps when boxes touch)
        fitz.Point(target.x0, target.y0),
        fitz.Point(target.x1, target.y0),
        fitz.Point(target.x0, target.y1),
        fitz.Point(target.x1, target.y1),
    ]


def _choose_target_attachment(
    start: fitz.Point,
    target: fitz.Rect,
    obstacles: List[fitz.Rect],
) -> fitz.Point:
    """
    FIX #2: Choose an endpoint on the target edge that avoids crossing other highlight boxes.
    Tries multiple edge points and picks the one with the fewest segment intersections.
    """
    best_pt = _center(target)
    best_hits = 10**9
    best_len = 10**9

    for pt in _target_edge_candidates(target):
        hits = 0
        for ob in obstacles:
            if _segment_intersects_rect(start, pt, ob):
                hits += 1

        seg_len = math.hypot(pt.x - start.x, pt.y - start.y)

        # primary: minimize crossings; secondary: shorter line
        if hits < best_hits or (hits == best_hits and seg_len < best_len):
            best_hits = hits
            best_len = seg_len
            best_pt = pt

    return best_pt


def _draw_connector_routed(
    page: fitz.Page,
    callout_rect: fitz.Rect,
    target_rect: fitz.Rect,
    obstacles: List[fitz.Rect],
):
    """
    Draw a connector from callout mid-edge to target edge.
    If the straight segment crosses obstacles or is too short, route as an elbow.
    """
    target_center = _center(target_rect)
    start = _callout_mid_edge_anchor(callout_rect, target_center)

    end = _choose_target_attachment(start, target_rect, obstacles)
    end = _pull_back_point(start, end, ENDPOINT_PULLBACK)

    seg_len = math.hypot(end.x - start.x, end.y - start.y)

    crosses = any(_segment_intersects_rect(start, end, ob) for ob in obstacles)

    if (not crosses) and seg_len >= MIN_CONNECTOR_LEN:
        page.draw_line(start, end, color=RED, width=1.0)
        return

    # Elbow routing: push outward in x away from content, then to end
    # Choose elbow x outside the target direction
    if start.x < target_center.x:
        elbow_x = min(callout_rect.x1 + 24.0, page.rect.width - EDGE_PAD)
    else:
        elbow_x = max(callout_rect.x0 - 24.0, EDGE_PAD)

    mid1 = fitz.Point(elbow_x, start.y)
    mid2 = fitz.Point(elbow_x, end.y)

    # If mid segments still cross, nudge elbow_y slightly
    if any(_segment_intersects_rect(start, mid1, ob) for ob in obstacles):
        mid1 = fitz.Point(mid1.x, mid1.y - 18.0)
    if any(_segment_intersects_rect(mid2, end, ob) for ob in obstacles):
        mid2 = fitz.Point(mid2.x, mid2.y + 18.0)

    page.draw_line(start, mid1, color=RED, width=1.0)
    page.draw_line(mid1, mid2, color=RED, width=1.0)
    page.draw_line(mid2, end, color=RED, width=1.0)


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
    Enforces minimum gaps by expanding blockers and occupied rectangles.
    Returns: (rect, wrapped_text, fontsize, safe_for_white_bg)
    """
    pr = page.rect
    target_union = _union_rect(targets)
    target_y = (target_union.y0 + target_union.y1) / 2

    margin_w = 120.0
    left_x = EDGE_PAD
    right_x = pr.width - EDGE_PAD - margin_w

    blockers = _page_blockers(page, pad=GAP_FROM_TEXT_BLOCKS)

    # keep away from red highlight boxes (targets)
    for t in targets:
        blockers.append(inflate_rect(t, GAP_FROM_HIGHLIGHTS))

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

    if best[0] >= 1e9:
        # fallback: pick "least overlapping" but don't paint background
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

        # draw red boxes
        for t in targets:
            page1.draw_rect(t, color=RED, width=1.5)
        total_meta_hits += len(targets)

        # choose callout location
        callout_rect, wrapped_text, fs, safe = _choose_best_margin_spot(
            page1, targets, occupied_callouts, label
        )

        if safe:
            page1.draw_rect(callout_rect, color=WHITE, fill=WHITE, overlay=True)

        page1.insert_textbox(
            callout_rect,
            wrapped_text,
            fontname="Times-Roman",
            fontsize=fs,
            color=RED,
            align=fitz.TEXT_ALIGN_LEFT,
        )

        # Obstacles for connectors:
        # everything highlighted on page1 (including OTHER targets) is an obstacle
        # so the connector doesn't cut through adjacent red boxes.
        obstacle_rects: List[fitz.Rect] = []
        try:
            # Use text blocks as general obstacles too (helps avoid cutting through text)
            for b in page1.get_text("blocks"):
                obstacle_rects.append(inflate_rect(fitz.Rect(b[:4]), 2.0))
        except Exception:
            pass

        # Add the red highlight rects as obstacles (expanded a bit)
        # IMPORTANT: exclude the target itself when connecting to it
        expanded_targets = [inflate_rect(t, 2.0) for t in targets]

        def connect_one(target_rect: fitz.Rect):
            obs = obstacle_rects[:]
            # other target boxes as obstacles (expanded)
            for ot in expanded_targets:
                if not ot.intersects(target_rect):
                    obs.append(ot)
            # also avoid callout boxes
            for oc in occupied_callouts:
                obs.append(inflate_rect(oc, 2.0))
            _draw_connector_routed(page1, callout_rect, target_rect, obs)

        if connector_policy == "all":
            for t in targets:
                connect_one(t)
        elif connector_policy == "single":
            connect_one(targets[0])
        else:
            connect_one(_union_rect(targets))

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
