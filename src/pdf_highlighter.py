# src/pdf_highlighter.py
import io
import math
import re
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# ============================================================
# Style + spacing knobs
# ============================================================
BOX_WIDTH = 1.8
LINE_WIDTH = 1.8
FONTNAME = "Times-Roman"
FONT_SIZES = [12, 11, 10]
EDGE_PAD = 18.0

GAP_FROM_TEXT_BLOCKS = 14.0
GAP_FROM_IMAGES = 12.0
GAP_FROM_DRAWINGS = 10.0
GAP_FROM_HIGHLIGHTS = 16.0
GAP_BETWEEN_CALLOUTS = 12.0
ENDPOINT_PULLBACK = 2.0

_MAX_TERM = 600
_CHUNK = 70
_CHUNK_OVERLAP = 22

# ============================================================
# Ink-check knobs (Progressive)
# ============================================================
INKCHECK_DPI = 90
INKCHECK_PAD = 1.5
INKCHECK_LEVELS = [
    (250, 0.002),  # strict
    (245, 0.006),  # relaxed
    (235, 0.010),  # more relaxed
]

# ============================================================
# Geometry helpers
# ============================================================

def inflate_rect(r: fitz.Rect, pad: float) -> fitz.Rect:
    rr = fitz.Rect(r)
    rr.x0 -= pad; rr.y0 -= pad; rr.x1 += pad; rr.y1 += pad
    return rr

def _union_rect(rects: List[fitz.Rect]) -> fitz.Rect:
    if not rects: return fitz.Rect(0, 0, 0, 0)
    r = fitz.Rect(rects[0])
    for x in rects[1:]: r |= x
    return r

def _center(rect: fitz.Rect) -> fitz.Point:
    return fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)

def _segment_hits_rect(p1: fitz.Point, p2: fitz.Point, r: fitz.Rect) -> bool:
    steps = 26
    for i in range(steps + 1):
        t = i / steps
        x = p1.x + (p2.x - p1.x) * t
        y = p1.y + (p2.y - p1.y) * t
        if r.contains(fitz.Point(x, y)): return True
    return False

def _pull_back_point(from_pt: fitz.Point, to_pt: fitz.Point, dist: float) -> fitz.Point:
    vx, vy = from_pt.x - to_pt.x, from_pt.y - to_pt.y
    d = math.hypot(vx, vy)
    if d == 0: return to_pt
    return fitz.Point(to_pt.x + (vx/d) * dist, to_pt.y + (vy/d) * dist)

def _edge_candidates(rect: fitz.Rect) -> List[fitz.Point]:
    cx, cy = rect.x0 + rect.width / 2.0, rect.y0 + rect.height / 2.0
    return [fitz.Point(rect.x0, cy), fitz.Point(rect.x1, cy), fitz.Point(cx, rect.y0), fitz.Point(cx, rect.y1)]

def _mid_height_anchor(callout: fitz.Rect, toward: fitz.Point) -> fitz.Point:
    y = callout.y0 + (callout.height / 2.0)
    return fitz.Point(callout.x1 if toward.x >= (callout.x0 + callout.width/2) else callout.x0, y)

def _straight_connector_best_pair(callout_rect, target_rect, obstacles):
    target_center = _center(target_rect)
    base = _mid_height_anchor(callout_rect, target_center)
    starts = [fitz.Point(base.x, min(max(callout_rect.y0 + 2.0, base.y + dy), callout_rect.y1 - 2.0)) for dy in [-14, 0, 14]]
    ends = _edge_candidates(target_rect)
    best_hits, best_len, best_s, best_e = 10**9, 10**9, starts[0], ends[0]
    for s in starts:
        for e in ends:
            hits = sum(1 for ob in obstacles if _segment_hits_rect(s, e, ob))
            length = math.hypot(e.x - s.x, e.y - s.y)
            if hits < best_hits or (hits == best_hits and length < best_len):
                best_hits, best_len, best_s, best_e = hits, length, s, e
    return best_s, _pull_back_point(best_s, best_e, ENDPOINT_PULLBACK), best_hits, best_len

def _draw_straight_connector(page, callout_rect, target_rect, obstacles):
    s, e, _, _ = _straight_connector_best_pair(callout_rect, target_rect, obstacles)
    page.draw_line(s, e, color=RED, width=LINE_WIDTH)

# ============================================================
# Ink check
# ============================================================

def _rect_has_ink(page, rect, dpi=INKCHECK_DPI, nonwhite_ratio=0.002, white_threshold=250) -> bool:
    r = fitz.Rect(rect) & page.rect
    if r.is_empty or r.width < 2: return True
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), clip=r, alpha=False)
    s = pix.samples
    if not s: return True
    nonwhite = sum(1 for i in range(0, len(s), 3) if any(v < white_threshold for v in s[i:i+3]))
    return (nonwhite / (pix.width * pix.height)) > nonwhite_ratio

# ============================================================
# Placement & Zones (The Fix)
# ============================================================

def _page_blockers(page):
    blockers = []
    for r in page.get_text("blocks"): blockers.append(inflate_rect(fitz.Rect(r[:4]), GAP_FROM_TEXT_BLOCKS))
    for img in page.get_images(): 
        for r in page.get_image_rects(img[0]): blockers.append(inflate_rect(r, GAP_FROM_IMAGES))
    return blockers

def _choose_best_spot(page, targets, occupied_callouts, label):
    pr = page.rect
    target_union = _union_rect(targets)
    tc = _center(target_union)
    blockers = _page_blockers(page)
    highlight_blockers = [inflate_rect(t, GAP_FROM_HIGHLIGHTS) for t in targets]
    occupied_buf = [inflate_rect(o, GAP_BETWEEN_CALLOUTS) for o in occupied_callouts]

    # Simple Zones
    zones = [
        ("left", fitz.Rect(EDGE_PAD, EDGE_PAD, 120, pr.height - EDGE_PAD)),
        ("right", fitz.Rect(pr.width - 120, EDGE_PAD, pr.width - EDGE_PAD, pr.height - EDGE_PAD))
    ]

    for threshold, ratio in INKCHECK_LEVELS:
        candidates = []
        for name, z in zones:
            fs, wrapped, w, h = 10, label, 100, 40 # Simplified for logic
            # Scan vertically near the target text
            for dy in [0, -40, 40, -80, 80, -120, 120]:
                cy = tc.y + dy
                if not (z.y0 + h < cy < z.y1 - h): continue
                cand = fitz.Rect(z.x0, cy - h/2, z.x0 + w, cy + h/2)
                
                if any(cand.intersects(b) for b in blockers + occupied_buf + highlight_blockers): continue
                if _rect_has_ink(page, cand, white_threshold=threshold, nonwhite_ratio=ratio): continue
                
                score = abs(dy) # Prioritize closest vertical match
                candidates.append((score, cand, wrapped, fs))
        
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1], candidates[0][2], candidates[0][3], True

    # Spread Fallback (Prevents top-corner stack)
    y_offset = EDGE_PAD + (len(occupied_callouts) * 60)
    return fitz.Rect(EDGE_PAD, y_offset, EDGE_PAD + 100, y_offset + 40), label, 10, False

# ============================================================
# Main Logic
# ============================================================

def annotate_pdf_bytes(pdf_bytes: bytes, quote_terms: List[str], criterion_id: str, meta: Dict):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(doc) == 0: return pdf_bytes, {}
    
    page1 = doc.load_page(0)
    occupied_callouts = []

    def _do_job(label, needles):
        nonlocal occupied_callouts
        targets = []
        for n in needles:
            if n: targets.extend(page1.search_for(n))
        if not targets: return

        for t in targets: page1.draw_rect(t, color=RED, width=BOX_WIDTH)
        
        rect, text, fs, safe = _choose_best_spot(page1, targets, occupied_callouts, label)
        if safe: page1.draw_rect(rect, color=WHITE, fill=WHITE, overlay=True)
        
        page1.insert_textbox(rect, text, fontname=FONTNAME, fontsize=fs, color=RED)
        _draw_straight_connector(page1, rect, _union_rect(targets), [])
        occupied_callouts.append(rect)

    # Example job
    source = meta.get("source_url", "")
    if source: _do_job("Source URL", [source])
    
    venue = meta.get("venue_name", "")
    if venue: _do_job("Organization", [venue])

    out = io.BytesIO()
    doc.save(out)
    return out.getvalue(), {"criterion_id": criterion_id}
