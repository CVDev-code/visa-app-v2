import io
import math
from typing import Dict, List, Tuple, Optional
import fitz  # PyMuPDF

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# ---- spacing knobs ----
GAP_FROM_TEXT_BLOCKS = 12.0   
GAP_FROM_HIGHLIGHTS = 18.0    
GAP_BETWEEN_CALLOUTS = 12.0   
EDGE_PAD = 20.0              
ENDPOINT_PULLBACK = 2.0      

# ============================================================
# Geometry helpers
# ============================================================

def inflate_rect(r: fitz.Rect, pad: float) -> fitz.Rect:
    return fitz.Rect(r.x0 - pad, r.y0 - pad, r.x1 + pad, r.y1 + pad)

def _union_rect(rects: List[fitz.Rect]) -> fitz.Rect:
    if not rects: return fitz.Rect(0, 0, 0, 0)
    r = fitz.Rect(rects[0])
    for x in rects[1:]: r |= x
    return r

def _center(rect: fitz.Rect) -> fitz.Point:
    return fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)

def _closest_point_on_rect_edge(rect: fitz.Rect, toward: fitz.Point) -> fitz.Point:
    x = max(rect.x0, min(toward.x, rect.x1))
    y = max(rect.y0, min(toward.y, rect.y1))
    if rect.x0 < x < rect.x1 and rect.y0 < y < rect.y1:
        dists = [abs(x - rect.x0), abs(rect.x1 - x), abs(y - rect.y0), abs(rect.y1 - y)]
        m = min(dists)
        if m == dists[0]: x = rect.x0
        elif m == dists[1]: x = rect.x1
        elif m == dists[2]: y = rect.y0
        else: y = rect.y1
    return fitz.Point(x, y)

def _draw_connector_edge_to_edge(page: fitz.Page, callout_rect: fitz.Rect, target_rect: fitz.Rect, fs: int):
    """
    FIX 1: Connects to the vertical center of the text content, not the box center.
    We assume the first line starts near the top, so we anchor to (y0 + fs).
    """
    # Anchor to the vertical middle of the first few lines of text
    anchor_y = callout_rect.y0 + (fs * 0.8) 
    callout_anchor = fitz.Point(_center(callout_rect).x, anchor_y)
    target_center = _center(target_rect)

    start = _closest_point_on_rect_edge(callout_rect, target_center)
    # Lock the start 'y' to our text-aligned anchor if it's a side-exit
    if start.x == callout_rect.x0 or start.x == callout_rect.x1:
        start.y = anchor_y

    end = _closest_point_on_rect_edge(target_rect, callout_anchor)
    
    # Visual pullback
    vx, vy = callout_anchor.x - end.x, callout_anchor.y - end.y
    mag = math.hypot(vx, vy)
    if mag > 0:
        end = fitz.Point(end.x + (vx/mag)*ENDPOINT_PULLBACK, end.y + (vy/mag)*ENDPOINT_PULLBACK)

    page.draw_line(start, end, color=RED, width=1.0)

# ============================================================
# Logic & Layout
# ============================================================

def _page_blockers(page: fitz.Page, pad: float) -> List[fitz.Rect]:
    blockers = []
    for b in page.get_text("blocks"):
        blockers.append(inflate_rect(fitz.Rect(b[:4]), pad))
    for img in page.get_images(full=True):
        for r in page.get_image_rects(img[0]):
            blockers.append(inflate_rect(fitz.Rect(r), pad))
    return blockers

def _choose_best_margin_spot(page: fitz.Page, targets: List[fitz.Rect], occupied: List[fitz.Rect], label: str) -> Tuple[fitz.Rect, str, int, bool]:
    pr = page.rect
    target_union = _union_rect(targets)
    target_y = (target_union.y0 + target_union.y1) / 2
    margin_w = 115.0
    
    # FIX 2: Create a 'shadow' blocker. 
    # Any text block between the margin and the target is a high-penalty zone.
    all_page_items = _page_blockers(page, GAP_FROM_TEXT_BLOCKS)
    
    candidates = []
    for x_start in [EDGE_PAD, pr.width - EDGE_PAD - margin_w]:
        fs, wrapped, w, h = _optimize_layout_for_margin(label, margin_w)
        cand = fitz.Rect(x_start, target_y - h/2, x_start + w, target_y + h/2)
        
        # Clamp to page
        if cand.y0 < EDGE_PAD: cand.y1 += (EDGE_PAD - cand.y0); cand.y0 = EDGE_PAD
        if cand.y1 > pr.height - EDGE_PAD: cand.y0 -= (cand.y1 - (pr.height - EDGE_PAD)); cand.y1 = pr.height - EDGE_PAD

        # Penalty for crossing other highlights
        overlap_penalty = 0
        for b in all_page_items + occupied + targets:
            # Check if the annotation overlaps OR if it's too close vertically
            if cand.intersects(inflate_rect(b, 5)):
                overlap_penalty += 1e7
            
            # Line-of-sight check: if a horizontal line from cand to target hits 'b'
            line_zone = fitz.Rect(min(cand.x1, b.x1), cand.y0, max(cand.x0, b.x0), cand.y1)
            if line_zone.intersects(b):
                overlap_penalty += 5000 # Significant but allows it if no other choice

        score = abs(target_y - (cand.y0 + cand.y1)/2) + overlap_penalty
        candidates.append((score, cand, wrapped, fs, overlap_penalty < 1e6))

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][2], candidates[0][3], candidates[0][4]

# [Layout Logic / Optimize Layout stays the same as previous working version]
def _optimize_layout_for_margin(text: str, box_width: float) -> Tuple[int, str, float, float]:
    for fs in [12, 11, 10]:
        words = text.split()
        lines, cur = [], []
        for w in words:
            if fitz.get_text_length(" ".join(cur + [w]), fontname="Times-Roman", fontsize=fs) <= (box_width - 10):
                cur.append(w)
            else:
                lines.append(" ".join(cur)); cur = [w]
        lines.append(" ".join(cur))
        h = (len(lines) * fs * 1.2) + 10.0
        if fs == 10 or h < 100: # heuristic
            return fs, "\n".join(lines), box_width, h
    return 10, text, box_width, 40.0

def annotate_pdf_bytes(pdf_bytes: bytes, quote_terms: List[str], criterion_id: str, meta: Dict) -> Tuple[bytes, Dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page1 = doc.load_page(0)
    occupied_callouts = []
    
    # Highlight logic... (simplified for brevity)
    # ...
    
    jobs = [
        ("Original source of publication.", meta.get("source_url"), "union"),
        ("Beneficiary lead role evidence.", meta.get("beneficiary_name"), "all"),
    ]

    for label, value, policy in jobs:
        if not value: continue
        targets = page1.search_for(str(value).strip())
        if not targets: continue
        
        for t in targets: page1.draw_rect(t, color=RED, width=1.5)
        
        callout_rect, wrapped_text, fs, safe = _choose_best_margin_spot(page1, targets, occupied_callouts, label)
        
        if safe:
            page1.draw_rect(callout_rect, color=WHITE, fill=WHITE, overlay=True)
        
        page1.insert_textbox(callout_rect, wrapped_text, fontname="Times-Roman", fontsize=fs, color=RED)
        
        # Connect
        if policy == "all":
            for t in targets: _draw_connector_edge_to_edge(page1, callout_rect, t, fs)
        else:
            _draw_connector_edge_to_edge(page1, callout_rect, _union_rect(targets), fs)
            
        occupied_callouts.append(callout_rect)

    out = io.BytesIO()
    doc.save(out)
    return out.getvalue(), {}
