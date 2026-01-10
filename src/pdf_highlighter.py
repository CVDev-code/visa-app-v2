import io
import math
import re
from typing import Dict, List, Tuple, Optional, Any

import fitz  # PyMuPDF

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# ============================================================
# Settings
# ============================================================
GAP_FROM_TEXT_BLOCKS = 10.0
GAP_FROM_HIGHLIGHTS = 14.0
GAP_BETWEEN_CALLOUTS = 10.0
EDGE_PAD = 18.0
HIGHLIGHT_WIDTH = 1.5
CONNECTOR_WIDTH = 1.0
FONTNAME = "Times-Roman"
FONT_SIZES = [12, 11, 10]

# Width of the annotation box
MARGIN_W = 130.0 

# ============================================================
# Geometry Helpers
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

def _intersects_any(r: fitz.Rect, others: List[fitz.Rect]) -> bool:
    return any(r.intersects(o) for o in others)

def _get_page_blockers(page: fitz.Page) -> List[fitz.Rect]:
    """Identify all forbidden zones (text, images, vector art)."""
    blockers = []
    # 1. Text Blocks
    for b in page.get_text("blocks"):
        blockers.append(inflate_rect(fitz.Rect(b[:4]), GAP_FROM_TEXT_BLOCKS))
    # 2. Images
    for img in page.get_images(full=True):
        try:
            for r in page.get_image_rects(img[0]):
                blockers.append(inflate_rect(r, GAP_FROM_TEXT_BLOCKS))
        except: pass
    # 3. Drawings/Lines
    for d in page.get_drawings():
        r = d.get("rect")
        if r: blockers.append(inflate_rect(r, GAP_FROM_TEXT_BLOCKS))
    return blockers

# ============================================================
# Layout Logic
# ============================================================

def _optimize_layout(text: str, width: float) -> Tuple[int, str, float, float]:
    """Finds best font size to fit text in width."""
    for fs in FONT_SIZES:
        words = text.split()
        lines, cur = [], []
        usable = width - 8
        for w in words:
            if fitz.get_text_length(" ".join(cur + [w]), fontname=FONTNAME, fontsize=fs) <= usable:
                cur.append(w)
            else:
                lines.append(" ".join(cur)) if cur else lines.append(w)
                cur = [w]
        if cur: lines.append(" ".join(cur))
        h = (len(lines) * fs * 1.25) + 8
        if h < 200: # Sanity check for height
            return fs, "\n".join(lines), width, h
    return 10, text, width, 50.0

def _find_best_margin_slot(
    page: fitz.Page, 
    targets: List[fitz.Rect], 
    occupied: List[fitz.Rect], 
    box_w: float, 
    box_h: float
) -> fitz.Rect:
    """
    Scans left and right margins for the best slot closest to the target Y.
    Falls back to bottom if margins are full.
    """
    pr = page.rect
    target_union = _union_rect(targets)
    target_cy = _center(target_union).y
    
    # 1. Define Candidate Zones
    # "Left" is strictly to the left of the main text body (heuristic)
    # We use a static gutter width here, but you can make this dynamic if needed.
    zones = [
        ("left", fitz.Rect(EDGE_PAD, EDGE_PAD, EDGE_PAD + box_w, pr.height - EDGE_PAD)),
        ("right", fitz.Rect(pr.width - EDGE_PAD - box_w, EDGE_PAD, pr.width - EDGE_PAD, pr.height - EDGE_PAD))
    ]
    
    # 2. Collect Obstacles
    blockers = _get_page_blockers(page)
    # Add highlighting boxes to blockers so we don't cover them
    blockers.extend([inflate_rect(t, GAP_FROM_HIGHLIGHTS) for t in targets])
    # Add already placed callouts
    blockers.extend([inflate_rect(o, GAP_BETWEEN_CALLOUTS) for o in occupied])

    best_cand = None
    min_score = float('inf')

    # 3. Sliding Window Search
    # We slide a box of size (box_w, box_h) down the left and right margins
    step = 20 # pixels
    
    for side, zone_rect in zones:
        # X is fixed for the column (left-aligned or right-aligned)
        x0 = zone_rect.x0
        
        # Scan Y from top to bottom
        for y in range(int(zone_rect.y0), int(zone_rect.y1 - box_h), step):
            cand = fitz.Rect(x0, y, x0 + box_w, y + box_h)
            
            # Check collision
            if _intersects_any(cand, blockers):
                continue
            
            # Score: Distance from target Y
            # We penalize "crossing over" the text body (not implemented here but logic exists)
            # Primary score is vertical alignment
            cand_cy = (cand.y0 + cand.y1) / 2
            score = abs(cand_cy - target_cy)
            
            # Bonus: If target is on the left, prefer left gutter slightly
            target_cx = _center(target_union).x
            if side == "left" and target_cx < pr.width/2: score -= 50
            if side == "right" and target_cx >= pr.width/2: score -= 50

            if score < min_score:
                min_score = score
                best_cand = cand

    # 4. Fallback (Bottom of page if side margins failed)
    if not best_cand:
        # Try to pack at bottom
        y_start = pr.height - EDGE_PAD - box_h
        x_start = EDGE_PAD
        while x_start + box_w < pr.width - EDGE_PAD:
            cand = fitz.Rect(x_start, y_start, x_start + box_w, y_start + box_h)
            if not _intersects_any(cand, blockers):
                return cand
            x_start += (box_w + 10)
            
        # Ultimate Fallback (Top Corner, but we tried our best)
        return fitz.Rect(EDGE_PAD, EDGE_PAD, EDGE_PAD + box_w, EDGE_PAD + box_h)

    return best_cand

# ============================================================
# Connector Logic (Clean Straight Lines)
# ============================================================

def _draw_connector(page, callout, target, obstacles):
    # Simple midpoint-to-midpoint or closest-edge logic
    # This ensures lines don't look "wild"
    p1 = _center(callout)
    p2 = _center(target)
    
    # Adjust start point to edge of callout
    if p1.x < p2.x: p1.x = callout.x1 
    else: p1.x = callout.x0
    
    # Draw
    page.draw_line(p1, p2, color=RED, width=CONNECTOR_WIDTH)

# ============================================================
# Main Entry Point
# ============================================================

def annotate_pdf_bytes(pdf_bytes: bytes, quote_terms: List[str], criterion_id: str, meta: Dict) -> Tuple[bytes, Dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if not doc: return pdf_bytes, {}
    
    page = doc[0] # Assuming single page processing for this snippet
    occupied = []

    # 1. Highlight Quotes (Yellow/Red)
    # (Existing logic...)

    # 2. Place Callouts
    jobs = [
        ("Source", meta.get("source_url")),
        ("Venue", meta.get("venue_name")),
        ("Date", meta.get("performance_date")),
    ]

    for label, val in jobs:
        if not val: continue
        
        # Search for text
        targets = page.search_for(val)
        if not targets: continue
        
        # Highlight Targets
        for t in targets:
            page.draw_rect(t, color=RED, width=1.5)

        # Layout Calculation
        fs, wrapped_text, w, h = _optimize_layout(label, MARGIN_W)
        
        # GEOMETRIC PLACEMENT (Replaces LLM)
        callout_rect = _find_best_margin_slot(page, targets, occupied, w, h)
        
        # Draw Callout
        page.draw_rect(callout_rect, color=WHITE, fill=WHITE, overlay=True) # Eraser
        page.insert_textbox(callout_rect, wrapped_text, fontsize=fs, color=RED, fontname=FONTNAME)
        
        # Draw Connector
        _draw_connector(page, callout_rect, _union_rect(targets), [])
        
        occupied.append(callout_rect)

    out = io.BytesIO()
    doc.save(out)
    return out.getvalue(), {}
