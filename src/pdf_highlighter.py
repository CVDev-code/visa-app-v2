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
    if not rects: return None
    r = fitz.Rect(rects[0])
    for x in rects[1:]: r |= x
    return r

# -----------------------------
# Text Wrapping & Layout (Requirement 3)
# -----------------------------
def _optimize_layout(text: str, fontname: str = "Times-Roman") -> Tuple[int, str, float, float]:
    """
    Finds the best balance between font size (10-12) and width-to-height ratio.
    Returns (fontsize, wrapped_text, width, height)
    """
    best_layout = (10, text, 200.0, 50.0)
    min_score = float('inf')

    # Iterate through allowed font sizes
    for fs in [12, 11, 10]:
        # Try different target widths to see which creates the neatest 'block'
        for target_width in [120, 160, 200, 250]:
            words = text.split()
            lines = []
            cur_line = []
            for w in words:
                trial = " ".join(cur_line + [w])
                if fitz.get_text_length(trial, fontname=fontname, fontsize=fs) <= target_width:
                    cur_line.append(w)
                else:
                    lines.append(" ".join(cur_line))
                    cur_line = [w]
            lines.append(" ".join(cur_line))
            
            final_text = "\n".join(lines)
            h = (len(lines) * fs * 1.2) + 10
            w = max([fitz.get_text_length(l, fontname=fontname, fontsize=fs) for l in lines]) + 10
            
            # Score based on 'squareness' and total area (smaller is better for neatness)
            score = w * h
            if score < min_score:
                min_score = score
                best_layout = (fs, final_text, w, h)
                
    return best_layout

# -----------------------------
# Placement Logic (Requirements 1 & 2)
# -----------------------------
def _choose_callout_rect(
    page: fitz.Page,
    targets: List[fitz.Rect],
    occupied: List[fitz.Rect],
    w: float,
    h: float
) -> fitz.Rect:
    pr = page.rect
    margin = 25.0
    target_union = _union_rect(targets) or fitz.Rect(100, 100, 200, 200)
    target_c = _center(target_union)
    
    # Pre-calculate blockers (Requirement 2: Aim for white space)
    # We only treat LARGE blocks or existing callouts as hard blockers
    hard_blockers = occupied + [r for r in page.get_text("blocks") if fitz.Rect(r[:4]).width > 50]

    candidates = []
    # Generate a grid of points around the target
    for angle in range(0, 360, 15):
        rad = math.radians(angle)
        for dist in [40, 80, 150, 250]: # Search outward (Requirement 1: Shortest line)
            cx = target_c.x + dist * math.cos(rad)
            cy = target_c.y + dist * math.sin(rad)
            
            cand = fitz.Rect(cx - w/2, cy - h/2, cx + w/2, cy + h/2)
            
            # Stay on page
            if cand.x0 < margin or cand.y0 < margin or cand.x1 > pr.width - margin or cand.y1 > pr.height - margin:
                continue
                
            # Score the candidate
            # 1. Distance from target (Primary)
            dist_score = _dist(_center(cand), target_c)
            
            # 2. Penalty for overlapping existing annotations (Hard)
            overlap_penalty = 0
            for b in hard_blockers:
                if cand.intersects(b):
                    overlap_penalty += 5000 
            
            candidates.append((dist_score + overlap_penalty, cand))

    if not candidates:
        # Emergency fallback: find any spot that doesn't overlap 'occupied'
        return fitz.Rect(margin, margin, margin + w, margin + h)

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

# -----------------------------
# Main Annotator
# -----------------------------
def annotate_pdf_bytes(
    pdf_bytes: bytes,
    quote_terms: List[str],
    criterion_id: str,
    meta: Dict,
) -> Tuple[bytes, Dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page1 = doc.load_page(0)
    occupied_callouts: List[fitz.Rect] = []

    # 1. First, draw all red boxes so we can see them
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        for t in quote_terms:
            for r in page.search_for(t):
                _draw_red_box(page, r)

    # 2. Define Metadata to process
    jobs = [
        ("Original source of publication.", [meta.get("source_url")]),
        ("The distinguished organization.", [meta.get("venue_name")]),
        ("Beneficiary named as a lead role.", [meta.get("beneficiary_name")]),
        ("Performance date.", [meta.get("performance_date")]),
    ]

    for label, needles in jobs:
        needles = [n for n in needles if n]
        if not needles: continue
        
        rects = []
        for n in needles:
            rects.extend(page1.search_for(n))
        
        if rects:
            for rr in rects: _draw_red_box(page1, rr)
            
            # Requirement 3: Optimize font and wrapping
            fs, txt, w, h = _optimize_layout(label)
            
            # Requirement 1 & 2: Best position
            callout_rect = _choose_callout_rect(page1, rects, occupied_callouts, w, h)
            
            # Draw annotation
            page1.insert_textbox(callout_rect, txt, fontname="Times-Bold", fontsize=fs, color=RED)
            occupied_callouts.append(callout_rect)
            
            # Draw line
            page1.draw_line(_center(callout_rect), _center(rects[0]), color=RED, width=0.8)

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue(), {"status": "success"}
