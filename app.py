import io
import math
import json
import os
from typing import Dict, List, Tuple, Optional
import fitz  # PyMuPDF
from openai import OpenAI

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# ============================================================
# GEOMETRY & MARGIN-FIRST PLACEMENT ENGINE
# ============================================================

def _center(rect: fitz.Rect) -> fitz.Point:
    return fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)

def _union_rect(rects: List[fitz.Rect]) -> fitz.Rect:
    r = fitz.Rect(rects[0])
    for x in rects[1:]: r |= x
    return r

def _get_closest_point(rect: fitz.Rect, target: fitz.Point) -> fitz.Point:
    """Returns the point on the rectangle boundary closest to the target."""
    x = max(rect.x0, min(target.x, rect.x1))
    y = max(rect.y0, min(target.y, rect.y1))
    return fitz.Point(x, y)

def _optimize_layout_for_margin(text: str, target_margin_width: float, fontname: str = "helv") -> Tuple[int, str, float, float]:
    """Wraps text to fit narrow side margins using 10-12pt font."""
    best_layout = (10, text, target_margin_width, 40.0)
    min_height = float('inf')

    for fs in [12, 11, 10]:
        words = text.split()
        lines, cur = [], []
        for w in words:
            trial = " ".join(cur + [w])
            if fitz.get_text_length(trial, fontname=fontname, fontsize=fs) <= target_margin_width:
                cur.append(w)
            else:
                lines.append(" ".join(cur))
                cur = [w]
        lines.append(" ".join(cur))
        
        h = (len(lines) * fs * 1.2) + 10
        w = max([fitz.get_text_length(l, fontname=fontname, fontsize=fs) for l in lines]) + 10
        
        if h < min_height:
            min_height = h
            best_layout = (fs, "\n".join(lines), w, h)
    return best_layout

def _choose_best_margin_spot(page: fitz.Page, targets: List[fitz.Rect], occupied: List[fitz.Rect], label: str) -> Tuple[fitz.Rect, str, int]:
    """Prioritizes left/right margins to avoid central document text."""
    pr = page.rect
    target_union = _union_rect(targets)
    target_y = (target_union.y0 + target_union.y1) / 2
    
    margin_w = 85.0 # Narrow width for side placement
    left_x = 25.0
    right_x = pr.width - 25.0 - margin_w

    text_blocks = [fitz.Rect(b[:4]) for b in page.get_text("blocks")]
    candidates = []

    for x_start in [left_x, right_x]:
        fs, txt, w, h = _optimize_layout_for_margin(label, margin_w)
        cand = fitz.Rect(x_start, target_y - h/2, x_start + w, target_y + h/2)
        
        # Keep on page Y-bounds
        if cand.y0 < 25: cand.y1 += (25 - cand.y0); cand.y0 = 25
        if cand.y1 > pr.height - 25: cand.y0 -= (cand.y1 - (pr.height - 25)); cand.y1 = pr.height - 25

        # Score by vertical distance + heavy collision penalties
        score = abs(target_y - _center(cand).y)
        for occ in occupied:
            if cand.intersects(occ): score += 50000 
        for block in text_blocks:
            if cand.intersects(block): score += 2000 # Strong preference for margin gaps
            
        candidates.append((score, cand, txt, fs))

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][2], candidates[0][3]

# ============================================================
# MAIN ANNOTATOR FUNCTION
# ============================================================

def annotate_pdf_bytes(
    pdf_bytes: bytes,
    quote_terms: List[str],
    criterion_id: str, # Fixes TypeError: parameter restored
    meta: Dict,
) -> Tuple[bytes, Dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(doc) == 0:
        return pdf_bytes, {}

    page1 = doc.load_page(0)
    occupied_callouts: List[fitz.Rect] = []

    # 1. Red box specific quotes throughout document
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        for term in (quote_terms or []):
            if not term.strip(): continue
            for r in page.search_for(term):
                page.draw_rect(r, color=RED, width=1.5)

    # 2. Metadata Labels (Prioritizing Margins)
    meta_jobs = [
        ("Original source of publication.", meta.get("source_url")),
        ("The distinguished organization.", meta.get("venue_name") or meta.get("org_name")),
        ("Performance date.", meta.get("performance_date")),
        ("Beneficiary lead role evidence.", meta.get("beneficiary_name")),
    ]

    for label, val in meta_jobs:
        if not val or not str(val).strip(): continue
        
        targets = page1.search_for(str(val))
        if not targets: continue
        
        # Highlight target
        for t in targets:
            page1.draw_rect(t, color=RED, width=1.5)
            
        # Get placement in the margin
        callout_rect, txt, fs = _choose_best_margin_spot(page1, targets, occupied_callouts, label)
        
        # Legibility Shield
        page1.draw_rect(callout_rect, color=WHITE, fill=WHITE, overlay=True)
        page1.insert_textbox(callout_rect, txt, fontname="helv", fontsize=fs, color=RED)
        
        # Draw connector from edge of box to target
        start_p = _get_closest_point(callout_rect, _center(targets[0]))
        page1.draw_line(start_p, _center(targets[0]), color=RED, width=1.0)
        
        occupied_callouts.append(callout_rect)

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue(), {"status": "success", "criterion": criterion_id}
