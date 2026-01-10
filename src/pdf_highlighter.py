import io
import math
import json
import os
import fitz  # PyMuPDF
from typing import Dict, List, Tuple, Optional
from openai import OpenAI

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# -----------------------------
# Layout & Margin Helpers
# -----------------------------

def _optimize_layout_for_margin(text: str, target_margin_width: float, fontname: str = "helv") -> Tuple[int, str, float, float]:
    """
    Requirement 3: Adapts line breaks to fit narrow margins.
    Tries to find the best font size (10-12) for the given target width.
    """
    best_layout = (10, text, target_margin_width, 40.0)
    min_height = float('inf')

    for fs in [12, 11, 10]:
        words = text.split()
        lines, cur = [], []
        for w in words:
            trial = " ".join(cur + [w])
            # Use target_margin_width to force a narrow, vertical layout if needed
            if fitz.get_text_length(trial, fontname=fontname, fontsize=fs) <= target_margin_width:
                cur.append(w)
            else:
                lines.append(" ".join(cur))
                cur = [w]
        lines.append(" ".join(cur))
        
        h = (len(lines) * fs * 1.2) + 10
        w = max([fitz.get_text_length(l, fontname=fontname, fontsize=fs) for l in lines]) + 10
        
        # We want the shortest height that fits in this width
        if h < min_height:
            min_height = h
            best_layout = (fs, "\n".join(lines), w, h)
            
    return best_layout

def _choose_best_margin_spot(page: fitz.Page, targets: List[fitz.Rect], occupied: List[fitz.Rect], label: str) -> Tuple[fitz.Rect, str, int]:
    """
    Prioritizes Left/Right margins near the target's Y-level.
    """
    pr = page.rect
    target_union = _union_rect(targets)
    target_y = (target_union.y0 + target_union.y1) / 2
    
    # Margin definitions (typical 72pt margins, but we check availability)
    margin_w = 70.0
    left_margin_x = 35.0
    right_margin_x = pr.width - 35.0 - margin_w

    # Options: [Left Margin, Right Margin]
    candidate_spots = [
        (left_margin_x, target_y),
        (right_margin_x, target_y)
    ]
    
    text_blocks = [fitz.Rect(b[:4]) for b in page.get_text("blocks")]
    final_candidates = []

    for x, y in candidate_spots:
        # Get optimal layout for a margin-width
        fs, txt, w, h = _optimize_layout_for_margin(label, margin_w)
        cand = fitz.Rect(x, y - h/2, x + w, y + h/2)
        
        # Ensure it's on page
        if cand.y0 < 20: cand.y1 += (20 - cand.y0); cand.y0 = 20
        if cand.y1 > pr.height - 20: cand.y0 -= (cand.y1 - (pr.height - 20)); cand.y1 = pr.height - 20

        # Scoring
        score = 0
        for occ in occupied:
            if cand.intersects(occ): score += 10000
        for block in text_blocks:
            if cand.intersects(block): score += 500 # Soft penalty for overlapping text

        final_candidates.append((score, cand, txt, fs))

    # Pick the lowest score (cleanest margin)
    final_candidates.sort(key=lambda x: x[0])
    best = final_candidates[0]
    return best[1], best[2], best[3]

# -----------------------------
# Main Annotation Logic
# -----------------------------

def annotate_pdf_bytes(pdf_bytes: bytes, quote_terms: List[str], meta: Dict) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page1 = doc.load_page(0)
    occupied_callouts: List[fitz.Rect] = []

    meta_jobs = [
        ("Original source of publication.", meta.get("source_url")),
        ("The distinguished organization.", meta.get("org_name") or meta.get("venue_name")),
        ("Beneficiary lead role evidence.", meta.get("beneficiary_name")),
    ]

    for label, val in meta_jobs:
        if not val: continue
        targets = page1.search_for(str(val))
        if not targets: continue
        
        # 1. Draw Red Box
        for t in targets:
            page1.draw_rect(t, color=RED, width=1.5)
            
        # 2. Find Margin Placement
        callout_rect, txt, fs = _choose_best_margin_spot(page1, targets, occupied_callouts, label)
        
        # 3. Draw "Shield" (White Background) and Text
        # Requirement 2: Aim for white sections but use shield if overlapping
        page1.draw_rect(callout_rect, color=WHITE, fill=WHITE, overlay=True)
        page1.insert_textbox(callout_rect, txt, fontname="helv", fontsize=fs, color=RED)
        
        # 4. Draw Shortest Connector (Requirement 1)
        # Connector points to the nearest edge of the callout box
        start_p = _get_closest_point(callout_rect, _center(targets[0]))
        page1.draw_line(start_p, _center(targets[0]), color=RED, width=1.0)
        
        occupied_callouts.append(callout_rect)

    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()

def _union_rect(rects: List[fitz.Rect]) -> fitz.Rect:
    r = fitz.Rect(rects[0])
    for x in rects[1:]: r |= x
    return r

def _center(rect: fitz.Rect) -> fitz.Point:
    return fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)

def _get_closest_point(rect: fitz.Rect, target: fitz.Point) -> fitz.Point:
    """Returns the point on the rectangle boundary closest to the target."""
    x = max(rect.x0, min(target.x, rect.x1))
    y = max(rect.y0, min(target.y, rect.y1))
    return fitz.Point(x, y)
