import io
import math
import fitz  # PyMuPDF
from typing import Dict, List, Tuple, Optional

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# -----------------------------
# Layout & Margin Helpers
# -----------------------------

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
    """Prioritizes left/right margins to keep center text clear."""
    pr = page.rect
    target_union = _union_rect(targets)
    target_y = (target_union.y0 + target_union.y1) / 2
    
    margin_w = 85.0 
    left_x = 25.0
    right_x = pr.width - 25.0 - margin_w

    text_blocks = [fitz.Rect(b[:4]) for b in page.get_text("blocks")]
    candidates = []

    for x_start in [left_x, right_x]:
        fs, txt, w, h = _optimize_layout_for_margin(label, margin_w)
        cand = fitz.Rect(x_start, target_y - h/2, x_start + w, target_y + h/2)
        
        if cand.y0 < 25: cand.y1 += (25 - cand.y0); cand.y0 = 25
        if cand.y1 > pr.height - 25: cand.y0 -= (cand.y1 - (pr.height - 25)); cand.y1 = pr.height - 25

        score = abs(target_y - (cand.y0 + cand.y1)/2)
        for occ in occupied:
            if cand.intersects(occ): score += 50000 
        for block in text_blocks:
            if cand.intersects(block): score += 2000 
            
        candidates.append((score, cand, txt, fs))

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][2], candidates[0][3]

# -----------------------------
# Main Annotation Logic
# -----------------------------

def annotate_pdf_bytes(
    pdf_bytes: bytes,
    quote_terms: List[str],
    criterion_id: str, # RESTORED: This stops the TypeError
    meta: Dict,
) -> Tuple[bytes, Dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(doc) == 0:
        return pdf_bytes, {}

    page1 = doc.load_page(0)
    occupied_callouts: List[fitz.Rect] = []
    total_quote_hits = 0
    total_meta_hits = 0

    # A) Global Quote Highlights
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        for term in (quote_terms or []):
            if not term.strip(): continue
            found = page.search_for(term)
            for r in found:
                page.draw_rect(r, color=RED, width=1.5)
                total_quote_hits += 1

    # B) Metadata Labels (Margin Priority)
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
        
        total_meta_hits += 1
        for t in targets:
            page1.draw_rect(t, color=RED, width=1.5)
            
        callout_rect, txt, fs = _choose_best_margin_spot(page1, targets, occupied_callouts, label)
        
        page1.draw_rect(callout_rect, color=WHITE, fill=WHITE, overlay=True)
        page1.insert_textbox(callout_rect, txt, fontname="helv", fontsize=fs, color=RED)
        
        start_p = _get_closest_point(callout_rect, _center(targets[0]))
        page1.draw_line(start_p, _center(targets[0]), color=RED, width=1.0)
        
        occupied_callouts.append(callout_rect)

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    
    return out.getvalue(), {"total_quote_hits": total_quote_hits, "total_meta_hits": total_meta_hits}

# Internal Geometry Helpers
def _union_rect(rects: List[fitz.Rect]) -> fitz.Rect:
    r = fitz.Rect(rects[0])
    for x in rects[1:]: r |= x
    return r

def _center(rect: fitz.Rect) -> fitz.Point:
    return fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)

def _get_closest_point(rect: fitz.Rect, target: fitz.Point) -> fitz.Point:
    x = max(rect.x0, min(target.x, rect.x1))
    y = max(rect.y0, min(target.y, rect.y1))
    return fitz.Point(x, y)
