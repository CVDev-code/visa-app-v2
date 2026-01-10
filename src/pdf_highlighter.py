import io
import math
import json
import os
import re
import csv
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF
from openai import OpenAI

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# ============================================================
# PART 1: SECRETS & METADATA HELPERS
# ============================================================

def _get_secret(name: str):
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)

def merge_metadata(
    filename: str,
    auto: Optional[Dict] = None,
    global_defaults: Optional[Dict] = None,
    csv_data: Optional[Dict[str, Dict]] = None,
    overrides: Optional[Dict] = None,
) -> Dict:
    auto = auto or {}
    global_defaults = global_defaults or {}
    overrides = overrides or {}
    row = (csv_data or {}).get(filename, {}) if csv_data else {}

    def pick(key: str):
        return (
            overrides.get(key)
            or row.get(key)
            or global_defaults.get(key)
            or auto.get(key)
            or None
        )

    return {
        "source_url": pick("source_url"),
        "venue_name": pick("venue_name"),
        "performance_date": pick("performance_date"),
        "org_name": pick("org_name"),
        "salary_amount": pick("salary_amount"),
        "beneficiary_name": pick("beneficiary_name"),
    }

# ============================================================
# PART 2: AI AUTO-DETECTION (Fixed OpenAI v1.0 Syntax)
# ============================================================

_AUTODETECT_SYSTEM = (
    "You extract structured metadata from USCIS O-1 evidence PDFs. "
    "Return ONLY valid JSON. If a field is not found, return an empty string."
)

_AUTODETECT_USER = """Extract metadata from the following document text.
Return JSON with keys: source_url, venue_name, performance_date, org_name, salary_amount.
DOCUMENT TEXT: {text}"""

def autodetect_metadata(document_text: str) -> Dict:
    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    model = _get_secret("OPENAI_MODEL") or "gpt-4o-mini"
    client = OpenAI(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _AUTODETECT_SYSTEM},
                {"role": "user", "content": _AUTODETECT_USER.format(text=(document_text or "")[:20000])},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw) if raw else {}
    except Exception:
        data = {}

    return {k: str(data.get(k, "")).strip() for k in ["source_url", "venue_name", "performance_date", "org_name", "salary_amount"]}

# ============================================================
# PART 3: GEOMETRY & PLACEMENT ENGINE
# ============================================================

def _center(rect: fitz.Rect) -> fitz.Point:
    return fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)

def _dist(a: fitz.Point, b: fitz.Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

def _union_rect(rects: List[fitz.Rect]) -> Optional[fitz.Rect]:
    if not rects: return None
    r = fitz.Rect(rects[0])
    for x in rects[1:]: r |= x
    return r

def _optimize_layout(text: str, fontname: str = "helv") -> Tuple[int, str, float, float]:
    """Requirement 3: Optimizes font size and wraps text for a neat block."""
    best_layout = (10, text, 180.0, 40.0)
    min_area = float('inf')

    for fs in [12, 11, 10]:
        for target_w in [140, 180, 220]:
            words = text.split()
            lines, cur = [], []
            for w in words:
                trial = " ".join(cur + [w])
                if fitz.get_text_length(trial, fontname=fontname, fontsize=fs) <= target_w:
                    cur.append(w)
                else:
                    lines.append(" ".join(cur))
                    cur = [w]
            lines.append(" ".join(cur))
            
            h = (len(lines) * fs * 1.2) + 10
            w = max([fitz.get_text_length(l, fontname=fontname, fontsize=fs) for l in lines]) + 10
            
            if w * h < min_area:
                min_area = w * h
                best_layout = (fs, "\n".join(lines), w, h)
    return best_layout

def _choose_callout_rect(page: fitz.Page, targets: List[fitz.Rect], occupied: List[fitz.Rect], w: float, h: float) -> fitz.Rect:
    """Requirement 1 & 2: Radial search for white space near the target."""
    pr = page.rect
    target_union = _union_rect(targets) or fitz.Rect(72, 72, 150, 150)
    target_c = _center(target_union)
    text_blocks = [fitz.Rect(b[:4]) for b in page.get_text("blocks")]
    
    candidates = []
    # Search in expanding circles (Requirement 1: keep lines short)
    for dist in range(40, 300, 20): 
        for angle in range(0, 360, 20):
            rad = math.radians(angle)
            cx = target_c.x + dist * math.cos(rad)
            cy = target_c.y + dist * math.sin(rad)
            cand = fitz.Rect(cx - w/2, cy - h/2, cx + w/2, cy + h/2)
            
            if cand.x0 < 25 or cand.y0 < 25 or cand.x1 > pr.width - 25 or cand.y1 > pr.height - 25:
                continue
                
            score = dist 
            for occ in occupied:
                if cand.intersects(occ): score += 10000 # Hard blocker: other callouts
            for block in text_blocks:
                if cand.intersects(block): score += 300 # Soft blocker: document text
            
            candidates.append((score, cand))

    if not candidates:
        offset = len(occupied) * 60
        return fitz.Rect(30, 30 + offset, 30 + w, 30 + h + offset)

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

# ============================================================
# PART 4: MAIN PDF ANNOTATOR
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

    # A) Global highlights for quote terms (all pages)
    for pno in range(len(doc)):
        page = doc.load_page(pno)
        for term in quote_terms:
            if not term.strip(): continue
            for r in page.search_for(term):
                page.draw_rect(r, color=RED, width=1.5)

    # B) Metadata Callouts (Page 1 only)
    meta_jobs = [
        ("Original source of publication.", meta.get("source_url")),
        ("The distinguished organization.", meta.get("org_name") or meta.get("venue_name")),
        ("Performance date.", meta.get("performance_date")),
        ("Beneficiary lead role evidence.", meta.get("beneficiary_name")),
    ]

    for label, val in meta_jobs:
        if not val or not str(val).strip(): continue
        
        targets = page1.search_for(str(val))
        if not targets: continue
        
        # Highlight matches
        for t in targets:
            page1.draw_rect(t, color=RED, width=1.5)
            
        # Optimize Layout & Position
        fs, wrapped_txt, w, h = _optimize_layout(label)
        callout_rect = _choose_callout_rect(page1, targets, occupied_callouts, w, h)
        
        # Draw opaque background for legibility
        page1.draw_rect(callout_rect, color=WHITE, fill=WHITE, overlay=True)
        
        # Draw annotation & Connector
        page1.insert_textbox(callout_rect, wrapped_txt, fontname="helv", fontsize=fs, color=RED)
        page1.draw_line(_center(callout_rect), _center(targets[0]), color=RED, width=1.0)
        
        occupied_callouts.append(callout_rect)

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue(), {"status": "success", "criterion": criterion_id}
