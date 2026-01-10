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

EDGE_PAD = 15.0 # Slightly tighter to allow more room

GAP_FROM_TEXT_BLOCKS = 14.0
GAP_FROM_IMAGES = 12.0
GAP_FROM_DRAWINGS = 10.0

LINE_BLOCKER_PAD_X = 3.0
LINE_BLOCKER_PAD_Y = 6.0 

GAP_FROM_HIGHLIGHTS = 16.0
GAP_BETWEEN_CALLOUTS = 10.0
ENDPOINT_PULLBACK = 2.0

_MAX_TERM = 600
_CHUNK = 70
_CHUNK_OVERLAP = 22

# ============================================================
# Ink-check knobs
# ============================================================
INKCHECK_DPI = 90
INKCHECK_PAD = 1.0

INKCHECK_LEVELS = [
    (250, 0.002),
    (245, 0.006),
    (235, 0.010),
]

# Gutters are more permissive to ignore page artifacts/noise
INKCHECK_LEVELS_GUTTER = [
    (250, 0.005),
    (245, 0.012),
    (230, 0.025),
]

# Increased gutter search widths
GUTTER_WIDTHS = [85.0, 110.0, 150.0] 

# ============================================================
# Geometry helpers
# ============================================================

def inflate_rect(r: fitz.Rect, pad: float) -> fitz.Rect:
    rr = fitz.Rect(r)
    rr.x0 -= pad; rr.y0 -= pad; rr.x1 += pad; rr.y1 += pad
    return rr

def inflate_rect_xy(r: fitz.Rect, padx: float, pady: float) -> fitz.Rect:
    rr = fitz.Rect(r)
    rr.x0 -= padx; rr.x1 += padx; rr.y0 -= pady; rr.y1 += pady
    return rr

def _union_rect(rects: List[fitz.Rect]) -> fitz.Rect:
    if not rects: return fitz.Rect(0, 0, 0, 0)
    r = fitz.Rect(rects[0])
    for x in rects[1:]: r |= x
    return r

def _center(rect: fitz.Rect) -> fitz.Point:
    return fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)

def _segment_hits_rect(p1: fitz.Point, p2: fitz.Point, r: fitz.Rect) -> bool:
    steps = 20
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
    cx, cy = (rect.x0 + rect.x1)/2, (rect.y0 + rect.y1)/2
    return [
        fitz.Point(rect.x0, cy), fitz.Point(rect.x1, cy),
        fitz.Point(cx, rect.y0), fitz.Point(cx, rect.y1),
        fitz.Point(rect.x0, rect.y0), fitz.Point(rect.x1, rect.y0),
        fitz.Point(rect.x0, rect.y1), fitz.Point(rect.x1, rect.y1),
    ]

def _mid_height_anchor(callout: fitz.Rect, toward: fitz.Point) -> fitz.Point:
    y = callout.y0 + (callout.height / 2.0)
    cx = callout.x0 + callout.width / 2.0
    return fitz.Point(callout.x1 if toward.x >= cx else callout.x0, y)

def _straight_connector_best_pair(
    callout_rect: fitz.Rect, target_rect: fitz.Rect, obstacles: List[fitz.Rect]
) -> Tuple[fitz.Point, fitz.Point, int, float]:
    target_center = _center(target_rect)
    base = _mid_height_anchor(callout_rect, target_center)
    starts = [fitz.Point(base.x, min(max(callout_rect.y0 + 2.0, base.y + dy), callout_rect.y1 - 2.0)) 
              for dy in [-14.0, -7.0, 0.0, 7.0, 14.0]]
    ends = _edge_candidates(target_rect)

    best_hits, best_len = 10**9, 10**9
    best_s, best_e = starts[0], ends[0]

    for s in starts:
        for e in ends:
            hits = sum(1 for ob in obstacles if _segment_hits_rect(s, e, ob))
            length = math.hypot(e.x - s.x, e.y - s.y)
            if hits < best_hits or (hits == best_hits and length < best_len):
                best_hits, best_len, best_s, best_e = hits, length, s, e

    return best_s, _pull_back_point(best_s, best_e, ENDPOINT_PULLBACK), best_hits, best_len

def _draw_straight_connector(page: fitz.Page, callout_rect: fitz.Rect, target_rect: fitz.Rect, obstacles: List[fitz.Rect]):
    s, e, _, _ = _straight_connector_best_pair(callout_rect, target_rect, obstacles)
    page.draw_line(s, e, color=RED, width=LINE_WIDTH)

def _rect_has_ink(page: fitz.Page, rect: fitz.Rect, *, dpi: int, white_threshold: int, nonwhite_ratio: float) -> bool:
    r = fitz.Rect(rect) & page.rect
    if r.is_empty or r.width < 2 or r.height < 2: return False
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), clip=r, alpha=False)
    s = pix.samples
    if not s: return False
    nonwhite = sum(1 for i in range(0, len(s), 3) if s[i] < white_threshold or s[i+1] < white_threshold or s[i+2] < white_threshold)
    return (nonwhite / (pix.width * pix.height)) > nonwhite_ratio

# ============================================================
# Blockers
# ============================================================

def _page_text_shapes(page: fitz.Page) -> List[fitz.Rect]:
    rects = []
    try:
        # Blocks
        for b in page.get_text("blocks"): rects.append(fitz.Rect(b[:4]))
        # Lines with padding
        d = page.get_text("dict")
        for b in d.get("blocks", []):
            for ln in b.get("lines", []):
                rects.append(inflate_rect_xy(fitz.Rect(ln.get("bbox")), LINE_BLOCKER_PAD_X, LINE_BLOCKER_PAD_Y))
    except: pass
    return rects

def _page_blockers(page: fitz.Page) -> List[fitz.Rect]:
    blockers = [inflate_rect(r, GAP_FROM_TEXT_BLOCKS) for r in _page_text_shapes(page)]
    try:
        for img in page.get_images(full=True):
            for r in page.get_image_rects(img[0]): blockers.append(inflate_rect(r, GAP_FROM_IMAGES))
        for d in page.get_drawings():
            rr = d.get("rect")
            if rr: blockers.append(inflate_rect(rr, GAP_FROM_DRAWINGS))
    except: pass
    return blockers

def _text_envelope(page: fitz.Page) -> Optional[fitz.Rect]:
    blocks = [fitz.Rect(b[:4]) for b in page.get_text("blocks")]
    return _union_rect(blocks) if blocks else None

def _intersects_any(r: fitz.Rect, others: List[fitz.Rect]) -> bool:
    return any(r.intersects(o) for o in others)

# ============================================================
# Layout and Zones
# ============================================================

def _optimize_layout(text: str, box_width: float) -> Tuple[int, str, float, float]:
    text = (text or "").strip()
    if not text: return 12, "", box_width, 24.0
    for fs in FONT_SIZES:
        words = text.split(); lines = []; cur = []; usable_w = max(10.0, box_width - 10.0)
        for w in words:
            if fitz.get_text_length(" ".join(cur + [w]), fontname=FONTNAME, fontsize=fs) <= usable_w:
                cur.append(w)
            else:
                lines.append(" ".join(cur)) if cur else lines.append(w)
                cur = [w]
        if cur: lines.append(" ".join(cur))
        h = (len(lines) * fs * 1.22) + 10.0
        if h <= 95.0 or fs == 10: return fs, "\n".join(lines), box_width, h
    return 10, text, box_width, 44.0

def _zones(page: fitz.Page) -> List[Tuple[str, fitz.Rect]]:
    pr = page.rect
    env = _text_envelope(page)
    zones = []
    for gw in GUTTER_WIDTHS:
        left = fitz.Rect(EDGE_PAD, EDGE_PAD, min(pr.width - EDGE_PAD, EDGE_PAD + gw), pr.height - EDGE_PAD)
        right = fitz.Rect(max(EDGE_PAD, pr.width - EDGE_PAD - gw), EDGE_PAD, pr.width - EDGE_PAD, pr.height - EDGE_PAD)
        if left.width > 45: zones.append((f"left_{int(gw)}", left))
        if right.width > 45: zones.append((f"right_{int(gw)}", right))
    
    if env:
        env2 = inflate_rect(env, 8.0)
        top = fitz.Rect(EDGE_PAD, EDGE_PAD, pr.width - EDGE_PAD, max(EDGE_PAD, env2.y0))
        bottom = fitz.Rect(EDGE_PAD, min(pr.height - EDGE_PAD, env2.y1), pr.width - EDGE_PAD, pr.height - EDGE_PAD)
        if top.height > 30: zones.append(("top", top))
        if bottom.height > 30: zones.append(("bottom", bottom))
    return zones if zones else [("full", fitz.Rect(EDGE_PAD, EDGE_PAD, pr.width-EDGE_PAD, pr.height-EDGE_PAD))]

def _grid_positions(z: fitz.Rect, w: float, h: float, step: float) -> List[fitz.Point]:
    pts = []; x, y = z.x0 + w/2, z.y0 + h/2
    while y <= (z.y1 - h/2):
        curr_x = x
        while curr_x <= (z.x1 - w/2):
            pts.append(fitz.Point(curr_x, y))
            curr_x += step
        y += step
    return pts

def _find_candidates_in_zone(page, zone_name, z, target_union, blockers, highlight_blockers, occupied_buf, connector_obstacles, label):
    bw = min(200.0, max(85.0, z.width - 10.0))
    fs, wrapped, w, h = _optimize_layout(label, bw)
    pts = _grid_positions(z, w, h, step=15.0)
    tc = _center(target_union)
    
    is_gutter = "left" in zone_name or "right" in zone_name
    # POINT SORTING: If gutter, favor the outer edges of the page first
    if "left" in zone_name: pts.sort(key=lambda p: p.x)
    elif "right" in zone_name: pts.sort(key=lambda p: -p.x)
    else: pts.sort(key=lambda p: abs(p.y - tc.y))

    levels = INKCHECK_LEVELS_GUTTER if is_gutter else INKCHECK_LEVELS
    out = []

    for thr, ratio in levels:
        for c in pts[:400]:
            cand = fitz.Rect(c.x - w/2, c.y - h/2, c.x + w/2, c.y + h/2)
            if _intersects_any(cand, blockers) or _intersects_any(cand, occupied_buf) or _intersects_any(cand, highlight_blockers):
                continue
            if _rect_has_ink(page, inflate_rect(cand, INKCHECK_PAD), dpi=INKCHECK_DPI, white_threshold=thr, nonwhite_ratio=ratio):
                continue

            s, e, hits, length = _straight_connector_best_pair(cand, target_union, connector_obstacles)
            
            # SCORING: 
            # 1. Huge bonus for being a gutter candidate.
            # 2. Lower penalty for 'hits' (1500 instead of 7000) so it doesn't disqualify side-placements.
            score = (hits * 1500.0) + length + (abs(c.y - tc.y) * 0.5)
            if is_gutter: score -= 50000.0 

            out.append((score, cand, wrapped, fs, True))
        if out: break
    return sorted(out, key=lambda x: x[0])

def _choose_best_spot(page, targets, occupied_callouts, label):
    target_union = _union_rect(targets)
    blockers = _page_blockers(page)
    highlight_blockers = [inflate_rect(t, GAP_FROM_HIGHLIGHTS) for t in targets]
    occupied_buf = [inflate_rect(o, GAP_BETWEEN_CALLOUTS) for o in occupied_callouts]
    
    conn_obs = [inflate_rect(r, 1.2) for r in _page_text_shapes(page)] + occupied_buf

    zones = _zones(page)
    side_zones = [z for z in zones if "left" in z[0] or "right" in z[0]]
    other_zones = [z for z in zones if z not in side_zones]

    # STAGE 1: Check Sides
    side_cands = []
    for zn, zr in side_zones:
        side_cands.extend(_find_candidates_in_zone(page, zn, zr, target_union, blockers, highlight_blockers, occupied_buf, conn_obs, label))
    
    if side_cands:
        best = sorted(side_cands, key=lambda x: x[0])[0]
        return best[1], best[2], best[3], best[4]

    # STAGE 2: Check Top/Bottom
    other_cands = []
    for zn, zr in other_zones:
        other_cands.extend(_find_candidates_in_zone(page, zn, zr, target_union, blockers, highlight_blockers, occupied_buf, conn_obs, label))
    
    if other_cands:
        best = sorted(other_cands, key=lambda x: x[0])[0]
        return best[1], best[2], best[3], best[4]

    # Fallback
    fs, wrapped, w, h = _optimize_layout(label, 170.0)
    fb = fitz.Rect(EDGE_PAD, EDGE_PAD, EDGE_PAD+w, EDGE_PAD+h)
    return fb, wrapped, fs, True

# ============================================================
# Search + Main
# ============================================================

def _search_term(page: fitz.Page, term: str) -> List[fitz.Rect]:
    t = (term or "").strip()
    if not t: return []
    res = page.search_for(t)
    if not res:
        t2 = re.sub(r"\s+", " ", t)
        res = page.search_for(t2)
    return res

def _url_variants(url: str) -> List[str]:
    u = (url or "").strip()
    if not u: return []
    v = [u, u.replace("https://","").replace("http://","").replace("www.","")]
    return list(set([x for x in v if x]))

def annotate_pdf_bytes(pdf_bytes: bytes, quote_terms: List[str], criterion_id: str, meta: Dict) -> Tuple[bytes, Dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if not doc: return pdf_bytes, {}
    
    page1 = doc[0]
    total_q, total_m = 0, 0
    occ = []

    for page in doc:
        for t in (quote_terms or []):
            rs = _search_term(page, t)
            for r in rs:
                page.draw_rect(r, color=RED, width=BOX_WIDTH)
                total_q += 1

    def _do_job(label, needles, policy="union"):
        nonlocal total_m, occ
        tgs = []
        for n in needles: tgs.extend(_search_term(page1, n))
        if not tgs: return
        for t in tgs: page1.draw_rect(t, color=RED, width=BOX_WIDTH)
        total_m += len(tgs)

        rect, text, fs, safe = _choose_best_spot(page1, tgs, occ, label)
        if safe: page1.draw_rect(rect, color=WHITE, fill=WHITE, overlay=True)
        page1.insert_textbox(rect, text, fontname=FONTNAME, fontsize=fs, color=RED)
        
        obs = [inflate_rect(r, 1.5) for r in _page_text_shapes(page1)] + occ
        if policy == "all":
            for t in tgs: _draw_straight_connector(page1, rect, t, obs)
        else:
            _draw_straight_connector(page1, rect, _union_rect(tgs), obs)
        occ.append(rect)

    _do_job("Original source of publication.", _url_variants(meta.get("source_url")))
    
    venue = (meta.get("venue_name") or "").strip()
    org = (meta.get("org_name") or "").strip()
    if venue or org: _do_job("The distinguished organization.", [venue, org])

    perf = (meta.get("performance_date") or "").strip()
    if perf: _do_job("Performance date.", [perf])

    sal = (meta.get("salary_amount") or "").strip()
    if sal: _do_job("Beneficiary salary evidence.", [sal])

    b_names = [meta.get("beneficiary_name")] + (meta.get("beneficiary_variants") or [])
    b_names = [n for n in b_names if n]
    if b_names: _do_job("Beneficiary lead role evidence.", list(set(b_names)), "all")

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue(), {"total_quote_hits": total_q, "total_meta_hits": total_m, "criterion_id": criterion_id}
