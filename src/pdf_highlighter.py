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

# EDIT B: reduce this because we already have line-level vertical padding
GAP_FROM_TEXT_BLOCKS = 6.0
GAP_FROM_IMAGES = 12.0
GAP_FROM_DRAWINGS = 10.0

LINE_BLOCKER_PAD_X = 3.0
LINE_BLOCKER_PAD_Y = 8.0

GAP_FROM_HIGHLIGHTS = 16.0
GAP_BETWEEN_CALLOUTS = 12.0

ENDPOINT_PULLBACK = 2.0

_MAX_TERM = 600
_CHUNK = 70
_CHUNK_OVERLAP = 22

# ============================================================
# Gutters / zones
# ============================================================
GUTTER_WIDTHS = [70.0, 100.0, 140.0, 180.0]

# ============================================================
# Adaptive ink detection
# ============================================================
INKCHECK_DPI = 96
INKCHECK_PAD = 1.5
INK_SAMPLE_STRIDE = 6
INK_DELTA = 18
INK_RATIO_MAX = 0.012
INK_RATIO_MAX_GUTTER = 0.020


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

def inflate_rect_xy(r: fitz.Rect, padx: float, pady: float) -> fitz.Rect:
    rr = fitz.Rect(r)
    rr.x0 -= padx
    rr.x1 += padx
    rr.y0 -= pady
    rr.y1 += pady
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

def _segment_hits_rect(p1: fitz.Point, p2: fitz.Point, r: fitz.Rect) -> bool:
    steps = 26
    for i in range(steps + 1):
        t = i / steps
        x = p1.x + (p2.x - p1.x) * t
        y = p1.y + (p2.y - p1.y) * t
        if r.contains(fitz.Point(x, y)):
            return True
    return False

def _pull_back_point(from_pt: fitz.Point, to_pt: fitz.Point, dist: float) -> fitz.Point:
    vx, vy = from_pt.x - to_pt.x, from_pt.y - to_pt.y
    d = math.hypot(vx, vy)
    if d == 0:
        return to_pt
    return fitz.Point(to_pt.x + (vx / d) * dist, to_pt.y + (vy / d) * dist)

def _edge_candidates(rect: fitz.Rect) -> List[fitz.Point]:
    cx, cy = (rect.x0 + rect.width / 2.0), (rect.y0 + rect.height / 2.0)
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
    starts = [
        fitz.Point(
            base.x,
            min(max(callout_rect.y0 + 2.0, base.y + dy), callout_rect.y1 - 2.0),
        )
        for dy in [-14.0, -7.0, 0.0, 7.0, 14.0]
    ]
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

def _draw_straight_connector(
    page: fitz.Page, callout_rect: fitz.Rect, target_rect: fitz.Rect, obstacles: List[fitz.Rect]
):
    s, e, _, _ = _straight_connector_best_pair(callout_rect, target_rect, obstacles)
    page.draw_line(s, e, color=RED, width=LINE_WIDTH)

# ============================================================
# EDIT A: clamp candidate rectangles to zone bounds
# ============================================================

def _clamp_rect_to_zone(r: fitz.Rect, z: fitz.Rect) -> Optional[fitz.Rect]:
    rr = fitz.Rect(r)

    # shift horizontally
    if rr.x0 < z.x0:
        rr.x1 += (z.x0 - rr.x0)
        rr.x0 = z.x0
    if rr.x1 > z.x1:
        rr.x0 -= (rr.x1 - z.x1)
        rr.x1 = z.x1

    # shift vertically
    if rr.y0 < z.y0:
        rr.y1 += (z.y0 - rr.y0)
        rr.y0 = z.y0
    if rr.y1 > z.y1:
        rr.y0 -= (rr.y1 - z.y1)
        rr.y1 = z.y1

    if rr.x0 < z.x0 or rr.x1 > z.x1 or rr.y0 < z.y0 or rr.y1 > z.y1:
        return None
    if rr.width < 10 or rr.height < 10:
        return None
    return rr


# ============================================================
# Adaptive ink check
# ============================================================

def _rect_has_ink_adaptive(
    page: fitz.Page,
    rect: fitz.Rect,
    *,
    dpi=INKCHECK_DPI,
    sample_stride=INK_SAMPLE_STRIDE,
    delta=INK_DELTA,
    ratio_max=INK_RATIO_MAX,
) -> bool:
    r = (fitz.Rect(rect) & page.rect)
    if r.is_empty or r.width < 2 or r.height < 2:
        return True

    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=r, alpha=False)
    s, w, h = pix.samples, pix.width, pix.height
    if not s or w < 2 or h < 2:
        return True

    step = max(1, sample_stride)

    total = 0
    sr = sg = sb = 0
    for y in range(0, h, step):
        row = y * w * 3
        for x in range(0, w, step):
            i = row + x * 3
            sr += s[i]
            sg += s[i + 1]
            sb += s[i + 2]
            total += 1

    if total <= 0:
        return True

    br, bg, bb = sr / total, sg / total, sb / total

    ink = 0
    total2 = 0
    for y in range(0, h, step):
        row = y * w * 3
        for x in range(0, w, step):
            i = row + x * 3
            if (abs(s[i] - br) > delta) or (abs(s[i + 1] - bg) > delta) or (abs(s[i + 2] - bb) > delta):
                ink += 1
            total2 += 1

    return (ink / total2) > ratio_max if total2 > 0 else True


# ============================================================
# Blockers
# ============================================================

def _page_text_shapes(page: fitz.Page) -> List[fitz.Rect]:
    rects = []
    try:
        for b in page.get_text("blocks"):
            rects.append(fitz.Rect(b[:4]))
        for w in page.get_text("words"):
            rects.append(fitz.Rect(w[:4]))

        # line-level blockers to prevent "between lines"
        d = page.get_text("dict")
        for b in d.get("blocks", []):
            for ln in b.get("lines", []):
                bbox = ln.get("bbox")
                if bbox:
                    rects.append(inflate_rect_xy(fitz.Rect(bbox), LINE_BLOCKER_PAD_X, LINE_BLOCKER_PAD_Y))
    except Exception:
        pass
    return rects

def _page_blockers(page: fitz.Page) -> List[fitz.Rect]:
    blockers = [inflate_rect(r, GAP_FROM_TEXT_BLOCKS) for r in _page_text_shapes(page)]
    try:
        for img in page.get_images(full=True):
            for r in page.get_image_rects(img[0]):
                blockers.append(inflate_rect(r, GAP_FROM_IMAGES))
        for d in page.get_drawings():
            rr = d.get("rect")
            if rr:
                blockers.append(inflate_rect(rr, GAP_FROM_DRAWINGS))
    except Exception:
        pass
    return blockers

def _text_envelope(page: fitz.Page) -> Optional[fitz.Rect]:
    blocks = [fitz.Rect(b[:4]) for b in page.get_text("blocks")]
    return _union_rect(blocks) if blocks else None

# ============================================================
# EDIT C: hard-forbid the whole text column envelope
# ============================================================

def _text_envelope_blocker(page: fitz.Page) -> Optional[fitz.Rect]:
    env = _text_envelope(page)
    if not env:
        return None
    # big enough to exclude inter-line gaps and "near column" whitespace
    return inflate_rect(env, 18.0)

def _intersects_any(r: fitz.Rect, others: List[fitz.Rect]) -> bool:
    return any(r.intersects(o) for o in others)


# ============================================================
# Layout + Zones
# ============================================================

def _optimize_layout(text: str, box_width: float) -> Tuple[int, str, float, float]:
    text = (text or "").strip()
    if not text:
        return 12, "", box_width, 24.0
    for fs in FONT_SIZES:
        words = text.split()
        lines, cur = [], []
        usable_w = max(10.0, box_width - 10.0)
        for w in words:
            if fitz.get_text_length(" ".join(cur + [w]), fontname=FONTNAME, fontsize=fs) <= usable_w:
                cur.append(w)
            else:
                lines.append(" ".join(cur)) if cur else lines.append(w)
                cur = [w]
        if cur:
            lines.append(" ".join(cur))
        h = (len(lines) * fs * 1.22) + 10.0
        if h <= 100.0 or fs == 10:
            return fs, "\n".join(lines), box_width, h
    return 10, text, box_width, 44.0

def _zones(page: fitz.Page) -> List[Tuple[str, fitz.Rect]]:
    pr, env = page.rect, _text_envelope(page)
    zones = []
    for gw in GUTTER_WIDTHS:
        left = fitz.Rect(EDGE_PAD, EDGE_PAD, min(pr.width - EDGE_PAD, EDGE_PAD + gw), pr.height - EDGE_PAD)
        right = fitz.Rect(max(EDGE_PAD, pr.width - EDGE_PAD - gw), EDGE_PAD, pr.width - EDGE_PAD, pr.height - EDGE_PAD)
        if left.width > 40:
            zones.append((f"left_{int(gw)}", left))
        if right.width > 40:
            zones.append((f"right_{int(gw)}", right))
    if env:
        env2 = inflate_rect(env, max(6.0, GAP_FROM_TEXT_BLOCKS * 0.4))
        top = fitz.Rect(EDGE_PAD, EDGE_PAD, pr.width - EDGE_PAD, max(EDGE_PAD, env2.y0))
        bottom = fitz.Rect(EDGE_PAD, min(pr.height - EDGE_PAD, env2.y1), pr.width - EDGE_PAD, pr.height - EDGE_PAD)
        if top.height > 30:
            zones.append(("top", top))
        if bottom.height > 30:
            zones.append(("bottom", bottom))
    return zones if zones else [("full", fitz.Rect(EDGE_PAD, EDGE_PAD, pr.width - EDGE_PAD, pr.height - EDGE_PAD))]


# ============================================================
# Placement logic (Side-Bias + Adaptive Ink)
# ============================================================

def _grid_positions(z, w, h, step):
    pts = []
    x_start = z.x0 + w / 2
    y = z.y0 + h / 2
    while y <= z.y1 - h / 2:
        x = x_start
        while x <= z.x1 - w / 2:
            pts.append(fitz.Point(x, y))
            x += step
        y += step
    return pts

def _choose_best_spot(page, targets, occupied_callouts, label):
    target_union = _union_rect(targets)
    tc = _center(target_union)

    blockers = _page_blockers(page)

    # EDIT C: forbid the whole text column
    env_block = _text_envelope_blocker(page)
    if env_block:
        blockers.append(env_block)

    highlight_blockers = [inflate_rect(t, GAP_FROM_HIGHLIGHTS) for t in targets]
    occupied_buf = [inflate_rect(o, GAP_BETWEEN_CALLOUTS) for o in occupied_callouts]

    conn_obs = (
        [inflate_rect(b, 1.5) for b in _page_text_shapes(page)]
        + [inflate_rect(o, 2.0) for o in occupied_callouts]
        + [inflate_rect(hb, 1.0) for hb in highlight_blockers]
    )

    zones = _zones(page)
    side_zones = [z for z in zones if z[0].startswith("left") or z[0].startswith("right")]
    other_zones = [z for z in zones if z not in side_zones]

    def find_in_zones(zone_list):
        cands = []
        for zn, zr in zone_list:
            is_gutter = zn.startswith("left") or zn.startswith("right")
            bw = (
                min(210.0, max(90.0, zr.width - 8.0))
                if is_gutter
                else min(300.0, max(140.0, zr.width - 12.0))
            )
            fs, wrapped, w, h = _optimize_layout(label, bw)

            points = _grid_positions(zr, w, h, step=18.0)
            points.sort(key=lambda p: abs(p.y - tc.y) + 0.2 * abs(p.x - tc.x))

            for c in points[: (950 if is_gutter else 550)]:
                cand = fitz.Rect(c.x - w / 2, c.y - h / 2, c.x + w / 2, c.y + h / 2)

                # EDIT A: clamp to the zone so it doesn't spill into the text area
                cand = _clamp_rect_to_zone(cand, zr)
                if not cand:
                    continue

                if _intersects_any(cand, blockers):
                    continue
                if _intersects_any(cand, occupied_buf):
                    continue
                if _intersects_any(cand, highlight_blockers):
                    continue

                ratio = INK_RATIO_MAX_GUTTER if is_gutter else INK_RATIO_MAX
                if _rect_has_ink_adaptive(page, inflate_rect(cand, INKCHECK_PAD), ratio_max=ratio):
                    continue

                s, e, hits, length = _straight_connector_best_pair(cand, target_union, conn_obs)

                # Keep crossings penalized, but not enough to override side placement.
                score = (hits * 2500.0) + length + abs((c.y - tc.y) * 0.7)

                # Strong side bias: if any gutter placement is possible, it should win.
                if is_gutter:
                    score -= 10000.0

                cands.append((score, cand, wrapped, fs, True))

        return sorted(cands, key=lambda x: x[0])

    # side first
    res = find_in_zones(side_zones)
    if not res:
        res = find_in_zones(other_zones)

    if res:
        return res[0][1], res[0][2], res[0][3], res[0][4]

    # ========================================================
    # EDIT D: fixed ultimate fallback rectangle (no geometry bug)
    # ========================================================
    fs, wrapped, w, h = _optimize_layout(label, 170.0)
    n = len(occupied_callouts)
    x0 = EDGE_PAD + (n % 6) * 20
    y0 = EDGE_PAD + (n // 6) * 40
    fb = fitz.Rect(x0, y0, x0 + w, y0 + h)
    return fb & page.rect, wrapped, fs, True


# ============================================================
# Search logic (including the chunking logic you provided)
# ============================================================

def _search_term(page: fitz.Page, term: str) -> List[fitz.Rect]:
    t = (term or "").strip()
    if not t:
        return []
    if len(t) > _MAX_TERM:
        t = t[:_MAX_TERM]

    # Standard search
    rs = page.search_for(t)
    if not rs:
        t2 = re.sub(r"\s+", " ", t)
        rs = page.search_for(t2)

    # Chunked fallback for long quotes
    if not rs and len(t) >= _CHUNK:
        hits = []
        step = max(12, _CHUNK - _CHUNK_OVERLAP)
        for i in range(0, len(t), step):
            chunk = t[i:i + _CHUNK].strip()
            if len(chunk) >= 18:
                hits.extend(page.search_for(chunk))
        if hits:
            hits.sort(key=lambda r: (r.y0, r.x0))
            merged = []
            for r in hits:
                if not merged:
                    merged.append(fitz.Rect(r))
                else:
                    if merged[-1].intersects(r) or abs(merged[-1].y0 - r.y0) < 3.0:
                        merged[-1] |= r
                    else:
                        merged.append(fitz.Rect(r))
            return merged

    return rs

def _url_variants(url: str) -> List[str]:
    u = (url or "").strip()
    if not u:
        return []
    v = [u, u.replace("https://", "").replace("http://", "").replace("www.", "")]
    if len(v[0]) > 45:
        v.append(v[0][:45])
    return list(dict.fromkeys(v))


def annotate_pdf_bytes(pdf_bytes: bytes, quote_terms: List[str], criterion_id: str, meta: Dict) -> Tuple[bytes, Dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if not doc:
        return pdf_bytes, {}
    page1 = doc[0]
    t_q, t_m, occ = 0, 0, []

    for page in doc:
        for t in (quote_terms or []):
            rs = _search_term(page, t)
            for r in rs:
                page.draw_rect(r, color=RED, width=BOX_WIDTH)
                t_q += 1

    def _do_job(label, needles, policy="union"):
        nonlocal t_m, occ
        tgs = []
        for n in needles:
            if n:
                tgs.extend(_search_term(page1, n))
        if not tgs:
            return
        for t in tgs:
            page1.draw_rect(t, color=RED, width=BOX_WIDTH)
        t_m += len(tgs)

        rect, txt, fs, safe = _choose_best_spot(page1, tgs, occ, label)
        if safe:
            page1.draw_rect(rect, color=WHITE, fill=WHITE, overlay=True)
        page1.insert_textbox(rect, txt, fontname=FONTNAME, fontsize=fs, color=RED)

        obs = (
            [inflate_rect(r, 1.5) for r in _page_text_shapes(page1)]
            + [inflate_rect(o, 2.0) for o in occ]
            + [inflate_rect(t, 2.5) for t in tgs]
        )

        if policy == "all":
            for t in tgs:
                _draw_straight_connector(page1, rect, t, obs)
        else:
            _draw_straight_connector(page1, rect, _union_rect(tgs), obs)

        occ.append(rect)

    _do_job("Original source of publication.", _url_variants(meta.get("source_url")))
    _do_job("The distinguished organization.", [n for n in [meta.get("venue_name"), meta.get("org_name")] if n])
    if meta.get("performance_date"):
        _do_job("Performance date.", [meta.get("performance_date")])
    if meta.get("salary_amount"):
        _do_job("Beneficiary salary evidence.", [meta.get("salary_amount")])

    b_names = list(filter(None, [meta.get("beneficiary_name")] + (meta.get("beneficiary_variants") or [])))
    if b_names:
        _do_job("Beneficiary lead role evidence.", list(set(b_names)), "all")

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    return out.getvalue(), {"total_quote_hits": t_q, "total_meta_hits": t_m, "criterion_id": criterion_id}
