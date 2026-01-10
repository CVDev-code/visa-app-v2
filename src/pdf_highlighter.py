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

# Must not overlap existing content (with padding)
GAP_FROM_TEXT_BLOCKS = 14.0
GAP_FROM_IMAGES = 12.0
GAP_FROM_DRAWINGS = 10.0

# Prevent "between lines" placements inside the text column
LINE_BLOCKER_PAD_X = 3.0
LINE_BLOCKER_PAD_Y = 8.0  # increase if you still see between-line placements

GAP_FROM_HIGHLIGHTS = 16.0
GAP_BETWEEN_CALLOUTS = 12.0

ENDPOINT_PULLBACK = 2.0

# Quote search robustness
_MAX_TERM = 600
_CHUNK = 70
_CHUNK_OVERLAP = 22

# ============================================================
# Gutters / zones
# ============================================================
GUTTER_WIDTHS = [70.0, 100.0, 140.0, 180.0]  # add 180 for very wide margins

# ============================================================
# Adaptive ink detection (THE big change)
# ============================================================
INKCHECK_DPI = 96
INKCHECK_PAD = 1.5

# We detect "ink" by contrast vs local background, not by "is it white?"
# - sample_stride: higher = faster but less sensitive
# - delta: how far from background counts as "ink" (lower = stricter)
# - ratio: max allowed fraction of "ink" pixels
INK_SAMPLE_STRIDE = 6
INK_DELTA = 18          # 14–22 typical
INK_RATIO_MAX = 0.012   # 0.006 stricter, 0.02 more permissive

# Gutters often have faint noise; allow slightly more
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
    vx = from_pt.x - to_pt.x
    vy = from_pt.y - to_pt.y
    d = math.hypot(vx, vy)
    if d == 0:
        return to_pt
    ux, uy = vx / d, vy / d
    return fitz.Point(to_pt.x + ux * dist, to_pt.y + uy * dist)

def _edge_candidates(rect: fitz.Rect) -> List[fitz.Point]:
    cx = rect.x0 + rect.width / 2.0
    cy = rect.y0 + rect.height / 2.0
    return [
        fitz.Point(rect.x0, cy),
        fitz.Point(rect.x1, cy),
        fitz.Point(cx, rect.y0),
        fitz.Point(cx, rect.y1),
        fitz.Point(rect.x0, rect.y0),
        fitz.Point(rect.x1, rect.y0),
        fitz.Point(rect.x0, rect.y1),
        fitz.Point(rect.x1, rect.y1),
    ]

def _mid_height_anchor(callout: fitz.Rect, toward: fitz.Point) -> fitz.Point:
    y = callout.y0 + (callout.height / 2.0)
    cx = callout.x0 + callout.width / 2.0
    if toward.x >= cx:
        return fitz.Point(callout.x1, y)
    return fitz.Point(callout.x0, y)

def _straight_connector_best_pair(
    callout_rect: fitz.Rect,
    target_rect: fitz.Rect,
    obstacles: List[fitz.Rect],
) -> Tuple[fitz.Point, fitz.Point, int, float]:
    target_center = _center(target_rect)
    base = _mid_height_anchor(callout_rect, target_center)

    nudges = [-14.0, -7.0, 0.0, 7.0, 14.0]
    starts: List[fitz.Point] = []
    for dy in nudges:
        y = min(max(callout_rect.y0 + 2.0, base.y + dy), callout_rect.y1 - 2.0)
        starts.append(fitz.Point(base.x, y))

    ends = _edge_candidates(target_rect)

    best_hits = 10**9
    best_len = 10**9
    best_s, best_e = starts[0], ends[0]

    for s in starts:
        for e in ends:
            hits = 0
            for ob in obstacles:
                if _segment_hits_rect(s, e, ob):
                    hits += 1
            length = math.hypot(e.x - s.x, e.y - s.y)
            if hits < best_hits or (hits == best_hits and length < best_len):
                best_hits, best_len = hits, length
                best_s, best_e = s, e

    best_e = _pull_back_point(best_s, best_e, ENDPOINT_PULLBACK)
    return best_s, best_e, best_hits, best_len

def _draw_straight_connector(page: fitz.Page, callout_rect: fitz.Rect, target_rect: fitz.Rect, obstacles: List[fitz.Rect]):
    s, e, _, _ = _straight_connector_best_pair(callout_rect, target_rect, obstacles)
    page.draw_line(s, e, color=RED, width=LINE_WIDTH)


# ============================================================
# Adaptive ink check (contrast-based, not whiteness-based)
# ============================================================

def _rect_has_ink_adaptive(
    page: fitz.Page,
    rect: fitz.Rect,
    *,
    dpi: int = INKCHECK_DPI,
    sample_stride: int = INK_SAMPLE_STRIDE,
    delta: int = INK_DELTA,
    ratio_max: float = INK_RATIO_MAX,
) -> bool:
    """
    True if region contains "ink" (content) above tolerance.
    Uses local background estimation: treat uniform grey as empty.
    """
    r = (fitz.Rect(rect) & page.rect)
    if r.is_empty or r.width < 2 or r.height < 2:
        return True

    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, clip=r, alpha=False)
    s = pix.samples
    if not s:
        return True

    w, h = pix.width, pix.height
    if w < 2 or h < 2:
        return True

    # Sample pixels (stride) to estimate background median-ish via running average + robust clamp.
    # We avoid numpy; this is fast enough because callout rects are small.
    step = max(1, sample_stride)
    total = 0
    sr = sg = sb = 0

    # First pass: average
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

    br = sr / total
    bg = sg / total
    bb = sb / total

    # Second pass: count pixels that deviate from background by > delta (any channel)
    ink = 0
    total2 = 0
    for y in range(0, h, step):
        row = y * w * 3
        for x in range(0, w, step):
            i = row + x * 3
            if (abs(s[i] - br) > delta) or (abs(s[i + 1] - bg) > delta) or (abs(s[i + 2] - bb) > delta):
                ink += 1
            total2 += 1

    if total2 <= 0:
        return True

    return (ink / total2) > ratio_max


# ============================================================
# Blockers (blocks + words + lines) to prevent between-lines
# ============================================================

def _page_text_blocks(page: fitz.Page) -> List[fitz.Rect]:
    blocks: List[fitz.Rect] = []
    try:
        for b in page.get_text("blocks"):
            blocks.append(fitz.Rect(b[:4]))
    except Exception:
        pass
    return blocks

def _page_word_rects(page: fitz.Page) -> List[fitz.Rect]:
    rects: List[fitz.Rect] = []
    try:
        for w in page.get_text("words"):
            rects.append(fitz.Rect(w[:4]))
    except Exception:
        pass
    return rects

def _page_line_rects(page: fitz.Page) -> List[fitz.Rect]:
    rects: List[fitz.Rect] = []
    try:
        d = page.get_text("dict")
        for b in d.get("blocks", []):
            for ln in b.get("lines", []):
                bbox = ln.get("bbox")
                if bbox:
                    rects.append(inflate_rect_xy(fitz.Rect(bbox), LINE_BLOCKER_PAD_X, LINE_BLOCKER_PAD_Y))
    except Exception:
        pass
    return rects

def _page_text_shapes(page: fitz.Page) -> List[fitz.Rect]:
    rects: List[fitz.Rect] = []
    rects.extend(_page_text_blocks(page))
    rects.extend(_page_word_rects(page))
    rects.extend(_page_line_rects(page))
    return rects

def _page_images(page: fitz.Page) -> List[fitz.Rect]:
    imgs: List[fitz.Rect] = []
    try:
        for img in page.get_images(full=True):
            xref = img[0]
            for r in page.get_image_rects(xref):
                imgs.append(fitz.Rect(r))
    except Exception:
        pass
    return imgs

def _page_drawings(page: fitz.Page) -> List[fitz.Rect]:
    dr: List[fitz.Rect] = []
    try:
        for d in page.get_drawings():
            rr = d.get("rect")
            if rr:
                dr.append(fitz.Rect(rr))
    except Exception:
        pass
    return dr

def _page_blockers(page: fitz.Page) -> List[fitz.Rect]:
    blockers: List[fitz.Rect] = []
    for r in _page_text_shapes(page):
        blockers.append(inflate_rect(r, GAP_FROM_TEXT_BLOCKS))
    for r in _page_images(page):
        blockers.append(inflate_rect(r, GAP_FROM_IMAGES))
    for r in _page_drawings(page):
        blockers.append(inflate_rect(r, GAP_FROM_DRAWINGS))
    return blockers

def _text_envelope(page: fitz.Page) -> Optional[fitz.Rect]:
    blocks = _page_text_blocks(page)
    if not blocks:
        return None
    return _union_rect(blocks)

def _intersects_any(r: fitz.Rect, others: List[fitz.Rect]) -> bool:
    return any(r.intersects(o) for o in others)


# ============================================================
# Callout text wrapping
# ============================================================

def _optimize_layout(text: str, box_width: float) -> Tuple[int, str, float, float]:
    text = (text or "").strip()
    if not text:
        return 12, "", box_width, 24.0

    for fs in FONT_SIZES:
        words = text.split()
        lines: List[str] = []
        cur: List[str] = []
        usable_w = max(10.0, box_width - 10.0)

        for w in words:
            trial = " ".join(cur + [w])
            if fitz.get_text_length(trial, fontname=FONTNAME, fontsize=fs) <= usable_w:
                cur.append(w)
            else:
                if cur:
                    lines.append(" ".join(cur))
                    cur = [w]
                else:
                    lines.append(w)
                    cur = []
        if cur:
            lines.append(" ".join(cur))

        h = (len(lines) * fs * 1.22) + 10.0
        if h <= 100.0 or fs == 10:
            return fs, "\n".join(lines), box_width, h

    return 10, text, box_width, 44.0


# ============================================================
# Zones: multi-width gutters + top/bottom derived from envelope
# ============================================================

def _zones(page: fitz.Page) -> List[Tuple[str, fitz.Rect]]:
    pr = page.rect
    env = _text_envelope(page)
    zones: List[Tuple[str, fitz.Rect]] = []

    for gw in GUTTER_WIDTHS:
        left = fitz.Rect(EDGE_PAD, EDGE_PAD, min(pr.width - EDGE_PAD, EDGE_PAD + gw), pr.height - EDGE_PAD)
        right = fitz.Rect(max(EDGE_PAD, pr.width - EDGE_PAD - gw), EDGE_PAD, pr.width - EDGE_PAD, pr.height - EDGE_PAD)
        if left.width > 40:
            zones.append((f"left_{int(gw)}", left))
        if right.width > 40:
            zones.append((f"right_{int(gw)}", right))

    if env is not None and env.get_area() > 0:
        env2 = inflate_rect(env, max(6.0, GAP_FROM_TEXT_BLOCKS * 0.4))
        top = fitz.Rect(EDGE_PAD, EDGE_PAD, pr.width - EDGE_PAD, max(EDGE_PAD, env2.y0))
        bottom = fitz.Rect(EDGE_PAD, min(pr.height - EDGE_PAD, env2.y1), pr.width - EDGE_PAD, pr.height - EDGE_PAD)
        if top.height > 30:
            zones.append(("top", top))
        if bottom.height > 30:
            zones.append(("bottom", bottom))

    if not zones:
        zones.append(("full", fitz.Rect(EDGE_PAD, EDGE_PAD, pr.width - EDGE_PAD, pr.height - EDGE_PAD)))

    return zones


# ============================================================
# Placement (HARD side-first + adaptive ink test)
# ============================================================

def _box_width_for_zone(zone_name: str, zone_rect: fitz.Rect) -> float:
    if zone_name.startswith("left") or zone_name.startswith("right"):
        return min(210.0, max(90.0, zone_rect.width - 8.0))
    if zone_name in ("top", "bottom"):
        return min(300.0, max(140.0, zone_rect.width - 12.0))
    return min(220.0, max(110.0, zone_rect.width - 12.0))

def _clamp_rect_to_zone(r: fitz.Rect, z: fitz.Rect) -> Optional[fitz.Rect]:
    rr = fitz.Rect(r)

    if rr.y0 < z.y0:
        rr.y1 += (z.y0 - rr.y0)
        rr.y0 = z.y0
    if rr.y1 > z.y1:
        rr.y0 -= (rr.y1 - z.y1)
        rr.y1 = z.y1

    if rr.x0 < z.x0:
        rr.x1 += (z.x0 - rr.x0)
        rr.x0 = z.x0
    if rr.x1 > z.x1:
        rr.x0 -= (rr.x1 - z.x1)
        rr.x1 = z.x1

    if rr.x0 < z.x0 or rr.x1 > z.x1 or rr.y0 < z.y0 or rr.y1 > z.y1:
        return None
    return rr

def _grid_positions(z: fitz.Rect, w: float, h: float, *, step: float) -> List[fitz.Point]:
    pts: List[fitz.Point] = []
    x0 = z.x0 + w / 2
    x1 = z.x1 - w / 2
    y0 = z.y0 + h / 2
    y1 = z.y1 - h / 2
    if x1 < x0 or y1 < y0:
        return pts
    y = y0
    while y <= y1 + 0.01:
        x = x0
        while x <= x1 + 0.01:
            pts.append(fitz.Point(x, y))
            x += step
        y += step
    return pts

def _choose_best_spot(
    page: fitz.Page,
    targets: List[fitz.Rect],
    occupied_callouts: List[fitz.Rect],
    label: str,
) -> Tuple[fitz.Rect, str, int, bool]:
    pr = page.rect
    target_union = _union_rect(targets)
    tc = _center(target_union)

    blockers = _page_blockers(page)
    highlight_blockers = [inflate_rect(t, GAP_FROM_HIGHLIGHTS) for t in targets]
    occupied_buf = [inflate_rect(o, GAP_BETWEEN_CALLOUTS) for o in occupied_callouts]

    connector_obstacles: List[fitz.Rect] = []
    for b in _page_text_shapes(page):
        connector_obstacles.append(inflate_rect(b, 1.5))
    for o in occupied_callouts:
        connector_obstacles.append(inflate_rect(o, 2.0))
    for hb in highlight_blockers:
        connector_obstacles.append(inflate_rect(hb, 1.0))

    zones = _zones(page)
    side_zones = [(n, r) for (n, r) in zones if n.startswith("left") or n.startswith("right")]
    other_zones = [(n, r) for (n, r) in zones if (n, r) not in side_zones]

    GRID_STEP = 18.0

    def score_candidate(cand: fitz.Rect) -> float:
        s, e, hits, length = _straight_connector_best_pair(cand, target_union, connector_obstacles)
        for tb in _page_text_shapes(page):
            if _segment_hits_rect(s, e, tb):
                hits += 2
        # heavily weight crossings; keep it readable
        return (hits * 7000.0) + length + abs((_center(cand).y - tc.y) * 0.7)

    def candidates_for_zone(zone_name: str, z: fitz.Rect) -> List[Tuple[float, fitz.Rect, str, int, bool]]:
        bw = _box_width_for_zone(zone_name, z)
        fs, wrapped, w, h = _optimize_layout(label, bw)
        if z.width < w + 2 or z.height < h + 2:
            return []

        pts = _grid_positions(z, w, h, step=GRID_STEP)
        pts.sort(key=lambda p: abs(p.y - tc.y) + 0.2 * abs(p.x - tc.x))

        is_gutter = zone_name.startswith("left") or zone_name.startswith("right")
        ratio_max = INK_RATIO_MAX_GUTTER if is_gutter else INK_RATIO_MAX

        out: List[Tuple[float, fitz.Rect, str, int, bool]] = []
        # Try more points in gutters because we really want side placement
        limit = 950 if is_gutter else 550

        for c in pts[:limit]:
            cand = fitz.Rect(c.x - w/2, c.y - h/2, c.x + w/2, c.y + h/2)
            cand = _clamp_rect_to_zone(cand, z)
            if not cand:
                continue

            if _intersects_any(cand, blockers):
                continue
            if _intersects_any(cand, occupied_buf):
                continue
            if _intersects_any(cand, highlight_blockers):
                continue

            if _rect_has_ink_adaptive(
                page,
                inflate_rect(cand, INKCHECK_PAD),
                dpi=INKCHECK_DPI,
                sample_stride=INK_SAMPLE_STRIDE,
                delta=INK_DELTA,
                ratio_max=ratio_max,
            ):
                continue

            sc = score_candidate(cand)

            # HARD preference: within gutters, bias even more so we pick side when possible
            if is_gutter:
                sc -= 1200.0

            out.append((sc, cand, wrapped, fs, True))

        out.sort(key=lambda x: x[0])
        return out

    # 1) HARD SIDE-FIRST
    side_candidates: List[Tuple[float, fitz.Rect, str, int, bool]] = []
    for zn, zr in side_zones:
        side_candidates.extend(candidates_for_zone(zn, zr))
    if side_candidates:
        _, best_rect, wrapped, fs, safe = sorted(side_candidates, key=lambda x: x[0])[0]
        return best_rect, wrapped, fs, safe

    # 2) If no side possible, use top/bottom/full
    other_candidates: List[Tuple[float, fitz.Rect, str, int, bool]] = []
    for zn, zr in other_zones:
        other_candidates.extend(candidates_for_zone(zn, zr))
    if other_candidates:
        _, best_rect, wrapped, fs, safe = sorted(other_candidates, key=lambda x: x[0])[0]
        return best_rect, wrapped, fs, safe

    # 3) final fallback: place somewhere safe-ish without stacking
    bw = 170.0
    fs, wrapped, w, h = _optimize_layout(label, bw)
    n = len(occupied_callouts)
    x0 = EDGE_PAD + (n % 6) * (w + 10.0)
    y0 = EDGE_PAD + (n // 6) * (h + 10.0)
    fb = fitz.Rect(x0, y0, x0 + w, y0 + h)
    fb = fb & page.rect
    return fb, wrapped, fs, True


# ============================================================
# Search helpers
# ============================================================

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _search_exact_or_normalized(page: fitz.Page, needle: str) -> List[fitz.Rect]:
    needle = (needle or "").strip()
    if not needle:
        return []
    flags = 0
    try:
        flags |= fitz.TEXT_DEHYPHENATE
    except Exception:
        pass
    try:
        flags |= fitz.TEXT_PRESERVE_WHITESPACE
    except Exception:
        pass

    try:
        rs = page.search_for(needle, flags=flags)
        if rs:
            return rs
    except Exception:
        pass

    n2 = _normalize_spaces(needle)
    if n2 and n2 != needle:
        try:
            rs = page.search_for(n2, flags=flags)
            if rs:
                return rs
        except Exception:
            pass

    return []

def _search_term(page: fitz.Page, term: str) -> List[fitz.Rect]:
    t = (term or "").strip()
    if not t:
        return []
    if len(t) > _MAX_TERM:
        t = t[:_MAX_TERM]

    rs = _search_exact_or_normalized(page, t)
    if rs:
        return rs

    t2 = _normalize_spaces(t)
    if len(t2) >= _CHUNK:
        hits: List[fitz.Rect] = []
        step = max(12, _CHUNK - _CHUNK_OVERLAP)
        for i in range(0, len(t2), step):
            chunk = t2[i:i + _CHUNK].strip()
            if len(chunk) < 18:
                continue
            hits.extend(_search_exact_or_normalized(page, chunk))

        if hits:
            hits_sorted = sorted(hits, key=lambda r: (r.y0, r.x0))
            merged: List[fitz.Rect] = []
            for r in hits_sorted:
                if not merged:
                    merged.append(fitz.Rect(r))
                else:
                    last = merged[-1]
                    if last.intersects(r) or abs(last.y0 - r.y0) < 3.0:
                        merged[-1] = last | r
                    else:
                        merged.append(fitz.Rect(r))
            return merged

    return []

def _url_variants(url: str) -> List[str]:
    u = (url or "").strip()
    if not u:
        return []
    out = [u]
    out.append(u.replace("https://", "").replace("http://", ""))
    out.append(u.replace("https://", "").replace("http://", "").replace("www.", ""))
    if len(out[0]) > 45:
        out.append(out[0][:45])
    seen = set()
    uniq = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


# ============================================================
# Main entrypoint
# ============================================================

def annotate_pdf_bytes(pdf_bytes: bytes, quote_terms: List[str], criterion_id: str, meta: Dict) -> Tuple[bytes, Dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(doc) == 0:
        return pdf_bytes, {}

    page1 = doc.load_page(0)

    total_quote_hits = 0
    total_meta_hits = 0
    occupied_callouts: List[fitz.Rect] = []

    # A) Quote highlights (ALL pages)
    for page in doc:
        for term in (quote_terms or []):
            rects = _search_term(page, term)
            for r in rects:
                page.draw_rect(r, color=RED, width=BOX_WIDTH)
                total_quote_hits += 1

    # B) Metadata callouts (page 1)
    def _do_job(label: str, needles: List[str], *, connect_policy: str = "union"):
        nonlocal total_meta_hits, occupied_callouts

        targets: List[fitz.Rect] = []
        for n in needles:
            if not n:
                continue
            targets.extend(_search_term(page1, n))

        if not targets:
            return

        for t in targets:
            page1.draw_rect(t, color=RED, width=BOX_WIDTH)
        total_meta_hits += len(targets)

        callout_rect, wrapped_text, fs, safe_for_white_bg = _choose_best_spot(page1, targets, occupied_callouts, label)

        if safe_for_white_bg:
            page1.draw_rect(callout_rect, color=WHITE, fill=WHITE, overlay=True)

        page1.insert_textbox(
            callout_rect,
            wrapped_text,
            fontname=FONTNAME,
            fontsize=fs,
            color=RED,
            align=fitz.TEXT_ALIGN_LEFT,
        )

        obstacles: List[fitz.Rect] = []
        for b in _page_text_shapes(page1):
            obstacles.append(inflate_rect(b, 1.5))
        for o in occupied_callouts:
            obstacles.append(inflate_rect(o, 2.0))
        obstacles.extend([inflate_rect(t, 2.5) for t in targets])

        def connect_to(rect: fitz.Rect):
            _draw_straight_connector(page1, callout_rect, rect, obstacles)

        if connect_policy == "all":
            for t in targets:
                connect_to(t)
        elif connect_policy == "single":
            connect_to(targets[0])
        else:
            connect_to(_union_rect(targets))

        occupied_callouts.append(callout_rect)

    _do_job(
        "Original source of publication.",
        _url_variants(str(meta.get("source_url") or "")),
        connect_policy="union",
    )

    venue = (meta.get("venue_name") or "").strip()
    org = (meta.get("org_name") or "").strip()
    if venue or org:
        _do_job("The distinguished organization.", [venue, org], connect_policy="union")

    perf = (meta.get("performance_date") or "").strip()
    if perf:
        _do_job("Performance date.", [perf], connect_policy="union")

    sal = (meta.get("salary_amount") or "").strip()
    if sal:
        _do_job("Beneficiary salary evidence.", [sal], connect_policy="union")

    bname = (meta.get("beneficiary_name") or "").strip()
    variants = meta.get("beneficiary_variants") or []
    needles: List[str] = []
    if bname:
        needles.append(bname)
    for v in variants:
        vv = (v or "").strip()
        if vv:
            needles.append(vv)

    if needles:
        seen = set()
        uniq = []
        for x in needles:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        _do_job("Beneficiary lead role evidence.", uniq, connect_policy="all")

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    out.seek(0)

    return out.getvalue(), {
        "total_quote_hits": total_quote_hits,
        "total_meta_hits": total_meta_hits,
        "criterion_id": criterion_id,
    }
