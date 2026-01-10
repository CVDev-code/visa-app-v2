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

# Keep these modest because we now hard-block the entire text column
GAP_FROM_TEXT_BLOCKS = 6.0
GAP_FROM_WORDS = 2.0
GAP_FROM_LINES = 0.0  # line bboxes already padded below

GAP_FROM_IMAGES = 12.0
GAP_FROM_DRAWINGS = 10.0

# Between-line protection (prevents "sneaking into gaps")
LINE_BLOCKER_PAD_X = 3.0
LINE_BLOCKER_PAD_Y = 8.0

GAP_FROM_HIGHLIGHTS = 16.0
GAP_BETWEEN_CALLOUTS = 12.0

ENDPOINT_PULLBACK = 2.0

# Quote search robustness
_MAX_TERM = 600
_CHUNK = 70
_CHUNK_OVERLAP = 22

# ============================================================
# Adaptive ink detection (relaxed for gutters)
# ============================================================
INKCHECK_DPI = 96
INKCHECK_PAD = 1.5
INK_SAMPLE_STRIDE = 6
INK_DELTA = 18
INK_RATIO_MAX = 0.012
INK_RATIO_MAX_GUTTER = 0.030  # permissive for scanned/pixelated margins
INK_DELTA_GUTTER_BONUS = 12   # ignore light-grey noise in gutters

# ============================================================
# Text column envelope behaviour
# ============================================================
TEXT_COLUMN_BUFFER_X = 15.0     # expands the no-go column left/right
SIDE_ZONE_GAP_FROM_TEXT = 10.0  # distance from envelope to start of side zones
MIN_SIDE_ZONE_W = 55.0
MIN_ZONE_H = 55.0
MIN_TOPBOTTOM_H = 40.0

# Grid search
GRID_STEP = 15.0
GRID_LIMIT_SIDE = 900
GRID_LIMIT_OTHER = 650


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
    callout_rect: fitz.Rect,
    target_rect: fitz.Rect,
    obstacles: List[fitz.Rect],
) -> Tuple[fitz.Point, fitz.Point, int, float]:
    target_center = _center(target_rect)
    base = _mid_height_anchor(callout_rect, target_center)

    starts = [
        fitz.Point(
            base.x,
            min(max(callout_rect.y0 + 2.0, base.y + dy), callout_rect.y1 - 2.0),
        )
        for dy in (-14.0, -7.0, 0.0, 7.0, 14.0)
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
    page: fitz.Page,
    callout_rect: fitz.Rect,
    target_rect: fitz.Rect,
    obstacles: List[fitz.Rect],
):
    s, e, _, _ = _straight_connector_best_pair(callout_rect, target_rect, obstacles)
    page.draw_line(s, e, color=RED, width=LINE_WIDTH)

def _clamp_rect_to_zone(r: fitz.Rect, z: fitz.Rect) -> Optional[fitz.Rect]:
    rr = fitz.Rect(r)

    if rr.x0 < z.x0:
        rr.x1 += (z.x0 - rr.x0)
        rr.x0 = z.x0
    if rr.x1 > z.x1:
        rr.x0 -= (rr.x1 - z.x1)
        rr.x1 = z.x1

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

def _intersects_any(r: fitz.Rect, others: List[fitz.Rect]) -> bool:
    return any(r.intersects(o) for o in others)


# ============================================================
# Adaptive Ink Check (contrast-based, relaxed for gutters)
# ============================================================

def _rect_has_ink_adaptive(
    page: fitz.Page,
    rect: fitz.Rect,
    *,
    is_gutter: bool = False,
    dpi: int = INKCHECK_DPI,
) -> bool:
    r = (fitz.Rect(rect) & page.rect)
    if r.is_empty or r.width < 2 or r.height < 2:
        return True

    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=r, alpha=False)
    s, w, h = pix.samples, pix.width, pix.height
    if not s or w < 2 or h < 2:
        return True

    ratio_max = INK_RATIO_MAX_GUTTER if is_gutter else INK_RATIO_MAX
    current_delta = INK_DELTA + (INK_DELTA_GUTTER_BONUS if is_gutter else 0)

    step = max(1, INK_SAMPLE_STRIDE)

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
            if (
                abs(s[i] - br) > current_delta
                or abs(s[i + 1] - bg) > current_delta
                or abs(s[i + 2] - bb) > current_delta
            ):
                ink += 1
            total2 += 1

    return (ink / total2) > ratio_max if total2 > 0 else True


# ============================================================
# Blockers + text envelope
# ============================================================

def _page_text_blocks(page: fitz.Page) -> List[fitz.Rect]:
    rects: List[fitz.Rect] = []
    try:
        for b in page.get_text("blocks"):
            rects.append(fitz.Rect(b[:4]))
    except Exception:
        pass
    return rects

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

    # Blocks are often huge; keep padding small.
    for r in _page_text_blocks(page):
        blockers.append(inflate_rect(r, GAP_FROM_TEXT_BLOCKS))

    # Words are precise; light padding.
    for r in _page_word_rects(page):
        blockers.append(inflate_rect(r, GAP_FROM_WORDS))

    # Lines are already padded.
    for r in _page_line_rects(page):
        blockers.append(inflate_rect(r, GAP_FROM_LINES))

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


# ============================================================
# Callout text layout
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
        if h <= 120.0 or fs == 10:
            return fs, "\n".join(lines), box_width, h

    return 10, text, box_width, 44.0

def _grid_positions(z: fitz.Rect, w: float, h: float, step: float) -> List[fitz.Point]:
    pts: List[fitz.Point] = []
    y = z.y0 + h / 2
    while y <= z.y1 - h / 2:
        x = z.x0 + w / 2
        while x <= z.x1 - w / 2:
            pts.append(fitz.Point(x, y))
            x += step
        y += step
    return pts


# ============================================================
# Zones built relative to the text envelope (side-first)
# ============================================================

def _build_zones_from_envelope(page: fitz.Page) -> List[Tuple[str, fitz.Rect]]:
    pr = page.rect
    safe = fitz.Rect(EDGE_PAD, EDGE_PAD, pr.width - EDGE_PAD, pr.height - EDGE_PAD)

    zones: List[Tuple[str, fitz.Rect]] = []
    env = _text_envelope(page)

    def _add_zone(name: str, r: fitz.Rect, *, min_w=MIN_SIDE_ZONE_W, min_h=MIN_ZONE_H):
        rr = fitz.Rect(r) & safe
        if rr.width >= min_w and rr.height >= min_h:
            zones.append((name, rr))

    if env:
        _add_zone("left", fitz.Rect(safe.x0, safe.y0, env.x0 - SIDE_ZONE_GAP_FROM_TEXT, safe.y1))
        _add_zone("right", fitz.Rect(env.x1 + SIDE_ZONE_GAP_FROM_TEXT, safe.y0, safe.x1, safe.y1))
        _add_zone("top", fitz.Rect(safe.x0, safe.y0, safe.x1, env.y0 - 10.0), min_h=MIN_TOPBOTTOM_H)
        _add_zone("bottom", fitz.Rect(safe.x0, env.y1 + 10.0, safe.x1, safe.y1), min_h=MIN_TOPBOTTOM_H)
    else:
        _add_zone("left", fitz.Rect(safe.x0, safe.y0, safe.x0 + 140.0, safe.y1))
        _add_zone("right", fitz.Rect(safe.x1 - 140.0, safe.y0, safe.x1, safe.y1))
        _add_zone("top", fitz.Rect(safe.x0, safe.y0, safe.x1, safe.y0 + 110.0), min_h=MIN_TOPBOTTOM_H)
        _add_zone("bottom", fitz.Rect(safe.x0, safe.y1 - 110.0, safe.x1, safe.y1), min_h=MIN_TOPBOTTOM_H)

    return zones if zones else [("full", safe)]


# ============================================================
# Placement (forces side when possible)
# ============================================================

def _choose_best_spot(page: fitz.Page, targets: List[fitz.Rect], occupied_callouts: List[fitz.Rect], label: str):
    target_union = _union_rect(targets)
    tc = _center(target_union)
    pr = page.rect

    env = _text_envelope(page)
    text_column_blocker = None
    if env:
        x0 = max(pr.x0, env.x0 - TEXT_COLUMN_BUFFER_X)
        x1 = min(pr.x1, env.x1 + TEXT_COLUMN_BUFFER_X)
        if x1 > x0 + 10:
            text_column_blocker = fitz.Rect(x0, pr.y0, x1, pr.y1)

    blockers = _page_blockers(page)
    if text_column_blocker:
        blockers.append(text_column_blocker)

    highlight_blockers = [inflate_rect(t, GAP_FROM_HIGHLIGHTS) for t in targets]
    occupied_buf = [inflate_rect(o, GAP_BETWEEN_CALLOUTS) for o in occupied_callouts]

    zones = _build_zones_from_envelope(page)
    side_zones = [z for z in zones if z[0] in ("left", "right")]
    other_zones = [z for z in zones if z[0] not in ("left", "right")]

    def find_in_zones(zone_list):
        candidates = []
        for zn, zr in zone_list:
            is_side = (zn in ("left", "right"))
            bw = min(200.0, max(80.0, zr.width - 10.0))
            fs, wrapped, w, h = _optimize_layout(label, bw)

            pts = _grid_positions(zr, w, h, step=GRID_STEP)
            pts.sort(key=lambda p: abs(p.y - tc.y))

            limit = GRID_LIMIT_SIDE if is_side else GRID_LIMIT_OTHER
            for c in pts[:limit]:
                cand = fitz.Rect(c.x - w / 2, c.y - h / 2, c.x + w / 2, c.y + h / 2)
                cand = _clamp_rect_to_zone(cand, zr)
                if not cand:
                    continue

                if _intersects_any(cand, blockers):
                    continue
                if _intersects_any(cand, occupied_buf):
                    continue
                if _intersects_any(cand, highlight_blockers):
                    continue

                if _rect_has_ink_adaptive(page, inflate_rect(cand, INKCHECK_PAD), is_gutter=is_side):
                    continue

                dist_y = abs(c.y - tc.y)
                score = dist_y - (20000 if is_side else 0)
                candidates.append((score, cand, wrapped, fs, True))

        candidates.sort(key=lambda x: x[0])
        return candidates

    res = find_in_zones(side_zones)
    if not res:
        res = find_in_zones(other_zones)

    if res:
        return res[0][1], res[0][2], res[0][3], res[0][4]

    # Packed fallback
    fs, wrapped, w, h = _optimize_layout(label, 150.0)
    safe = fitz.Rect(EDGE_PAD, EDGE_PAD, pr.width - EDGE_PAD, pr.height - EDGE_PAD)
    occ_buf2 = [inflate_rect(o, GAP_BETWEEN_CALLOUTS) for o in occupied_callouts]

    for y in range(int(safe.y0), int(safe.y1 - h), 30):
        for x in range(int(safe.x0), int(safe.x1 - w), 30):
            cand = fitz.Rect(x, y, x + w, y + h)
            if _intersects_any(cand, blockers):
                continue
            if _intersects_any(cand, occ_buf2):
                continue
            if _intersects_any(cand, highlight_blockers):
                continue
            return cand, wrapped, fs, True

    fb = fitz.Rect(EDGE_PAD, EDGE_PAD, EDGE_PAD + w, EDGE_PAD + h) & pr
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

        callout_rect, wrapped_text, fs, safe_for_white_bg = _choose_best_spot(
            page1, targets, occupied_callouts, label
        )

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

        # Connector obstacles (avoid crossing text and existing callouts)
        obstacles: List[fitz.Rect] = []
        for r in _page_text_blocks(page1):
            obstacles.append(inflate_rect(r, 1.5))
        for r in _page_word_rects(page1):
            obstacles.append(inflate_rect(r, 1.0))
        for r in _page_line_rects(page1):
            obstacles.append(inflate_rect(r, 1.0))
        for o in occupied_callouts:
            obstacles.append(inflate_rect(o, 2.0))

        highlight_obs = [inflate_rect(t, 2.5) for t in targets]
        obstacles.extend(highlight_obs)

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

    # Source URL
    _do_job("Original source of publication.", _url_variants(str(meta.get("source_url") or "")), connect_policy="union")

    # Venue / org
    venue = (meta.get("venue_name") or "").strip()
    org = (meta.get("org_name") or "").strip()
    if venue or org:
        _do_job("The distinguished organization.", [venue, org], connect_policy="union")

    # Performance date
    perf = (meta.get("performance_date") or "").strip()
    if perf:
        _do_job("Performance date.", [perf], connect_policy="union")

    # Salary
    sal = (meta.get("salary_amount") or "").strip()
    if sal:
        _do_job("Beneficiary salary evidence.", [sal], connect_policy="union")

    # Beneficiary name + variants
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
