import io
import math
import re
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# ---- style knobs ----
BOX_WIDTH = 1.7
LINE_WIDTH = 1.6
FONTNAME = "Times-Bold"
FONT_SIZES = [12, 11, 10]

# ---- spacing knobs ----
EDGE_PAD = 18.0
GAP_FROM_TEXT_BLOCKS = 10.0         # keep callout away from original text
GAP_FROM_HIGHLIGHTS = 14.0
GAP_BETWEEN_CALLOUTS = 10.0
ENDPOINT_PULLBACK = 1.5             # line stops at box edge (pulled back)

# ---- quote search robustness ----
_MAX_TERM = 600
_CHUNK = 60
_CHUNK_OVERLAP = 18

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
    steps = 22
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

def _mid_edge_anchor(callout: fitz.Rect, toward: fitz.Point) -> fitz.Point:
    """Mid-height on left or right edge of callout box."""
    y = callout.y0 + (callout.height / 2.0)
    cx = callout.x0 + callout.width / 2.0
    if toward.x >= cx:
        return fitz.Point(callout.x1, y)
    return fitz.Point(callout.x0, y)

def _edge_candidates(rect: fitz.Rect) -> List[fitz.Point]:
    """Edge points to sample endpoints on the target box."""
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

def _straight_connector_best_pair(
    callout_rect: fitz.Rect,
    target_rect: fitz.Rect,
    obstacles: List[fitz.Rect],
) -> Tuple[fitz.Point, fitz.Point]:
    """Always returns a straight segment (start,end) minimizing crossings then length."""
    target_center = _center(target_rect)

    base = _mid_edge_anchor(callout_rect, target_center)
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
    return best_s, best_e

def _draw_straight_connector(
    page: fitz.Page,
    callout_rect: fitz.Rect,
    target_rect: fitz.Rect,
    obstacles: List[fitz.Rect],
):
    s, e = _straight_connector_best_pair(callout_rect, target_rect, obstacles)
    page.draw_line(s, e, color=RED, width=LINE_WIDTH)

# ============================================================
# Page blockers (for avoiding overlaps) + TEXT ENVELOPE
# ============================================================

def _page_text_blocks(page: fitz.Page) -> List[fitz.Rect]:
    blocks: List[fitz.Rect] = []
    try:
        for b in page.get_text("blocks"):
            blocks.append(fitz.Rect(b[:4]))
    except Exception:
        pass
    return blocks

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

def _page_blockers(page: fitz.Page, pad: float = GAP_FROM_TEXT_BLOCKS) -> List[fitz.Rect]:
    blockers: List[fitz.Rect] = []
    for r in _page_text_blocks(page):
        blockers.append(inflate_rect(r, pad))
    for r in _page_images(page):
        blockers.append(inflate_rect(r, pad))
    try:
        for d in page.get_drawings():
            rr = d.get("rect")
            if rr:
                blockers.append(inflate_rect(fitz.Rect(rr), pad))
    except Exception:
        pass
    return blockers

def _text_envelope(page: fitz.Page) -> Optional[fitz.Rect]:
    """
    A conservative rectangle covering all text on the page.
    We will ONLY place callouts outside this envelope (margins/top/bottom).
    """
    blocks = _page_text_blocks(page)
    if not blocks:
        return None
    env = _union_rect(blocks)
    return env

def _intersects_any(r: fitz.Rect, others: List[fitz.Rect]) -> bool:
    return any(r.intersects(o) for o in others)

# ============================================================
# Callout text wrapping (prefers 12, then 11, then 10)
# ============================================================

def _optimize_layout_for_margin(text: str, box_width: float) -> Tuple[int, str, float, float]:
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
        if h <= 86.0 or fs == 10:
            return fs, "\n".join(lines), box_width, h

    return 10, text, box_width, 44.0

# ============================================================
# Callout placement: MARGINS/TOP/BOTTOM ONLY (never between lines)
# ============================================================

def _allowed_zones(page: fitz.Page) -> List[fitz.Rect]:
    """
    Returns placement zones that are strictly:
    - Left margin band (left of text envelope)
    - Right margin band (right of text envelope)
    - Top band (above text envelope)
    - Bottom band (below text envelope)
    If no text envelope, whole page is allowed (minus edge pads).
    """
    pr = page.rect
    env = _text_envelope(page)

    # default: whole page inset
    if env is None or env.get_area() <= 0:
        return [fitz.Rect(EDGE_PAD, EDGE_PAD, pr.width - EDGE_PAD, pr.height - EDGE_PAD)]

    # Expand the envelope slightly so we don't sit between lines / close to text
    env2 = inflate_rect(env, GAP_FROM_TEXT_BLOCKS)

    zones: List[fitz.Rect] = []

    # left margin: x < env2.x0
    if env2.x0 - EDGE_PAD > 60:
        zones.append(fitz.Rect(EDGE_PAD, EDGE_PAD, env2.x0, pr.height - EDGE_PAD))

    # right margin: x > env2.x1
    if (pr.width - EDGE_PAD) - env2.x1 > 60:
        zones.append(fitz.Rect(env2.x1, EDGE_PAD, pr.width - EDGE_PAD, pr.height - EDGE_PAD))

    # top band: y < env2.y0
    if env2.y0 - EDGE_PAD > 40:
        zones.append(fitz.Rect(EDGE_PAD, EDGE_PAD, pr.width - EDGE_PAD, env2.y0))

    # bottom band: y > env2.y1
    if (pr.height - EDGE_PAD) - env2.y1 > 40:
        zones.append(fitz.Rect(EDGE_PAD, env2.y1, pr.width - EDGE_PAD, pr.height - EDGE_PAD))

    # If everything is too tight, fall back to margins-only based on page edge
    if not zones:
        zones = [fitz.Rect(EDGE_PAD, EDGE_PAD, pr.width - EDGE_PAD, pr.height - EDGE_PAD)]

    return zones

def _choose_best_spot_margins_only(
    page: fitz.Page,
    targets: List[fitz.Rect],
    occupied: List[fitz.Rect],
    label: str,
) -> Tuple[fitz.Rect, str, int, bool]:
    pr = page.rect
    target_union = _union_rect(targets)
    target_center = _center(target_union)
    target_y = target_center.y

    blockers = _page_blockers(page, pad=GAP_FROM_TEXT_BLOCKS)
    for t in targets:
        blockers.append(inflate_rect(t, GAP_FROM_HIGHLIGHTS))
    occupied_buf = [inflate_rect(o, GAP_BETWEEN_CALLOUTS) for o in occupied]

    zones = _allowed_zones(page)

    # candidate widths differ: left/right margins are narrow; top/bottom can be wider
    def zone_box_width(z: fitz.Rect) -> float:
        return min(160.0, max(110.0, z.width - 8.0))

    # We’ll try a few Y positions around target_y and pick best (safe, then connector cost)
    y_offsets = [-70.0, -40.0, -20.0, 0.0, 20.0, 40.0, 70.0]

    candidates = []
    for z in zones:
        bw = zone_box_width(z)
        fs, wrapped, w, h = _optimize_layout_for_margin(label, bw)

        for dy in y_offsets:
            cy = target_y + dy
            y0 = cy - h / 2
            y1 = cy + h / 2

            # clamp within zone
            if y0 < z.y0:
                y1 += (z.y0 - y0)
                y0 = z.y0
            if y1 > z.y1:
                y0 -= (y1 - z.y1)
                y1 = z.y1
            if y0 < z.y0 or y1 > z.y1:
                continue

            # choose x: if zone is left margin, place against left edge; if right margin, against right edge; else center-ish
            if z.x1 <= (_text_envelope(page) or pr).x0:
                x0 = z.x0
                x1 = min(z.x1, x0 + w)
            elif z.x0 >= (_text_envelope(page) or pr).x1:
                x1 = z.x1
                x0 = max(z.x0, x1 - w)
            else:
                # top/bottom bands: keep near left/right depending on target
                if target_center.x < pr.width / 2:
                    x0 = z.x0
                    x1 = min(z.x1, x0 + w)
                else:
                    x1 = z.x1
                    x0 = max(z.x0, x1 - w)

            cand = fitz.Rect(x0, y0, x1, y1)

            safe = (not _intersects_any(cand, blockers)) and (not _intersects_any(cand, occupied_buf))

            # connector cost: fewer crossings then shortest
            # obstacles = text blocks + other targets + occupied
            obstacles: List[fitz.Rect] = []
            for b in _page_text_blocks(page):
                obstacles.append(inflate_rect(b, 1.5))
            for oc in occupied:
                obstacles.append(inflate_rect(oc, 2.0))
            for t in targets:
                obstacles.append(inflate_rect(t, 2.5))

            # approximate best straight-line (using target union)
            s, e = _straight_connector_best_pair(cand, target_union, obstacles)
            crossings = 0
            for ob in obstacles:
                if _segment_hits_rect(s, e, ob):
                    crossings += 1
            length = math.hypot(e.x - s.x, e.y - s.y)

            # score: must be safe first; then crossings; then length; then keep close to target_y
            score = (0 if safe else 1e9) + crossings * 5000.0 + length + abs((_center(cand).y) - target_y) * 0.8

            candidates.append((score, cand, wrapped, fs, safe))

    candidates.sort(key=lambda x: x[0])
    if not candidates:
        # emergency fallback
        fs, wrapped, w, h = _optimize_layout_for_margin(label, 140.0)
        fallback = fitz.Rect(EDGE_PAD, EDGE_PAD, EDGE_PAD + w, EDGE_PAD + h)
        return fallback, wrapped, fs, False

    _, cand, wrapped, fs, safe = candidates[0]
    return cand, wrapped, fs, safe

# ============================================================
# Search helpers (robust quote matching)
# ============================================================

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _search_term(page: fitz.Page, term: str) -> List[fitz.Rect]:
    t = (term or "").strip()
    if not t:
        return []
    if len(t) > _MAX_TERM:
        t = t[:_MAX_TERM]

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
        rects = page.search_for(t, flags=flags)
        if rects:
            return rects
    except Exception:
        pass

    t2 = _normalize_spaces(t)
    if t2 and t2 != t:
        try:
            rects = page.search_for(t2, flags=flags)
            if rects:
                return rects
        except Exception:
            pass

    if len(t2) >= _CHUNK:
        hits: List[fitz.Rect] = []
        step = max(10, _CHUNK - _CHUNK_OVERLAP)
        for i in range(0, len(t2), step):
            chunk = t2[i:i + _CHUNK].strip()
            if len(chunk) < 18:
                continue
            try:
                rs = page.search_for(chunk, flags=flags)
                hits.extend(rs)
            except Exception:
                continue

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

    # A) Quote highlights (all pages)
    for page in doc:
        for term in (quote_terms or []):
            rects = _search_term(page, term)
            for r in rects:
                page.draw_rect(r, color=RED, width=BOX_WIDTH)
                total_quote_hits += 1

    # B) Metadata callouts (page 1)
    def _do_job(
        label: str,
        value: Optional[str],
        *,
        connect_policy: str = "union",  # "single" | "union" | "all"
        also_try_variants: Optional[List[str]] = None,
    ):
        nonlocal total_meta_hits

        needles: List[str] = []
        if value and str(value).strip():
            needles.append(str(value).strip())
        if also_try_variants:
            for v in also_try_variants:
                vv = (v or "").strip()
                if vv:
                    needles.append(vv)
        needles = list(dict.fromkeys(needles))
        if not needles:
            return

        targets: List[fitz.Rect] = []
        for needle in needles:
            try:
                ts = page1.search_for(needle)
            except Exception:
                ts = []
            if ts:
                targets.extend(ts)

        if not targets:
            return

        # Box targets
        for t in targets:
            page1.draw_rect(t, color=RED, width=BOX_WIDTH)
        total_meta_hits += len(targets)

        # Place callout in margins/top/bottom ONLY
        callout_rect, wrapped_text, fs, safe = _choose_best_spot_margins_only(
            page1, targets, occupied_callouts, label
        )

        # White background only if safe (won’t cover text in allowed zones anyway, but keep the logic)
        if safe:
            page1.draw_rect(callout_rect, color=WHITE, fill=WHITE, overlay=True)

        page1.insert_textbox(
            callout_rect,
            wrapped_text,
            fontname=FONTNAME,
            fontsize=fs,
            color=RED,
            align=fitz.TEXT_ALIGN_LEFT,
        )

        # Obstacles for connector scoring (we still draw one straight line)
        obstacles: List[fitz.Rect] = []
        for b in _page_text_blocks(page1):
            obstacles.append(inflate_rect(b, 1.5))
        for oc in occupied_callouts:
            obstacles.append(inflate_rect(oc, 2.0))
        # treat all target boxes as obstacles (except the one we connect to is okay; we pull back anyway)
        expanded_all = [inflate_rect(t, 2.5) for t in targets]

        def connect_to(rect: fitz.Rect):
            obs = obstacles[:]
            for ot in expanded_all:
                if not ot.intersects(rect):
                    obs.append(ot)
            _draw_straight_connector(page1, callout_rect, rect, obs)

        if connect_policy == "all":
            for t in targets:
                connect_to(t)
        elif connect_policy == "single":
            connect_to(targets[0])
        else:
            connect_to(_union_rect(targets))

        occupied_callouts.append(callout_rect)

    _do_job("Original source of publication.", meta.get("source_url"), connect_policy="union")
    _do_job("The distinguished organization.", meta.get("venue_name") or meta.get("org_name"), connect_policy="union")
    _do_job("Performance date.", meta.get("performance_date"), connect_policy="union")
    _do_job("Beneficiary salary evidence.", meta.get("salary_amount"), connect_policy="union")
    _do_job(
        "Beneficiary lead role evidence.",
        meta.get("beneficiary_name"),
        connect_policy="all",
        also_try_variants=meta.get("beneficiary_variants") or [],
    )

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    out.seek(0)

    return out.getvalue(), {
        "total_quote_hits": total_quote_hits,
        "total_meta_hits": total_meta_hits,
        "criterion_id": criterion_id,
    }
