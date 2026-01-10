# src/pdf_highlighter.py
import io
import math
import re
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# ============================================================
# Visual style (match your "good" version)
# ============================================================
BOX_WIDTH = 1.5      # highlight rectangle stroke
LINE_WIDTH = 1.0     # connector stroke (thin)
FONTNAME = "Times-Roman"
FONT_SIZES = [12, 11, 10]

# ============================================================
# Placement knobs
# ============================================================
EDGE_PAD = 18.0
GAP_FROM_TEXT_BLOCKS = 10.0
GAP_FROM_IMAGES = 12.0
GAP_FROM_DRAWINGS = 10.0

# Keep callouts away from highlight boxes and each other
GAP_FROM_HIGHLIGHTS = 14.0
GAP_BETWEEN_CALLOUTS = 10.0

# Prevent “between lines” placement by hard-blocking the entire text column
TEXT_COLUMN_BUFFER_X = 14.0       # expand envelope left/right
SIDE_ZONE_GAP_FROM_TEXT = 10.0    # gutter starts this far from envelope

# Callout sizing defaults
MIN_CALLOUT_W = 110.0
MAX_CALLOUT_W = 160.0  # keep margin-ish
MAX_CALLOUT_H = 130.0

# Search sampling for candidate spots
Y_SCAN_STEP = 14.0
Y_SCAN_SPAN = 260.0   # scan around target_y +/- span

# Prefer left margin when both sides are equally good
PREFER_LEFT = True

# Pull connector endpoint slightly away from target edge
ENDPOINT_PULLBACK = 1.5

# ============================================================
# Ink check (relaxed in gutters, to ignore scanner noise)
# ============================================================
INKCHECK_DPI = 96
INKCHECK_PAD = 1.5
INK_SAMPLE_STRIDE = 6
INK_DELTA = 18
INK_RATIO_MAX = 0.012
INK_RATIO_MAX_GUTTER = 0.030
INK_DELTA_GUTTER_BONUS = 12

# ============================================================
# Quote search robustness
# ============================================================
_MAX_TERM = 600
_CHUNK = 70
_CHUNK_OVERLAP = 22


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

def _mid_height_anchor(callout: fitz.Rect, toward: fitz.Point) -> fitz.Point:
    """Start point on callout edge at mid-height."""
    y = callout.y0 + (callout.height / 2.0)
    cx = callout.x0 + callout.width / 2.0
    if toward.x >= cx:
        return fitz.Point(callout.x1, y)
    return fitz.Point(callout.x0, y)

def _target_edge_candidates(target: fitz.Rect) -> List[fitz.Point]:
    cx = target.x0 + target.width / 2.0
    cy = target.y0 + target.height / 2.0
    return [
        fitz.Point(target.x0, cy),
        fitz.Point(target.x1, cy),
        fitz.Point(cx, target.y0),
        fitz.Point(cx, target.y1),
        fitz.Point(target.x0, target.y0),
        fitz.Point(target.x1, target.y0),
        fitz.Point(target.x0, target.y1),
        fitz.Point(target.x1, target.y1),
    ]

def _choose_target_attachment(
    start: fitz.Point,
    target: fitz.Rect,
    obstacles: List[fitz.Rect],
) -> fitz.Point:
    """Pick target edge point minimizing obstacle crossings then length."""
    best_pt = _center(target)
    best_hits = 10**9
    best_len = 10**9
    for pt in _target_edge_candidates(target):
        hits = 0
        for ob in obstacles:
            if _segment_hits_rect(start, pt, ob):
                hits += 1
        seg_len = math.hypot(pt.x - start.x, pt.y - start.y)
        if hits < best_hits or (hits == best_hits and seg_len < best_len):
            best_hits, best_len = hits, seg_len
            best_pt = pt
    return best_pt

def _draw_straight_connector(
    page: fitz.Page,
    callout_rect: fitz.Rect,
    target_rect: fitz.Rect,
    obstacles: List[fitz.Rect],
):
    """Always straight line, but choose best endpoint to reduce crossings."""
    tc = _center(target_rect)
    start = _mid_height_anchor(callout_rect, tc)
    end = _choose_target_attachment(start, target_rect, obstacles)
    end = _pull_back_point(start, end, ENDPOINT_PULLBACK)
    page.draw_line(start, end, color=RED, width=LINE_WIDTH)


# ============================================================
# Ink detection (relaxed for gutters)
# ============================================================

def _rect_has_ink_adaptive(page: fitz.Page, rect: fitz.Rect, *, is_gutter: bool) -> bool:
    r = (fitz.Rect(rect) & page.rect)
    if r.is_empty or r.width < 2 or r.height < 2:
        return True

    pix = page.get_pixmap(matrix=fitz.Matrix(INKCHECK_DPI / 72, INKCHECK_DPI / 72), clip=r, alpha=False)
    s, w, h = pix.samples, pix.width, pix.height
    if not s or w < 2 or h < 2:
        return True

    ratio_max = INK_RATIO_MAX_GUTTER if is_gutter else INK_RATIO_MAX
    delta = INK_DELTA + (INK_DELTA_GUTTER_BONUS if is_gutter else 0)
    step = max(1, INK_SAMPLE_STRIDE)

    total = 0
    sr = sg = sb = 0
    for y in range(0, h, step):
        row = y * w * 3
        for x in range(0, w, step):
            i = row + x * 3
            sr += s[i]; sg += s[i + 1]; sb += s[i + 2]
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
# Blockers + envelope
# ============================================================

def _page_text_blocks(page: fitz.Page) -> List[fitz.Rect]:
    out: List[fitz.Rect] = []
    try:
        for b in page.get_text("blocks"):
            out.append(fitz.Rect(b[:4]))
    except Exception:
        pass
    return out

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

def _text_envelope(page: fitz.Page) -> Optional[fitz.Rect]:
    blocks = _page_text_blocks(page)
    if not blocks:
        return None
    return _union_rect(blocks)

def _page_blockers(page: fitz.Page) -> List[fitz.Rect]:
    blockers: List[fitz.Rect] = []
    for r in _page_text_blocks(page):
        blockers.append(inflate_rect(r, GAP_FROM_TEXT_BLOCKS))
    for r in _page_images(page):
        blockers.append(inflate_rect(r, GAP_FROM_IMAGES))
    for r in _page_drawings(page):
        blockers.append(inflate_rect(r, GAP_FROM_DRAWINGS))
    return blockers

def _intersects_any(r: fitz.Rect, others: List[fitz.Rect]) -> bool:
    return any(r.intersects(o) for o in others)

def _clamp_to_safe(r: fitz.Rect, safe: fitz.Rect) -> Optional[fitz.Rect]:
    rr = fitz.Rect(r)
    if rr.x0 < safe.x0:
        rr.x1 += (safe.x0 - rr.x0); rr.x0 = safe.x0
    if rr.x1 > safe.x1:
        rr.x0 -= (rr.x1 - safe.x1); rr.x1 = safe.x1
    if rr.y0 < safe.y0:
        rr.y1 += (safe.y0 - rr.y0); rr.y0 = safe.y0
    if rr.y1 > safe.y1:
        rr.y0 -= (rr.y1 - safe.y1); rr.y1 = safe.y1
    if rr.x0 < safe.x0 or rr.x1 > safe.x1 or rr.y0 < safe.y0 or rr.y1 > safe.y1:
        return None
    if rr.width < 20 or rr.height < 18:
        return None
    return rr


# ============================================================
# Callout text wrapping
# ============================================================

def _optimize_layout(text: str, box_width: float) -> Tuple[int, str, float, float]:
    text = (text or "").strip()
    if not text:
        return 12, "", box_width, 24.0

    box_width = max(MIN_CALLOUT_W, min(MAX_CALLOUT_W, box_width))
    best = (10, text, box_width, 40.0)
    best_h = 10**9

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

        h = (len(lines) * fs * 1.2) + 10.0
        if h < best_h:
            best_h = h
            best = (fs, "\n".join(lines), box_width, h)

    fs, wrapped, w, h = best
    h = min(h, MAX_CALLOUT_H)
    return fs, wrapped, w, h


# ============================================================
# Placement: side gutters derived from envelope, hard-block text column
# ============================================================

def _choose_best_side_spot(
    page: fitz.Page,
    targets: List[fitz.Rect],
    occupied: List[fitz.Rect],
    label: str,
) -> Tuple[fitz.Rect, str, int, bool]:
    pr = page.rect
    safe = fitz.Rect(EDGE_PAD, EDGE_PAD, pr.width - EDGE_PAD, pr.height - EDGE_PAD)

    target_union = _union_rect(targets)
    ty = (target_union.y0 + target_union.y1) / 2.0

    blockers = _page_blockers(page)
    # keep away from highlight targets
    highlight_blockers = [inflate_rect(t, GAP_FROM_HIGHLIGHTS) for t in targets]
    blockers.extend(highlight_blockers)
    # keep away from other callouts
    occupied_buf = [inflate_rect(o, GAP_BETWEEN_CALLOUTS) for o in occupied]

    # Hard block the entire text column (prevents “between lines”)
    env = _text_envelope(page)
    if env:
        x0 = max(pr.x0, env.x0 - TEXT_COLUMN_BUFFER_X)
        x1 = min(pr.x1, env.x1 + TEXT_COLUMN_BUFFER_X)
        if x1 > x0 + 10:
            blockers.append(fitz.Rect(x0, pr.y0, x1, pr.y1))

        # Build real gutters relative to envelope
        left_zone = fitz.Rect(safe.x0, safe.y0, env.x0 - SIDE_ZONE_GAP_FROM_TEXT, safe.y1)
        right_zone = fitz.Rect(env.x1 + SIDE_ZONE_GAP_FROM_TEXT, safe.y0, safe.x1, safe.y1)
    else:
        # No text detected: use generic gutters
        left_zone = fitz.Rect(safe.x0, safe.y0, safe.x0 + 140.0, safe.y1)
        right_zone = fitz.Rect(safe.x1 - 140.0, safe.y0, safe.x1, safe.y1)

    # Validate zones
    zones: List[Tuple[str, fitz.Rect]] = []
    if left_zone.width >= 55 and left_zone.height >= 80:
        zones.append(("left", left_zone))
    if right_zone.width >= 55 and right_zone.height >= 80:
        zones.append(("right", right_zone))

    # If neither gutter is usable, fall back to top band (still non-overlapping)
    if not zones:
        top = fitz.Rect(safe.x0, safe.y0, safe.x1, min(safe.y1, safe.y0 + 140.0))
        zones = [("top", top)]

    # Candidate generation: scan Y positions around the target y within each zone
    candidates: List[Tuple[float, fitz.Rect, str, int, bool]] = []

    for zn, z in zones:
        # width is the zone width (capped), so we don't end up huge boxes
        bw_guess = min(MAX_CALLOUT_W, max(MIN_CALLOUT_W, z.width - 6.0))
        fs, wrapped, w, h = _optimize_layout(label, bw_guess)

        # anchor x by zone
        if zn == "left":
            x0 = z.x0
            x1 = x0 + w
        elif zn == "right":
            x1 = z.x1
            x0 = x1 - w
        else:
            # top fallback: place near left or right depending on preference
            if PREFER_LEFT:
                x0 = z.x0
                x1 = x0 + w
            else:
                x1 = z.x1
                x0 = x1 - w

        # scan y positions
        for dy in _frange(-Y_SCAN_SPAN, Y_SCAN_SPAN, Y_SCAN_STEP):
            cy = ty + dy
            cand = fitz.Rect(x0, cy - h / 2, x1, cy + h / 2)
            cand = _clamp_to_safe(cand, safe)
            if not cand:
                continue

            # Hard constraints: no overlap with blockers or occupied
            if _intersects_any(cand, blockers):
                continue
            if _intersects_any(cand, occupied_buf):
                continue

            # Only draw white bg if the region looks clean enough
            is_gutter = (zn in ("left", "right"))
            has_ink = _rect_has_ink_adaptive(page, inflate_rect(cand, INKCHECK_PAD), is_gutter=is_gutter)
            safe_for_white = (not has_ink)

            # Score: prefer being close in Y, and prefer left/right strongly over top fallback
            score = abs(dy)
            if zn == "left":
                score -= 2500.0 if PREFER_LEFT else 2000.0
            elif zn == "right":
                score -= 2500.0 if not PREFER_LEFT else 2000.0
            else:
                score += 4000.0  # top is truly last resort

            candidates.append((score, cand, wrapped, fs, safe_for_white))

    candidates.sort(key=lambda x: x[0])
    if not candidates:
        # Absolute fallback: packed placement in safe area, still non-overlapping
        fs, wrapped, w, h = _optimize_layout(label, 140.0)
        step = 26.0
        for y in _frange(safe.y0, safe.y1 - h, step):
            for x in _frange(safe.x0, safe.x1 - w, step):
                cand = fitz.Rect(x, y, x + w, y + h)
                if _intersects_any(cand, blockers):
                    continue
                if _intersects_any(cand, occupied_buf):
                    continue
                return cand, wrapped, fs, False

        fb = fitz.Rect(safe.x0, safe.y0, safe.x0 + w, safe.y0 + h)
        return fb, wrapped, fs, False

    _, best_rect, wrapped, fs, safe_for_white = candidates[0]
    return best_rect, wrapped, fs, safe_for_white


def _frange(a: float, b: float, step: float):
    x = a
    if step == 0:
        return
    if a <= b:
        while x <= b + 1e-9:
            yield x
            x += step
    else:
        while x >= b - 1e-9:
            yield x
            x -= step


# ============================================================
# Search helpers (robust)
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

        callout_rect, wrapped_text, fs, safe_for_white_bg = _choose_best_side_spot(
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

        # Obstacles for straight connector endpoint choice
        obstacles: List[fitz.Rect] = []
        for b in _page_text_blocks(page1):
            obstacles.append(inflate_rect(b, 2.0))
        for o in occupied_callouts:
            obstacles.append(inflate_rect(o, 2.0))
        # treat other target boxes as obstacles (expanded slightly)
        for t in targets:
            obstacles.append(inflate_rect(t, 2.0))

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

    # Beneficiary name + variants (connect to all matches)
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
