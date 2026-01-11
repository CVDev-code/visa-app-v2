import io
import math
import re
from typing import Dict, List, Tuple, Optional

import fitz  # PyMuPDF

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# ---- style knobs ----
BOX_WIDTH = 1.7          # thickness for red rectangles
LINE_WIDTH = 1.6         # thickness for connectors
FONTNAME = "Times-Bold"  # thicker than Times-Roman
FONT_SIZES = [12, 11, 10]

# ---- spacing knobs ----
EDGE_PAD = 18.0
GAP_FROM_TEXT_BLOCKS = 10.0
GAP_FROM_HIGHLIGHTS = 14.0
GAP_BETWEEN_CALLOUTS = 10.0
ENDPOINT_PULLBACK = 1.5   # pull line end slightly away from box interior

# For quote search robustness
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
    """Sample points along segment; good enough for routing decisions."""
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
    """Edge points for sampling endpoints (midpoints + slight corner bias)."""
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
    """
    ALWAYS returns a straight segment (start,end) that minimizes crossings.
    We sample multiple start points along the callout side and multiple end points on target edge.
    """
    target_center = _center(target_rect)

    # sample a few start points around the mid-height to avoid cutting through adjacent boxes
    base = _mid_edge_anchor(callout_rect, target_center)
    # allow small vertical nudges (still on edge)
    nudges = [-14.0, -7.0, 0.0, 7.0, 14.0]
    starts: List[fitz.Point] = []
    for dy in nudges:
        y = min(max(callout_rect.y0 + 2.0, base.y + dy), callout_rect.y1 - 2.0)
        starts.append(fitz.Point(base.x, y))

    ends = _edge_candidates(target_rect)

    best = (10**9, 10**9, starts[0], ends[0])  # (hits, length, start, end)
    for s in starts:
        for e in ends:
            # Count obstacle crossings
            hits = 0
            for ob in obstacles:
                if _segment_hits_rect(s, e, ob):
                    hits += 1

            length = math.hypot(e.x - s.x, e.y - s.y)

            # primary: minimize hits, secondary: shortest line
            if hits < best[0] or (hits == best[0] and length < best[1]):
                best = (hits, length, s, e)

    s = best[2]
    e = best[3]
    e = _pull_back_point(s, e, ENDPOINT_PULLBACK)
    return s, e


def _draw_straight_connector(
    page: fitz.Page,
    callout_rect: fitz.Rect,
    target_rect: fitz.Rect,
    obstacles: List[fitz.Rect],
):
    s, e = _straight_connector_best_pair(callout_rect, target_rect, obstacles)
    page.draw_line(s, e, color=RED, width=LINE_WIDTH)


# ============================================================
# Page blockers (for callout placement)
# ============================================================

def _page_blockers(page: fitz.Page, pad: float = GAP_FROM_TEXT_BLOCKS) -> List[fitz.Rect]:
    blockers: List[fitz.Rect] = []

    # Text blocks
    try:
        for b in page.get_text("blocks"):
            blockers.append(inflate_rect(fitz.Rect(b[:4]), pad))
    except Exception:
        pass

    # Images
    try:
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                for r in page.get_image_rects(xref):
                    blockers.append(inflate_rect(fitz.Rect(r), pad))
            except Exception:
                pass
    except Exception:
        pass

    # Vector drawings
    try:
        for d in page.get_drawings():
            r = d.get("rect")
            if r:
                blockers.append(inflate_rect(fitz.Rect(r), pad))
    except Exception:
        pass

    return blockers


def _intersects_any(r: fitz.Rect, others: List[fitz.Rect]) -> bool:
    return any(r.intersects(o) for o in others)


# ============================================================
# Callout text wrapping (prefers 12, then 11, then 10)
# ============================================================

def _optimize_layout_for_margin(text: str, box_width: float) -> Tuple[int, str, float, float]:
    """
    Returns (fontsize, wrapped_text, width, height). Prefers 12 unless too tall.
    """
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
        # If we can keep this reasonably compact, accept this fs
        if h <= 86.0 or fs == 10:
            return fs, "\n".join(lines), box_width, h

    return 10, text, box_width, 44.0


def _choose_best_margin_spot(
    page: fitz.Page,
    targets: List[fitz.Rect],
    occupied: List[fitz.Rect],
    label: str,
) -> Tuple[fitz.Rect, str, int, bool]:
    pr = page.rect
    target_union = _union_rect(targets)
    target_y = (target_union.y0 + target_union.y1) / 2

    margin_w = 130.0
    left_x = EDGE_PAD
    right_x = pr.width - EDGE_PAD - margin_w

    blockers = _page_blockers(page, pad=GAP_FROM_TEXT_BLOCKS)
    for t in targets:
        blockers.append(inflate_rect(t, GAP_FROM_HIGHLIGHTS))
    occupied_buf = [inflate_rect(o, GAP_BETWEEN_CALLOUTS) for o in occupied]

    def clamp(r: fitz.Rect) -> fitz.Rect:
        rr = fitz.Rect(r)
        if rr.y0 < EDGE_PAD:
            rr.y1 += (EDGE_PAD - rr.y0)
            rr.y0 = EDGE_PAD
        if rr.y1 > pr.height - EDGE_PAD:
            rr.y0 -= (rr.y1 - (pr.height - EDGE_PAD))
            rr.y1 = pr.height - EDGE_PAD
        return rr

    candidates = []
    for x_start in [left_x, right_x]:
        fs, wrapped, w, h = _optimize_layout_for_margin(label, margin_w)
        cand = fitz.Rect(x_start, target_y - h / 2, x_start + w, target_y + h / 2)
        cand = clamp(cand)

        safe = (not _intersects_any(cand, blockers)) and (not _intersects_any(cand, occupied_buf))
        score = abs(target_y - (cand.y0 + cand.y1) / 2) + (0 if safe else 1e9)
        candidates.append((score, cand, wrapped, fs, safe))

    candidates.sort(key=lambda x: x[0])
    score, cand, wrapped, fs, safe = candidates[0]

    # If not safe, don't paint white background (still place text)
    if score >= 1e9:
        safe = False

    return cand, wrapped, fs, safe


# ============================================================
# Search helpers (robust quote matching)
# ============================================================

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _search_term(page: fitz.Page, term: str) -> List[fitz.Rect]:
    """
    Robust search:
    1) exact search (with dehyphenate)
    2) whitespace-normalized search
    3) chunk fallback for long phrases (line breaks/hyphenation)
    """
    t = (term or "").strip()
    if not t:
        return []

    # cap extreme length
    if len(t) > _MAX_TERM:
        t = t[:_MAX_TERM]

    flags = 0
    # some PyMuPDF builds provide these
    try:
        flags |= fitz.TEXT_DEHYPHENATE
    except Exception:
        pass
    try:
        flags |= fitz.TEXT_PRESERVE_WHITESPACE
    except Exception:
        pass

    # pass 1: exact
    try:
        rects = page.search_for(t, flags=flags)
        if rects:
            return rects
    except Exception:
        pass

    # pass 2: normalized spaces
    t2 = _normalize_spaces(t)
    if t2 and t2 != t:
        try:
            rects = page.search_for(t2, flags=flags)
            if rects:
                return rects
        except Exception:
            pass

    # pass 3: chunk search for long phrases
    if len(t2) >= _CHUNK:
        hits: List[fitz.Rect] = []
        step = max(10, _CHUNK - _CHUNK_OVERLAP)
        for i in range(0, len(t2), step):
            chunk = t2[i:i + _CHUNK].strip()
            if len(chunk) < 18:
                continue
            try:
                rs = page.search_for(chunk, flags=flags)
                for r in rs:
                    hits.append(r)
            except Exception:
                continue

        # de-dup rectangles
        if hits:
            # merge close/overlapping
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
# Main annotation entrypoint
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

    # -----------------------------
    # A) Quote highlights (all pages)
    # -----------------------------
    for page in doc:
        for term in (quote_terms or []):
            rects = _search_term(page, term)
            for r in rects:
                page.draw_rect(r, color=RED, width=BOX_WIDTH)
                total_quote_hits += 1

    # -----------------------------
    # B) Metadata callouts (page 1)
    # Always attempt whatever values we receive in meta.
    # -----------------------------
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

        needles = list(dict.fromkeys(needles))  # de-dupe
        if not needles:
            return

        # Find targets on page 1 (try all needles)
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

        # Box all target hits
        for t in targets:
            page1.draw_rect(t, color=RED, width=BOX_WIDTH)
        total_meta_hits += len(targets)

        # Place callout
        callout_rect, wrapped_text, fs, safe = _choose_best_margin_spot(page1, targets, occupied_callouts, label)

        # White background only if safe
        if safe:
            page1.draw_rect(callout_rect, color=WHITE, fill=WHITE, overlay=True)

        # Thicker-looking text: Times-Bold and slightly larger leading in wrap already handled
        page1.insert_textbox(
            callout_rect,
            wrapped_text,
            fontname=FONTNAME,
            fontsize=fs,
            color=RED,
            align=fitz.TEXT_ALIGN_LEFT,
        )

        # Obstacles for “avoid crossing” scoring: other target boxes + existing text blocks
        obstacles: List[fitz.Rect] = []
        try:
            for b in page1.get_text("blocks"):
                obstacles.append(inflate_rect(fitz.Rect(b[:4]), 1.5))
        except Exception:
            pass

        # treat ALL highlighted targets as obstacles EXCEPT the one we're connecting to.
        expanded_all = [inflate_rect(t, 2.5) for t in targets]

        def connect_to(rect: fitz.Rect):
            obs = obstacles[:]
            for ot in expanded_all:
                # if it's not basically the same rect, keep it as obstacle
                if not (ot.intersects(rect) and (ot | rect).get_area() < (ot.get_area() + rect.get_area() + 3.0)):
                    obs.append(ot)
            for oc in occupied_callouts:
                obs.append(inflate_rect(oc, 2.0))

            _draw_straight_connector(page1, callout_rect, rect, obs)

        if connect_policy == "all":
            for t in targets:
                connect_to(t)
        elif connect_policy == "single":
            connect_to(targets[0])
        else:
            connect_to(_union_rect(targets))

        occupied_callouts.append(callout_rect)

    # Always do URL if present
    _do_job("Original source of publication.", meta.get("source_url"), connect_policy="union")

    # Venue / org name
    _do_job("The distinguished organization.", meta.get("venue_name") or meta.get("org_name"), connect_policy="union")

    # Date
    _do_job("Performance date.", meta.get("performance_date"), connect_policy="union")

    # Salary
    _do_job("Beneficiary salary evidence.", meta.get("salary_amount"), connect_policy="union")

    # Beneficiary name (try variants too) – connect to all hits (straight lines)
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
