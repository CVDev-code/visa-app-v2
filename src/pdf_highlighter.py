# src/pdf_highlighter.py
import io
import math
import os
import re
import json
from typing import Dict, List, Tuple, Optional, Any

import fitz  # PyMuPDF

# If you deploy on Streamlit Cloud, install openai in requirements.txt
# pip install openai
from openai import OpenAI

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# ============================================================
# Style knobs (keep close to your preferred look)
# ============================================================
BOX_WIDTH = 1.5
LINE_WIDTH = 1.0
FONTNAME = "Times-Roman"
FONT_SIZES = [12, 11, 10]

EDGE_PAD = 18.0

# ============================================================
# Hard safety gaps
# ============================================================
GAP_FROM_TEXT_BLOCKS = 10.0
GAP_FROM_IMAGES = 12.0
GAP_FROM_DRAWINGS = 10.0

GAP_FROM_HIGHLIGHTS = 14.0
GAP_BETWEEN_CALLOUTS = 10.0

# Hard-block the whole text column to stop "between lines"
TEXT_COLUMN_BUFFER_X = 16.0

# Connector cosmetics
ENDPOINT_PULLBACK = 1.5

# Candidate generation
GRID_STEP_Y = 14.0
GRID_STEP_X = 14.0
MAX_CANDIDATES_TO_LLM = 60

# Prefer side gutters strongly; top should be last resort.
ZONE_ORDER = ["left", "right", "bottom", "top"]

# ============================================================
# Quote search robustness
# ============================================================
_MAX_TERM = 600
_CHUNK = 70
_CHUNK_OVERLAP = 22

# ============================================================
# LLM config
# ============================================================
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")  # pick what you have access to
LLM_TEMPERATURE = 0


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

def _intersects_any(r: fitz.Rect, others: List[fitz.Rect]) -> bool:
    return any(r.intersects(o) for o in others)

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

def _choose_target_attachment(start: fitz.Point, target: fitz.Rect, obstacles: List[fitz.Rect]) -> fitz.Point:
    best_pt = _center(target)
    best_hits = 10**9
    best_len = 10**9
    for pt in _target_edge_candidates(target):
        hits = sum(1 for ob in obstacles if _segment_hits_rect(start, pt, ob))
        seg_len = math.hypot(pt.x - start.x, pt.y - start.y)
        if hits < best_hits or (hits == best_hits and seg_len < best_len):
            best_hits, best_len, best_pt = hits, seg_len, pt
    return best_pt

def _draw_straight_connector(page: fitz.Page, callout_rect: fitz.Rect, target_rect: fitz.Rect, obstacles: List[fitz.Rect]):
    tc = _center(target_rect)
    start = _mid_height_anchor(callout_rect, tc)
    end = _choose_target_attachment(start, target_rect, obstacles)
    end = _pull_back_point(start, end, ENDPOINT_PULLBACK)
    page.draw_line(start, end, color=RED, width=LINE_WIDTH)


# ============================================================
# Page blockers + envelope
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


# ============================================================
# Callout text layout
# ============================================================

def _optimize_layout(text: str, box_width: float) -> Tuple[int, str, float, float]:
    text = (text or "").strip()
    if not text:
        return 12, "", box_width, 24.0

    best_fs = 10
    best_wrapped = text
    best_w = box_width
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
            best_fs = fs
            best_wrapped = "\n".join(lines)
            best_w = box_width

    return best_fs, best_wrapped, best_w, best_h


# ============================================================
# Zones: compute gutters from envelope and produce candidate rects
# ============================================================

def _build_zones(page: fitz.Page) -> Dict[str, fitz.Rect]:
    pr = page.rect
    safe = fitz.Rect(EDGE_PAD, EDGE_PAD, pr.width - EDGE_PAD, pr.height - EDGE_PAD)

    env = _text_envelope(page)
    zones: Dict[str, fitz.Rect] = {}

    if env:
        left = fitz.Rect(safe.x0, safe.y0, env.x0 - 10.0, safe.y1)
        right = fitz.Rect(env.x1 + 10.0, safe.y0, safe.x1, safe.y1)
        top = fitz.Rect(safe.x0, safe.y0, safe.x1, env.y0 - 10.0)
        bottom = fitz.Rect(safe.x0, env.y1 + 10.0, safe.x1, safe.y1)
    else:
        left = fitz.Rect(safe.x0, safe.y0, safe.x0 + 140.0, safe.y1)
        right = fitz.Rect(safe.x1 - 140.0, safe.y0, safe.x1, safe.y1)
        top = fitz.Rect(safe.x0, safe.y0, safe.x1, safe.y0 + 140.0)
        bottom = fitz.Rect(safe.x0, safe.y1 - 140.0, safe.x1, safe.y1)

    # keep only non-trivial zones
    if left.width > 55 and left.height > 80:
        zones["left"] = left & safe
    if right.width > 55 and right.height > 80:
        zones["right"] = right & safe
    if bottom.height > 45:
        zones["bottom"] = bottom & safe
    if top.height > 45:
        zones["top"] = top & safe

    return zones

def _grid_points(zone: fitz.Rect) -> List[fitz.Point]:
    pts: List[fitz.Point] = []
    y = zone.y0
    while y <= zone.y1:
        x = zone.x0
        while x <= zone.x1:
            pts.append(fitz.Point(x, y))
            x += GRID_STEP_X
        y += GRID_STEP_Y
    return pts


# ============================================================
# Candidate generation (Python hard constraints)
# ============================================================

def _estimate_connector_metrics(
    callout_rect: fitz.Rect,
    target_rect: fitz.Rect,
    obstacles: List[fitz.Rect],
) -> Tuple[float, int]:
    tc = _center(target_rect)
    start = _mid_height_anchor(callout_rect, tc)
    end = _choose_target_attachment(start, target_rect, obstacles)
    length = math.hypot(end.x - start.x, end.y - start.y)
    hits = sum(1 for ob in obstacles if _segment_hits_rect(start, end, ob))
    return length, hits

def _generate_candidates(
    page: fitz.Page,
    target_union: fitz.Rect,
    label: str,
    occupied_callouts: List[fitz.Rect],
    highlight_blockers: List[fitz.Rect],
) -> List[Dict[str, Any]]:
    pr = page.rect
    safe = fitz.Rect(EDGE_PAD, EDGE_PAD, pr.width - EDGE_PAD, pr.height - EDGE_PAD)

    blockers = _page_blockers(page)

    # Hard-block full text column
    env = _text_envelope(page)
    if env:
        x0 = max(pr.x0, env.x0 - TEXT_COLUMN_BUFFER_X)
        x1 = min(pr.x1, env.x1 + TEXT_COLUMN_BUFFER_X)
        if x1 > x0 + 10:
            blockers.append(fitz.Rect(x0, pr.y0, x1, pr.y1))

    # Also block highlights and existing callouts
    blockers.extend(highlight_blockers)
    blockers.extend([inflate_rect(o, GAP_BETWEEN_CALLOUTS) for o in occupied_callouts])

    zones = _build_zones(page)

    # size the callout to fit the zone width (but not huge)
    def callout_dims_for_zone(z: fitz.Rect) -> Tuple[int, str, float, float]:
        bw = min(MAX_CALLOUT_W, max(MIN_CALLOUT_W, z.width - 8.0))
        fs, wrapped, w, h = _optimize_layout(label, bw)
        return fs, wrapped, w, h

    # connector obstacles: text blocks + existing callouts + highlight blockers
    conn_obstacles: List[fitz.Rect] = []
    for b in _page_text_blocks(page):
        conn_obstacles.append(inflate_rect(b, 2.0))
    for o in occupied_callouts:
        conn_obstacles.append(inflate_rect(o, 2.0))
    for hb in highlight_blockers:
        conn_obstacles.append(inflate_rect(hb, 2.0))

    candidates: List[Dict[str, Any]] = []
    tid = 1
    tc = _center(target_union)

    for zone_name in ZONE_ORDER:
        if zone_name not in zones:
            continue
        z = zones[zone_name]
        fs, wrapped, w, h = callout_dims_for_zone(z)

        pts = _grid_points(z)
        # Sort by "close to target Y" first
        pts.sort(key=lambda p: abs(p.y - tc.y))

        for p in pts:
            # Anchor callout inside the zone
            if zone_name == "left":
                x0 = z.x0
                x1 = x0 + w
            elif zone_name == "right":
                x1 = z.x1
                x0 = x1 - w
            elif zone_name == "bottom":
                # bottom: align left so it doesn't overlay, but model can still choose bottom if best
                x0 = z.x0
                x1 = x0 + w
            else:  # top
                x0 = z.x0
                x1 = x0 + w

            cand = fitz.Rect(x0, p.y - h / 2, x1, p.y + h / 2)
            cand = cand & safe
            if cand.is_empty or cand.width < 30 or cand.height < 18:
                continue

            # Hard constraints
            if _intersects_any(cand, blockers):
                continue

            # Prefer not to be too far from target vertically (still allow URL to go bottom)
            dist_y = abs((_center(cand).y) - tc.y)

            # Connector metrics
            conn_len, conn_hits = _estimate_connector_metrics(cand, target_union, conn_obstacles)

            candidates.append({
                "id": tid,
                "zone": zone_name,
                "rect": {"x0": cand.x0, "y0": cand.y0, "x1": cand.x1, "y1": cand.y1},
                "fontsize": fs,
                "wrapped_text": wrapped,
                "dist_y": dist_y,
                "connector_len": conn_len,
                "connector_crossings_est": conn_hits,
            })
            tid += 1

            # keep candidate list bounded per zone
            if len(candidates) >= 400:
                break
        if len(candidates) >= 400:
            break

    return candidates


# ============================================================
# LLM ranking: choose best candidate ID
# Uses Structured Outputs / json_schema so parsing is reliable.
# ============================================================

def _llm_choose_candidate(
    *,
    label: str,
    target_union: fitz.Rect,
    candidates: List[Dict[str, Any]],
) -> int:
    if not candidates:
        return -1

    client = OpenAI()  # uses OPENAI_API_KEY from env automatically

    # send only top N by a crude heuristic before asking the LLM
    # (LLM should decide, but this keeps payload small)
    def pre_score(c: Dict[str, Any]) -> float:
        # prefer left/right, small connector, small dist_y, fewer crossings
        zone = c["zone"]
        zone_bonus = 0.0
        if zone == "left":
            zone_bonus = -9000.0
        elif zone == "right":
            zone_bonus = -8000.0
        elif zone == "bottom":
            zone_bonus = -2000.0
        else:
            zone_bonus = 3000.0
        # Special case: URL/source label tends to bottom
        if "source of publication" in label.lower() and zone == "bottom":
            zone_bonus -= 4000.0
        return (
            zone_bonus
            + c["dist_y"] * 1.1
            + c["connector_len"] * 0.6
            + c["connector_crossings_est"] * 2500.0
        )

    candidates_sorted = sorted(candidates, key=pre_score)[:MAX_CANDIDATES_TO_LLM]

    # Prompt priorities (what you asked for)
    prompt = {
        "task": "Choose the best callout placement candidate.",
        "label": label,
        "priorities": [
            "Prefer LEFT or RIGHT side gutters over top/bottom whenever possible.",
            "Prefer the candidate that is vertically closest to the highlighted (target) text (small dist_y).",
            "Prefer the shortest connector line (small connector_len).",
            "Avoid connector lines that cross content (connector_crossings_est should be minimal).",
            "Top placements are a last resort.",
            "If label is 'Original source of publication.' prefer BOTTOM if it yields a shorter connector and keeps the callout near the URL.",
        ],
        "target_union": {"x0": target_union.x0, "y0": target_union.y0, "x1": target_union.x1, "y1": target_union.y1},
        "candidates": candidates_sorted,
        "output_rule": "Return ONLY the chosen_id (from the provided candidates) and a short reason.",
    }

    schema = {
        "name": "callout_choice",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "chosen_id": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["chosen_id", "reason"],
        },
    }

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a layout assistant that selects the best callout placement candidate."},
            {"role": "user", "content": json.dumps(prompt)},
        ],
        temperature=LLM_TEMPERATURE,
        response_format={"type": "json_schema", "json_schema": schema},
    )

    content = resp.choices[0].message.content
    data = json.loads(content)
    chosen_id = int(data.get("chosen_id", -1))
    return chosen_id


# ============================================================
# Placement wrapper: Python generates candidates, LLM chooses, Python draws
# ============================================================

def _choose_best_spot_llm(
    page: fitz.Page,
    targets: List[fitz.Rect],
    occupied_callouts: List[fitz.Rect],
    label: str,
) -> Tuple[fitz.Rect, str, int, bool]:
    target_union = _union_rect(targets)

    # Blockers around highlights
    highlight_blockers = [inflate_rect(t, GAP_FROM_HIGHLIGHTS) for t in targets]

    candidates = _generate_candidates(
        page=page,
        target_union=target_union,
        label=label,
        occupied_callouts=occupied_callouts,
        highlight_blockers=highlight_blockers,
    )

    chosen_id = _llm_choose_candidate(label=label, target_union=target_union, candidates=candidates)

    chosen = None
    for c in candidates:
        if c["id"] == chosen_id:
            chosen = c
            break

    # Fallback if LLM fails or returns invalid id
    if chosen is None:
        # safest: pick the first candidate
        if not candidates:
            # ultimate fallback: top-left tiny box (no bg)
            pr = page.rect
            fs, wrapped, w, h = _optimize_layout(label, 140.0)
            rect = fitz.Rect(EDGE_PAD, EDGE_PAD, EDGE_PAD + w, EDGE_PAD + h) & pr
            return rect, wrapped, fs, False
        chosen = candidates[0]

    r = chosen["rect"]
    rect = fitz.Rect(r["x0"], r["y0"], r["x1"], r["y1"])
    wrapped_text = chosen["wrapped_text"]
    fs = int(chosen["fontsize"])

    # White bg: safe if not intersecting any blockers (it shouldn't) AND the zone is likely clean
    # Since candidates were hard-filtered, we allow bg. If you want stricter, you can add an ink check here.
    safe_for_white_bg = True

    return rect, wrapped_text, fs, safe_for_white_bg


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

    # B) Metadata callouts (page 1) using LLM placement
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

        callout_rect, wrapped_text, fs, safe_for_white_bg = _choose_best_spot_llm(
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

        # Obstacles for endpoint choice (keep connectors from crossing content)
        obstacles: List[fitz.Rect] = []
        for b in _page_text_blocks(page1):
            obstacles.append(inflate_rect(b, 2.0))
        for o in occupied_callouts:
            obstacles.append(inflate_rect(o, 2.0))
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

    _do_job("Original source of publication.", _url_variants(str(meta.get("source_url") or "")), connect_policy="union")

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
        uniq = list(dict.fromkeys(needles))
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
