# src/pdf_highlighter.py
import io
import math
import os
import json
from typing import Dict, List, Tuple, Optional, Any

import fitz  # PyMuPDF
from openai import OpenAI

RED = (1, 0, 0)
WHITE = (1, 1, 1)

# ---- spacing knobs (tweakable) ----
GAP_FROM_TEXT_BLOCKS = 10.0   # min gap from existing text/images for callout placement
GAP_FROM_HIGHLIGHTS = 14.0    # min gap from red highlight boxes for callout placement
GAP_BETWEEN_CALLOUTS = 10.0   # min gap between callouts
EDGE_PAD = 18.0              # padding from page edge
ENDPOINT_PULLBACK = 1.5      # pull connector end slightly away from target edge

# Visuals (match your “good” look)
HIGHLIGHT_WIDTH = 1.5
CONNECTOR_WIDTH = 1.0
FONTNAME = "Times-Roman"
FONT_SIZES = [12, 11, 10]
MARGIN_W = 120.0

# AI
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
AI_TEMPERATURE = 0

# Candidate generation (this is the main “tweak surface”)
Y_OFFSETS = [-180, -140, -110, -80, -55, -30, 0, 30, 55, 80, 110, 140, 180]
BOTTOM_Y_OFFSETS = [-120, -90, -60, -30, 0]  # relative to page bottom band
MAX_CANDIDATES_TO_LLM = 60


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

def _segment_intersects_rect(p1: fitz.Point, p2: fitz.Point, r: fitz.Rect) -> bool:
    steps = 18
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

def _callout_mid_edge_anchor(callout: fitz.Rect, target_center: fitz.Point) -> fitz.Point:
    y = callout.y0 + (callout.height / 2.0)
    if target_center.x >= (callout.x0 + callout.width / 2.0):
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
    best_pt = _center(target)
    best_hits = 10**9
    best_len = 10**9

    for pt in _target_edge_candidates(target):
        hits = 0
        for ob in obstacles:
            if _segment_intersects_rect(start, pt, ob):
                hits += 1

        seg_len = math.hypot(pt.x - start.x, pt.y - start.y)
        if hits < best_hits or (hits == best_hits and seg_len < best_len):
            best_hits, best_len, best_pt = hits, seg_len, pt

    return best_pt

def _draw_connector_straight(
    page: fitz.Page,
    callout_rect: fitz.Rect,
    target_rect: fitz.Rect,
    obstacles: List[fitz.Rect],
):
    """Straight line only (never elbow). Choose best endpoint to reduce crossings."""
    target_center = _center(target_rect)
    start = _callout_mid_edge_anchor(callout_rect, target_center)
    end = _choose_target_attachment(start, target_rect, obstacles)
    end = _pull_back_point(start, end, ENDPOINT_PULLBACK)
    page.draw_line(start, end, color=RED, width=CONNECTOR_WIDTH)


# ============================================================
# Blockers (avoid covering existing content)
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
            for r in page.get_image_rects(xref):
                blockers.append(inflate_rect(fitz.Rect(r), pad))
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
# Margin wrapping
# ============================================================

def _optimize_layout_for_margin(
    text: str,
    box_width: float,
    fontname: str = FONTNAME,
) -> Tuple[int, str, float, float]:
    best = (10, (text or ""), box_width, 40.0)
    best_h = float("inf")

    for fs in FONT_SIZES:
        words = (text or "").split()
        if not words:
            return fs, "", box_width, 30.0

        lines: List[str] = []
        cur: List[str] = []
        usable_w = max(10.0, box_width - 10.0)

        for w in words:
            trial = " ".join(cur + [w])
            if fitz.get_text_length(trial, fontname=fontname, fontsize=fs) <= usable_w:
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

    return best


# ============================================================
# AI placement: generate multiple safe candidates then let AI pick
# ============================================================

def _clamp_to_page(page: fitz.Page, r: fitz.Rect) -> fitz.Rect:
    pr = page.rect
    rr = fitz.Rect(r)
    # clamp vertically
    if rr.y0 < EDGE_PAD:
        rr.y1 += (EDGE_PAD - rr.y0)
        rr.y0 = EDGE_PAD
    if rr.y1 > pr.height - EDGE_PAD:
        rr.y0 -= (rr.y1 - (pr.height - EDGE_PAD))
        rr.y1 = pr.height - EDGE_PAD
    # clamp horizontally just in case
    if rr.x0 < EDGE_PAD:
        rr.x1 += (EDGE_PAD - rr.x0)
        rr.x0 = EDGE_PAD
    if rr.x1 > pr.width - EDGE_PAD:
        rr.x0 -= (rr.x1 - (pr.width - EDGE_PAD))
        rr.x1 = pr.width - EDGE_PAD
    return rr

def _estimate_connector_metrics(
    page: fitz.Page,
    callout_rect: fitz.Rect,
    target_rect: fitz.Rect,
    obstacles: List[fitz.Rect],
) -> Tuple[float, int]:
    target_center = _center(target_rect)
    start = _callout_mid_edge_anchor(callout_rect, target_center)
    end = _choose_target_attachment(start, target_rect, obstacles)
    length = math.hypot(end.x - start.x, end.y - start.y)
    crossings = sum(1 for ob in obstacles if _segment_intersects_rect(start, end, ob))
    return length, crossings

def _llm_choose_candidate(label: str, candidates: List[Dict[str, Any]]) -> int:
    """Returns chosen candidate id. LLM can ONLY pick from the list."""
    if not candidates:
        return -1

    client = OpenAI()

    # Keep payload small and focused
    cand_payload = candidates[:MAX_CANDIDATES_TO_LLM]

    prompt = {
        "task": "Pick the best callout placement candidate for a PDF annotation.",
        "label": label,
        "rules": [
            "Prefer LEFT or RIGHT margin candidates over TOP. TOP is a last resort.",
            "Prefer smaller vertical distance to the red-boxed target (dist_y).",
            "Prefer shorter connector line (connector_len).",
            "Prefer fewer line crossings (connector_crossings_est).",
            "Avoid placements very near the top unless the target is near the top.",
            "Special rule: If label is 'Original source of publication.' then BOTTOM is often best if it keeps the callout near the URL.",
        ],
        "candidates": cand_payload,
        "output": "Return the chosen_id from the provided candidates."
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
            {"role": "system", "content": "You select the best candidate rectangle for a callout label."},
            {"role": "user", "content": json.dumps(prompt)},
        ],
        temperature=AI_TEMPERATURE,
        response_format={"type": "json_schema", "json_schema": schema},
    )

    data = json.loads(resp.choices[0].message.content)
    return int(data.get("chosen_id", -1))

def _choose_best_margin_spot_ai(
    page: fitz.Page,
    targets: List[fitz.Rect],
    occupied: List[fitz.Rect],
    label: str,
) -> Tuple[fitz.Rect, str, int, bool]:
    """
    Generates multiple safe candidates in left/right (and bottom for URL-ish label),
    then uses LLM to pick best among them.
    Returns: (rect, wrapped_text, fontsize, safe_for_white_bg)
    """
    pr = page.rect
    target_union = _union_rect(targets)
    target_center = _center(target_union)
    target_y = target_center.y

    fs, wrapped, w, h = _optimize_layout_for_margin(label, MARGIN_W, fontname=FONTNAME)

    left_x = EDGE_PAD
    right_x = pr.width - EDGE_PAD - w

    blockers = _page_blockers(page, pad=GAP_FROM_TEXT_BLOCKS)

    # Keep away from the highlight boxes
    highlight_blockers = [inflate_rect(t, GAP_FROM_HIGHLIGHTS) for t in targets]
    blockers.extend(highlight_blockers)

    # Keep away from other callouts
    occupied_buf = [inflate_rect(o, GAP_BETWEEN_CALLOUTS) for o in occupied]
    blockers.extend(occupied_buf)

    # Obstacles for connector scoring (text blocks + other callouts + other targets)
    connector_obstacles: List[fitz.Rect] = []
    try:
        for b in page.get_text("blocks"):
            connector_obstacles.append(inflate_rect(fitz.Rect(b[:4]), 2.0))
    except Exception:
        pass
    for oc in occupied:
        connector_obstacles.append(inflate_rect(oc, 2.0))
    for hb in highlight_blockers:
        connector_obstacles.append(inflate_rect(hb, 2.0))

    candidates: List[Dict[str, Any]] = []
    cid = 1

    def maybe_add_candidate(zone: str, rect: fitz.Rect):
        nonlocal cid
        rr = _clamp_to_page(page, rect)
        if rr.width < 30 or rr.height < 18:
            return
        if _intersects_any(rr, blockers):
            return

        conn_len, conn_cross = _estimate_connector_metrics(page, rr, target_union, connector_obstacles)

        # simple “too-top” penalty heuristic included as a feature
        topness = (rr.y0 - EDGE_PAD) / max(1.0, (pr.height - 2 * EDGE_PAD))

        candidates.append({
            "id": cid,
            "zone": zone,
            "rect": {"x0": rr.x0, "y0": rr.y0, "x1": rr.x1, "y1": rr.y1},
            "dist_y": abs((_center(rr).y) - target_y),
            "connector_len": conn_len,
            "connector_crossings_est": conn_cross,
            "topness": topness,
        })
        cid += 1

    # Left/right candidates at multiple Y offsets
    for dy in Y_OFFSETS:
        cy = target_y + dy
        maybe_add_candidate("left", fitz.Rect(left_x, cy - h / 2, left_x + w, cy + h / 2))
        maybe_add_candidate("right", fitz.Rect(right_x, cy - h / 2, right_x + w, cy + h / 2))

    # Bottom candidates (only if label suggests URL/source)
    is_source_label = "source of publication" in (label or "").lower()
    if is_source_label:
        bottom_band_top = pr.height - EDGE_PAD - 180.0
        bottom_band_bottom = pr.height - EDGE_PAD
        bottom_y_center = (bottom_band_top + bottom_band_bottom) / 2.0
        for dy in BOTTOM_Y_OFFSETS:
            cy = bottom_y_center + dy
            # allow both left and right bottom
            maybe_add_candidate("bottom_left", fitz.Rect(left_x, cy - h / 2, left_x + w, cy + h / 2))
            maybe_add_candidate("bottom_right", fitz.Rect(right_x, cy - h / 2, right_x + w, cy + h / 2))

    # If nothing survived, revert to your old two-shot logic (no white bg if overlaps)
    if not candidates:
        cand_left = _clamp_to_page(page, fitz.Rect(left_x, target_y - h / 2, left_x + w, target_y + h / 2))
        if not _intersects_any(cand_left, blockers):
            return cand_left, wrapped, fs, True
        cand_right = _clamp_to_page(page, fitz.Rect(right_x, target_y - h / 2, right_x + w, target_y + h / 2))
        if not _intersects_any(cand_right, blockers):
            return cand_right, wrapped, fs, True
        # ultimate fallback (no white bg)
        fallback = _clamp_to_page(page, fitz.Rect(EDGE_PAD, EDGE_PAD, EDGE_PAD + w, EDGE_PAD + h))
        return fallback, wrapped, fs, False

    # Sort by a light heuristic before sending to LLM (keeps best candidates in payload)
    def pre_score(c: Dict[str, Any]) -> float:
        zone = c["zone"]
        zone_bonus = 0.0
        if zone == "left":
            zone_bonus = -3000.0
        elif zone == "right":
            zone_bonus = -2800.0
        elif zone.startswith("bottom"):
            zone_bonus = -2400.0 if is_source_label else -1200.0
        else:
            zone_bonus = 500.0
        # discourage extreme top unless target is top
        top_penalty = 1200.0 * max(0.0, 0.22 - c["topness"])  # penalty if very close to top
        return zone_bonus + c["dist_y"] * 1.0 + c["connector_len"] * 0.7 + c["connector_crossings_est"] * 2500.0 + top_penalty

    candidates = sorted(candidates, key=pre_score)

    chosen_id = _llm_choose_candidate(label, candidates)
    chosen = None
    for c in candidates:
        if c["id"] == chosen_id:
            chosen = c
            break
    if chosen is None:
        chosen = candidates[0]

    r = chosen["rect"]
    rect = fitz.Rect(r["x0"], r["y0"], r["x1"], r["y1"])

    # Safe for white bg because it was filtered against blockers/occupied.
    return rect, wrapped, fs, True


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
    occupied_callouts: List[fitz.Rect] = []
    total_quote_hits = 0
    total_meta_hits = 0

    # A) Quote highlights across all pages
    for page in doc:
        for term in (quote_terms or []):
            t = (term or "").strip()
            if not t:
                continue
            try:
                rects = page.search_for(t)
            except Exception:
                rects = []
            for r in rects:
                page.draw_rect(r, color=RED, width=HIGHLIGHT_WIDTH)
                total_quote_hits += 1

    # B) Metadata callouts (page 1)
    def _do_job(
        label: str,
        value: Optional[str],
        *,
        connector_policy: str = "union",  # "single" | "union" | "all"
    ):
        nonlocal total_meta_hits, occupied_callouts

        if not value or not str(value).strip():
            return

        needle = str(value).strip()
        try:
            targets = page1.search_for(needle)
        except Exception:
            targets = []
        if not targets:
            return

        # draw red boxes
        for t in targets:
            page1.draw_rect(t, color=RED, width=HIGHLIGHT_WIDTH)
        total_meta_hits += len(targets)

        # choose callout location (AI)
        callout_rect, wrapped_text, fs, safe = _choose_best_margin_spot_ai(
            page1, targets, occupied_callouts, label
        )

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

        # Obstacles for connectors (to choose best endpoint)
        obstacle_rects: List[fitz.Rect] = []
        try:
            for b in page1.get_text("blocks"):
                obstacle_rects.append(inflate_rect(fitz.Rect(b[:4]), 2.0))
        except Exception:
            pass

        expanded_targets = [inflate_rect(t, 2.0) for t in targets]

        def connect_one(target_rect: fitz.Rect):
            obs = obstacle_rects[:]
            for ot in expanded_targets:
                if not ot.intersects(target_rect):
                    obs.append(ot)
            for oc in occupied_callouts:
                obs.append(inflate_rect(oc, 2.0))
            _draw_connector_straight(page1, callout_rect, target_rect, obs)

        if connector_policy == "all":
            for t in targets:
                connect_one(t)
        elif connector_policy == "single":
            connect_one(targets[0])
        else:
            connect_one(_union_rect(targets))

        occupied_callouts.append(callout_rect)

    jobs = [
        ("Original source of publication.", meta.get("source_url"), "union"),
        ("The distinguished organization.", meta.get("venue_name") or meta.get("org_name"), "union"),
        ("Performance date.", meta.get("performance_date"), "union"),
        ("Beneficiary lead role evidence.", meta.get("beneficiary_name"), "all"),
    ]

    for label, value, policy in jobs:
        _do_job(label, value, connector_policy=policy)

    out = io.BytesIO()
    doc.save(out)
    doc.close()
    out.seek(0)

    return out.getvalue(), {
        "total_quote_hits": total_quote_hits,
        "total_meta_hits": total_meta_hits,
        "criterion_id": criterion_id,
    }
