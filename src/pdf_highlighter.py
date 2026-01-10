# ============================================================
# Candidate generation + scoring (FIXED)
# ============================================================

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

    # Obstacles for connector line scoring
    connector_obstacles = []
    for b in _page_text_shapes(page): connector_obstacles.append(inflate_rect(b, 1.5))
    for o in occupied_callouts: connector_obstacles.append(inflate_rect(o, 2.0))

    zones = _zones(page)
    
    # We broaden the search: check every 20 points vertically to find a gap
    # and try different horizontal alignments within the zones
    best_overall_cand = None
    min_overall_score = float('inf')

    # Iterate through ink-check levels: first try pure white, then slightly "dirty" white
    for threshold, ratio in INKCHECK_LEVELS:
        candidates = []
        
        for zone_name, z in zones:
            bw = _box_width_for_zone(zone_name, z)
            fs, wrapped, w, h = _optimize_layout(label, bw)

            # Scan the zone vertically in increments of 20pts
            # starting from the target Y and moving outwards
            y_range = sorted(range(int(z.y0 + h/2), int(z.y1 - h/2), 20), 
                            key=lambda y: abs(y - tc.y))

            for cy in y_range:
                y0, y1 = cy - h/2, cy + h/2
                
                # Align to the "inner" edge of the margin (closer to text)
                if zone_name == "left":
                    x1 = z.x1
                    x0 = x1 - w
                elif zone_name == "right":
                    x0 = z.x0
                    x1 = x0 + w
                else: # Top/Bottom
                    x0 = max(z.x0, min(tc.x - w/2, z.x1 - w))
                    x1 = x0 + w

                cand = fitz.Rect(x0, y0, x1, y1)
                
                # Check Hard Constraints
                if _intersects_any(cand, occupied_buf): continue
                if _intersects_any(cand, highlight_blockers): continue
                
                # Use current ink-check level
                if _rect_has_ink(page, inflate_rect(cand, INKCHECK_PAD), 
                                white_threshold=threshold, nonwhite_ratio=ratio):
                    continue

                # Scoring
                s, e, hits, length = _straight_connector_best_pair(cand, target_union, connector_obstacles)
                
                # Score components: 
                # 1. Huge penalty for crossing text
                # 2. Medium penalty for vertical distance from target
                # 3. Small penalty for line length
                score = (hits * 10000.0) + (abs(cy - tc.y) * 50.0) + length
                
                candidates.append((score, cand, wrapped, fs))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            score, rect, wrapped, fs = candidates[0]
            return rect, wrapped, fs, True # Found a "safe" spot

    # EMERGENCY FALLBACK: If ink-check fails everywhere, 
    # ignore ink-check but still respect other callouts
    # (Prevents the "top-left stack" bug)
    for zone_name, z in zones:
        bw = _box_width_for_zone(zone_name, z)
        fs, wrapped, w, h = _optimize_layout(label, bw)
        
        # Just place it at the target's Y level, clamped to the zone
        target_y = max(z.y0, min(tc.y - h/2, z.y1 - h))
        cand = fitz.Rect(z.x0, target_y, z.x0 + w, target_y + h)
        
        # If this overlaps another callout, nudge it down until it fits
        attempts = 0
        while _intersects_any(cand, occupied_buf) and attempts < 10:
            cand.y0 += (h + GAP_BETWEEN_CALLOUTS)
            cand.y1 += (h + GAP_BETWEEN_CALLOUTS)
            attempts += 1
            
        if not _intersects_any(cand, occupied_buf):
            return cand, wrapped, fs, False

    # Ultimate fallback (spread out vertically)
    offset = len(occupied_callouts) * 60
    fallback = fitz.Rect(EDGE_PAD, EDGE_PAD + offset, EDGE_PAD + 150, EDGE_PAD + offset + 50)
    return fallback, label, 10, False
