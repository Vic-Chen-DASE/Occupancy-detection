"""IoU + center-distance association between YOLO boxes and radar projections."""

from __future__ import annotations

import math


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def center_dist(a, b):
    ax, ay = _center(a)
    bx, by = _center(b)
    return math.hypot(ax - bx, ay - by)


def associate(thermal_dets, radar_proj, iou_min=0.15, center_max_px=40.0):
    """
    Greedy match thermal dets to radar projected boxes.

    Prefer IoU >= iou_min; remaining pairs matched by center distance
    if within center_max_px (in the same pixel space as the boxes).

    thermal_dets: list[{box:[x1,y1,x2,y2], conf, ...}]
    radar_proj:   list[{box:[...], id, in_view, ...}] from project_track

    Returns:
      confirmed: list of {thermal, radar, iou, match}
      thermal_only: unmatched thermal
      radar_only: unmatched in-view radar (always drawable as projection)
    """
    dets = list(thermal_dets or [])
    rads = [r for r in (radar_proj or []) if r.get("in_view", True) and r.get("box")]
    used_t, used_r = set(), set()
    confirmed = []

    # Pass 1: IoU
    pairs = []
    for ti, td in enumerate(dets):
        for ri, rd in enumerate(rads):
            score = iou(td["box"], rd["box"])
            if score >= iou_min:
                pairs.append((score, ti, ri))
    pairs.sort(reverse=True)
    for score, ti, ri in pairs:
        if ti in used_t or ri in used_r:
            continue
        used_t.add(ti)
        used_r.add(ri)
        confirmed.append({
            "thermal": dets[ti],
            "radar": rads[ri],
            "iou": score,
            "match": "iou",
        })

    # Pass 2: center distance for leftovers
    if center_max_px is not None and center_max_px > 0:
        c_pairs = []
        for ti, td in enumerate(dets):
            if ti in used_t:
                continue
            for ri, rd in enumerate(rads):
                if ri in used_r:
                    continue
                d = center_dist(td["box"], rd["box"])
                if d <= center_max_px:
                    c_pairs.append((d, ti, ri))
        c_pairs.sort()  # nearest first
        for d, ti, ri in c_pairs:
            if ti in used_t or ri in used_r:
                continue
            used_t.add(ti)
            used_r.add(ri)
            confirmed.append({
                "thermal": dets[ti],
                "radar": rads[ri],
                "iou": iou(dets[ti]["box"], rads[ri]["box"]),
                "match": "center",
                "center_px": d,
            })

    thermal_only = [d for i, d in enumerate(dets) if i not in used_t]
    radar_only = [r for i, r in enumerate(rads) if i not in used_r]
    return confirmed, thermal_only, radar_only
