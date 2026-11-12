#!/usr/bin/env python3
"""Headless smoke for vic_fusion_ui: synthetic thermal + JSONL -> PNG."""

from __future__ import annotations

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SAMPLE = os.path.join(HERE, "samples", "synthetic_tracks.jsonl")


def main():
    out_dir = tempfile.mkdtemp(prefix="vic_ui_")
    out_png = os.path.join(out_dir, "composite.png")
    # Import after path setup
    import vic_fusion_ui as ui

    rc = ui.main([
        "--smoke",
        "--tracks-jsonl", SAMPLE,
        "--smoke-out", out_png,
        "--smoke-seconds", "0.5",
        "--headless",
        "--no-camera",
    ])
    if rc != 0:
        print(f"FAIL ui.main rc={rc}")
        return rc
    if not os.path.isfile(out_png):
        print(f"FAIL missing png {out_png}")
        return 1
    size = os.path.getsize(out_png)
    print(f"ok  smoke png {out_png} bytes={size}")

    # Also unit-test associate + render without camera
    from vic_ui.associate import associate, iou
    assert abs(iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-6
    confirmed, t_only, r_only = associate(
        [{"box": [10, 10, 40, 50], "conf": 0.9}],
        [{"box": [12, 12, 38, 48], "id": 5, "in_view": True, "range_m": 3.0}],
        iou_min=0.1,
    )
    assert len(confirmed) == 1 and confirmed[0]["radar"]["id"] == 5
    assert not t_only and not r_only
    print("ok  associate IoU match")

    # Center-distance fallback when IoU is tiny / zero
    confirmed2, t2, r2 = associate(
        [{"box": [10, 10, 30, 40], "conf": 0.8}],
        [{"box": [28, 28, 50, 60], "id": 7, "in_view": True, "range_m": 3.4}],
        iou_min=0.5,  # IoU too strict
        center_max_px=40.0,
    )
    assert len(confirmed2) == 1 and confirmed2[0]["match"] == "center"
    assert confirmed2[0]["radar"]["id"] == 7
    print("ok  associate center fallback")

    # YOLO empty → radar_only populated (projections must still draw)
    confirmed3, t3, r3 = associate(
        [],
        [{"box": [40, 30, 80, 90], "id": 5, "in_view": True, "range_m": 3.4}],
        iou_min=0.15,
    )
    assert not confirmed3 and not t3 and len(r3) == 1
    print("ok  associate YOLO-empty → radar_only")

    import vic_fusion_ui as vui
    c, mode = vui._occupancy([], [{"id": 5}], dets=[], show_unconfirmed=False)
    assert c == 1 and mode == "radar", (c, mode)
    c2, mode2 = vui._occupancy(
        [{"thermal": {}, "radar": {"id": 5}, "iou": 0.5}],
        [{"id": 5}, {"id": 1000}],
        dets=[{"box": [0, 0, 1, 1]}],
    )
    assert c2 == 2 and mode2 in ("fusion", "fusion+radar"), (c2, mode2)
    print("ok  occupancy radar fallback + max(confirmed, radar_n)")

    from radar_link_from_tracks import TracksRadarLink
    link = TracksRadarLink()
    link.apply_dict({
        "frame": 1, "generation": 0,
        "tracks": [
            {"id": 5, "x": 0.0, "y": 4.0, "z": 0.9,
             "vx": 0, "vy": -0.5, "vz": 0, "conf": 1,
             "z_min": 0, "z_max": 1.7, "range": 4.0},
            {"id": 1000, "x": 1.0, "y": 3.5, "z": 0.9,
             "vx": 0, "vy": 0, "vz": 0, "conf": 1,
             "z_min": 0, "z_max": 1.7, "range": 3.6},
        ],
    })
    from vic_ui import render as R
    from vic_ui.camera import synthetic_thermal, frame_to_bgr
    left = R.draw_radar_pane(link.tracks(), count=2)
    right = R.draw_thermal_pane(
        frame_to_bgr(synthetic_thermal(blobs=[(80, 60, 12, 20, 200)])),
        confirmed=[{
            "thermal": {"box": [60, 40, 100, 90], "conf": 0.8},
            "radar": {"box": [62, 42, 98, 88], "id": 5, "range_m": 4.0},
            "iou": 0.7,
        }],
        thermal_only=[],
        radar_only=[{
            "box": [20, 20, 50, 70], "id": 1000, "range_m": 3.4,
        }],
        always_draw_radar=True,
    )
    canvas = R.compose(left, right, ["smoke=unit", "radar_n=2"], fps=30.0)
    assert canvas.shape[0] == R.PANEL_H + R.STATUS_H
    assert canvas.shape[1] == R.LEFT_W + R.RIGHT_W
    print(f"ok  compose shape={canvas.shape}")

    # 3D pane should not be flat black; yellow volume + orange boxes paint pixels
    import numpy as np
    left3 = R.draw_radar_pane(link.tracks(), count=2, points=[
        {"x": 0.0, "y": 4.0, "z": 0.9, "snr": 20},
        {"x": 0.1, "y": 4.1, "z": 1.0, "snr": 15},
    ])
    assert left3.shape == (R.PANEL_H, R.LEFT_W, 3)
    assert int(left3.max()) > 40, "3D pane looks empty"
    # flipped thermal differs from unflipped
    syn = frame_to_bgr(synthetic_thermal(blobs=[(80, 30, 12, 20, 200)]))
    a = R.draw_thermal_pane(syn, [], [], [], flip_v=False)
    b = R.draw_thermal_pane(syn, [], [], [], flip_v=True)
    assert not np.array_equal(a, b), "flip_v should change thermal pane"
    print("ok  3D radar pane + thermal flip")

    args = vui.parse_args([])
    assert args.tracks_udp == "0.0.0.0:9100", args.tracks_udp
    assert args.flip is True, args.flip
    args2 = vui.parse_args(["--no-flip"])
    assert args2.flip is False
    print("ok  default --tracks-udp 0.0.0.0:9100; flip default ON / --no-flip")

    print("ALL VIC_UI SMOKE PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
