#!/usr/bin/env python3
"""
Vic one-window fusion UI — radar 3D-ish (left) + thermal (right).

Single process:
  * listen for v6.12 locked-box tracks (UDP and/or JSONL)
  * open thermal camera (OpenCV; prefer device 1 / DSHOW on Windows)
  * optional YOLO if ultralytics + weights are available
  * associate thermal dets with radar projections (IoU + center);
    occupancy = confirmed when matched, else radar lock count (YOLO-miss fallback)
  * always draw radar projections on thermal even when ass=0

Keys:
  T  toggle trails (default OFF)
  R  reset trails / association state
  + / -  YOLO conf
  U  toggle show-unconfirmed thermal-only boxes (default OFF)
  Q / Esc  quit

UDP port: default --tracks-udp is 0.0.0.0:9100 to match v6.12
  --export-tracks-udp 127.0.0.1:9100. Binding 8100 (or any other port) with
  A still exporting 9100 yields a silent mismatch — radar_n stays 0/stale.

Smoke (no camera, no display required if --smoke-out set):
  python vic_fusion_ui.py --smoke --tracks-jsonl samples/synthetic_tracks.jsonl \
      --smoke-out /tmp/vic_ui_smoke.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

VENDOR = os.path.join(HERE, "vendor")
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from radar_link_from_tracks import TracksRadarLink  # noqa: E402
from vic_ui.associate import associate  # noqa: E402
from vic_ui.camera import open_camera, frame_to_bgr, synthetic_thermal  # noqa: E402
from vic_ui.detect import Detector  # noqa: E402
from vic_ui import render as R  # noqa: E402

try:
    import project as P
    _HAS_PROJECT = True
except Exception as _pe:  # noqa: BLE001
    P = None
    _HAS_PROJECT = False
    _PROJECT_ERR = _pe


WINDOW = "Vic Fusion"


def _build_cam_ext(pitch=0.0, tz=-0.08):
    if not _HAS_PROJECT:
        return None, None
    return P.Camera(), P.Extrinsics(tz=tz, pitch=pitch)


def _project_tracks(link, cam, ext, body_width=0.50):
    if cam is None or ext is None or P is None:
        return []
    out = []
    for t in link.tracks():
        b = P.project_track(t, cam, ext, body_width)
        if b:
            out.append(b)
    return out


def _occupancy(confirmed, tracks, dets, show_unconfirmed=False, thermal_only=None):
    """
    Occupancy policy:
      * confirmed (IoU/center match) when any association exists → mode fusion
      * else if radar has locks (YOLO empty or no overlap) → radar fallback
      * optional U: add thermal-only dets on top of confirmed (debug)

    Returns (count, mode_label) where mode_label is "fusion" | "radar" | "debug".
    """
    n_conf = len(confirmed or [])
    n_radar = len(tracks or [])
    n_t_only = len(thermal_only or [])
    if show_unconfirmed and n_conf + n_t_only > 0:
        return n_conf + n_t_only, "debug"
    if n_conf > 0:
        # Prefer fusion, but never under-count vs live radar locks
        return max(n_conf, n_radar), ("fusion" if n_conf >= n_radar else "fusion+radar")
    if n_radar > 0:
        return n_radar, "radar"
    return 0, "none"


def run_loop(args, link, cap, detector, smoke_frames=None, smoke_out=None):
    import cv2
    import numpy as np

    cam, ext = _build_cam_ext(pitch=args.pitch, tz=args.tz)
    trails = R.TrailStore(max_len=50)
    fpsm = R.FpsMeter()
    show_trails = False
    show_unconfirmed = False
    conf = float(args.conf)
    detector.set_conf(conf)

    headless = bool(smoke_out) or bool(args.headless)
    if not headless:
        try:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, R.LEFT_W + R.RIGHT_W, R.PANEL_H + R.STATUS_H)
        except Exception:
            headless = True

    smoke_i = 0
    last_frame = None
    t_end = time.time() + float(args.smoke_seconds) if args.smoke else None

    while True:
        # --- tracks ---
        tracks = link.tracks()
        trails.update(tracks)
        st = link.status()

        # --- thermal frame ---
        thermal_bgr = None
        cam_note = "cam=off"
        if smoke_frames is not None:
            if smoke_i >= len(smoke_frames):
                break
            raw = smoke_frames[smoke_i]
            smoke_i += 1
            thermal_bgr = frame_to_bgr(raw) if raw.ndim == 2 else raw
            cam_note = "cam=synthetic"
        elif cap is not None:
            ok, frame = cap.read()
            if ok and frame is not None:
                thermal_bgr = frame_to_bgr(frame)
                cam_note = f"cam={args.device}"
            else:
                cam_note = "cam=read-fail"

        # Scale dets to native thermal resolution for association with project boxes
        # project boxes are in Camera() 160x120 space by default.
        dets = []
        if detector.ready and thermal_bgr is not None:
            # Run YOLO on display-sized or native; map boxes back to 160x120 if needed
            dets_raw = detector.detect(thermal_bgr)
            th, tw = thermal_bgr.shape[0], thermal_bgr.shape[1]
            # project uses 160x120 — scale dets into that space for IoU
            sx, sy = 160.0 / tw, 120.0 / th
            for d in dets_raw:
                x1, y1, x2, y2 = d["box"]
                dets.append({
                    "box": [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                    "conf": d["conf"],
                    "cls": d.get("cls", 0),
                })
            # Also keep pixel boxes for drawing on the scaled pane: use native
            # We redraw using dets scaled to thermal_bgr size — store both.
            for i, d in enumerate(dets_raw):
                dets[i]["box_px"] = d["box"]

        proj = _project_tracks(link, cam, ext, body_width=args.body_width)

        # Associate in 160x120 space
        confirmed, thermal_only, radar_only = associate(
            dets, proj, iou_min=args.iou_min, center_max_px=args.center_max
        )

        # For drawing, prefer native-pixel thermal boxes when present
        confirmed_draw = []
        for c in confirmed:
            td = dict(c["thermal"])
            if "box_px" in td:
                td["box"] = td["box_px"]
            # radar box is in 160x120 — scale to thermal_bgr for draw_thermal_pane
            rd = dict(c["radar"])
            if thermal_bgr is not None and rd.get("box"):
                th, tw = thermal_bgr.shape[0], thermal_bgr.shape[1]
                bx = rd["box"]
                rd["box"] = [
                    bx[0] / 160.0 * tw, bx[1] / 120.0 * th,
                    bx[2] / 160.0 * tw, bx[3] / 120.0 * th,
                ]
            confirmed_draw.append({"thermal": td, "radar": rd, "iou": c["iou"]})

        thermal_only_draw = []
        for td0 in thermal_only:
            td = dict(td0)
            if "box_px" in td:
                td["box"] = td["box_px"]
            thermal_only_draw.append(td)

        radar_only_draw = []
        for rd0 in radar_only:
            rd = dict(rd0)
            if thermal_bgr is not None and rd.get("box"):
                th, tw = thermal_bgr.shape[0], thermal_bgr.shape[1]
                bx = rd["box"]
                rd["box"] = [
                    bx[0] / 160.0 * tw, bx[1] / 120.0 * th,
                    bx[2] / 160.0 * tw, bx[3] / 120.0 * th,
                ]
            radar_only_draw.append(rd)

        # Occupancy: confirmed when associated; else radar lock fallback so a
        # YOLO miss with a live lock does not stick people=0.
        count, occ_mode = _occupancy(
            confirmed, tracks, dets,
            show_unconfirmed=show_unconfirmed,
            thermal_only=thermal_only,
        )
        count_label = {
            "fusion": "people",
            "fusion+radar": "people*",
            "radar": "radar",
            "debug": "debug",
            "none": "people",
        }.get(occ_mode, "people")

        points = None
        if hasattr(link, "points"):
            try:
                points = link.points()
            except Exception:
                points = None
        left = R.draw_radar_pane(
            tracks,
            trails=trails.data,
            show_trails=show_trails,
            count=count,
            count_label=count_label,
            boundary=getattr(args, "boundary", None),
            points=points,
            sensor_height=float(getattr(args, "sensor_height", 2.4)),
        )
        right = R.draw_thermal_pane(
            thermal_bgr,
            confirmed_draw,
            thermal_only_draw,
            radar_only_draw,
            show_unconfirmed=show_unconfirmed,
            show_trails=False,
            always_draw_radar=True,
            flip_v=bool(getattr(args, "flip", True)),
        )

        udp = (st.get("source") or {}).get("udp") or "—"
        jsonl = (st.get("source") or {}).get("jsonl")
        src = f"udp={udp}" if udp and udp != "—" else (
            f"jsonl={os.path.basename(jsonl)}" if jsonl else "tracks=—"
        )
        yolo_note = (
            f"yolo={detector.backend} conf={conf:.2f}"
            if detector.ready else "yolo=off"
        )
        stale = st.get("stale_s")
        stale_s = f"stale={stale:.1f}s" if stale is not None else "stale=—"
        status = [
            src,
            f"radar_n={len(tracks)} gen={st.get('generation', 0)}",
            f"ass={len(confirmed)} yolo_n={len(dets)}",
            cam_note,
            yolo_note,
            f"occ={count}({occ_mode})",
            stale_s,
            "trails=on" if show_trails else "trails=off",
            "flip=on" if getattr(args, "flip", True) else "flip=off",
        ]
        fps = fpsm.tick()
        canvas = R.compose(left, right, status, fps=fps if fps else None)
        last_frame = canvas

        if not headless:
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("t"), ord("T")):
                show_trails = not show_trails
            if key in (ord("r"), ord("R")):
                trails.clear()
            if key in (ord("u"), ord("U")):
                show_unconfirmed = not show_unconfirmed
            if key in (ord("+"), ord("=")):
                conf = min(0.95, conf + 0.05)
                detector.set_conf(conf)
            if key in (ord("-"), ord("_")):
                conf = max(0.05, conf - 0.05)
                detector.set_conf(conf)
        else:
            time.sleep(0.01)

        if t_end is not None and time.time() >= t_end:
            break
        if args.smoke and smoke_frames is None and smoke_i > 0:
            # jsonl-driven smoke without synthetic frames list uses timed exit
            pass

    if smoke_out and last_frame is not None:
        os.makedirs(os.path.dirname(os.path.abspath(smoke_out)) or ".", exist_ok=True)
        cv2.imwrite(smoke_out, last_frame)
        print(f"wrote {smoke_out} shape={last_frame.shape}")

    if not headless:
        try:
            cv2.destroyWindow(WINDOW)
        except Exception:
            pass
    return 0


def build_smoke_frames(link, n=4):
    """Synthetic thermal greyscale with blobs near projected track centers."""
    import numpy as np

    frames = []
    cam, ext = _build_cam_ext()
    # Seed frames from current + advance via apply if needed — caller feeds link
    tracks = link.tracks()
    blobs = []
    if cam and ext and P is not None:
        for t in tracks:
            b = P.project_track(t, cam, ext, 0.50)
            if not b or not b.get("in_view"):
                continue
            x1, y1, x2, y2 = b["box"]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            rw, rh = max(6, int((x2 - x1) / 2)), max(8, int((y2 - y1) / 2))
            blobs.append((cx, cy, rw, rh, 200))
    if not blobs:
        blobs = [(80, 60, 12, 20, 200), (110, 55, 10, 18, 190)]
    for i in range(n):
        # Slight jitter
        jb = [(x + (i % 3) - 1, y, rw, rh, temp) for (x, y, rw, rh, temp) in blobs]
        frames.append(synthetic_thermal(120, 160, jb))
    return frames


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Vic dual-view fusion UI")
    p.add_argument(
        "--tracks-udp", default="0.0.0.0:9100",
        help="UDP bind for v6.12 export (default 0.0.0.0:9100). "
             "Must match A's --export-tracks-udp port; 8100≠9100 is a common mismatch.",
    )
    p.add_argument("--tracks-jsonl", default=None,
                   help="Follow v6.12 JSONL export")
    p.add_argument("--device", type=int, default=1,
                   help="OpenCV camera index (default 1 = PureThermal on many PCs)")
    p.add_argument("--no-camera", action="store_true",
                   help="Radar-only dual view (thermal pane blank/synthetic)")
    p.add_argument("--weights", default=None,
                   help="YOLO .pt path (optional)")
    p.add_argument("--conf", type=float, default=0.37)
    p.add_argument("--iou-min", type=float, default=0.15)
    p.add_argument("--center-max", type=float, default=40.0,
                   help="Center-match fallback max distance in 160x120 px (0=off)")
    p.add_argument("--body-width", type=float, default=0.50)
    p.add_argument("--pitch", type=float, default=0.0)
    p.add_argument("--tz", type=float, default=-0.08)
    p.add_argument("--flip", dest="flip", action="store_true", default=True,
                   help="Vertically flip thermal (default ON for Lepton mount)")
    p.add_argument("--no-flip", dest="flip", action="store_false",
                   help="Disable thermal vertical flip")
    p.add_argument("--boundary-box", default=None,
                   help="minX maxX minY maxY minZ maxZ (metres); "
                        "default matches my_occupancy.cfg")
    p.add_argument("--sensor-height", type=float, default=2.4,
                   help="Sensor mount height for 3D pane marker (m)")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Synthetic thermal + JSONL/UDP; no real camera")
    p.add_argument("--smoke-out", default=None,
                   help="Write final composite PNG (implies headless)")
    p.add_argument("--smoke-seconds", type=float, default=1.5)
    return p.parse_args(argv)


def _parse_boundary(s):
    if not s:
        return None
    parts = [float(x) for x in str(s).replace(",", " ").split()]
    if len(parts) != 6:
        raise SystemExit(
            f"--boundary-box needs 6 floats (minX maxX minY maxY minZ maxZ), got {parts!r}"
        )
    return {
        "minX": parts[0], "maxX": parts[1],
        "minY": parts[2], "maxY": parts[3],
        "minZ": parts[4], "maxZ": parts[5],
    }


def main(argv=None):
    args = parse_args(argv)
    args.boundary = _parse_boundary(args.boundary_box)

    if args.smoke and not args.tracks_jsonl:
        args.tracks_jsonl = os.path.join(HERE, "samples", "synthetic_tracks.jsonl")
    # Headless JSONL smoke: do not bind the live UDP default unless asked.
    if args.smoke and args.tracks_jsonl and args.tracks_udp == "0.0.0.0:9100":
        # Only drop default if caller did not also pass an explicit non-default.
        # argparse cannot tell "default" vs "explicit same value"; smoke uses JSONL.
        args.tracks_udp = None

    # Empty string disables the default UDP bind (JSONL-only / smoke).
    if args.tracks_udp is not None and str(args.tracks_udp).strip() == "":
        args.tracks_udp = None
    if not args.tracks_udp and not args.tracks_jsonl:
        print("Need --tracks-udp and/or --tracks-jsonl (or --smoke)", file=sys.stderr)
        return 2

    link = TracksRadarLink(
        jsonl=args.tracks_jsonl,
        udp=args.tracks_udp,
        jsonl_from_start=bool(args.smoke),
        poll_s=0.02,
    )
    link.start()

    # Wait briefly for first frame when using JSONL smoke
    if args.tracks_jsonl:
        t0 = time.time()
        while link.frames < 1 and time.time() - t0 < 2.0:
            time.sleep(0.02)

    detector = Detector(weights=args.weights, conf=args.conf)
    if args.weights and not detector.ready:
        print(f"[vic-ui] YOLO unavailable: {detector.err} (continuing radar-only count)")

    cap = None
    smoke_frames = None
    if args.smoke or args.no_camera:
        smoke_frames = build_smoke_frames(link, n=8)
        print(f"[vic-ui] smoke thermal frames={len(smoke_frames)} "
              f"tracks={len(link.tracks())} project={_HAS_PROJECT}")
    else:
        cap, info = open_camera(device=args.device)
        if cap is None:
            print(f"[vic-ui] camera open failed: {info}; use --smoke or --no-camera",
                  file=sys.stderr)
            # Still allow radar-only UI
            smoke_frames = build_smoke_frames(link, n=1)
        else:
            print(f"[vic-ui] camera ok: {info}")

    if not _HAS_PROJECT:
        print(f"[vic-ui] warning: project.py missing ({_PROJECT_ERR!r})")

    try:
        rc = run_loop(
            args, link, cap, detector,
            smoke_frames=smoke_frames,
            smoke_out=args.smoke_out,
        )
    finally:
        link.stop()
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
