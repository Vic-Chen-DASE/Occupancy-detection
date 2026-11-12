#!/usr/bin/env python3
"""
Wire v6.12 track export into Haneef Fusion without editing upstream much.

Modes:
  python fuse_vic.py --self-test
  python fuse_vic.py --smoke-jsonl [path]
  python fuse_vic.py --install-check
  python fuse_vic.py --live ...   # forwards to Haneef fuse.py after patching

How the drop-in works
---------------------
integrated_launcher does `from radar_link import RadarLink` after inserting
Fusion/ on sys.path. If `radar_link` is **already** in sys.modules, Python
reuses our module. fuse_vic --live does that, then calls Fusion/fuse.py live
with the same argv shape.

Env (set automatically from flags):
  VIC_TRACKS_UDP=0.0.0.0:9100
  VIC_TRACKS_JSONL=/path/tracks.jsonl

Typical two-process run on Windows (VIC01):
  # terminal A — radar GUI + export
  python live_32d_track_v6.py --data-port COM6 --cli-port COM5 \\
    --cfg my_occupancy.cfg --export-tracks tracks.jsonl \\
    --export-tracks-udp 127.0.0.1:9100

  # terminal B — thermal fusion, radar half = this bridge
  set VIC_TRACKS_UDP=0.0.0.0:9100
  python fuse_vic.py --live --tracks-udp 0.0.0.0:9100 --cli COMx --data COMy ...

Blockers for full live: Thermal venv (YOLO, lepton), PureThermal device,
Windows COM ports for the *thermal* stack. The bridge itself needs neither.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def install_radar_link_module():
    """Make `import radar_link` resolve to this bridge (before Fusion path wins)."""
    import importlib
    import radar_link as bridge_rl  # fusion_bridge/radar_link.py
    sys.modules["radar_link"] = bridge_rl
    # Also ensure vendor/project is importable as `project` for IL.
    vendor = os.path.join(HERE, "vendor")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    return bridge_rl


def cmd_self_test():
    from radar_link_from_tracks import self_test as core
    from radar_link import self_test as dropin
    rc1 = core()
    # dropin self_test re-runs core; call constructor check only once more lightly
    link_mod = __import__("radar_link")
    link = link_mod.RadarLink("/dev/null", "/dev/null", "c.cfg")
    assert link.mode == "vic"
    print("fuse_vic drop-in wiring ok")
    return rc1


def cmd_smoke_jsonl(path, udp_port=None):
    """Follow a JSONL (and optional UDP) without Thermal/Qt."""
    from radar_link_from_tracks import TracksRadarLink, P, _HAS_PROJECT

    if not os.path.exists(path):
        print(f"missing JSONL: {path}", file=sys.stderr)
        return 1
    link = TracksRadarLink(jsonl=path, udp=udp_port, jsonl_from_start=True)
    link.start()
    # Give the follower a moment to read existing lines.
    deadline = time.time() + 2.0
    while time.time() < deadline and link.frames < 1:
        time.sleep(0.05)
    print(f"frames={link.frames} n={len(link.tracks())} gen={link.generation}")
    for t in link.tracks():
        print(f"  id={t['id']} xy=({t['x']:.2f},{t['y']:.2f}) "
              f"range={t['range']:.2f} conf={t['conf']}")
    if _HAS_PROJECT:
        cam = P.Camera()
        ext = P.Extrinsics(tz=-0.08)
        boxes = link.projected(cam, ext)
        print(f"projected={len(boxes)} "
              f"in_view={sum(1 for b in boxes if b['in_view'])}")
    else:
        print("projected skipped (no project.py)")
    link.stop()
    return 0 if link.frames >= 1 else 2


def cmd_install_check():
    install_radar_link_module()
    import radar_link
    print(f"radar_link -> {radar_link.__file__}")
    print(f"RadarLink  -> {radar_link.RadarLink}")
    try:
        import project as P
        print(f"project    -> {P.__file__}")
    except Exception as e:  # noqa: BLE001
        print(f"project    FAIL: {e}")
        return 1
    return 0


def cmd_live(args, extra):
    """
    Patch sys.modules['radar_link'], then hand off to Haneef fuse.live /
    integrated_launcher. Requires the Occupancy-Detection-Program tree.
    """
    jsonl = args.tracks_jsonl
    udp = args.tracks_udp
    if jsonl:
        os.environ["VIC_TRACKS_JSONL"] = os.path.abspath(jsonl)
    if udp:
        os.environ["VIC_TRACKS_UDP"] = str(udp)
    if not os.environ.get("VIC_TRACKS_JSONL") and not os.environ.get("VIC_TRACKS_UDP"):
        print("Need --tracks-udp and/or --tracks-jsonl (or env VIC_TRACKS_*)",
              file=sys.stderr)
        return 2

    install_radar_link_module()

    odp = args.odp_root
    if not odp:
        # Common local layouts
        candidates = [
            os.environ.get("ODP_ROOT"),
            os.path.join(os.path.expanduser("~"), "Occupancy-Detection-Program"),
            r"C:\Users\13055\Occupancy-Detection-Program",
            "/tmp/odp_src",
        ]
        for c in candidates:
            if c and os.path.isdir(c):
                # Prefer a tree that has Thermal/integrated_launcher.py
                if os.path.isfile(os.path.join(c, "Thermal", "integrated_launcher.py")) \
                        or os.path.isfile(os.path.join(c, "Fusion", "fuse.py")):
                    odp = c
                    break
    if not odp or not os.path.isdir(odp):
        print(
            "Haneef repo root not found. Pass --odp-root /path/to/"
            "Occupancy-Detection-Program (needs Thermal/ + Fusion/).\n"
            "Bridge self-test does not need it; full --live does.",
            file=sys.stderr,
        )
        return 3

    fusion_dir = os.path.join(odp, "Fusion")
    thermal_dir = os.path.join(odp, "Thermal")
    mmwave_dir = os.path.join(odp, "mmWave")
    for p in (mmwave_dir, thermal_dir, fusion_dir, HERE):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    # Re-assert our radar_link after path inserts.
    install_radar_link_module()

    fuse_py = os.path.join(fusion_dir, "fuse.py")
    if not os.path.isfile(fuse_py):
        print(f"missing {fuse_py}", file=sys.stderr)
        return 3

    # Import fuse from that tree and call live() with a reconstructed args-like
    # namespace. Easier: build argv and exec fuse main with --live.
    argv = ["fuse.py", "--live"]
    # Map our flags into fuse flags; unknown go to extra.
    if args.cli:
        argv += ["--cli", args.cli]
    if args.data:
        argv += ["--data", args.data]
    if args.close_cfg:
        argv += ["--close-cfg", args.close_cfg]
    if args.far_cfg:
        argv += ["--far-cfg", args.far_cfg]
    if args.adaptive:
        argv += ["--adaptive"]
    if args.weights:
        argv += ["--weights", args.weights]
    if args.conf is not None:
        argv += ["--conf", str(args.conf)]
    if args.pitch is not None:
        argv += ["--pitch", str(args.pitch)]
    if args.yaw is not None:
        argv += ["--yaw", str(args.yaw)]
    if args.tz is not None:
        argv += ["--tz", str(args.tz)]
    if args.assist_alpha is not None:
        argv += ["--assist-alpha", str(args.assist_alpha)]
    if args.no_thermal_veto:
        argv += ["--no-thermal-veto"]
    argv += list(extra)

    print("FLUXNET fusion (vic bridge)")
    print(f"  odp_root  {odp}")
    print(f"  tracks    UDP={os.environ.get('VIC_TRACKS_UDP')}  "
          f"JSONL={os.environ.get('VIC_TRACKS_JSONL')}")
    print(f"  radar_link is {sys.modules['radar_link'].__file__}")
    print(f"  forwarding: {' '.join(argv[1:])}\n")

    # Run fuse.main with patched argv.
    import runpy
    sys.argv = argv
    # Ensure cwd so relative cfg paths in fuse defaults still work somewhat.
    os.chdir(fusion_dir)
    runpy.run_path(fuse_py, run_name="__main__")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="v6.12 -> Haneef Fusion RadarLink bridge")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke-jsonl", nargs="?", const="",
                    metavar="PATH",
                    help="Follow JSONL and print tracks "
                         "(default: samples/synthetic_tracks.jsonl)")
    ap.add_argument("--smoke-udp", type=int, default=None, metavar="PORT",
                    help="Also bind UDP during --smoke-jsonl")
    ap.add_argument("--install-check", action="store_true",
                    help="Show which radar_link/project would be imported")
    ap.add_argument("--live", action="store_true",
                    help="Patch radar_link and run Haneef fuse.py --live")

    ap.add_argument("--tracks-jsonl", default=None)
    ap.add_argument("--tracks-udp", default=None,
                    help="Bind host:port or port (e.g. 0.0.0.0:9100)")
    ap.add_argument("--odp-root", default=None,
                    help="Path to Haneef Occupancy-Detection-Program")

    # Forwarded live flags (subset)
    ap.add_argument("--cli", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--close-cfg", default=None)
    ap.add_argument("--far-cfg", default=None)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--pitch", type=float, default=None)
    ap.add_argument("--yaw", type=float, default=None)
    ap.add_argument("--tz", type=float, default=None)
    ap.add_argument("--assist-alpha", type=float, default=None)
    ap.add_argument("--no-thermal-veto", action="store_true")

    args, extra = ap.parse_known_args(argv)

    if args.self_test:
        return cmd_self_test()
    if args.install_check:
        return cmd_install_check()
    if args.smoke_jsonl is not None:
        path = args.smoke_jsonl or os.path.join(
            HERE, "samples", "synthetic_tracks.jsonl")
        return cmd_smoke_jsonl(path, udp_port=args.smoke_udp)
    if args.live:
        return cmd_live(args, extra)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
