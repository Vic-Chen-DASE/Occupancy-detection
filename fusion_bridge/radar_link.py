#!/usr/bin/env python3
"""
Drop-in replacement for Haneef Fusion/radar_link.py.

Thermal/integrated_launcher.py does:

    sys.path.insert(0, .../Fusion)
    from radar_link import RadarLink
    link = RadarLink(cli, data, close_cfg, far_cfg, adaptive=..., on_event=...)
    link.start()

Install this module as `radar_link` **before** that import (see fuse_vic.py),
or copy/symlink this file over Fusion/radar_link.py. Serial CLI/data/cfg args
are accepted and ignored; tracks come from v6.12 export:

    VIC_TRACKS_UDP=0.0.0.0:9100          # preferred live
    VIC_TRACKS_JSONL=path/to/tracks.jsonl

Optional kwargs (if a patched launcher passes them): tracks_udp=, tracks_jsonl=.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from radar_link_from_tracks import TracksRadarLink  # noqa: E402


def _env_source():
    jsonl = os.environ.get("VIC_TRACKS_JSONL") or None
    udp = os.environ.get("VIC_TRACKS_UDP") or None
    if jsonl:
        jsonl = os.path.expanduser(jsonl)
    return jsonl, udp


class RadarLink(TracksRadarLink):
    """
    Same constructor signature as Haneef RadarLink so integrated_launcher
    needs no edits. Adaptive cfg switching is intentionally disabled — v6.12
    owns identity; bumping generation here would fight the lock boxes.
    """

    def __init__(self, cli, data, close_cfg, far_cfg=None, adaptive=False,
                 baud=921600, cli_baud=115200, ti_common=None,
                 near_min=1.0, far_enter=6.0, far_exit=5.0, far_count=2,
                 hold=3.0, dwell=30.0, metric="ground", body_width=0.50,
                 on_event=None, tracks_jsonl=None, tracks_udp=None,
                 jsonl_from_start=False):
        env_jsonl, env_udp = _env_source()
        jsonl = tracks_jsonl or env_jsonl
        udp = tracks_udp or env_udp
        # Stash ignored Haneef args for status/debug only.
        self.cli, self.data = cli, data
        self.close_cfg = close_cfg
        self.far_cfg = far_cfg
        self._requested_adaptive = bool(adaptive and far_cfg is not None)
        super().__init__(
            jsonl=jsonl,
            udp=udp,
            body_width=body_width,
            on_event=on_event,
            jsonl_from_start=jsonl_from_start,
        )
        if self._requested_adaptive:
            self.on_event(
                "[vic-radar] adaptive cfg ignored — v6.12 lock boxes own IDs")

    def status(self):
        st = super().status()
        st["haneef_cli"] = self.cli
        st["haneef_data"] = self.data
        st["requested_adaptive"] = self._requested_adaptive
        return st


def self_test():
    # Constructor must accept Haneef call shape even with no source yet.
    link = RadarLink("/dev/null", "/dev/null", "close.cfg", "far.cfg",
                     adaptive=True)
    assert link.mode == "vic"
    assert link.generation == 0
    st = link.status()
    assert "generation" in st and st["requested_adaptive"] is True
    print("drop-in RadarLink constructor ok")
    # Re-run track self-test from the core module.
    from radar_link_from_tracks import self_test as core
    return core()


if __name__ == "__main__":
    sys.exit(self_test())
