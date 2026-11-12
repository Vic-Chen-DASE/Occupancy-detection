#!/usr/bin/env python3
"""
RadarLink-compatible backend that consumes v6.12 locked-box track snapshots.

Sources (either or both):
  * UDP datagrams — preferred for live (latest frame wins)
  * JSONL file   — tail follow, or last line on open

Public surface matches Haneef Fusion/radar_link.RadarLink enough for
Thermal/integrated_launcher.py:

    link.start() / link.stop() / link.alive
    link.tracks() -> list[dict]   # id,x,y,z,vx,vy,vz,...
    link.projected(cam, ext)      # via mmWave.project.project_track
    link.status() / link.generation / link.frames / ...

Identity truth is the v6.12 locked **box** id (firmware tid or local 1000+),
not board TLV-1010 alone. generation is taken from each snapshot frame.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "vendor")
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

try:
    import project as P  # Haneef mmWave/project.py (vendored)
    _HAS_PROJECT = True
except Exception as _e:  # noqa: BLE001 — optional dep for projected()
    P = None
    _HAS_PROJECT = False
    _PROJECT_ERR = _e


def _parse_udp_bind(spec):
    """
    Accept:
      9100
      "9100"
      "0.0.0.0:9100"
      ("0.0.0.0", 9100)
    -> (host, port) to bind.
    """
    if spec is None:
        return None
    if isinstance(spec, tuple) and len(spec) == 2:
        host, port = spec
        return str(host), int(port)
    if isinstance(spec, int):
        return "0.0.0.0", int(spec)
    s = str(spec).strip()
    if not s:
        return None
    if ":" in s:
        host, _, port_s = s.rpartition(":")
        return (host or "0.0.0.0"), int(port_s)
    return "0.0.0.0", int(s)


def _normalize_track(t):
    """Keep Haneef summarise field names 1:1; coerce types lightly."""
    if not isinstance(t, dict) or "id" not in t:
        return None
    try:
        out = {
            "id": int(t["id"]),
            "x": float(t.get("x", 0.0)),
            "y": float(t.get("y", 0.0)),
            "z": float(t.get("z", 0.0)),
            "vx": float(t.get("vx", 0.0)),
            "vy": float(t.get("vy", 0.0)),
            "vz": float(t.get("vz", 0.0)),
            "speed": float(t["speed"]) if "speed" in t else None,
            "range": float(t["range"]) if "range" in t else None,
            "conf": float(t.get("conf", 1.0)),
            "z_min": t.get("z_min", 0.0),
            "z_max": t.get("z_max", 1.7),
        }
    except (TypeError, ValueError):
        return None
    if out["speed"] is None:
        out["speed"] = (out["vx"] ** 2 + out["vy"] ** 2) ** 0.5
    if out["range"] is None:
        out["range"] = (out["x"] ** 2 + out["y"] ** 2) ** 0.5
    if out["z_min"] is not None:
        try:
            out["z_min"] = float(out["z_min"])
        except (TypeError, ValueError):
            out["z_min"] = 0.0
    if out["z_max"] is not None:
        try:
            out["z_max"] = float(out["z_max"])
        except (TypeError, ValueError):
            out["z_max"] = 1.7
    return out


def parse_frame_payload(raw):
    """
    Bytes or str -> frame dict {frame, t, generation, tracks} or None.
    Tolerates a trailing newline; one JSON object per datagram / line.
    """
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    s = raw.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "tracks" not in obj:
        return None
    tracks = []
    for t in obj.get("tracks") or []:
        nt = _normalize_track(t)
        if nt is not None:
            tracks.append(nt)
    points = obj.get("points")
    if points is not None and not isinstance(points, list):
        points = None
    return {
        "frame": int(obj.get("frame", -1) or -1),
        "t": float(obj.get("t") or time.time()),
        "generation": int(obj.get("generation", 0) or 0),
        "tracks": tracks,
        "points": points,
    }


class TracksRadarLink:
    """
    Drop-in radar half for Fusion: no serial, no TI parser, no adaptive cfg.

    Preferred live path is UDP (v6.12 --export-tracks-udp). JSONL is fine for
    replay and for smoke tests without a radar.
    """

    def __init__(self, jsonl=None, udp=None, body_width=0.50, on_event=None,
                 jsonl_from_start=False, poll_s=0.02):
        self.jsonl = jsonl
        self.udp_bind = _parse_udp_bind(udp)
        self.body_width = float(body_width)
        self.on_event = on_event or (lambda s: None)
        self.jsonl_from_start = bool(jsonl_from_start)
        self.poll_s = float(poll_s)

        self._lock = threading.Lock()
        self._tracks = []
        self._points = []
        self._stop = False
        self._threads = []
        self.mode = "vic"          # not CLOSE/FAR — host-side lock boxes
        self.generation = 0
        self.frames = 0
        self.err = None
        self.switching = False
        self.switches = 0
        self.last_blackout = 0.0
        self.t_last = 0.0
        self.adaptive = False
        self._frame_num = -1
        self._project_ok = _HAS_PROJECT

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        if not self.jsonl and not self.udp_bind:
            self.err = ("no track source: pass jsonl= and/or udp=, "
                        "or set VIC_TRACKS_JSONL / VIC_TRACKS_UDP")
            self.on_event(f"[vic-radar] {self.err}")
            return self
        self._stop = False
        if self.udp_bind:
            th = threading.Thread(target=self._run_udp, daemon=True,
                                  name="vic-tracks-udp")
            th.start()
            self._threads.append(th)
            self.on_event(
                f"[vic-radar] UDP listen {self.udp_bind[0]}:{self.udp_bind[1]}")
        if self.jsonl:
            th = threading.Thread(target=self._run_jsonl, daemon=True,
                                  name="vic-tracks-jsonl")
            th.start()
            self._threads.append(th)
            self.on_event(f"[vic-radar] JSONL follow {self.jsonl}")
        return self

    def stop(self):
        self._stop = True

    @property
    def alive(self):
        return any(t.is_alive() for t in self._threads)

    # -- what the launcher asks for ---------------------------------------

    def tracks(self):
        with self._lock:
            return list(self._tracks)

    def points(self):
        """Optional point cloud from the latest frame (list or empty)."""
        with self._lock:
            return list(self._points)

    def projected(self, cam, ext):
        """
        Same contract as Haneef RadarLink.projected: every track returned,
        with in_view set by project_track (do not filter here).
        """
        if not self._project_ok or P is None:
            # Soft stub: empty list so Fusion still runs; status carries err.
            if self.err is None:
                self.err = (
                    "projected() needs mmWave/project.py on PYTHONPATH "
                    f"(vendor load failed: {_PROJECT_ERR!r})"
                )
            return []
        out = []
        for t in self.tracks():
            b = P.project_track(t, cam, ext, self.body_width)
            if b:
                out.append(b)
        return out

    def status(self):
        return {
            "mode": self.mode,
            "frames": self.frames,
            "err": self.err,
            "switching": self.switching,
            "switches": self.switches,
            "generation": self.generation,
            "n": len(self.tracks()),
            "reason": "vic-lock-boxes",
            "adaptive": self.adaptive,
            "blackout": self.last_blackout,
            "stale_s": (time.time() - self.t_last) if self.t_last else None,
            "frame": self._frame_num,
            "project": self._project_ok,
            "source": {
                "jsonl": self.jsonl,
                "udp": (f"{self.udp_bind[0]}:{self.udp_bind[1]}"
                        if self.udp_bind else None),
            },
        }

    # -- apply one parsed frame -------------------------------------------

    def _apply_frame(self, rec):
        if rec is None:
            return
        with self._lock:
            self._tracks = list(rec["tracks"])
            pts = rec.get("points")
            self._points = list(pts) if isinstance(pts, list) else []
            gen = int(rec.get("generation", 0) or 0)
            if gen != self.generation:
                self.generation = gen
            self._frame_num = int(rec.get("frame", -1) or -1)
            self.frames += 1
            self.t_last = time.time()

    def apply_raw(self, raw):
        """Test hook: feed one JSON line / datagram without sockets."""
        self._apply_frame(parse_frame_payload(raw))

    def apply_dict(self, rec):
        """Test hook: feed an already-parsed frame dict."""
        if not isinstance(rec, dict) or "tracks" not in rec:
            return
        tracks = []
        for t in rec.get("tracks") or []:
            nt = _normalize_track(t)
            if nt is not None:
                tracks.append(nt)
        pts = rec.get("points")
        self._apply_frame({
            "frame": int(rec.get("frame", -1) or -1),
            "t": float(rec.get("t") or time.time()),
            "generation": int(rec.get("generation", 0) or 0),
            "tracks": tracks,
            "points": pts if isinstance(pts, list) else None,
        })

    # -- readers -----------------------------------------------------------

    def _run_udp(self):
        host, port = self.udp_bind
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.settimeout(0.4)
        except Exception as e:  # noqa: BLE001
            self.err = f"UDP bind failed: {type(e).__name__}: {e}"
            self.on_event(f"[vic-radar] {self.err}")
            return
        try:
            while not self._stop:
                try:
                    data, _addr = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except Exception as e:  # noqa: BLE001
                    self.err = f"UDP recv: {type(e).__name__}: {e}"
                    break
                rec = parse_frame_payload(data)
                if rec is not None:
                    self._apply_frame(rec)
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def _run_jsonl(self):
        path = self.jsonl
        # Wait for file to appear (exporter may start slightly later).
        while not self._stop and not os.path.exists(path):
            time.sleep(self.poll_s)
        if self._stop:
            return
        pos = 0
        buf = ""
        try:
            if not self.jsonl_from_start:
                # Seed from last complete line, then follow.
                try:
                    with open(path, "rb") as f:
                        f.seek(0, os.SEEK_END)
                        size = f.tell()
                        back = min(size, 1 << 16)
                        f.seek(size - back)
                        chunk = f.read().decode("utf-8", errors="replace")
                        pos = size
                    lines = [ln for ln in chunk.splitlines() if ln.strip()]
                    if lines:
                        rec = parse_frame_payload(lines[-1])
                        if rec is not None:
                            self._apply_frame(rec)
                except Exception as e:  # noqa: BLE001
                    self.err = f"JSONL seed: {type(e).__name__}: {e}"
            else:
                pos = 0

            while not self._stop:
                try:
                    size = os.path.getsize(path)
                except OSError:
                    time.sleep(self.poll_s)
                    continue
                if size < pos:
                    # Truncated / rotated.
                    pos = 0
                    buf = ""
                if size == pos:
                    time.sleep(self.poll_s)
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        f.seek(pos)
                        data = f.read()
                        pos = f.tell()
                except Exception as e:  # noqa: BLE001
                    self.err = f"JSONL read: {type(e).__name__}: {e}"
                    time.sleep(self.poll_s)
                    continue
                buf += data
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    rec = parse_frame_payload(line)
                    if rec is not None:
                        self._apply_frame(rec)
        except Exception as e:  # noqa: BLE001
            self.err = f"JSONL follow: {type(e).__name__}: {e}"
            self.on_event(f"[vic-radar] {self.err}")


def self_test():
    """No radio, no Qt — synthetic frames only."""
    ok = True

    def check(label, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {label} {extra}")

    link = TracksRadarLink()
    check("starts generation 0", link.generation == 0)
    st = link.status()
    check("status keys",
          {"mode", "frames", "n", "switching", "generation"} <= set(st))

    link.apply_dict({
        "frame": 1, "t": 1.0, "generation": 0,
        "tracks": [
            {"id": 5, "x": 0.0, "y": 4.0, "z": 0.9,
             "vx": 0.0, "vy": -0.8, "vz": 0.0,
             "speed": 0.8, "range": 4.0, "conf": 1.0,
             "z_min": 0.0, "z_max": 1.7},
            {"id": 1000, "x": 1.0, "y": 4.0, "z": 0.9,
             "vx": 0.0, "vy": 0.0, "vz": 0.0,
             "conf": 1.0, "z_min": 0.0, "z_max": 1.7},
        ],
    })
    tr = link.tracks()
    check("two locked-box ids", len(tr) == 2, len(tr))
    check("local 1000+ kept", {t["id"] for t in tr} == {5, 1000})
    check("frames++", link.frames == 1)

    link.apply_raw(
        '{"frame":2,"t":2.0,"generation":3,"tracks":'
        '[{"id":5,"x":0.1,"y":3.9,"z":0.9,"vx":0,"vy":0,"vz":0,'
        '"conf":1,"z_min":0,"z_max":1.7}]}'
    )
    check("generation from frame", link.generation == 3, link.generation)
    check("latest tracks only", len(link.tracks()) == 1)

    if _HAS_PROJECT:
        cam = P.Camera()
        ext = P.Extrinsics(tz=-0.08)
        link.apply_dict({
            "frame": 3, "generation": 3,
            "tracks": [
                {"id": 1, "x": 0.0, "y": 4.0, "z": 0.9,
                 "vx": 0.0, "vy": -0.8, "vz": 0.0,
                 "z_min": 0.0, "z_max": 1.7},
                {"id": 2, "x": 9.0, "y": 1.0, "z": 0.9,
                 "vx": 0.0, "vy": 0.0, "vz": 0.0,
                 "z_min": 0.0, "z_max": 1.7},
            ],
        })
        b = link.projected(cam, ext)
        check("projected returns both", len(b) == 2, len(b))
        check("in_view flags differ",
              sorted(x["in_view"] for x in b) == [False, True],
              [x["in_view"] for x in b])
        check("vel carried", b[0]["vel"] == [0.0, -0.8, 0.0])
    else:
        check("project.py available", False, str(_PROJECT_ERR))

    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(self_test())
