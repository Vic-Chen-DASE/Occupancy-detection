#!/usr/bin/env python3
"""Smoke test: synthetic JSONL + optional UDP round-trip. No Qt / no radar."""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from radar_link_from_tracks import TracksRadarLink, P, _HAS_PROJECT  # noqa: E402


SAMPLE = os.path.join(HERE, "samples", "synthetic_tracks.jsonl")


def test_jsonl_follow():
    link = TracksRadarLink(jsonl=SAMPLE, jsonl_from_start=True, poll_s=0.01)
    link.start()
    t0 = time.time()
    while link.frames < 4 and time.time() - t0 < 2.0:
        time.sleep(0.02)
    link.stop()
    assert link.frames >= 4, link.frames
    assert link.generation == 1, link.generation
    ids = {t["id"] for t in link.tracks()}
    # Last frame only has id 5
    assert ids == {5}, ids
    print("ok  jsonl follow -> last frame id=5 generation=1")


def test_udp_roundtrip():
    # Bind ephemeral port
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    link = TracksRadarLink(udp=("127.0.0.1", port))
    link.start()
    time.sleep(0.05)
    payload = {
        "frame": 99, "t": time.time(), "generation": 7,
        "tracks": [
            {"id": 1000, "x": 1.0, "y": 2.0, "z": 0.9,
             "vx": 0, "vy": 0, "vz": 0, "conf": 1.0,
             "z_min": 0.0, "z_max": 1.7},
        ],
    }
    line = json.dumps(payload).encode("utf-8")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(line, ("127.0.0.1", port))
    s.close()
    t0 = time.time()
    while link.frames < 1 and time.time() - t0 < 2.0:
        time.sleep(0.02)
    link.stop()
    assert link.frames >= 1, link.frames
    assert link.generation == 7
    assert link.tracks()[0]["id"] == 1000
    print("ok  udp round-trip id=1000 generation=7")


def test_projected():
    assert _HAS_PROJECT, "vendor/project.py missing"
    link = TracksRadarLink()
    link.apply_dict({
        "frame": 1, "generation": 0,
        "tracks": [
            {"id": 5, "x": 0.0, "y": 4.0, "z": 0.9,
             "vx": 0.0, "vy": -0.8, "vz": 0.0,
             "z_min": 0.0, "z_max": 1.7},
            {"id": 1000, "x": 9.0, "y": 1.0, "z": 0.9,
             "vx": 0, "vy": 0, "vz": 0,
             "z_min": 0.0, "z_max": 1.7},
        ],
    })
    cam = P.Camera()
    ext = P.Extrinsics(tz=-0.08)
    boxes = link.projected(cam, ext)
    assert len(boxes) == 2
    assert sorted(b["in_view"] for b in boxes) == [False, True]
    print("ok  projected() in_view split")


def test_append_while_following():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "live.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "frame": 1, "t": 1.0, "generation": 0,
                "tracks": [{"id": 5, "x": 0, "y": 3, "z": 0.9,
                            "vx": 0, "vy": 0, "vz": 0, "conf": 1,
                            "z_min": 0, "z_max": 1.7}],
            }) + "\n")
        link = TracksRadarLink(jsonl=path, jsonl_from_start=False, poll_s=0.01)
        link.start()
        t0 = time.time()
        while link.frames < 1 and time.time() - t0 < 2.0:
            time.sleep(0.02)
        assert link.frames >= 1
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "frame": 2, "t": 2.0, "generation": 2,
                "tracks": [{"id": 1000, "x": 1, "y": 3, "z": 0.9,
                            "vx": 0, "vy": 0, "vz": 0, "conf": 1,
                            "z_min": 0, "z_max": 1.7}],
            }) + "\n")
        t0 = time.time()
        while link.frames < 2 and time.time() - t0 < 2.0:
            time.sleep(0.02)
        link.stop()
        assert link.frames >= 2, link.frames
        assert link.tracks()[0]["id"] == 1000
        assert link.generation == 2
        print("ok  append-while-follow")


def main():
    test_jsonl_follow()
    test_udp_roundtrip()
    test_projected()
    test_append_while_following()
    # Also run module self_tests
    from radar_link_from_tracks import self_test as core
    rc = core()
    if rc != 0:
        return rc
    print("ALL SMOKE PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
