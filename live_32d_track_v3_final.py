#!/usr/bin/env python3
# =============================================================================
# live_3d_radar_v10.py
# -----------------------------------------------------------------------------
# Real-time 3D mmWave Radar Visualizer — PyQtGraph + OpenGL edition
# (TI IWR6843AOP, 3D People Tracking demo)
#
# Upgrades this brings over live_3d_radar_v9.py (Matplotlib):
#   - PySide6/PySide2 + pyqtgraph.opengl GLViewWidget for a fast, GPU-backed
#     3D view, side-by-side with a pyqtgraph 2D top-down PlotWidget — the
#     same dual-view layout as TI's own Industrial Visualizer.
#   - QThread + Signal producer/consumer parsing thread instead of a raw
#     Python thread + shared-lock polling loop: the parser thread owns the
#     serial port and all TI-parser calls, and pushes ready-to-render
#     snapshots to the GUI thread via a queued Qt signal (no polling, no
#     race conditions on Qt widgets).
#   - Hybrid TLV parsing: point cloud (TLV 1020, compressed spherical) is
#     read via the official `parseStandardFrame` (TI's own parser already
#     implements the correct '2bh2H' per-point struct for that TLV). Track
#     list (TLV 1010) is parsed *manually* straight from the raw UART frame,
#     because parseStandardFrame's `trackData` output is unreliable in
#     practice for this TLV. A robust extractor still tries the official
#     parser's output first (handling plain lists, dicts, and numpy
#     structured arrays) and falls back to the manual TLV-1010 parse
#     whenever that comes back empty.
#   - Kept from v9: CVKalman3D constant-velocity Kalman filter, frame
#     accumulation for point-cloud density, and the tilt/height coordinate
#     transform for a mounted sensor (now defaulting to 18 deg, ceiling
#     mount).
#
# -----------------------------------------------------------------------------
# CHANGELOG v10 -> v11  (debug / stability / sensitivity pass)
# -----------------------------------------------------------------------------
#  [TLV 1010 PARSING - the main fix]
#   - manual_parse_tracks() no longer assumes a single fixed struct size.
#     A new _parse_track_payload() tries several known TI target-list struct
#     layouts (3D full '<I27f'=112B, 2D '<I17f'=72B) and picks whichever one
#     evenly divides the actual TLV payload length. If NONE divide evenly
#     (firmware sends something we don't recognise), it falls back to the
#     smallest-remainder candidate and still parses the whole-number portion
#     instead of throwing the whole TLV away.
#   - Per-track sanity checks (finite values, plausible x/y/z range, plausible
#     tid) reject garbage caused by a struct-size mismatch instead of feeding
#     wild numbers into the Kalman filter.
#   - struct.unpack_from() failures now just stop *that* TLV's loop (via a
#     targeted try/except) instead of raising out of the parser.
#   - Verbose, opt-in debug logging (--debug flag) prints TLV length, which
#     struct layout was chosen, how many tracks were found, how many passed
#     sanity checks, and a hex preview of the first bytes -- exactly the
#     information needed to tune this against a new/unknown .cfg profile
#     such as a custom my_occupancy.cfg. If your device emits a track-list
#     size that isn't 112 or 72 bytes, run with --debug once, read the
#     logged payload length, and add that size to TRACK_STRUCT_CANDIDATES.
#
#  [STABILITY]
#   - _process_frame() now has its own top-level try/except so a crash deep
#     in point-cloud extraction or Kalman math can never kill the parser
#     thread; it's logged and that single frame is dropped.
#   - on_snapshot() (GUI thread) now strips non-finite (NaN/Inf) rows out of
#     the cloud and track positions before handing them to the GL widgets --
#     a NaN vertex can otherwise crash or hang some OpenGL drivers.
#   - closeEvent() now waits longer, force-terminates as an absolute last
#     resort, and QApplication.aboutToQuit is also wired to stop the thread,
#     so the "QThread: Destroyed while thread is still running" warning
#     should no longer appear regardless of how the app is closed.
#
#  [SENSITIVITY]
#   - ROI relaxed slightly (both Y and Z) so borderline/edge-of-room tracks
#     aren't filtered out before they even reach the Kalman filter.
#   - Kalman filter retuned: lower measurement noise (trust fresh detections
#     a bit more -> faster convergence), higher process noise (adapts to
#     real movement quicker), wider gate (fewer valid detections get treated
#     as "missed"), and MAX_MISSED raised so a track survives brief gaps
#     instead of being deleted and restarted (which read as "unstable" IDs).
#   - ACCUM_FRAMES raised moderately (12 -> 16) for a denser point cloud.
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
# CHANGELOG v11 -> v3 (detection volume / per-track color / stability pass)
# -----------------------------------------------------------------------------
#  [1. DETECTION VOLUME BOX]  (see also the FIX note further down for the
#   axis-alignment correction applied after this was first added)
#   - New dashed, semi-transparent wireframe box in the 3D view
#     (gl.GLLinePlotItem, mode='lines') showing the radar's configured
#     detection volume -- roughly the .cfg `boundaryBox`.
#     Configurable via --boundary-box "minX,maxX,minY,maxY,minZ,maxZ" if it
#     doesn't match your .cfg. True GL line-stippling isn't exposed by
#     pyqtgraph's GLLinePlotItem, so "dashed" is approximated by chopping
#     each edge into short dash/gap sub-segments (see _dash_segment()) --
#     visually dashed, still a single cheap static GLLinePlotItem.
#
#  [1a. FIX] Detection volume box was tilting with the sensor
#   - The box was originally defined in RAW sensor-frame coordinates and run
#     through the full tilt+height transform (make_transform), which
#     rotated it to match the sensor's tilt angle -- so it appeared tilted
#     relative to the floor grid. That's wrong: it's meant to be a fixed,
#     floor-aligned reference volume of the room (the monitored area), not
#     something seen through the sensor's tilted view.
#   - BOUNDARY_BOX_RAW was renamed/redefined as BOUNDARY_BOX_ROOM and is now
#     specified directly in ROOM coordinates (X=left/right, Y=forward/depth,
#     Z=height above the floor, Z=0 at the floor). Default: X -5..5,
#     Y 0.5..10, Z 0..2.5.
#   - A new make_height_only_transform() applies ONLY a height translation
#     (in this Z-is-already-absolute-height convention, that's an identity
#     shift) with NO tilt rotation, and is used exclusively for the boundary
#     box. make_transform() (tilt + height) is unchanged and still used for
#     the point cloud and tracks, which genuinely are captured through the
#     sensor's tilted view.
#   - Color changed to bright yellow (1.0, 1.0, 0.0, 0.7) for higher
#     visibility as a reference overlay.
#
#  [2. PER-TRACK UNIQUE COLORS]
#   - TrackArtists3D and TrackArtists2D now both take a color derived from
#     `pg.intColor(tid, hues=12)`, computed once when a track first appears
#     and reused every frame after -- so a given person keeps the same
#     color for as long as their track ID is alive, in BOTH the 3D box and
#     the 2D top-down box/trail. The trail uses the same hue at reduced
#     alpha (instead of a fixed color) so it reads as "this person's trail"
#     while staying visually distinct from the solid box outline.
#
#  [3. STABILITY]
#   - TrackArtists3D.update()/remove() and TrackArtists2D.update()/remove()
#     are now wrapped in try/except and validate their inputs (finite
#     position, non-empty history) before touching any GL/Qt item, so a
#     track that appears/disappears mid-frame or carries a momentarily bad
#     position can't raise out of a render call.
#   - on_snapshot() is now split into independently try/except-wrapped
#     sections (point cloud 3D, point cloud 2D, track render/GC) so a
#     failure in one section (e.g. a bad color, a malformed history) can't
#     take down the whole GUI update or leave stale artists on screen.
#   - Track render data is sanitized before use: any track whose current
#     position isn't finite is skipped for that frame, and history points
#     are filtered to finite values before being handed to the trail items.
#   - ParserThread.run()'s data-port-open step now also catches generic
#     Exception (not just SerialException), and the port-close is wrapped
#     in a try/finally-style pattern so it's never skipped even if something
#     upstream misbehaves.
# -----------------------------------------------------------------------------
#

# -----------------------------------------------------------------------------
# CHANGELOG v3 -> v3.1  (people counter / ghost-track fix)
# -----------------------------------------------------------------------------
#  [PEOPLE COUNTER]
#   - Large always-visible "People: N" QLabel in a top toolbar.
#   - N = ACTIVE tracks only: missed == 0 in the snapshot AND room-frame Z
#     inside the human plausibility band (same rule used for drawing boxes).
#   - Status bar still shows "Frame | K active people | points | errors".
#
#  [GHOST TRACK FIX]  (green box floating above the yellow volume)
#   - After make_transform(), room-frame Z must satisfy
#     TRACK_Z_ROOM_MIN < Z < TRACK_Z_ROOM_MAX (default 0.3..2.2 m).
#     Tracks outside that band are dropped from track_state (not coasted).
#   - Only ACTIVE tracks (missed == 0) that pass the Z band are drawn as
#     solid person boxes. Coasting tracks (missed > 0) are hidden — never
#     left as a solid floating box.
#   - After every Kalman predict/update, Z is clamped to TRACK_Z_CLAMP
#     (default 0..2.5 m, matching typical boundaryBox Z) and vz is zeroed
#     if the clamp fired, so a bad vertical velocity cannot climb into the
#     sky over a few missed frames.
#   - Person wireframe is floor-anchored: box bottom near z=0 (or just
#     above), top at bottom + PERSON_HEIGHT — no more centering a 1.75 m
#     box on a high track point.
#   - Accumulated point cloud drops points whose Z is outside the
#     detection-volume Z range (+ CLOUD_Z_MARGIN), killing the vertical
#     "ghost trail" streak.
#   - --debug logs ghost rejects: tid, raw xyz, room xyz, missed, reason.
#
# REQUIREMENTS
#   pip install pyserial numpy pyqtgraph PyOpenGL
#   pip install PySide6      # preferred
#   # or, if PySide6 isn't available:
#   pip install PySide2
#
#   The official TI parser files must be importable (same directory as this
#   script, or a `ti_parser/` subfolder next to it, or anywhere on
#   PYTHONPATH):
#       parseFrame.py, parseTLVs.py, tlv_defines.py, demo_defines.py,
#       gui_common.py
#   NOTE: parseFrame.py imports
#       Demo_Classes.Helper_Classes.classification.ClassificationSupplement
#   which must also be importable (part of TI's Industrial Visualizer repo).
#
# USAGE
#   python live_3d_radar_v10.py --data-port COM6 --cli-port COM5 --cfg my_occupancy.cfg
#
#   Optional flags:
#     --data-baud       Data-port baud rate (default 921600)
#     --cli-baud        CLI-port baud rate (default 115200)
#     --demo            Demo name for parseFrame.py header selection
#                        (default: '3D People Tracking')
#     --accum-frames    Frames of points kept for a dense cloud (default 12)
#     --max-points      Hard cap on points rendered per draw (default 4000)
#     --sensor-height   Sensor mount height above the floor, metres (default 3.0)
#     --tilt            Sensor downward tilt, degrees (default 18, ceiling mount)
#     --debug           Verbose logging: TLV-1010 payload sizes, struct
#                        layout chosen, track counts, sanity-check rejects,
#                        hex preview of raw bytes. Use this to tune
#                        TRACK_STRUCT_CANDIDATES against a new .cfg profile.
#     --boundary-box    "minX,maxX,minY,maxY,minZ,maxZ" (metres, ROOM
#                        coordinates: floor-aligned, Z=0 at the floor)
#                        overriding the default detection-volume wireframe
#                        shown in the 3D view. This is a fixed reference
#                        box for the monitored area and does not rotate
#                        with the sensor's tilt. Should roughly match the
#                        `boundaryBox` line in your .cfg.
# =============================================================================

import os
import sys
import time
import math
import struct
import argparse
import logging
from collections import deque

import numpy as np
import serial

# -----------------------------------------------------------------------------
# Qt binding: PySide6 preferred, PySide2 fallback. Must happen before
# `import pyqtgraph` so pyqtgraph picks up the same binding.
# -----------------------------------------------------------------------------
try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Signal, QThread
    QT_LIB = "PySide6"
except ImportError:
    from PySide2 import QtCore, QtGui, QtWidgets
    from PySide2.QtCore import Signal, QThread
    QT_LIB = "PySide2"

os.environ["PYQTGRAPH_QT_LIB"] = QT_LIB

import pyqtgraph as pg
import pyqtgraph.opengl as gl

# -----------------------------------------------------------------------------
# Official TI parser modules. Look alongside this script and in a
# `ti_parser/` subfolder, matching v9's search path convention.
# -----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in (HERE, os.path.join(HERE, "ti_parser")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from parseFrame import parseStandardFrame           # official TI frame parser
import demo_defines as demo_defines_mod              # official demo name table

log = logging.getLogger("live_3d_radar_v10")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# =============================================================================
# TUNABLE PARAMETERS
# =============================================================================
SENSOR_HEIGHT = 2.4           # metres above the floor (matches typical sensorPosition)
TILT_DEG = 18.0                # downward tilt of the sensor, degrees

MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"
FRAME_HEADER_LEN = 40          # 'Q8I' -> 8 + 8*4 bytes
TOTAL_LEN_OFFSET = 12          # totalPacketLen field offset in the header
NUM_TLV_OFFSET = 32            # numTLVs field offset in the header
TLV_HEADER_LEN = 8             # '2I' -> type, length
MAX_REASONABLE_FRAME_LEN = 1 << 20

TLV_TARGET_LIST_1010 = 1010    # MMWDEMO_OUTPUT_MSG_TRACKERPROC_3D_TARGET_LIST

# [TLV 1010 FIX] Known TI target-list struct layouts, tried in order against
# the actual TLV payload length (see _parse_track_payload). Add more entries
# here if --debug reveals a payload size that doesn't match either of these
# (e.g. a firmware variant with a smaller/larger error-covariance matrix).
TRACK_STRUCT_CANDIDATES = [
    ("3D-full", "<I27f"),   # tid + pos/vel/acc(9) + ec[16] + g + confidence = 112B
    ("2D", "<I17f"),        # tid + pos/vel/acc(6) + ec[9]  + g + confidence =  72B
]

ACCUM_FRAMES = 16              # [SENSITIVITY] was 12: denser cloud for visualization
PC_MAX_POINTS = 5000           # raised alongside ACCUM_FRAMES to avoid over-culling
PC_SNR_MIN = 0.0               # drop points below this SNR (0 = keep all)

TRAIL_SECONDS = 3.5
MAX_MISSED = 15                # [SENSITIVITY] was 10: tolerate brief gaps without
                                # deleting/re-creating the track (more stable IDs)
PERSON_HALF_W = 0.30
PERSON_HEIGHT = 1.75

# [SENSITIVITY] Kalman retune: lower measurement noise trusts fresh detections
# more (faster convergence onto a real person); higher process noise lets the
# filter follow real movement instead of lagging behind it; wider gate means
# fewer genuine detections get rejected as "not the same track".
MEASURE_STD = 0.10             # was 0.12
PROCESS_STD = 1.6              # was 1.2
GATE_SIGMA = 4.5               # was 3.5
GATE_MIN_M = 1.2               # was 0.8

# [SENSITIVITY] ROI relaxed slightly so people near the edge of the covered
# area aren't dropped before they ever reach the Kalman filter.
ROI_Y_RAW = (0.1, 18.0)        # was (0.2, 15.0)
ROI_Z_RAW = (-4.0, 2.0)        # was (-3.5, 1.0)

ROOM = {"x": (-8, 8), "y": (-2, 16), "z": (-0.5, 3.5)}

# ---- [1. DETECTION VOLUME BOX] ---------------------------------------------
# [FIX] Previously this box was defined in RAW sensor-frame coordinates and
# run through the full tilt+height transform, which rotated it with the
# sensor's tilt angle -- so it appeared tilted relative to the floor grid.
# That's wrong for a "monitoring area" reference box: it's a room-frame
# concept (e.g. "people in this 5m x 9.5m x 2.5m volume of the room"), not a
# sensor-frame one, and it should stay axis-aligned with the floor regardless
# of how the sensor is mounted/tilted.
#
# BOUNDARY_BOX_ROOM is therefore now defined directly in ROOM coordinates
# (metres): X is left/right, Y is forward/depth, Z is height above the
# floor (Z=0). Only a height translation is ever applied to it (see
# boundary_box_line_vertices() / make_height_only_transform() below) --
# never the tilt rotation. Override with --boundary-box if it doesn't match
# your monitored area.
# Default matches a typical 3D people-tracking .cfg boundaryBox
# (e.g. "boundaryBox -2.8 2.8 0.6 5.5 0 2.5"). Override with --boundary-box.
BOUNDARY_BOX_ROOM = {
    "minX": -2.8, "maxX": 2.8,
    "minY": 0.6, "maxY": 5.5,
    "minZ": 0.0, "maxZ": 2.5,
}
# Back-compat alias: some code/tooling may still reference the old name.
BOUNDARY_BOX_RAW = BOUNDARY_BOX_ROOM

BOUNDARY_BOX_COLOR = (1.0, 1.0, 0.0, 0.7)         # [FIX] bright, high-visibility yellow
BOUNDARY_BOX_WIDTH = 1.0                          # thin lines
BOUNDARY_BOX_DASHED = True                        # approximate dashing (see _dash_segment)
BOUNDARY_BOX_DASH_LEN = 0.28
BOUNDARY_BOX_GAP_LEN = 0.18


# ---- [GHOST TRACK FIX / PEOPLE COUNTER] ------------------------------------
# Room-frame Z band for a plausible standing human (after make_transform).
# Tracks outside this band are rejected / deleted instead of coasting upward.
TRACK_Z_ROOM_MIN = 0.20
TRACK_Z_ROOM_MAX = 2.40
# Hard clamp applied after every Kalman predict/update (metres, room frame).
# Matches typical .cfg boundaryBox Z (0 .. 2.5).
TRACK_Z_CLAMP = (0.0, 2.5)
# Drop accumulated cloud points outside detection-volume Z +/- this margin.
CLOUD_Z_MARGIN = 0.30



# =============================================================================
# Geometry: sensor -> room-frame transform (tilt compensation + height offset)
# =============================================================================
def make_transform(sensor_height, tilt_deg):
    theta = math.radians(tilt_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def transform_array(pts):
        """pts: (N,3) raw sensor-frame [x,y,z] -> (N,3) room-frame."""
        if pts is None or len(pts) == 0:
            return np.empty((0, 3), dtype=float)
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        y_new = y * cos_t - z * sin_t
        z_new = y * sin_t + z * cos_t + sensor_height
        return np.column_stack([x, y_new, z_new])

    return transform_array


def make_height_only_transform(sensor_height):
    """[FIX] Height-translation-only transform, deliberately WITHOUT the
    tilt rotation applied by make_transform(). Used exclusively for the
    detection-volume reference box: that box is defined directly in room
    coordinates (floor-aligned), so it must stay axis-aligned with the
    floor grid regardless of the sensor's mounting tilt. `make_transform`
    remains unchanged and is still the one used for the point cloud and
    tracks, which genuinely are captured in the sensor's tilted frame.

    Kept as a coordinate-shift function (rather than a no-op) in case a
    future BOUNDARY_BOX_ROOM origin convention needs a Z offset; today it's
    an identity shift since BOUNDARY_BOX_ROOM's Z values are already
    "height above the floor" and don't need re-basing to sensor height.
    """
    def transform_array(pts):
        if pts is None or len(pts) == 0:
            return np.empty((0, 3), dtype=float)
        return np.asarray(pts, dtype=float).reshape(-1, 3).copy()

    return transform_array



# =============================================================================
# Constant-velocity Kalman filter (kept from v9)
# =============================================================================
class CVKalman3D:
    """Simple constant-velocity Kalman filter over 3D position."""

    def __init__(self, pos, meas_std=MEASURE_STD, proc_std=PROCESS_STD):
        self.x = np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0], dtype=float)
        self.P = np.diag([0.2, 0.2, 0.2, 2.0, 2.0, 2.0]).astype(float)
        self.q = float(proc_std) ** 2
        self.r = float(meas_std) ** 2
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
        self.R = np.eye(3) * self.r

    def _F(self, dt):
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        return F

    def _Q(self, dt):
        q = self.q
        dt2, dt3, dt4 = dt * dt, dt * dt * dt, dt * dt * dt * dt
        Q = np.zeros((6, 6))
        for i in range(3):
            p, v = i, i + 3
            Q[p, p] = dt4 / 4.0 * q
            Q[p, v] = dt3 / 2.0 * q
            Q[v, p] = dt3 / 2.0 * q
            Q[v, v] = dt2 * q
        return Q

    def predict(self, dt):
        dt = max(dt, 1e-3)
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def innovation(self, z):
        y = np.asarray(z, dtype=float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def update(self, z):
        y, S = self.innovation(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return y, S

    @property
    def pos(self):
        return self.x[0], self.x[1], self.x[2]

    def clamp_z(self, z_min=None, z_max=None):
        """[GHOST FIX] Keep room-frame Z inside [z_min, z_max]. If the clamp
        fires, also zero vertical velocity so coasting cannot climb further.
        Returns True if a clamp was applied.
        """
        z_min = TRACK_Z_CLAMP[0] if z_min is None else z_min
        z_max = TRACK_Z_CLAMP[1] if z_max is None else z_max
        z = float(self.x[2])
        if z < z_min or z > z_max:
            self.x[2] = min(max(z, z_min), z_max)
            self.x[5] = 0.0  # vz
            return True
        return False


# =============================================================================
# TLV / frame helpers
# =============================================================================
def iter_tlvs(frame_bytes):
    """Yield (tlv_type, payload_bytes) for every TLV in a complete frame."""
    if len(frame_bytes) < FRAME_HEADER_LEN:
        return
    num_tlvs = struct.unpack_from("<I", frame_bytes, NUM_TLV_OFFSET)[0]
    offset = FRAME_HEADER_LEN
    for _ in range(num_tlvs):
        if offset + TLV_HEADER_LEN > len(frame_bytes):
            break
        tlv_type, tlv_len = struct.unpack_from("<II", frame_bytes, offset)
        offset += TLV_HEADER_LEN
        payload = frame_bytes[offset:offset + tlv_len]
        if len(payload) < tlv_len:
            break
        yield tlv_type, payload
        offset += tlv_len


def manual_parse_tracks(frame_bytes):
    """Manually decode every TLV 1010 (3D Track List) found in the raw frame.

    [TLV 1010 FIX] This is intentionally defensive: TLV parsing here must
    NEVER be allowed to raise out of this function, because a single bad or
    unexpectedly-shaped TLV would otherwise take down the whole parser
    thread. Each TLV's payload is handed to _parse_track_payload(), which
    does its own internal error handling and sanity checking.
    """
    tracks = []
    try:
        for tlv_type, payload in iter_tlvs(frame_bytes):
            if tlv_type != TLV_TARGET_LIST_1010:
                continue
            try:
                tracks.extend(_parse_track_payload(payload))
            except Exception as e:
                # Should be unreachable (_parse_track_payload guards itself),
                # but kept as a last-resort net so one malformed TLV can never
                # crash frame parsing.
                log.debug("TLV1010: unexpected failure parsing payload (%d bytes): %s",
                          len(payload), e)
    except Exception as e:
        log.debug("TLV1010: failed while iterating TLVs in frame: %s", e)
    return tracks


def _parse_track_payload(payload):
    """Robustly decode one TLV-1010 payload of unknown/variable per-track
    size, tolerating firmware/demo variants and partial/misaligned data.

    [TLV 1010 FIX] Strategy:
      1. Try each known struct layout in TRACK_STRUCT_CANDIDATES; use the
         first one whose size evenly divides the payload length (that's the
         one that's actually correct for this frame).
      2. If none divide evenly (unknown layout / corrupt TLV), fall back to
         whichever candidate leaves the smallest remainder, and still parse
         the whole-number portion rather than discarding all of it.
      3. Decode each record with struct.unpack_from inside its own
         try/except -- a failure there just stops this TLV's loop instead of
         raising.
      4. Sanity-check every decoded record (finite values, plausible
         position range, plausible track ID). A struct-size mismatch
         produces wild/garbage numbers rather than a clean parse error, so
         this is the check that actually catches most misalignment bugs.

    Returns list of (tid, x, y, z, vx, vy, vz).
    """
    results = []
    plen = len(payload)
    if plen < 4:
        log.debug("TLV1010: payload too short (%d bytes), skipping", plen)
        return results

    sizes = [(name, fmt, struct.calcsize(fmt)) for name, fmt in TRACK_STRUCT_CANDIDATES]

    chosen = None
    for name, fmt, size in sizes:
        if size > 0 and plen % size == 0 and plen // size > 0:
            chosen = (name, fmt, size)
            break

    if chosen is None:
        # No exact fit -- use the candidate with the smallest leftover bytes
        # and just parse the whole-number portion of it.
        name, fmt, size = min(sizes, key=lambda c: (plen % c[2]) if c[2] else plen)
        chosen = (name, fmt, size)
        log.debug("TLV1010: no exact struct match for payload=%dB; "
                  "falling back to %s (item=%dB, remainder=%dB)",
                  plen, name, size, plen % size if size else plen)

    name, fmt, size = chosen
    n = plen // size if size else 0
    log.debug("TLV1010: payload=%dB using '%s' struct (item=%dB) -> %d candidate track(s); "
              "first bytes=%s", plen, name, size, n, payload[:16].hex())

    accepted = 0
    for i in range(n):
        offset = i * size
        try:
            vals = struct.unpack_from(fmt, payload, offset)
        except struct.error as e:
            log.debug("TLV1010: unpack failed at record %d/%d (%s); stopping this TLV", i, n, e)
            break

        try:
            tid = int(vals[0])
            x, y, z = float(vals[1]), float(vals[2]), float(vals[3])
            vx, vy, vz = float(vals[4]), float(vals[5]), float(vals[6])
        except (IndexError, ValueError) as e:
            log.debug("TLV1010: record %d field extraction failed: %s", i, e)
            continue

        # ---- sanity checks: catch struct-size mismatches (garbage values) --
        if not all(math.isfinite(v) for v in (x, y, z, vx, vy, vz)):
            log.debug("TLV1010: record %d has non-finite value(s), rejected", i)
            continue
        if not (-100.0 < x < 100.0 and -100.0 < y < 100.0 and -100.0 < z < 100.0):
            log.debug("TLV1010: record %d out of plausible range (%.2f, %.2f, %.2f), rejected",
                      i, x, y, z)
            continue
        if not (0 <= tid < 250):
            log.debug("TLV1010: record %d has implausible tid=%d, rejected", i, tid)
            continue

        results.append((tid, x, y, z, vx, vy, vz))
        accepted += 1

    log.debug("TLV1010: %d/%d record(s) passed sanity checks", accepted, n)
    return results


def extract_pointcloud(out):
    """Return (N,4) [x, y, z, snr] from the official parser's outputDict,
    sliced to numDetectedPoints. Relies on TI's own TLV-1020 decoding."""
    pc = out.get("pointCloud", None) if out else None
    if pc is None:
        return np.empty((0, 4), dtype=float)
    arr = np.asarray(pc, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
        return np.empty((0, 4), dtype=float)
    n = int(out.get("numDetectedPoints", arr.shape[0]) or 0)
    n = max(0, min(n, arr.shape[0]))
    if n == 0:
        return np.empty((0, 4), dtype=float)
    arr = arr[:n]
    xyz = arr[:, :3]
    snr = arr[:, 4] if arr.shape[1] >= 5 else np.zeros(len(arr))
    return np.column_stack([xyz, snr])


def _row_to_track(row, fallback_idx):
    """Pull (tid, x, y, z) out of one track record, whatever shape it's in:
    a numpy structured record, a dict, or a plain indexable sequence."""
    dtype_names = getattr(row, "dtype", None)
    dtype_names = dtype_names.names if dtype_names is not None else None

    if dtype_names:  # numpy structured array row
        tid = int(row["tid"]) if "tid" in dtype_names else int(row[0])
        x = float(row["posX"]) if "posX" in dtype_names else float(row[1])
        y = float(row["posY"]) if "posY" in dtype_names else float(row[2])
        z = float(row["posZ"]) if "posZ" in dtype_names else float(row[3])
        return tid, x, y, z
    if isinstance(row, dict):
        tid = int(row.get("tid", row.get("targetId", fallback_idx)))
        x = float(row.get("posX", row.get("x")))
        y = float(row.get("posY", row.get("y")))
        z = float(row.get("posZ", row.get("z")))
        return tid, x, y, z
    # plain list / tuple / 1D numpy array: [tid, x, y, z, ...]
    tid = int(row[0])
    x, y, z = float(row[1]), float(row[2]), float(row[3])
    return tid, x, y, z


def extract_tracks_robust(out, frame_bytes):
    """Robust track extraction with TID dedupe.

    [GHOST FIX] Official parseStandardFrame trackData and the manual TLV-1010
    path can both yield tracks for the same person. If we naively trusted one
    then the other across frames (or ever merged without dedupe), one person
    could produce two boxes (e.g. the floating green ghost beside a valid red
    one). Strategy:
      1. Collect official tracks (if any).
      2. Collect manual TLV-1010 tracks.
      3. Merge by tid into a dict — manual values WIN when both report the
         same tid (manual parse is the reliable path for this TLV).
      4. Return the deduped list.

    Returns list of (tid, x, y, z).
    """
    by_tid = {}

    data = out.get("trackData", None) if out else None
    num = int(out.get("numDetectedTracks", 0) or 0) if out else 0
    if data is not None and num > 0:
        try:
            n = min(num, len(data))
            for i in range(n):
                try:
                    tid, x, y, z = _row_to_track(data[i], i)
                    by_tid[int(tid)] = (int(tid), float(x), float(y), float(z))
                except (TypeError, ValueError, IndexError, KeyError):
                    continue
        except TypeError:
            pass

    try:
        manual = manual_parse_tracks(frame_bytes)
        for (tid, x, y, z, vx, vy, vz) in manual:
            # Manual wins on tid collision.
            by_tid[int(tid)] = (int(tid), float(x), float(y), float(z))
    except Exception as e:
        log.debug("manual_parse_tracks failed inside extract_tracks_robust: %s", e)

    return list(by_tid.values())


# =============================================================================
# Serial helper: send a TI .cfg profile over the CLI/user UART
# =============================================================================
def send_cfg(cli_port, cfg_path, baud=115200):
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    log.info("Opening CLI port %s @ %d baud to send config...", cli_port, baud)
    with serial.Serial(cli_port, baud, timeout=0.3) as cli:
        with open(cfg_path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("%"):
                    continue
                cli.write((line + "\n").encode())
                time.sleep(0.03)
    log.info("Config sent successfully.")



def room_z_plausible(z):
    """True if room-frame Z looks like a standing human body/torso."""
    try:
        z = float(z)
    except (TypeError, ValueError):
        return False
    return math.isfinite(z) and TRACK_Z_ROOM_MIN < z < TRACK_Z_ROOM_MAX


def in_boundary_box(x, y, z, bb=None, margin=0.15):
    """[GHOST FIX] True if room-frame (x,y,z) is inside the detection volume
    (BOUNDARY_BOX_ROOM), with a small margin. Ghosts that climb above the
    yellow lid or drift outside the monitored XY area are rejected here.
    """
    bb = bb or BOUNDARY_BOX_ROOM
    try:
        x, y, z = float(x), float(y), float(z)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(v) for v in (x, y, z)):
        return False
    return (
        (bb["minX"] - margin) <= x <= (bb["maxX"] + margin)
        and (bb["minY"] - margin) <= y <= (bb["maxY"] + margin)
        and (bb["minZ"] - margin) <= z <= (bb["maxZ"] + margin)
    )


def filter_cloud_by_volume_z(cloud, bb=None, margin=CLOUD_Z_MARGIN):
    """[GHOST FIX] Drop accumulated points outside the detection volume
    (full AABB, not only Z). Removes the vertical streak that otherwise
    accumulates toward a floating ghost track.
    """
    if cloud is None or len(cloud) == 0:
        return np.empty((0, 4), dtype=float)
    bb = bb or BOUNDARY_BOX_ROOM
    x0, x1 = float(bb["minX"]) - margin, float(bb["maxX"]) + margin
    y0, y1 = float(bb["minY"]) - margin, float(bb["maxY"]) + margin
    z0, z1 = float(bb["minZ"]) - margin, float(bb["maxZ"]) + margin
    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    mask = finite & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (z >= z0) & (z <= z1)
    return cloud[mask]


def log_ghost_reject(tid, raw_xyz, room_xyz, missed, reason):
    """Opt-in (--debug) one-liner when a track is rejected as a ghost."""
    try:
        rx, ry, rz = raw_xyz
        qx, qy, qz = room_xyz
        log.debug(
            "GHOST reject tid=%s raw=(%.2f,%.2f,%.2f) room=(%.2f,%.2f,%.2f) "
            "missed=%s reason=%s",
            tid, rx, ry, rz, qx, qy, qz, missed, reason)
    except Exception:
        log.debug("GHOST reject tid=%s reason=%s (log fields unavailable)", tid, reason)


def _trim_history(hist, now):
    cutoff = now - TRAIL_SECONDS
    while hist and hist[0][0] < cutoff:
        hist.popleft()


# =============================================================================
# Parser thread: producer. Owns the serial port, runs the TI parser + the
# manual TLV-1010 fallback, does the Kalman filtering, and emits ready-to-
# render snapshots to the GUI thread over a queued Qt signal.
# =============================================================================
class ParserThread(QThread):
    snapshotReady = Signal(dict)     # {'cloud':.., 'tracks':.., 'frame_count':.., 'errors':..}
    statusMessage = Signal(str)
    errorOccurred = Signal(str)

    def __init__(self, data_port_name, data_baud, demo_name,
                 sensor_height, tilt_deg, accum_frames, max_points,
                 boundary_box=None, parent=None):
        super().__init__(parent)
        self.data_port_name = data_port_name
        self.data_baud = data_baud
        self.demo_name = demo_name
        self.transform_array = make_transform(sensor_height, tilt_deg)
        self.cloud_frames = deque(maxlen=max(1, accum_frames))
        self.max_points = max_points
        # Detection volume used for ghost rejection + cloud gating.
        self.boundary_box = boundary_box or dict(BOUNDARY_BOX_ROOM)
        self._running = True
        self.track_state = {}         # tid -> {kf, history, missed, last_update}
        self.frame_count = 0
        self.parse_errors = 0

    def stop(self):
        self._running = False

    def run(self):
        # [3. STABILITY] Catch generic Exception here too, not just
        # SerialException -- any failure opening the port (bad name,
        # permissions, already in use, etc.) should surface as a clean
        # errorOccurred signal instead of an unhandled exception in the
        # QThread.
        data_port = None
        try:
            self.statusMessage.emit(f"Opening data port {self.data_port_name} @ {self.data_baud} baud...")
            data_port = serial.Serial(self.data_port_name, self.data_baud, timeout=0.05)
        except serial.SerialException as e:
            self.errorOccurred.emit(f"Could not open data port {self.data_port_name}: {e}")
            return
        except Exception as e:
            self.errorOccurred.emit(f"Unexpected error opening data port {self.data_port_name}: {e}")
            return

        self.statusMessage.emit("Data port open. Streaming...")
        buffer = bytearray()

        while self._running:
            try:
                chunk = data_port.read(4096)
                if chunk:
                    buffer.extend(chunk)

                while True:
                    start = buffer.find(MAGIC_WORD)
                    if start == -1:
                        if len(buffer) > 1 << 20:
                            del buffer[:-8]
                        break
                    if start > 0:
                        del buffer[:start]
                    if len(buffer) < FRAME_HEADER_LEN:
                        break

                    total_len = int.from_bytes(
                        buffer[TOTAL_LEN_OFFSET:TOTAL_LEN_OFFSET + 4], "little")
                    if total_len <= 0 or total_len > MAX_REASONABLE_FRAME_LEN:
                        del buffer[:8]
                        continue
                    if len(buffer) < total_len:
                        break

                    frame = bytes(buffer[:total_len])
                    del buffer[:total_len]
                    self._process_frame(frame)

            except serial.SerialException as e:
                self.errorOccurred.emit(f"Serial error: {e}")
                time.sleep(0.2)
            except Exception as e:
                self.parse_errors += 1
                log.debug("Parse-loop error: %s", e)
                time.sleep(0.05)

        try:
            data_port.close()
        except Exception:
            pass
        self.statusMessage.emit("Data port closed.")

    def _process_frame(self, frame):
        # [STABILITY FIX] The entire method body is now wrapped so that any
        # unexpected failure anywhere in point-cloud extraction, track
        # extraction, or Kalman-filter math is caught, logged, and counted
        # as a dropped frame -- it can never propagate up and kill the
        # parser thread.
        try:
            self._process_frame_inner(frame)
        except Exception as e:
            self.parse_errors += 1
            log.error("Unexpected error processing frame (frame dropped): %s", e)

    def _process_frame_inner(self, frame):
        # ---- Point cloud: official TI parser (TLV 1020) --------------------
        try:
            out = parseStandardFrame(frame, demo=self.demo_name)
        except Exception as e:
            self.parse_errors += 1
            log.debug("parseStandardFrame failed: %s", e)
            out = None

        raw_pts = extract_pointcloud(out) if out else np.empty((0, 4))
        if PC_SNR_MIN > 0 and len(raw_pts) > 0:
            raw_pts = raw_pts[raw_pts[:, 3] >= PC_SNR_MIN]
        if len(raw_pts) > 0:
            room_xyz = self.transform_array(raw_pts[:, :3])
            room_pts = np.column_stack([room_xyz, raw_pts[:, 3]])
            # [GHOST FIX] Drop high/low-Z outliers before they accumulate into
            # the vertical "ghost trail" streak toward a floating box.
            room_pts = filter_cloud_by_volume_z(room_pts, bb=self.boundary_box)
        else:
            room_pts = np.empty((0, 4))

        self.cloud_frames.append(room_pts)
        non_empty = [c for c in self.cloud_frames if len(c) > 0]
        stacked = np.vstack(non_empty) if non_empty else np.empty((0, 4))
        if len(stacked) > 0:
            stacked = filter_cloud_by_volume_z(stacked, bb=self.boundary_box)
        if len(stacked) > self.max_points:
            idx = np.random.choice(len(stacked), self.max_points, replace=False)
            stacked = stacked[idx]

        # ---- Track list: robust extraction, manual TLV-1010 fallback ------
        # [STABILITY FIX] extra safety net on top of manual_parse_tracks()'s
        # own internal error handling.
        try:
            tracks = extract_tracks_robust(out, frame)
        except Exception as e:
            log.debug("extract_tracks_robust failed, treating frame as track-less: %s", e)
            tracks = []

        now = time.time()
        seen_ids = set()
        for tid, x_raw, y_raw, z_raw in tracks:
            try:
                if not (ROI_Y_RAW[0] < y_raw < ROI_Y_RAW[1]
                        and ROI_Z_RAW[0] < z_raw < ROI_Z_RAW[1]):
                    continue
                meas = self.transform_array(np.array([[x_raw, y_raw, z_raw]]))[0]
                raw_xyz = (float(x_raw), float(y_raw), float(z_raw))
                room_xyz = (float(meas[0]), float(meas[1]), float(meas[2]))

                # [GHOST FIX] Reject measurements whose room-frame pose is not a
                # plausible standing human inside the detection volume.
                # Do NOT create / update state for them, and drop any existing
                # state so we never coast into the sky / outside the yellow box.
                if not room_z_plausible(meas[2]):
                    log_ghost_reject(tid, raw_xyz, room_xyz,
                                     self.track_state.get(tid, {}).get("missed", 0),
                                     "room_z_out_of_band_on_measure")
                    if tid in self.track_state:
                        del self.track_state[tid]
                    continue
                if not in_boundary_box(meas[0], meas[1], meas[2], bb=self.boundary_box):
                    log_ghost_reject(tid, raw_xyz, room_xyz,
                                     self.track_state.get(tid, {}).get("missed", 0),
                                     "outside_boundary_box_on_measure")
                    if tid in self.track_state:
                        del self.track_state[tid]
                    continue

                seen_ids.add(tid)
                st = self.track_state.get(tid)
                if st is None:
                    kf = CVKalman3D(meas)
                    kf.clamp_z()
                    st = {"kf": kf, "history": deque(), "missed": 0,
                          "last_update": now, "last_raw": raw_xyz}
                    self.track_state[tid] = st
                    px, py, pz = kf.pos
                    st["history"].append((now, px, py, pz))
                else:
                    kf = st["kf"]
                    dt = max(now - st["last_update"], 1e-3)
                    kf.predict(dt)
                    kf.clamp_z()
                    y_res, S = kf.innovation(meas)
                    try:
                        dist2 = float(y_res.T @ np.linalg.inv(S) @ y_res)
                    except np.linalg.LinAlgError:
                        dist2 = 0.0
                    gate = max(GATE_SIGMA ** 2, (GATE_MIN_M ** 2) / MEASURE_STD ** 2)
                    if dist2 <= gate:
                        kf.update(meas)
                        kf.clamp_z()
                        st["missed"] = 0
                    else:
                        st["missed"] += 1
                    st["last_update"] = now
                    st["last_raw"] = raw_xyz
                    px, py, pz = kf.pos
                    # If clamp still left us outside the human band / volume, drop.
                    if not room_z_plausible(pz):
                        log_ghost_reject(tid, raw_xyz, (px, py, pz),
                                         st["missed"], "room_z_out_of_band_after_update")
                        del self.track_state[tid]
                        continue
                    if not in_boundary_box(px, py, pz, bb=self.boundary_box):
                        log_ghost_reject(tid, raw_xyz, (px, py, pz),
                                         st["missed"], "outside_boundary_box_after_update")
                        del self.track_state[tid]
                        continue
                    st["history"].append((now, px, py, pz))
                _trim_history(st["history"], now)
            except Exception as e:
                log.debug("track update failed for tid=%s: %s", tid, e)

        for tid in list(self.track_state.keys()):
            if tid in seen_ids:
                continue
            try:
                st = self.track_state[tid]
                kf = st["kf"]
                dt = max(now - st["last_update"], 1e-3)
                kf.predict(dt)
                kf.clamp_z()
                st["last_update"] = now
                st["missed"] += 1
                px, py, pz = kf.pos
                raw_xyz = st.get("last_raw", (float("nan"),) * 3)
                # [GHOST FIX] Coasting with bad Z / outside volume -> delete.
                if not room_z_plausible(pz):
                    log_ghost_reject(tid, raw_xyz, (px, py, pz),
                                     st["missed"], "room_z_out_of_band_while_coasting")
                    del self.track_state[tid]
                    continue
                if not in_boundary_box(px, py, pz, bb=self.boundary_box):
                    log_ghost_reject(tid, raw_xyz, (px, py, pz),
                                     st["missed"], "outside_boundary_box_while_coasting")
                    del self.track_state[tid]
                    continue
                st["history"].append((now, px, py, pz))
                _trim_history(st["history"], now)
                if st["missed"] > MAX_MISSED:
                    del self.track_state[tid]
            except Exception as e:
                log.debug("coast update failed for tid=%s: %s", tid, e)
                self.track_state.pop(tid, None)

        self.frame_count += 1

        # Snapshot includes all surviving tracks; the GUI only *draws* active
        # ones (missed == 0) that still pass room_z_plausible — that is also
        # how the People counter is defined.
        snapshot = {
            "cloud": stacked,
            "tracks": {
                tid: {"pos": st["kf"].pos,
                      "history": [(h[1], h[2], h[3]) for h in st["history"]],
                      "missed": st["missed"]}
                for tid, st in self.track_state.items()
            },
            "frame_count": self.frame_count,
            "errors": self.parse_errors,
        }
        self.snapshotReady.emit(snapshot)


# =============================================================================
# Color mapping: SNR -> viridis-like RGBA, implemented purely with
# pyqtgraph's own ColorMap (no matplotlib dependency).
# =============================================================================
_VIRIDIS_STOPS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_VIRIDIS_COLORS = np.array([
    [68, 1, 84, 255],
    [59, 82, 139, 255],
    [33, 144, 140, 255],
    [93, 201, 99, 255],
    [253, 231, 37, 255],
], dtype=float)
_CMAP = pg.ColorMap(_VIRIDIS_STOPS, _VIRIDIS_COLORS)


def snr_to_rgba_float(snr):
    """snr: (N,) array -> (N,4) RGBA floats in [0,1], for GLScatterPlotItem."""
    if snr is None or len(snr) == 0:
        return np.zeros((0, 4), dtype=float)
    lo, hi = float(np.min(snr)), float(np.max(snr))
    norm = (snr - lo) / (hi - lo) if hi > lo else np.zeros_like(snr)
    return _CMAP.map(norm, mode="float")


def snr_to_brushes(snr):
    """snr: (N,) array -> list of QColor, for pg.ScatterPlotItem in 2D view."""
    if snr is None or len(snr) == 0:
        return []
    lo, hi = float(np.min(snr)), float(np.max(snr))
    norm = (snr - lo) / (hi - lo) if hi > lo else np.zeros_like(snr)
    return list(_CMAP.map(norm, mode="qcolor"))


# =============================================================================
# Geometry helpers for wireframe boxes
# =============================================================================
# Shared 12-edge connectivity for any axis-aligned box defined by 8 corners
# in the order [(x,y,z) for z in (z0,z1) for x in (x0,x1) for y in (y0,y1)].
# Used by box_line_vertices(), cube_line_vertices(), and the new
# boundary_box_line_vertices() so all three stay in sync.
_BOX_EDGES_12 = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]


def box_line_vertices(cx, cy, cz, half_w=PERSON_HALF_W, height=PERSON_HEIGHT):
    """(2*12, 3) vertex-pair array for a GLLinePlotItem(mode='lines') box.

    [GHOST FIX] Floor-anchored: the track point `cz` is treated as a torso /
    body height hint, but the wireframe BOTTOM sits near the floor (z≈0)
    and the TOP is bottom + PERSON_HEIGHT. Centering a 1.75 m box on a
    high/spiky Z was one way a "ghost" looked like it floated above the
    yellow volume even when XY was fine.
    """
    # Prefer feet near the floor; if cz is a plausible torso, keep a mild
    # offset so the box still roughly covers the body, but never go below 0.
    z0 = max(0.0, float(cz) - 0.95)
    # If the torso hint itself is near the floor, still draw a full-height box.
    if z0 + height < float(cz) + 0.2:
        z0 = max(0.0, float(cz) - height * 0.55)
    z1 = z0 + height
    xs, ys = (cx - half_w, cx + half_w), (cy - half_w, cy + half_w)
    corners = [(x, y, z) for z in (z0, z1) for x in xs for y in ys]
    pts = []
    for a, b in _BOX_EDGES_12:
        pts.append(corners[a])
        pts.append(corners[b])
    return np.array(pts, dtype=float)


def cube_line_vertices(cx, cy, cz, size):
    h = size / 2.0
    xs, ys, zs = (cx - h, cx + h), (cy - h, cy + h), (cz - h, cz + h)
    corners = [(x, y, z) for z in zs for x in xs for y in ys]
    pts = []
    for a, b in _BOX_EDGES_12:
        pts.append(corners[a])
        pts.append(corners[b])
    return np.array(pts, dtype=float)


def box_outline_2d(cx, cy, half_w=PERSON_HALF_W):
    """Closed-loop (x[], y[]) outline of the top-down bounding box."""
    xs = [cx - half_w, cx + half_w, cx + half_w, cx - half_w, cx - half_w]
    ys = [cy - half_w, cy - half_w, cy + half_w, cy + half_w, cy - half_w]
    return xs, ys


# ---- [1. DETECTION VOLUME BOX] geometry ------------------------------------
def _aabb_corners(minx, maxx, miny, maxy, minz, maxz):
    xs, ys, zs = (minx, maxx), (miny, maxy), (minz, maxz)
    return [(x, y, z) for z in zs for x in xs for y in ys]


def _dash_segment(p1, p2, dash_len=BOUNDARY_BOX_DASH_LEN, gap_len=BOUNDARY_BOX_GAP_LEN):
    """Chop one edge into short dash/gap sub-segments. pyqtgraph's
    GLLinePlotItem doesn't expose real GL line-stippling, so this is how we
    approximate a dashed line: a list of short, separated (p1,p2) pairs
    covering the edge instead of one continuous segment."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    length = float(np.linalg.norm(p2 - p1))
    if length < 1e-6:
        return []
    direction = (p2 - p1) / length
    segments = []
    t = 0.0
    while t < length:
        seg_end_t = min(t + dash_len, length)
        segments.append(p1 + direction * t)
        segments.append(p1 + direction * seg_end_t)
        t = seg_end_t + gap_len
    return segments


def boundary_box_line_vertices(bb, transform_fn, dashed=BOUNDARY_BOX_DASHED):
    """Build the vertex-pair array for the detection-volume wireframe.

    [FIX] `bb` is now a dict with minX/maxX/minY/maxY/minZ/maxZ in ROOM
    coordinates (floor-aligned, Z=0 at the floor) -- see BOUNDARY_BOX_ROOM.
    `transform_fn` should be a HEIGHT-ONLY transform (see
    make_height_only_transform()), NOT the tilt-rotating make_transform()
    used for the point cloud/tracks. This keeps the box axis-aligned with
    the floor grid regardless of the sensor's mounting tilt -- it's a
    fixed reference volume of the room, not something captured through the
    sensor's tilted view.
    """
    corners_raw = np.array(
        _aabb_corners(bb["minX"], bb["maxX"], bb["minY"], bb["maxY"], bb["minZ"], bb["maxZ"]),
        dtype=float)
    corners_room = transform_fn(corners_raw)


    pts = []
    for a, b in _BOX_EDGES_12:
        p1, p2 = corners_room[a], corners_room[b]
        if dashed:
            pts.extend(_dash_segment(p1, p2))
        else:
            pts.append(p1)
            pts.append(p2)

    if not pts:
        return np.zeros((2, 3), dtype=float)
    return np.array(pts, dtype=float)


# =============================================================================
# Per-track render artists (one set for the 3D view, one for the 2D view)
# =============================================================================
def track_qcolor(tid, hues=12):
    """[2. PER-TRACK COLOR] Deterministic, stable color for a track ID.
    Same tid always -> same color, so a tracked person keeps their color
    for as long as their track ID is alive, and it's consistent between
    the 3D and 2D views since both derive from this single function."""
    return pg.intColor(int(tid) % 997, hues=hues)


def _qcolor_to_gl_rgba(qcolor, alpha=None):
    """QColor -> (r,g,b,a) floats in [0,1] for GL items."""
    a = qcolor.alphaF() if alpha is None else alpha
    return (qcolor.redF(), qcolor.greenF(), qcolor.blueF(), a)


class TrackArtists3D:
    """[2. PER-TRACK COLOR] `color` is a per-tid RGBA float tuple, shared by
    the box outline and the marker; the trail reuses the same hue at lower
    alpha so it reads as "this person's trail" while staying visually
    distinct from the solid box.

    [3. STABILITY] update()/remove() validate their inputs and never raise:
    a track that vanishes mid-update or carries a bad position is simply
    skipped for that frame instead of crashing the render callback.
    """

    def __init__(self, gl_view, color):
        self.color = color
        trail_color = (color[0], color[1], color[2], min(color[3], 0.75) * 0.7)
        self.box = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=color,
                                      width=2.0, mode="lines", antialias=True)
        self.trail = gl.GLLinePlotItem(pos=np.zeros((1, 3)), color=trail_color,
                                        width=2.0, mode="line_strip", antialias=True)
        self.marker = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=color, size=12)
        gl_view.addItem(self.box)
        gl_view.addItem(self.trail)
        gl_view.addItem(self.marker)

    def update(self, pos, history):
        # [3. STABILITY] Defensive checks: don't let a bad position or an
        # empty/garbled history propagate into a GL call.
        try:
            if pos is None or len(pos) != 3:
                return
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            if not all(math.isfinite(v) for v in (x, y, z)):
                return

            self.box.setData(pos=box_line_vertices(x, y, z))
            self.marker.setData(pos=np.array([[x, y, z]]))

            hist_arr = np.asarray(history, dtype=float) if history else np.empty((0, 3))
            if hist_arr.ndim == 2 and hist_arr.shape[1] == 3 and len(hist_arr) > 0:
                finite = np.all(np.isfinite(hist_arr), axis=1)
                hist_arr = hist_arr[finite]
            else:
                hist_arr = np.empty((0, 3))

            if len(hist_arr) > 1:
                self.trail.setData(pos=hist_arr)
            else:
                self.trail.setData(pos=np.array([[x, y, z], [x, y, z]]))
        except Exception as e:
            log.debug("TrackArtists3D.update failed, skipping this frame: %s", e)

    def remove(self, gl_view):
        for item in (self.box, self.trail, self.marker):
            try:
                gl_view.removeItem(item)
            except Exception as e:
                log.debug("TrackArtists3D.remove: failed to remove item: %s", e)


class TrackArtists2D:
    """[2. PER-TRACK COLOR] `qcolor` is the same per-tid QColor used for the
    3D marker/box (via track_qcolor()), so the 2D view matches the 3D view
    for a given person. The trail uses the same color at reduced alpha.

    [3. STABILITY] update()/remove() are defensive for the same reasons as
    TrackArtists3D.
    """

    def __init__(self, plot_widget, qcolor):
        trail_qcolor = QtGui.QColor(qcolor)
        trail_qcolor.setAlphaF(max(0.25, trail_qcolor.alphaF() * 0.55))
        pen_box = pg.mkPen(color=qcolor, width=2)
        pen_trail = pg.mkPen(color=trail_qcolor, width=2)
        self.box = pg.PlotCurveItem(pen=pen_box)
        self.trail = pg.PlotCurveItem(pen=pen_trail)
        self.marker = pg.ScatterPlotItem(size=10, brush=qcolor, pen=pg.mkPen(None))
        plot_widget.addItem(self.box)
        plot_widget.addItem(self.trail)
        plot_widget.addItem(self.marker)

    def update(self, pos, history):
        try:
            if pos is None or len(pos) != 3:
                return
            x, y = float(pos[0]), float(pos[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                return

            bx, by = box_outline_2d(x, y)
            self.box.setData(bx, by)
            self.marker.setData([x], [y])

            hist_arr = np.asarray(history, dtype=float) if history else np.empty((0, 3))
            if hist_arr.ndim == 2 and hist_arr.shape[1] >= 2 and len(hist_arr) > 0:
                finite = np.all(np.isfinite(hist_arr[:, :2]), axis=1)
                hist_arr = hist_arr[finite]
            else:
                hist_arr = np.empty((0, 2))

            if len(hist_arr) > 1:
                self.trail.setData(hist_arr[:, 0], hist_arr[:, 1])
            else:
                self.trail.setData([x], [y])
        except Exception as e:
            log.debug("TrackArtists2D.update failed, skipping this frame: %s", e)

    def remove(self, plot_widget):
        for item in (self.box, self.trail, self.marker):
            try:
                plot_widget.removeItem(item)
            except Exception as e:
                log.debug("TrackArtists2D.remove: failed to remove item: %s", e)


# =============================================================================
# Main window
# =============================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("mmWave Radar Visualizer v3.2 - PyQtGraph/OpenGL")
        self.resize(1500, 850)

        # ---- [PEOPLE COUNTER] Always-visible occupancy readout ---------------
        # Count definition (must match drawn solid boxes):
        #   ACTIVE tracks only — missed == 0 AND room-frame Z in the human band.
        # Coasting / ghost / ROI-rejected tracks are NOT counted.
        toolbar = QtWidgets.QToolBar("Occupancy")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setIconSize(QtCore.QSize(0, 0))
        # PySide6 prefers ToolBarArea.TopToolBarArea; PySide2 has TopToolBarArea.
        try:
            _top = QtCore.Qt.ToolBarArea.TopToolBarArea
        except AttributeError:
            _top = QtCore.Qt.TopToolBarArea
        self.addToolBar(_top, toolbar)
        self.people_label = QtWidgets.QLabel("People: 0")
        self.people_label.setStyleSheet(
            "QLabel { font-size: 28px; font-weight: 700; padding: 6px 16px; "
            "color: #111; background: #f5f5f5; border: 1px solid #ccc; "
            "border-radius: 6px; }")
        toolbar.addWidget(self.people_label)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        self.setCentralWidget(central)

        # ---- Left: 3D OpenGL view -------------------------------------
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setCameraPosition(distance=18, elevation=25, azimuth=-90)
        grid = gl.GLGridItem()
        grid.setSize(x=ROOM["x"][1] - ROOM["x"][0], y=ROOM["y"][1] - ROOM["y"][0])
        grid.setSpacing(x=1, y=1)
        self.gl_view.addItem(grid)

        self.cloud_item_3d = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=4,
                                                    color=(1, 1, 1, 1), pxMode=True)
        self.gl_view.addItem(self.cloud_item_3d)

        sensor_pos = (0, 0, args.sensor_height)
        self.sensor_item = gl.GLLinePlotItem(
            pos=cube_line_vertices(*sensor_pos, size=0.3),
            color=(1.0, 0.0, 0.0, 1.0), width=2.5, mode="lines", antialias=True)
        self.gl_view.addItem(self.sensor_item)

        # ---- [1. DETECTION VOLUME BOX] ---------------------------------
        # [FIX] This box is a fixed, floor-aligned reference volume of the
        # ROOM (e.g. "the monitored area"), not something seen through the
        # sensor's tilted view -- so it must NOT rotate with the sensor's
        # tilt angle. We therefore build it with make_height_only_transform()
        # (height translation only) instead of make_transform() (which also
        # applies the tilt rotation and is reserved for the point cloud and
        # tracks -- see the call in ParserThread). Bright yellow, dashed,
        # semi-transparent so it reads as a reference overlay rather than
        # real sensor data. Static geometry -- computed once here, not
        # per-frame. Visible by default; toggle self.boundary_box_item if
        # you want a UI control to hide/show it later.
        self.boundary_box = args.boundary_box or BOUNDARY_BOX_ROOM
        boundary_box_room = self.boundary_box
        boundary_box_transform = make_height_only_transform(args.sensor_height)
        self.boundary_box_item = gl.GLLinePlotItem(
            pos=boundary_box_line_vertices(boundary_box_room, boundary_box_transform),
            color=BOUNDARY_BOX_COLOR, width=BOUNDARY_BOX_WIDTH,
            mode="lines", antialias=True)
        self.gl_view.addItem(self.boundary_box_item)

        # ---- Right: 2D top-down view ------------------------------------
        self.plot_2d = pg.PlotWidget()
        self.plot_2d.setBackground("w")
        self.plot_2d.showGrid(x=True, y=True, alpha=0.3)
        self.plot_2d.setLabel("bottom", "X (m)")
        self.plot_2d.setLabel("left", "Y (m)")
        self.plot_2d.setXRange(*ROOM["x"])
        self.plot_2d.setYRange(*ROOM["y"])
        self.plot_2d.setAspectLocked(True)

        self.cloud_item_2d = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None))
        self.plot_2d.addItem(self.cloud_item_2d)

        sensor_marker_2d = pg.ScatterPlotItem(
            size=14, symbol="s", brush=pg.mkBrush(255, 0, 0), pen=pg.mkPen(None))
        sensor_marker_2d.setData([0], [0])
        self.plot_2d.addItem(sensor_marker_2d)

        layout.addWidget(self.gl_view, stretch=3)
        layout.addWidget(self.plot_2d, stretch=2)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Starting...")

        self.artists_3d = {}
        self.artists_2d = {}

        # ---- Parser thread (producer) ------------------------------------
        self.thread = ParserThread(
            data_port_name=args.data_port,
            data_baud=args.data_baud,
            demo_name=args.demo,
            sensor_height=args.sensor_height,
            tilt_deg=args.tilt,
            accum_frames=args.accum_frames,
            max_points=args.max_points,
            boundary_box=self.boundary_box,
        )
        self.thread.snapshotReady.connect(self.on_snapshot)
        self.thread.statusMessage.connect(self.on_status)
        self.thread.errorOccurred.connect(self.on_error)
        self.thread.start()

    # -------------------------------------------------------------------
    def on_status(self, msg):
        log.info(msg)
        self.status_bar.showMessage(msg)

    def on_error(self, msg):
        log.error(msg)
        self.status_bar.showMessage("ERROR: " + msg)

    def on_snapshot(self, snapshot):
        # [3. STABILITY] Top-level guard: whatever else happens, a single
        # bad snapshot must never crash the GUI thread / the whole app.
        try:
            self._on_snapshot_inner(snapshot)
        except Exception as e:
            log.error("on_snapshot failed unexpectedly (frame skipped): %s", e)

    def _on_snapshot_inner(self, snapshot):
        cloud = snapshot.get("cloud", np.empty((0, 4)))
        tracks = snapshot.get("tracks", {})

        # [STABILITY FIX] Strip any non-finite rows before they reach the GL
        # widgets -- a NaN/Inf vertex can crash or hang some OpenGL drivers,
        # and this is cheap insurance against any parser edge case that
        # slips through upstream sanity checks.
        if len(cloud) > 0:
            finite_mask = np.all(np.isfinite(cloud), axis=1)
            if not finite_mask.all():
                log.debug("Dropping %d non-finite point(s) before render",
                          int((~finite_mask).sum()))
            cloud = cloud[finite_mask]
            # [GHOST FIX] Belt-and-suspenders: strip out-of-volume-Z points
            # that may still be sitting in the accumulation buffer.
            try:
                cloud = filter_cloud_by_volume_z(cloud, bb=getattr(self, "boundary_box", None))
            except Exception as e:
                log.debug("cloud Z volume filter failed: %s", e)

        # ---- point cloud, 3D -------------------------------------------
        # [3. STABILITY] Each render section is independently guarded: a
        # failure updating the 3D cloud shouldn't prevent the 2D cloud or
        # the tracks from still being updated this frame.
        try:
            if len(cloud) > 0:
                colors = snr_to_rgba_float(cloud[:, 3])
                self.cloud_item_3d.setData(pos=cloud[:, :3], color=colors, size=4)
            else:
                self.cloud_item_3d.setData(pos=np.zeros((0, 3)))
        except Exception as e:
            log.debug("3D point-cloud render failed, skipping: %s", e)

        # ---- point cloud, 2D top-down -----------------------------------
        try:
            if len(cloud) > 0:
                brushes = snr_to_brushes(cloud[:, 3])
                self.cloud_item_2d.setData(x=cloud[:, 0], y=cloud[:, 1], brush=brushes)
            else:
                self.cloud_item_2d.setData(x=[], y=[])
        except Exception as e:
            log.debug("2D point-cloud render failed, skipping: %s", e)

        # ---- tracks: bounding boxes + trails, 3D and 2D -----------------
        # [PEOPLE COUNTER] / [GHOST FIX]
        # Draw AND count only ACTIVE tracks: missed == 0 AND room Z plausible.
        # Coasting tracks (missed > 0) are hidden — never left as a solid
        # floating box above the yellow volume.
        active = 0
        drawable_ids = set()
        for tid, st in tracks.items():
            try:
                pos = st.get("pos")
                missed = int(st.get("missed", 0) or 0)
                if pos is None or len(pos) != 3 or not all(math.isfinite(v) for v in pos):
                    continue
                if missed != 0:
                    # Hide coasting / aged tracks (remove artists if present).
                    continue
                if not room_z_plausible(pos[2]):
                    log_ghost_reject(tid, (float("nan"),) * 3, pos, missed,
                                     "gui_skip_room_z_out_of_band")
                    continue
                if not in_boundary_box(pos[0], pos[1], pos[2], bb=getattr(self, "boundary_box", None)):
                    log_ghost_reject(tid, (float("nan"),) * 3, pos, missed,
                                     "gui_skip_outside_boundary_box")
                    continue

                drawable_ids.add(tid)
                if tid not in self.artists_3d:
                    qcolor = track_qcolor(tid)
                    gl_color = _qcolor_to_gl_rgba(qcolor, alpha=1.0)
                    self.artists_3d[tid] = TrackArtists3D(self.gl_view, gl_color)
                    self.artists_2d[tid] = TrackArtists2D(self.plot_2d, qcolor)

                self.artists_3d[tid].update(pos, st.get("history", []))
                self.artists_2d[tid].update(pos, st.get("history", []))
                active += 1
            except Exception as e:
                log.debug("Failed to render track %s, skipping it this frame: %s", tid, e)

        # ---- garbage-collect artists not in the drawable active set -----
        try:
            for tid in list(self.artists_3d.keys()):
                if tid not in drawable_ids:
                    self.artists_3d[tid].remove(self.gl_view)
                    self.artists_2d[tid].remove(self.plot_2d)
                    del self.artists_3d[tid]
                    del self.artists_2d[tid]
        except Exception as e:
            log.debug("Track artist cleanup failed: %s", e)

        try:
            # [PEOPLE COUNTER] Keep the big toolbar label in sync with `active`.
            if hasattr(self, "people_label") and self.people_label is not None:
                self.people_label.setText(f"People: {active}")
        except Exception as e:
            log.debug("People counter update failed: %s", e)

        try:
            self.status_bar.showMessage(
                f"Frame {snapshot.get('frame_count', 0)} | {active} active people | "
                f"{len(cloud)} points | parse errors: {snapshot.get('errors', 0)}")
        except Exception:
            pass

    # -------------------------------------------------------------------
    def closeEvent(self, event):
        # [3. STABILITY] Give the thread a real chance to exit its serial
        # read loop (it polls self._running every ~50ms via the read
        # timeout, so this should normally return almost immediately). Only
        # as an absolute last resort do we force-terminate, which is unsafe
        # in general (it can leave the serial port in a bad state) but far
        # better than leaking a running QThread when the window is closed.
        # The whole thing is wrapped so a shutdown-path error still lets the
        # window close instead of hanging the app.
        try:
            log.info("Window closing: stopping parser thread...")
            self.thread.stop()
            if not self.thread.wait(3000):
                log.warning("Parser thread did not stop within 3s; forcing termination.")
                self.thread.terminate()
                self.thread.wait(1000)
        except Exception as e:
            log.error("Error while stopping parser thread on close: %s", e)
        event.accept()


# =============================================================================
# CLI / entry point
# =============================================================================
def parse_boundary_box_arg(s):
    """--boundary-box "minX,maxX,minY,maxY,minZ,maxZ" -> dict, for the
    [1. DETECTION VOLUME BOX] wireframe. Raises ArgumentTypeError (which
    argparse turns into a clean CLI error) on malformed input."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "--boundary-box needs exactly 6 comma-separated values: "
            "minX,maxX,minY,maxY,minZ,maxZ")
    try:
        vals = [float(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--boundary-box values must be numbers: {e}")
    keys = ["minX", "maxX", "minY", "maxY", "minZ", "maxZ"]
    return dict(zip(keys, vals))


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Real-time 3D mmWave radar visualizer (PyQtGraph + OpenGL).")
    ap.add_argument("--data-port", required=True,
                     help="Serial port for the radar's DATA UART (e.g. COM6 or /dev/ttyUSB1)")
    ap.add_argument("--data-baud", type=int, default=921600)
    ap.add_argument("--cli-port", default=None,
                     help="Serial port for the radar's CLI/user UART (needed to send --cfg)")
    ap.add_argument("--cli-baud", type=int, default=115200)
    ap.add_argument("--cfg", default=None, help="Path to a TI .cfg profile to send before streaming")
    ap.add_argument("--demo", default=demo_defines_mod.DEMO_3D_PEOPLE_TRACKING,
                     help="Demo name (selects frame-header layout in parseFrame.py)")
    ap.add_argument("--accum-frames", type=int, default=ACCUM_FRAMES)
    ap.add_argument("--max-points", type=int, default=PC_MAX_POINTS)
    ap.add_argument("--sensor-height", type=float, default=SENSOR_HEIGHT)
    ap.add_argument("--tilt", type=float, default=TILT_DEG)
    ap.add_argument("--debug", action="store_true",
                     help="Verbose logging: TLV-1010 struct selection, track "
                          "counts, sanity-check rejects, raw byte previews.")
    ap.add_argument("--boundary-box", type=parse_boundary_box_arg, default=None,
                     help="[1. DETECTION VOLUME BOX] Override the 3D-view detection "
                          "volume wireframe: 'minX,maxX,minY,maxY,minZ,maxZ' in ROOM "
                          "coordinates, metres (floor-aligned: X=left/right, "
                          "Y=forward/depth, Z=height above the floor, Z=0 at the "
                          "floor). This is a fixed visual reference box for the "
                          "monitored area -- it does NOT rotate with the sensor's "
                          "tilt. Roughly match it to your .cfg's boundaryBox / "
                          f"occupancy zone if needed. Default: {BOUNDARY_BOX_ROOM}")
    return ap


def main():
    args = build_arg_parser().parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        log.debug("Debug logging enabled.")

    if args.cfg:
        if not args.cli_port:
            log.error("--cfg requires --cli-port to be specified.")
            sys.exit(1)
        try:
            send_cfg(args.cli_port, args.cfg, args.cli_baud)
        except Exception as e:
            log.error("Failed to send config: %s", e)
            sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(args)

    # [STABILITY FIX] Belt-and-suspenders: closeEvent() handles the normal
    # "user closed the window" path, but aboutToQuit also fires for other
    # exit paths (e.g. app.quit() called elsewhere, some OS window-manager
    # close sequences). Stopping the thread here too guarantees it's never
    # left running when the QThread object is destroyed at interpreter exit.
    def _ensure_thread_stopped():
        if win.thread.isRunning():
            log.info("aboutToQuit: stopping parser thread...")
            win.thread.stop()
            win.thread.wait(2000)

    app.aboutToQuit.connect(_ensure_thread_stopped)

    win.show()
    sys.exit(app.exec() if QT_LIB == "PySide6" else app.exec_())


if __name__ == "__main__":
    main()