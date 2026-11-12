"""Dual-pane compositor: radar 3D-ish (left) + thermal (right).

Left pane approximates live_32d_track OpenGL style with OpenCV:
  perspective floor grid, dashed yellow detection volume, orange person
  wire boxes, optional point cloud when the UDP/JSONL frame carries points.
"""

from __future__ import annotations

import math
import time


# Panel sizes (final window = LEFT_W + RIGHT_W  x  PANEL_H + STATUS_H)
LEFT_W = 480
RIGHT_W = 640
PANEL_H = 480
STATUS_H = 36
TITLE_H = 0

# Default room detection volume (matches my_occupancy.cfg boundaryBox /
# live_32d_track BOUNDARY_BOX_ROOM).
BOUNDARY_DEFAULT = {
    "minX": -2.8, "maxX": 2.8,
    "minY": 0.6, "maxY": 5.5,
    "minZ": 0.0, "maxZ": 2.5,
}

PERSON_HALF_W = 0.30
PERSON_HEIGHT = 1.75

# 12 edges of an AABB whose corners are ordered
# [(x,y,z) for z in (z0,z1) for x in (x0,x1) for y in (y0,y1)].
_BOX_EDGES_12 = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# Floor grid extents for drawing (metres)
GRID_X = (-4.0, 4.0)
GRID_Y = (0.0, 8.0)
GRID_STEP = 1.0


# ---------------------------------------------------------------------------
# Perspective camera (room frame: X right, Y forward, Z up)
# ---------------------------------------------------------------------------
class _PerspCam:
    """Simple look-at perspective projector → pixel coords."""

    def __init__(self, w=LEFT_W, h=PANEL_H,
                 eye=(-3.8, -4.2, 5.8),
                 target=(0.0, 3.2, 0.9),
                 up=(0.0, 0.0, 1.0),
                 fov_deg=48.0):
        self.w = int(w)
        self.h = int(h)
        self.eye = _v3(eye)
        self.target = _v3(target)
        self.up = _v3(up)
        self.fov = float(fov_deg)
        self._build()

    def _build(self):
        f = _norm(_sub(self.target, self.eye))
        s = _norm(_cross(f, self.up))
        u = _cross(s, f)
        self._R = (s, u, f)  # camera axes in world
        self._f = 0.5 * self.h / math.tan(math.radians(self.fov) * 0.5)

    def project(self, p):
        """World (x,y,z) -> (u, v, depth) or None if behind camera."""
        d = _sub(p, self.eye)
        s, u, f = self._R
        x = _dot(d, s)
        y = _dot(d, u)
        z = _dot(d, f)  # forward depth (+ in front)
        if z < 0.15:
            return None
        u_px = self.w * 0.5 + self._f * (x / z)
        v_px = self.h * 0.5 - self._f * (y / z)
        return int(round(u_px)), int(round(v_px)), float(z)


def _v3(p):
    return (float(p[0]), float(p[1]), float(p[2]))


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a):
    n = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


def _aabb_corners(minx, maxx, miny, maxy, minz, maxz):
    xs, ys, zs = (minx, maxx), (miny, maxy), (minz, maxz)
    return [(x, y, z) for z in zs for x in xs for y in ys]


def _dash_segment(p1, p2, dash_len=0.28, gap_len=0.18):
    p1 = _v3(p1)
    p2 = _v3(p2)
    d = _sub(p2, p1)
    length = math.sqrt(_dot(d, d))
    if length < 1e-6:
        return []
    direction = _mul(d, 1.0 / length)
    segs = []
    t = 0.0
    while t < length:
        te = min(t + dash_len, length)
        segs.append(_add(p1, _mul(direction, t)))
        segs.append(_add(p1, _mul(direction, te)))
        t = te + gap_len
    return segs


def _draw_line_3d(img, cam, p1, p2, color, thickness=1):
    import cv2
    a = cam.project(p1)
    b = cam.project(p2)
    if a is None or b is None:
        return
    cv2.line(img, (a[0], a[1]), (b[0], b[1]), color, thickness, cv2.LINE_AA)


def _draw_polyline_pairs(img, cam, pairs, color, thickness=1):
    """pairs: flat list [p0,p1, p2,p3, ...] of world points."""
    for i in range(0, len(pairs) - 1, 2):
        _draw_line_3d(img, cam, pairs[i], pairs[i + 1], color, thickness)


def _person_box_corners(t, half_w=PERSON_HALF_W):
    """Floor-anchored person AABB from a track dict (uses z_min/z_max)."""
    cx = float(t.get("x", 0.0))
    cy = float(t.get("y", 0.0))
    z_min = t.get("z_min")
    z_max = t.get("z_max")
    cz = float(t.get("z", 0.9))
    if z_min is None:
        z0 = max(0.0, cz - 0.95)
    else:
        z0 = max(0.0, float(z_min))
    if z_max is None:
        z1 = z0 + PERSON_HEIGHT
    else:
        z1 = max(z0 + 0.4, float(z_max))
    return _aabb_corners(cx - half_w, cx + half_w,
                         cy - half_w, cy + half_w, z0, z1)


def _track_color(tid):
    # Orange-ish palette matching live_32d_track person boxes; vary by tid.
    palette = [
        (40, 140, 255),   # orange (BGR)
        (50, 80, 230),    # red-orange
        (30, 180, 255),   # light orange
        (60, 120, 220),
        (20, 160, 240),
        (80, 100, 255),
        (40, 200, 200),
        (90, 140, 255),
    ]
    return palette[int(tid) % len(palette)]


def draw_radar_pane(tracks, trails=None, show_trails=False, count=None,
                    count_label="people", boundary=None, points=None,
                    sensor_height=2.4):
    """Perspective 3D-ish radar pane (BGR), live_32d_track style."""
    import numpy as np
    import cv2

    img = np.full((PANEL_H, LEFT_W, 3), 16, dtype=np.uint8)
    cam = _PerspCam(LEFT_W, PANEL_H)
    bb = dict(BOUNDARY_DEFAULT if boundary is None else boundary)

    # --- floor grid ---
    grid_col = (55, 55, 62)
    x0, x1 = GRID_X
    y0, y1 = GRID_Y
    x = x0
    while x <= x1 + 1e-6:
        _draw_line_3d(img, cam, (x, y0, 0.0), (x, y1, 0.0), grid_col, 1)
        x += GRID_STEP
    y = y0
    while y <= y1 + 1e-6:
        _draw_line_3d(img, cam, (x0, y, 0.0), (x1, y, 0.0), grid_col, 1)
        y += GRID_STEP

    # Axis hint on floor
    _draw_line_3d(img, cam, (0, 0, 0), (1.2, 0, 0), (70, 70, 180), 2)  # +X
    _draw_line_3d(img, cam, (0, 0, 0), (0, 1.5, 0), (70, 180, 70), 2)  # +Y

    # --- sensor marker (elevated at origin XY) ---
    sh = float(sensor_height)
    _draw_line_3d(img, cam, (0, 0, 0), (0, 0, sh), (160, 100, 80), 1)
    p_s = cam.project((0.0, 0.0, sh))
    if p_s is not None:
        cv2.circle(img, (p_s[0], p_s[1]), 5, (100, 100, 220), -1, cv2.LINE_AA)
        cv2.putText(img, "sensor", (p_s[0] + 6, p_s[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 180), 1,
                    cv2.LINE_AA)

    # --- dashed yellow detection volume ---
    yellow = (0, 230, 255)  # BGR bright yellow
    corners = _aabb_corners(bb["minX"], bb["maxX"], bb["minY"], bb["maxY"],
                            bb["minZ"], bb["maxZ"])
    dash_pts = []
    for a, b in _BOX_EDGES_12:
        dash_pts.extend(_dash_segment(corners[a], corners[b]))
    _draw_polyline_pairs(img, cam, dash_pts, yellow, thickness=1)

    # --- optional point cloud ---
    if points:
        for pt in points:
            try:
                if isinstance(pt, dict):
                    px, py, pz = float(pt["x"]), float(pt["y"]), float(pt.get("z", 0))
                    snr = float(pt.get("snr", pt.get("s", 12)))
                else:
                    px, py, pz = float(pt[0]), float(pt[1]), float(pt[2])
                    snr = float(pt[3]) if len(pt) > 3 else 12.0
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            pr = cam.project((px, py, pz))
            if pr is None:
                continue
            # SNR → purple/blue/white-ish (BGR)
            t = max(0.0, min(1.0, (snr - 5.0) / 25.0))
            col = (
                int(180 + 60 * t),          # B
                int(80 + 100 * (1 - t)),    # G
                int(200 + 55 * t),          # R → purple-white
            )
            cv2.circle(img, (pr[0], pr[1]), 1 if t < 0.5 else 2, col, -1)

    # --- trails on floor (optional) ---
    if show_trails and trails:
        for tid, pts in trails.items():
            if len(pts) < 2:
                continue
            color = _track_color(tid)
            for i in range(1, len(pts)):
                p0 = (pts[i - 1][0], pts[i - 1][1], 0.02)
                p1 = (pts[i][0], pts[i][1], 0.02)
                _draw_line_3d(img, cam, p0, p1, color, 2)

    # --- person wire boxes ---
    for t in tracks or []:
        tid = int(t.get("id", 0))
        color = _track_color(tid)
        corners_p = _person_box_corners(t)
        for a, b in _BOX_EDGES_12:
            _draw_line_3d(img, cam, corners_p[a], corners_p[b], color, 2)
        # Label at top of box
        top = (
            float(t.get("x", 0.0)),
            float(t.get("y", 0.0)),
            float(t.get("z_max") if t.get("z_max") is not None
                  else float(t.get("z", 0.9)) + 0.8),
        )
        pr = cam.project(top)
        if pr is not None:
            label = f"{tid}"
            rng = t.get("range")
            if rng is not None:
                label += f" {float(rng):.1f}m"
            cv2.putText(img, label, (pr[0] + 4, pr[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # Title + occupancy count overlay
    cv2.putText(img, "radar 3D", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (160, 160, 170), 1, cv2.LINE_AA)
    if count is not None:
        cv2.putText(img, str(int(count)), (14, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.4, (40, 220, 120), 4, cv2.LINE_AA)
        cv2.putText(img, str(count_label), (18, 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 180, 150), 1, cv2.LINE_AA)

    return img


def draw_thermal_pane(bgr, confirmed, thermal_only, radar_only,
                      show_unconfirmed=False, show_trails=False,
                      trail_pts=None, scale_to=(RIGHT_W, PANEL_H),
                      always_draw_radar=True, flip_v=False):
    """Scale thermal BGR and overlay confirmed / radar boxes.

    flip_v: vertically flip the thermal image (Lepton often mounts upside-down).
    Radar projections (radar_only) are drawn even when ass=0 /
    show_unconfirmed is False when always_draw_radar is True.
    """
    import numpy as np
    import cv2

    tw, th = scale_to
    if bgr is None:
        canvas = np.full((th, tw, 3), 28, dtype=np.uint8)
        cv2.putText(canvas, "no thermal", (tw // 2 - 70, th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 130), 1, cv2.LINE_AA)
    else:
        src = cv2.flip(bgr, 0) if flip_v else bgr
        canvas = cv2.resize(src, (tw, th), interpolation=cv2.INTER_NEAREST)

    src_h, src_w = (bgr.shape[0], bgr.shape[1]) if bgr is not None else (120, 160)
    sx, sy = tw / float(src_w), th / float(src_h)

    def scale_box(box, flip=False):
        x1, y1, x2, y2 = box
        if flip:
            # Vertical flip in source pixel space before scaling to pane.
            y1, y2 = (src_h - 1 - y2), (src_h - 1 - y1)
        return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

    cv2.putText(canvas, "thermal", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 210), 1, cv2.LINE_AA)
    if flip_v:
        cv2.putText(canvas, "flip", (tw - 50, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (120, 120, 140), 1, cv2.LINE_AA)

    # Confirmed: solid green thermal box + cyan radar proj
    for c in confirmed or []:
        td, rd = c["thermal"], c["radar"]
        x1, y1, x2, y2 = scale_box(td["box"], flip=flip_v)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 220, 120), 2)
        rid = rd.get("id", "?")
        rng = rd.get("range_m", rd.get("range"))
        conf = td.get("conf", 0)
        label = f"#{rid}"
        if rng is not None:
            label += f" {float(rng):.1f}m"
        label += f" {conf:.2f}"
        cv2.putText(canvas, label, (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 220, 120), 1, cv2.LINE_AA)
        if rd.get("box"):
            rx1, ry1, rx2, ry2 = scale_box(rd["box"], flip=flip_v)
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (220, 200, 40), 1)

    if always_draw_radar:
        for rd in radar_only or []:
            if not rd.get("box"):
                continue
            x1, y1, x2, y2 = scale_box(rd["box"], flip=flip_v)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 200, 220), 2)
            rid = rd.get("id", "?")
            rng = rd.get("range_m", rd.get("range"))
            label = f"R#{rid}"
            if rng is not None:
                label += f" {float(rng):.1f}m"
            cv2.putText(canvas, label, (x1, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 200, 220), 1,
                        cv2.LINE_AA)

    if show_unconfirmed:
        for td in thermal_only or []:
            x1, y1, x2, y2 = scale_box(td["box"], flip=flip_v)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 80, 200), 1)

    if show_trails and trail_pts:
        for tid, pts in trail_pts.items():
            if len(pts) < 2:
                continue
            for i in range(1, len(pts)):
                def _flip_pt(p):
                    x, y = p[0], p[1]
                    if flip_v:
                        y = src_h - 1 - y
                    return (int(x * sx), int(y * sy))
                p0 = _flip_pt(pts[i - 1])
                p1 = _flip_pt(pts[i])
                cv2.line(canvas, p0, p1, (180, 180, 80), 1)

    return canvas


def draw_status(width, lines, fps=None):
    import numpy as np
    import cv2

    bar = np.full((STATUS_H, width, 3), 12, dtype=np.uint8)
    text = "  |  ".join(lines)
    if fps is not None:
        text += f"  |  {fps:.0f} fps"
    cv2.putText(bar, text[:140], (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 190), 1, cv2.LINE_AA)
    return bar


def compose(radar_bgr, thermal_bgr, status_lines, fps=None):
    import numpy as np
    import cv2

    left = radar_bgr
    right = thermal_bgr
    if left.shape[0] != PANEL_H or left.shape[1] != LEFT_W:
        left = cv2.resize(left, (LEFT_W, PANEL_H))
    if right.shape[0] != PANEL_H or right.shape[1] != RIGHT_W:
        right = cv2.resize(right, (RIGHT_W, PANEL_H))
    row = np.hstack([left, right])
    cv2.line(row, (LEFT_W, 0), (LEFT_W, PANEL_H - 1), (60, 60, 70), 2)
    bar = draw_status(LEFT_W + RIGHT_W, status_lines, fps=fps)
    return np.vstack([row, bar])


class TrailStore:
    """Keep short XY histories per track id (radar metres)."""

    def __init__(self, max_len=40):
        self.max_len = max_len
        self._pts = {}

    def update(self, tracks):
        seen = set()
        for t in tracks or []:
            tid = int(t["id"])
            seen.add(tid)
            lst = self._pts.setdefault(tid, [])
            lst.append((float(t["x"]), float(t["y"])))
            if len(lst) > self.max_len:
                del lst[: len(lst) - self.max_len]
        for tid in list(self._pts):
            if tid not in seen:
                del self._pts[tid]

    def clear(self):
        self._pts.clear()

    @property
    def data(self):
        return self._pts


class FpsMeter:
    def __init__(self, span=1.0):
        self.span = span
        self._t0 = time.time()
        self._n = 0
        self.value = 0.0

    def tick(self):
        self._n += 1
        now = time.time()
        dt = now - self._t0
        if dt >= self.span:
            self.value = self._n / dt
            self._t0 = now
            self._n = 0
        return self.value
