"""OpenCV camera open helpers (Windows DSHOW / Linux V4L2)."""

from __future__ import annotations

import sys


def open_camera(device=1, width=160, height=120, backend=None):
    """
    Open a capture. Prefer device 1 (PureThermal often lands there on Windows
    when device 0 is a laptop webcam). Returns (cap, info_str) or (None, err).
    """
    try:
        import cv2
    except ImportError:
        return None, "opencv not installed"

    backends = []
    if backend:
        backends.append(backend)
    if sys.platform.startswith("win"):
        backends.extend([getattr(cv2, "CAP_DSHOW", 700), getattr(cv2, "CAP_MSMF", 1400)])
    backends.append(getattr(cv2, "CAP_ANY", 0))

    last_err = "no backend"
    for be in backends:
        try:
            cap = cv2.VideoCapture(int(device), be)
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            continue
        if not cap.isOpened():
            cap.release()
            last_err = f"device {device} backend {be} failed to open"
            continue
        # Prefer GREY / Y16 when the driver allows it (radiometric path).
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        except Exception:
            pass
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            last_err = f"device {device} opened but read failed"
            continue
        info = (
            f"dev={device} be={be} shape={tuple(frame.shape)} "
            f"dtype={frame.dtype}"
        )
        return cap, info
    return None, last_err


def frame_to_bgr(frame):
    """Normalize greyscale / Y16 / BGR into a displayable BGR uint8 image."""
    import numpy as np
    import cv2

    if frame is None:
        return None
    if frame.ndim == 2:
        if frame.dtype == np.uint16:
            # Rough radiometric-ish stretch for display only.
            lo, hi = np.percentile(frame, (2, 98))
            if hi <= lo:
                hi = lo + 1
            norm = np.clip((frame.astype(np.float32) - lo) / (hi - lo), 0, 1)
            u8 = (norm * 255).astype(np.uint8)
        else:
            u8 = frame.astype(np.uint8)
        return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.applyColorMap(frame[:, :, 0].astype(np.uint8), cv2.COLORMAP_INFERNO)
    return None


def synthetic_thermal(h=120, w=160, blobs=None):
    """Dark canvas with optional warm blobs for smoke tests (no camera)."""
    import numpy as np

    img = np.full((h, w), 40, dtype=np.uint8)
    for b in blobs or []:
        x, y, rw, rh, temp = b
        x1, y1 = max(0, x - rw), max(0, y - rh)
        x2, y2 = min(w, x + rw), min(h, y + rh)
        img[y1:y2, x1:x2] = int(temp)
    return img
