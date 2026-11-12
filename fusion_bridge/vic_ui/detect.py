"""Optional YOLO person detector (ultralytics). Soft-fails if missing."""

from __future__ import annotations

import os


class Detector:
    def __init__(self, weights=None, conf=0.20, device="cpu"):
        self.weights = weights
        self.conf = float(conf)
        self.device = device
        self.model = None
        self.err = None
        self.backend = "none"
        if weights and os.path.isfile(weights):
            self._try_load(weights)
        elif weights:
            self.err = f"weights missing: {weights}"

    def _try_load(self, path):
        try:
            from ultralytics import YOLO  # type: ignore
            self.model = YOLO(path)
            self.backend = "ultralytics"
            self.err = None
        except Exception as e:  # noqa: BLE001
            self.model = None
            self.backend = "none"
            self.err = f"YOLO load failed: {type(e).__name__}: {e}"

    @property
    def ready(self):
        return self.model is not None

    def set_conf(self, conf):
        self.conf = float(max(0.05, min(0.95, conf)))

    def detect(self, bgr):
        """Return list[{box:[x1,y1,x2,y2], conf, cls}]. Empty if no model."""
        if self.model is None or bgr is None:
            return []
        try:
            results = self.model.predict(
                source=bgr, conf=self.conf, verbose=False, device=self.device
            )
        except Exception as e:  # noqa: BLE001
            self.err = f"predict: {type(e).__name__}: {e}"
            return []
        out = []
        for r in results or []:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                try:
                    xyxy = b.xyxy[0].tolist()
                    conf = float(b.conf[0]) if b.conf is not None else 0.0
                    cls = int(b.cls[0]) if b.cls is not None else 0
                except Exception:
                    continue
                out.append({
                    "box": [float(xyxy[0]), float(xyxy[1]),
                            float(xyxy[2]), float(xyxy[3])],
                    "conf": conf,
                    "cls": cls,
                })
        return out
