# Occupancy-detection

TI IWR6843 mmWave occupancy / people tracking (PySide + PyQtGraph OpenGL).

## Current versions

Use one of these two scripts at the repository root (pick the line you are running):

- **`live_32d_track_v6.17.py`** — IWR6843 only, OpenGL 3D + 2D top-down. Start here if you are not using the thermal camera.
- **`live_32d_track_v6.16.py`** — IWR6843 + FLIR Lepton / PureThermal fusion (3D + thermal pane).

Earlier snapshots are in [`history/`](history/), listed in version order in [`history/README.md`](history/README.md).

## Run (radar-only, v6.17)

```bash
python live_32d_track_v6.17.py --data-port COM6 --cli-port COM5 --cfg my_occupancy.cfg
```

On Linux, use the corresponding serial devices (for example `--data-port /dev/ttyUSB1 --cli-port /dev/ttyUSB0`).

## Run (radar + thermal, v6.16)

```bash
python live_32d_track_v6.16.py --data-port COM6 --cli-port COM5 --cfg my_occupancy.cfg --thermal --thermal-device 1 --thermal-weights models/v5/best.pt
```

YOLO / OpenCV are optional: the app soft-fails if they are missing. Default thermal weights are Haneef v5 (`models/v5/best.pt`).

## Dependencies

```bash
pip install pyserial numpy pyqtgraph PyOpenGL PySide6
```

v6.16 additionally uses `opencv-python` and `ultralytics` when the thermal pane is enabled. NumPy must be installed for the same Python version as the interpreter in your venv.

## Version order

| Version | Line | Layout | Where |
|---|---|---|---|
| v3 … v5 | Radar | Historical dual-view | `history/` |
| v6.3 … v6.11 | Radar | 3D + 2D | `history/` |
| v6.12 | Radar | 3D + 2D, JSONL/UDP export | `history/` |
| v6.13 | Radar + thermal | 3D + 2D + thermal | `history/` |
| v6.14 | Radar + thermal | 3D + thermal | `history/` |
| v6.15 | Radar + thermal | 3D + thermal, Live E | `history/` |
| **v6.16** | Radar + thermal | 3D + thermal (current fusion) | repository root |
| **v6.17** | Radar | 3D + 2D (current radar-only) | repository root |

## Fusion export

See [`TRACK_EXPORT.md`](TRACK_EXPORT.md) and [`fusion_bridge/README.md`](fusion_bridge/README.md).
