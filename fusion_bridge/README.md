# fusion_bridge — v6.12 tracks → Haneef RadarLink

Step 2 of the Vic / Haneef merge: a thin adapter so Fusion’s thermal half
consumes **locked-box** snapshots from `live_32d_track` v6.12 instead of the
board tracker over serial.

| File | Role |
|---|---|
| `radar_link_from_tracks.py` | `TracksRadarLink` — UDP / JSONL → `tracks()` / `projected()` / `generation` |
| `radar_link.py` | Drop-in `RadarLink` with Haneef constructor signature |
| `fuse_vic.py` | Install drop-in + smoke / optional `--live` handoff to Haneef `fuse.py` |
| `vendor/project.py` | Copy of Haneef `mmWave/project.py` for `projected()` |
| `smoke_test.py` | Synthetic JSONL + UDP, no Qt / no radar |
| `samples/synthetic_tracks.jsonl` | Fixture |

## Quick check (no Thermal, no radar)

```bash
cd fusion_bridge
python smoke_test.py
python fuse_vic.py --self-test
python fuse_vic.py --smoke-jsonl samples/synthetic_tracks.jsonl
```

## Live wiring (two processes)

**A — radar (v6.12):**

```bash
python live_32d_track_v6.py --data-port COM6 --cli-port COM5 --cfg my_occupancy.cfg \
  --export-tracks tracks.jsonl \
  --export-tracks-udp 127.0.0.1:9100
```

**B — fusion (this bridge as RadarLink):**

```bash
# Env alternative: set VIC_TRACKS_UDP=0.0.0.0:9100
python fuse_vic.py --live --tracks-udp 0.0.0.0:9100 \
  --odp-root /path/to/Occupancy-Detection-Program \
  --cli COMx --data COMy   # still passed through; ignored by bridge
```

`fuse_vic --live` puts our `radar_link` in `sys.modules` **before**
`integrated_launcher` imports Haneef’s, so Fusion/Thermal need not be edited.
See `DIFF.md` for the optional file-copy alternative.

## Field map

Snapshot track fields are 1:1 with Haneef `ti_track.summarise` /
`RadarLink._tracks`. `id` is the v6.12 **locked box** id (may be local 1000+).
`generation` is read from each frame object.

## Blockers for full product

- Haneef **Thermal** venv (YOLO weights, lepton/UVC) and a PureThermal camera
- Windows COM ports only matter for the *thermal* launcher / if you still run
  Haneef’s native radar path; this bridge does **not** open radar serial
- `--adaptive` cfg switching is ignored on purpose (lock boxes own identity)

## One-window Vic UI (radar + thermal)

`vic_fusion_ui.py` is a **single** OpenCV window — left radar top-down, right
thermal — instead of Haneef’s cluttered launcher. Trails off by default;
unconfirmed thermal noise hidden unless you press `U`.

| Key | Action |
|---|---|
| `T` | trails on/off (default off) |
| `R` | reset trails |
| `+` / `-` | YOLO conf |
| `U` | show unconfirmed dets |
| `Q` / Esc | quit |

### Smoke (no camera)

```bash
cd fusion_bridge
python vic_ui_smoke.py
python vic_fusion_ui.py --smoke --smoke-out /tmp/vic_ui.png
```

### Live on Windows (VIC01) — still two processes for radar hardware

Radar GUI stays in v6.12 (serial + lock boxes). The UI process only consumes
UDP tracks + the PureThermal camera.

**UDP port must match.** v6.12 exports to `127.0.0.1:9100`; `vic_fusion_ui`
defaults to `--tracks-udp 0.0.0.0:9100`. Binding **8100** (or any other port)
while A still sends **9100** is a silent mismatch — you may see stale/empty
tracks or wonder why association never fires. Check the status bar `udp=…`.

**Occupancy when YOLO misses:** if thermal has a person heat blob but YOLO
returns no boxes (`ass=0`), the big count falls back to radar lock-box `n`
(label `radar`) and cyan `R#id` projections are still drawn on the thermal
pane. Matched pairs use label `people` / `fusion`.

```powershell
# A — radar (mmwave venv)
cd C:\Users\13055\mmwave_occupancy
.\venv\Scripts\python.exe live_32d_track_v6.12.py --data-port COM6 --cli-port COM5 `
  --cfg my_occupancy.cfg --export-tracks tracks.jsonl --export-tracks-udp 127.0.0.1:9100

# B — one-window UI (Thermal .venv-win has opencv + ultralytics)
cd C:\Users\13055\mmwave_occupancy\fusion_bridge
C:\Users\13055\Documents\GitHub\Occupancy-Detection-Program\Thermal\.venv-win\Scripts\python.exe `
  vic_fusion_ui.py --device 1 `
  --weights C:\Users\13055\Documents\GitHub\Occupancy-Detection-Program\Thermal\models\v2\best.pt `
  --conf 0.37
# default UDP bind is already 0.0.0.0:9100 — do NOT use 8100
```

Copy this folder to VIC01:

`C:\Users\13055\mmwave_occupancy\fusion_bridge\` (includes `vic_fusion_ui.py`, `vic_ui\`, bridge modules).

If YOLO weights are missing **or YOLO returns no dets**, occupancy falls back
to the radar lock-box count (status `occ=N(radar)`), and radar projections are
still drawn on the thermal pane as cyan `R#id` boxes.
