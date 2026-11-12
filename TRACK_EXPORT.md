# Track snapshot export (v6.12) — Fusion / RadarLink bridge

Locked boxes from `assign_boxes_to_clusters` can be written as JSONL (and
optionally UDP) in the same shape Haneef `mmWave/ti_track.summarise` /
`Fusion/radar_link.py` expects. Tracking logic is unchanged (no new
Kalman/EMA/hysteresis; same-tid distance veto stays).

## Step 1 — export from v6.12 (GUI + export)

```bash
python live_32d_track_v6.py --data-port COM6 --cli-port COM5 --cfg my_occupancy.cfg \
  --export-tracks tracks.jsonl

# optional UDP mirror of the same frame object
python live_32d_track_v6.py --data-port COM6 \
  --export-tracks tracks.jsonl \
  --export-tracks-udp 127.0.0.1:9100
```

Export runs on the GUI thread after each box assignment. There is no
`--headless` yet; use the normal Qt window.

## JSONL format

One JSON object per line (one radar frame):

```json
{"frame":42,"t":1234.5,"generation":0,"tracks":[{"id":5,"x":1.0,"y":2.0,"z":0.9,"vx":2.0,"vy":2.0,"vz":0.0,"speed":2.828,"range":2.236,"conf":1.0,"z_min":0.0,"z_max":1.75}]}
```

| Field | Source |
|---|---|
| `id` | Locked **box** id (firmware tid **or** local 1000+). Not raw TLV-1010 alone. |
| `x,y,z` | Box lock pose (`pos` / cluster centroid) |
| `vx,vy,vz` | Finite difference of last two trail points / ~0.1 s (0 if unknown) |
| `speed` | `hypot(vx, vy)` |
| `range` | `hypot(x, y)` |
| `conf` | `1.0` if locked this frame, `0.5` if holding |
| `z_min` / `z_max` | `0.0` / person height |
| `generation` | Frame-level; default `0` (bump later on cfg switch) |

Pending (one-frame) local boxes are skipped. Holding boxes within
`DRAW_HIDE_FRAMES` are included.

## Step 2 — feed Fusion (`fusion_bridge/`)

Thin adapter implements Haneef `RadarLink.tracks()` / `projected()` /
`generation` by reading the export stream (UDP preferred live; JSONL for
replay / smoke). Identity stays on lock boxes — board tid is not re-trusted.

```bash
cd fusion_bridge
python smoke_test.py
python fuse_vic.py --self-test
python fuse_vic.py --smoke-jsonl samples/synthetic_tracks.jsonl

# Live (radar process exporting UDP + thermal Fusion):
python fuse_vic.py --live --tracks-udp 0.0.0.0:9100 \
  --odp-root /path/to/Occupancy-Detection-Program
```

`fuse_vic --live` installs `fusion_bridge/radar_link.py` into `sys.modules`
before Haneef `integrated_launcher` imports `RadarLink`, so upstream Fusion
files need not be edited. Details: `fusion_bridge/README.md`, `DIFF.md`.

Env vars the drop-in reads:

- `VIC_TRACKS_UDP` — bind `host:port` or `port` (e.g. `0.0.0.0:9100`)
- `VIC_TRACKS_JSONL` — path to the JSONL file to follow

## Blockers (full live product)

- Thermal stack: YOLO weights, PureThermal / lepton, Haneef Thermal venv
- This bridge does **not** open radar COM ports (v6.12 does)
- `--radar-adaptive` is ignored by the bridge on purpose
