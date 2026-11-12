# Version history

Scripts here are dated snapshots, listed in development order (not GitHub's filename sort).

| File | Line | Notes |
|---|---|---|
| `live_32d_track_v3.py` / `live_32d_track_v3_final.py` | Radar | Early dual-view tracker |
| `live_32d_track_v5.py` | Radar | Pre-v6 snapshot |
| `live_32d_track_v6.3.py` … `live_32d_track_v6.10.py` | Radar | Iterations of the OpenGL 3D + 2D tracker |
| `live_32d_track_v6.11.py` | Radar | Same-tid distance veto (`TWO_PEOPLE_DIST=0.8` m) |
| `live_32d_track_v6.12.py` | Radar | JSONL / UDP export; 3D + 2D |
| `live_32d_track_v6.13.py` | Radar + thermal | First integrated thermal pane; still 3D + 2D |
| `live_32d_track_v6.14.py` | Radar + thermal | 2D pane removed; 3D + thermal |
| `live_32d_track_v6.15.py` | Radar + thermal | INFERNO / flip / room-frame extrinsics / Live E |

Current recommended scripts live in the repository root:

- `live_32d_track_v6.17.py` — IWR6843-only, 3D + 2D
- `live_32d_track_v6.16.py` — radar + thermal fusion

An older root file named `live_32d_track_v6.py` was only an alias of whatever was then latest (v6.11) and is no longer used.
