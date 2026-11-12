# How this attaches to Haneef (minimal upstream change)

## Preferred: no file edits (runtime patch)

`fuse_vic.py --live` does:

1. Set `VIC_TRACKS_UDP` / `VIC_TRACKS_JSONL`
2. `sys.modules["radar_link"] = fusion_bridge.radar_link`
3. Run Haneef `Fusion/fuse.py --live` → `integrated_launcher.main`

When the launcher executes `from radar_link import RadarLink`, Python reuses the
already-loaded bridge module even after `Fusion/` is prepended to `sys.path`.

Constructor call site (unchanged):

```python
link = RadarLink(args.radar_cli, args.radar_data,
                 args.radar_close_cfg,
                 args.radar_far_cfg if args.radar_adaptive else None,
                 adaptive=args.radar_adaptive,
                 on_event=...)
link.start()
```

Bridge accepts those args, ignores serial/cfg, reads tracks from env/kwargs.

## Alternative: replace one file in the Haneef tree

Copy both into `Occupancy-Detection-Program/Fusion/`:

- `radar_link.py`          (overwrites Haneef’s)
- `radar_link_from_tracks.py`
- optionally `vendor/` or ensure `mmWave/project.py` stays importable

Then set env and run stock `python fuse.py --live ...`.

## Not done (on purpose)

- No edits to `integrated_launcher.py` / `fuse.py` association logic
- No rewrite of `live_32d_track`
- Adaptive CLOSE↔FAR reconfigure disabled in the bridge (would reset board
  tids and fight lock-box IDs)
