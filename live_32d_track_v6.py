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
# -----------------------------------------------------------------------------
# CHANGELOG v3.2 -> v3.3  (upgrade from official TI / live_3d_radar_v9 pack)
# -----------------------------------------------------------------------------
#  Aligned with zip reference (parseTLVs / people_tracking / live_3d_radar_v9):
#   - Room transform now matches TI eulerRot(elev_tilt, azi_tilt=0) then
#     +sensorHeight (was the opposite elev-tilt sign from v9's local copy).
#   - Point cloud: prefer parseStandardFrame, then corrected TLV-1020 fallback
#     (5x float32 units + per-point '<2bh2H', spherical->cartesian) from v9.
#   - Tracks: prefer official trackData (I27f / parseTrackTLV); manual TLV-1010
#     only fills missing tids. Keep velocity and seed Kalman with it.
#   - Parse TLV 1012 heightData (tid, maxZ, minZ); size person boxes from it.
#   - Keep people counter + ghost gates from v3.2.
#
# -----------------------------------------------------------------------------
# CHANGELOG v3.3 -> v5  (cumulative visit counter)
# -----------------------------------------------------------------------------
#  The main number is NO LONGER current occupancy ("how many people are in
#  the room right now"). It is a CUMULATIVE detection-event counter:
#
#    Total Detected N = how many times a person has been newly confirmed
#    present since app start (or since Reset).
#
#  Rising-edge rules:
#    - A track is CONFIRMED when missed==0, pose finite, room Z in the human
#      band, and inside BOUNDARY_BOX_ROOM.
#    - After CONFIRM_FRAMES consecutive confirmed frames (default 5), fire
#      N += 1 once for this visit (ignore ID flicker).
#    - While that tid stays continuously confirmed, do NOT increment again.
#    - A track is GONE when the tid leaves track_state, missed > LEFT_FRAMES
#      (default 8), or it fails the Z/boundary gates for LEFT_FRAMES frames.
#    - After GONE, the SAME tid returning later is a NEW visit -> N += 1 again.
#    - Leaving the room does NOT decrease N.
#
#  UI: "Total Detected: N" (big) + "In room now: K" (secondary) + Reset.
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
# CHANGELOG v5 -> v5.1  (fast-walk reacquire + ghost point-support)
# -----------------------------------------------------------------------------
#  FIX 1 — Fast walking was splitting one person into many Total Detected +1s
#    because the tracker briefly dropped the tid and minted a new one.
#    Visits now stay OPEN for LEFT_TIME (2.5 s). A newly confirmed tid within
#    REACQUIRE_DIST (1.2 m) of a recently-lost open visit REUSES that visit
#    (no +1). CONFIRM_FRAMES raised to 8. Simultaneous tids closer than
#    MERGE_DIST (0.8 m) share one occupancy / visit slot.
#
#  FIX 2 — Empty red ghost box (no nearby cloud) is no longer drawn and never
#    increments N. Require MIN_NEAR_POINTS (3) cloud points within
#    POINT_SUPPORT_RADIUS (0.7 m) of the track XY. Fail for 3 frames -> hide /
#    drop. Person-box height from TLV 1012 clamped to 0.8–2.2 m. Never draw a
#    person-sized box on the sensor marker at (0,0,sensor_height).
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
# CHANGELOG v5.1 -> v5.2  (entry-edge double-count lock)
# -----------------------------------------------------------------------------
#  Walking in from the yellow boundary edge could mint two visits: first as a
#  brief edge detection, then again after the tracker reacquired deeper inside.
#  Entry-band visits now stay locked until inward progress / time commit, and
#  nearby reappears stitch/reacquire into the same visit (ENTRY_* constants).
#  CONFIRM_FRAMES=10, MERGE_DIST=1.0.
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
# CHANGELOG v5.2 -> v5.3  (multi-person merge fix + live param panel)
# -----------------------------------------------------------------------------
#  BUG: 3+ simultaneous people were collapsed by over-aggressive merge/stitch
#    (MERGE_DIST=1.0, ENTRY_STITCH_DIST=1.6, REACQUIRE gluing). Defaults now:
#    MERGE_DIST=0.45, ENTRY_STITCH_DIST=0.9, REACQUIRE_DIST=0.9,
#    ENTRY_REACQUIRE_DIST=1.2, MIN_NEAR_POINTS=1, POINT_SUPPORT_RADIUS=0.9.
#    Never merge/stitch two tids that are BOTH confirmed this frame; only
#    stitch a NEW tid onto a visit whose members are all GONE/unconfirmed,
#    or distance < MERGE_DIST with an inactive/gone visit. _merge_slots only
#    clusters below MERGE_DIST so three people 0.6–1.0 m apart stay K=3.
#    Status bar: Frame | in-room K | total N | raw_tids R | confirmed C |
#    drop:merge=M stitch=S gate=G unsup=U. --debug logs merge/stitch/reject.
#
#  FEATURE: right-side live GUI parameter panel (QScrollArea) for Kalman,
#    boundary box, SENSOR_HEIGHT/TILT_DEG. Thread-safe ParamSnapshot under
#    MainWindow._param_lock; ParserThread.apply_params / set_geometry;
#    CVKalman3D.set_noise; VisitCounter.apply_params. "Reset params" restores
#    code defaults (Total Detected Reset stays separate). Title → v5.3.
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
# CHANGELOG v5.3 -> v5.4  (TLV 52 / unknown-TLV must not drop frames)
# -----------------------------------------------------------------------------
#  Official parseStandardFrame aborts the TLV loop on unknown types (prints
#  "Invalid TLV type: 52" / "Dropping frame..."), which also trips the
#  totalPacketLen mismatch warning. We patch unusedTLVs so unknowns are
#  skipped at DEBUG and the rest of the frame (1010/1012/1020) still parses.
#  UART framing unchanged: consume header totalPacketLen, resync on MAGIC_WORD.
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
# CHANGELOG v5.4 -> v6  (TLV header failure must not empty the stream)
# -----------------------------------------------------------------------------
#  Runtime "TLV Header Parsing Failure: Ignored frame..." comes from
#  parseFrame.parseStandardFrame when tlvHeaderDecode fails — it returns {}.
#  v6: demote that WARNING to DEBUG, keep unknown-TLV skip, and if official
#  output is empty / no payload, always fall back to manual_parse_frame()
#  which walks TLVs, skips unknowns (incl. 52) by exact length, and still
#  decodes 1020/1010/1012. UART framing unchanged (totalPacketLen + MAGIC_WORD).
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
# CHANGELOG v6 -> v6.1  (current-frame support + split-ghost suppress)
# -----------------------------------------------------------------------------
#  Ghost tests must use THIS-frame cloud only (cloud_frame), never the
#  accumulated display cloud. MIN_NEAR_POINTS raised to 2; POINT_SUPPORT_RADIUS
#  tightened to 0.6. MIN_SEPARATION / DUPLICATE_SUPPRESS_DIST: one-person lock
#  — score = near*10+snr, stronger-tid wins; weaker splits too close to a
#  supported centroid are dropped (In room now / Total Detected stay honest).
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
# CHANGELOG v6.1 -> v6.2  (raw cloud filter + median-SNR confirm + GHOST_MERGE)
# -----------------------------------------------------------------------------
#  [POINT CLOUD] filter_raw_cloud() before accumulation: drop non-finite,
#  SNR < PC_SNR_MIN (default 8.0), outside BOUNDARY_BOX+/-margin, room-Z
#  outside [0,2.5], optional |doppler|>3. Display stays [x,y,z,snr].
#  [CONFIRM] tid drawable/countable only with missed==0, Z in 0.2–2.4,
#  inside box, >=MIN_NEAR_POINTS within 0.6 m XY, median nearby SNR >=
#  PC_SNR_MIN. near_point_stats -> (count, snr_sum, snr_median, centroid).
#  [GHOST] suppress_split_ghosts uses GHOST_MERGE_DIST=1.5; score =
#  near*max(median_snr,1); clearly-stronger (ratio>=1.3) wins without
#  MIN_SEPARATION; equal strength (within 20%) needs MIN_SEPARATION.
#  [TEMPORAL] CONFIRM_FRAMES=8; fail_support_streak>=3 hides; REACQUIRE
#  TIME=2.0 / DIST=1.5. Live panel: PC_SNR_MIN, MIN_NEAR_POINTS,
#  GHOST_MERGE_DIST, POINT_SUPPORT_RADIUS.
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# CHANGELOG v6.10 -> v6.11  (same-tid distance veto, no reservation, SWITCH_FRAMES=4)
# -----------------------------------------------------------------------------
#  BUGS: v6.10 box stuck on an old blob then popped; trails fragmented; two
#  distant clusters could share ONE box when TLV 1011 tagged both with the same
#  tid (firmware sometimes gives two bodies, or body + multipath, one tid).
#  FIX:
#   * Same-tid distance veto: clusters >= TWO_PEOPLE_DIST apart are ALWAYS two
#     boxes; a shared TLV-1011 tid is not enough to merge or jump.
#     - pass 0 exact-tid lock only onto the same-tid cluster NEAREST the box's
#       last lock pose, and only if it is within LOCK_RADIUS and closer than
#       TWO_PEOPLE_DIST (brand-new tid: nearest to Kalman pose within
#       LOCK_RADIUS).  Other same-tid clusters stay unclaimed -> pass 3 spawns
#       new local boxes (1000+).  A far same-tid cluster never makes the box
#       jump; the box holds its last pose (DRAW_HIDE_FRAMES).
#     - _same_tid_immediate() applies the same veto.
#     - cluster_cloud_by_track_index() splits each tid's points into XY
#       sub-clusters (cluster_cloud_xy flood/split, merge_dist=TWO_PEOPLE_DIST)
#       so two far blobs sharing one tid become two clusters with that tid.
#   * Removed v6.10 weak->strong promote and the reservation of an alternate
#     cluster during a switch streak (it stole the second person's box).
#     Unclaimed clusters always go to pass 3.
#   * Switch gating simplified: a box re-locks immediately onto the nearest
#     cluster to its last lock pose (walking < SAME_CLUSTER_DIST updates the
#     same frame).  Only a switch to a cluster >= SAME_CLUSTER_DIST away while
#     a nearer old cluster still exists needs SWITCH_FRAMES.  Dropped the
#     "old.n < MIN_CLUSTER_POINTS" gate (caused stick-then-pop).
#   * SWITCH_FRAMES default 8 -> 4 (v6.9 value); GUI default follows.
#   * _lock(): a new lock pose > TWO_PEOPLE_DIST from the previous lock pose
#     clears the trail (history=[pos], trail_gen++); the jump is never drawn.
#     TRAIL_MAX_STEP stays 0.45 as a safety net.
#   * append_trail_pose(prev_raw=...): the > TRAIL_MAX_STEP teleport test uses
#     the raw lock-to-lock step, not the EMA-lagged last trail point (a 0.2 m/
#     frame walker was clearing its trail every ~4 frames = fragmented trails).
#   * Live Parameters / window title -> v6.11.
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
# CHANGELOG v6.9 -> v6.10  (own-cluster lock + hard trail max-step)
# -----------------------------------------------------------------------------
#  BUGS: Yellow/red trail still connected TWO separate cloud clusters; box/trail
#  teleported between bodies. v6.9 SWITCH_MARGIN hysteresis + EMA did not stop
#  oscillating centroids / brighter-neighbor pulls while the old body remained.
#  FIX:
#   * TRAIL_MAX_STEP default 0.45 (was 0.6). append_trail_pose() hard-rejects any
#     consecutive history segment > max_step (clear to [new]/ after append,
#     sanitize adjacent pairs and truncate to [last] if any jump remains).
#   * Own-cluster lock (pass 2b): switch ONLY if SWITCH_FRAMES=8 AND preferred/old
#     cluster missing OR old.n < MIN_CLUSTER_POINTS. Brighter/closer neighbor must
#     NOT pull while the old body still has enough points. TLV-1011 same-tid
#     preferred cluster always re-locks immediately; never switch away while the
#     own-tid cluster is alive with n >= MIN_CLUSTER_POINTS.
#   * Rejected far switch leaves the other cluster UNCLAIMED so pass-3 spawns a
#     second box (>= TWO_PEOPLE_DIST clear separation). Do not reserve/steal it.
#   * TrackArtists3D/2D: when hist has <=1 point, explicitly clear trail geometry
#     (empty setData) — never leave the previous polyline.
#   * Trail remains EMA path of THIS lock only (never firmware Kalman history).
#   * Live Parameters / window title -> v6.10.
# -----------------------------------------------------------------------------
#
# CHANGELOG v6.8 -> v6.9  (lock hysteresis + filtered trail + occupancy sync)
# -----------------------------------------------------------------------------
#  BUGS: (1) person trail jumped between two cloud blobs (lock flicker;
#  history appended every jumped centroid). (2) Box+trail visible but
#  "In room now: 0".
#  FIX:
#   * assign_boxes_to_clusters(): SWITCH_MARGIN / SWITCH_FRAMES hysteresis —
#     an already-locked box may switch clusters only if the new one is closer
#     by SWITCH_MARGIN for SWITCH_FRAMES consecutive frames. TLV-1011 same-tid
#     re-lock is immediate; switching to a different tid still needs hysteresis.
#     Box stores switch_candidate / switch_streak / lock_cx / lock_cy.
#   * append_trail_pose(): replace raw _hist_for. TRAIL_MIN_STEP skips micro
#     jitter; TRAIL_MAX_STEP clears history (no teleport segment); else EMA
#     0.7/0.3. draw_pos is EMA-smoothed for painting. trail_gen bumps on
#     reset so MainWindow clears the artist once (one trail per lock id).
#   * Occupancy: only paint non-pending boxes inside the yellow volume;
#     In room now = drawable count. VisitCounter locked path treats cluster
#     support as OK (no PC_SNR_MIN / MIN_NEAR drop) so Total Detected still
#     rises; holding boxes that are still painted also count in-room.
#   * Live Parameters: SWITCH_MARGIN, SWITCH_FRAMES, TRAIL_MIN_STEP,
#     TRAIL_MAX_STEP. Version bump to v6.9.
# -----------------------------------------------------------------------------
#
# CHANGELOG v6.7 -> v6.8  (TLV 1011 point-to-track index clustering)
# -----------------------------------------------------------------------------
#  Parse MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX (TLV 1011): one uint8 per
#  detected point in the same order as the point cloud. Values 0–249 are the
#  associated firmware track tid; 253/254/255 are unassociated sentinels
#  (WEAK_SNR / BOUNDS / NOISE). Prefer out["trackIndexes"] from the official
#  TI parser; fall back to walking TLV 1011 in the raw frame.
#  When association is present, cluster_cloud_by_track_index() groups points by
#  firmware tid (inside BOUNDARY_BOX) and assign_boxes_to_clusters() prefers an
#  exact tid lock before greedy XY distance matching. When TLV 1011 is absent
#  or all points are unassociated, keep the v6.7 cluster_cloud_xy() fallback.
#  point_tids stay 1:1 with cloud_frame via the same filter mask.
#  Status bar shows 1011=yes/no and ptid_assoc=N.
# -----------------------------------------------------------------------------
#
# CHANGELOG v6.6 -> v6.7  (cluster-first box lock: one stable box per person)
# -----------------------------------------------------------------------------
#  BUG: with 2+ people in the yellow box a single box jumped between cloud
#  clusters.  Cause: associate_cloud_cluster()'s global in-box fallback
#  snapped a tid onto the strongest blob anywhere in the box,
#  display_snap_pose() moved the drawn box there, and suppress_split_ghosts()
#  ONE_PERSON_LOCK then deleted the other tids.
#  FIX -- cluster first, then assign boxes:
#   * cluster_cloud_xy(): numpy-only greedy flood (CLUSTER_EPS=0.7,
#     MIN_CLUSTER_POINTS=3) of the CURRENT-frame cloud inside BOUNDARY_BOX,
#     density-valley split of chained bodies, same-body merge < TWO_PEOPLE_DIST.
#   * assign_boxes_to_clusters(): 1-to-1 lock.  N clusters -> N boxes on the
#     cluster centroids.  A tid / box attaches only to the nearest unmatched
#     cluster within LOCK_RADIUS=1.0 (anchor = last locked pose, Kalman pose
#     only for the first lock -> no stealing, no jumping).  Unmatched cluster
#     -> local box id 1000+ (2-frame confirm).  Unmatched box holds its pose
#     DRAW_HIDE_FRAMES then drops.  Two ids on one cluster (< 0.5 m) -> keep
#     the stronger.  NO global strongest-blob fallback.
#   * associate_cloud_cluster(): fallback hard-disabled when n_clusters >= 2;
#     display_snap_pose(): never snaps farther than LOCK_RADIUS.
#   * suppress_split_ghosts(): narrowed.  < SAME_CLUSTER_DIST (0.5) -> same
#     cluster, keep stronger.  >= TWO_PEOPLE_DIST (0.8) -> two people, keep
#     both.  ONE_PERSON_LOCK removed.
#   * VisitCounter.update(locked_boxes=...): COUNT runs on stable box ids;
#     In room now K = locked boxes (3+ possible); Total Detected N is still a
#     rising edge per visit after CONFIRM_FRAMES; a jumping tid cannot +1.
#   * Live Parameters: new group D (CLUSTER_EPS, MIN_CLUSTER_POINTS,
#     LOCK_RADIUS, SAME_CLUSTER_DIST, TWO_PEOPLE_DIST, DRAW_HIDE_FRAMES).
#   * Status bar shows clusters / boxes per frame.
# -----------------------------------------------------------------------------
#
# CHANGELOG v6.5 -> v6.6  (surgical review fixes 1–4 + 6)
# -----------------------------------------------------------------------------
#  1) manual_parse_frame: keep [x,y,z,doppler,snr] — was stuffing Doppler into
#     the SNR column and zeroing Doppler (starved filter_raw_cloud).
#  2) Innovation gate: accept if Mahalanobis² <= GATE_SIGMA² OR (young track
#     and ||innovation|| <= GATE_MIN_M). Old max(sigma², min_m²/R) was ~12σ.
#  3) heightData merge: never truth-test a numpy array (ambiguous ValueError).
#  4) Bad pose on measure/update/coast: increment missed and coast — do NOT
#     delete until missed > MAX_MISSED (stops ID churn / double visits).
#  5) Remove synthetic tid=-1 cloud box and in_room=max(drawable) override.
#  6) Latest-only snapshot slot + QTimer drain (no unbounded GUI queue).
#  Also: time.monotonic for Kalman dt; use link_reason in debug log.
# -----------------------------------------------------------------------------
#
# CHANGELOG v6.4 -> v6.5  (draw vs count + snap box to cloud)
# -----------------------------------------------------------------------------
#  DRAW the person box with looser gates (DRAW_MIN_NEAR_POINTS=1,
#  DRAW_SUPPORT_RADIUS=1.0, DRAW_SNR_MIN=3) as soon as current-frame support
#  exists — do NOT wait for CONFIRM_FRAMES. COUNT Total Detected only after
#  CONFIRM_FRAMES with the stricter count/ghost gates. Display-snap: if the
#  official track XY is > SNAP_TO_CLOUD_DIST (0.5 m) from the associated
#  current-frame cluster centroid, draw the box on the centroid. Hide only
#  after DRAW_HIDE_FRAMES=5 unsupported/missed frames. Live panel exposes
#  DRAW_MIN_NEAR_POINTS, DRAW_SUPPORT_RADIUS, DRAW_SNR_MIN.
# -----------------------------------------------------------------------------
#
# CHANGELOG v6.3 -> v6.4  (four-filter operator checklist)
# -----------------------------------------------------------------------------
#  Documented and verified the four filters above. Defaults unchanged from
#  v6.3: PC_SNR_MIN=8, MIN_NEAR_POINTS=2, GHOST_MERGE_DIST=1.5,
#  CONFIRM_FRAMES=8, REACQUIRE 2.0 s / 1.5 m, point support radius 0.6 m.
#  Live panel already exposes PC_SNR_MIN, MIN_NEAR_POINTS, GHOST_MERGE_DIST.
# -----------------------------------------------------------------------------
#
# CHANGELOG v6.2 -> v6.3  (one-person lock for ~2 m radial cyan ghost)
# -----------------------------------------------------------------------------
#  Screenshot: green@y≈1.5 real, cyan@y≈3.5 ghost (~2 m apart). Within-1.5 m
#  stronger-wins alone missed them. ONE-PERSON LOCK: if the strongest tid
#  already has STRONG_SUPPORT_NEAR current-frame points, a second tid must
#  reach SECOND_SCORE_RATIO of its score AND be >= MIN_SEPARATION away —
#  otherwise dropped same frame (no box/trail/+1). GHOST_MERGE_DIST stays 1.5
#  so two real people ~1.8 m apart are not glued.
# -----------------------------------------------------------------------------
#

# =============================================================================
# FOUR FILTERS (operator: GREEN = real person, CYAN = split ghost)
# =============================================================================
# 1) POINT-CLOUD FILTER  — filter_raw_cloud() BEFORE accumulation
#    Drop non-finite; SNR < PC_SNR_MIN (default 8, live GUI); outside live
#    yellow BOUNDARY_BOX +/- margin; room-Z outside [0.0, 2.5]; optional
#    Doppler gate |doppler| <= 3 (near-static + walking; wild outliers out).
#
# 2) TRACK CONFIRMATION  — same frame, CURRENT-frame cloud only (cloud_frame)
#    Drawable / countable only if: missed==0, room Z in 0.2–2.4 m, inside
#    BOUNDARY_BOX, >= MIN_NEAR_POINTS points within 0.6 m XY, and those
#    nearby points have median SNR >= PC_SNR_MIN. Never use ACCUM cloud to
#    keep a dead/coasting tid alive.
#
# 3) SPLIT-GHOST FILTER  — suppress_split_ghosts()
#    Score = (current-frame near count) * median SNR. When two tids are
#    closer than GHOST_MERGE_DIST (default 1.5 m): keep the winner (GREEN),
#    delete the loser the same frame (no box, no trail, no +1). Do NOT
#    require 1.2 m centroid separation when one cluster is clearly stronger.
#    Extra one-person lock: a weak second tid far in Y (classic cyan) is
#    also dropped if the strongest body already has STRONG_SUPPORT_NEAR.
#
# 4) TEMPORAL FILTER
#    CONFIRM_FRAMES=8 before Total Detected += 1. Fail point-support for
#    UNSUPPORTED_HIDE_FRAMES=3 -> hide. Reacquire within REACQUIRE_TIME=2.0 s
#    and REACQUIRE_DIST=1.5 m = same visit (no extra +1). True later re-entry
#    after leave still +1. Fast wall walk-in still +1 once (entry-band lock).
# =============================================================================

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
import threading
from collections import deque
from dataclasses import dataclass, field, replace

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

import parseFrame as parseFrame_mod                  # module — we patch unusedTLVs
from parseFrame import parseStandardFrame as _ti_parseStandardFrame
import demo_defines as demo_defines_mod              # official demo name table

log = logging.getLogger("live_32d_track_v6")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =============================================================================
# [v6] Tolerant TLV parsing — unknown TLV / header glitch must NOT kill stream
# =============================================================================
# Trace of the runtime warning:
#   parseFrame.parseStandardFrame() -> tlvHeaderDecode() raises/fails
#   -> log.warning("TLV Header Parsing Failure: Ignored frame due to parsing error")
#   -> return {}   (error=2)
# That empties the official snapshot. Separately, unknown types (TLV 52) hit:
#   log.info("Invalid TLV type"); print("Dropping frame..."); break
# One bad/unrecognized TLV must never abort the whole frame: skip `length`
# bytes and keep decoding 1010 / 1012 / 1020. UART stays synced by consuming
# header totalPacketLen and resyncing on MAGIC_WORD.

class _AcceptUnknownTLVs:
    """Membership always True → official parser takes the unusedTLVs (skip) path."""
    def __contains__(self, tlv_type):
        return True


class _DemoteParseFrameNoise(logging.Filter):
    """Downgrade official parseFrame spam to DEBUG (we recover via manual walk)."""
    _MARKERS = (
        "TLV Header Parsing Failure",
        "totalPacketLen in frame header",
        "Invalid TLV type",
    )
    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return True
        if any(m in msg for m in self._MARKERS):
            record.levelno = logging.DEBUG
            record.levelname = "DEBUG"
        return True


def _install_tolerant_unknown_tlv_skip():
    parseFrame_mod.unusedTLVs = _AcceptUnknownTLVs()
    # Also demote the official logger so WARNING/INFO spam does not flood.
    pf_log = logging.getLogger(getattr(parseFrame_mod, "__name__", "parseFrame"))
    pf_log.addFilter(_DemoteParseFrameNoise())
    log.debug("Unknown TLV types skipped; parseFrame header warnings demoted to DEBUG")


_install_tolerant_unknown_tlv_skip()

# Rate-limit unknown-TLV DEBUG lines (once per N occurrences).
_UNKNOWN_TLV_LOG_EVERY = 60
_unknown_tlv_hits = {}


def _log_unknown_tlv(tlv_type, length):
    n = _unknown_tlv_hits.get(tlv_type, 0) + 1
    _unknown_tlv_hits[tlv_type] = n
    if n == 1 or (n % _UNKNOWN_TLV_LOG_EVERY) == 0:
        log.debug("Skipping unknown TLV type=%s length=%s (count=%s)",
                  tlv_type, length, n)


def _out_has_sensor_payload(out):
    if not out:
        return False
    try:
        n_pts = int(out.get("numDetectedPoints", 0) or 0)
        n_trk = int(out.get("numDetectedTracks", 0) or 0)
        if n_pts > 0 or n_trk > 0:
            return True
        pc = out.get("pointCloud")
        if pc is not None and len(pc) > 0:
            return True
        td = out.get("trackData")
        if td is not None and len(td) > 0:
            return True
    except Exception:
        return False
    return False


def manual_parse_frame(frame_bytes):
    """Best-effort full-frame TLV walk (no official parser).

    Rules:
      - Frame already synced on MAGIC_WORD by the UART reader; we trust the
        slice length (header totalPacketLen) from the caller.
      - For each TLV: read type+length; unknown (incl. 52) -> skip exactly
        `length` bytes and continue (DEBUG, rate-limited).
      - Decode TLV 1020 point cloud, 1010 tracks, 1011 track indexes, 1012
        height when present.
      - Never return empty solely because one TLV type is unrecognized.
    """
    out = {
        "error": 0,
        "pointCloud": None,
        "numDetectedPoints": 0,
        "trackData": None,
        "numDetectedTracks": 0,
        "trackIndexes": None,
        "heightData": None,
        "numDetectedHeights": 0,
        "source": "manual_tlv_walk",
    }
    if not frame_bytes or len(frame_bytes) < FRAME_HEADER_LEN:
        out["error"] = 1
        return out

    # Point cloud via existing corrected 1020 decoder
    try:
        pc = _manual_parse_tlv1020(frame_bytes)
        if pc is not None and len(pc) > 0:
            # [v6.6] _manual_parse_tlv1020 already returns [x,y,z,doppler,snr].
            # Do NOT map pc[:,3] (doppler) into the SNR column — that made
            # filter_raw_cloud treat walking speeds as SNR and drop the cloud.
            if pc.ndim == 2 and pc.shape[1] >= 5:
                out["pointCloud"] = np.asarray(pc[:, :5], dtype=float)
            elif pc.ndim == 2 and pc.shape[1] == 4:
                # Legacy [x,y,z,snr] — pad doppler=0
                arr = np.zeros((len(pc), 5), dtype=float)
                arr[:, :3] = pc[:, :3]
                arr[:, 4] = pc[:, 3]
                out["pointCloud"] = arr
            else:
                out["pointCloud"] = np.asarray(pc, dtype=float)
            out["numDetectedPoints"] = len(pc)
    except Exception as e:
        log.debug("manual 1020 parse failed: %s", e)

    try:
        tracks = manual_parse_tracks(frame_bytes)
        if tracks:
            # Rows: tid,x,y,z,vx,vy,vz (official-like)
            td = np.zeros((len(tracks), 7), dtype=float)
            for i, rec in enumerate(tracks):
                td[i, 0] = rec[0]
                td[i, 1] = rec[1]
                td[i, 2] = rec[2]
                td[i, 3] = rec[3]
                if len(rec) >= 7:
                    td[i, 4] = rec[4]
                    td[i, 5] = rec[5]
                    td[i, 6] = rec[6]
            out["trackData"] = td
            out["numDetectedTracks"] = len(tracks)
    except Exception as e:
        log.debug("manual 1010 parse failed: %s", e)

    try:
        heights = extract_height_data(None, frame_bytes)
        if heights:
            rows = []
            for tid, h in heights.items():
                rows.append((tid, h, 0.0))
            out["heightData"] = np.array(rows, dtype=float)
            out["numDetectedHeights"] = len(rows)
    except Exception as e:
        log.debug("manual 1012 parse failed: %s", e)

    # TLV 1011: one uint8 per point (same order as point cloud).
    # 0–249 = associated track tid; 253/254/255 = unassociated sentinels.
    try:
        for tlv_type, payload in iter_tlvs(frame_bytes):
            if tlv_type != TLV_TARGET_INDEX_1011:
                continue
            if payload:
                out["trackIndexes"] = np.frombuffer(
                    payload, dtype=np.uint8).astype(int)
            break
    except Exception as e:
        log.debug("manual 1011 parse failed: %s", e)

    # Enumerate TLVs once more for rate-limited unknown logging only.
    try:
        for tlv_type, payload in iter_tlvs(frame_bytes):
            if tlv_type not in (TLV_POINT_CLOUD_1020, TLV_TARGET_LIST_1010,
                                TLV_TARGET_INDEX_1011,
                                TLV_TARGET_HEIGHT_1012):
                _log_unknown_tlv(tlv_type, len(payload))
    except Exception:
        pass

    return out


def parseStandardFrame(frame, demo=None):
    """Try official TI parser (with unknown-TLV skip); on empty/error==2 or
    missing payload, fall back to manual_parse_frame so streaming continues.
    """
    out = None
    try:
        out = _ti_parseStandardFrame(frame, demo=demo)
    except Exception as e:
        log.debug("parseStandardFrame raised; using manual TLV walk: %s", e)
        out = None

    # Official returns {} on TLV header failure (error was set then cleared by return {}).
    # Also treat error 2 / empty payload as failure.
    ti_ok = bool(out) and _out_has_sensor_payload(out)
    if out and out.get("error") == 3:
        log.debug(
            "Frame length check mismatch (non-fatal). "
            "UART still advances by header totalPacketLen / MAGIC_WORD.")
        out["error"] = 0

    if ti_ok:
        return out

    manual = manual_parse_frame(frame)
    if ti_ok is False and _out_has_sensor_payload(manual):
        log.debug("Using manual TLV walk (official empty/error or no payload)")
        return manual
    # Prefer whichever has more signal; merge if TI had partial data.
    if out and isinstance(out, dict):
        if not _out_has_sensor_payload(out) and _out_has_sensor_payload(manual):
            return manual
        # Fill holes from manual
        if out.get("numDetectedPoints", 0) in (0, None) and manual.get("numDetectedPoints", 0):
            out["pointCloud"] = manual["pointCloud"]
            out["numDetectedPoints"] = manual["numDetectedPoints"]
        if out.get("numDetectedTracks", 0) in (0, None) and manual.get("numDetectedTracks", 0):
            out["trackData"] = manual["trackData"]
            out["numDetectedTracks"] = manual["numDetectedTracks"]
        # [v6.6] Never truth-test a numpy array (ambiguous ValueError on multi-row).
        hd = out.get("heightData", None)
        need_h = hd is None or (hasattr(hd, "__len__") and len(hd) == 0)
        if need_h and manual.get("heightData") is not None:
            out["heightData"] = manual["heightData"]
            out["numDetectedHeights"] = manual.get("numDetectedHeights", 0)
        ti = out.get("trackIndexes", None)
        need_ti = ti is None or (hasattr(ti, "__len__") and len(ti) == 0)
        if need_ti and manual.get("trackIndexes") is not None:
            out["trackIndexes"] = manual["trackIndexes"]
        out["error"] = 0
        return out
    return manual

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
TLV_POINT_CLOUD_1020 = 1020    # MMWDEMO_OUTPUT_MSG_COMPRESSED_POINTS / POINT_CLOUD
TLV_TARGET_HEIGHT_1012 = 1012  # MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_HEIGHT
TLV_TARGET_INDEX_1011 = 1011   # MMWDEMO_OUTPUT_MSG_TRACKERPROC_TARGET_INDEX
# One uint8 per detected point (same order as point cloud). Unassoc sentinels:
TRACK_INDEX_WEAK_SNR = 253     # Point not associated, SNR too weak
TRACK_INDEX_BOUNDS = 254       # Point not associated, outside boundary
TRACK_INDEX_NOISE = 255        # Point not associated, considered noise
TRACK_INDEX_UNASSOC = frozenset({253, 254, 255})

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
PC_SNR_MIN = 8.0               # drop points below this SNR (was 0.0; v6.2)

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
def make_transform(sensor_height, tilt_deg, azi_tilt_deg=0.0):
    """Sensor-frame -> room-frame, matching TI Industrial Visualizer.

    people_tracking.py does:
        rotX, rotY, rotZ = eulerRot(x, y, z, elev_tilt, az_tilt)
        z = rotZ + sensorHeight
    With az_tilt=0, eulerRot reduces to an elevation rotation:
        x' = x
        y' =  y*cos(e) + z*sin(e)
        z' = -y*sin(e) + z*cos(e) + sensor_height
    (Earlier local copies used the opposite elev sign; that mismatch can shove
    tracks vertically and feed the "ghost above the yellow box" symptom.)
    """
    elev = math.radians(tilt_deg)
    azi = math.radians(azi_tilt_deg)
    ce, se = math.cos(elev), math.sin(elev)
    ca, sa = math.cos(azi), math.sin(azi)
    # Full elevAziRotMatrix from graph_utilities.eulerRot
    R = np.array([
        [ca, ce * sa, se * sa],
        [-sa, ce * ca, se * ca],
        [0.0, -se, ce],
    ], dtype=float)

    def transform_array(pts):
        """pts: (N,3) raw sensor-frame [x,y,z] -> (N,3) room-frame."""
        if pts is None or len(pts) == 0:
            return np.empty((0, 3), dtype=float)
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        out = pts @ R.T
        out[:, 2] = out[:, 2] + sensor_height
        return out

    def transform_velocity(vels):
        """Rotate velocities with the same R (no height translation)."""
        if vels is None or len(vels) == 0:
            return np.empty((0, 3), dtype=float)
        vels = np.asarray(vels, dtype=float).reshape(-1, 3)
        return vels @ R.T

    transform_array.velocity = transform_velocity  # type: ignore[attr-defined]
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

    def set_noise(self, meas_std, proc_std):
        """[v5.3] Live-update measurement / process noise on an existing filter."""
        self.q = float(proc_std) ** 2
        self.r = float(meas_std) ** 2
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
    """Yield (tlv_type, payload_bytes) for every TLV in a complete frame.

    [v6] If a TLV length is absurd or overruns the frame, skip/stop without
    raising. Unknown types are still yielded so callers can ignore them; the
    UART layer has already sliced exactly header totalPacketLen.
    """
    if len(frame_bytes) < FRAME_HEADER_LEN:
        return
    try:
        num_tlvs = struct.unpack_from("<I", frame_bytes, NUM_TLV_OFFSET)[0]
    except struct.error:
        return
    # Sanity: firmware usually sends a handful of TLVs.
    if num_tlvs > 64:
        num_tlvs = 64
    offset = FRAME_HEADER_LEN
    for _ in range(num_tlvs):
        if offset + TLV_HEADER_LEN > len(frame_bytes):
            break
        try:
            tlv_type, tlv_len = struct.unpack_from("<II", frame_bytes, offset)
        except struct.error:
            break
        offset += TLV_HEADER_LEN
        if tlv_len > len(frame_bytes) or tlv_len > MAX_REASONABLE_FRAME_LEN:
            # Corrupt length — stop this frame's TLV walk; caller keeps what
            # was already decoded and UART resyncs on next MAGIC_WORD.
            break
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


def extract_pointcloud(out, frame_bytes=None):
    """Return RAW sensor-frame [x,y,z,snr] or [x,y,z,doppler,snr].

    Pack-aligned preference (live_3d_radar_v9):
      1. Official parseStandardFrame pointCloud sliced to numDetectedPoints
         (cols: x,y,z, doppler, snr, ...). When >=5 cols, keep doppler.
      2. Corrected manual TLV-1020 fallback ('2bh2H' compressed spherical).
    """
    pts = _pointcloud_from_parser(out)
    if pts is not None and len(pts) > 0:
        return pts
    if frame_bytes:
        try:
            return _manual_parse_tlv1020(frame_bytes)
        except Exception as e:
            log.debug("TLV1020 manual fallback failed: %s", e)
    return np.empty((0, 4), dtype=float)


def extract_track_indexes(out, frame_bytes=None, n_points=None):
    """[v6.8] TLV 1011 point-to-track indexes aligned with the point cloud.

    One uint8 per detected point, SAME ORDER as the point cloud in that frame.
    Values 0–249 (typical) = associated track tid. Unassociated sentinels:
      253 = TRACK_INDEX_WEAK_SNR, 254 = TRACK_INDEX_BOUNDS,
      255 = TRACK_INDEX_NOISE.

    Prefer ``out['trackIndexes']`` from the official TI parser; else walk
    TLV type 1011. Truncate/pad to ``n_points`` (pad with 255) when given.
    Returns int ndarray shape (N,) or empty.
    """
    idxs = None
    if out is not None:
        ti = out.get("trackIndexes", None)
        if ti is not None:
            try:
                arr = np.asarray(ti).ravel()
                if len(arr) > 0:
                    idxs = arr.astype(int, copy=False)
            except Exception:
                idxs = None
    if idxs is None and frame_bytes:
        try:
            for tlv_type, payload in iter_tlvs(frame_bytes):
                if tlv_type != TLV_TARGET_INDEX_1011:
                    continue
                if payload:
                    idxs = np.frombuffer(payload, dtype=np.uint8).astype(int)
                break
        except Exception as e:
            log.debug("extract_track_indexes TLV walk failed: %s", e)
            idxs = None
    if idxs is None:
        idxs = np.empty(0, dtype=int)
    else:
        idxs = np.asarray(idxs, dtype=int).ravel()
    if n_points is not None:
        n = int(n_points)
        if n <= 0:
            return np.empty(0, dtype=int)
        if len(idxs) > n:
            idxs = idxs[:n].copy()
        elif len(idxs) < n:
            pad = np.full(n - len(idxs), TRACK_INDEX_NOISE, dtype=int)
            idxs = np.concatenate([idxs, pad]) if len(idxs) else pad
    return idxs


def _pointcloud_from_parser(out):
    pc = out.get("pointCloud", None) if out else None
    if pc is None:
        return None
    arr = np.asarray(pc, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
        return None
    n = int(out.get("numDetectedPoints", arr.shape[0]) or arr.shape[0])
    n = max(0, min(n, arr.shape[0]))
    if n == 0:
        return None
    arr = arr[:n]
    xyz = arr[:, :3]
    # TI layout: [x,y,z,doppler,snr,...] — keep doppler when present for
    # filter_raw_cloud; display path strips to [x,y,z,snr].
    if arr.shape[1] >= 5:
        doppler = arr[:, 3]
        snr = arr[:, 4]
        return np.column_stack([xyz, doppler, snr])
    elif arr.shape[1] == 4:
        snr = arr[:, 3]  # best-effort (may be doppler on some firmwares)
        return np.column_stack([xyz, snr])
    else:
        snr = np.zeros(len(arr))
        return np.column_stack([xyz, snr])


def _manual_parse_tlv1020(frame_bytes):
    """CORRECTED TLV-1020 compressed spherical cloud (from live_3d_radar_v9 /
    parseCompressedSphericalPointCloudTLV).

    header = 5 x float32 units: elev_u, azim_u, doppler_u, range_u, snr_u
    each point = '<2bh2H' (8 bytes): elev i8, azim i8, doppler i16,
                 range u16, snr u16
    Then spherical -> cartesian (TI convention).
    """
    for tlv_type, payload in iter_tlvs(frame_bytes):
        if tlv_type != TLV_POINT_CLOUD_1020:
            continue
        unit_struct = "<5f"
        unit_size = struct.calcsize(unit_struct)
        point_struct = "<2bh2H"
        point_size = struct.calcsize(point_struct)
        if len(payload) < unit_size:
            return np.empty((0, 4), dtype=float)
        elev_u, azim_u, dopp_u, range_u, snr_u = struct.unpack_from(unit_struct, payload, 0)
        body = payload[unit_size:]
        n = len(body) // point_size
        if n <= 0:
            return np.empty((0, 4), dtype=float)
        xs = np.empty(n); ys = np.empty(n); zs = np.empty(n)
        dd = np.empty(n); ss = np.empty(n)
        for i in range(n):
            elev_i, azim_i, dopp_i, rng_i, snr_i = struct.unpack_from(
                point_struct, body, i * point_size)
            elev = elev_i * elev_u
            azim = azim_i * azim_u
            rng = rng_i * range_u
            snr = snr_i * snr_u
            # Same as gui_common.sphericalToCartesianPointCloud (range, az, el)
            xs[i] = rng * math.sin(azim) * math.cos(elev)
            ys[i] = rng * math.cos(azim) * math.cos(elev)
            zs[i] = rng * math.sin(elev)
            dd[i] = dopp_i * dopp_u
            ss[i] = snr
        return np.column_stack([xs, ys, zs, dd, ss])
    return np.empty((0, 4), dtype=float)


def extract_height_data(out, frame_bytes):
    """Map tid -> person height (metres) from TLV 1012 heightData.

    Official parseTrackHeightTLV: each record is (tid, maxZ, minZ).
    people_tracking displays maxZ as height; we prefer |maxZ-minZ| when that
    looks like a body extent, else abs(maxZ).
    """
    heights = {}
    data = out.get("heightData", None) if out else None
    num = int(out.get("numDetectedHeights", 0) or 0) if out else 0
    if data is not None and num > 0:
        try:
            n = min(num, len(data))
            for i in range(n):
                row = data[i]
                try:
                    if isinstance(row, dict):
                        tid = int(row.get("tid", row.get("targetId", i)))
                        max_z = float(row.get("maxZ", row.get("height", 0)))
                        min_z = float(row.get("minZ", 0))
                    else:
                        tid = int(row[0])
                        max_z = float(row[1])
                        min_z = float(row[2]) if len(row) > 2 else 0.0
                    span = abs(max_z - min_z)
                    h = span if span >= 0.4 else abs(max_z)
                    if 0.6 <= h <= 2.4:
                        heights[tid] = clamp_person_height(h)
                except (TypeError, ValueError, IndexError, KeyError):
                    continue
        except TypeError:
            pass
    if heights:
        return heights
    # Manual TLV-1012 fallback
    try:
        for tlv_type, payload in iter_tlvs(frame_bytes or b""):
            if tlv_type != TLV_TARGET_HEIGHT_1012:
                continue
            fmt = "<I2f"
            size = struct.calcsize(fmt)
            n = len(payload) // size
            for i in range(n):
                try:
                    tid, max_z, min_z = struct.unpack_from(fmt, payload, i * size)
                    span = abs(max_z - min_z)
                    h = span if span >= 0.4 else abs(max_z)
                    if 0.6 <= h <= 2.4:
                        heights[int(tid)] = clamp_person_height(h)
                except struct.error:
                    break
    except Exception as e:
        log.debug("heightData manual parse failed: %s", e)
    return heights


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


def _row_to_track_full(row, fallback_idx):
    """Like _row_to_track but also pulls velocity when present.
    Returns (tid, x, y, z, vx, vy, vz).
    """
    tid, x, y, z = _row_to_track(row, fallback_idx)
    vx = vy = vz = 0.0
    dtype_names = getattr(row, "dtype", None)
    dtype_names = dtype_names.names if dtype_names is not None else None
    try:
        if dtype_names:
            if "velX" in dtype_names:
                vx, vy, vz = float(row["velX"]), float(row["velY"]), float(row["velZ"])
        elif isinstance(row, dict):
            vx = float(row.get("velX", row.get("vx", 0.0)))
            vy = float(row.get("velY", row.get("vy", 0.0)))
            vz = float(row.get("velZ", row.get("vz", 0.0)))
        else:
            # Official parseTrackTLV packs [tid,x,y,z,vx,vy,vz,...]
            if len(row) >= 7:
                vx, vy, vz = float(row[4]), float(row[5]), float(row[6])
    except (TypeError, ValueError, IndexError, KeyError):
        vx = vy = vz = 0.0
    return tid, x, y, z, vx, vy, vz


def extract_tracks_robust(out, frame_bytes):
    """Robust track extraction with TID dedupe (pack-aligned).

    Strategy (v3.3, matching official people_tracking preference):
      1. Prefer official trackData from parseTrackTLV (I27f → cols tid..vz).
      2. Manual TLV-1010 fills only tids the official path missed.
      3. Always dedupe by tid so one person never becomes two boxes.
      4. [v5.3] Keep ALL tids — never clamp/truncate to 2 (support 8+ artists).

    Returns list of (tid, x, y, z, vx, vy, vz) in SENSOR frame.
    """
    by_tid = {}

    data = out.get("trackData", None) if out else None
    num = int(out.get("numDetectedTracks", 0) or 0) if out else 0
    if data is not None and num > 0:
        try:
            n = min(num, len(data))
            for i in range(n):
                try:
                    tid, x, y, z, vx, vy, vz = _row_to_track_full(data[i], i)
                    # Sanity: reject obviously garbage official rows
                    if not all(math.isfinite(v) for v in (x, y, z)):
                        continue
                    by_tid[int(tid)] = (int(tid), float(x), float(y), float(z),
                                        float(vx), float(vy), float(vz))
                except (TypeError, ValueError, IndexError, KeyError):
                    continue
        except TypeError:
            pass

    try:
        manual = manual_parse_tracks(frame_bytes)
        for (tid, x, y, z, vx, vy, vz) in manual:
            tid = int(tid)
            if tid in by_tid:
                continue  # official already has this tid
            by_tid[tid] = (tid, float(x), float(y), float(z),
                           float(vx), float(vy), float(vz))
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


def filter_raw_cloud_with_mask(cloud, bb, snr_min, z_lo=0.0, z_hi=2.5,
                                margin=0.15, doppler_max=3.0):
    """[v6.8] Same gate as filter_raw_cloud, also returns the boolean keep mask.

    The mask is aligned with the INPUT cloud rows so parallel arrays (e.g.
    TLV-1011 point_tids) stay 1:1 with cloud_frame after filtering.
    Returns (kept_cloud [x,y,z,snr], mask).
    """
    if cloud is None or len(cloud) == 0:
        return np.empty((0, 4), dtype=float), np.zeros(0, dtype=bool)
    arr = np.asarray(cloud, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        n = 0 if arr.ndim < 1 else int(arr.shape[0])
        return np.empty((0, 4), dtype=float), np.zeros(n, dtype=bool)
    bb = bb or BOUNDARY_BOX_ROOM
    ncols = arr.shape[1]
    # layout: [x,y,z,snr] OR [x,y,z,doppler,snr] when >=5 cols
    if ncols >= 5:
        xyz = arr[:, :3]
        doppler = arr[:, 3]
        snr = arr[:, 4]
    else:
        xyz = arr[:, :3]
        doppler = None
        snr = arr[:, 3]

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    finite = (np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
              & np.isfinite(snr))
    snr_ok = snr >= float(snr_min)
    x0 = float(bb["minX"]) - float(margin)
    x1 = float(bb["maxX"]) + float(margin)
    y0 = float(bb["minY"]) - float(margin)
    y1 = float(bb["maxY"]) + float(margin)
    in_xy = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    in_z = (z >= float(z_lo)) & (z <= float(z_hi))
    mask = finite & snr_ok & in_xy & in_z
    if doppler is not None:
        d_ok = np.isfinite(doppler) & (np.abs(doppler) <= float(doppler_max))
        mask = mask & d_ok
    kept = xyz[mask]
    snr_k = snr[mask]
    if len(kept) == 0:
        return np.empty((0, 4), dtype=float), mask
    return np.column_stack([kept, snr_k]), mask


def filter_raw_cloud(cloud, bb, snr_min, z_lo=0.0, z_hi=2.5, margin=0.15,
                     doppler_max=3.0):
    """[v6.4] Filter 1 — point-cloud gate BEFORE accumulation / display stack.

    Accepts room-frame cloud as [x,y,z,snr] or [x,y,z,doppler,snr].
    Drops: non-finite; SNR < snr_min; outside BOUNDARY_BOX +/- margin;
    room-Z outside [z_lo, z_hi]; optional |doppler| > doppler_max.
    Returns [x,y,z,snr] for display / support tests.
    """
    kept, _mask = filter_raw_cloud_with_mask(
        cloud, bb, snr_min, z_lo=z_lo, z_hi=z_hi, margin=margin,
        doppler_max=doppler_max)
    return kept


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
        self._geom_lock = threading.Lock()
        self._param_lock = threading.Lock()
        self.sensor_height = float(sensor_height)
        self.tilt_deg = float(tilt_deg)
        self.transform_array = make_transform(sensor_height, tilt_deg)
        self.cloud_frames = deque(maxlen=max(1, accum_frames))
        self.max_points = max_points
        # Detection volume used for ghost rejection + cloud gating.
        self.boundary_box = dict(boundary_box or BOUNDARY_BOX_ROOM)
        self._running = True
        self.track_state = {}         # tid -> {kf, history, missed, last_update}
        self.frame_count = 0
        self.parse_errors = 0
        # [v5.3] live-tunable Kalman / gate params (copied from ParamSnapshot)
        self.measure_std = float(MEASURE_STD)
        self.process_std = float(PROCESS_STD)
        self.gate_sigma = float(GATE_SIGMA)
        self.gate_min_m = float(GATE_MIN_M)
        self.max_missed = int(MAX_MISSED)
        self.pc_snr_min = float(PC_SNR_MIN)  # [v6.2] live cloud SNR gate

    def stop(self):
        self._running = False

    def set_geometry(self, height, tilt):
        """[v5.3] Rebuild sensor->room transform under lock (live GUI)."""
        height = float(height)
        tilt = float(tilt)
        with self._geom_lock:
            self.sensor_height = height
            self.tilt_deg = tilt
            self.transform_array = make_transform(height, tilt)

    def apply_params(self, snap):
        """[v5.3] Apply a ParamSnapshot copy (Kalman + boundary + geometry)."""
        if snap is None:
            return
        with self._param_lock:
            self.measure_std = float(snap.measure_std)
            self.process_std = float(snap.process_std)
            self.gate_sigma = float(snap.gate_sigma)
            self.gate_min_m = float(snap.gate_min_m)
            self.max_missed = int(snap.max_missed)
            self.boundary_box = dict(snap.boundary_box)
            self.pc_snr_min = float(getattr(snap, "pc_snr_min", PC_SNR_MIN))
        self.set_geometry(snap.sensor_height, snap.tilt_deg)
        # Update noise on existing filters
        for st in list(self.track_state.values()):
            try:
                st["kf"].set_noise(self.measure_std, self.process_std)
            except Exception:
                pass

    def _copy_runtime_params(self):
        with self._param_lock:
            return (
                float(self.measure_std), float(self.process_std),
                float(self.gate_sigma), float(self.gate_min_m),
                int(self.max_missed), dict(self.boundary_box),
                float(self.pc_snr_min),
            )

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

                    # Always consume header totalPacketLen so one bad/odd TLV
                    # inside the frame cannot desync the UART stream. Next
                    # search is MAGIC_WORD again (never delete short of magic).
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
        # ---- Point cloud / tracks via tolerant TI parser --------------------
        # Unknown TLVs (e.g. type 52) are skipped — see parseStandardFrame wrapper.
        # Manual TLV-1010 / TLV-1020 fallbacks still run if official output is thin.
        try:
            out = parseStandardFrame(frame, demo=self.demo_name)
        except Exception as e:
            self.parse_errors += 1
            log.debug("parseStandardFrame failed: %s", e)
            out = None
        # Belt-and-suspenders: never leave this frame with a blank official
        # dict if the raw UART frame still has 1010/1020 bytes to walk.
        if not _out_has_sensor_payload(out):
            try:
                out = manual_parse_frame(frame)
            except Exception as e:
                log.debug("manual_parse_frame failed: %s", e)

        (meas_std, proc_std, gate_sigma, gate_min_m, max_missed,
         boundary_box, snr_min) = self._copy_runtime_params()
        with self._geom_lock:
            transform_array = self.transform_array

        raw_pts = extract_pointcloud(out, frame) if out is not None else extract_pointcloud(None, frame)
        # [v6.8] TLV 1011 indexes: one uint8 per point, same order as raw cloud.
        # Apply the SAME keep-mask as filter_raw_cloud so point_tids ↔ cloud_frame.
        try:
            raw_tids = extract_track_indexes(
                out, frame, n_points=len(raw_pts) if len(raw_pts) > 0 else 0)
        except Exception as e:
            log.debug("extract_track_indexes failed: %s", e)
            raw_tids = np.empty(0, dtype=int)
        if len(raw_pts) > 0:
            room_xyz = transform_array(raw_pts[:, :3])
            # Preserve optional doppler col for filter_raw_cloud
            if raw_pts.shape[1] >= 5:
                room_pts = np.column_stack(
                    [room_xyz, raw_pts[:, 3], raw_pts[:, 4]])
            else:
                room_pts = np.column_stack([room_xyz, raw_pts[:, 3]])
            # [v6.2] filter BEFORE accumulation (SNR / box / Z / doppler)
            # [v6.8] mask alignment for point_tids
            room_pts, keep_mask = filter_raw_cloud_with_mask(
                room_pts, bb=boundary_box, snr_min=snr_min,
                z_lo=0.0, z_hi=2.5, margin=0.15)
            if len(raw_tids) == len(keep_mask):
                point_tids = np.asarray(raw_tids, dtype=int)[keep_mask]
            else:
                point_tids = np.empty(0, dtype=int)
        else:
            room_pts = np.empty((0, 4))
            point_tids = np.empty(0, dtype=int)

        self.cloud_frames.append(room_pts)
        non_empty = [c for c in self.cloud_frames if len(c) > 0]
        stacked = np.vstack(non_empty) if non_empty else np.empty((0, 4))
        if len(stacked) > 0:
            stacked = filter_cloud_by_volume_z(stacked, bb=boundary_box)
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

        try:
            height_by_tid = extract_height_data(out, frame)
        except Exception as e:
            log.debug("extract_height_data failed: %s", e)
            height_by_tid = {}

        now = time.monotonic()  # [v6.6] not wall clock — avoid negative dt
        seen_ids = set()
        for track_rec in tracks:
            try:
                if len(track_rec) >= 7:
                    tid, x_raw, y_raw, z_raw, vx_raw, vy_raw, vz_raw = track_rec[:7]
                else:
                    tid, x_raw, y_raw, z_raw = track_rec[:4]
                    vx_raw = vy_raw = vz_raw = 0.0
                if not (ROI_Y_RAW[0] < y_raw < ROI_Y_RAW[1]
                        and ROI_Z_RAW[0] < z_raw < ROI_Z_RAW[1]):
                    continue
                meas = transform_array(np.array([[x_raw, y_raw, z_raw]]))[0]
                raw_xyz = (float(x_raw), float(y_raw), float(z_raw))
                room_xyz = (float(meas[0]), float(meas[1]), float(meas[2]))

                # [v6.6] Bad pose on MEASUREMENT: do not create a new track,
                # and do not delete an existing one — leave it for the coast
                # path (missed++) so Kalman/velocity/trail survive glitches.
                pose_ok = (
                    room_z_plausible(meas[2])
                    and in_boundary_box(meas[0], meas[1], meas[2], bb=boundary_box))
                if not pose_ok:
                    reason = ("room_z_out_of_band_on_measure"
                              if not room_z_plausible(meas[2])
                              else "outside_boundary_box_on_measure")
                    log_ghost_reject(
                        tid, raw_xyz, room_xyz,
                        self.track_state.get(tid, {}).get("missed", 0), reason)
                    # Existing tid: omit from seen_ids -> coast below.
                    continue

                seen_ids.add(tid)
                st = self.track_state.get(tid)
                if st is None:
                    kf = CVKalman3D(meas, meas_std=meas_std, proc_std=proc_std)
                    # Seed velocity from radar track (rotated like position).
                    try:
                        v_room = transform_array.velocity(
                            np.array([[vx_raw, vy_raw, vz_raw]], dtype=float))[0]
                        kf.x[3], kf.x[4], kf.x[5] = float(v_room[0]), float(v_room[1]), float(v_room[2])
                    except Exception:
                        pass
                    kf.clamp_z()
                    person_h = float(height_by_tid.get(tid, PERSON_HEIGHT))
                    st = {"kf": kf, "history": deque(), "missed": 0,
                          "last_update": now, "last_raw": raw_xyz,
                          "height": person_h, "updates": 1}
                    self.track_state[tid] = st
                    px, py, pz = kf.pos
                    st["history"].append((now, px, py, pz))
                else:
                    if tid in height_by_tid:
                        st["height"] = float(height_by_tid[tid])
                    kf = st["kf"]
                    dt = max(now - st["last_update"], 1e-3)
                    kf.predict(dt)
                    kf.clamp_z()
                    y_res, S = kf.innovation(meas)
                    try:
                        dist2 = float(y_res.T @ np.linalg.inv(S) @ y_res)
                    except np.linalg.LinAlgError:
                        dist2 = 0.0
                    # [v6.6] Real Mahalanobis gate + absolute OR for young tracks.
                    # Old: max(sigma^2, min_m^2/R) ≈ 144 (~12σ) — never rejected.
                    innov_norm = float(np.linalg.norm(y_res))
                    young = int(st.get("updates", len(st.get("history", ())))) < 5
                    accept = (dist2 <= float(gate_sigma) ** 2) or (
                        young and innov_norm <= float(gate_min_m))
                    if accept:
                        kf.update(meas)
                        kf.clamp_z()
                        st["missed"] = 0
                        st["updates"] = int(st.get("updates", 0)) + 1
                    else:
                        st["missed"] += 1
                    st["last_update"] = now
                    st["last_raw"] = raw_xyz
                    px, py, pz = kf.pos
                    # After update: if pose left the band, coast — do not delete.
                    if not room_z_plausible(pz) or not in_boundary_box(
                            px, py, pz, bb=boundary_box):
                        reason = ("room_z_out_of_band_after_update"
                                  if not room_z_plausible(pz)
                                  else "outside_boundary_box_after_update")
                        log_ghost_reject(tid, raw_xyz, (px, py, pz),
                                         st["missed"], reason)
                        st["missed"] += 1
                    else:
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
                # [v6.6] Coast with bad Z/box: keep filter until MAX_MISSED.
                # GUI gates still hide the box; deleting here caused ID churn.
                if not room_z_plausible(pz) or not in_boundary_box(
                        px, py, pz, bb=boundary_box):
                    reason = ("room_z_out_of_band_while_coasting"
                              if not room_z_plausible(pz)
                              else "outside_boundary_box_while_coasting")
                    log_ghost_reject(tid, raw_xyz, (px, py, pz),
                                     st["missed"], reason)
                else:
                    st["history"].append((now, px, py, pz))
                    _trim_history(st["history"], now)
                if st["missed"] > max_missed:
                    del self.track_state[tid]
            except Exception as e:
                log.debug("coast update failed for tid=%s: %s", tid, e)
                self.track_state.pop(tid, None)

        self.frame_count += 1

        # Snapshot includes all surviving tracks; the GUI only *draws* active
        # ones (missed == 0) that still pass room_z_plausible — that is also
        # how the People counter is defined.
        # [v6.8] Association stats for status bar (associated = not 253/254/255).
        try:
            _pt = np.asarray(point_tids, dtype=int).ravel()
            _assoc = int(np.sum(~np.isin(_pt, list(TRACK_INDEX_UNASSOC)))) if len(_pt) else 0
            _tlv1011 = bool(len(_pt) > 0 and _assoc > 0)
        except Exception:
            _pt = np.empty(0, dtype=int)
            _assoc = 0
            _tlv1011 = False
        snapshot = {
            "cloud": stacked,  # accumulated (display)
            # [v6.1] THIS frame only (after volume Z filter) — ghost tests use this
            "cloud_frame": np.array(room_pts, copy=True),
            # [v6.8] int array aligned 1:1 with cloud_frame (empty if no TLV 1011)
            "point_tids": np.array(_pt, copy=True),
            "tlv1011": _tlv1011,
            "ptid_assoc": _assoc,
            "tracks": {
                tid: {"pos": st["kf"].pos,
                      "history": [(h[1], h[2], h[3]) for h in st["history"]],
                      "missed": st["missed"],
                      "height": st.get("height", PERSON_HEIGHT),
                      # [v6.7] track strength for the same-cluster ghost rule
                      "updates": int(st.get("updates", 0) or 0)}
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
def track_qcolor(tid, hues=16):
    """[2. PER-TRACK COLOR] Deterministic, stable color for a track ID.
    Same tid always -> same color, so a tracked person keeps their color
    for as long as their track ID is alive, and it's consistent between
    the 3D and 2D views since both derive from this single function.
    hues=16 supports at least 8 simultaneous artists without collisions."""
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

    def update(self, pos, history, height=None):
        # [3. STABILITY] Defensive checks: don't let a bad position or an
        # empty/garbled history propagate into a GL call.
        try:
            if pos is None or len(pos) != 3:
                return
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            if not all(math.isfinite(v) for v in (x, y, z)):
                return
            h = clamp_person_height(height if height is not None else PERSON_HEIGHT)

            self.box.setData(pos=box_line_vertices(x, y, z, height=h))
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
                # [v6.10] Clear trail geometry — do not leave prior polyline
                # or a degenerate 2-point stub when history is empty/short.
                self.trail.setData(pos=np.zeros((0, 3)))
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
                # [v6.10] Clear trail curve when history is empty/short.
                self.trail.setData([], [])
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

# ---- [CUMULATIVE VISIT COUNTER + GHOST GATES] --------------------------------
# Rising-edge debounce before a visit can fire +1.
CONFIRM_FRAMES = 8  # Filter 4: frames before Total Detected += 1
# Legacy frame threshold (still used as a hard missed cutoff).
LEFT_FRAMES = 8
# Visit stays open this long after last confirm (seconds). Brief tracker drops
# while walking must NOT close the visit / cause a second +1.
LEFT_TIME = 2.5
# New tid within this distance (m) of a recently-lost open visit = reacquire.
REACQUIRE_DIST = 1.5  # align with GHOST_MERGE_DIST (was 0.9)
# Look-back window (s) for reacquire matching.
REACQUIRE_TIME = 2.0
# Merge simultaneous confirmed tids closer than this (m) into one slot.
# [v5.3] was 1.0 — too aggressive; 0.45 keeps people ~0.6–1.0 m apart as K=3.
MERGE_DIST = 0.45
# Point-cloud support: ghost boxes have almost no nearby points.
# [v5.3] prefer tracker presence when missed==0; 1 nearby point is enough.
POINT_SUPPORT_RADIUS = 0.6  # COUNT/ghost: current-frame support radius
MIN_NEAR_POINTS = 2  # COUNT/ghost: need a few current-frame points
# --- DRAW vs COUNT (v6.5) -------------------------------------------------
# DRAW uses looser gates so the box appears on the purple body cluster fast.
# COUNT / Total Detected still uses CONFIRM_FRAMES + MIN_NEAR / PC_SNR_MIN.
DRAW_MIN_NEAR_POINTS = 1
DRAW_SUPPORT_RADIUS = 1.0
DRAW_SNR_MIN = 3.0
SNAP_TO_CLOUD_DIST = 0.5  # if track XY farther than this from cluster, snap draw
DRAW_HIDE_FRAMES = 5  # hide box only after this many missed/unsupported frames
MIN_SEPARATION = 1.2  # second person must be this far from a strong first body
# Stronger-tid wins inside this radius (user: prefer stronger within 1.5 m).
# Farther radial splits (~2 m green/cyan) are killed by ONE-PERSON LOCK below,
# not by raising this — raising it would glue two real people ~1–2 m apart.
GHOST_MERGE_DIST = 1.5
DUPLICATE_SUPPRESS_DIST = GHOST_MERGE_DIST  # back-compat alias
# First tid is "strong" if it has at least this many current-frame near points;
# then a second tid must also be competitively strong (see SECOND_SCORE_RATIO).
STRONG_SUPPORT_NEAR = 4
# Second person vs strongest: score must be >= this fraction or it is a split ghost.
SECOND_SCORE_RATIO = 0.50
# Hide / reject after this many consecutive unsupported frames.
UNSUPPORTED_HIDE_FRAMES = 3  # COUNT path: support-fail streak (visit)
# DRAW path uses DRAW_HIDE_FRAMES (default 5), not this.
# Never treat a track this close to the sensor XY origin as a person box.
SENSOR_XY_EXCLUDE = 0.35
# Entry-edge lock: suppress double +1 when walking in from the yellow wall.
ENTRY_MARGIN = 0.8
ENTRY_COMMIT_DIST = 0.6
ENTRY_COMMIT_TIME = 1.2
# [v5.3] was 1.6 — only stitch onto GONE/unconfirmed entry visits.
ENTRY_STITCH_DIST = 0.9
ENTRY_STITCH_TIME = 3.0
ENTRY_LEFT_TIME = 3.5
ENTRY_REACQUIRE_DIST = 1.2



def count_near_points_xy(cloud, x, y, radius=POINT_SUPPORT_RADIUS):
    """How many cloud points lie within `radius` of (x, y) in XY."""
    if cloud is None or len(cloud) == 0:
        return 0
    try:
        dx = cloud[:, 0] - float(x)
        dy = cloud[:, 1] - float(y)
        return int(np.count_nonzero((dx * dx + dy * dy) <= (radius * radius)))
    except Exception:
        return 0


def near_point_stats(cloud, x, y, radius=POINT_SUPPORT_RADIUS):
    """Return (count, snr_sum, snr_median, centroid_xy or None) — CURRENT frame only."""
    if cloud is None or len(cloud) == 0:
        return 0, 0.0, 0.0, None
    try:
        dx = cloud[:, 0] - float(x)
        dy = cloud[:, 1] - float(y)
        mask = (dx * dx + dy * dy) <= (float(radius) * float(radius))
        count = int(np.count_nonzero(mask))
        if count == 0:
            return 0, 0.0, 0.0, None
        pts = cloud[mask]
        if pts.shape[1] > 3:
            snr_vals = pts[:, 3]
            snr_sum = float(np.sum(snr_vals))
            snr_median = float(np.median(snr_vals))
        else:
            snr_sum = 0.0
            snr_median = 0.0
        centroid_xy = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        return count, snr_sum, snr_median, centroid_xy
    except Exception:
        return 0, 0.0, 0.0, None



def associate_cloud_cluster(cloud, x, y, radius, min_near, snr_min,
                             bb=None, max_assoc_dist=2.5, n_clusters=None,
                             allow_global_fallback=True):
    """[v6.5] Associate strongest nearby CURRENT-frame cluster to a tid pose.

    Tries ``radius`` around (x, y), then 1.5x / 2x radius.

    [v6.7] The old "global in-box fallback" (snap to the strongest cloud mass
    anywhere in the yellow box) is what made ONE box wander between people.
    It is now only allowed when the scene provably holds a single body:
    ``n_clusters`` (from cluster_cloud_xy) must be < 2 and
    ``allow_global_fallback`` True.  The draw path no longer calls this at
    all -- it uses cluster_cloud_xy() + assign_boxes_to_clusters().
    Returns (near, snr_sum, snr_median, centroid_xy or None, ok:bool).
    """
    r0 = float(radius)
    near = snr_sum = snr_median = 0.0
    centroid = None
    for r in (r0, r0 * 1.5, r0 * 2.0):
        near, snr_sum, snr_median, centroid = near_point_stats(
            cloud, x, y, radius=r)
        if near >= int(min_near) and snr_median >= float(snr_min):
            return near, snr_sum, snr_median, centroid, True

    # Global in-box fallback: strongest cloud mass inside the yellow box.
    # [v6.7] Hard-disabled while 2+ clusters exist (multi-person scene).
    if not allow_global_fallback or (n_clusters is not None and int(n_clusters) >= 2):
        return near, snr_sum, snr_median, centroid, False
    if cloud is None or len(cloud) == 0:
        return near, snr_sum, snr_median, centroid, False
    try:
        arr = np.asarray(cloud, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return near, snr_sum, snr_median, centroid, False
        snr_col = arr[:, 3] if arr.shape[1] > 3 else np.ones(len(arr))
        mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & (snr_col >= float(snr_min))
        if bb is not None:
            mask = (
                mask
                & (arr[:, 0] >= float(bb["minX"])) & (arr[:, 0] <= float(bb["maxX"]))
                & (arr[:, 1] >= float(bb["minY"])) & (arr[:, 1] <= float(bb["maxY"]))
            )
            if "minZ" in bb and arr.shape[1] > 2:
                mask = mask & (arr[:, 2] >= float(bb["minZ"])) & (arr[:, 2] <= float(bb["maxZ"]))
        pts = arr[mask]
        if len(pts) < int(min_near):
            return near, snr_sum, snr_median, centroid, False
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        if math.hypot(float(x) - cx, float(y) - cy) > float(max_assoc_dist):
            # Track too far from the only body cluster — still OK if track is
            # coasting and this is the sole dense blob (one-person scene).
            if len(pts) < max(8, int(min_near) * 3):
                return near, snr_sum, snr_median, centroid, False
        med = float(np.median(pts[:, 3])) if pts.shape[1] > 3 else float(snr_min)
        return len(pts), float(np.sum(pts[:, 3])) if pts.shape[1] > 3 else 0.0, med, (cx, cy), True
    except Exception:
        return near, snr_sum, snr_median, centroid, False


def display_snap_pose(track_pos, centroid_xy, snap_dist=SNAP_TO_CLOUD_DIST):
    """[v6.5] Pull drawn box XY onto cloud centroid when Kalman drifts.

    Keep Kalman state untouched; only the displayed pose is snapped.
    Z stays from the track (floor-anchored box uses height separately).
    """
    try:
        tx, ty, tz = float(track_pos[0]), float(track_pos[1]), float(track_pos[2])
    except Exception:
        return track_pos
    if centroid_xy is None:
        return (tx, ty, tz)
    try:
        cx, cy = float(centroid_xy[0]), float(centroid_xy[1])
    except Exception:
        return (tx, ty, tz)
    if not all(math.isfinite(v) for v in (tx, ty, tz, cx, cy)):
        return (tx, ty, tz)
    d = math.hypot(tx - cx, ty - cy)
    # [v6.7] Never teleport a box: snapping is only allowed onto a centroid
    # within LOCK_RADIUS of the track.  Farther centroids belong to someone
    # else; the caller must keep the last pose instead.
    if d > float(LOCK_RADIUS):
        return (tx, ty, tz)
    if d > float(snap_dist):
        return (cx, cy, tz)
    return (tx, ty, tz)


def suppress_split_ghosts(candidates, min_near, support_radius, min_separation,
                          ghost_merge_dist, pc_snr_min=None,
                          strong_near=None, second_score_ratio=None,
                          same_cluster_dist=None, two_people_dist=None):
    """[v6.7] Filter 3 -- split-ghost, NARROWED.

    Point support must already be from CURRENT-frame cloud only (Filter 2).

    Rules:
      1. Drop tids with near < min_near or median SNR < pc_snr_min.
      2. Score = near_count * max(median_snr, 1). Sort strongest first.
      3. Two candidates closer than SAME_CLUSTER_DIST (0.5 m) claim the SAME
         cluster -> keep the stronger, drop the other (this is the cyan split).
      4. Two candidates >= TWO_PEOPLE_DIST (0.8 m) apart are TWO PEOPLE.
         Always keep both.  The old ONE_PERSON_LOCK (kill any weak second
         body while a strong first exists, even far away) is REMOVED -- it
         is what deleted real person #2 / #3.
      5. In the 0.5-0.8 m band (ambiguous): drop the second only if it is
         clearly weak (score < SECOND_SCORE_RATIO * the stronger one).

    ``min_separation`` / ``ghost_merge_dist`` / ``strong_near`` are accepted
    for call compatibility but no longer widen the kill radius.
    Returns set of surviving tids.
    """
    snr_floor = float(PC_SNR_MIN if pc_snr_min is None else pc_snr_min)
    same_d = float(SAME_CLUSTER_DIST if same_cluster_dist is None else same_cluster_dist)
    two_d = float(TWO_PEOPLE_DIST if two_people_dist is None else two_people_dist)
    ratio_need = float(SECOND_SCORE_RATIO if second_score_ratio is None else second_score_ratio)

    scored = []
    for c in candidates or ():
        try:
            near = int(c.get("near", 0) or 0)
        except Exception:
            near = 0
        if near < int(min_near):
            log.debug(
                "ghost drop tid=%s reason=no_frame_support near=%d need=%d",
                c.get("tid"), near, int(min_near))
            continue
        try:
            med = float(c.get("snr_median", c.get("snr", 0.0)) or 0.0)
        except Exception:
            med = 0.0
        if med < snr_floor:
            log.debug(
                "ghost drop tid=%s reason=low_median_snr med=%.1f need=%.1f",
                c.get("tid"), med, snr_floor)
            continue
        score = float(near) * max(med, 1.0)
        scored.append((score, near, med, c))
    scored.sort(key=lambda t: (-t[0], str(t[3].get("tid", 0))))

    kept = []  # (tid, cx, cy, score, near)
    survivors = set()
    for score, near, med, c in scored:
        tid = c.get("tid")
        cent = c.get("centroid")
        try:
            if cent is not None:
                cx, cy = float(cent[0]), float(cent[1])
            else:
                cx, cy = float(c["x"]), float(c["y"])
        except Exception:
            log.debug("ghost drop tid=%s reason=no_frame_support (bad centroid)", tid)
            continue

        drop_reason = None
        for ktid, kx, ky, kscore, knear in kept:
            d = math.hypot(cx - kx, cy - ky)
            if d < same_d:
                # Same cluster as a stronger kept body -> split ghost.
                drop_reason = "same_cluster_weaker"
                break
            if d < two_d and score < ratio_need * max(kscore, 1e-9):
                # Ambiguous band and clearly weaker -> likely a split.
                drop_reason = "ambiguous_band_weak"
                break
            # d >= two_d: two people. Keep both -- no one-person lock.

        if drop_reason is not None:
            log.debug(
                "ghost drop tid=%s reason=%s near=%s med_snr=%.1f score=%.1f vs kept=%s",
                tid, drop_reason, near, med, score,
                [k[0] for k in kept])
            continue
        kept.append((tid, cx, cy, score, near))
        survivors.add(tid)
    return survivors


# --- [v6.7] CLUSTER-FIRST BOX LOCK ---------------------------------------------
# Design: cluster the CURRENT-frame cloud inside the yellow box first, then
# assign boxes to clusters 1-to-1.  A box may only lock onto the nearest
# unmatched cluster within LOCK_RADIUS; it can never be pulled to the
# "strongest blob in the box" while another cluster exists.
CLUSTER_EPS = 0.7            # XY neighbour radius for greedy flood clustering (m)
MIN_CLUSTER_POINTS = 3       # clusters smaller than this are noise, not bodies
LOCK_RADIUS = 1.0            # a box/tid may attach only to a cluster within this (m)
SAME_CLUSTER_DIST = 0.5      # two boxes/tids closer than this share ONE cluster -> keep stronger
TWO_PEOPLE_DIST = 0.8        # clusters this far apart are two people: never merge / never lock-kill
LOCK_STICKY_BONUS = 0.3      # cost bonus (m) for re-locking onto last frame's cluster
LOCAL_BOX_ID_BASE = 1000     # box ids for clusters that have no firmware tid
BOX_TRAIL_FRAMES = 60        # per-box pose history kept for the trail
SWITCH_MARGIN = 0.4          # [v6.10] kept for logging only
SWITCH_FRAMES = 4            # [v6.11] back to v6.9 value (v6.10's 8 made follow slow); streak only for a >= SAME_CLUSTER_DIST switch while the nearer old cluster still exists
TRAIL_MIN_STEP = 0.08        # [v6.9] ignore micro-jitter below this (m)
TRAIL_MAX_STEP = 0.45        # [v6.10] jump larger than this clears trail (no teleport)
SINGLE_BODY_EXTENT = 1.0     # flood cluster wider than this (m) may be 2 people -> try split (accept only if halves >= TWO_PEOPLE_DIST apart)


def _split_wide_cluster(cl, min_pts, two_people_dist, extent_max, depth=0,
                        bin_m=0.1, valley_ratio=0.6):
    """[v6.7] Flood fill at CLUSTER_EPS chains two bodies standing ~0.8-1.0 m
    apart (their edges are closer than eps).  If a cluster's XY extent is
    wider than one person, project the points onto the principal axis and
    cut at density VALLEYS between peaks that are >= two_people_dist apart
    (valley must drop below ``valley_ratio`` of the lower peak).  Each side
    must keep >= min_pts.  Recurses so 3+ chained bodies split into 3+.
    Returns a list of point arrays.  A single person never produces two
    peaks >= 0.8 m apart with a real valley, so it stays one cluster.
    """
    n = len(cl)
    if n < 2 * int(min_pts) or depth > 4:
        return [cl]
    xy = cl[:, :2]
    mu = xy.mean(axis=0)
    cen = xy - mu
    cov = cen.T @ cen / max(n - 1, 1)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(-w)
    # Peaks only need to be ~70 % of two_people_dist apart here: the peak
    # position jitters by a bin or two, and the final centroid merge in
    # cluster_cloud_xy() (< two_people_dist -> one body) is the real arbiter.
    peak_sep = 0.7 * float(two_people_dist)
    kern = np.array([1.0, 2.0, 1.0]) / 4.0  # ~0.3 m smoothing
    cuts = []
    proj = None
    for ai in order:  # principal axis first, then the secondary one
        axis = v[:, int(ai)]
        proj = cen @ axis
        lo, hi = float(proj.min()), float(proj.max())
        if (hi - lo) <= float(extent_max):
            continue
        nb = max(3, int(math.ceil((hi - lo) / float(bin_m))) + 1)
        hist, edges = np.histogram(proj, bins=nb, range=(lo - bin_m / 2, hi + bin_m / 2))
        h = np.convolve(hist.astype(float), kern, mode="same")
        centers = 0.5 * (edges[:-1] + edges[1:])
        peaks = [i for i in range(len(h))
                 if h[i] > 0
                 and (i == 0 or h[i] >= h[i - 1])
                 and (i == len(h) - 1 or h[i] > h[i + 1])]
        if len(peaks) < 2:
            continue
        # walk peaks left->right: merge peaks closer than peak_sep (keep the
        # higher), cut at the valley between peaks that are far enough apart.
        kept = [peaks[0]]
        cuts = []
        for p in peaks[1:]:
            q = kept[-1]
            if centers[p] - centers[q] < peak_sep:
                if h[p] > h[q]:
                    kept[-1] = p
                continue
            seg = h[q:p + 1]
            vi = q + int(np.argmin(seg))
            if h[vi] <= float(valley_ratio) * min(h[p], h[q]):
                cuts.append(centers[vi])
                kept.append(p)
            elif h[p] > h[q]:
                kept[-1] = p
        if cuts:
            break
    if not cuts:
        return [cl]
    bounds = [-np.inf] + sorted(cuts) + [np.inf]
    parts = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = (proj >= a) & (proj < b)
        if int(m.sum()) >= int(min_pts):
            parts.append(cl[m])
    if len(parts) < 2:
        return [cl]
    out = []
    for part in parts:
        out.extend(_split_wide_cluster(part, min_pts, two_people_dist, extent_max, depth + 1,
                                       bin_m=bin_m, valley_ratio=valley_ratio))
    return out


def cluster_cloud_by_track_index(cloud, point_tids, bb=None,
                                  min_pts=MIN_CLUSTER_POINTS, snr_min=None,
                                  two_people_dist=TWO_PEOPLE_DIST):
    """[v6.8] Cluster CURRENT-frame cloud using firmware TLV-1011 point tids.

    For each unique tid NOT in TRACK_INDEX_UNASSOC (253/254/255), gather
    points with that tid inside BOUNDARY_BOX. If count >= min_pts, emit a
    cluster dict like cluster_cloud_xy plus a ``tid`` key for preferred lock.
    Sorted strongest-first. Returns [] when association is absent/useless so
    the caller can fall back to XY clustering.

    [v6.11] Firmware sometimes tags two bodies (or a body + multipath blob)
    with ONE tid.  Each tid's points are therefore split into XY sub-clusters
    (cluster_cloud_xy flood + density-valley split, merge_dist =
    two_people_dist): blobs >= two_people_dist apart come out as separate
    clusters that all carry the same ``tid``.  assign_boxes_to_clusters()
    then lets the box keep only the one nearest its last lock pose.
    """
    out = []
    if cloud is None or point_tids is None:
        return out
    try:
        arr = np.asarray(cloud, dtype=float)
        tids = np.asarray(point_tids, dtype=int).ravel()
        if arr.ndim != 2 or arr.shape[0] == 0 or len(tids) != len(arr):
            return out
        has_z = arr.shape[1] > 2
        has_snr = arr.shape[1] > 3
        mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
        if has_z:
            mask &= np.isfinite(arr[:, 2])
        if has_snr and snr_min is not None:
            mask &= arr[:, 3] >= float(snr_min)
        if bb is not None:
            mask &= (arr[:, 0] >= float(bb["minX"])) & (arr[:, 0] <= float(bb["maxX"]))
            mask &= (arr[:, 1] >= float(bb["minY"])) & (arr[:, 1] <= float(bb["maxY"]))
            if has_z and "minZ" in bb and "maxZ" in bb:
                mask &= (arr[:, 2] >= float(bb["minZ"])) & (arr[:, 2] <= float(bb["maxZ"]))
        # Drop unassociated sentinels (253 WEAK_SNR / 254 BOUNDS / 255 NOISE).
        assoc = ~np.isin(tids, list(TRACK_INDEX_UNASSOC))
        mask &= assoc
        if not mask.any():
            return out
        pts = arr[mask]
        tids_k = tids[mask]
        for tid in np.unique(tids_k):
            m = tids_k == tid
            cnt = int(np.count_nonzero(m))
            if cnt < int(min_pts):
                continue
            cl = pts[m]
            # [v6.11] Same tid, far-apart blobs -> separate sub-clusters (each
            # keeps this tid).  Points were already SNR/box filtered above.
            subs = cluster_cloud_xy(cl, bb=None, min_pts=min_pts, snr_min=None,
                                    merge_dist=float(two_people_dist))
            if len(subs) >= 2:
                for sc in subs:
                    sc = dict(sc)
                    sc["tid"] = int(tid)
                    out.append(sc)
                continue
            # One blob (or sub-split lost points below min_pts) -> whole tid set.
            cx = float(np.mean(cl[:, 0]))
            cy = float(np.mean(cl[:, 1]))
            cz = float(np.mean(cl[:, 2])) if has_z else 0.0
            if has_snr:
                snr_sum = float(np.sum(cl[:, 3]))
                snr_med = float(np.median(cl[:, 3]))
            else:
                snr_sum, snr_med = 0.0, 0.0
            out.append({
                "cx": cx, "cy": cy, "cz": cz, "n": cnt,
                "snr_sum": snr_sum, "snr_median": snr_med,
                "score": float(cnt) * max(snr_med, 1.0),
                "tid": int(tid),
            })
        out.sort(key=lambda c: -c["score"])
        return out
    except Exception as e:
        log.debug("cluster_cloud_by_track_index failed: %s", e)
        return out


def cluster_cloud_xy(cloud, bb=None, eps=CLUSTER_EPS, min_pts=MIN_CLUSTER_POINTS,
                     snr_min=None, merge_dist=TWO_PEOPLE_DIST, max_points=1500):
    """[v6.7] Greedy flood-fill (DBSCAN-style, numpy only) of CURRENT-frame
    cloud in XY inside the boundary box.

    Returns a list of cluster dicts sorted strongest-first::

        {"cx", "cy", "cz", "n", "snr_sum", "snr_median", "score"}

    Two clusters whose centroids end up closer than ``merge_dist`` (default
    TWO_PEOPLE_DIST) are treated as a split of the same body and absorbed
    into the stronger one.  Clusters >= merge_dist apart are kept separate.
    """
    out = []
    if cloud is None or len(cloud) == 0:
        return out
    try:
        arr = np.asarray(cloud, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
            return out
        has_z = arr.shape[1] > 2
        has_snr = arr.shape[1] > 3
        mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
        if has_z:
            mask &= np.isfinite(arr[:, 2])
        if has_snr and snr_min is not None:
            mask &= arr[:, 3] >= float(snr_min)
        if bb is not None:
            mask &= (arr[:, 0] >= float(bb["minX"])) & (arr[:, 0] <= float(bb["maxX"]))
            mask &= (arr[:, 1] >= float(bb["minY"])) & (arr[:, 1] <= float(bb["maxY"]))
            if has_z and "minZ" in bb and "maxZ" in bb:
                mask &= (arr[:, 2] >= float(bb["minZ"])) & (arr[:, 2] <= float(bb["maxZ"]))
        pts = arr[mask]
        n = len(pts)
        if n < int(min_pts):
            return out
        if n > int(max_points):
            # Keep the strongest points; keeps the N x N adjacency bounded.
            if has_snr:
                keep = np.argsort(-pts[:, 3])[: int(max_points)]
            else:
                keep = np.arange(int(max_points))
            pts = pts[keep]
            n = len(pts)

        xy = pts[:, :2]
        diff = xy[:, None, :] - xy[None, :, :]
        adj = (diff[..., 0] ** 2 + diff[..., 1] ** 2) <= float(eps) ** 2
        labels = np.full(n, -1, dtype=int)
        cur = 0
        for seed in range(n):
            if labels[seed] >= 0:
                continue
            members = np.zeros(n, dtype=bool)
            members[seed] = True
            frontier = members.copy()
            while frontier.any():
                nb = adj[frontier].any(axis=0) & ~members
                members |= nb
                frontier = nb
            labels[members] = cur
            cur += 1

        raw = []
        parts = []
        for k in range(cur):
            m = labels == k
            if int(np.count_nonzero(m)) < int(min_pts):
                continue
            sub = _split_wide_cluster(pts[m], min_pts, merge_dist, SINGLE_BODY_EXTENT)
            # remember which parts were separated by a density valley
            parts.extend((cl, len(sub) > 1) for cl in sub)
        for cl, was_split in parts:
            cnt = int(len(cl))
            if cnt < int(min_pts):
                continue
            cx = float(np.mean(cl[:, 0]))
            cy = float(np.mean(cl[:, 1]))
            cz = float(np.mean(cl[:, 2])) if has_z else 0.0
            if has_snr:
                snr_sum = float(np.sum(cl[:, 3]))
                snr_med = float(np.median(cl[:, 3]))
            else:
                snr_sum, snr_med = 0.0, 0.0
            raw.append({
                "cx": cx, "cy": cy, "cz": cz, "n": cnt,
                "snr_sum": snr_sum, "snr_median": snr_med,
                "score": float(cnt) * max(snr_med, 1.0),
                "split": bool(was_split),
            })
        raw.sort(key=lambda c: -c["score"])

        # Same-body merge: two flood fragments with centroids closer than
        # merge_dist (TWO_PEOPLE_DIST) are one body.  Parts that a density
        # valley separated carry positive evidence of two bodies, so they may
        # sit a little closer (0.9 x) -- mixing pulls their centroids inward.
        for c in raw:
            host = None
            for m in out:
                d = math.hypot(c["cx"] - m["cx"], c["cy"] - m["cy"])
                lim = float(merge_dist) * (0.9 if (c["split"] and m["split"]) else 1.0)
                if d < lim:
                    host = m
                    break
            if host is None:
                out.append(dict(c))
                continue
            tot = host["n"] + c["n"]
            host["cx"] = (host["cx"] * host["n"] + c["cx"] * c["n"]) / tot
            host["cy"] = (host["cy"] * host["n"] + c["cy"] * c["n"]) / tot
            host["cz"] = (host["cz"] * host["n"] + c["cz"] * c["n"]) / tot
            host["n"] = tot
            host["snr_sum"] += c["snr_sum"]
            host["snr_median"] = max(host["snr_median"], c["snr_median"])
            host["score"] = float(tot) * max(host["snr_median"], 1.0)
        for c in out:
            c.pop("split", None)
        out.sort(key=lambda c: -c["score"])
        return out
    except Exception as e:
        log.debug("cluster_cloud_xy failed: %s", e)
        return out


def _greedy_match(pairs, claimed):
    """pairs: list of (cost, tiebreak, key, cluster_idx) sorted ascending.
    1-to-1 greedy: each key and each cluster used at most once."""
    used_keys = set()
    match = {}
    for _cost, _tb, key, ci in pairs:
        if key in used_keys or claimed[ci] is not None:
            continue
        used_keys.add(key)
        claimed[ci] = key
        match[key] = ci
    return match


def append_trail_pose(prev_hist, new_pos, min_step=TRAIL_MIN_STEP,
                      max_step=TRAIL_MAX_STEP, max_len=BOX_TRAIL_FRAMES,
                      prev_raw=None):
    """[v6.10] Filtered trail history. Returns ``(hist, trail_reset)``.

    - No history -> start with the first point.
    - step < min_step -> do not append (keep previous hist).
    - step > max_step -> CLEAR history; start with new pose only (no teleport
      segment across a lock jump / room-scale zig-zag).
    - else append EMA-smoothed XY (0.7*prev + 0.3*new), capped to max_len.
    - After any append, sanitize adjacent pairs: if ANY consecutive step
      exceeds max_step, truncate to ``[last_pt]`` and set trail_reset.
      Never accept a history that contains a jump > max_step.
    - [v6.11] ``prev_raw`` (optional): previous RAW lock pose.  When given,
      the teleport test (step > max_step) uses the raw lock-to-lock step
      instead of the EMA-lagged last trail point, so a steady walker
      (~0.2 m/frame, EMA lag ~0.5 m) no longer clears the trail every few
      frames.  Drawn segments are still sanitized against max_step.
    """
    try:
        nx = float(new_pos[0]); ny = float(new_pos[1])
        nz = float(new_pos[2]) if len(new_pos) > 2 else 0.0
    except Exception:
        return list(prev_hist or []), False
    if not (math.isfinite(nx) and math.isfinite(ny) and math.isfinite(nz)):
        return list(prev_hist or []), False
    new_pt = (nx, ny, nz)
    max_step = float(max_step)
    hist = list(prev_hist or [])
    if not hist:
        return [new_pt], False
    last = hist[-1]
    try:
        lx, ly = float(last[0]), float(last[1])
    except Exception:
        return [new_pt], True
    step = math.hypot(nx - lx, ny - ly)
    if step < float(min_step):
        return hist, False
    jump = step
    if prev_raw is not None:
        try:
            rx, ry = float(prev_raw[0]), float(prev_raw[1])
            if math.isfinite(rx) and math.isfinite(ry):
                jump = math.hypot(nx - rx, ny - ry)
        except Exception:
            pass
    if jump > max_step:
        return [new_pt], True
    lz = float(last[2]) if len(last) > 2 else nz
    ema = (0.7 * lx + 0.3 * nx, 0.7 * ly + 0.3 * ny, 0.7 * lz + 0.3 * nz)
    # Hard reject EMA result that would still jump > max_step (assert-equivalent).
    if math.hypot(ema[0] - lx, ema[1] - ly) > max_step:
        return [new_pt], True
    hist = hist + [ema]
    if len(hist) > int(max_len):
        hist = hist[-int(max_len):]
    # Sanitize: never keep a history with any adjacent jump > max_step.
    for i in range(1, len(hist)):
        try:
            x0, y0 = float(hist[i - 1][0]), float(hist[i - 1][1])
            x1, y1 = float(hist[i][0]), float(hist[i][1])
        except Exception:
            return [hist[-1]], True
        if math.hypot(x1 - x0, y1 - y0) > max_step:
            return [hist[-1]], True
    return hist, False


def assign_boxes_to_clusters(tracks, clusters, prev_boxes, lock_radius=LOCK_RADIUS,
                             hide_frames=DRAW_HIDE_FRAMES,
                             same_cluster_dist=SAME_CLUSTER_DIST,
                             sticky_bonus=LOCK_STICKY_BONUS,
                             next_local_id=LOCAL_BOX_ID_BASE,
                             default_height=PERSON_HEIGHT,
                             switch_margin=SWITCH_MARGIN,
                             switch_frames=SWITCH_FRAMES,
                             trail_min_step=TRAIL_MIN_STEP,
                             trail_max_step=TRAIL_MAX_STEP,
                             min_cluster_points=MIN_CLUSTER_POINTS,
                             two_people_dist=TWO_PEOPLE_DIST):
    """[v6.7/v6.11] 1-to-1 box <-> cluster lock.  Returns (boxes, next_local_id).

    ``boxes`` is a dict ``box_id -> state`` shaped like the parser ``tracks``
    dict (``pos``, ``missed``, ``history``, ``height``) plus ``tid``,
    ``locked``, ``near``, ``snr_median``, and [v6.9+] ``switch_candidate``,
    ``switch_streak``, ``lock_cx``/``lock_cy``, ``draw_pos``, ``trail_gen``.

    The lock, in priority order:
      0. [v6.8/v6.11] Clusters with a firmware ``tid`` key (TLV 1011) lock onto
         that exact track id -- but only the same-tid cluster nearest the
         box's last lock pose, within lock_radius and < two_people_dist
         (new tid: nearest to Kalman pose within lock_radius).
      1. Remaining firmware tids (TLV 1010) lock onto the nearest UNMATCHED
         cluster within ``lock_radius``.  Anchor = the box's last locked pose
         (with a sticky bonus) once it exists, else the Kalman pose for the
         first lock.  Nearest pair wins; a tid that loses can NOT take another
         cluster it is farther from -> no stealing.  Tids sitting ON the same
         cluster (< same_cluster_dist) tie on distance, so the STRONGER track
         wins.
      2. Existing local boxes re-lock onto remaining clusters the same way.
      3. [v6.11] Simple switch gate: a box re-locks immediately onto the
         nearest cluster to its last lock pose (same frame).  Only a switch
         to a cluster >= same_cluster_dist away while a nearer old cluster
         still exists needs ``switch_frames`` consecutive frames; meanwhile
         the proposed cluster stays unclaimed so pass 3 spawns a box on it.
         Same-tid distance veto everywhere: clusters >= two_people_dist apart
         are two boxes even when TLV 1011 gives them the same tid.
         ``switch_margin`` / ``min_cluster_points`` are accepted for API
         compatibility only.
      4. Every cluster still unmatched becomes a NEW local box (1000+).  It is
         ``pending`` (not drawn, not counted) until it re-locks on the next
         frame, so a one-frame spurious split cannot flash a box.
      5. A box that found no cluster this frame:
         - is dropped at once if it sits within ``same_cluster_dist`` of a
           cluster that another box locked (two ids on ONE body -> keep the
           one that locked, drop the other = the split ghost);
         - otherwise holds its last pose for ``hide_frames`` frames, then drops.
    There is deliberately NO global "strongest blob in the box" fallback.
    """
    prev_boxes = prev_boxes or {}
    tracks = tracks or {}
    clusters = clusters or []
    claimed = [None] * len(clusters)
    boxes = {}
    switch_margin = float(switch_margin)
    switch_frames = int(switch_frames)
    trail_min_step = float(trail_min_step)
    trail_max_step = float(trail_max_step)
    min_cluster_points = int(min_cluster_points)
    two_people_dist = float(two_people_dist)

    def _finite_xy(p):
        try:
            return (p is not None and len(p) >= 2
                    and math.isfinite(float(p[0])) and math.isfinite(float(p[1])))
        except Exception:
            return False

    # ---- pass 0: exact firmware tid from TLV-1011 clusters ---------------
    tid_match = {}
    tid_info = {}
    for tid, st in tracks.items():
        pos = st.get("pos")
        if not _finite_xy(pos):
            continue
        kx, ky = float(pos[0]), float(pos[1])
        if track_near_sensor_xy(kx, ky):
            continue
        prev = prev_boxes.get(tid)
        px = py = None
        if prev is not None:
            # [v6.9] Prefer last locked cluster centroid over raw pos.
            if prev.get("lock_cx") is not None and prev.get("lock_cy") is not None:
                try:
                    px, py = float(prev["lock_cx"]), float(prev["lock_cy"])
                except Exception:
                    px = py = None
            if px is None and _finite_xy(prev.get("pos")):
                px, py = float(prev["pos"][0]), float(prev["pos"][1])
        strength = len(st.get("history", []) or []) + int(st.get("updates", 0) or 0)
        tid_info[tid] = (kx, ky, px, py, strength)

    def _cluster_tid(ci):
        ctid = clusters[ci].get("tid", None)
        if ctid is None:
            return None
        try:
            return int(ctid)
        except (TypeError, ValueError):
            return None

    def _nearest_same_tid_cluster(tid, ax, ay, has_prev, only_unclaimed=True):
        """[v6.11] SAME-TID DISTANCE VETO.

        Firmware TLV 1011 sometimes tags two bodies (or a body + multipath
        blob) with ONE tid, so "same tid" alone must never merge two clusters
        or make a box jump across the room.  Among all clusters carrying
        ``tid``, return the one NEAREST the anchor (box's last lock pose, or
        the Kalman pose for a brand-new tid) -- but only if it is within
        lock_radius and, for an existing box, closer than two_people_dist
        (clusters >= two_people_dist apart are two people).  Other same-tid
        clusters are left alone so pass 3 gives them their own local box.
        Returns (ci, dist) or (None, None).
        """
        best_ci, best_d = None, None
        for ci in range(len(clusters)):
            if only_unclaimed and claimed[ci] is not None:
                continue
            if _cluster_tid(ci) != tid:
                continue
            d = math.hypot(ax - clusters[ci]["cx"], ay - clusters[ci]["cy"])
            if best_d is None or d < best_d:
                best_ci, best_d = ci, d
        if best_ci is None:
            return None, None
        if best_d > float(lock_radius):
            return None, None       # beyond LOCK_RADIUS -> not this box's body
        if has_prev and best_d >= float(two_people_dist):
            return None, None       # >= TWO_PEOPLE_DIST -> another person; no jump
        return best_ci, best_d

    for tid, (kx, ky, px, py, strength) in tid_info.items():
        try:
            itid = int(tid)
        except (TypeError, ValueError):
            continue
        if itid in TRACK_INDEX_UNASSOC:
            continue
        # [v6.11] Exact-tid lock is distance-vetoed: anchor = last lock pose
        # (lock_cx/lock_cy, else prev pos); brand-new tid -> Kalman pose.
        # A same-tid cluster that is too far is NOT taken: the box holds its
        # last pose (pass 4) and the far cluster becomes a new local box.
        has_prev = px is not None
        ax, ay = (px, py) if has_prev else (kx, ky)
        ci, _d = _nearest_same_tid_cluster(itid, ax, ay, has_prev)
        if ci is None:
            continue
        claimed[ci] = tid
        tid_match[tid] = ci

    # ---- pass 1: remaining firmware tids (greedy distance) ---------------
    pairs = []
    for tid, (kx, ky, px, py, strength) in tid_info.items():
        if tid in tid_match:
            continue  # already exact-locked via TLV 1011 cluster tid
        for ci, c in enumerate(clusters):
            if claimed[ci] is not None:
                continue
            if px is not None:
                dist = math.hypot(px - c["cx"], py - c["cy"])
                cost = dist - float(sticky_bonus)
            else:
                dist = math.hypot(kx - c["cx"], ky - c["cy"])
                cost = dist
            if dist <= float(lock_radius):
                # [v6.10/v6.11] Never greedily assign a locked box across a
                # two-people gap (>= two_people_dist from its last lock pose),
                # even if the cluster carries this box's tid; leave that
                # cluster unclaimed so pass 3 spawns its own box.
                if px is not None and dist >= float(two_people_dist):
                    continue
                if dist < float(same_cluster_dist):
                    cost = 0.0
                pairs.append((cost, -strength, tid, ci))
    pairs.sort(key=lambda t: (t[0], t[1], str(t[2])))
    tid_match.update(_greedy_match(pairs, claimed))

    # ---- pass 2: existing local boxes re-lock ------------------------------
    pairs = []
    for bid, prev in prev_boxes.items():
        if bid in tracks or prev.get("tid") is not None:
            continue  # tid-backed boxes handled in pass 1
        # [v6.9] Prefer lock_cx/lock_cy anchor so a single jumped pos cannot
        # permanently retarget sticky matching onto the wrong blob.
        if prev.get("lock_cx") is not None and prev.get("lock_cy") is not None:
            try:
                px, py = float(prev["lock_cx"]), float(prev["lock_cy"])
            except Exception:
                px = py = None
        else:
            px = py = None
        if px is None:
            if not _finite_xy(prev.get("pos")):
                continue
            px, py = float(prev["pos"][0]), float(prev["pos"][1])
        for ci, c in enumerate(clusters):
            if claimed[ci] is not None:
                continue
            d = math.hypot(px - c["cx"], py - c["cy"])
            if d <= float(lock_radius):
                if d >= float(two_people_dist):
                    continue  # [v6.10] clear separation -> second box, not steal
                pairs.append((d - float(sticky_bonus), 0, bid, ci))
    pairs.sort(key=lambda t: (t[0], str(t[2])))
    local_match = _greedy_match(pairs, claimed)

    # ---- pass 2b: [v6.11] simple switch gate (simplified v6.9/v6.10 layer) ----
    def _lock_anchor(prev):
        if prev is None:
            return None, None
        if prev.get("lock_cx") is not None and prev.get("lock_cy") is not None:
            try:
                return float(prev["lock_cx"]), float(prev["lock_cy"])
            except Exception:
                pass
        if _finite_xy(prev.get("pos")):
            return float(prev["pos"][0]), float(prev["pos"][1])
        return None, None

    def _own_tid_for(bid, prev):
        """Firmware tid this box owns (box id for tid-backed, else prev.tid)."""
        if bid in tracks:
            try:
                return int(bid)
            except (TypeError, ValueError):
                return None
        if prev is not None and prev.get("tid") is not None:
            try:
                return int(prev["tid"])
            except (TypeError, ValueError):
                return None
        return None

    def _same_tid_immediate(bid, ci):
        """TLV-1011: firmware tid on cluster matches this box -> instant re-lock.

        [v6.11] SAME-TID DISTANCE VETO: a matching tid is NOT enough.  The
        cluster must also be within lock_radius AND closer than
        two_people_dist to the box's lock anchor (last lock pose).  A
        same-tid cluster >= two_people_dist away is a second body (or a
        multipath blob) that the firmware mislabelled -> return False so the
        box never jumps to it.
        """
        ctid = _cluster_tid(ci)
        if ctid is None:
            return False
        prev = prev_boxes.get(bid)
        own = _own_tid_for(bid, prev)
        if own is None or own != ctid:
            return False
        ax, ay = _lock_anchor(prev)
        if ax is None:
            return True  # no anchor yet (brand-new box) -> pass 0 already vetoed vs Kalman
        d = math.hypot(ax - clusters[ci]["cx"], ay - clusters[ci]["cy"])
        if d > float(lock_radius) or d >= float(two_people_dist):
            return False
        return True

    def _cand_key_of(ci):
        ctid = _cluster_tid(ci)
        if ctid is not None:
            return ("tid", ctid)
        return ("xy", round(float(clusters[ci]["cx"]), 1),
                round(float(clusters[ci]["cy"]), 1))

    def _hysteresis_filter(match):
        """[v6.11] Simple switch gate; rewrites ``match`` in place.

        - Nearest cluster to the last lock pose (or < same_cluster_dist, or
          same tid within the veto) -> re-lock IMMEDIATELY (walking updates
          the box the same frame; no hold, no streak).
        - Proposed cluster >= same_cluster_dist away while a NEARER old
          cluster still exists (and is free for this box) -> keep the old
          cluster until the proposal has won ``switch_frames`` frames in a
          row.  The rejected cluster is left UNCLAIMED so pass 3 gives it its
          own box.  No weak->strong promote, no reservation (removed v6.10).
        """
        for bid, ci in list(match.items()):
            prev = prev_boxes.get(bid)
            if prev is None or prev.get("pending"):
                continue
            ax, ay = _lock_anchor(prev)
            if ax is None:
                continue
            if _same_tid_immediate(bid, ci):
                continue
            d_cur = math.hypot(ax - clusters[ci]["cx"], ay - clusters[ci]["cy"])
            if d_cur < float(same_cluster_dist):
                continue  # same body / walking step -> same-frame update
            # Nearest OTHER cluster to the anchor that is closer than the
            # proposal and still free for this box = "old cluster still exists".
            old_ci, old_d = None, None
            for oj, c in enumerate(clusters):
                if oj == ci:
                    continue
                if claimed[oj] is not None and claimed[oj] != bid:
                    continue
                d = math.hypot(ax - c["cx"], ay - c["cy"])
                if d < d_cur and d <= float(lock_radius) and (old_d is None or d < old_d):
                    old_ci, old_d = oj, d
            if old_ci is None:
                continue  # proposal is the nearest available -> immediate
            cand_key = _cand_key_of(ci)
            prev_cand = prev.get("switch_candidate")
            try:
                if prev_cand is not None and tuple(prev_cand) == tuple(cand_key):
                    streak = int(prev.get("switch_streak", 0) or 0) + 1
                else:
                    streak = 1
            except Exception:
                streak = 1
            if streak >= switch_frames:
                log.debug("box switch id=%s streak=%d d_new=%.2f", bid, streak, d_cur)
                continue
            # Keep the old cluster this frame; free the proposal for pass 3.
            if claimed[ci] == bid:
                claimed[ci] = None
            claimed[old_ci] = bid
            match[bid] = old_ci
            _hyst_state[bid] = (cand_key, streak)

    _hyst_state = {}  # bid -> (switch_candidate, switch_streak)
    _hysteresis_filter(tid_match)
    _hysteresis_filter(local_match)

    def _ema_draw(prev, pos):
        if prev is not None and prev.get("draw_pos") is not None:
            try:
                dp = prev["draw_pos"]
                return (0.7 * float(dp[0]) + 0.3 * float(pos[0]),
                        0.7 * float(dp[1]) + 0.3 * float(pos[1]),
                        0.7 * float(dp[2]) + 0.3 * float(pos[2]))
            except Exception:
                pass
        return tuple(pos)

    def _lock(bid, tid, ci, height, pending=False):
        c = clusters[ci]
        pos = (float(c["cx"]), float(c["cy"]), float(c["cz"]))
        prev = prev_boxes.get(bid)
        prev_hist = list(prev.get("history", []) or []) if prev else []
        prev_gen = int(prev.get("trail_gen", 0) or 0) if prev else 0
        # New box id / pending -> clear trail.
        if prev is None or pending:
            hist, trail_reset = [pos], True
        else:
            # [v6.11] teleport test vs previous RAW lock pose (not EMA trail).
            _raw = _lock_anchor(prev)
            hist, trail_reset = append_trail_pose(
                prev_hist, pos, trail_min_step, trail_max_step, BOX_TRAIL_FRAMES,
                prev_raw=_raw if _raw[0] is not None else None)
        # Tid swap (box reused with different firmware tid) -> reset trail.
        if prev is not None and tid is not None and prev.get("tid") is not None:
            try:
                if int(prev["tid"]) != int(tid):
                    hist, trail_reset = [pos], True
            except (TypeError, ValueError):
                pass
        # [v6.11] Lock jump guard: a new lock pose > two_people_dist from the
        # previous lock pose is a different body -> clear the trail and never
        # append (draw) the jump segment.
        if prev is not None and not pending:
            plx, ply = _lock_anchor(prev)
            if plx is not None and math.hypot(pos[0] - plx, pos[1] - ply) > two_people_dist:
                hist, trail_reset = [pos], True
        trail_gen = prev_gen + 1 if trail_reset else prev_gen
        # Streak bookkeeping: kept only while a switch is being counted.
        # [v6.11] The lock anchor always follows the cluster the box is on
        # (v6.10 froze it during a streak, which made the box stick).
        hs = _hyst_state.get(bid)
        if hs is not None and match_has_pending_hyst(bid, ci):
            sw_cand, sw_streak = hs
        else:
            sw_cand, sw_streak = None, 0
        lcx, lcy = float(c["cx"]), float(c["cy"])
        boxes[bid] = {
            "tid": tid, "pos": pos, "missed": 0, "locked": True,
            "pending": bool(pending),
            "history": hist, "height": height,
            "near": int(c["n"]), "snr_median": float(c["snr_median"]),
            "cluster": ci,
            "lock_cx": lcx, "lock_cy": lcy,
            "switch_candidate": sw_cand, "switch_streak": int(sw_streak),
            "draw_pos": _ema_draw(prev, pos),
            "trail_gen": int(trail_gen),
        }

    def match_has_pending_hyst(bid, ci):
        """True when we kept old cluster but are still counting a candidate."""
        hs = _hyst_state.get(bid)
        if hs is None:
            return False
        cand, streak = hs
        if cand is None or streak <= 0:
            return False
        # If we accepted the NEW cluster this frame, clear.
        if cand == _cand_key_of(ci):
            return False  # we switched onto the candidate — reset
        return True

    for tid, ci in tid_match.items():
        h = tracks[tid].get("height", default_height)
        _lock(tid, tid, ci, h)
    for bid, ci in local_match.items():
        _lock(bid, None, ci, prev_boxes[bid].get("height", default_height))

    # ---- pass 3: leftover clusters -> new local boxes -----------------------
    for ci, c in enumerate(clusters):
        if claimed[ci] is not None:
            continue
        bid = int(next_local_id)
        next_local_id += 1
        claimed[ci] = bid
        _lock(bid, None, ci, default_height, pending=True)

    # ---- pass 4: boxes without a cluster this frame: ghost-drop or hold -----
    locked_xy = [(clusters[b["cluster"]]["cx"], clusters[b["cluster"]]["cy"])
                 for b in boxes.values() if b.get("cluster") is not None]

    def _near_locked(x, y):
        for lx, ly in locked_xy:
            if math.hypot(x - lx, y - ly) < float(same_cluster_dist):
                return True
        return False

    for bid, prev in prev_boxes.items():
        if bid in boxes:
            continue
        pos = prev.get("pos")
        if not _finite_xy(pos):
            continue
        px, py = float(pos[0]), float(pos[1])
        near_claimed = _near_locked(px, py)
        if not near_claimed and bid in tid_info:
            kx, ky = tid_info[bid][0], tid_info[bid][1]
            near_claimed = _near_locked(kx, ky)
        if near_claimed:
            log.debug("box drop id=%s reason=same_cluster_as_locked_box", bid)
            continue
        if prev.get("pending"):
            continue  # never confirmed -> no hold
        missed = int(prev.get("missed", 0) or 0) + 1
        if missed >= int(hide_frames):
            log.debug("box drop id=%s reason=no_cluster_for_%d_frames", bid, missed)
            continue
        b = dict(prev)
        b["missed"] = missed
        b["locked"] = False
        b["cluster"] = None
        # Preserve / update hysteresis streak if we rejected a switch.
        hs = _hyst_state.get(bid)
        if hs is not None:
            b["switch_candidate"], b["switch_streak"] = hs[0], int(hs[1])
        else:
            b["switch_candidate"] = None
            b["switch_streak"] = 0
        boxes[bid] = b
    return boxes, next_local_id



def clamp_person_height(h, default=PERSON_HEIGHT):
    """Clamp TLV-1012 height into a plausible standing-human range."""
    try:
        h = float(h)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(h):
        return default
    return min(max(h, 0.8), 2.2)


def track_near_sensor_xy(x, y, exclude=SENSOR_XY_EXCLUDE):
    """True if (x,y) is on/near the sensor marker — never a person box."""
    try:
        return (float(x) * float(x) + float(y) * float(y)) <= (exclude * exclude)
    except (TypeError, ValueError):
        return False


def _bb_xy(bb):
    """Return (minX, maxX, minY, maxY) from a boundary-box dict."""
    bb = bb or BOUNDARY_BOX_ROOM
    return (
        float(bb["minX"]), float(bb["maxX"]),
        float(bb["minY"]), float(bb["maxY"]),
    )


def dist_to_boundary_wall_xy(x, y, bb):
    """Inward distance (m) to the nearest XY wall if inside the box, else -1."""
    try:
        x, y = float(x), float(y)
        minX, maxX, minY, maxY = _bb_xy(bb)
    except (TypeError, ValueError, KeyError):
        return -1.0
    if not (math.isfinite(x) and math.isfinite(y)):
        return -1.0
    if not (minX <= x <= maxX and minY <= y <= maxY):
        return -1.0
    return min(x - minX, maxX - x, y - minY, maxY - y)


def in_entry_band(x, y, bb, margin=ENTRY_MARGIN):
    """True if (x,y) is inside the box and within `margin` of an XY wall."""
    d = dist_to_boundary_wall_xy(x, y, bb)
    return d >= 0.0 and d <= float(margin)


def inward_progress(x, y, first_pos, bb):
    """How much farther from the nearest wall than `first_pos` (meters)."""
    if not first_pos:
        return 0.0
    try:
        d_now = dist_to_boundary_wall_xy(x, y, bb)
        d_first = dist_to_boundary_wall_xy(first_pos[0], first_pos[1], bb)
    except (TypeError, ValueError, IndexError):
        return 0.0
    if d_now < 0.0 or d_first < 0.0:
        return 0.0
    return max(0.0, d_now - d_first)


# =============================================================================
# Live-tunable parameter snapshot (thread-safe copy for GUI <-> parser)
# =============================================================================
@dataclass
class ParamSnapshot:
    """Immutable-ish snapshot of live GUI parameters.

    MainWindow holds the current instance under ``_param_lock``. ParserThread
    and VisitCounter read a ``replace()``/dict copy each frame / via setters.
    """
    measure_std: float = MEASURE_STD
    process_std: float = PROCESS_STD
    gate_sigma: float = GATE_SIGMA
    gate_min_m: float = GATE_MIN_M
    max_missed: int = MAX_MISSED
    confirm_frames: int = CONFIRM_FRAMES
    left_time: float = LEFT_TIME
    entry_left_time: float = ENTRY_LEFT_TIME
    merge_dist: float = MERGE_DIST
    min_near_points: int = MIN_NEAR_POINTS
    reacquire_dist: float = REACQUIRE_DIST
    entry_stitch_dist: float = ENTRY_STITCH_DIST
    entry_reacquire_dist: float = ENTRY_REACQUIRE_DIST
    point_support_radius: float = POINT_SUPPORT_RADIUS
    min_separation: float = MIN_SEPARATION
    ghost_merge_dist: float = GHOST_MERGE_DIST
    duplicate_suppress_dist: float = DUPLICATE_SUPPRESS_DIST
    pc_snr_min: float = PC_SNR_MIN
    # [v6.5] DRAW gates (separate from count/ghost)
    draw_min_near_points: int = DRAW_MIN_NEAR_POINTS
    draw_support_radius: float = DRAW_SUPPORT_RADIUS
    draw_snr_min: float = DRAW_SNR_MIN
    snap_to_cloud_dist: float = SNAP_TO_CLOUD_DIST
    draw_hide_frames: int = DRAW_HIDE_FRAMES
    # [v6.7] cluster-first box lock
    cluster_eps: float = CLUSTER_EPS
    min_cluster_points: int = MIN_CLUSTER_POINTS
    lock_radius: float = LOCK_RADIUS
    same_cluster_dist: float = SAME_CLUSTER_DIST
    two_people_dist: float = TWO_PEOPLE_DIST
    switch_margin: float = SWITCH_MARGIN
    switch_frames: int = SWITCH_FRAMES
    trail_min_step: float = TRAIL_MIN_STEP
    trail_max_step: float = TRAIL_MAX_STEP
    boundary_box: dict = field(default_factory=lambda: dict(BOUNDARY_BOX_ROOM))
    sensor_height: float = SENSOR_HEIGHT
    tilt_deg: float = TILT_DEG

    def copy(self):
        return replace(self, boundary_box=dict(self.boundary_box))


def default_param_snapshot(boundary_box=None, sensor_height=None, tilt_deg=None):
    """Build a ParamSnapshot from code defaults (+ optional CLI overrides)."""
    bb = dict(boundary_box or BOUNDARY_BOX_ROOM)
    return ParamSnapshot(
        measure_std=MEASURE_STD,
        process_std=PROCESS_STD,
        gate_sigma=GATE_SIGMA,
        gate_min_m=GATE_MIN_M,
        max_missed=MAX_MISSED,
        confirm_frames=CONFIRM_FRAMES,
        left_time=LEFT_TIME,
        entry_left_time=ENTRY_LEFT_TIME,
        merge_dist=MERGE_DIST,
        min_near_points=MIN_NEAR_POINTS,
        reacquire_dist=REACQUIRE_DIST,
        entry_stitch_dist=ENTRY_STITCH_DIST,
        entry_reacquire_dist=ENTRY_REACQUIRE_DIST,
        point_support_radius=POINT_SUPPORT_RADIUS,
        min_separation=MIN_SEPARATION,
        ghost_merge_dist=GHOST_MERGE_DIST,
        duplicate_suppress_dist=DUPLICATE_SUPPRESS_DIST,
        pc_snr_min=PC_SNR_MIN,
        draw_min_near_points=DRAW_MIN_NEAR_POINTS,
        draw_support_radius=DRAW_SUPPORT_RADIUS,
        draw_snr_min=DRAW_SNR_MIN,
        snap_to_cloud_dist=SNAP_TO_CLOUD_DIST,
        draw_hide_frames=DRAW_HIDE_FRAMES,
        cluster_eps=CLUSTER_EPS,
        min_cluster_points=MIN_CLUSTER_POINTS,
        lock_radius=LOCK_RADIUS,
        same_cluster_dist=SAME_CLUSTER_DIST,
        two_people_dist=TWO_PEOPLE_DIST,
        switch_margin=SWITCH_MARGIN,
        switch_frames=SWITCH_FRAMES,
        trail_min_step=TRAIL_MIN_STEP,
        trail_max_step=TRAIL_MAX_STEP,
        boundary_box=bb,
        sensor_height=float(SENSOR_HEIGHT if sensor_height is None else sensor_height),
        tilt_deg=float(TILT_DEG if tilt_deg is None else tilt_deg),
    )


# =============================================================================
# Cumulative visit counter (Total Detected)
# =============================================================================
class VisitCounter:
    """Count detection EVENTS (visits), not unique tids forever.

    Rising-edge +1 when a NEW visit is confirmed. A visit stays OPEN through
    brief tracker drops (fast walking). A new tid that reappears near a
    recently-lost open visit REUSES that visit (no second +1).

    Entry-edge lock: a visit that first appears in the yellow-wall entry band
    does not commit +1 until the person moves inward (or ENTRY_COMMIT_TIME
    elapses). Nearby reappears stitch/reacquire into that same visit so a
    walk-in from the boundary cannot mint two Total Detected increments.

    Only after the visit has been fully closed for LEFT_TIME / ENTRY_LEFT_TIME
    (and the person is gone) will a later return of the same / nearby person
    count again.

    [v5.3] NEVER merge/stitch two tids that are BOTH currently confirmed
    (missed==0 this frame). Only stitch a NEW tid onto a visit whose linked
    tids are all GONE/unconfirmed, OR distance < MERGE_DIST and the target
    visit is inactive/gone. _merge_slots only clusters below MERGE_DIST.
    """

    def __init__(self,
                 confirm_frames=CONFIRM_FRAMES,
                 left_frames=LEFT_FRAMES,
                 left_time=LEFT_TIME,
                 reacquire_dist=REACQUIRE_DIST,
                 reacquire_time=REACQUIRE_TIME,
                 merge_dist=MERGE_DIST,
                 entry_margin=ENTRY_MARGIN,
                 entry_commit_dist=ENTRY_COMMIT_DIST,
                 entry_commit_time=ENTRY_COMMIT_TIME,
                 entry_stitch_dist=ENTRY_STITCH_DIST,
                 entry_stitch_time=ENTRY_STITCH_TIME,
                 entry_left_time=ENTRY_LEFT_TIME,
                 entry_reacquire_dist=ENTRY_REACQUIRE_DIST,
                 min_near_points=MIN_NEAR_POINTS,
                 point_support_radius=POINT_SUPPORT_RADIUS,
                 min_separation=MIN_SEPARATION,
                 ghost_merge_dist=GHOST_MERGE_DIST,
                 duplicate_suppress_dist=DUPLICATE_SUPPRESS_DIST,
                 pc_snr_min=PC_SNR_MIN):
        self.confirm_frames = int(confirm_frames)
        self.left_frames = int(left_frames)
        self.left_time = float(left_time)
        self.reacquire_dist = float(reacquire_dist)
        self.reacquire_time = float(reacquire_time)
        self.merge_dist = float(merge_dist)
        self.entry_margin = float(entry_margin)
        self.entry_commit_dist = float(entry_commit_dist)
        self.entry_commit_time = float(entry_commit_time)
        self.entry_stitch_dist = float(entry_stitch_dist)
        self.entry_stitch_time = float(entry_stitch_time)
        self.entry_left_time = float(entry_left_time)
        self.entry_reacquire_dist = float(entry_reacquire_dist)
        self.min_near_points = int(min_near_points)
        self.point_support_radius = float(point_support_radius)
        self.min_separation = float(min_separation)
        self.ghost_merge_dist = float(ghost_merge_dist)
        self.duplicate_suppress_dist = float(duplicate_suppress_dist)
        self.pc_snr_min = float(pc_snr_min)
        self.draw_min_near_points = int(DRAW_MIN_NEAR_POINTS)
        self.draw_support_radius = float(DRAW_SUPPORT_RADIUS)
        self.draw_snr_min = float(DRAW_SNR_MIN)
        self.total = 0
        # tid -> per-track confirm/gone streaks (for this tid only)
        self._tid_state = {}
        # Open visits: list of dicts
        #   {visit_id, tids:set, first_pos, first_seen, last_pos, last_seen,
        #    counted, closed, active, confirm_streak, is_entry}
        self._visits = []
        self._next_visit_id = 1
        # [v5.3] per-snapshot drop counters for status bar
        self.drop_merge = 0
        self.drop_stitch = 0
        self.drop_gate = 0
        self.drop_unsup = 0
        self.last_raw_tids = 0
        self.last_confirmed = 0
        # tids confirmed (missed==0 + gates) this frame — set in update()
        self._confirmed_this_frame = set()

    def reset(self):
        self.total = 0
        self._tid_state.clear()
        self._visits.clear()
        self._next_visit_id = 1
        self.drop_merge = 0
        self.drop_stitch = 0
        self.drop_gate = 0
        self.drop_unsup = 0
        self.last_raw_tids = 0
        self.last_confirmed = 0
        self._confirmed_this_frame = set()

    def apply_params(self, snap):
        """[v5.3] Pull live-tunable association params from ParamSnapshot."""
        if snap is None:
            return
        self.confirm_frames = int(snap.confirm_frames)
        self.left_time = float(snap.left_time)
        self.entry_left_time = float(snap.entry_left_time)
        self.merge_dist = float(snap.merge_dist)
        self.min_near_points = int(snap.min_near_points)
        self.reacquire_dist = float(getattr(snap, "reacquire_dist", self.reacquire_dist))
        self.entry_stitch_dist = float(getattr(snap, "entry_stitch_dist", self.entry_stitch_dist))
        self.entry_reacquire_dist = float(
            getattr(snap, "entry_reacquire_dist", self.entry_reacquire_dist))
        self.point_support_radius = float(
            getattr(snap, "point_support_radius", self.point_support_radius))
        self.min_separation = float(
            getattr(snap, "min_separation", self.min_separation))
        self.ghost_merge_dist = float(
            getattr(snap, "ghost_merge_dist",
                    getattr(snap, "duplicate_suppress_dist", self.ghost_merge_dist)))
        self.duplicate_suppress_dist = float(
            getattr(snap, "duplicate_suppress_dist", self.ghost_merge_dist))
        self.pc_snr_min = float(getattr(snap, "pc_snr_min", self.pc_snr_min))
        self.draw_min_near_points = int(
            getattr(snap, "draw_min_near_points", getattr(self, "draw_min_near_points", DRAW_MIN_NEAR_POINTS)))
        self.draw_support_radius = float(
            getattr(snap, "draw_support_radius", getattr(self, "draw_support_radius", DRAW_SUPPORT_RADIUS)))
        self.draw_snr_min = float(
            getattr(snap, "draw_snr_min", getattr(self, "draw_snr_min", DRAW_SNR_MIN)))
        self.draw_hide_frames = int(
            getattr(snap, "draw_hide_frames",
                    getattr(self, "draw_hide_frames", DRAW_HIDE_FRAMES)))

    def _tid_st(self, tid):
        st = self._tid_state.get(tid)
        if st is None:
            st = {"confirmed_streak": 0, "gone_streak": 0, "visit_id": None,
                  "fail_support_streak": 0}
            self._tid_state[tid] = st
        return st

    def is_gate_ok(self, st, boundary_box, cloud=None, require_points=True):
        """CONFIRMED pose gates (+ optional point-cloud support)."""
        try:
            if st is None:
                return False
            if int(st.get("missed", 0) or 0) != 0:
                return False
            pos = st.get("pos")
            if pos is None or len(pos) != 3:
                return False
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            if not all(math.isfinite(v) for v in (x, y, z)):
                return False
            if track_near_sensor_xy(x, y):
                return False
            if not room_z_plausible(z):
                return False
            if not in_boundary_box(x, y, z, bb=boundary_box):
                return False
            if require_points:
                near, _sum, med, _c = near_point_stats(
                    cloud, x, y, radius=self.point_support_radius)
                if near < self.min_near_points:
                    return False
                if med < float(self.pc_snr_min):
                    return False
            return True
        except Exception:
            return False

    def _close_timeout(self, visit, bb):
        """How long an inactive visit stays open before full close."""
        if visit.get("is_entry"):
            if not visit.get("counted"):
                return self.entry_left_time
            lp = visit.get("last_pos")
            if lp and in_entry_band(lp[0], lp[1], bb, margin=self.entry_margin):
                return self.entry_left_time
        return self.left_time

    def _reacquire_radius(self, visit):
        if visit.get("is_entry"):
            return self.entry_reacquire_dist
        return self.reacquire_dist

    def _visit_has_confirmed_member(self, visit, exclude_tid=None):
        """True if visit already has a different confirmed tid this frame."""
        confirmed = self._confirmed_this_frame
        for tid in visit.get("tids", ()):
            if exclude_tid is not None and tid == exclude_tid:
                continue
            if tid in confirmed:
                return True
        return False

    def _visit_members_all_gone(self, visit):
        """True if every linked tid is GONE / unconfirmed this frame."""
        tids = visit.get("tids") or set()
        if not tids:
            return True
        confirmed = self._confirmed_this_frame
        for tid in tids:
            if tid in confirmed:
                return False
        return True

    def _find_merge_visit(self, x, y, candidate_tid=None, candidate_confirmed=True):
        """Open visit within MERGE_DIST — never glue two live confirmed people.

        Only merge onto a visit if:
          - its members are all GONE/unconfirmed, OR
          - distance < MERGE_DIST AND the visit is inactive/gone,
        and never when candidate is confirmed AND visit already has another
        confirmed member this frame.
        """
        best = None
        best_d = None
        for v in self._visits:
            if v.get("closed"):
                continue
            if candidate_confirmed and self._visit_has_confirmed_member(
                    v, exclude_tid=candidate_tid):
                log.debug(
                    "merge SKIP visit=%s: already has confirmed member "
                    "(candidate tid=%s also confirmed)",
                    v.get("visit_id"), candidate_tid)
                continue
            # Only stitch onto inactive/gone visits, or visits whose members
            # are all unconfirmed this frame (never two live confirmed tids).
            if not ((not v.get("active")) or self._visit_members_all_gone(v)):
                continue
            lp = v.get("last_pos")
            if not lp:
                continue
            d = math.hypot(x - lp[0], y - lp[1])
            if d < self.merge_dist and (best_d is None or d < best_d):
                best, best_d = v, d
        return best

    def _find_entry_stitch_visit(self, x, y, now, candidate_tid=None,
                                 candidate_confirmed=True):
        """Stitch a NEW tid onto a recent entry visit (GONE/unconfirmed only)."""
        best = None
        best_d = None
        for v in self._visits:
            if v.get("closed"):
                continue
            if not v.get("is_entry"):
                continue
            # [v5.3] only stitch onto GONE/unconfirmed entry visits
            if not self._visit_members_all_gone(v):
                log.debug(
                    "entry_stitch SKIP visit=%s: has confirmed members "
                    "(candidate tid=%s)", v.get("visit_id"), candidate_tid)
                continue
            if candidate_confirmed and self._visit_has_confirmed_member(
                    v, exclude_tid=candidate_tid):
                continue
            age = now - float(v.get("last_seen", 0.0))
            if age > self.entry_stitch_time:
                continue
            lp = v.get("last_pos") or v.get("first_pos")
            if not lp:
                continue
            d = math.hypot(x - lp[0], y - lp[1])
            if d <= self.entry_stitch_dist and (best_d is None or d < best_d):
                best, best_d = v, d
        return best

    def _find_reacquire_visit(self, x, y, now, bb=None, candidate_tid=None,
                              candidate_confirmed=True):
        """Open (or recently lost) visit within reacquire dist/time.
        Entry visits use ENTRY_REACQUIRE_DIST. Skip visits with other
        confirmed members when the candidate is also confirmed.
        """
        best = None
        best_d = None
        for v in self._visits:
            if v.get("closed"):
                continue
            if candidate_confirmed and self._visit_has_confirmed_member(
                    v, exclude_tid=candidate_tid):
                log.debug(
                    "reacquire SKIP visit=%s: has confirmed member "
                    "(candidate tid=%s)", v.get("visit_id"), candidate_tid)
                continue
            # Prefer gone/inactive visits for reacquire gluing
            if v.get("active") and not self._visit_members_all_gone(v):
                continue
            age = now - float(v.get("last_seen", 0.0))
            timeout = self._close_timeout(v, bb) if bb is not None else (
                self.entry_left_time if v.get("is_entry") else self.left_time)
            if age > self.reacquire_time and not v.get("active"):
                if age > timeout:
                    continue
            lp = v.get("last_pos")
            if not lp:
                continue
            rdist = self._reacquire_radius(v)
            d = math.hypot(x - lp[0], y - lp[1])
            if d <= rdist and (best_d is None or d < best_d):
                best, best_d = v, d
        return best

    def _merge_slots(self, confirmed_list):
        """Cluster simultaneous confirmed tracks within MERGE_DIST only.

        Returns list of representatives: (tid, x, y, members).
        Three people 0.6–1.0 m apart with MERGE_DIST=0.45 must yield K=3.
        """
        slots = []
        merge_absorbs = 0
        for tid, x, y in confirmed_list:
            placed = False
            for slot in slots:
                if math.hypot(x - slot["x"], y - slot["y"]) < self.merge_dist:
                    # NEVER merge two currently-confirmed tids that are both
                    # live unless they are truly < MERGE_DIST (already checked).
                    # Distance gate is the only cluster rule.
                    slot["members"].append(tid)
                    merge_absorbs += 1
                    n = len(slot["members"])
                    slot["x"] = (slot["x"] * (n - 1) + x) / n
                    slot["y"] = (slot["y"] * (n - 1) + y) / n
                    log.debug(
                        "merge cluster: tid=%s into slot tid=%s d<%.2f members=%s",
                        tid, slot["tid"], self.merge_dist, slot["members"])
                    placed = True
                    break
            if not placed:
                slots.append({"tid": tid, "x": x, "y": y, "members": [tid]})
        self.drop_merge += merge_absorbs
        return slots

    def _try_commit(self, visit, x, y, now, bb):
        """Rising-edge +1 once per visit after debounce (+ entry lock)."""
        if visit.get("counted"):
            return
        if int(visit.get("confirm_streak", 0)) < self.confirm_frames:
            return
        if visit.get("is_entry"):
            prog = inward_progress(x, y, visit.get("first_pos"), bb)
            age = now - float(visit.get("first_seen", now))
            if prog < self.entry_commit_dist and age < self.entry_commit_time:
                return
        self.total += 1
        visit["counted"] = True
        log.debug(
            "VISIT +1 id=%s entry=%s tids=%s total=%d",
            visit["visit_id"], bool(visit.get("is_entry")),
            sorted(visit["tids"]), self.total)

    def _new_visit(self, x, y, now, bb):
        is_entry = in_entry_band(x, y, bb, margin=self.entry_margin)
        visit = {
            "visit_id": self._next_visit_id,
            "tids": set(),
            "first_pos": (x, y),
            "first_seen": now,
            "last_pos": (x, y),
            "last_seen": now,
            "counted": False,
            "closed": False,
            "active": True,
            "confirm_streak": 0,
            "is_entry": is_entry,
        }
        self._next_visit_id += 1
        self._visits.append(visit)
        log.debug(
            "VISIT open id=%s entry=%s at (%.2f,%.2f)",
            visit["visit_id"], is_entry, x, y)
        return visit

    def update(self, tracks, boundary_box, cloud=None, now=None, cloud_frame=None,
               locked_boxes=None):
        """Process one snapshot. Returns (total_N, in_room_K).

        [v6.7] When ``locked_boxes`` (from assign_boxes_to_clusters) is given,
        the COUNT path runs on the locked cluster-boxes instead of raw
        firmware tids: each box id is stable while it stays on its cluster,
        so a tid whose Kalman pose jumps between clusters cannot open a new
        visit (no spurious +1).  ``In room now`` = number of locked boxes.
        ``Total Detected`` is still a rising edge per visit after
        CONFIRM_FRAMES (DRAW is fast, COUNT waits).
        """
        import time as _time
        now = float(now if now is not None else _time.time())
        raw_tids = len(tracks or {})
        use_boxes = locked_boxes is not None
        if use_boxes:
            # Only confirmed (non-pending) boxes take part in counting.
            tracks = {bid: b for bid, b in (locked_boxes or {}).items()
                      if not b.get("pending")}
        tracks = tracks or {}
        bb = boundary_box
        # [v6.1] Ghost / point-support MUST use current-frame cloud, never accum.
        cloud_frame = cloud_frame if cloud_frame is not None else cloud

        # per-frame drop counters
        self.drop_merge = 0
        self.drop_stitch = 0
        self.drop_gate = 0
        self.drop_unsup = 0
        self.last_raw_tids = raw_tids

        # --- build confirmed set (must pass gates + point support) ---
        confirmed = []  # (tid, x, y)
        candidates = []  # for suppress_split_ghosts
        for tid, st in tracks.items():
            missed = int(st.get("missed", 0) or 0)
            # pose-only check for gate/unsup accounting
            pose_ok = False
            try:
                if missed == 0:
                    pos = st.get("pos")
                    if pos is not None and len(pos) == 3:
                        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                        pose_ok = (
                            all(math.isfinite(v) for v in (x, y, z))
                            and not track_near_sensor_xy(x, y)
                            and room_z_plausible(z)
                            and in_boundary_box(x, y, z, bb=boundary_box))
            except Exception:
                pose_ok = False

            if missed != 0:
                continue
            if not pose_ok:
                self.drop_gate += 1
                log.debug("gate reject tid=%s", tid)
                continue
            pos = st["pos"]
            if use_boxes and st.get("locked"):
                # [v6.7/v6.9] Box is ON its cluster: support = the cluster itself.
                # Do NOT drop locked boxes on PC_SNR_MIN / MIN_NEAR — the
                # cluster already validated the body; COUNT needs these for
                # Total Detected rising-edge.
                near = int(st.get("near", 0) or 0)
                snr_median = float(st.get("snr_median", 0.0) or 0.0)
                snr_sum = near * snr_median
                centroid = (float(pos[0]), float(pos[1]))
                locked_support_ok = True
            else:
                near, snr_sum, snr_median, centroid = near_point_stats(
                    cloud_frame, pos[0], pos[1],
                    radius=self.point_support_radius)
                locked_support_ok = False
            tst = self._tid_st(tid)
            # [v6.4] Filter 2 — current-frame support + median SNR + fail_support_streak
            # [v6.9] Locked cluster boxes skip the near/snr gate.
            support_ok = locked_support_ok or (
                near >= self.min_near_points
                and snr_median >= float(self.pc_snr_min))
            if not support_ok:
                tst["fail_support_streak"] = int(
                    tst.get("fail_support_streak", 0)) + 1
                self.drop_unsup += 1
                log.debug(
                    "ghost drop tid=%s reason=no_frame_support near=%d "
                    "med_snr=%.1f need_near=%d need_snr=%.1f streak=%d",
                    tid, near, snr_median, self.min_near_points,
                    self.pc_snr_min, tst["fail_support_streak"])
                continue
            tst["fail_support_streak"] = 0
            candidates.append({
                "tid": tid,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "near": near,
                "snr": snr_sum,
                "snr_median": snr_median,
                "centroid": centroid if centroid is not None else (float(pos[0]), float(pos[1])),
            })

        # [v6.8] When locked_boxes are provided, boxes are already unique by
        # cluster (incl. firmware-tid 1011 clusters). Skipping suppress /
        # merge avoids under-counting two nearby locked people.
        if use_boxes:
            survivors = {c["tid"] for c in candidates}
        else:
            survivors = suppress_split_ghosts(
                candidates,
                min_near=self.min_near_points,
                support_radius=self.point_support_radius,
                min_separation=self.min_separation,
                ghost_merge_dist=self.ghost_merge_dist,
                pc_snr_min=self.pc_snr_min)
        for c in candidates:
            if c["tid"] not in survivors:
                self.drop_unsup += 1
                continue
            # Hide if fail_support_streak previously hit threshold (safety)
            if int(self._tid_st(c["tid"]).get("fail_support_streak", 0)) >= UNSUPPORTED_HIDE_FRAMES:
                continue
            confirmed.append((c["tid"], c["x"], c["y"]))

        self._confirmed_this_frame = {t for t, _, _ in confirmed}
        self.last_confirmed = len(confirmed)

        # COUNT path: visit +1 only uses confirmed (strict gates + CONFIRM_FRAMES)
        if use_boxes:
            slots = [{"tid": tid, "x": x, "y": y, "members": [tid]}
                     for tid, x, y in confirmed]
        else:
            slots = self._merge_slots(confirmed)

        # [v6.5] DRAW / occupancy path: looser gates so In room now tracks the
        # visible purple body even before CONFIRM_FRAMES / strict SNR.
        # [v6.7] With locked boxes this whole path is skipped: K is simply the
        # number of boxes currently locked onto a cluster inside the box.
        draw_confirmed = []
        draw_cands = []
        d_near = int(getattr(self, "draw_min_near_points", DRAW_MIN_NEAR_POINTS))
        d_rad = float(getattr(self, "draw_support_radius", DRAW_SUPPORT_RADIUS))
        d_snr = float(getattr(self, "draw_snr_min", DRAW_SNR_MIN))
        n_clusters_hint = None
        if use_boxes:
            n_clusters_hint = sum(1 for b in tracks.values() if b.get("locked"))
        for tid, st in ({} if use_boxes else tracks).items():
            try:
                missed = int(st.get("missed", 0) or 0)
                pos = st.get("pos")
                if pos is None or len(pos) != 3:
                    continue
                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                if not all(math.isfinite(v) for v in (x, y, z)):
                    continue
                if track_near_sensor_xy(x, y):
                    continue
                if not room_z_plausible(z):
                    continue
                if not in_boundary_box(x, y, z, bb=boundary_box):
                    continue
                # Allow brief coast (missed>0) for occupancy if cloud still supports
                near, snr_sum, snr_median, centroid, ok = associate_cloud_cluster(
                    cloud_frame, x, y, d_rad, d_near, d_snr,
                    bb=boundary_box, max_assoc_dist=2.5,
                    n_clusters=n_clusters_hint)
                if not ok and missed != 0:
                    continue
                if not ok:
                    continue
                cx, cy = (centroid if centroid is not None else (x, y))
                draw_cands.append({
                    "tid": tid, "x": float(cx), "y": float(cy),
                    "near": near, "snr": snr_sum, "snr_median": snr_median,
                    "centroid": (float(cx), float(cy)),
                })
            except Exception:
                continue
        draw_surv = suppress_split_ghosts(
            draw_cands,
            min_near=d_near,
            support_radius=d_rad,
            min_separation=self.min_separation,
            ghost_merge_dist=self.ghost_merge_dist,
            pc_snr_min=d_snr)
        for c in draw_cands:
            if c["tid"] in draw_surv:
                draw_confirmed.append((c["tid"], c["x"], c["y"]))
        if use_boxes:
            # [v6.7/v6.9] In room now K = non-pending boxes that are locked OR
            # still holding (missed < hide, painted) with a finite pose inside
            # the yellow volume.  MainWindow may further override from
            # drawable_ids after paint.
            hide_n = int(getattr(self, "draw_hide_frames", DRAW_HIDE_FRAMES))
            in_room = 0
            for st in tracks.values():
                try:
                    pos = st.get("pos")
                    if pos is None or len(pos) < 3:
                        continue
                    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                    if not in_boundary_box(x, y, z, bb=boundary_box):
                        continue
                    missed = int(st.get("missed", 0) or 0)
                    if st.get("locked") or (missed > 0 and missed < hide_n):
                        in_room += 1
                except Exception:
                    continue
        else:
            draw_slots = self._merge_slots(draw_confirmed)
            # In room now = DRAW occupancy (not waiting on CONFIRM_FRAMES)
            in_room = len(draw_slots) if draw_slots else len(slots)
        confirmed_tids = set()
        for slot in slots:
            confirmed_tids.update(slot["members"])

        # mark visits inactive this frame; reactivate via slots
        for v in self._visits:
            v["active"] = False

        for slot in slots:
            tid, x, y = slot["tid"], slot["x"], slot["y"]
            # Update streaks for all members
            for mtid in slot["members"]:
                tst = self._tid_st(mtid)
                tst["gone_streak"] = 0
                tst["confirmed_streak"] = int(tst.get("confirmed_streak", 0)) + 1

            # Linking order:
            # existing tid visit_id -> merge -> entry stitch -> reacquire -> new
            visit = None
            link_reason = None
            for mtid in slot["members"]:
                vid = self._tid_st(mtid).get("visit_id")
                if vid is not None:
                    for v in self._visits:
                        if v["visit_id"] == vid and not v.get("closed"):
                            # If this visit already has a *different* confirmed
                            # member and our slot brings another confirmed tid
                            # that isn't already in the visit, do not join.
                            foreign = False
                            for other in slot["members"]:
                                if other in v.get("tids", ()):
                                    continue
                                if (other in self._confirmed_this_frame
                                        and self._visit_has_confirmed_member(
                                            v, exclude_tid=other)):
                                    foreign = True
                                    log.debug(
                                        "reject join: confirmed tid=%s onto "
                                        "visit=%s that already has confirmed",
                                        other, v["visit_id"])
                                    break
                            if not foreign:
                                visit = v
                                link_reason = "existing"
                            break
                if visit is not None:
                    break

            cand_tid = tid
            cand_confirmed = True

            if visit is None:
                visit = self._find_merge_visit(
                    x, y, candidate_tid=cand_tid,
                    candidate_confirmed=cand_confirmed)
                if visit is not None:
                    link_reason = "merge"
                    log.debug(
                        "stitch/merge: tid=%s -> visit=%s via MERGE_DIST=%.2f",
                        cand_tid, visit["visit_id"], self.merge_dist)

            if visit is None:
                visit = self._find_entry_stitch_visit(
                    x, y, now, candidate_tid=cand_tid,
                    candidate_confirmed=cand_confirmed)
                if visit is not None:
                    link_reason = "entry_stitch"
                    self.drop_stitch += 1
                    log.debug(
                        "entry_stitch: tid=%s -> visit=%s d<=%.2f",
                        cand_tid, visit["visit_id"], self.entry_stitch_dist)

            if visit is None:
                visit = self._find_reacquire_visit(
                    x, y, now, bb=bb, candidate_tid=cand_tid,
                    candidate_confirmed=cand_confirmed)
                if visit is not None:
                    link_reason = "reacquire"
                    self.drop_stitch += 1
                    log.debug(
                        "reacquire: tid=%s -> visit=%s",
                        cand_tid, visit["visit_id"])

            if visit is None:
                # Last chance: inactive/gone visit within MERGE_DIST only
                near = self._find_merge_visit(
                    x, y, candidate_tid=cand_tid,
                    candidate_confirmed=cand_confirmed)
                if near is not None and (
                        (not near.get("active"))
                        or self._visit_members_all_gone(near)):
                    visit = near
                    link_reason = "merge_inactive"
                    log.debug(
                        "merge inactive: tid=%s -> visit=%s",
                        cand_tid, visit["visit_id"])
                else:
                    visit = self._new_visit(x, y, now, bb)
                    link_reason = "new"

            visit["active"] = True
            visit["closed"] = False
            visit["last_pos"] = (x, y)
            visit["last_seen"] = now
            visit["tids"].update(slot["members"])
            visit["confirm_streak"] = int(visit.get("confirm_streak", 0)) + 1
            if visit.get("first_pos") is None:
                visit["first_pos"] = (x, y)
                visit["first_seen"] = now
            for mtid in slot["members"]:
                self._tid_st(mtid)["visit_id"] = visit["visit_id"]

            log.debug(
                "visit link tid=%s visit=%s reason=%s",
                cand_tid, visit.get("visit_id"), link_reason)
            self._try_commit(visit, x, y, now, bb)

        # --- tids / visits that are not confirmed this frame ---
        all_tids = set(tracks.keys()) | set(self._tid_state.keys())
        for tid in list(all_tids):
            if tid in confirmed_tids:
                continue
            tst = self._tid_st(tid)
            tst["confirmed_streak"] = 0
            in_snapshot = tid in tracks
            missed = int(tracks.get(tid, {}).get("missed", 0) or 0) if in_snapshot else self.left_frames + 1
            if (not in_snapshot) or missed > self.left_frames:
                tst["gone_streak"] = self.left_frames
            else:
                tst["gone_streak"] = int(tst.get("gone_streak", 0)) + 1
            # keep last_pos on linked visit from track if available
            if in_snapshot:
                pos = tracks[tid].get("pos")
                vid = tst.get("visit_id")
                if pos is not None and vid is not None:
                    for v in self._visits:
                        if v["visit_id"] == vid and not v.get("active"):
                            try:
                                v["last_pos"] = (float(pos[0]), float(pos[1]))
                            except Exception:
                                pass

        # Close visits that have been inactive longer than LEFT_TIME /
        # ENTRY_LEFT_TIME (entry, uncounted / still in band).
        still = []
        for v in self._visits:
            if v.get("active"):
                still.append(v)
                continue
            age = now - float(v.get("last_seen", now))
            timeout = self._close_timeout(v, bb)
            if age >= timeout:
                v["closed"] = True
                log.debug("VISIT closed id=%s after %.2fs (timeout=%.2f)",
                          v["visit_id"], age, timeout)
                for tid in list(v.get("tids", ())):
                    tst = self._tid_state.get(tid)
                    if tst and tst.get("visit_id") == v["visit_id"]:
                        tst["visit_id"] = None
            else:
                still.append(v)
        self._visits = still

        # prune orphan tid state
        for tid in list(self._tid_state.keys()):
            if tid not in tracks and self._tid_state[tid].get("visit_id") is None:
                if self._tid_state[tid].get("gone_streak", 0) >= self.left_frames:
                    self._tid_state.pop(tid, None)

        return self.total, in_room


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("mmWave Radar Visualizer v6.11 - PyQtGraph/OpenGL")
        self.resize(1680, 900)

        # ---- [v5.3] thread-safe live params ---------------------------------
        self._param_lock = threading.Lock()
        self.params = default_param_snapshot(
            boundary_box=args.boundary_box or BOUNDARY_BOX_ROOM,
            sensor_height=args.sensor_height,
            tilt_deg=args.tilt)
        self._default_params = self.params.copy()

        # ---- [CUMULATIVE VISIT COUNTER] --------------------------------------
        # Primary: Total Detected N = cumulative visit events (never decreases
        # when someone leaves; same tid returning after GONE increments again).
        # Secondary: In room now K = currently CONFIRMED tracks this frame.
        self.visit_counter = VisitCounter(
            confirm_frames=CONFIRM_FRAMES, left_frames=LEFT_FRAMES,
            left_time=LEFT_TIME, reacquire_dist=REACQUIRE_DIST,
            reacquire_time=REACQUIRE_TIME, merge_dist=MERGE_DIST,
            entry_margin=ENTRY_MARGIN,
            entry_commit_dist=ENTRY_COMMIT_DIST,
            entry_commit_time=ENTRY_COMMIT_TIME,
            entry_stitch_dist=ENTRY_STITCH_DIST,
            entry_stitch_time=ENTRY_STITCH_TIME,
            entry_left_time=ENTRY_LEFT_TIME,
            entry_reacquire_dist=ENTRY_REACQUIRE_DIST,
            min_near_points=MIN_NEAR_POINTS,
            point_support_radius=POINT_SUPPORT_RADIUS,
            min_separation=MIN_SEPARATION,
            ghost_merge_dist=GHOST_MERGE_DIST,
            duplicate_suppress_dist=DUPLICATE_SUPPRESS_DIST,
            pc_snr_min=PC_SNR_MIN)
        self._unsup_streak = {}  # tid -> consecutive frames failing point support
        # [v5.3] GUI-side drop counters (mirrored from VisitCounter each frame)
        self.drop_merge = 0
        self.drop_stitch = 0
        self.drop_gate = 0
        self.drop_unsup = 0

        toolbar = QtWidgets.QToolBar("Detections")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setIconSize(QtCore.QSize(0, 0))
        try:
            _top = QtCore.Qt.ToolBarArea.TopToolBarArea
        except AttributeError:
            _top = QtCore.Qt.TopToolBarArea
        self.addToolBar(_top, toolbar)

        self.total_label = QtWidgets.QLabel("Total Detected: 0")
        self.total_label.setStyleSheet(
            "QLabel { font-size: 28px; font-weight: 700; padding: 6px 16px; "
            "color: #111; background: #f5f5f5; border: 1px solid #ccc; "
            "border-radius: 6px; }")
        toolbar.addWidget(self.total_label)

        self.inroom_label = QtWidgets.QLabel("In room now: 0")
        self.inroom_label.setStyleSheet(
            "QLabel { font-size: 16px; font-weight: 600; padding: 4px 12px; "
            "color: #333; margin-left: 8px; }")
        toolbar.addWidget(self.inroom_label)

        toolbar.addSeparator()
        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.reset_btn.setToolTip("Reset Total Detected to 0 and clear visit state")
        self.reset_btn.clicked.connect(self.on_reset_counter)
        toolbar.addWidget(self.reset_btn)

        # Back-compat alias so older patches looking for people_label don't break.
        self.people_label = self.total_label

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
        self.boundary_box = dict(args.boundary_box or BOUNDARY_BOX_ROOM)
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

        # [v5.3] wrap 3D|2D views; put live param panel on the right
        views = QtWidgets.QWidget()
        views_layout = QtWidgets.QHBoxLayout(views)
        views_layout.setContentsMargins(0, 0, 0, 0)
        views_layout.addWidget(self.gl_view, stretch=3)
        views_layout.addWidget(self.plot_2d, stretch=2)
        layout.addWidget(views, stretch=5)

        self._spin_boxes = {}
        param_panel = self._build_param_panel()
        layout.addWidget(param_panel, stretch=2)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Starting...")

        self.artists_3d = {}
        self.artists_2d = {}
        # [v6.6] coalesce cross-thread snapshots to latest-only
        self._pending_snapshot = None
        self._snapshot_timer = QtCore.QTimer(self)
        self._snapshot_timer.setInterval(16)  # ~60 Hz drain; drops extras
        self._snapshot_timer.timeout.connect(self._drain_snapshot)
        self._snapshot_timer.start()

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
        # Push initial params into the parser thread
        self.thread.apply_params(self._params_copy())
        self.thread.start()

    # -------------------------------------------------------------------
    # [v5.3] Live parameter panel + thread-safe snapshot plumbing
    # -------------------------------------------------------------------
    def _params_copy(self):
        with self._param_lock:
            return self.params.copy()

    def _update_param(self, name, value, *, rebuild_box=False, rebuild_geom=False):
        """Immediate update for Kalman / box / geometry spins."""
        with self._param_lock:
            if name.startswith("bb_"):
                key = name[3:]
                bb = dict(self.params.boundary_box)
                bb[key] = float(value)
                self.params = replace(self.params, boundary_box=bb)
            else:
                kwargs = {name: value}
                self.params = replace(self.params, **kwargs)
            snap = self.params.copy()
        # Push to VisitCounter (GUI thread) and ParserThread
        try:
            self.visit_counter.apply_params(snap)
        except Exception as e:
            log.debug("VisitCounter.apply_params failed: %s", e)
        try:
            self.thread.apply_params(snap)
        except Exception as e:
            log.debug("ParserThread.apply_params failed: %s", e)
        if rebuild_box or name.startswith("bb_"):
            self._rebuild_boundary_box(snap)
        if rebuild_geom or name in ("sensor_height", "tilt_deg"):
            self._rebuild_sensor_marker(snap)

    def _rebuild_boundary_box(self, snap=None):
        snap = snap or self._params_copy()
        self.boundary_box = dict(snap.boundary_box)
        try:
            xform = make_height_only_transform(snap.sensor_height)
            pos = boundary_box_line_vertices(self.boundary_box, xform)
            self.boundary_box_item.setData(pos=pos)
        except Exception as e:
            log.debug("boundary box rebuild failed: %s", e)

    def _rebuild_sensor_marker(self, snap=None):
        snap = snap or self._params_copy()
        try:
            sensor_pos = (0, 0, float(snap.sensor_height))
            self.sensor_item.setData(pos=cube_line_vertices(*sensor_pos, size=0.3))
        except Exception as e:
            log.debug("sensor marker rebuild failed: %s", e)

    def _make_dspin(self, name, value, lo, hi, step=0.05, decimals=2):
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(float(lo), float(hi))
        box.setSingleStep(float(step))
        box.setDecimals(int(decimals))
        box.setValue(float(value))
        box.setKeyboardTracking(False)
        box.valueChanged.connect(
            lambda v, n=name: self._update_param(n, float(v)))
        self._spin_boxes[name] = box
        return box

    def _make_ispin(self, name, value, lo, hi, step=1):
        box = QtWidgets.QSpinBox()
        box.setRange(int(lo), int(hi))
        box.setSingleStep(int(step))
        box.setValue(int(value))
        box.setKeyboardTracking(False)
        box.valueChanged.connect(
            lambda v, n=name: self._update_param(n, int(v)))
        self._spin_boxes[name] = box
        return box

    def _add_spin_row(self, form, label, widget):
        form.addRow(label, widget)

    def _build_param_panel(self):
        """Right-side QScrollArea with live Kalman / box / geometry spins."""
        snap = self._params_copy()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(360)
        host = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(host)

        title = QtWidgets.QLabel("Live Parameters (v6.11)")
        title.setStyleSheet("QLabel { font-weight: 700; font-size: 14px; }")
        vbox.addWidget(title)

        # A. Kalman / tracker
        box_a = QtWidgets.QGroupBox("A. Kalman / tracker")
        form_a = QtWidgets.QFormLayout(box_a)
        self._add_spin_row(form_a, "MEASURE_STD",
            self._make_dspin("measure_std", snap.measure_std, 0.01, 2.0, 0.01))
        self._add_spin_row(form_a, "PROCESS_STD",
            self._make_dspin("process_std", snap.process_std, 0.01, 10.0, 0.05))
        self._add_spin_row(form_a, "GATE_SIGMA",
            self._make_dspin("gate_sigma", snap.gate_sigma, 0.5, 20.0, 0.1))
        self._add_spin_row(form_a, "GATE_MIN_M",
            self._make_dspin("gate_min_m", snap.gate_min_m, 0.05, 5.0, 0.05))
        self._add_spin_row(form_a, "MAX_MISSED",
            self._make_ispin("max_missed", snap.max_missed, 1, 60))
        self._add_spin_row(form_a, "CONFIRM_FRAMES",
            self._make_ispin("confirm_frames", snap.confirm_frames, 1, 60))
        self._add_spin_row(form_a, "LEFT_TIME",
            self._make_dspin("left_time", snap.left_time, 0.1, 30.0, 0.1))
        self._add_spin_row(form_a, "ENTRY_LEFT_TIME",
            self._make_dspin("entry_left_time", snap.entry_left_time, 0.1, 30.0, 0.1))
        self._add_spin_row(form_a, "MERGE_DIST",
            self._make_dspin("merge_dist", snap.merge_dist, 0.05, 3.0, 0.05))
        self._add_spin_row(form_a, "MIN_NEAR_POINTS",
            self._make_ispin("min_near_points", snap.min_near_points, 0, 20))
        self._add_spin_row(form_a, "POINT_SUPPORT_RADIUS",
            self._make_dspin("point_support_radius", snap.point_support_radius,
                             0.1, 3.0, 0.05))
        self._add_spin_row(form_a, "PC_SNR_MIN",
            self._make_dspin("pc_snr_min", snap.pc_snr_min, 0.0, 40.0, 0.5))
        self._add_spin_row(form_a, "GHOST_MERGE_DIST",
            self._make_dspin("ghost_merge_dist", snap.ghost_merge_dist,
                             0.2, 5.0, 0.05))
        self._add_spin_row(form_a, "MIN_SEPARATION",
            self._make_dspin("min_separation", snap.min_separation,
                             0.2, 5.0, 0.05))
        self._add_spin_row(form_a, "DRAW_MIN_NEAR_POINTS",
            self._make_ispin("draw_min_near_points",
                             getattr(snap, "draw_min_near_points", DRAW_MIN_NEAR_POINTS),
                             0, 20))
        self._add_spin_row(form_a, "DRAW_SUPPORT_RADIUS",
            self._make_dspin("draw_support_radius",
                             getattr(snap, "draw_support_radius", DRAW_SUPPORT_RADIUS),
                             0.1, 5.0, 0.05))
        self._add_spin_row(form_a, "DRAW_SNR_MIN",
            self._make_dspin("draw_snr_min",
                             getattr(snap, "draw_snr_min", DRAW_SNR_MIN),
                             0.0, 40.0, 0.5))
        vbox.addWidget(box_a)

        # B. Boundary box
        box_b = QtWidgets.QGroupBox("B. Boundary box (room m)")
        form_b = QtWidgets.QFormLayout(box_b)
        bb = snap.boundary_box
        for key, lo, hi in (
            ("minX", -20.0, 20.0), ("maxX", -20.0, 20.0),
            ("minY", -5.0, 30.0), ("maxY", -5.0, 30.0),
            ("minZ", -1.0, 5.0), ("maxZ", -1.0, 5.0),
        ):
            spin = self._make_dspin(f"bb_{key}", bb[key], lo, hi, 0.1)
            # Force rebuild on boundary change
            spin.valueChanged.disconnect()
            spin.valueChanged.connect(
                lambda v, n=f"bb_{key}": self._update_param(
                    n, float(v), rebuild_box=True))
            self._add_spin_row(form_b, key, spin)
        vbox.addWidget(box_b)

        # C. Sensor geometry
        box_c = QtWidgets.QGroupBox("C. Sensor geometry")
        form_c = QtWidgets.QFormLayout(box_c)
        hspin = self._make_dspin("sensor_height", snap.sensor_height, 0.5, 5.0, 0.05)
        hspin.valueChanged.disconnect()
        hspin.valueChanged.connect(
            lambda v: self._update_param(
                "sensor_height", float(v), rebuild_box=True, rebuild_geom=True))
        self._add_spin_row(form_c, "SENSOR_HEIGHT", hspin)
        tspin = self._make_dspin("tilt_deg", snap.tilt_deg, -45.0, 45.0, 0.5, decimals=1)
        tspin.valueChanged.disconnect()
        tspin.valueChanged.connect(
            lambda v: self._update_param(
                "tilt_deg", float(v), rebuild_geom=True))
        self._add_spin_row(form_c, "TILT_DEG", tspin)
        vbox.addWidget(box_c)

        # D. [v6.7/v6.11] Cluster-first box lock + trail hysteresis
        box_d = QtWidgets.QGroupBox("D. Cluster lock / trail (v6.11)")
        form_d = QtWidgets.QFormLayout(box_d)
        self._add_spin_row(form_d, "CLUSTER_EPS",
            self._make_dspin("cluster_eps",
                             getattr(snap, "cluster_eps", CLUSTER_EPS), 0.1, 3.0, 0.05))
        self._add_spin_row(form_d, "MIN_CLUSTER_POINTS",
            self._make_ispin("min_cluster_points",
                             getattr(snap, "min_cluster_points", MIN_CLUSTER_POINTS), 1, 50))
        self._add_spin_row(form_d, "LOCK_RADIUS",
            self._make_dspin("lock_radius",
                             getattr(snap, "lock_radius", LOCK_RADIUS), 0.2, 3.0, 0.05))
        self._add_spin_row(form_d, "SAME_CLUSTER_DIST",
            self._make_dspin("same_cluster_dist",
                             getattr(snap, "same_cluster_dist", SAME_CLUSTER_DIST), 0.1, 2.0, 0.05))
        self._add_spin_row(form_d, "TWO_PEOPLE_DIST",
            self._make_dspin("two_people_dist",
                             getattr(snap, "two_people_dist", TWO_PEOPLE_DIST), 0.2, 3.0, 0.05))
        self._add_spin_row(form_d, "DRAW_HIDE_FRAMES",
            self._make_ispin("draw_hide_frames",
                             getattr(snap, "draw_hide_frames", DRAW_HIDE_FRAMES), 1, 60))
        self._add_spin_row(form_d, "SWITCH_MARGIN",
            self._make_dspin("switch_margin",
                             getattr(snap, "switch_margin", SWITCH_MARGIN), 0.0, 2.0, 0.05))
        self._add_spin_row(form_d, "SWITCH_FRAMES",
            self._make_ispin("switch_frames",
                             getattr(snap, "switch_frames", SWITCH_FRAMES), 1, 30))
        self._add_spin_row(form_d, "TRAIL_MIN_STEP",
            self._make_dspin("trail_min_step",
                             getattr(snap, "trail_min_step", TRAIL_MIN_STEP), 0.0, 1.0, 0.01))
        self._add_spin_row(form_d, "TRAIL_MAX_STEP",
            self._make_dspin("trail_max_step",
                             getattr(snap, "trail_max_step", TRAIL_MAX_STEP), 0.1, 5.0, 0.05))
        vbox.addWidget(box_d)

        reset_params_btn = QtWidgets.QPushButton("Reset params")
        reset_params_btn.setToolTip("Restore code defaults (does NOT reset Total Detected)")
        reset_params_btn.clicked.connect(self.on_reset_params)
        vbox.addWidget(reset_params_btn)
        vbox.addStretch(1)

        scroll.setWidget(host)
        return scroll

    def on_reset_params(self):
        """Restore code-default ParamSnapshot into spins + threads."""
        try:
            defaults = default_param_snapshot(
                boundary_box=BOUNDARY_BOX_ROOM,
                sensor_height=SENSOR_HEIGHT,
                tilt_deg=TILT_DEG)
            # Prefer original CLI defaults captured at startup when available
            if hasattr(self, "_default_params") and self._default_params is not None:
                defaults = self._default_params.copy()
            with self._param_lock:
                self.params = defaults.copy()
                snap = self.params.copy()
            # Silence spin signals while restoring
            for name, box in self._spin_boxes.items():
                box.blockSignals(True)
                try:
                    if name.startswith("bb_"):
                        box.setValue(float(snap.boundary_box[name[3:]]))
                    else:
                        val = getattr(snap, name)
                        box.setValue(float(val) if isinstance(box, QtWidgets.QDoubleSpinBox)
                                     else int(val))
                finally:
                    box.blockSignals(False)
            self.visit_counter.apply_params(snap)
            self.thread.apply_params(snap)
            self._rebuild_boundary_box(snap)
            self._rebuild_sensor_marker(snap)
            self.status_bar.showMessage("Parameters reset to defaults.")
            log.info("Live parameters reset to defaults.")
        except Exception as e:
            log.debug("Reset params failed: %s", e)

    # -------------------------------------------------------------------
    def on_status(self, msg):
        log.info(msg)
        self.status_bar.showMessage(msg)

    def on_error(self, msg):
        log.error(msg)
        self.status_bar.showMessage("ERROR: " + msg)

    def on_snapshot(self, snapshot):
        # [v6.6] Latest-only: parser may emit ~18 Hz; keep one pending slot so
        # the GUI queue cannot grow without bound when render is slower.
        self._pending_snapshot = snapshot

    def _drain_snapshot(self):
        snap = self._pending_snapshot
        if snap is None:
            return
        self._pending_snapshot = None
        try:
            self._on_snapshot_inner(snap)
        except Exception as e:
            log.error("on_snapshot failed unexpectedly (frame skipped): %s", e)

    def _on_snapshot_inner(self, snapshot):
        cloud = snapshot.get("cloud", np.empty((0, 4)))
        # [v6.1] current-frame cloud for ghost / near_n tests (never accum)
        cloud_frame = snapshot.get("cloud_frame", None)
        if cloud_frame is None:
            cloud_frame = np.empty((0, 4))
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
        # [v6.8] Keep point_tids 1:1 with cloud_frame through the same masks.
        point_tids = snapshot.get("point_tids", None)
        if point_tids is None:
            point_tids = np.empty(0, dtype=int)
        else:
            point_tids = np.asarray(point_tids, dtype=int).ravel()

        if len(cloud_frame) > 0:
            try:
                finite_mask = np.all(np.isfinite(cloud_frame), axis=1)
                if len(point_tids) == len(cloud_frame):
                    point_tids = point_tids[finite_mask]
                cloud_frame = cloud_frame[finite_mask]
                bb_gui = getattr(self, "boundary_box", None) or BOUNDARY_BOX_ROOM
                margin = CLOUD_Z_MARGIN
                x0 = float(bb_gui["minX"]) - margin
                x1 = float(bb_gui["maxX"]) + margin
                y0 = float(bb_gui["minY"]) - margin
                y1 = float(bb_gui["maxY"]) + margin
                z0 = float(bb_gui["minZ"]) - margin
                z1 = float(bb_gui["maxZ"]) + margin
                x, y, z = cloud_frame[:, 0], cloud_frame[:, 1], cloud_frame[:, 2]
                vol = (np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                       & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
                       & (z >= z0) & (z <= z1))
                if len(point_tids) == len(cloud_frame):
                    point_tids = point_tids[vol]
                cloud_frame = cloud_frame[vol]
            except Exception as e:
                log.debug("cloud_frame filter failed: %s", e)

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
        # [v6.7] CLUSTER FIRST, THEN ASSIGN BOXES.
        #   1. cluster_cloud_xy(): greedy flood (CLUSTER_EPS) of the CURRENT
        #      frame cloud inside the yellow box, density-split of chained
        #      bodies, same-body merge < TWO_PEOPLE_DIST.
        #   2. assign_boxes_to_clusters(): 1-to-1 lock.  A firmware tid or an
        #      existing box may attach ONLY to the nearest unmatched cluster
        #      within LOCK_RADIUS; N clusters -> N boxes, each on its cluster
        #      centroid.  Unmatched clusters get a local id (1000+).  A box
        #      with no cluster holds its last pose DRAW_HIDE_FRAMES, then drops.
        #      Two ids on the same cluster (< SAME_CLUSTER_DIST) -> keep the
        #      stronger.  There is NO global strongest-blob fallback anymore.
        #   DRAW vs COUNT: boxes appear at once (local boxes after 2 frames);
        #   VisitCounter still waits CONFIRM_FRAMES before Total Detected += 1.
        drawable_ids = set()
        bb = getattr(self, "boundary_box", None)
        if not hasattr(self, "_boxes"):
            self._boxes = {}            # box_id -> locked box state (last frame)
            self._next_local_id = LOCAL_BOX_ID_BASE

        with self._param_lock:
            draw_snr = float(getattr(
                self.params, "draw_snr_min", DRAW_SNR_MIN))
            hide_n = int(getattr(
                self.params, "draw_hide_frames", DRAW_HIDE_FRAMES))
            eps = float(getattr(self.params, "cluster_eps", CLUSTER_EPS))
            min_cpts = int(getattr(self.params, "min_cluster_points", MIN_CLUSTER_POINTS))
            lock_r = float(getattr(self.params, "lock_radius", LOCK_RADIUS))
            same_d = float(getattr(self.params, "same_cluster_dist", SAME_CLUSTER_DIST))
            two_d = float(getattr(self.params, "two_people_dist", TWO_PEOPLE_DIST))
            sw_margin = float(getattr(self.params, "switch_margin", SWITCH_MARGIN))
            sw_frames = int(getattr(self.params, "switch_frames", SWITCH_FRAMES))
            t_min = float(getattr(self.params, "trail_min_step", TRAIL_MIN_STEP))
            t_max = float(getattr(self.params, "trail_max_step", TRAIL_MAX_STEP))

        # [v6.8] Prefer firmware TLV-1011 point-to-track clustering when
        # association is present; fall back to XY flood when absent/empty.
        try:
            clusters = []
            if (point_tids is not None
                    and len(point_tids) == len(cloud_frame)
                    and np.any(~np.isin(point_tids, list(TRACK_INDEX_UNASSOC)))):
                clusters = cluster_cloud_by_track_index(
                    cloud_frame, point_tids, bb=bb,
                    min_pts=min_cpts, snr_min=draw_snr, two_people_dist=two_d)
            if not clusters:
                clusters = cluster_cloud_xy(
                    cloud_frame, bb=bb, eps=eps, min_pts=min_cpts,
                    snr_min=draw_snr, merge_dist=two_d)
        except Exception as e:
            log.debug("cluster (1011/xy) failed: %s", e)
            clusters = []
        try:
            boxes, self._next_local_id = assign_boxes_to_clusters(
                tracks, clusters, self._boxes,
                lock_radius=lock_r, hide_frames=hide_n,
                same_cluster_dist=same_d,
                next_local_id=self._next_local_id,
                switch_margin=sw_margin, switch_frames=sw_frames,
                trail_min_step=t_min, trail_max_step=t_max,
                min_cluster_points=min_cpts, two_people_dist=two_d)
        except Exception as e:
            log.debug("assign_boxes_to_clusters failed: %s", e)
            boxes = {}
        self._boxes = boxes
        self.last_clusters = len(clusters)
        if not hasattr(self, "_trail_gen"):
            self._trail_gen = {}

        def _paint(tid, draw_pos, st, hist_override=None):
            drawable_ids.add(tid)
            if tid not in self.artists_3d:
                qcolor = track_qcolor(tid)
                gl_color = _qcolor_to_gl_rgba(qcolor, alpha=1.0)
                self.artists_3d[tid] = TrackArtists3D(self.gl_view, gl_color)
                self.artists_2d[tid] = TrackArtists2D(self.plot_2d, qcolor)
            h = clamp_person_height(st.get("height", PERSON_HEIGHT))
            # [v6.9] Use filtered history from the box; do NOT re-append raw pose.
            if hist_override is not None:
                hist = list(hist_override)
            else:
                hist = list(st.get("history", []) or [])
            self.artists_3d[tid].update(draw_pos, hist, height=h)
            self.artists_2d[tid].update(draw_pos, hist)

        for bid, b in boxes.items():
            if b.get("pending"):
                continue  # new local box: wait one more frame on its cluster
            try:
                draw_pos = b.get("draw_pos") or b.get("pos")
                if draw_pos is None or len(draw_pos) < 3:
                    continue
                # [v6.9] Only paint boxes inside the yellow volume so K matches
                # what the operator sees (no box while In room now: 0).
                if not in_boundary_box(
                        float(draw_pos[0]), float(draw_pos[1]), float(draw_pos[2]),
                        bb=bb):
                    continue
                gen = int(b.get("trail_gen", 0) or 0)
                prev_gen = self._trail_gen.get(bid)
                self._trail_gen[bid] = gen
                # On trail_gen change: clear artist trail once, then draw new hist.
                if prev_gen is not None and prev_gen != gen:
                    _paint(bid, draw_pos, b, hist_override=[])
                _paint(bid, draw_pos, b)
            except Exception as e:
                log.debug("Failed to render box %s: %s", bid, e)

        try:
            for tid in list(self.artists_3d.keys()):
                if tid in drawable_ids:
                    continue
                self.artists_3d[tid].remove(self.gl_view)
                self.artists_2d[tid].remove(self.plot_2d)
                del self.artists_3d[tid]
                del self.artists_2d[tid]
                if hasattr(self, "_trail_gen"):
                    self._trail_gen.pop(tid, None)
        except Exception as e:
            log.debug("Track artist cleanup failed: %s", e)

        # ---- cumulative visit counter (reacquire-aware rising edge) -----
        try:
            # Keep VisitCounter in sync with live params each snapshot
            self.visit_counter.apply_params(self._params_copy())
            total_n, in_room = self.visit_counter.update(
                tracks, bb, cloud=cloud, cloud_frame=cloud_frame, now=time.time(),
                locked_boxes=boxes)
            # [v6.9] Sync In room now with painted drawable boxes inside volume.
            in_room = len(drawable_ids)
            self.total_label.setText(f"Total Detected: {total_n}")
            self.inroom_label.setText(f"In room now: {in_room}")
            self.drop_merge = int(getattr(self.visit_counter, "drop_merge", 0))
            self.drop_stitch = int(getattr(self.visit_counter, "drop_stitch", 0))
            self.drop_gate = int(getattr(self.visit_counter, "drop_gate", 0))
            self.drop_unsup = int(getattr(self.visit_counter, "drop_unsup", 0))
            raw_tids = int(getattr(self.visit_counter, "last_raw_tids", len(tracks)))
            confirmed_c = int(getattr(self.visit_counter, "last_confirmed", 0))
        except Exception as e:
            log.debug("Visit counter update failed: %s", e)
            total_n = getattr(self.visit_counter, "total", 0)
            in_room = len(drawable_ids)
            raw_tids = len(tracks)
            confirmed_c = len(drawable_ids)

        try:
            tlv1011 = "yes" if snapshot.get("tlv1011") else "no"
            ptid_assoc = int(snapshot.get("ptid_assoc", 0) or 0)
            self.status_bar.showMessage(
                f"Frame {snapshot.get('frame_count', 0)} | in-room {in_room} | "
                f"total {getattr(self.visit_counter, 'total', total_n)} | "
                f"clusters {getattr(self, 'last_clusters', 0)} | boxes {len(drawable_ids)} | "
                f"1011={tlv1011} ptid_assoc={ptid_assoc} | "
                f"raw_tids {raw_tids} | confirmed {confirmed_c} | "
                f"drop:merge={self.drop_merge} stitch={self.drop_stitch} "
                f"gate={self.drop_gate} unsup={self.drop_unsup}")
        except Exception:
            pass

    # -------------------------------------------------------------------
    def on_reset_counter(self):
        """Reset button: Total Detected -> 0; clear per-tid confirmed/gone state."""
        try:
            self.visit_counter.reset()
            self._unsup_streak = {}
            self._last_draw = {}
            self._boxes = {}
            self._next_local_id = LOCAL_BOX_ID_BASE
            self._trail_gen = {}
            self.total_label.setText("Total Detected: 0")
            self.inroom_label.setText("In room now: 0")
            self.status_bar.showMessage("Counter reset.")
            log.info("Visit counter reset.")
        except Exception as e:
            log.debug("Reset counter failed: %s", e)

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