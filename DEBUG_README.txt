ANPR Debug Patch (MVP) - CCTV style tracking / best-shot

1) Replace files in your project with these (minimum):
   - camera_worker.py

   If you are using the older debug patch, you can also keep:
   - main.py / db.py / enhance_worker.py / alpr.py / gpu_enhance.py

2) Run:
   python main.py

3) Logs:
   - File: anpr_debug.log (or config.json: log_path)
   - Console: INFO by default
     Set env: ANPR_CONSOLE_DEBUG=1 to see DEBUG in console.

4) Debug knobs (ENV):
   General:
   - ANPR_DEBUG=1               (more logs in camera worker)
   - ANPR_KEEP_PEND=1           (keep PEND_*.jpg images for inspection)
   - ANPR_DRAW_TRACK=1          (draw green boxes on main camera)

   Commit / stability (recommend production defaults below):
   - ANPR_FOCUS_TIMEOUT=6.0     (max seconds to wait before forcing commit/drop)
   - ANPR_VOTE_NEED=2           (>=2 helps avoid early wrong reads)
   - ANPR_MIN_FINAL_CONF=0.62
   - ANPR_MIN_CAP=3             (min number of snapshots before commit)
   - ANPR_MIN_TRACK_TIME=0.6    (seconds)
   - ANPR_STABLE_HOLD=0.6       (commit only after no improvement for N seconds)

   Tracking (important to avoid "1 xe ghi 2 biển"):
   - ANPR_TRACK_IOU=0.22        (lower if bbox jitter causes re-track)
   - ANPR_DEDUP_IOU=0.65        (dedup 2 bbox in same frame)
   - ANPR_TRACK_MAX_GAP=1.2     (seconds)
   - ANPR_TRACK_LOST=0.9        (seconds)

   Enhance trigger:
   - ANPR_ENHANCE_TRIG=0.75
   - ANPR_SHARP_TRIG=110
   - ANPR_ENHANCE_SUBMIT_IV=0.25
   - ANPR_MAX_PEND=10

Example (Windows PowerShell):
   $env:ANPR_DEBUG="1"
   $env:ANPR_CONSOLE_DEBUG="1"
   $env:ANPR_KEEP_PEND="1"
   $env:ANPR_DRAW_TRACK="1"
   python main.py

Notes:
- New logic will HOLD the same tracked object after COMMIT until it disappears.
  This prevents the scenario "read wrong plate -> later read correct -> insert second log".
- Best snapshot is selected using (OCR conf + sharpness + crop area) across multiple frames.
