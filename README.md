# Real-Time Object Detection & Tracking

YOLOv8 detection + a **from-scratch SORT tracker** (custom Kalman filter,
no `filterpy` black box) with motion trails, per-object speed estimates,
and a line-crossing in/out counter.

## What makes this build different

- **Kalman filter written by hand** (`kalman_filter.py`) — a constant-velocity
  model over `[cx, cy, area, aspect_ratio]`, so you can see and tune the exact
  process/measurement noise instead of trusting an imported package.
- **Hungarian-algorithm IoU matching** (`tracker.py`) via `scipy.optimize.linear_sum_assignment`.
- **Motion trails** that fade with age, drawn per track ID.
- **Live speed estimate** (px/frame) per object, pulled straight from the
  Kalman filter's velocity state — no extra computation needed.
- **Line-crossing counter** (IN/OUT) — the "how many people/cars crossed
  this line" feature most tutorials skip.
- **Snapshot + recording hotkeys** while the window is open.
- **Deterministic per-ID colors** (hashed, not random) so a track keeps the
  same color across the whole video.

## Project layout

```
object_tracking_project/
├── main.py            # CLI entry point / video loop
├── detector.py         # YOLOv8 wrapper -> plain-dict detections
├── tracker.py           # SORT: predict, IoU-match, update, spawn, prune
├── kalman_filter.py     # custom constant-velocity Kalman filter
├── utils.py             # drawing, trails, line counter, FPS/HUD
├── config.py             # every tunable constant lives here
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads `yolov8n.pt` (~6MB) automatically via `ultralytics`.

## Usage

```bash
# Webcam (default)
python main.py

# A video file
python main.py --source path/to/video.mp4

# Bigger, more accurate model + higher confidence
python main.py --weights yolov8s.pt --conf 0.5

# Only track people (0), cars (2), and motorbikes (3) — COCO class IDs
python main.py --classes 0 2 3

# Disable trails / the counting line
python main.py --no-trails --no-line
```

### Keyboard controls (while the window is focused)

| Key | Action |
|-----|--------|
| `q` / `Esc` | Quit |
| `s` | Save a snapshot (PNG) to `output/` |
| `r` | Toggle recording (MP4) to `output/` |
| `t` | Toggle motion trails |
| `l` | Toggle the counting line |

## Tuning

Everything lives in `config.py`:
- `CONFIDENCE_THRESHOLD` — raise to cut false positives, lower to catch more.
- `MAX_AGE` — how many frames a track survives without a matching detection
  (raise if objects get briefly occluded a lot).
- `MIN_HITS` — how many consecutive matches before a track is "confirmed"
  and drawn (raise to suppress flickery false tracks).
- `IOU_MATCH_THRESHOLD` — how much overlap is required to match a detection
  to an existing track.
- `COUNT_LINE_P1` / `COUNT_LINE_P2` — normalized (0–1) endpoints of the
  counting line, so it scales to any resolution.

## Swapping in Faster R-CNN instead of YOLO

`detector.py` is the only file that talks to the detection model. Replace
its `detect()` method with a Faster R-CNN inference call (e.g. via
`torchvision.models.detection.fasterrcnn_resnet50_fpn`) that returns the
same `{bbox, confidence, class_id, class_name}` dict shape, and every
downstream file (tracker, utils, main) works unchanged.

## How the tracking math works, briefly

1. **Predict**: every existing track's Kalman filter predicts where its box
   should be this frame based on its last known velocity.
2. **Associate**: IoU is computed between all predicted boxes and all new
   YOLO detections; the Hungarian algorithm finds the assignment that
   maximizes total IoU.
3. **Update**: matched tracks get a Kalman update pulling their state toward
   the new detection.
4. **Spawn/prune**: unmatched detections start new tracks; tracks unmatched
   for more than `MAX_AGE` frames are dropped.
