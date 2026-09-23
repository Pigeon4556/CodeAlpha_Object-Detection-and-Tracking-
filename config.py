"""
config.py
---------
Central place for every tunable knob in the project. Nothing else in the
codebase hardcodes a magic number that belongs here — change behavior by
editing this file, not by hunting through the pipeline.
"""

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
MODEL_WEIGHTS = "yolov8n.pt"      # swap for yolov8s.pt / yolov8m.pt for more accuracy
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD_NMS = 0.5           # NMS threshold used internally by YOLO
CLASSES_TO_DETECT = None          # None = all COCO classes, or e.g. [0, 2, 3] for person/car/motorbike

# ---------------------------------------------------------------------------
# Tracking (custom SORT: Kalman filter + Hungarian assignment on IoU)
# ---------------------------------------------------------------------------
MAX_AGE = 30            # frames a track survives with no matching detection
MIN_HITS = 3            # consecutive matches required before a track is "confirmed"
IOU_MATCH_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
TRAIL_LENGTH = 30        # how many past centroids to draw per track (motion trail)
BOX_THICKNESS = 2
FONT_SCALE = 0.55
SHOW_TRAILS = True
SHOW_SPEED = True
SHOW_FPS = True
SHOW_CLASS_SUMMARY = True

# Counting line: two normalized points (fraction of frame width/height),
# so it scales to any video resolution automatically.
COUNT_LINE_ENABLED = True
COUNT_LINE_P1 = (0.0, 0.6)
COUNT_LINE_P2 = (1.0, 0.6)

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
DEFAULT_SOURCE = 0        # 0 = default webcam; or a path/URL to a video file
OUTPUT_DIR = "output"
RECORD_FOURCC = "mp4v"
