"""
main.py
-------
Real-time object detection + tracking.

Usage:
    python main.py                          # webcam
    python main.py --source path/to/video.mp4
    python main.py --source 0 --weights yolov8s.pt --conf 0.5
    python main.py --no-trails --no-line

Keyboard controls while running:
    q / ESC  -> quit
    s        -> save a snapshot (PNG) of the current frame to output/
    r        -> toggle video recording (MP4) to output/
    t        -> toggle motion trails on/off
    l        -> toggle the counting line on/off
"""

import argparse
import os
import time
from datetime import datetime

import cv2

import config
from detector import Detector
from tracker import SortTracker
from utils import FPSCounter, LineCounter, draw_track, draw_hud


def parse_args():
    p = argparse.ArgumentParser(description="Real-time object detection and tracking")
    p.add_argument("--source", default=config.DEFAULT_SOURCE,
                    help="Webcam index (e.g. 0) or path/URL to a video file")
    p.add_argument("--weights", default=config.MODEL_WEIGHTS, help="YOLO weights file")
    p.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD, help="Detection confidence threshold")
    p.add_argument("--classes", type=int, nargs="*", default=config.CLASSES_TO_DETECT,
                    help="Restrict detection to these COCO class IDs, e.g. --classes 0 2 3")
    p.add_argument("--no-trails", action="store_true", help="Disable motion trails")
    p.add_argument("--no-line", action="store_true", help="Disable the counting line")
    p.add_argument("--save-video", action="store_true", help="Start recording immediately")
    return p.parse_args()


def open_source(source):
    # allow numeric strings ("0") to mean a webcam index
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def main():
    args = parse_args()
    config.SHOW_TRAILS = not args.no_trails
    config.COUNT_LINE_ENABLED = not args.no_line

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("Loading model...")
    detector = Detector(weights=args.weights, conf=args.conf, classes=args.classes)
    tracker = SortTracker()
    line_counter = LineCounter(config.COUNT_LINE_P1, config.COUNT_LINE_P2) if config.COUNT_LINE_ENABLED else None
    fps_counter = FPSCounter()

    cap = open_source(args.source)
    writer = None
    recording = args.save_video

    print("Running. Press 'q' to quit, 's' snapshot, 'r' record, 't' trails, 'l' line.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("End of stream / cannot read frame.")
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections)

            if line_counter is not None and config.COUNT_LINE_ENABLED:
                line_counter.update(tracks, frame.shape)
                line_counter.draw(frame)

            for t in tracks:
                draw_track(frame, t)

            fps = fps_counter.tick()
            draw_hud(frame, fps, tracks, recording=recording)

            if recording:
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*config.RECORD_FOURCC)
                    out_path = os.path.join(config.OUTPUT_DIR, f"recording_{int(time.time())}.mp4")
                    writer = cv2.VideoWriter(out_path, fourcc, 20.0, (w, h))
                    print(f"Recording -> {out_path}")
                writer.write(frame)

            cv2.imshow("Object Detection & Tracking", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # q or ESC
                break
            elif key == ord("s"):
                fname = os.path.join(config.OUTPUT_DIR, f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.png")
                cv2.imwrite(fname, frame)
                print(f"Saved snapshot -> {fname}")
            elif key == ord("r"):
                recording = not recording
                if not recording and writer is not None:
                    writer.release()
                    writer = None
                    print("Recording stopped.")
            elif key == ord("t"):
                config.SHOW_TRAILS = not config.SHOW_TRAILS
            elif key == ord("l"):
                config.COUNT_LINE_ENABLED = not config.COUNT_LINE_ENABLED

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
