"""
utils.py
--------
Everything that touches pixels: consistent per-ID colors, bounding boxes,
motion trails, the line-crossing counter, and the on-screen HUD (FPS,
per-class counts). Keeping this separate from main.py keeps the render
logic testable and the control loop short.
"""

import time
import hashlib
import cv2
import numpy as np
import config


def id_to_color(track_id):
    """Deterministic, visually distinct BGR color derived from the track ID
    (so the same object keeps the same color for its whole life, and different
    IDs rarely collide in color)."""
    h = hashlib.md5(str(track_id).encode()).hexdigest()
    b = int(h[0:2], 16)
    g = int(h[2:4], 16)
    r = int(h[4:6], 16)
    # push away from very dark colors so boxes stay visible on dark video
    return (max(b, 60), max(g, 60), max(r, 60))


class FPSCounter:
    def __init__(self, smoothing=0.9):
        self.smoothing = smoothing
        self.fps = 0.0
        self._last = time.time()

    def tick(self):
        now = time.time()
        dt = max(now - self._last, 1e-6)
        instant = 1.0 / dt
        self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant if self.fps else instant
        self._last = now
        return self.fps


class LineCounter:
    """Counts tracks that cross a user-defined line, split into in/out
    based on which side of the line the centroid moved to."""

    def __init__(self, p1_norm, p2_norm):
        self.p1_norm = p1_norm
        self.p2_norm = p2_norm
        self.count_in = 0
        self.count_out = 0
        self._sides = {}  # track_id -> last known side (+1/-1)

    def _line_pts(self, frame_shape):
        h, w = frame_shape[:2]
        p1 = (int(self.p1_norm[0] * w), int(self.p1_norm[1] * h))
        p2 = (int(self.p2_norm[0] * w), int(self.p2_norm[1] * h))
        return p1, p2

    @staticmethod
    def _side(p1, p2, pt):
        # sign of the cross product tells which side of the line pt is on
        val = (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])
        return 1 if val > 0 else -1

    def update(self, tracks, frame_shape):
        p1, p2 = self._line_pts(frame_shape)
        for t in tracks:
            side = self._side(p1, p2, t.centroid())
            prev = self._sides.get(t.id)
            if prev is not None and prev != side and not t.counted:
                if side == 1:
                    self.count_in += 1
                else:
                    self.count_out += 1
                t.counted = True
            self._sides[t.id] = side

    def draw(self, frame):
        p1, p2 = self._line_pts(frame.shape)
        cv2.line(frame, p1, p2, (0, 255, 255), 2)
        cv2.putText(frame, f"IN: {self.count_in}  OUT: {self.count_out}",
                    (p1[0] + 10, p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)


def draw_track(frame, track):
    x1, y1, x2, y2 = [int(v) for v in track.bbox()]
    color = id_to_color(track.id)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)

    label = f"ID {track.id} {track.class_name}"
    if config.SHOW_SPEED:
        vx, vy = track.kf.velocity()
        speed = (vx ** 2 + vy ** 2) ** 0.5
        label += f" | {speed:.1f}px/f"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)

    if config.SHOW_TRAILS and len(track.centroid_history) > 1:
        pts = np.array(track.centroid_history, dtype=np.int32)
        for i in range(1, len(pts)):
            # fade the trail: older segments are thinner/dimmer
            alpha = i / len(pts)
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), color, thickness)


def draw_hud(frame, fps, tracks, recording=False):
    h, w = frame.shape[:2]
    if config.SHOW_FPS:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    if config.SHOW_CLASS_SUMMARY:
        counts = {}
        for t in tracks:
            counts[t.class_name] = counts.get(t.class_name, 0) + 1
        summary = " | ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "no objects"
        cv2.putText(frame, summary, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 0), 1, cv2.LINE_AA)

    if recording:
        cv2.circle(frame, (w - 20, 20), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 60, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
