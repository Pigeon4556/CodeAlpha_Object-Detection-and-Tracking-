"""
kalman_filter.py
-----------------
A small, self-contained constant-velocity Kalman filter for tracking a
bounding box represented as [cx, cy, area, aspect_ratio].

Most SORT tutorials just `pip install filterpy` and call it a black box.
Here the filter is implemented directly in numpy so the state model,
process noise, and measurement update are all visible and easy to tune.

State vector (7): [cx, cy, s, r, vcx, vcy, vs]
  cx, cy -> center of the box
  s      -> scale/area of the box
  r      -> aspect ratio (assumed constant, no velocity term needed)
  vcx,vcy,vs -> velocities of the above
"""

import numpy as np


class KalmanBoxTracker:
    """Tracks a single object's bounding box using a constant-velocity model."""

    count = 0  # class-level counter to hand out unique track IDs

    def __init__(self, bbox):
        """
        bbox: [x1, y1, x2, y2]
        """
        # State transition matrix (constant velocity model)
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0

        # Measurement function: we observe [cx, cy, s, r] directly
        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        # Measurement noise covariance
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0

        # Process noise covariance
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

        # State covariance — high initial uncertainty on velocities
        self.P = np.eye(7)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0

        self.x = np.zeros((7, 1))
        self.x[:4, 0] = self._bbox_to_z(bbox).flatten()

        self.time_since_update = 0
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    # -- geometry helpers ----------------------------------------------
    @staticmethod
    def _bbox_to_z(bbox):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        s = w * h
        r = w / float(h) if h != 0 else 0.0
        return np.array([[cx], [cy], [s], [r]])

    @staticmethod
    def _x_to_bbox(x):
        cx, cy, s, r = x[0, 0], x[1, 0], x[2, 0], x[3, 0]
        s = max(s, 1e-6)
        w = np.sqrt(s * r) if r > 0 else 0.0
        h = s / w if w > 0 else 0.0
        return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]

    # -- Kalman steps -----------------------------------------------------
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        bbox = self._x_to_bbox(self.x)
        self.history.append(bbox)
        return bbox

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        z = self._bbox_to_z(bbox)
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R       # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)      # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def get_state(self):
        return self._x_to_bbox(self.x)

    def velocity(self):
        """Returns (vx, vy) in pixels/frame, useful for speed estimation."""
        return float(self.x[4, 0]), float(self.x[5, 0])
