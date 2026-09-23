"""
tracker.py
----------
Multi-object tracker (SORT: Simple Online and Realtime Tracking).

Pipeline each frame:
  1. Predict the next position of every existing track (Kalman predict step).
  2. Match new detections to predicted tracks by IoU using the Hungarian
     algorithm (scipy.optimize.linear_sum_assignment).
  3. Update matched tracks with their detection (Kalman update step).
  4. Spawn new tracks for unmatched detections.
  5. Drop tracks that haven't been matched in `max_age` frames.

This is a from-scratch, readable implementation rather than an imported
package, so every step above maps to a clearly-named method below.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from kalman_filter import KalmanBoxTracker
import config


def iou_batch(bb_test, bb_gt):
    """Vectorized IoU between two sets of boxes, shape (N,4) and (M,4)."""
    bb_test = np.expand_dims(bb_test, 1)
    bb_gt = np.expand_dims(bb_gt, 0)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    union = area_test + area_gt - inter

    return inter / np.maximum(union, 1e-6)


class Track:
    """Thin wrapper pairing a KalmanBoxTracker with app-level metadata."""

    def __init__(self, bbox, class_id, class_name):
        self.kf = KalmanBoxTracker(bbox)
        self.class_id = class_id
        self.class_name = class_name
        self.centroid_history = []  # for drawing motion trails
        self.counted = False        # for the line-crossing counter

    @property
    def id(self):
        return self.kf.id

    def predict(self):
        bbox = self.kf.predict()
        return bbox

    def update(self, bbox, class_id, class_name):
        self.kf.update(bbox)
        self.class_id = class_id
        self.class_name = class_name

    def bbox(self):
        return self.kf.get_state()

    def centroid(self):
        x1, y1, x2, y2 = self.bbox()
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def push_centroid(self):
        self.centroid_history.append(self.centroid())
        if len(self.centroid_history) > config.TRAIL_LENGTH:
            self.centroid_history.pop(0)


class SortTracker:
    def __init__(self, max_age=None, min_hits=None, iou_threshold=None):
        self.max_age = max_age or config.MAX_AGE
        self.min_hits = min_hits or config.MIN_HITS
        self.iou_threshold = iou_threshold or config.IOU_MATCH_THRESHOLD
        self.tracks = []

    def _associate(self, detections, predicted_boxes):
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        iou_matrix = iou_batch(detections[:, :4], np.array(predicted_boxes))
        # Hungarian algorithm maximizes assignment by minimizing cost -> use -iou
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matches, unmatched_dets, unmatched_trks = [], [], []
        matched_det_idx, matched_trk_idx = set(), set()

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matches.append((r, c))
                matched_det_idx.add(r)
                matched_trk_idx.add(c)

        unmatched_dets = [i for i in range(len(detections)) if i not in matched_det_idx]
        unmatched_trks = [i for i in range(len(self.tracks)) if i not in matched_trk_idx]
        return matches, unmatched_dets, unmatched_trks

    def update(self, detections):
        """
        detections: list of dicts with keys 'bbox' [x1,y1,x2,y2], 'class_id', 'class_name'
        Returns the list of currently confirmed Track objects.
        """
        predicted_boxes = [t.predict() for t in self.tracks]

        det_array = np.array([d["bbox"] for d in detections]) if detections else np.empty((0, 4))
        matches, unmatched_dets, unmatched_trks = self._associate(det_array, predicted_boxes)

        for det_idx, trk_idx in matches:
            d = detections[det_idx]
            self.tracks[trk_idx].update(d["bbox"], d["class_id"], d["class_name"])

        for det_idx in unmatched_dets:
            d = detections[det_idx]
            self.tracks.append(Track(d["bbox"], d["class_id"], d["class_name"]))

        # drop stale tracks
        alive = []
        for t in self.tracks:
            if t.kf.time_since_update <= self.max_age:
                alive.append(t)
        self.tracks = alive

        confirmed = []
        for t in self.tracks:
            if t.kf.time_since_update == 0:
                t.push_centroid()
            if t.kf.hits >= self.min_hits or t.kf.age <= self.min_hits:
                if t.kf.time_since_update == 0:
                    confirmed.append(t)
        return confirmed
