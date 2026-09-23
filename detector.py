"""
detector.py
-----------
Thin wrapper around Ultralytics YOLOv8 that turns a raw frame into a list
of plain-dict detections the tracker can consume. Keeping this isolated
means swapping in Faster R-CNN, YOLOv5, or a custom model later only
requires rewriting this one file.
"""

from ultralytics import YOLO
import config


class Detector:
    def __init__(self, weights=None, conf=None, classes=None):
        weights = weights or config.MODEL_WEIGHTS
        self.conf = conf or config.CONFIDENCE_THRESHOLD
        self.classes = classes if classes is not None else config.CLASSES_TO_DETECT
        self.model = YOLO(weights)
        self.names = self.model.names  # class_id -> class_name

    def detect(self, frame):
        """
        Runs inference on a single BGR frame.
        Returns: list of dicts {bbox: [x1,y1,x2,y2], confidence, class_id, class_name}
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            classes=self.classes,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": self.names.get(cls_id, str(cls_id)),
            })
        return detections
