"""
detector.py — YOLO11 Object Detection Engine
============================================
WHY THIS FILE EXISTS:
  Wraps the Ultralytics YOLO model. All raw detection logic lives here.
  Returns clean Python objects (bounding boxes, confidences, class IDs)
  that tracker.py and utils.py can consume without knowing YOLO internals.

HOW IT CONNECTS:
  main.py → creates Detector() → calls detector.detect(frame)
  tracker.py → receives the raw YOLO results and feeds them to ByteTrack

KEY CONCEPTS EXPLAINED:
  ─────────────────────────────────────────────────────────────────────
  WHAT IS YOLO?
    YOLO = You Only Look Once. A neural network that looks at the entire
    image ONCE (unlike older "sliding window" methods) and predicts:
      • Bounding boxes  → WHERE is the object? (x, y, width, height)
      • Confidence      → HOW SURE is the model? (0.0–1.0)
      • Class           → WHAT is the object? (person, phone, etc.)

  BOUNDING BOX:
    A rectangle: (x_min, y_min, x_max, y_max) in pixel coordinates.
    x_min,y_min = top-left corner
    x_max,y_max = bottom-right corner

  CONFIDENCE:
    A probability score. conf=0.9 means the model is 90% sure.
    We filter out detections below CONF_THRESHOLD.

  IoU (Intersection over Union):
    When YOLO produces overlapping boxes for the same object, IoU
    measures how much they overlap (intersection area / union area).
    Boxes with IoU > threshold are merged — this is called NMS.

  NMS (Non-Maximum Suppression):
    Keeps the BEST bounding box when several overlap the same object.
    Without NMS you'd see 5 boxes on one student instead of 1.

  YOLO11n vs other sizes:
    Model      Parameters   Speed (CPU)   Accuracy
    YOLO11n    2.6M         ~45ms/frame   Good
    YOLO11s    9.4M         ~70ms/frame   Better
    YOLO11m    20M          ~120ms/frame  Great
    YOLO11l    25M          ~180ms/frame  Excellent
    → We use 'n' (nano) for real-time CPU performance.
  ─────────────────────────────────────────────────────────────────────
"""

from ultralytics import YOLO
import numpy as np
import config


class Detector:
    """
    Wraps YOLO11n for object detection.

    Usage:
        detector = Detector()
        results  = detector.detect(frame)
        boxes    = results.boxes  # bounding boxes
    """

    def __init__(self):
        """
        Load the YOLO model from disk.
        On first run, Ultralytics auto-downloads yolo11n.pt if not found.
        """
        print(f"[Detector] Loading model: {config.MODEL_PATH}")
        self.model = YOLO(config.MODEL_PATH)

        # Warm up the model with a dummy forward pass.
        # This pre-loads weights into memory so the first real frame
        # doesn't have an unexpected delay.
        dummy = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        print("[Detector] Model warmed up and ready.")

    def detect(self, frame: np.ndarray):
        """
        Run YOLO detection on a single BGR frame (from OpenCV).

        Parameters
        ----------
        frame : np.ndarray
            A single video frame in BGR colour space (H × W × 3).

        Returns
        -------
        results : ultralytics.engine.results.Results
            Raw Ultralytics result object. Caller can access:
              results.boxes.xyxy   → bounding boxes as (x1,y1,x2,y2) tensors
              results.boxes.conf   → confidence scores
              results.boxes.cls    → class IDs
              results.boxes.id     → track IDs (after tracker.track() call)

        WHY return the raw result?
            tracker.py needs the whole result object to pass into
            ByteTrack. Splitting it here would force reconstruction.
        """
        results = self.model(
            frame,
            conf=config.CONF_THRESHOLD,    # Discard detections below threshold
            iou=config.IOU_THRESHOLD,      # NMS overlap threshold
            imgsz=config.IMG_SIZE,         # Resize input to this before inference
            device=config.DEVICE,          # "cpu" or "cuda:0"
            classes=config.CLASSES_OF_INTEREST,  # Only detect person + cell phone
            verbose=False,                 # Suppress per-frame console spam
        )
        return results[0]   # results is a list (one item per image); we send one frame

    def get_class_name(self, class_id: int) -> str:
        """
        Convert a COCO class integer ID to its human-readable name.

        Parameters
        ----------
        class_id : int
            e.g. 0 → "person", 67 → "cell phone"
        """
        return self.model.names.get(int(class_id), "unknown")

    def is_person(self, class_id: int) -> bool:
        """Return True if this detection is a person."""
        return int(class_id) == config.CLASS_PERSON

    def is_phone(self, class_id: int) -> bool:
        """Return True if this detection is a cell phone."""
        return int(class_id) == config.CLASS_CELL_PHONE
