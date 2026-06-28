"""
tracker.py — ByteTrack Student Tracking Module
===============================================
WHY THIS FILE EXISTS:
  Detection alone gives us "there is a person at position X in THIS frame."
  Tracking adds: "that person is STUDENT #3, and they were also at position Y
  in the LAST 50 frames." This persistent identity is what makes the suspicion
  score meaningful over time.

HOW IT CONNECTS:
  main.py → creates Tracker() → calls tracker.track(frame)
           → passes the result to utils.py BehaviourAnalyser

KEY CONCEPTS EXPLAINED:
  ─────────────────────────────────────────────────────────────────────
  DETECTION vs TRACKING — What's the difference?

    Detection  → "I see a person at (100,200)–(300,400) in THIS frame."
                 Every frame is independent. No memory.
    Tracking   → "The person at (100,200) in frame 1 is the SAME as
                 the person at (105,195) in frame 2 → they are Student #3."
                 Tracking builds memory across time.

  WHY IS TRACKING NEEDED?
    • To compute "Student #3 has been looking away for 4 seconds" you must
      know which detection in each frame corresponds to Student #3.
    • Without tracking, you can't tell students apart across frames.
    • Tracking also handles occlusion: if a student briefly hides behind
      a pillar, the tracker predicts where they should be and re-links
      the detection when they reappear.

  WHAT IS ByteTrack?
    ByteTrack (2022) is a high-performance multi-object tracker that works
    by associating detections to existing tracks using IoU matching.
    Key innovation: it uses BOTH high-confidence AND low-confidence
    detections (the "bytes") to maintain tracks through occlusion.

    For example:
      Frame 10: Student #3 detected with conf=0.8 → track maintained.
      Frame 11: Student #3 partially hidden, conf=0.25 → ByteTrack still
                keeps the track alive using this low-confidence detection.
      Frame 12: Student #3 fully visible, conf=0.9 → track seamlessly
                continues as Student #3.

  TRACK ID:
    A unique integer assigned to each student by the tracker.
    Student #1 → track_id = 1
    Student #2 → track_id = 2
    etc.
    IDs are persistent for the session (persist=True).

  RE-IDENTIFICATION:
    If a student leaves the frame and returns, ByteTrack tries to give them
    back their original ID. This isn't perfect (they might get a new ID),
    but it works well in a controlled exam room setting.

  persist=True:
    Tells Ultralytics to keep the same ByteTrack state object across
    consecutive calls to model.track(). Without this, every frame resets
    the tracker and students get new IDs every frame — useless!
  ─────────────────────────────────────────────────────────────────────
"""

import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Optional
import config


@dataclass
class TrackedStudent:
    """
    Represents one tracked student in the current frame.

    This is a plain data container — no logic here.
    BehaviourAnalyser in utils.py reads these to compute suspicion.

    Fields
    ------
    track_id    : int   — Unique ID assigned by ByteTrack (1, 2, 3, ...)
    bbox        : tuple — (x1, y1, x2, y2) pixel coordinates
    confidence  : float — YOLO detection confidence (0.0–1.0)
    label       : str   — "Student #<id>"
    cx, cy      : int   — Centre pixel (for distance calculations)
    """
    track_id   : int
    bbox       : tuple          # (x1, y1, x2, y2)
    confidence : float
    label      : str
    cx         : int            # Centre x pixel
    cy         : int            # Centre y pixel
    has_phone  : bool = False   # Will be set by BehaviourAnalyser


@dataclass
class TrackedPhone:
    """Represents a detected cell phone in the current frame."""
    bbox       : tuple
    confidence : float
    cx         : int
    cy         : int


class Tracker:
    """
    Wraps YOLO11n + ByteTrack for multi-student tracking.

    WHY wrap the detector here too?
      model.track() calls detection AND tracking in one step.
      Separating them would require re-implementing Ultralytics internals.
      So Tracker owns its own YOLO model instance (separate from Detector)
      specifically for the track() call.

    Usage:
        tracker   = Tracker()
        students, phones = tracker.track(frame)
        for student in students:
            print(student.track_id, student.bbox)
    """

    def __init__(self):
        print(f"[Tracker] Loading model for tracking: {config.MODEL_PATH}")
        self.model = YOLO(config.MODEL_PATH)

        # Warm up
        dummy = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
        self.model.track(
            dummy,
            persist=config.PERSIST,
            tracker=config.TRACKER_CONFIG,
            classes=config.CLASSES_OF_INTEREST,
            verbose=False,
        )
        print("[Tracker] ByteTrack initialised.")

    def track(self, frame: np.ndarray):
        """
        Run ByteTrack multi-object tracking on one frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from OpenCV.

        Returns
        -------
        students : list[TrackedStudent]
            All person detections with assigned track IDs.
        phones   : list[TrackedPhone]
            All cell phone detections (phones are not tracked by ID —
            they don't need persistent IDs, we just need to know IF
            a phone is visible near a student).
        """
        # persist=True: reuse ByteTrack state from the previous frame.
        # This is what makes tracking "remember" across frames.
        results = self.model.track(
            frame,
            persist=config.PERSIST,
            tracker=config.TRACKER_CONFIG,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            imgsz=config.IMG_SIZE,
            device=config.DEVICE,
            classes=config.CLASSES_OF_INTEREST,
            verbose=False,
        )

        students = []
        phones   = []

        if results[0].boxes is None:
            return students, phones

        boxes = results[0].boxes

        for i in range(len(boxes)):
            # Extract raw values from the YOLO result tensor
            cls_id     = int(boxes.cls[i].item())
            conf_val   = float(boxes.conf[i].item())
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if cls_id == config.CLASS_PERSON:
                # Get tracking ID — may be None if tracker lost the track
                tid = None
                if boxes.id is not None:
                    tid = int(boxes.id[i].item())

                if tid is None:
                    # Skip untracked detections (happens on very first frame)
                    continue

                student = TrackedStudent(
                    track_id   = tid,
                    bbox       = (x1, y1, x2, y2),
                    confidence = conf_val,
                    label      = f"Student #{tid}",
                    cx         = cx,
                    cy         = cy,
                )
                students.append(student)

            elif cls_id == config.CLASS_CELL_PHONE:
                phone = TrackedPhone(
                    bbox       = (x1, y1, x2, y2),
                    confidence = conf_val,
                    cx         = cx,
                    cy         = cy,
                )
                phones.append(phone)

        return students, phones
