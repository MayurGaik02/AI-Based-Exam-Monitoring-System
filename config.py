"""
config.py — Central Configuration for AI Exam Monitoring System
================================================================
WHY THIS FILE EXISTS:
  All magic numbers, file paths, thresholds, and weights live here.
  Every other module imports from config so you change settings in
  ONE place and the whole system updates. No hardcoded values anywhere.

HOW IT CONNECTS:
  detector.py  → imports MODEL_PATH, CONF_THRESHOLD, IMG_SIZE
  tracker.py   → imports TRACKER_CONFIG, PERSIST
  utils.py     → imports SUSPICION_WEIGHTS, ALERT_THRESHOLD, BEHAVIOUR_*
  main.py      → imports VIDEO_SOURCE, OUTPUT_PATH, DISPLAY_*

ENGINEERING DECISIONS EXPLAINED:
  - YOLO11n (nano): lightest model, runs on CPU, real-time at 640px
  - conf=0.6: balanced — avoids ghost detections and misses
  - 640x640: YOLO's native training resolution → best accuracy
  - ByteTrack: state-of-the-art multi-object tracker, handles occlusion
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "yolo11n.pt")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
LOG_DIR     = os.path.join(OUTPUT_DIR, "logs")
REPORT_DIR  = os.path.join(OUTPUT_DIR, "reports")
FRAMES_DIR  = os.path.join(OUTPUT_DIR, "annotated_frames")

# ─────────────────────────────────────────────
# VIDEO SOURCE
# ─────────────────────────────────────────────
# 0 = default webcam, 1 = secondary camera
# Or pass a string path to a video file, e.g. "videos/exam.mp4"
VIDEO_SOURCE = 0

# ─────────────────────────────────────────────
# YOLO DETECTION SETTINGS
# ─────────────────────────────────────────────
CONF_THRESHOLD = 0.6    # Confidence threshold (0.0 – 1.0)
                         # 0.3 → many detections, more false positives
                         # 0.5 → balanced (recommended for testing)
                         # 0.6 → production quality, fewer false alarms
                         # 0.7 → very strict, may miss partially occluded students
IOU_THRESHOLD  = 0.45   # Intersection-over-Union for NMS (Non-Max Suppression)
                         # If two boxes overlap more than 45%, keep only the best one
IMG_SIZE       = 640    # Input resolution sent to YOLO (pixels)
                         # 640 is YOLO's native training size → highest accuracy
                         # Use 320 for faster but less accurate inference
DEVICE         = "cpu"  # "cpu" for CPU-only laptops; "cuda:0" for GPU

# ─────────────────────────────────────────────
# TRACKER SETTINGS
# ─────────────────────────────────────────────
TRACKER_CONFIG = "bytetrack.yaml"   # ByteTrack config bundled with Ultralytics
PERSIST        = True               # Keep track IDs across consecutive frames
                                    # Without persist=True, each frame re-assigns IDs

# ─────────────────────────────────────────────
# COCO CLASS IDs (YOLO pretrained COCO dataset)
# ─────────────────────────────────────────────
# YOLO COCO has 80 classes. We care about:
CLASS_PERSON     = 0    # COCO class 0 = "person"
CLASS_CELL_PHONE = 67   # COCO class 67 = "cell phone"
CLASSES_OF_INTEREST = [CLASS_PERSON, CLASS_CELL_PHONE]

# ─────────────────────────────────────────────
# SUSPICION SCORE WEIGHTS
# ─────────────────────────────────────────────
# Formula:
#   Score = W_GAZE*looking_away + W_PHONE*phone + W_INTERACT*interaction + W_SEAT*seat
# All weights must sum to 1.0
SUSPICION_WEIGHTS = {
    "looking_away":  0.40,   # Highest weight — most common cheating indicator
    "phone_detected": 0.35,  # Very strong indicator
    "interaction":   0.15,   # Talking/crowding with another student
    "seat_leaving":  0.10,   # Getting up from seat
}

# ─────────────────────────────────────────────
# RISK THRESHOLDS
# ─────────────────────────────────────────────
LOW_RISK_MAX    = 0.30   # 0% – 30%   → Green label: "Low Risk"
MEDIUM_RISK_MAX = 0.60   # 31% – 60%  → Yellow label: "Medium Risk"
HIGH_RISK_MIN   = 0.61   # 61%+       → Red label: "High Risk" + Alert

# ─────────────────────────────────────────────
# BEHAVIOUR DETECTION PARAMETERS
# ─────────────────────────────────────────────
# How many consecutive seconds of looking away triggers suspicion
GAZE_AWAY_SECONDS   = 3.0

# Pixel distance threshold between two student bounding box CENTRES
# If two students' box centres are within this many pixels → "interaction"
INTERACTION_DIST_PX = 120

# If a tracked student disappears from frame for this many seconds → "absent"
ABSENCE_TIMEOUT_SEC = 5.0

# ─────────────────────────────────────────────
# DISPLAY SETTINGS
# ─────────────────────────────────────────────
SHOW_FPS        = True
SHOW_CONF       = True
SHOW_TRACK_ID   = True
SHOW_SUSPICION  = True
WINDOW_NAME     = "AI Exam Monitor"

# Colour palette (BGR format for OpenCV)
COLOUR_LOW      = (0,   200,  0)    # Green  → Low Risk
COLOUR_MEDIUM   = (0,   165, 255)   # Orange → Medium Risk
COLOUR_HIGH     = (0,     0, 255)   # Red    → High Risk
COLOUR_PHONE    = (255,   0, 255)   # Magenta → Phone detected
COLOUR_TEXT     = (255, 255, 255)   # White   → General text
COLOUR_ALERT    = (0,     0, 200)   # Dark red → Alert banner

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
SAVE_LOG_CSV    = True    # Save per-event CSV log
SAVE_REPORT     = True    # Save JSON summary report at end of session
LOG_INTERVAL_S  = 1.0    # Log state of each tracked student every N seconds
