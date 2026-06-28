# AI-Based Exam Monitoring System

> Real-time exam hall surveillance using YOLO11n + ByteTrack on CPU

---

## Overview

This system monitors students during an examination using a webcam or CCTV
video feed and automatically detects suspicious behaviour in real time.

**Model:** YOLO11 Nano (`yolo11n.pt`) — pretrained on COCO dataset  
**Tracker:** ByteTrack (multi-object tracking with occlusion handling)  
**Language:** Python 3.10+  
**Hardware:** CPU-only laptop supported  

---

## Features

| Feature | Detection Method |
|---------|----------------|
| Student detection | YOLO11n person class |
| Cell phone detection | YOLO11n cell phone class |
| Student tracking across frames | ByteTrack (unique Student #ID) |
| Gaze / head turn detection | Bounding box position heuristic |
| Interaction detection | Euclidean distance between students |
| Seat absence detection | Track disappearance timeout |
| Suspicion scoring | Weighted formula (0–100%) |
| Alert generation | Rule-based threshold triggers |
| Per-session CSV log | Automatic |
| JSON report | Saved on exit |

---

## Suspicion Score Formula

```
Score = 0.40 × (Looking Away)
      + 0.35 × (Phone Detected)
      + 0.15 × (Interacting with Student)
      + 0.10 × (Left Seat / Absent)
```

| Score | Risk Level |
|-------|-----------|
| 0–30% | 🟢 Low Risk |
| 31–60% | 🟡 Medium Risk |
| 61%+ | 🔴 High Risk — Alert fired |

---

## Project Structure

```
ExamMonitoring/
├── models/               ← Place yolo11n.pt here
├── dataset/
│   ├── raw/              ← Original collected images/videos
│   ├── annotated/        ← Labelled in Roboflow YOLO format
│   └── splits/
│       ├── train/        ← 80% of data
│       ├── val/          ← 10% of data
│       └── test/         ← 10% of data
├── videos/               ← Place test video files here
├── output/
│   ├── logs/             ← Per-session CSV logs (auto-generated)
│   ├── reports/          ← JSON session reports (auto-generated)
│   └── annotated_frames/ ← Optional saved frames
├── src/
│   ├── main.py           ← Entry point — run this
│   ├── detector.py       ← YOLO11 detection wrapper
│   ├── tracker.py        ← ByteTrack tracking + TrackedStudent data class
│   ├── utils.py          ← Behaviour analysis, scoring, alerts, logging
│   ├── config.py         ← All settings, thresholds, weights
│   └── __init__.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone / Download
```bash
git clone https://github.com/yourname/ExamMonitoring.git
cd ExamMonitoring
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model
The model downloads automatically on first run. Or manually:
```bash
# In Python:
from ultralytics import YOLO
YOLO("yolo11n.pt")  # Downloads to current directory
```
Then move `yolo11n.pt` to the `models/` folder.

---

## Running

```bash
# Webcam (default)
python src/main.py

# Video file
python src/main.py --source videos/exam.mp4

# Secondary camera
python src/main.py --source 1
```

**Press `q` to quit.** A JSON report is saved automatically on exit.

---

## Output

Each frame shows:
- Bounding boxes (colour-coded by risk: green / orange / red)
- Student #ID labels
- Confidence scores
- Suspicion score percentage + risk level
- Behaviour flags (PHONE | LOOK AWAY | INTERACTING | ABSENT)
- Alert banner at bottom of screen
- FPS counter

---

## Dataset Preparation (Custom Training)

### Option A: COCO (zero effort, baseline)
The pretrained `yolo11n.pt` already detects `person` and `cell phone`
from the COCO dataset. No additional training needed for basic monitoring.

### Option B: Custom Dataset (Roboflow)

1. **Collect images** of an exam hall — various angles, lighting conditions,
   student positions.

2. **Upload to Roboflow** at [roboflow.com](https://roboflow.com)
   - Create a new project → Object Detection
   - Upload images

3. **Annotate** using Roboflow's online annotation tool:
   - Draw bounding boxes for: `student`, `phone`, `looking_left`,
     `looking_right`, `cheating` (custom classes)

4. **Split dataset**: 80% train / 10% val / 10% test

5. **Export in YOLO11 format** — download `data.yaml` + images + labels

6. **Fine-tune** on top of the pretrained model:
```bash
yolo detect train \
  model=yolo11n.pt \
  data=dataset/data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=8 \
  lr0=0.01 \
  device=cpu \
  project=output/training \
  name=exam_monitor_v1
```

**Parameter explanations:**
| Parameter | Meaning |
|-----------|---------|
| `model=yolo11n.pt` | Start from pretrained weights (transfer learning) |
| `data=data.yaml` | Path to dataset config |
| `epochs=50` | Number of full passes through the training data |
| `imgsz=640` | Resize all images to 640×640 before training |
| `batch=8` | 8 images per batch (lower if CPU runs out of memory) |
| `lr0=0.01` | Initial learning rate |
| `device=cpu` | Use CPU (change to `0` for GPU) |

---

## Confidence Threshold Guide

| conf | Description |
|------|-------------|
| 0.3 | Loose — detects almost everything, many false positives |
| 0.5 | Balanced — good for testing |
| **0.6** | **Production — our default** |
| 0.7 | Strict — only very confident detections; may miss occluded students |

---

## Common Errors & Fixes

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `Cannot open video source: 0` | No webcam detected | Check camera connection or use `--source videos/test.mp4` |
| `ModuleNotFoundError: ultralytics` | Not installed | Run `pip install ultralytics` |
| Very low FPS (<5) | CPU overloaded | Reduce `IMG_SIZE` to 320 in config.py |
| All students flagged as "High Risk" | `CONF_THRESHOLD` too low | Raise to 0.65 in config.py |
| `bytetrack.yaml not found` | Old Ultralytics version | `pip install ultralytics --upgrade` |

---

## Future Improvements

- **Face Recognition** — Link tracks to student roll numbers
- **MediaPipe Face Mesh** — Precise eye gaze estimation (yaw/pitch/roll)
- **YOLO Pose Estimation** — Body pose for "leaning over" / "looking behind"
- **Audio Detection** — Flag whispering using microphone input
- **LLM Report** — GPT-4 generates invigilator-ready PDF report
- **Cloud Dashboard** — Stream alerts to a web interface in real time
- **Email Alerts** — SMTP notification to invigilator's phone
- **Attendance** — Auto-mark attendance from face recognition
- **Edge Deployment** — ONNX export for Raspberry Pi / Jetson Nano

---

## Converting to a Research Paper

To publish this as a research paper (IEEE / Springer / Elsevier):

1. **Baseline comparison**: Compare your system against manual invigilation.
2. **Dataset contribution**: Annotate and publish a novel "ExamHall" dataset.
3. **Ablation study**: Remove each component (tracking / scoring / alerts)
   and measure accuracy drop.
4. **Metrics**: Report mAP@50, precision, recall, FPS, latency.
5. **User study**: Have real invigilators evaluate the alert accuracy.
6. **Limitations section**: Address edge cases (poor lighting, occlusion).

Suggested venues: IEEE ICIP, CVPR Workshops, Pattern Recognition Letters.

---

## Screenshots

*(Add screenshots of the running system here)*

---

## Author

Your Name | College Name | Final Year Project | 2024–25
