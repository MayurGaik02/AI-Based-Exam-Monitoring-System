"""
main.py — Entry Point for AI Exam Monitoring System
====================================================
WHY THIS FILE EXISTS:
  Orchestrates all modules. Runs the main video loop:
    1. Capture a frame from webcam/video
    2. Run ByteTrack tracking (tracker.py)
    3. Analyse behaviour (utils.BehaviourAnalyser)
    4. Compute suspicion scores (utils.SuspicionScorer)
    5. Check for alerts (utils.AlertEngine)
    6. Draw all annotations on the frame
    7. Display the frame in a window
    8. Log events (utils.Logger)
    9. Repeat until 'q' is pressed

HOW IT CONNECTS:
  Imports and calls all other modules.
  This is the ONLY file the user needs to run:
      python src/main.py
      python src/main.py --source videos/exam.mp4
      python src/main.py --source 0   (webcam)

RUNNING:
  Activate virtualenv, then:
      python src/main.py
  Press 'q' to quit. Report is auto-saved on exit.
"""

import argparse
import time
import cv2
import numpy as np
import os
import sys

# Add src/ to path so imports work from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from tracker import Tracker
from utils  import BehaviourAnalyser, SuspicionScorer, AlertEngine, Logger


# ═══════════════════════════════════════════════════════════════════════
# CLASS: ExamMonitor
# ═══════════════════════════════════════════════════════════════════════
class ExamMonitor:
    """
    Main controller class. Creates all components and runs the video loop.

    OOP DESIGN:
      Using a class (instead of bare functions) lets us:
        • Store state (components, FPS counter, alert history)
        • Cleanly initialise and tear down resources (camera, CSV log)
        • Easily extend (e.g. add a web dashboard attribute later)
    """

    def __init__(self, source):
        """
        Initialise all components.

        Parameters
        ----------
        source : int or str
            0 for webcam, or path to a video file.
        """
        print("=" * 55)
        print("  AI-Based Exam Monitoring System")
        print("  YOLO11n + ByteTrack | Press 'q' to quit")
        print("=" * 55)

        # ── Core modules ──────────────────────────────────────────────
        self.tracker   = Tracker()
        self.analyser  = BehaviourAnalyser()
        self.scorer    = SuspicionScorer()
        self.alerter   = AlertEngine()
        self.logger    = Logger()

        # ── Video capture ──────────────────────────────────────────────
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"[main] Cannot open video source: {source}")

        # Get frame dimensions for gaze heuristic
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.analyser.set_frame_size(w, h)

        # ── FPS tracking ──────────────────────────────────────────────
        self._fps         = 0.0
        self._fps_counter = 0
        self._fps_start   = time.time()

        # ── Active alerts list (shown on screen, fades after 3 seconds) ──
        self._active_alerts = []   # list of (message, expire_time)

    # ── Main loop ─────────────────────────────────────────────────────
    def run(self):
        """Read frames in a loop until 'q' is pressed or video ends."""
        print("[Monitor] Starting video loop...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[Monitor] End of video or camera disconnected.")
                break

            now = time.time()

            # ── Step 1: Track students and phones ─────────────────────
            students, phones = self.tracker.track(frame)

            # ── Step 2: Analyse behaviour ─────────────────────────────
            states = self.analyser.update(students, phones, now)

            # ── Step 3: Score + alert ─────────────────────────────────
            new_alerts = []
            for tid, state in states.items():
                self.scorer.score(state)                     # Updates state.suspicion_score
                alerts = self.alerter.check(state, now)      # Returns list of alert strings
                new_alerts.extend(alerts)

            # Add new alerts to the on-screen display queue
            for alert in new_alerts:
                self._active_alerts.append((alert, now + 3.0))  # Visible for 3s
            # Remove expired alerts
            self._active_alerts = [(a, t) for a, t in self._active_alerts if t > now]

            # ── Step 4: Draw annotations ──────────────────────────────
            self._draw(frame, students, phones, states)

            # ── Step 5: Log ───────────────────────────────────────────
            if config.SAVE_LOG_CSV:
                self.logger.log_frame(states, new_alerts, now)

            # ── Step 6: Update FPS counter ───────────────────────────
            self._fps_counter += 1
            elapsed = now - self._fps_start
            if elapsed >= 1.0:
                self._fps       = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_start = now

            # ── Step 7: Show frame ────────────────────────────────────
            cv2.imshow(config.WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self._cleanup()

    # ── Drawing ───────────────────────────────────────────────────────
    def _draw(self, frame, students, phones, states):
        """
        Draw all annotations on the frame in-place.

        Drawing order matters: backgrounds first, then borders, then text.
        This prevents text being hidden behind boxes.
        """
        h, w = frame.shape[:2]

        # ── FPS banner ────────────────────────────────────────────────
        if config.SHOW_FPS:
            cv2.putText(
                frame, f"FPS: {self._fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                config.COLOUR_TEXT, 2
            )

        # ── Header bar ───────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 45), (20, 20, 20), -1)
        cv2.putText(
            frame, "AI EXAM MONITOR",
            (w // 2 - 100, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 200, 255), 2
        )

        # ── Draw each tracked student ─────────────────────────────────
        for student in students:
            tid   = student.track_id
            state = states.get(tid)
            if state is None:
                continue

            x1, y1, x2, y2 = student.bbox

            # Choose box colour based on risk level
            if state.risk_level == "High Risk":
                colour = config.COLOUR_HIGH
            elif state.risk_level == "Medium Risk":
                colour = config.COLOUR_MEDIUM
            else:
                colour = config.COLOUR_LOW

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            # ── Labels above box ──────────────────────────────────────
            label_y = max(y1 - 10, 50)

            # Student name + confidence
            name_label = f"{student.label}"
            if config.SHOW_CONF:
                name_label += f"  ({student.confidence:.2f})"
            cv2.putText(
                frame, name_label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2
            )

            # Suspicion score + risk level
            if config.SHOW_SUSPICION:
                score_pct  = int(state.suspicion_score * 100)
                score_label = f"Risk: {state.risk_level}  {score_pct}%"
                cv2.putText(
                    frame, score_label,
                    (x1, label_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1
                )

            # Behaviour flags below the box
            flags = []
            if state.has_phone:       flags.append("PHONE")
            if state.is_looking_away: flags.append("LOOK AWAY")
            if state.is_interacting:  flags.append("INTERACTING")
            if state.is_absent:       flags.append("ABSENT")
            if flags:
                flag_text = " | ".join(flags)
                cv2.putText(
                    frame, flag_text,
                    (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1
                )

        # ── Draw phone bounding boxes ─────────────────────────────────
        for phone in phones:
            px1, py1, px2, py2 = phone.bbox
            cv2.rectangle(frame, (px1, py1), (px2, py2), config.COLOUR_PHONE, 2)
            cv2.putText(
                frame, f"PHONE {phone.confidence:.2f}",
                (px1, py1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOUR_PHONE, 1
            )

        # ── Alert banner (bottom of screen) ──────────────────────────
        alert_y = h - 10
        for msg, _ in reversed(self._active_alerts[-4:]):  # Show last 4 alerts
            cv2.putText(
                frame, msg,
                (10, alert_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
            )
            alert_y -= 22

    # ── Cleanup ───────────────────────────────────────────────────────
    def _cleanup(self):
        """Release camera and save final report."""
        print("[Monitor] Shutting down...")
        self.cap.release()
        cv2.destroyAllWindows()

        if config.SAVE_REPORT:
            self.logger.save_report(self.analyser.states)
        self.logger.close()
        print("[Monitor] Done. Goodbye!")


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="AI Exam Monitoring System")
    parser.add_argument(
        "--source",
        default=str(config.VIDEO_SOURCE),
        help="Video source: 0 (webcam) or path to video file (default: 0)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Convert source to int if it's a digit (webcam index)
    source = int(args.source) if args.source.isdigit() else args.source

    monitor = ExamMonitor(source=source)
    monitor.run()
