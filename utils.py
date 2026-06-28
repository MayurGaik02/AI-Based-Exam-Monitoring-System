"""
utils.py — Behaviour Analysis, Suspicion Scoring, Alert Engine & Logger
=======================================================================
WHY THIS FILE EXISTS:
  The brain of the system. Takes raw tracking data (students, phones, frame)
  and applies domain-specific logic to produce suspicion scores, risk labels,
  alerts, and event logs.

  Three classes live here:
    1. BehaviourAnalyser  → detects WHAT each student is doing
    2. SuspicionScorer    → converts behaviours into a numeric risk score
    3. AlertEngine        → decides WHEN to fire an alert
    4. Logger             → writes events to CSV and JSON report

HOW IT CONNECTS:
  main.py → creates one of each → calls them in order each frame
  tracker.py output (students, phones) → BehaviourAnalyser → SuspicionScorer
  → AlertEngine → Logger → main.py draws the results on-screen
"""

import time
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import config
from tracker import TrackedStudent, TrackedPhone


# ═══════════════════════════════════════════════════════════════════════
# DATA CLASS: StudentState
# Holds all runtime state for ONE tracked student across the session.
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class StudentState:
    """
    Persistent memory for one student across all frames.

    Why not store this in TrackedStudent?
      TrackedStudent is recreated every frame from YOLO output.
      StudentState persists in a dictionary keyed by track_id.
    """
    track_id           : int
    first_seen_time    : float = field(default_factory=time.time)
    last_seen_time     : float = field(default_factory=time.time)

    # Behaviour flags (current frame)
    is_looking_away    : bool  = False
    has_phone          : bool  = False
    is_interacting     : bool  = False
    is_absent          : bool  = False

    # Timers (used to measure duration of behaviours)
    look_away_start    : Optional[float] = None   # When they started looking away
    look_away_total_s  : float = 0.0              # Accumulated look-away seconds
    phone_detected_at  : Optional[float] = None  # First time phone seen near them
    interaction_start  : Optional[float] = None

    # Suspicion score (0.0–1.0)
    suspicion_score    : float = 0.0
    risk_level         : str   = "Low Risk"       # "Low Risk" | "Medium Risk" | "High Risk"

    # Alert log for this student
    alerts             : List[str] = field(default_factory=list)

    # Bounding box (updated each frame)
    bbox               : tuple = (0, 0, 0, 0)
    cx                 : int   = 0
    cy                 : int   = 0


# ═══════════════════════════════════════════════════════════════════════
# CLASS: BehaviourAnalyser
# ═══════════════════════════════════════════════════════════════════════
class BehaviourAnalyser:
    """
    Analyses student behaviour from tracking data each frame.

    HOW DOES GAZE DETECTION WORK WITHOUT FACE RECOGNITION?
      We use a clever proxy: the bounding box of the person relative to the
      frame centre. This is a simplified heuristic:
        • If a student's bounding box centre moves significantly to the LEFT
          of where it normally is → they may be looking left.
        • This is NOT true gaze tracking (which requires MediaPipe or face mesh).
        • For a college project this is acceptable and demonstrable.
        • Future improvement: add MediaPipe Face Mesh for precise eye gaze.

    HOW DOES INTERACTION DETECTION WORK?
      If two students' bounding box CENTRES are within INTERACTION_DIST_PX
      pixels of each other → they are flagged as interacting.
      This works because students sitting close enough to talk/share answers
      will have overlapping or very close bounding boxes.

    HOW IS PHONE DETECTION LINKED TO A STUDENT?
      For each detected phone, find the student whose bounding box CONTAINS
      or is NEAREST to the phone's centre. If distance < threshold → that
      student "has a phone."
    """

    def __init__(self):
        self.states: Dict[int, StudentState] = {}   # track_id → StudentState
        self._frame_width  = 640
        self._frame_height = 480

    def set_frame_size(self, w: int, h: int):
        """Call this once after getting the first frame."""
        self._frame_width  = w
        self._frame_height = h

    def update(
        self,
        students : List[TrackedStudent],
        phones   : List[TrackedPhone],
        now      : float,   # current wall-clock time (time.time())
    ) -> Dict[int, StudentState]:
        """
        Process one frame: update all student states.

        Parameters
        ----------
        students : detections from tracker.py
        phones   : phone detections from tracker.py
        now      : current time in seconds (for duration calculations)

        Returns
        -------
        Dict[track_id → StudentState] — updated state for all students
        """
        current_ids = {s.track_id for s in students}

        # ── Mark absent students ──────────────────────────────────────
        for tid, state in self.states.items():
            if tid not in current_ids:
                elapsed = now - state.last_seen_time
                if elapsed > config.ABSENCE_TIMEOUT_SEC:
                    state.is_absent = True
                    state.is_looking_away = False
                    state.has_phone       = False
                    state.is_interacting  = False

        # ── Process each visible student ─────────────────────────────
        for student in students:
            tid = student.track_id

            # Create state if first time seeing this student
            if tid not in self.states:
                self.states[tid] = StudentState(track_id=tid)

            state = self.states[tid]
            state.last_seen_time = now
            state.is_absent      = False
            state.bbox = student.bbox
            state.cx   = student.cx
            state.cy   = student.cy

            # ── Phone detection ───────────────────────────────────────
            state.has_phone = self._check_phone(student, phones)
            if state.has_phone and state.phone_detected_at is None:
                state.phone_detected_at = now

            # ── Gaze / head pose (heuristic) ─────────────────────────
            #   We estimate gaze by checking how far the student's bbox
            #   centre is from a "frontal" expected position.
            #   A more accurate method: MediaPipe Face Mesh (future work).
            looking_away = self._check_gaze(student)
            state.is_looking_away = looking_away

            if looking_away:
                if state.look_away_start is None:
                    state.look_away_start = now
            else:
                if state.look_away_start is not None:
                    # Accumulate the time they looked away
                    state.look_away_total_s += (now - state.look_away_start)
                state.look_away_start = None

            # ── Interaction detection ─────────────────────────────────
            state.is_interacting = self._check_interaction(student, students)

        return self.states

    # ── Private helpers ───────────────────────────────────────────────

    def _check_phone(
        self,
        student : TrackedStudent,
        phones  : List[TrackedPhone],
    ) -> bool:
        """
        Return True if any phone is inside or very close to this student's box.

        Strategy:
          1. Check if the phone's centre is INSIDE the student's bounding box.
          2. Or if the distance between phone centre and student centre < threshold.
        """
        x1, y1, x2, y2 = student.bbox
        for phone in phones:
            px, py = phone.cx, phone.cy
            # Is the phone centre inside the student's bounding box?
            if x1 <= px <= x2 and y1 <= py <= y2:
                return True
            # Is the phone close to the student's centre?
            dist = np.hypot(px - student.cx, py - student.cy)
            if dist < 80:   # pixels — adjust based on camera resolution
                return True
        return False

    def _check_gaze(self, student: TrackedStudent) -> bool:
        """
        Heuristic gaze detection using bounding box position.

        REAL APPROACH (for future improvement):
          Use MediaPipe Face Mesh to get 468 facial landmarks, then
          compute head pose using the PnP algorithm (cv2.solvePnP).
          This gives yaw/pitch/roll angles which precisely indicate
          left/right/down gaze.

        HEURISTIC APPROACH (used here):
          We assume the camera is centred in front of students (like a typical
          exam hall setup). If a student's bbox centre deviates significantly
          from the frame's horizontal midline, they're likely looking sideways.

          This is a PROXY — not accurate for every student.
          Works best when: single student per seat, camera frontal.
        """
        frame_cx = self._frame_width // 2
        # How far is this student from the frame centre horizontally?
        deviation = abs(student.cx - frame_cx)

        # If the student is very close to the left or right edge →
        # they're likely turned to one side
        if student.cx < self._frame_width * 0.2:    # Far left
            return True
        if student.cx > self._frame_width * 0.8:    # Far right
            return True

        # If deviation is large relative to their own box width → turned
        box_width = student.bbox[2] - student.bbox[0]
        if deviation > box_width * 0.8:
            return True

        return False

    def _check_interaction(
        self,
        student  : TrackedStudent,
        all_students : List[TrackedStudent],
    ) -> bool:
        """
        Return True if this student is very close to another student.

        Measures Euclidean distance between bounding box centres.
        Distance < INTERACTION_DIST_PX → flagged as potential interaction.
        """
        for other in all_students:
            if other.track_id == student.track_id:
                continue    # Don't compare to self
            dist = np.hypot(
                other.cx - student.cx,
                other.cy - student.cy
            )
            if dist < config.INTERACTION_DIST_PX:
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════
# CLASS: SuspicionScorer
# ═══════════════════════════════════════════════════════════════════════
class SuspicionScorer:
    """
    Converts behaviour flags into a numeric suspicion score per student.

    THE FORMULA:
      score = Σ (weight_i × flag_i)

    Where flag_i is 1.0 if the behaviour is active, or a continuous value
    for time-based behaviours (e.g. looking away for 5s vs 30s).

    WEIGHTS (from config.py):
      looking_away  → 0.40  (most suspicious)
      phone         → 0.35
      interaction   → 0.15
      seat_leaving  → 0.10

    WHY THESE WEIGHTS?
      Based on typical exam invigilation rules:
      • Looking away is the most common indicator of copying.
      • Phone is a severe violation — but harder to detect precisely at distance.
      • Interaction is significant but could be accidental proximity.
      • Seat leaving alone is minor (student may be stretching).

    RANGE: 0.0 (totally fine) → 1.0 (all behaviours active simultaneously)
    """

    def score(self, state: StudentState) -> float:
        """
        Compute the suspicion score for one student in the current frame.

        Returns
        -------
        float in range [0.0, 1.0]
        """
        w = config.SUSPICION_WEIGHTS

        # Looking away: scaled by time (more time = higher score component)
        # Max contribution capped at weight value (1.0 × weight)
        look_away_flag = 0.0
        if state.is_looking_away:
            # If they've been looking away > GAZE_AWAY_SECONDS → full weight
            if state.look_away_start is not None:
                duration = time.time() - state.look_away_start
                look_away_flag = min(1.0, duration / config.GAZE_AWAY_SECONDS)
        elif state.look_away_total_s > 0:
            # They looked away in the past — partial contribution
            look_away_flag = min(0.5, state.look_away_total_s / 30.0)

        phone_flag       = 1.0 if state.has_phone      else 0.0
        interaction_flag = 1.0 if state.is_interacting else 0.0
        seat_flag        = 1.0 if state.is_absent       else 0.0

        score = (
            w["looking_away"]   * look_away_flag
          + w["phone_detected"] * phone_flag
          + w["interaction"]    * interaction_flag
          + w["seat_leaving"]   * seat_flag
        )

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        # Update the state object
        state.suspicion_score = score
        state.risk_level      = self._risk_label(score)

        return score

    @staticmethod
    def _risk_label(score: float) -> str:
        """Map numeric score to human-readable risk label."""
        if score <= config.LOW_RISK_MAX:
            return "Low Risk"
        elif score <= config.MEDIUM_RISK_MAX:
            return "Medium Risk"
        else:
            return "High Risk"


# ═══════════════════════════════════════════════════════════════════════
# CLASS: AlertEngine
# ═══════════════════════════════════════════════════════════════════════
class AlertEngine:
    """
    Decides when to fire an alert based on student state.

    ALERT RULES (from config.py):
      1. Phone detected near any student
      2. Suspicion score exceeds HIGH_RISK_MIN (0.61 by default)
      3. Student absent for > ABSENCE_TIMEOUT_SEC seconds
      4. Student is interacting with another student

    COOLDOWN:
      To avoid flooding the screen with repeated alerts for the same event,
      we apply a cooldown: once an alert fires for a student+event pair,
      it won't fire again for 10 seconds.
    """

    COOLDOWN_SECONDS = 10.0

    def __init__(self):
        # last_alert_time[track_id][alert_type] = timestamp
        self._last_alert: Dict[int, Dict[str, float]] = defaultdict(dict)

    def check(self, state: StudentState, now: float) -> List[str]:
        """
        Check one student's state and return a list of new alert strings.

        Parameters
        ----------
        state : StudentState
        now   : current time.time()

        Returns
        -------
        list of alert message strings (may be empty)
        """
        alerts = []
        tid    = state.track_id

        def should_fire(alert_type: str) -> bool:
            """Only fire if cooldown has passed."""
            last = self._last_alert[tid].get(alert_type, 0)
            if now - last > self.COOLDOWN_SECONDS:
                self._last_alert[tid][alert_type] = now
                return True
            return False

        if state.has_phone and should_fire("phone"):
            alerts.append(f"⚠ Student #{tid}: PHONE DETECTED")

        if state.suspicion_score >= config.HIGH_RISK_MIN and should_fire("high_risk"):
            pct = int(state.suspicion_score * 100)
            alerts.append(f"⚠ Student #{tid}: HIGH RISK ({pct}% suspicion)")

        if state.is_absent and should_fire("absent"):
            alerts.append(f"⚠ Student #{tid}: LEFT EXAM ZONE")

        if state.is_interacting and should_fire("interaction"):
            alerts.append(f"⚠ Student #{tid}: INTERACTING WITH STUDENT")

        # Append to the student's personal alert log
        state.alerts.extend(alerts)

        return alerts


# ═══════════════════════════════════════════════════════════════════════
# CLASS: Logger
# ═══════════════════════════════════════════════════════════════════════
class Logger:
    """
    Writes per-event CSV logs and a session-end JSON summary report.

    CSV log columns:
      timestamp, track_id, risk_level, suspicion_score,
      has_phone, is_looking_away, is_interacting, is_absent, alert

    JSON report:
      Session-level summary: duration, total alerts, per-student stats.
    """

    def __init__(self):
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(config.REPORT_DIR, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        self._csv_path    = os.path.join(config.LOG_DIR,    f"session_{ts}.csv")
        self._report_path = os.path.join(config.REPORT_DIR, f"report_{ts}.json")

        self._csv_file   = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "timestamp", "track_id", "risk_level", "suspicion_score",
            "has_phone", "is_looking_away", "is_interacting", "is_absent", "alert"
        ])

        self._session_start = time.time()
        self._last_log: Dict[int, float] = {}   # track_id → last log time
        print(f"[Logger] CSV log: {self._csv_path}")

    def log_frame(
        self,
        states : Dict[int, StudentState],
        alerts : List[str],
        now    : float,
    ):
        """Log the current state of all students at LOG_INTERVAL_S frequency."""
        for tid, state in states.items():
            last = self._last_log.get(tid, 0)
            if now - last < config.LOG_INTERVAL_S:
                continue   # Not time to log this student yet

            alert_str = "; ".join(a for a in alerts if f"#{tid}" in a)
            self._csv_writer.writerow([
                time.strftime("%H:%M:%S"),
                tid,
                state.risk_level,
                f"{state.suspicion_score:.2f}",
                int(state.has_phone),
                int(state.is_looking_away),
                int(state.is_interacting),
                int(state.is_absent),
                alert_str,
            ])
            self._last_log[tid] = now

    def save_report(self, states: Dict[int, StudentState]):
        """Write the JSON summary report at end of session."""
        duration = time.time() - self._session_start
        report = {
            "session_duration_seconds": round(duration, 1),
            "total_students_detected":  len(states),
            "students": {}
        }
        for tid, state in states.items():
            report["students"][str(tid)] = {
                "label":              state.risk_level,
                "final_score":        round(state.suspicion_score, 3),
                "total_look_away_s":  round(state.look_away_total_s, 1),
                "phone_detected":     state.has_phone,
                "total_alerts":       len(state.alerts),
                "alerts":             state.alerts[-10:],   # Last 10 alerts
            }
        with open(self._report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Logger] Report saved: {self._report_path}")

    def close(self):
        """Flush and close the CSV file."""
        self._csv_file.flush()
        self._csv_file.close()
