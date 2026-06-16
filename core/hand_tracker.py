"""
core/hand_tracker.py
---------------------
Thin, stateless wrapper around Google MediaPipe Hands.

Responsibilities
----------------
- Initialise and teardown the MediaPipe Hands solution.
- Process BGR frames and return raw MediaPipe results.
- Extract typed, frame-relative landmark data.
- Count how many fingers are raised.
- Provide convenience accessors (index tip, pinch midpoint, hand bbox).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Landmark:
    x: int
    y: int
    z: float

LandmarkList = List[Landmark]


# ── Constants ─────────────────────────────────────────────────────────────────

_WRIST          = 0
_THUMB_TIP      = 4
_THUMB_IP       = 3
_INDEX_TIP      = 8
_INDEX_PIP      = 6
_MIDDLE_TIP     = 12
_MIDDLE_PIP     = 10
_RING_TIP       = 16
_RING_PIP       = 14
_PINKY_TIP      = 20
_PINKY_PIP      = 18


# ── HandTracker ───────────────────────────────────────────────────────────────

class HandTracker:
    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.80,
        tracking_confidence: float = 0.75,
    ) -> None:
        
        # Standard, clean assignment using the golden version
        self._mp_hands = mp.solutions.hands
        
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._last_results = None

    # ── Core processing ───────────────────────────────────────────────────────

    def process(self, bgr_frame: np.ndarray):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True
        self._last_results = results
        return results

    # ── Hand counting ─────────────────────────────────────────────────────────

    def count_hands(self, results) -> int:
        if results.multi_hand_landmarks is None:
            return 0
        return len(results.multi_hand_landmarks)

    # ── Landmark extraction ───────────────────────────────────────────────────

    def get_landmarks(
        self,
        results,
        frame_shape: Tuple[int, int, int],
        hand_index: int = 0,
    ) -> Optional[LandmarkList]:
        if results.multi_hand_landmarks is None:
            return None
        if hand_index >= len(results.multi_hand_landmarks):
            return None

        h, w = frame_shape[:2]
        raw = results.multi_hand_landmarks[hand_index].landmark

        return [
            Landmark(
                x=int(lm.x * w),
                y=int(lm.y * h),
                z=lm.z,
            )
            for lm in raw
        ]

    # ── Finger counting ───────────────────────────────────────────────────────

    def count_fingers(self, landmarks: LandmarkList) -> int:
        if len(landmarks) < 21:
            return 0

        count = 0

        if landmarks[_THUMB_TIP].x < landmarks[_THUMB_IP].x:
            count += 1

        for tip_idx, pip_idx in (
            (_INDEX_TIP,  _INDEX_PIP),
            (_MIDDLE_TIP, _MIDDLE_PIP),
            (_RING_TIP,   _RING_PIP),
            (_PINKY_TIP,  _PINKY_PIP),
        ):
            if landmarks[tip_idx].y < landmarks[pip_idx].y:
                count += 1

        return count

    # ── Convenience accessors ─────────────────────────────────────────────────

    def index_tip(self, landmarks: LandmarkList) -> Tuple[int, int]:
        lm = landmarks[_INDEX_TIP]
        return (lm.x, lm.y)

    def pinch_midpoint(self, landmarks: LandmarkList) -> Tuple[int, int]:
        ix, iy = landmarks[_INDEX_TIP].x, landmarks[_INDEX_TIP].y
        mx, my = landmarks[_MIDDLE_TIP].x, landmarks[_MIDDLE_TIP].y
        return ((ix + mx) // 2, (iy + my) // 2)

    def hand_bbox(
        self,
        landmarks: LandmarkList,
        frame_shape: Tuple[int, int, int],
        padding: int = 20,
    ) -> Tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x1 = max(0,     min(xs) - padding)
        y1 = max(0,     min(ys) - padding)
        x2 = min(w - 1, max(xs) + padding)
        y2 = min(h - 1, max(ys) + padding)
        return (x1, y1, x2, y2)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()