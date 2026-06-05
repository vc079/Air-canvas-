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

Design notes
------------
- All pixel coordinates are returned as plain (int, int) tuples so callers
  never need to import mediapipe themselves.
- Frame dimensions are required only when converting normalised landmarks to
  pixel space; they are never stored on the instance.
- `process()` expects an *unmodified* BGR frame (OpenCV default). The method
  converts to RGB internally and marks the frame non-writeable for speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


# ── Types ─────────────────────────────────────────────────────────────────────

# A single landmark expressed in pixel coordinates.
# z is normalised depth (unused for 2-D drawing but kept for future use).
@dataclass(frozen=True)
class Landmark:
    x: int
    y: int
    z: float


LandmarkList = List[Landmark]   # 21 landmarks, indices match MediaPipe spec


# ── Constants ─────────────────────────────────────────────────────────────────

# MediaPipe landmark indices
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
    """
    Wraps MediaPipe Hands with a minimal, type-safe interface.

    Parameters
    ----------
    max_hands : int
        Maximum number of hands to detect. The gesture engine only acts on
        single-hand frames, but we detect up to 2 to trigger the multi-hand
        warning correctly.
    detection_confidence : float
        Minimum confidence for the initial hand detection.
    tracking_confidence : float
        Minimum confidence for subsequent landmark tracking.
    """

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.80,
        tracking_confidence: float = 0.75,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        # Cache the last results so callers can call accessors without
        # re-processing the same frame.
        self._last_results = None

    # ── Core processing ───────────────────────────────────────────────────────

    def process(self, bgr_frame: np.ndarray):
        """
        Run MediaPipe on *bgr_frame* and return the raw results object.

        The frame is converted to RGB and marked non-writeable before
        inference, then restored — this avoids an unnecessary copy.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True
        self._last_results = results
        return results

    # ── Hand counting ─────────────────────────────────────────────────────────

    def count_hands(self, results) -> int:
        """Return the number of hands detected in *results*."""
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
        """
        Convert normalised MediaPipe landmarks to pixel-space ``Landmark``
        objects for the hand at *hand_index*.

        Returns ``None`` if no hand is detected.
        """
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
        """
        Count the number of extended fingers (0-5).

        Method: a finger is "up" when its tip's Y-coordinate is *above*
        (smaller value) its PIP knuckle's Y-coordinate.  The thumb uses a
        horizontal comparison (X-axis) because it moves laterally.

        This approach is robust to hand rotation within ±30° of upright.
        """
        if len(landmarks) < 21:
            return 0

        count = 0

        # Thumb: compare X to handle mirrored (selfie) orientation.
        # When the hand is mirrored, tip.x < ip.x means thumb is extended.
        if landmarks[_THUMB_TIP].x < landmarks[_THUMB_IP].x:
            count += 1

        # Four fingers: tip above (lower Y) than PIP knuckle
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
        """Return the (x, y) pixel position of the index fingertip."""
        lm = landmarks[_INDEX_TIP]
        return (lm.x, lm.y)

    def pinch_midpoint(self, landmarks: LandmarkList) -> Tuple[int, int]:
        """
        Return the midpoint between the index tip and middle tip.
        Used as the drag handle position in 2-finger drag mode.
        """
        ix, iy = landmarks[_INDEX_TIP].x, landmarks[_INDEX_TIP].y
        mx, my = landmarks[_MIDDLE_TIP].x, landmarks[_MIDDLE_TIP].y
        return ((ix + mx) // 2, (iy + my) // 2)

    def hand_bbox(
        self,
        landmarks: LandmarkList,
        frame_shape: Tuple[int, int, int],
        padding: int = 20,
    ) -> Tuple[int, int, int, int]:
        """
        Return the axis-aligned bounding box (x1, y1, x2, y2) that encloses
        all 21 landmarks, expanded by *padding* pixels on each side.

        Used by the open-hand wipe eraser.
        """
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
        """Release MediaPipe resources. Call once on application exit."""
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()