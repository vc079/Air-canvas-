"""
utils/finger_counter.py
------------------------
A standalone, importable finger-counting utility that operates directly
on a MediaPipe landmark list without requiring a ``HandTracker`` instance.

Why a separate module?
-----------------------
``HandTracker.count_fingers()`` is the primary runtime path.  This module
exists for two separate concerns:

1. **Unit testing** — tests can feed synthetic landmark data to
   ``count_fingers_from_landmarks()`` without instantiating MediaPipe.
2. **Alternative entry points** — if a future integration (e.g. a FastAPI
   endpoint accepting pre-processed landmarks) needs finger counting
   without the webcam pipeline, it can import this module directly.

Both ``HandTracker`` and this module implement the same Y-coordinate
comparison algorithm so behaviour is guaranteed identical.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence


# ── Data types ────────────────────────────────────────────────────────────────

class Point2D(NamedTuple):
    """Minimal 2-D landmark with pixel coordinates."""
    x: int
    y: int


# MediaPipe landmark indices (subset used for finger counting)
_THUMB_TIP  = 4;  _THUMB_IP   = 3
_INDEX_TIP  = 8;  _INDEX_PIP  = 6
_MIDDLE_TIP = 12; _MIDDLE_PIP = 10
_RING_TIP   = 16; _RING_PIP   = 14
_PINKY_TIP  = 20; _PINKY_PIP  = 18

_FINGER_PAIRS = (
    (_INDEX_TIP,  _INDEX_PIP),
    (_MIDDLE_TIP, _MIDDLE_PIP),
    (_RING_TIP,   _RING_PIP),
    (_PINKY_TIP,  _PINKY_PIP),
)


# ── Public functions ──────────────────────────────────────────────────────────

def count_fingers_from_landmarks(
    landmarks: Sequence[Point2D],
    mirrored: bool = True,
) -> int:
    """
    Count the number of extended fingers (0–5) from a 21-point landmark list.

    Parameters
    ----------
    landmarks : Sequence[Point2D]
        Ordered list of 21 (x, y) pixel-space landmarks following the
        MediaPipe Hands index convention.
    mirrored : bool
        Set to ``True`` (default) when the webcam feed has been horizontally
        flipped (selfie orientation).  Affects the thumb X-axis comparison
        direction.

    Returns
    -------
    int
        Number of fingers detected as extended (0–5).

    Raises
    ------
    ValueError
        If fewer than 21 landmarks are provided.

    Examples
    --------
    >>> pts = [Point2D(0, 0)] * 21
    >>> count_fingers_from_landmarks(pts)
    0
    """
    if len(landmarks) < 21:
        raise ValueError(
            f"Expected 21 landmarks, got {len(landmarks)}. "
            "Ensure MediaPipe has fully detected the hand."
        )

    count = 0

    # Thumb: horizontal comparison (moves laterally, not vertically)
    # In mirrored (selfie) mode, tip.x < ip.x → extended
    thumb_tip = landmarks[_THUMB_TIP]
    thumb_ip  = landmarks[_THUMB_IP]
    if mirrored:
        if thumb_tip.x < thumb_ip.x:
            count += 1
    else:
        if thumb_tip.x > thumb_ip.x:
            count += 1

    # Four fingers: tip above (lower Y) than PIP knuckle
    for tip_idx, pip_idx in _FINGER_PAIRS:
        if landmarks[tip_idx].y < landmarks[pip_idx].y:
            count += 1

    return count


def fingers_state(
    landmarks: Sequence[Point2D],
    mirrored: bool = True,
) -> dict[str, bool]:
    """
    Return the extended/curled state of each individual finger.

    Useful for debugging and for building more granular gesture rules
    in future iterations.

    Returns
    -------
    dict[str, bool]
        Keys: ``"thumb"``, ``"index"``, ``"middle"``, ``"ring"``, ``"pinky"``.
        Values: ``True`` if the finger is extended.
    """
    if len(landmarks) < 21:
        raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}.")

    thumb_up = (
        landmarks[_THUMB_TIP].x < landmarks[_THUMB_IP].x
        if mirrored
        else landmarks[_THUMB_TIP].x > landmarks[_THUMB_IP].x
    )

    return {
        "thumb":  thumb_up,
        "index":  landmarks[_INDEX_TIP].y  < landmarks[_INDEX_PIP].y,
        "middle": landmarks[_MIDDLE_TIP].y < landmarks[_MIDDLE_PIP].y,
        "ring":   landmarks[_RING_TIP].y   < landmarks[_RING_PIP].y,
        "pinky":  landmarks[_PINKY_TIP].y  < landmarks[_PINKY_PIP].y,
    }