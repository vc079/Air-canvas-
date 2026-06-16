"""
config/gestures.py
------------------
Maps finger counts (0 – 5) to named application modes and stores
per-gesture metadata consumed by the main loop and the HUD overlay.

Design
------
``GESTURE_MAP`` is the single source of truth.  The main loop calls
``get_mode(finger_count)`` and switches behaviour based on the returned
``GestureMode`` value.  The HUD overlay calls ``get_meta(mode)`` to
look up the label and hint text to display on screen.

Adding a new gesture
--------------------
1. Add a new member to the ``GestureMode`` enum.
2. Add an entry to ``GESTURE_MAP`` keyed by finger count.
3. The rest of the system picks it up automatically.
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass


# ======================================================================
# Mode enumeration
# ======================================================================

class GestureMode(Enum):
    """
    All operational modes, one per unique hand gesture.
    The integer values are arbitrary; use the names for comparisons.
    """
    IDLE          = auto()   # 0 fingers  – fist, no action
    DRAW          = auto()   # 1 finger   – index tip draws
    DRAG          = auto()   # 2 fingers  – grab & pan canvas
    COLOR_PICKER  = auto()   # 3 fingers  – hover over palette swatch
    SNAPSHOT      = auto()   # 4 fingers  – save to disk
    ERASE         = auto()   # 5 fingers  – open-hand bounding-box wipe
    MULTI_HAND    = auto()   # sentinel   – two hands in frame (safety)
    UNKNOWN       = auto()   # sentinel   – unrecognised count


# ======================================================================
# Per-gesture metadata
# ======================================================================

@dataclass(frozen=True)
class GestureMeta:
    """
    Display and behaviour metadata for a single gesture.

    Attributes
    ----------
    mode        : The ``GestureMode`` this record describes.
    label       : Short name shown in the HUD mode badge (≤ 12 chars).
    hint        : One-line instruction shown below the mode badge.
    finger_count: Expected raised-finger count (-1 = not finger-triggered).
    hud_color   : BGR colour used for the mode badge background.
    draw_landmarks: Whether MediaPipe landmarks should be drawn on screen.
    """
    mode:             GestureMode
    label:            str
    hint:             str
    finger_count:     int
    hud_color:        tuple[int, int, int]
    draw_landmarks:   bool = True


# ======================================================================
# Gesture map  (finger_count → GestureMeta)
# ======================================================================
# Edit labels / hints here to customise the on-screen HUD text.

GESTURE_MAP: dict[int, GestureMeta] = {
    0: GestureMeta(
        mode          = GestureMode.IDLE,
        label         = "Idle",
        hint          = "Fist closed — move freely without drawing",
        finger_count  = 0,
        hud_color     = ( 60,  60,  60),   # dark grey
    ),
    1: GestureMeta(
        mode          = GestureMode.DRAW,
        label         = "Draw",
        hint          = "Index finger traces lines on the canvas",
        finger_count  = 1,
        hud_color     = (  0, 180,   0),   # green
    ),
    2: GestureMeta(
        mode          = GestureMode.DRAG,
        label         = "Drag",
        hint          = "Pinch & move — pans the entire canvas",
        finger_count  = 2,
        hud_color     = (200, 130,   0),   # blue-ish teal
    ),
    3: GestureMeta(
        mode          = GestureMode.COLOR_PICKER,
        label         = "Colour",
        hint          = "Hover index finger over a swatch to pick colour",
        finger_count  = 3,
        hud_color     = (180,   0, 180),   # purple
    ),
    4: GestureMeta(
        mode          = GestureMode.SNAPSHOT,
        label         = "Snapshot",
        hint          = "Hold still — saves canvas PNG to disk",
        finger_count  = 4,
        hud_color     = (  0, 180, 255),   # amber/orange
    ),
    5: GestureMeta(
        mode          = GestureMode.ERASE,
        label         = "Erase",
        hint          = "Open hand — bounding box wipes ink beneath",
        finger_count  = 5,
        hud_color     = (  0,   0, 220),   # red
    ),
}

# Sentinels for special states (not driven by a finger count)
_MULTI_HAND_META = GestureMeta(
    mode          = GestureMode.MULTI_HAND,
    label         = "!! STOP !!",
    hint          = "Multiple hands detected — drawing paused",
    finger_count  = -1,
    hud_color     = (  0,   0, 255),   # bright red
    draw_landmarks= True,
)

_UNKNOWN_META = GestureMeta(
    mode          = GestureMode.UNKNOWN,
    label         = "Unknown",
    hint          = "",
    finger_count  = -1,
    hud_color     = ( 40,  40,  40),
    draw_landmarks= False,
)


# ======================================================================
# Public helper functions
# ======================================================================

def get_mode(finger_count: int) -> GestureMode:
    """
    Convert a raw finger count into a ``GestureMode``.

    Parameters
    ----------
    finger_count : int
        Output of ``FingerCounter.count()``.  Values outside 0-5
        return ``GestureMode.UNKNOWN``.

    Returns
    -------
    GestureMode
    """
    meta = GESTURE_MAP.get(finger_count)
    return meta.mode if meta else GestureMode.UNKNOWN


def get_meta(mode: GestureMode) -> GestureMeta:
    """
    Retrieve display metadata for the given mode.

    Parameters
    ----------
    mode : GestureMode

    Returns
    -------
    GestureMeta
        Never raises; falls back to the UNKNOWN sentinel.
    """
    if mode == GestureMode.MULTI_HAND:
        return _MULTI_HAND_META

    for meta in GESTURE_MAP.values():
        if meta.mode == mode:
            return meta

    return _UNKNOWN_META


def get_meta_by_count(finger_count: int) -> GestureMeta:
    """
    Convenience: look up metadata directly from a finger count.

    Parameters
    ----------
    finger_count : int

    Returns
    -------
    GestureMeta
    """
    return GESTURE_MAP.get(finger_count, _UNKNOWN_META)


def all_modes() -> list[GestureMode]:
    """Return all gesture modes in finger-count order (0 → 5)."""
    return [meta.mode for meta in sorted(GESTURE_MAP.values(),
                                         key=lambda m: m.finger_count)]


# ----------------------------------------------------------------------
# Compatibility aliases expected by older modules/tests
# ----------------------------------------------------------------------
# Backwards-compatible mapping: finger count -> short mode name string
FINGER_TO_MODE: dict[int, str] = {
    0: "idle",
    1: "draw",
    2: "drag",
    3: "palette",
    4: "snapshot",
    5: "erase",
}

# Number of consecutive frames required to confirm a mode. Tests expect
# fast response so set to 1 by default (can be overridden in runtime config).
DEBOUNCE_FRAMES: int = 1
