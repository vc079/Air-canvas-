"""
ui/multi_hand_warn.py
----------------------
Renders a full-width warning banner when more than one hand is detected
in the frame, informing the user that drawing is paused.

Visual design
-------------
- A semi-transparent red bar spans the full width of the frame at the top.
- Bold white text reads "⚠  MULTI-HAND DETECTED — Drawing Paused".
- A pulsing red border vignette around the frame edges reinforces urgency
  without being distracting.

The actual gesture-pause logic lives in main.py (which calls
``GestureEngine.reset()``).  This class is purely presentational.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


# ── Visual constants ──────────────────────────────────────────────────────────

_BAR_HEIGHT     = 52
_BAR_COLOR      = (30, 30, 200)       # deep red in BGR
_BAR_ALPHA      = 0.72
_TEXT_PRIMARY   = "  !!  MULTI-HAND DETECTED"
_TEXT_SECONDARY = "Drawing paused — show only ONE hand"
_FONT           = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE_P   = 0.70
_FONT_SCALE_S   = 0.48
_FONT_THICK_P   = 2
_FONT_THICK_S   = 1
_TEXT_COLOR     = (220, 220, 255)
_BORDER_COLOR   = (40, 40, 220)
_BORDER_THICK   = 4

# Pulse: border alpha oscillates via a frame counter
_PULSE_SPEED    = 0.18   # radians per frame


class MultiHandWarning:
    """
    Draws the multi-hand safety warning overlay onto a frame.

    The pulsing border effect is driven by an internal frame counter so
    each instance pulses independently.
    """

    def __init__(self) -> None:
        self._tick: int = 0

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Render the warning overlay onto *frame* and return the result.

        Parameters
        ----------
        frame : np.ndarray
            Current BGR output frame (after compositing).
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # ── Top warning bar ───────────────────────────────────────────────────
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, _BAR_HEIGHT), _BAR_COLOR, cv2.FILLED)
        cv2.addWeighted(overlay, _BAR_ALPHA, out, 1 - _BAR_ALPHA, 0, out)

        # Primary text (centred)
        (pw, ph), _ = cv2.getTextSize(
            _TEXT_PRIMARY, _FONT, _FONT_SCALE_P, _FONT_THICK_P
        )
        px = (w - pw) // 2
        py = int(_BAR_HEIGHT * 0.55)
        cv2.putText(
            out, _TEXT_PRIMARY, (px, py),
            _FONT, _FONT_SCALE_P, _TEXT_COLOR, _FONT_THICK_P, cv2.LINE_AA,
        )

        # Secondary text (centred, smaller)
        (sw, sh), _ = cv2.getTextSize(
            _TEXT_SECONDARY, _FONT, _FONT_SCALE_S, _FONT_THICK_S
        )
        sx = (w - sw) // 2
        sy = int(_BAR_HEIGHT * 0.90)
        cv2.putText(
            out, _TEXT_SECONDARY, (sx, sy),
            _FONT, _FONT_SCALE_S, _TEXT_COLOR, _FONT_THICK_S, cv2.LINE_AA,
        )

        # ── Pulsing frame border ──────────────────────────────────────────────
        pulse_alpha = 0.35 + 0.30 * math.sin(self._tick * _PULSE_SPEED)
        border_overlay = out.copy()
        # Draw thick border rectangle
        cv2.rectangle(
            border_overlay,
            (_BORDER_THICK // 2, _BORDER_THICK // 2),
            (w - _BORDER_THICK // 2, h - _BORDER_THICK // 2),
            _BORDER_COLOR,
            _BORDER_THICK,
        )
        cv2.addWeighted(
            border_overlay, pulse_alpha,
            out,            1 - pulse_alpha,
            0, out,
        )

        self._tick += 1
        return out

    def reset_pulse(self) -> None:
        """Reset the pulse phase — call when switching back to single-hand."""
        self._tick = 0