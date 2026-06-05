"""
ui/eraser_box.py
-----------------
Renders the animated bounding-box eraser visualisation during the open-hand
(5-finger) wipe gesture.

Visual design
-------------
- A dashed, animated red rectangle outlines the eraser zone.
- Corner brackets are drawn in solid white to emphasise the corners.
- A centred ``ERASE`` label floats inside the box.
- The box animates via a marching-dashes effect (phase offset increments each
  frame) to make the eraser area immediately distinguishable from ink.

The actual canvas erasure is performed by ``CanvasRenderer.erase()``; this
class is purely presentational.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# ── Visual constants ──────────────────────────────────────────────────────────

_BOX_COLOR      = (60, 60, 220)      # bold red-ish in BGR
_CORNER_COLOR   = (255, 255, 255)
_LABEL_COLOR    = (60, 60, 220)
_LABEL_BG       = (255, 255, 255)
_DASH_LEN       = 12                 # length of each dash segment (px)
_DASH_GAP       = 8                  # gap between dashes (px)
_BOX_THICK      = 2
_CORNER_LEN     = 18                 # length of each corner bracket arm
_CORNER_THICK   = 3
_LABEL_FONT     = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE    = 0.55
_LABEL_THICK    = 2
_FILL_ALPHA     = 0.08               # very subtle red tint inside box


class EraserBox:
    """
    Animated dashed-rectangle overlay for the open-hand eraser.

    Call ``draw(frame, bbox)`` once per frame while in erase mode.
    The internal phase counter auto-increments to animate the dashes.
    """

    def __init__(self) -> None:
        self._phase: int = 0   # marching-ants offset

    def draw(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Render the animated eraser box onto *frame* and return the result.

        Parameters
        ----------
        frame : np.ndarray
            Current BGR output frame.
        bbox : Tuple[int, int, int, int]
            (x1, y1, x2, y2) bounding box in pixel coordinates, as returned
            by ``HandTracker.hand_bbox()``.
        """
        out = frame.copy()
        x1, y1, x2, y2 = bbox

        # Subtle red tint fill inside the eraser zone
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), _BOX_COLOR, cv2.FILLED)
        cv2.addWeighted(overlay, _FILL_ALPHA, out, 1 - _FILL_ALPHA, 0, out)

        # Dashed border
        self._draw_dashed_rect(out, x1, y1, x2, y2)

        # Corner brackets
        self._draw_corners(out, x1, y1, x2, y2)

        # Centred label
        self._draw_label(out, x1, y1, x2, y2)

        # Advance marching phase
        self._phase = (self._phase + 2) % (_DASH_LEN + _DASH_GAP)

        return out

    # ── Private helpers ───────────────────────────────────────────────────────

    def _draw_dashed_rect(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
    ) -> None:
        """Draw a marching-ants dashed rectangle in-place."""
        period = _DASH_LEN + _DASH_GAP

        def dashed_line(pt_a: Tuple[int, int], pt_b: Tuple[int, int]) -> None:
            ax, ay = pt_a
            bx, by = pt_b
            length = int(np.hypot(bx - ax, by - ay))
            if length == 0:
                return
            dx = (bx - ax) / length
            dy = (by - ay) / length

            t = -self._phase  # start offset so dashes appear to march
            while t < length:
                ts  = max(t, 0)
                te  = min(t + _DASH_LEN, length)
                if te > 0:
                    p1 = (int(ax + dx * ts), int(ay + dy * ts))
                    p2 = (int(ax + dx * te), int(ay + dy * te))
                    cv2.line(frame, p1, p2, _BOX_COLOR, _BOX_THICK, cv2.LINE_AA)
                t += period

        dashed_line((x1, y1), (x2, y1))   # top
        dashed_line((x2, y1), (x2, y2))   # right
        dashed_line((x2, y2), (x1, y2))   # bottom
        dashed_line((x1, y2), (x1, y1))   # left

    def _draw_corners(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
    ) -> None:
        """Draw solid white corner brackets for visual clarity."""
        L = _CORNER_LEN
        T = _CORNER_THICK
        corners = [
            # top-left
            ((x1, y1 + L), (x1, y1), (x1 + L, y1)),
            # top-right
            ((x2 - L, y1), (x2, y1), (x2, y1 + L)),
            # bottom-right
            ((x2, y2 - L), (x2, y2), (x2 - L, y2)),
            # bottom-left
            ((x1 + L, y2), (x1, y2), (x1, y2 - L)),
        ]
        for arm1, corner, arm2 in corners:
            cv2.line(frame, arm1,   corner, _CORNER_COLOR, T, cv2.LINE_AA)
            cv2.line(frame, corner, arm2,   _CORNER_COLOR, T, cv2.LINE_AA)

    def _draw_label(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
    ) -> None:
        """Render a centred 'ERASE' label inside the bounding box."""
        label = "ERASE"
        (tw, th), _ = cv2.getTextSize(label, _LABEL_FONT, _LABEL_SCALE, _LABEL_THICK)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # White pill background for legibility
        pad = 6
        cv2.rectangle(
            frame,
            (cx - tw // 2 - pad, cy - th - pad),
            (cx + tw // 2 + pad, cy + pad),
            _LABEL_BG,
            cv2.FILLED,
        )
        cv2.putText(
            frame, label,
            (cx - tw // 2, cy),
            _LABEL_FONT, _LABEL_SCALE, _LABEL_COLOR, _LABEL_THICK, cv2.LINE_AA,
        )