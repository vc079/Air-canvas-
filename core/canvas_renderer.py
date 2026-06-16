"""
core/canvas_renderer.py
------------------------
Owns the persistent numpy canvas (the "ink layer") and all operations on it:
drawing strokes, erasing regions, dragging the entire canvas, and compositing
it onto a live webcam frame each tick.

Responsibilities
----------------
- Maintain a pitch-black BGR canvas the same size as the webcam frame.
- Draw anti-aliased, smoothed polylines for the draw gesture.
- Translate the entire canvas for the drag gesture.
- Erase a rectangular region for the open-hand wipe eraser.
- Composite the canvas onto a BGR frame using bitwise operations.

Design notes
------------
- Smoothing is delegated to ``utils/smoothing.py``; this class only decides
  *when* to flush a stroke.
- The composite step keeps ink "on top of" the video without alpha blending:
  wherever the canvas is non-black the video pixel is replaced.  This gives
  fully opaque, vibrant ink that reads well over any background.
- All coordinates are (int, int) tuples — no numpy indexing arithmetic leaks
  into callers.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from config.colors import DEFAULT_COLOR, COLOR_BGR
from config.settings import STROKE_THICKNESS
from utils.smoothing import SmoothingBuffer
from utils.compositing import composite_layers


# ── CanvasRenderer ────────────────────────────────────────────────────────────

class CanvasRenderer:
    """
    Manages the persistent ink layer and all drawing operations.

    Parameters
    ----------
    width, height : int
        Dimensions of the webcam frame (and therefore the canvas).
    """

    def __init__(self, width: int, height: int) -> None:
        self._w = width
        self._h = height

        # The ink layer: pitch-black BGR array.  Non-black pixels = ink.
        self.canvas: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)

        # Active drawing color (BGR tuple)
        self.color: Tuple[int, int, int] = COLOR_BGR[DEFAULT_COLOR]

        # Stroke state ─────────────────────────────────────────────
        # Previous tip position; None = start of a new stroke
        self._prev_point: Optional[Tuple[int, int]] = None
        # Smoothing buffer for the current stroke
        self._smoother = SmoothingBuffer(window=5)

        # Drag state ────────────────────────────────────────────────
        self._last_drag_anchor: Optional[Tuple[int, int]] = None

    # ── Color ─────────────────────────────────────────────────────────────────

    def set_color(self, color_name: str) -> None:
        """
        Switch the active ink colour by name.

        *color_name* must exist in ``config/colors.py :: COLOR_BGR``.
        Unknown names are silently ignored (preserving the current colour).
        """
        if color_name in COLOR_BGR:
            self.color = COLOR_BGR[color_name]
            self.reset_stroke()

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, tip: Tuple[int, int]) -> None:
        """
        Extend the current stroke to *tip*.

        A Catmull-Rom–style smoothed position is computed first, then a
        thick anti-aliased line is painted from the previous smoothed point
        to the new one.  If this is the first point of a stroke, nothing is
        drawn yet (we need two points to form a segment).
        """
        smooth_tip = self._smoother.push(tip)

        if self._prev_point is not None:
            cv2.line(
                self.canvas,
                self._prev_point,
                smooth_tip,
                self.color,
                thickness=STROKE_THICKNESS,
                lineType=cv2.LINE_AA,
            )

        self._prev_point = smooth_tip

    def reset_stroke(self) -> None:
        """
        Break the current stroke.  Call when switching modes or lifting the
        finger — prevents a straight "teleport" line on re-entry.
        """
        self._prev_point = None
        self._smoother.reset()

    # ── Drag ──────────────────────────────────────────────────────────────────

    def drag(self, midpoint: Tuple[int, int]) -> None:
        """
        Translate the entire canvas by the delta between *midpoint* and the
        previous drag anchor.

        On the first call of a new drag gesture the anchor is set and no
        translation occurs (avoids a snap jump).
        """
        if self._last_drag_anchor is None:
            self._last_drag_anchor = midpoint
            return

        dx = midpoint[0] - self._last_drag_anchor[0]
        dy = midpoint[1] - self._last_drag_anchor[1]
        self._last_drag_anchor = midpoint

        if dx == 0 and dy == 0:
            return

        # Build a 2×3 affine translation matrix and warp the canvas in-place
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        self.canvas = cv2.warpAffine(
            self.canvas,
            M,
            (self._w, self._h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def stop_drag(self) -> None:
        """Release the drag anchor — call when exiting drag mode."""
        self._last_drag_anchor = None

    # ── Erase ─────────────────────────────────────────────────────────────────

    def erase(self, bbox: Tuple[int, int, int, int]) -> None:
        """
        Zero-out (erase) a rectangular region of the canvas.

        *bbox* is (x1, y1, x2, y2) in pixel coordinates.
        """
        x1, y1, x2, y2 = bbox
        # Clamp to canvas bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(self._w - 1, x2); y2 = min(self._h - 1, y2)
        if x2 > x1 and y2 > y1:
            self.canvas[y1:y2, x1:x2] = 0

    def clear(self) -> None:
        """Wipe the entire canvas black."""
        self.canvas[:] = 0

    # ── Composite ─────────────────────────────────────────────────────────────

    def composite(self, frame: np.ndarray) -> np.ndarray:
        """
        Blend the ink canvas onto *frame* and return the result.

        Uses ``utils.compositing.composite_layers`` so the blending logic
        lives in one tested place.

        Parameters
        ----------
        frame : np.ndarray
            Live BGR webcam frame (already flipped).

        Returns
        -------
        np.ndarray
            The composited output frame (same shape as *frame*).
        """
        return composite_layers(frame, self.canvas)

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CanvasRenderer({self._w}×{self._h}, "
            f"color={self.color}, "
            f"stroke_active={self._prev_point is not None})"
        )