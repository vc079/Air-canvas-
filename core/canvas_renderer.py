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

from typing import Deque, List, Optional, Tuple
from collections import deque

import cv2
import numpy as np

from config.colors import DEFAULT_COLOR, COLOR_BGR
from config.settings import STROKE_THICKNESS
from utils.smoothing import SmoothingBuffer
from utils.compositing import composite_layers


# Maximum number of canvas states kept on the undo stack. Each entry is a
# full copy of the canvas array, so this caps memory use — at 1280×720×3
# bytes (~2.7MB) per entry, 20 entries is ~55MB, a reasonable ceiling for
# a desktop app. Older states are dropped once the cap is exceeded.
_MAX_HISTORY = 20


# ── CanvasRenderer ────────────────────────────────────────────────────────────

class CanvasRenderer:
    """
    Manages the persistent ink layer and all drawing operations.

    Undo / redo
    -----------
    Undo/redo operates on whole-canvas snapshots rather than individual
    pixels or line segments. ``push_history()`` must be called by the
    caller (``main.py``) at the boundary of each discrete user action —
    i.e. immediately *before* a new stroke/erase/drag/clear begins, so
    the pushed state represents "the canvas as it looked right before
    this action." ``undo()`` then restores that prior state, and the
    canvas as it looked *before* the undo* is pushed onto the redo stack
    so ``redo()`` can restore it again.

    A snapshot-per-action design (rather than per-pixel or per-frame) is
    intentional: drawing is a continuous stream of small line segments,
    and undoing one segment at a time would feel meaningless to the user
    — pressing "z" should erase the whole last stroke, not one tiny
    fragment of it.

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
        self._smoother = SmoothingBuffer()

        # Drag state ────────────────────────────────────────────────
        self._last_drag_anchor: Optional[Tuple[int, int]] = None

        # Undo / redo history ──────────────────────────────────────
        # Bounded deques of full canvas snapshots (numpy arrays).
        self._undo_stack: Deque[np.ndarray] = deque(maxlen=_MAX_HISTORY)
        self._redo_stack: Deque[np.ndarray] = deque(maxlen=_MAX_HISTORY)
        # Tracks whether a snapshot has already been pushed for the
        # in-progress action, so a single stroke/erase/drag doesn't
        # push a new history entry on every frame it spans.
        self._action_open: bool = False

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

    # ── Undo / redo ───────────────────────────────────────────────────────────

    def push_history(self) -> None:
        """
        Snapshot the current canvas onto the undo stack, marking the start
        of a new discrete action (stroke, erase, drag, or clear).

        Safe to call once per frame while a multi-frame action is in
        progress — internally guarded by ``_action_open`` so only the
        *first* call for a given action actually pushes a snapshot; later
        calls within the same continuous action are no-ops. The action is
        considered closed (and a future call will push again) once
        ``close_action()`` is called.

        Any time a new history snapshot is successfully pushed, the redo
        stack is cleared — the standard undo/redo convention: once the
        user does something new, the previously-undone "future" is no
        longer reachable.
        """
        if self._action_open:
            return
        self._undo_stack.append(self.canvas.copy())
        self._redo_stack.clear()
        self._action_open = True

    def close_action(self) -> None:
        """
        Mark the current action as finished, so the next ``push_history()``
        call starts a fresh snapshot rather than being a no-op.

        Call this whenever a gesture ends (e.g. on the frame the mode
        changes away from "draw"/"erase"/"drag", or right after a
        single-shot action like ``clear()`` completes).
        """
        self._action_open = False

    def undo(self) -> bool:
        """
        Revert the canvas to the state before the most recent action.

        Returns
        -------
        bool
            ``True`` if an undo was performed, ``False`` if the undo
            stack was empty (nothing to undo).
        """
        if not self._undo_stack:
            return False
        self._redo_stack.append(self.canvas.copy())
        self.canvas = self._undo_stack.pop()
        self.reset_stroke()
        self.stop_drag()
        self._action_open = False
        return True

    def redo(self) -> bool:
        """
        Re-apply the most recently undone action.

        Returns
        -------
        bool
            ``True`` if a redo was performed, ``False`` if the redo
            stack was empty (nothing to redo).
        """
        if not self._redo_stack:
            return False
        self._undo_stack.append(self.canvas.copy())
        self.canvas = self._redo_stack.pop()
        self.reset_stroke()
        self.stop_drag()
        self._action_open = False
        return True

    def can_undo(self) -> bool:
        """``True`` if there is at least one action available to undo."""
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        """``True`` if there is at least one undone action available to redo."""
        return len(self._redo_stack) > 0

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