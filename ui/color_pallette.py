"""
ui/color_palette.py
--------------------
Renders the colour-selection palette that appears when the user holds up
three fingers, and detects hover / selection events.

Layout
------
Colour swatches are arranged in a horizontal strip pinned to the top of the
frame.  Each swatch is a filled rounded rectangle with a thin white border.
The currently active colour is highlighted with a bright ring.

Hover detection
---------------
When the index-tip enters a swatch's hit-box, the colour is returned
immediately from ``hover()``.  There is intentionally no click or dwell
requirement — the gesture itself (3 fingers → palette mode) means the user
is in selection intent.

Design notes
------------
- Swatch geometry is computed once in ``__init__`` from ``config/colors.py``
  so resizing or reordering colours requires no changes here.
- Drawing uses OpenCV primitives only (no PIL dependency) for consistency
  with the rest of the pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.colors import COLOR_BGR, COLOR_ORDER, DEFAULT_COLOR
from config.settings import FRAME_WIDTH


# ── Layout constants ──────────────────────────────────────────────────────────

_SWATCH_H      = 60           # height of each swatch (px)
_SWATCH_W      = 80           # width of each swatch (px)
_SWATCH_Y      = 10           # top margin (px)
_SWATCH_RADIUS = 10           # corner radius for rounded rect
_BORDER_W      = 2            # default border thickness
_ACTIVE_RING   = 4            # active colour ring thickness
_LABEL_SCALE   = 0.45         # font scale for colour name label
_LABEL_COLOR   = (255, 255, 255)
_PANEL_ALPHA   = 0.55         # translucency of the dark panel behind swatches


# ── ColorPalette ──────────────────────────────────────────────────────────────

class ColorPalette:
    """
    On-screen colour picker rendered during the 3-finger palette gesture.

    Attributes
    ----------
    _swatches : List[Dict]
        Pre-computed swatch geometries and metadata.
    """

    def __init__(self) -> None:
        self._swatches: List[Dict] = self._build_swatches()

    # ── Public API ────────────────────────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        active_color: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Overlay the palette onto *frame* and return the annotated frame.

        Parameters
        ----------
        frame : np.ndarray
            The current composited BGR frame.
        active_color : Tuple[int, int, int]
            The currently selected ink colour (BGR) used to draw the
            active-colour ring around the matching swatch.
        """
        out = frame.copy()

        # Semi-transparent dark panel behind all swatches
        total_w = len(self._swatches) * (_SWATCH_W + 8) + 16
        panel = out[
            max(0, _SWATCH_Y - 8) : _SWATCH_Y + _SWATCH_H + 16,
            0:total_w,
        ].copy()
        dark = np.zeros_like(panel)
        cv2.addWeighted(dark, _PANEL_ALPHA, panel, 1 - _PANEL_ALPHA, 0, panel)
        out[
            max(0, _SWATCH_Y - 8) : _SWATCH_Y + _SWATCH_H + 16,
            0:total_w,
        ] = panel

        for sw in self._swatches:
            is_active = sw["bgr"] == active_color
            self._draw_swatch(out, sw, is_active)

        return out

    def hover(
        self,
        index_tip: Tuple[int, int],
    ) -> Optional[str]:
        """
        Test whether *index_tip* is inside any swatch hit-box.

        Returns the colour name string on a hit, ``None`` otherwise.
        """
        tx, ty = index_tip
        for sw in self._swatches:
            x1, y1, x2, y2 = sw["bbox"]
            if x1 <= tx <= x2 and y1 <= ty <= y2:
                return sw["name"]
        return None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_swatches(self) -> List[Dict]:
        """
        Pre-compute swatch positions from ``config/colors.py :: COLOR_ORDER``.
        """
        swatches = []
        gap = 8
        x_cursor = gap

        for name in COLOR_ORDER:
            bgr = COLOR_BGR[name]
            x1 = x_cursor
            y1 = _SWATCH_Y
            x2 = x1 + _SWATCH_W
            y2 = y1 + _SWATCH_H
            swatches.append({
                "name":  name,
                "bgr":   bgr,
                "bbox":  (x1, y1, x2, y2),
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
            })
            x_cursor += _SWATCH_W + gap

        return swatches

    def _draw_swatch(
        self,
        frame: np.ndarray,
        swatch: Dict,
        is_active: bool,
    ) -> None:
        """Draw a single swatch rectangle onto *frame* in-place."""
        x1, y1, x2, y2 = swatch["bbox"]
        bgr  = swatch["bgr"]
        name = swatch["name"]
        cx, cy = swatch["center"]

        # Filled body
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, cv2.FILLED)

        # Border
        border_color = (255, 255, 255) if is_active else (180, 180, 180)
        border_thick = _ACTIVE_RING if is_active else _BORDER_W
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thick)

        # Active indicator — bright dot at centre
        if is_active:
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), 7, bgr, 2)

        # Colour name label (small, centred below dot)
        label = name.capitalize()
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, _LABEL_SCALE, 1
        )
        lx = cx - tw // 2
        ly = y2 - 8
        cv2.putText(
            frame, label, (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX, _LABEL_SCALE,
            _LABEL_COLOR, 1, cv2.LINE_AA,
        )