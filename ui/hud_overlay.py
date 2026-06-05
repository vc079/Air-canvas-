"""
ui/hud_overlay.py
------------------
Heads-up display (HUD) rendered on every frame regardless of gesture mode.

Elements
--------
1. **Mode badge** — bottom-left pill showing the current gesture mode name
   and its associated icon character.
2. **Colour swatch** — small filled circle next to the mode badge showing the
   active ink colour.
3. **Snapshot flash** — full-frame white vignette that fires for a few frames
   after a snapshot is saved, simulating a camera shutter.
4. **Drag handle** — translucent green circle rendered at the pinch midpoint
   during drag mode (drawn by ``draw_drag_handle``; called separately from
   main.py so the handle is only shown when landmarks are live).

Design notes
------------
- All drawing is done on a *copy* of the frame passed in; the original is
  never mutated.
- The mode badge uses a filled rounded-rectangle background so it remains
  legible over both dark and bright video regions.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from config.settings import FRAME_WIDTH, FRAME_HEIGHT


# ── HUD constants ─────────────────────────────────────────────────────────────

_MODE_ICONS: dict[str, str] = {
    "idle":     "✋",   # rendered as text fallback below (OpenCV has no emoji)
    "draw":     "[D]",
    "drag":     "[M]",
    "palette":  "[C]",
    "snapshot": "[S]",
    "erase":    "[E]",
}

_BADGE_BG        = (30, 30, 30)          # dark fill for mode pill
_BADGE_ALPHA     = 0.70                  # translucency
_BADGE_PAD_X     = 16
_BADGE_PAD_Y     = 10
_BADGE_RADIUS    = 12
_FONT            = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE      = 0.65
_FONT_THICK      = 2
_TEXT_COLOR      = (230, 230, 230)
_COLOR_DOT_R     = 10                    # colour indicator dot radius

_FLASH_COLOR     = (255, 255, 255)
_FLASH_ALPHA_MAX = 0.55

_DRAG_HANDLE_R   = 18
_DRAG_HANDLE_COL = (50, 220, 80)         # bright green
_DRAG_HANDLE_ALPHA = 0.55


# ── HUDOverlay ────────────────────────────────────────────────────────────────

class HUDOverlay:
    """
    Composites all persistent HUD elements onto each output frame.
    """

    def draw(
        self,
        frame: np.ndarray,
        mode: str,
        active_color: Tuple[int, int, int],
        snapshot_flash: bool = False,
    ) -> np.ndarray:
        """
        Draw the HUD onto *frame* and return the annotated copy.

        Parameters
        ----------
        frame : np.ndarray
            The composited BGR output frame for this tick.
        mode : str
            Current gesture mode name (e.g. ``"draw"``, ``"idle"``).
        active_color : Tuple[int, int, int]
            BGR tuple of the currently selected ink colour.
        snapshot_flash : bool
            When ``True``, render the white shutter-flash vignette.
        """
        out = frame.copy()

        if snapshot_flash:
            out = self._draw_flash(out)

        out = self._draw_mode_badge(out, mode, active_color)
        return out

    def draw_drag_handle(
        self,
        frame: np.ndarray,
        midpoint: Tuple[int, int],
    ) -> np.ndarray:
        """
        Draw a translucent green circle at *midpoint*.

        Called separately (and only when landmarks are live) so the handle
        disappears naturally when the hand leaves the frame.
        """
        out = frame.copy()
        overlay = out.copy()
        cv2.circle(overlay, midpoint, _DRAG_HANDLE_R, _DRAG_HANDLE_COL, cv2.FILLED)
        cv2.addWeighted(overlay, _DRAG_HANDLE_ALPHA, out, 1 - _DRAG_HANDLE_ALPHA, 0, out)
        # Crisp outer ring
        cv2.circle(out, midpoint, _DRAG_HANDLE_R, _DRAG_HANDLE_COL, 2, cv2.LINE_AA)
        return out

    # ── Private helpers ───────────────────────────────────────────────────────

    def _draw_mode_badge(
        self,
        frame: np.ndarray,
        mode: str,
        active_color: Tuple[int, int, int],
    ) -> np.ndarray:
        """Render the mode pill and colour dot in the bottom-left corner."""
        h, w = frame.shape[:2]
        icon  = _MODE_ICONS.get(mode, "[?]")
        label = f"{icon}  {mode.upper()}"

        (tw, th), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICK)

        # Badge rectangle dimensions
        dot_space = _COLOR_DOT_R * 2 + 10   # space reserved for colour dot
        bw = tw + _BADGE_PAD_X * 2 + dot_space
        bh = th + baseline + _BADGE_PAD_Y * 2

        # Pin to bottom-left with a small margin
        margin = 16
        x1 = margin
        y1 = h - bh - margin
        x2 = x1 + bw
        y2 = y1 + bh

        # Semi-transparent dark pill
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), _BADGE_BG, cv2.FILLED)
        cv2.addWeighted(overlay, _BADGE_ALPHA, frame, 1 - _BADGE_ALPHA, 0, frame)

        # Mode text
        tx = x1 + _BADGE_PAD_X
        ty = y2 - _BADGE_PAD_Y - baseline
        cv2.putText(frame, label, (tx, ty), _FONT, _FONT_SCALE, _TEXT_COLOR, _FONT_THICK, cv2.LINE_AA)

        # Colour dot (right side of badge)
        dot_cx = x2 - _BADGE_PAD_X - _COLOR_DOT_R
        dot_cy = (y1 + y2) // 2
        cv2.circle(frame, (dot_cx, dot_cy), _COLOR_DOT_R, active_color, cv2.FILLED)
        cv2.circle(frame, (dot_cx, dot_cy), _COLOR_DOT_R, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def _draw_flash(self, frame: np.ndarray) -> np.ndarray:
        """Blend a white vignette over the frame for the snapshot flash."""
        flash = np.full_like(frame, 255)
        cv2.addWeighted(flash, _FLASH_ALPHA_MAX, frame, 1 - _FLASH_ALPHA_MAX, 0, frame)
        return frame