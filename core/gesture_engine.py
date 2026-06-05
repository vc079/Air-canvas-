"""
core/gesture_engine.py
-----------------------
Translates a raw finger count + landmark snapshot into a named gesture
mode, applying debouncing and drag-anchor state management.

Responsibilities
----------------
- Map finger count (0–5) to a canonical mode name string.
- Debounce noisy transitions so a single flicker frame cannot switch modes.
- Track the drag anchor so the canvas renderer can compute relative deltas.
- Expose ``reset()`` for the multi-hand safety pause.

Design notes
------------
- The engine is intentionally *stateful*: it owns the debounce counter and
  drag anchor. Everything else (rendering, overlays) is stateless per frame.
- Mode names are string literals so callers use simple ``if gesture == "draw"``
  checks without importing an enum — keeping main.py readable.
- Drag mode stores the previous midpoint to compute (dx, dy) deltas, which
  the canvas renderer applies via a canvas translate.
"""

from __future__ import annotations

from typing import Optional, Tuple

from config.gestures import FINGER_TO_MODE, DEBOUNCE_FRAMES
from core.hand_tracker import LandmarkList


# ── GestureEngine ─────────────────────────────────────────────────────────────

class GestureEngine:
    """
    Stateful resolver: finger count → stable gesture mode.

    Attributes
    ----------
    current_mode : str
        The currently active, debounce-confirmed mode name.
    drag_anchor : Optional[Tuple[int, int]]
        The previous pinch midpoint, used by the renderer to compute the
        canvas drag delta. ``None`` when not in drag mode.
    """

    def __init__(self) -> None:
        self.current_mode: str = "idle"
        self.drag_anchor: Optional[Tuple[int, int]] = None

        # Pending mode and how many consecutive frames it has been seen
        self._pending_mode: str = "idle"
        self._pending_count: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve(
        self,
        finger_count: int,
        landmarks: LandmarkList,
    ) -> str:
        """
        Given the current *finger_count* and raw *landmarks*, return the
        stable gesture name for this frame.

        The returned string is one of:
            "idle"     — fist, no action
            "draw"     — index finger only
            "drag"     — two-finger pinch
            "palette"  — three fingers
            "snapshot" — four fingers
            "erase"    — open hand

        Side-effects
        ------------
        - Updates ``self.current_mode`` after debounce confirmation.
        - Manages ``self.drag_anchor`` in drag mode.
        """
        candidate = FINGER_TO_MODE.get(finger_count, "idle")
        self._advance_debounce(candidate)

        # Maintain drag anchor while in drag mode
        if self.current_mode == "drag":
            self._update_drag_anchor(landmarks)
        else:
            self.drag_anchor = None

        return self.current_mode

    def get_drag_delta(
        self,
        current_midpoint: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        Return the (dx, dy) offset between *current_midpoint* and the last
        recorded drag anchor.  Returns (0, 0) if no anchor is set.
        """
        if self.drag_anchor is None:
            return (0, 0)
        dx = current_midpoint[0] - self.drag_anchor[0]
        dy = current_midpoint[1] - self.drag_anchor[1]
        return (dx, dy)

    def reset(self) -> None:
        """
        Force an immediate reset to idle — called by the multi-hand guard.
        Clears debounce state and drag anchor.
        """
        self.current_mode = "idle"
        self._pending_mode = "idle"
        self._pending_count = 0
        self.drag_anchor = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _advance_debounce(self, candidate: str) -> None:
        """
        Increment the consecutive-frame counter for *candidate*.
        Confirm the mode only after ``DEBOUNCE_FRAMES`` consecutive frames.
        """
        if candidate == self._pending_mode:
            self._pending_count += 1
        else:
            # New candidate — restart the counter
            self._pending_mode = candidate
            self._pending_count = 1

        if self._pending_count >= DEBOUNCE_FRAMES:
            self.current_mode = self._pending_mode

    def _update_drag_anchor(self, landmarks: LandmarkList) -> None:
        """Store the current pinch midpoint as the new drag anchor."""
        if len(landmarks) < 13:
            return
        ix, iy = landmarks[8].x, landmarks[8].y   # index tip
        mx, my = landmarks[12].x, landmarks[12].y  # middle tip
        self.drag_anchor = ((ix + mx) // 2, (iy + my) // 2)

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GestureEngine(mode={self.current_mode!r}, "
            f"pending={self._pending_mode!r}×{self._pending_count})"
        )