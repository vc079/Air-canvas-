"""
utils/snapshot.py
------------------
Handles saving the canvas as a timestamped PNG file when the user holds
up four fingers, with a built-in cooldown to prevent accidental rapid-fire
saves and a short flash signal for the HUD to display.

Responsibilities
----------------
- Write the canvas numpy array to ``assets/snapshots/`` as a PNG file.
- Enforce a configurable cooldown period between saves.
- Expose a ``flash_active`` flag so the HUD can render the shutter effect.

File naming
-----------
Snapshots are saved as::

    assets/snapshots/snapshot_YYYYMMDD_HHMMSS.png

Using a timestamp guarantees uniqueness without requiring a counter file
or database.

Design notes
------------
- ``capture()`` is a no-op if the cooldown hasn't expired, making it safe
  to call on every frame while 4 fingers are raised.
- The flash lasts ``FLASH_FRAMES`` frames (default 6 ≈ 100 ms @ 60 fps),
  which is long enough to be visible but short enough to feel snappy.
- Directory creation is handled automatically on first save.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config.settings import SNAPSHOT_DIR, SNAPSHOT_COOLDOWN_FRAMES, SNAPSHOT_FLASH_FRAMES


class SnapshotManager:
    """
    Manages on-demand canvas saves with cooldown and HUD flash signalling.

    Attributes
    ----------
    flash_active : bool
        ``True`` for ``SNAPSHOT_FLASH_FRAMES`` frames after a successful
        save.  Consumed by ``HUDOverlay.draw()``.
    last_save_path : str | None
        Absolute path of the most recently saved file, or ``None``.
    """

    def __init__(self) -> None:
        self._cooldown_remaining: int = 0
        self._flash_remaining: int = 0
        self.last_save_path: str | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def flash_active(self) -> bool:
        """``True`` while the HUD shutter flash should be displayed."""
        return self._flash_remaining > 0

    @property
    def ready(self) -> bool:
        """``True`` if the cooldown has expired and a save can proceed."""
        return self._cooldown_remaining <= 0

    def capture(self, canvas: np.ndarray) -> bool:
        """
        Attempt to save *canvas* as a PNG file.

        Parameters
        ----------
        canvas : np.ndarray
            The ink-layer BGR array from ``CanvasRenderer``.

        Returns
        -------
        bool
            ``True`` if the file was saved, ``False`` if on cooldown.
        """
        if not self.ready:
            return False

        path = self._build_path()
        self._ensure_dir(path)

        success = cv2.imwrite(str(path), canvas)
        if success:
            self.last_save_path = str(path)
            self._cooldown_remaining = SNAPSHOT_COOLDOWN_FRAMES
            self._flash_remaining = SNAPSHOT_FLASH_FRAMES
            print(f"[Snapshot] Saved → {path}")
        else:
            print(f"[Snapshot] ERROR: failed to write {path}")

        return success

    def tick(self) -> None:
        """
        Decrement the cooldown and flash timers by one frame.

        Must be called exactly once per frame in the main loop, *after*
        ``HUDOverlay.draw()`` has consumed ``flash_active``.
        """
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
        if self._flash_remaining > 0:
            self._flash_remaining -= 1

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_path() -> Path:
        """Return a unique timestamped output path."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(SNAPSHOT_DIR) / f"snapshot_{ts}.png"

    @staticmethod
    def _ensure_dir(path: Path) -> None:
        """Create the snapshot directory if it doesn't already exist."""
        path.parent.mkdir(parents=True, exist_ok=True)

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SnapshotManager("
            f"ready={self.ready}, "
            f"cooldown={self._cooldown_remaining}, "
            f"flash={self._flash_remaining})"
        )