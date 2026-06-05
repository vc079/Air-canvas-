"""
tests/test_snapshot.py
-----------------------
Comprehensive unit tests for ``utils.snapshot.SnapshotManager``.

Test strategy
-------------
- No real files are written during tests.  ``cv2.imwrite`` and
  ``Path.mkdir`` are patched via ``unittest.mock`` so the suite runs
  in any CI environment without a filesystem side-effect.
- Timer behaviour (cooldown, flash) is exercised by calling ``tick()``
  the exact number of times dictated by the config constants, confirming
  the state machine transitions at the right frames.
- Edge cases covered: zero-size canvas, imwrite failure, repeated
  rapid-fire capture calls, repr string format.

Run with:
    pytest tests/test_snapshot.py -v
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make project root importable regardless of where pytest is invoked from.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.snapshot import SnapshotManager
from config.settings import SNAPSHOT_COOLDOWN_FRAMES, SNAPSHOT_FLASH_FRAMES


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def manager() -> SnapshotManager:
    """Fresh SnapshotManager instance for each test."""
    return SnapshotManager()


@pytest.fixture()
def blank_canvas() -> np.ndarray:
    """A realistic all-black 720p BGR canvas (no ink)."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture()
def inked_canvas() -> np.ndarray:
    """A canvas with a visible white stroke — simulates real usage."""
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas[100:110, 200:600] = (255, 255, 255)   # horizontal white line
    return canvas


# ── Helper ───────────────────────────────────────────────────────────────────

def _do_capture(manager: SnapshotManager, canvas: np.ndarray) -> bool:
    """Patch away I/O and attempt a capture. Returns capture() result."""
    with patch("cv2.imwrite", return_value=True), \
         patch("pathlib.Path.mkdir"):
        return manager.capture(canvas)


# ── Initial state ─────────────────────────────────────────────────────────────

class TestInitialState:
    def test_ready_on_construction(self, manager):
        assert manager.ready is True

    def test_flash_inactive_on_construction(self, manager):
        assert manager.flash_active is False

    def test_last_save_path_is_none_on_construction(self, manager):
        assert manager.last_save_path is None

    def test_cooldown_private_counter_zero(self, manager):
        assert manager._cooldown_remaining == 0

    def test_flash_private_counter_zero(self, manager):
        assert manager._flash_remaining == 0


# ── Successful capture ────────────────────────────────────────────────────────

class TestSuccessfulCapture:
    def test_capture_returns_true_on_success(self, manager, blank_canvas):
        result = _do_capture(manager, blank_canvas)
        assert result is True

    def test_capture_sets_last_save_path(self, manager, blank_canvas):
        _do_capture(manager, blank_canvas)
        assert manager.last_save_path is not None

    def test_last_save_path_is_png(self, manager, blank_canvas):
        _do_capture(manager, blank_canvas)
        assert manager.last_save_path.endswith(".png")

    def test_last_save_path_contains_snapshot_prefix(self, manager, blank_canvas):
        _do_capture(manager, blank_canvas)
        filename = Path(manager.last_save_path).name
        assert filename.startswith("snapshot_")

    def test_flash_active_immediately_after_capture(self, manager, blank_canvas):
        _do_capture(manager, blank_canvas)
        assert manager.flash_active is True

    def test_not_ready_immediately_after_capture(self, manager, blank_canvas):
        _do_capture(manager, blank_canvas)
        assert manager.ready is False

    def test_cooldown_set_to_constant_after_capture(self, manager, blank_canvas):
        _do_capture(manager, blank_canvas)
        assert manager._cooldown_remaining == SNAPSHOT_COOLDOWN_FRAMES

    def test_flash_counter_set_to_constant_after_capture(self, manager, blank_canvas):
        _do_capture(manager, blank_canvas)
        assert manager._flash_remaining == SNAPSHOT_FLASH_FRAMES

    def test_inked_canvas_captures_successfully(self, manager, inked_canvas):
        result = _do_capture(manager, inked_canvas)
        assert result is True

    def test_imwrite_called_with_correct_canvas(self, manager, inked_canvas):
        with patch("cv2.imwrite", return_value=True) as mock_write, \
             patch("pathlib.Path.mkdir"):
            manager.capture(inked_canvas)
            args, _ = mock_write.call_args
            # Second argument to imwrite is the array
            assert np.array_equal(args[1], inked_canvas)

    def test_imwrite_called_with_png_path(self, manager, blank_canvas):
        with patch("cv2.imwrite", return_value=True) as mock_write, \
             patch("pathlib.Path.mkdir"):
            manager.capture(blank_canvas)
            path_arg = mock_write.call_args[0][0]
            assert path_arg.endswith(".png")

    def test_mkdir_called_with_parents_exist_ok(self, manager, blank_canvas):
        with patch("cv2.imwrite", return_value=True), \
             patch("pathlib.Path.mkdir") as mock_mkdir:
            manager.capture(blank_canvas)
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


# ── Cooldown enforcement ──────────────────────────────────────────────────────

class TestCooldown:
    def test_second_capture_blocked_immediately(self, manager, blank_canvas):
        _do_capture(manager, blank_canvas)
        result = _do_capture(manager, blank_canvas)
        assert result is False

    def test_capture_blocked_one_tick_before_cooldown_expires(
        self, manager, blank_canvas
    ):
        _do_capture(manager, blank_canvas)
        for _ in range(SNAPSHOT_COOLDOWN_FRAMES - 1):
            manager.tick()
        assert manager.ready is False
        result = _do_capture(manager, blank_canvas)
        assert result is False

    def test_capture_allowed_after_full_cooldown(self, manager, blank_canvas):
        _do_capture(manager, blank_canvas)
        for _ in range(SNAPSHOT_COOLDOWN_FRAMES):
            manager.tick()
        assert manager.ready is True
        result = _do_capture(manager, blank_canvas)
        assert result is True

    def test_ready_flag_false_during_cooldown(self, manager, blank_canvas):
        _do_capture(manager, blank_canvas)
        for i in range(SNAPSHOT_COOLDOWN_FRAMES - 1):
            manager.tick()
            assert manager.ready is False, f"Should still be on cooldown at tick {i+1}"

    def test_ready_flag_true_at_exact_cooldown_boundary(self, manager, blank_canvas):
        _do_capture(manager, blank_canvas)
        for _ in range(SNAPSHOT_COOLDOWN_FRAMES):
            manager.tick()
        assert manager.ready is True

    def test_last_save_path_unchanged_when_blocked(self, manager, blank_canvas):
        _do_capture(manager, blank_canvas)
        first_path = manager.last_save_path
        # Attempt blocked capture
        with patch("cv2.imwrite", return_value=True), \
             patch("pathlib.Path.mkdir"):
            manager.capture(blank_canvas)
        assert manager.last_save_path == first_path


# ── Flash timer ───────────────────────────────────────────────────────────────

class TestFlashTimer:
    def test_flash_active_for_all_flash_frames(self, manager, blank_canvas):
        _do_capture(manager, blank_canvas)
        for i in range(SNAPSHOT_FLASH_FRAMES):
            assert manager.flash_active is True, \
                f"Flash should be active at tick {i} (before tick() call)"
            manager.tick()

    def test_flash_inactive_after_flash_frames_expire(self, manager, blank_canvas):
        _do_capture(manager, blank_canvas)
        for _ in range(SNAPSHOT_FLASH_FRAMES):
            manager.tick()
        assert manager.flash_active is False

    def test_flash_inactive_one_tick_after_expiry(self, manager, blank_canvas):
        _do_capture(manager, blank_canvas)
        for _ in range(SNAPSHOT_FLASH_FRAMES + 1):
            manager.tick()
        assert manager.flash_active is False

    def test_flash_shorter_than_cooldown(self):
        """Flash must expire before cooldown so the HUD clears before re-enable."""
        assert SNAPSHOT_FLASH_FRAMES < SNAPSHOT_COOLDOWN_FRAMES


# ── tick() boundary behaviour ─────────────────────────────────────────────────

class TestTick:
    def test_tick_does_not_go_below_zero_cooldown(self, manager):
        """Calling tick() on a fresh manager must not cause negative counters."""
        for _ in range(10):
            manager.tick()
        assert manager._cooldown_remaining == 0

    def test_tick_does_not_go_below_zero_flash(self, manager):
        for _ in range(10):
            manager.tick()
        assert manager._flash_remaining == 0

    def test_tick_decrements_cooldown_by_one(self, manager, blank_canvas):
        _do_capture(manager, blank_canvas)
        before = manager._cooldown_remaining
        manager.tick()
        assert manager._cooldown_remaining == before - 1

    def test_tick_decrements_flash_by_one(self, manager, blank_canvas):
        _do_capture(manager, blank_canvas)
        before = manager._flash_remaining
        manager.tick()
        assert manager._flash_remaining == before - 1


# ── imwrite failure handling ──────────────────────────────────────────────────

class TestImwriteFailure:
    def test_capture_returns_false_when_imwrite_fails(self, manager, blank_canvas):
        with patch("cv2.imwrite", return_value=False), \
             patch("pathlib.Path.mkdir"):
            result = manager.capture(blank_canvas)
        assert result is False

    def test_last_save_path_unchanged_on_imwrite_failure(self, manager, blank_canvas):
        with patch("cv2.imwrite", return_value=False), \
             patch("pathlib.Path.mkdir"):
            manager.capture(blank_canvas)
        assert manager.last_save_path is None

    def test_flash_not_activated_on_imwrite_failure(self, manager, blank_canvas):
        with patch("cv2.imwrite", return_value=False), \
             patch("pathlib.Path.mkdir"):
            manager.capture(blank_canvas)
        assert manager.flash_active is False

    def test_cooldown_not_started_on_imwrite_failure(self, manager, blank_canvas):
        with patch("cv2.imwrite", return_value=False), \
             patch("pathlib.Path.mkdir"):
            manager.capture(blank_canvas)
        assert manager.ready is True

    def test_capture_retried_after_imwrite_failure(self, manager, blank_canvas):
        """After a failed write the manager must still be ready for a retry."""
        with patch("cv2.imwrite", return_value=False), \
             patch("pathlib.Path.mkdir"):
            manager.capture(blank_canvas)

        # Now let imwrite succeed
        result = _do_capture(manager, blank_canvas)
        assert result is True


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_size_canvas_does_not_raise(self, manager):
        """A degenerate empty canvas should not crash capture()."""
        tiny = np.zeros((0, 0, 3), dtype=np.uint8)
        with patch("cv2.imwrite", return_value=True), \
             patch("pathlib.Path.mkdir"):
            try:
                manager.capture(tiny)
            except Exception as exc:
                pytest.fail(f"capture() raised unexpectedly: {exc}")

    def test_single_pixel_canvas(self, manager):
        one_px = np.zeros((1, 1, 3), dtype=np.uint8)
        with patch("cv2.imwrite", return_value=True), \
             patch("pathlib.Path.mkdir"):
            result = manager.capture(one_px)
        assert result is True

    def test_multiple_sequential_captures_after_cooldown(
        self, manager, blank_canvas
    ):
        """Three full capture+cooldown cycles must all succeed."""
        for cycle in range(3):
            result = _do_capture(manager, blank_canvas)
            assert result is True, f"Cycle {cycle}: expected capture to succeed"
            for _ in range(SNAPSHOT_COOLDOWN_FRAMES):
                manager.tick()

    def test_repr_contains_ready_field(self, manager):
        assert "ready=" in repr(manager)

    def test_repr_contains_cooldown_field(self, manager):
        assert "cooldown=" in repr(manager)

    def test_repr_contains_flash_field(self, manager):
        assert "flash=" in repr(manager)

    def test_repr_is_string(self, manager):
        assert isinstance(repr(manager), str)