"""
tests/test_compositing.py
--------------------------
Comprehensive unit tests for ``utils.compositing.composite_layers``.

What is being tested
--------------------
The compositing pipeline takes a live BGR webcam frame and a pitch-black
ink canvas, and returns a single frame where every non-black canvas pixel
replaces the corresponding frame pixel — fully opaque, no alpha blending.

The six logical steps inside the function give us clear test targets:

    Step 1-2  mask creation        → ink pixels detected correctly
    Step 3    mask inversion       → background pixels preserved correctly
    Step 4-5  bitwise AND          → no cross-contamination between layers
    Step 6    bitwise OR merge     → correct final pixel values

Test classes
------------
    TestOutputContract         shape, dtype, new array guarantees
    TestBlankCanvas            black canvas  → frame returned unchanged
    TestFullCanvas             all-white canvas → all pixels replaced
    TestWhiteInkOnGrayBg       white stroke pixels land correctly
    TestColoredInk             colored ink (red, green, blue) transfers exactly
    TestBackgroundPreservation  pixels outside ink regions stay untouched
    TestInkReplacement          ink pixels fully overwrite the frame
    TestInputImmutability       neither input array is mutated
    TestShapeMismatch           mismatched shapes raise ValueError
    TestEdgeCases               1×1, non-square, near-black threshold

Run with:
    pytest tests/test_compositing.py -v
"""

from __future__ import annotations

import sys
import os

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make project root importable regardless of where pytest is invoked from.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.compositing import composite_layers


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def dims():
    """Standard test frame dimensions."""
    return 480, 640   # height, width


@pytest.fixture()
def gray_frame(dims):
    """Solid mid-gray BGR frame — easy to distinguish from ink."""
    h, w = dims
    return np.ones((h, w, 3), dtype=np.uint8) * 128


@pytest.fixture()
def blank_canvas(dims):
    """All-black canvas with no ink."""
    h, w = dims
    return np.zeros((h, w, 3), dtype=np.uint8)


@pytest.fixture()
def white_line_canvas(dims):
    """Canvas with a single white diagonal line from (10,10) → (100,100)."""
    h, w = dims
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.line(canvas, (10, 10), (100, 100), (255, 255, 255), thickness=5)
    return canvas


@pytest.fixture()
def red_rect_canvas(dims):
    """Canvas with a filled red rectangle at a known position."""
    h, w = dims
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    # BGR: red = (0, 0, 220)
    cv2.rectangle(canvas, (200, 150), (400, 300), (0, 0, 220), cv2.FILLED)
    return canvas


@pytest.fixture()
def full_white_canvas(dims):
    """Canvas where every pixel is white — total replacement."""
    h, w = dims
    return np.ones((h, w, 3), dtype=np.uint8) * 255


# ── TestOutputContract ────────────────────────────────────────────────────────

class TestOutputContract:
    """The function must return a correctly typed, correctly shaped new array."""

    def test_output_shape_matches_frame(self, gray_frame, white_line_canvas):
        out = composite_layers(gray_frame, white_line_canvas)
        assert out.shape == gray_frame.shape

    def test_output_dtype_is_uint8(self, gray_frame, white_line_canvas):
        out = composite_layers(gray_frame, white_line_canvas)
        assert out.dtype == np.uint8

    def test_output_is_new_array_not_frame(self, gray_frame, white_line_canvas):
        out = composite_layers(gray_frame, white_line_canvas)
        assert out is not gray_frame

    def test_output_is_new_array_not_canvas(self, gray_frame, white_line_canvas):
        out = composite_layers(gray_frame, white_line_canvas)
        assert out is not white_line_canvas

    def test_output_has_three_channels(self, gray_frame, blank_canvas):
        out = composite_layers(gray_frame, blank_canvas)
        assert out.ndim == 3
        assert out.shape[2] == 3


# ── TestBlankCanvas ───────────────────────────────────────────────────────────

class TestBlankCanvas:
    """A pitch-black canvas (no ink) must leave every frame pixel unchanged."""

    def test_blank_canvas_output_equals_frame(self, gray_frame, blank_canvas):
        out = composite_layers(gray_frame, blank_canvas)
        assert np.array_equal(out, gray_frame)

    def test_blank_canvas_no_pixel_altered(self, gray_frame, blank_canvas):
        out = composite_layers(gray_frame, blank_canvas)
        diff = np.sum(out != gray_frame)
        assert diff == 0

    def test_blank_canvas_with_black_frame(self, blank_canvas):
        """Both layers black → output is all black."""
        black_frame = np.zeros_like(blank_canvas)
        out = composite_layers(black_frame, blank_canvas)
        assert np.array_equal(out, black_frame)

    def test_blank_canvas_with_white_frame(self, dims):
        """White frame + blank canvas → white output."""
        h, w = dims
        white_frame = np.ones((h, w, 3), dtype=np.uint8) * 255
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        out = composite_layers(white_frame, blank)
        assert np.array_equal(out, white_frame)


# ── TestFullCanvas ────────────────────────────────────────────────────────────

class TestFullCanvas:
    """A fully-white canvas replaces every pixel regardless of the frame."""

    def test_full_white_canvas_output_is_all_white(self, gray_frame, full_white_canvas):
        out = composite_layers(gray_frame, full_white_canvas)
        assert np.all(out == 255)

    def test_full_white_canvas_no_gray_pixels_remain(self, gray_frame, full_white_canvas):
        out = composite_layers(gray_frame, full_white_canvas)
        gray_mask = np.all(out == 128, axis=2)
        assert not np.any(gray_mask)

    def test_full_white_canvas_all_pixels_replaced(self, gray_frame, full_white_canvas):
        out = composite_layers(gray_frame, full_white_canvas)
        assert np.array_equal(out, full_white_canvas)


# ── TestWhiteInkOnGrayBg ──────────────────────────────────────────────────────

class TestWhiteInkOnGrayBg:
    """White diagonal line on a gray background — the canonical use case."""

    def test_on_line_pixel_is_white(self, gray_frame, white_line_canvas):
        """The centre of the drawn line (50, 50) must be white in the output."""
        out = composite_layers(gray_frame, white_line_canvas)
        assert np.array_equal(out[50, 50], [255, 255, 255])

    def test_on_line_pixel_is_not_gray(self, gray_frame, white_line_canvas):
        out = composite_layers(gray_frame, white_line_canvas)
        assert not np.array_equal(out[50, 50], [128, 128, 128])

    def test_off_line_pixel_remains_gray(self, gray_frame, white_line_canvas):
        """A pixel far from the stroke (300, 500) must stay gray."""
        out = composite_layers(gray_frame, white_line_canvas)
        assert np.array_equal(out[300, 500], [128, 128, 128])

    def test_multiple_on_line_pixels_are_white(self, gray_frame, white_line_canvas):
        """Several sampled points along the line must all be white."""
        out = composite_layers(gray_frame, white_line_canvas)
        for row, col in [(15, 15), (30, 30), (50, 50), (70, 70), (90, 90)]:
            assert np.array_equal(out[row, col], [255, 255, 255]), \
                f"Expected white at ({row},{col}), got {out[row, col]}"

    def test_corner_pixel_top_left_is_gray(self, gray_frame, white_line_canvas):
        """Top-left corner is not on the line — must be unchanged."""
        out = composite_layers(gray_frame, white_line_canvas)
        assert np.array_equal(out[0, 0], [128, 128, 128])

    def test_corner_pixel_bottom_right_is_gray(self, gray_frame, white_line_canvas):
        out = composite_layers(gray_frame, white_line_canvas)
        assert np.array_equal(out[479, 639], [128, 128, 128])


# ── TestColoredInk ────────────────────────────────────────────────────────────

class TestColoredInk:
    """Colored ink must transfer with exact BGR values, not blended."""

    def test_red_ink_pixel_value(self, gray_frame, red_rect_canvas):
        out = composite_layers(gray_frame, red_rect_canvas)
        # Centre of the red rectangle
        assert np.array_equal(out[225, 300], [0, 0, 220])

    def test_red_ink_is_not_blended_with_gray(self, gray_frame, red_rect_canvas):
        out = composite_layers(gray_frame, red_rect_canvas)
        pixel = out[225, 300]
        # If alpha-blended it would be something like (64, 64, 192); not here
        assert np.array_equal(pixel, [0, 0, 220])

    def test_green_ink_transfers_exactly(self, gray_frame, dims):
        h, w = dims
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (50, 50), (150, 150), (0, 200, 0), cv2.FILLED)
        out = composite_layers(gray_frame, canvas)
        assert np.array_equal(out[100, 100], [0, 200, 0])

    def test_blue_ink_transfers_exactly(self, gray_frame, dims):
        h, w = dims
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (50, 50), (150, 150), (220, 0, 0), cv2.FILLED)
        out = composite_layers(gray_frame, canvas)
        assert np.array_equal(out[100, 100], [220, 0, 0])

    def test_multiple_colors_on_same_canvas(self, gray_frame, dims):
        """Two non-overlapping colored regions both transfer correctly."""
        h, w = dims
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (50,  50),  (150, 150), (0, 0, 200),   cv2.FILLED)  # red
        cv2.rectangle(canvas, (300, 200), (450, 350), (0, 200, 0),   cv2.FILLED)  # green
        out = composite_layers(gray_frame, canvas)
        assert np.array_equal(out[100, 100],  [0, 0,   200])
        assert np.array_equal(out[275, 375],  [0, 200, 0  ])


# ── TestBackgroundPreservation ────────────────────────────────────────────────

class TestBackgroundPreservation:
    """Every pixel that has no ink above it must be identical to the frame."""

    def test_background_region_outside_rect_unchanged(self, gray_frame, red_rect_canvas):
        out = composite_layers(gray_frame, red_rect_canvas)
        # Well outside the red rectangle
        assert np.array_equal(out[10, 10], gray_frame[10, 10])

    def test_all_non_ink_pixels_unchanged(self, gray_frame, red_rect_canvas):
        """Build a mask of canvas non-ink pixels; all must match the frame."""
        out = composite_layers(gray_frame, red_rect_canvas)
        gray_canvas = cv2.cvtColor(red_rect_canvas, cv2.COLOR_BGR2GRAY)
        _, ink_mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
        bg_mask = cv2.bitwise_not(ink_mask).astype(bool)
        assert np.array_equal(out[bg_mask], gray_frame[bg_mask])

    def test_ink_region_differs_from_frame(self, gray_frame, red_rect_canvas):
        """Sanity-check: ink pixels must differ from the gray background."""
        out = composite_layers(gray_frame, red_rect_canvas)
        gray_canvas = cv2.cvtColor(red_rect_canvas, cv2.COLOR_BGR2GRAY)
        _, ink_mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
        ink_bool = ink_mask.astype(bool)
        assert not np.array_equal(out[ink_bool], gray_frame[ink_bool])


# ── TestInkReplacement ────────────────────────────────────────────────────────

class TestInkReplacement:
    """Ink pixels must fully overwrite frame pixels with no blending."""

    def test_ink_pixel_matches_canvas_not_frame(self, dims):
        """On an ink pixel the output must equal canvas, not frame, not a mix."""
        h, w = dims
        frame  = np.ones((h, w, 3), dtype=np.uint8) * 200   # bright gray
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (100, 100), (200, 200), (0, 128, 255), cv2.FILLED)
        out = composite_layers(frame, canvas)
        assert np.array_equal(out[150, 150], [0, 128, 255])
        assert not np.array_equal(out[150, 150], [200, 200, 200])

    def test_ink_on_bright_frame_not_washed_out(self, dims):
        """Dark ink on a bright frame must remain dark — no blending lightens it."""
        h, w = dims
        frame  = np.ones((h, w, 3), dtype=np.uint8) * 255   # white frame
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (100, 100), (300, 300), (30, 30, 30), cv2.FILLED)
        out = composite_layers(frame, canvas)
        assert np.array_equal(out[200, 200], [30, 30, 30])


# ── TestInputImmutability ─────────────────────────────────────────────────────

class TestInputImmutability:
    """Neither input array should be mutated by the function."""

    def test_frame_not_mutated(self, gray_frame, white_line_canvas):
        original_frame = gray_frame.copy()
        composite_layers(gray_frame, white_line_canvas)
        assert np.array_equal(gray_frame, original_frame)

    def test_canvas_not_mutated(self, gray_frame, white_line_canvas):
        original_canvas = white_line_canvas.copy()
        composite_layers(gray_frame, white_line_canvas)
        assert np.array_equal(white_line_canvas, original_canvas)

    def test_frame_not_mutated_with_colored_ink(self, gray_frame, red_rect_canvas):
        original_frame = gray_frame.copy()
        composite_layers(gray_frame, red_rect_canvas)
        assert np.array_equal(gray_frame, original_frame)

    def test_canvas_not_mutated_with_colored_ink(self, gray_frame, red_rect_canvas):
        original_canvas = red_rect_canvas.copy()
        composite_layers(gray_frame, red_rect_canvas)
        assert np.array_equal(red_rect_canvas, original_canvas)


# ── TestShapeMismatch ─────────────────────────────────────────────────────────

class TestShapeMismatch:
    """Mismatched shapes must raise ValueError with a descriptive message."""

    def test_different_height_raises(self, gray_frame):
        bad_canvas = np.zeros((100, 640, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            composite_layers(gray_frame, bad_canvas)

    def test_different_width_raises(self, gray_frame):
        bad_canvas = np.zeros((480, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            composite_layers(gray_frame, bad_canvas)

    def test_different_both_dims_raises(self, gray_frame):
        bad_canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            composite_layers(gray_frame, bad_canvas)

    def test_error_message_mentions_shapes(self, gray_frame):
        bad_canvas = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"[Ss]hape"):
            composite_layers(gray_frame, bad_canvas)

    def test_swapped_args_shape_mismatch_raises(self):
        """Passing (small, large) should also raise if dimensions differ."""
        small = np.zeros((100, 100, 3), dtype=np.uint8)
        large = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            composite_layers(small, large)


# ── TestEdgeCases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Boundary and unusual inputs that must not crash or silently misbehave."""

    def test_single_pixel_blank_canvas(self):
        """1×1 frame with blank canvas → unchanged."""
        frame  = np.array([[[100, 150, 200]]], dtype=np.uint8)
        canvas = np.array([[[0, 0, 0]]], dtype=np.uint8)
        out = composite_layers(frame, canvas)
        assert np.array_equal(out, frame)

    def test_single_pixel_ink_canvas(self):
        """1×1 canvas with ink → frame replaced by ink pixel."""
        frame  = np.array([[[100, 150, 200]]], dtype=np.uint8)
        canvas = np.array([[[255, 255, 255]]], dtype=np.uint8)
        out = composite_layers(frame, canvas)
        assert np.array_equal(out, canvas)

    def test_non_square_frame(self):
        """Wide-screen (1280×720) dimensions must work without errors."""
        frame  = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.line(canvas, (0, 0), (1279, 719), (255, 0, 0), 3)
        out = composite_layers(frame, canvas)
        assert out.shape == frame.shape

    def test_near_black_ink_threshold(self):
        """Pixel value (1,1,1) is above the threshold (>1) — NOT inked."""
        frame  = np.ones((10, 10, 3), dtype=np.uint8) * 128
        canvas = np.zeros((10, 10, 3), dtype=np.uint8)
        canvas[5, 5] = [1, 1, 1]   # threshold is > 1, so this is background
        out = composite_layers(frame, canvas)
        assert np.array_equal(out[5, 5], [128, 128, 128])

    def test_just_above_threshold_is_inked(self):
        """Pixel value (2,2,2) is above threshold — treated as ink."""
        frame  = np.ones((10, 10, 3), dtype=np.uint8) * 128
        canvas = np.zeros((10, 10, 3), dtype=np.uint8)
        canvas[5, 5] = [2, 2, 2]
        out = composite_layers(frame, canvas)
        assert np.array_equal(out[5, 5], [2, 2, 2])

    def test_identical_frame_and_canvas_colors(self):
        """When ink color == frame color, the output pixel still matches both."""
        frame  = np.ones((10, 10, 3), dtype=np.uint8) * 200
        canvas = np.zeros((10, 10, 3), dtype=np.uint8)
        canvas[5, 5] = [200, 200, 200]  # same as frame
        out = composite_layers(frame, canvas)
        assert np.array_equal(out[5, 5], [200, 200, 200])

    def test_called_twice_gives_same_result(self, gray_frame, white_line_canvas):
        """Pure function: two calls with same inputs must give identical outputs."""
        out1 = composite_layers(gray_frame, white_line_canvas)
        out2 = composite_layers(gray_frame, white_line_canvas)
        assert np.array_equal(out1, out2)