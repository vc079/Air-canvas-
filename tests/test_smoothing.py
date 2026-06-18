"""
tests/test_smoothing.py
-------------------------
Unit tests for utils.smoothing.SmoothingBuffer.

Run via terminal: pytest tests/test_smoothing.py
"""

import sys
import os
import math

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.smoothing import SmoothingBuffer


@pytest.fixture()
def buf():
    return SmoothingBuffer()


class TestBasicBehavior:
    def test_first_point_returned_unchanged(self, buf):
        assert buf.push((50, 60)) == (50, 60)

    def test_window_default_is_three(self, buf):
        assert buf.window == 3

    def test_is_empty_before_any_push(self, buf):
        assert buf.is_empty is True

    def test_is_empty_false_after_push(self, buf):
        buf.push((1, 1))
        assert buf.is_empty is False

    def test_len_tracks_buffer_size_up_to_window(self, buf):
        buf.push((1, 1))
        buf.push((2, 2))
        assert len(buf) == 2
        buf.push((3, 3))
        buf.push((4, 4))
        assert len(buf) == 3  # capped at window size

    def test_reset_clears_buffer(self, buf):
        buf.push((1, 1))
        buf.push((2, 2))
        buf.reset()
        assert buf.is_empty is True
        assert len(buf) == 0

    def test_reset_allows_unconditional_first_point_again(self, buf):
        buf.push((1, 1))
        buf.push((500, 500))  # within first window, no rejection logic yet active issue
        buf.reset()
        # After reset, even a "far" point must be accepted unconditionally
        result = buf.push((9999, 9999))
        assert result == (9999, 9999)


class TestWeightedAveraging:
    def test_recent_point_weighted_more_than_old(self, buf):
        """With window=3, pushing (0,0),(0,0),(100,0) should NOT average to 33.3 (flat) but lean toward 100."""
        buf.push((0, 0))
        buf.push((0, 0))
        result = buf.push((100, 0))
        # Weighted: (0*1 + 0*2 + 100*3) / 6 = 50.0 -- not the flat-average 33.3
        assert result[0] == 50

    def test_constant_input_returns_same_value(self, buf):
        for _ in range(5):
            result = buf.push((42, 84))
        assert result == (42, 84)

    def test_single_point_returns_exact_value(self, buf):
        assert buf.push((7, 9)) == (7, 9)


class TestOutlierRejection:
    def test_large_jump_is_rejected(self, buf):
        buf.push((100, 100))
        buf.push((105, 102))
        before = buf.push((110, 104))
        glitch = buf.push((500, 480))  # > max_jump (120px) away
        assert glitch == before

    def test_buffer_not_modified_by_rejected_point(self, buf):
        buf.push((100, 100))
        size_before = len(buf)
        buf.push((900, 900))  # glitch, should be rejected
        assert len(buf) == size_before

    def test_recovery_after_glitch_uses_real_point(self, buf):
        buf.push((100, 100))
        buf.push((110, 100))
        buf.push((900, 900))  # rejected glitch
        result = buf.push((120, 101))  # genuine next point
        # Should now include the new genuine point in the average
        assert result != (900, 900)

    def test_first_point_after_reset_never_rejected_even_if_far(self):
        sb = SmoothingBuffer(max_jump=10)
        sb.push((0, 0))
        sb.reset()
        result = sb.push((10000, 10000))
        assert result == (10000, 10000)

    def test_custom_max_jump_respected(self):
        sb = SmoothingBuffer(max_jump=5)
        sb.push((0, 0))
        rejected = sb.push((100, 0))  # jump of 100 > max_jump of 5
        assert rejected == (0, 0)


class TestLagCharacteristics:
    """
    Confirms the weighted average tracks a moving fingertip more tightly
    than a flat average would -- this is the core usability fix: ink
    should follow the real fingertip closely, not lag behind it.
    """

    def _simulate_stroke(self, sb: SmoothingBuffer):
        raw_path = []
        for i in range(40):
            t = i / 40
            x = 300 + 150 * math.sin(t * math.pi * 2)
            y = 200 + t * 200
            raw_path.append((int(x), int(y)))
        smoothed = [sb.push(p) for p in raw_path]
        lags = [math.hypot(r[0] - s[0], r[1] - s[1]) for r, s in zip(raw_path, smoothed)]
        return sum(lags) / len(lags), max(lags)

    def test_average_lag_under_15px_on_fast_stroke(self):
        sb = SmoothingBuffer()  # default window=3, weighted
        avg_lag, max_lag = self._simulate_stroke(sb)
        assert avg_lag < 15, f"Average lag too high: {avg_lag:.1f}px"

    def test_max_lag_under_25px_on_fast_stroke(self):
        sb = SmoothingBuffer()
        avg_lag, max_lag = self._simulate_stroke(sb)
        assert max_lag < 25, f"Max lag too high: {max_lag:.1f}px"


class TestValidation:
    def test_window_zero_raises(self):
        with pytest.raises(ValueError):
            SmoothingBuffer(window=0)

    def test_negative_window_raises(self):
        with pytest.raises(ValueError):
            SmoothingBuffer(window=-1)

    def test_max_jump_zero_raises(self):
        with pytest.raises(ValueError):
            SmoothingBuffer(max_jump=0)

    def test_window_one_is_valid(self):
        sb = SmoothingBuffer(window=1)
        assert sb.push((5, 5)) == (5, 5)
        assert sb.push((10, 10)) == (10, 10)  # window=1 means no averaging at all