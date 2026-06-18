"""
tests/test_undo_redo.py
-------------------------
Unit tests for the undo/redo history stack on CanvasRenderer.

Run via terminal: pytest tests/test_undo_redo.py
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.canvas_renderer import CanvasRenderer


@pytest.fixture()
def renderer():
    return CanvasRenderer(640, 480)


def _do_stroke(r, points):
    """Simulate one complete draw gesture across several frames."""
    r.push_history()
    for p in points:
        r.draw(p)
    r.close_action()


class TestBasicUndo:
    def test_undo_on_empty_stack_returns_false(self, renderer):
        assert renderer.undo() is False

    def test_undo_reverts_a_single_stroke(self, renderer):
        blank = renderer.canvas.copy()
        _do_stroke(renderer, [(10, 10), (50, 50)])
        assert np.any(renderer.canvas != blank)
        assert renderer.undo() is True
        assert np.array_equal(renderer.canvas, blank)

    def test_can_undo_reflects_stack_state(self, renderer):
        assert renderer.can_undo() is False
        _do_stroke(renderer, [(1, 1), (2, 2)])
        assert renderer.can_undo() is True


class TestBasicRedo:
    def test_redo_on_empty_stack_returns_false(self, renderer):
        assert renderer.redo() is False

    def test_redo_restores_undone_stroke(self, renderer):
        _do_stroke(renderer, [(10, 10), (60, 60)])
        after_stroke = renderer.canvas.copy()
        renderer.undo()
        assert renderer.redo() is True
        assert np.array_equal(renderer.canvas, after_stroke)

    def test_can_redo_false_before_any_undo(self, renderer):
        _do_stroke(renderer, [(1, 1), (2, 2)])
        assert renderer.can_redo() is False

    def test_can_redo_true_after_undo(self, renderer):
        _do_stroke(renderer, [(1, 1), (2, 2)])
        renderer.undo()
        assert renderer.can_redo() is True


class TestMultiStepHistory:
    def test_two_strokes_undo_twice_reaches_blank(self, renderer):
        blank = renderer.canvas.copy()
        _do_stroke(renderer, [(10, 10), (20, 20)])
        _do_stroke(renderer, [(100, 100), (120, 120)])
        renderer.undo()
        renderer.undo()
        assert np.array_equal(renderer.canvas, blank)

    def test_redo_chain_restores_each_step_in_order(self, renderer):
        _do_stroke(renderer, [(10, 10), (20, 20)])
        state1 = renderer.canvas.copy()
        _do_stroke(renderer, [(100, 100), (120, 120)])
        state2 = renderer.canvas.copy()

        renderer.undo()
        renderer.undo()
        renderer.redo()
        assert np.array_equal(renderer.canvas, state1)
        renderer.redo()
        assert np.array_equal(renderer.canvas, state2)

    def test_new_action_after_undo_clears_redo_stack(self, renderer):
        _do_stroke(renderer, [(10, 10), (20, 20)])
        _do_stroke(renderer, [(100, 100), (120, 120)])
        renderer.undo()
        assert renderer.can_redo() is True
        _do_stroke(renderer, [(200, 200), (220, 220)])
        assert renderer.can_redo() is False


class TestActionBoundaries:
    def test_continuous_action_pushes_single_history_entry(self, renderer):
        for _ in range(15):
            renderer.push_history()
            renderer.draw((5, 5))
        renderer.close_action()
        assert len(renderer._undo_stack) == 1

    def test_close_action_allows_new_push(self, renderer):
        renderer.push_history()
        renderer.draw((1, 1))
        renderer.close_action()
        renderer.push_history()
        renderer.draw((2, 2))
        renderer.close_action()
        assert len(renderer._undo_stack) == 2

    def test_push_without_close_does_not_duplicate(self, renderer):
        renderer.push_history()
        renderer.push_history()
        renderer.push_history()
        assert len(renderer._undo_stack) == 1


class TestHistoryCap:
    def test_undo_stack_capped_at_max_history(self, renderer):
        for i in range(30):
            renderer.push_history()
            renderer.draw((i, i))
            renderer.close_action()
        assert len(renderer._undo_stack) <= 20


class TestUndoRedoSideEffects:
    def test_undo_resets_in_progress_stroke(self, renderer):
        _do_stroke(renderer, [(10, 10), (20, 20)])
        renderer.push_history()
        renderer.draw((30, 30))  # mid-stroke, prev_point set
        renderer.undo()
        assert renderer._prev_point is None

    def test_undo_clears_drag_anchor(self, renderer):
        renderer.push_history()
        renderer.drag((50, 50))
        renderer.drag((60, 60))
        renderer.undo()
        assert renderer._last_drag_anchor is None