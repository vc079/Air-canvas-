"""
Unit tests for the GestureEngine logic.
Run via terminal: pytest tests/test_gestures.py
"""

import pytest
from core.gesture_engine import GestureEngine

@pytest.fixture
def engine():
    """Fixture to initialize a fresh GestureEngine for each test."""
    return GestureEngine()

def test_idle_gesture(engine):
    """0 fingers should resolve to 'idle' mode."""
    mode = engine.resolve(finger_count=0, landmarks=None)
    assert mode == "idle"

def test_draw_gesture(engine):
    """1 finger should resolve to 'draw' mode."""
    mode = engine.resolve(finger_count=1, landmarks=None)
    assert mode == "draw"

def test_drag_gesture(engine):
    """2 fingers should resolve to 'drag' mode."""
    mode = engine.resolve(finger_count=2, landmarks=None)
    assert mode == "drag"

def test_palette_gesture(engine):
    """3 fingers should resolve to 'palette' mode."""
    mode = engine.resolve(finger_count=3, landmarks=None)
    assert mode == "palette"

def test_snapshot_gesture(engine):
    """4 fingers should resolve to 'snapshot' mode."""
    mode = engine.resolve(finger_count=4, landmarks=None)
    assert mode == "snapshot"

def test_erase_gesture(engine):
    """5 fingers should resolve to 'erase' mode."""
    mode = engine.resolve(finger_count=5, landmarks=None)
    assert mode == "erase"

def test_engine_reset_state(engine):
    """Ensure resetting the engine reverts it to a safe default state."""
    # Simulate entering a drawing state
    engine.resolve(finger_count=1, landmarks=None)
    assert engine.current_mode == "draw"
    
    # Trigger reset (e.g., multi-hand detected)
    engine.reset()
    assert engine.current_mode == "idle"