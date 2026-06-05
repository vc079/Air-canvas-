"""
Unit tests for image compositing utility functions.
Run via terminal: pytest tests/test_compositing.py
"""

import pytest
import numpy as np
import cv2
from utils.compositing import merge_canvas

@pytest.fixture
def mock_frames():
    """Provides mock video frames and canvas arrays for testing."""
    height, width, channels = 480, 640, 3
    
    # Create a mock webcam frame (gray background)
    bg_frame = np.ones((height, width, channels), dtype=np.uint8) * 128
    
    # Create a mock canvas (pitch black background with a white drawing)
    canvas = np.zeros((height, width, channels), dtype=np.uint8)
    # Draw a mock white line on the canvas
    cv2.line(canvas, (10, 10), (100, 100), (255, 255, 255), 5)
    
    return bg_frame, canvas

def test_merge_canvas_output_shape(mock_frames):
    """Ensure the composited output matches the original frame dimensions."""
    bg_frame, canvas = mock_frames
    
    output = merge_canvas(bg_frame, canvas)
    
    assert output.shape == bg_frame.shape
    assert output.dtype == np.uint8

def test_merge_canvas_ink_transfer(mock_frames):
    """Ensure the drawn ink on the canvas successfully transfers to the background."""
    bg_frame, canvas = mock_frames
    
    output = merge_canvas(bg_frame, canvas)
    
    # Check a pixel where the line was drawn (should not be the background color)
    # (50, 50) is on the line from (10,10) to (100,100)
    line_pixel = output[50, 50]
    
    # The pixel should no longer be the gray background (128, 128, 128)
    assert not np.array_equal(line_pixel, [128, 128, 128])
    # The pixel should match the white ink (255, 255, 255)
    assert np.array_equal(line_pixel, [255, 255, 255])