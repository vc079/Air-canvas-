"""
utils/compositing.py
---------------------
Single-responsibility module for blending the ink canvas onto a live
webcam frame using OpenCV bitwise operations.

Algorithm
---------
Given:
    frame  — live BGR webcam image  (H × W × 3)
    canvas — ink layer BGR image    (H × W × 3, black background)

Steps:
    1. Convert canvas to greyscale → ``gray``
    2. Threshold ``gray`` → binary mask (any non-black pixel = 255)
    3. Invert the mask → ``inv_mask``
    4. Black-out the ink region in the frame using ``inv_mask``
       → ``frame_bg``  (frame with "holes" where ink will sit)
    5. Isolate ink pixels from the canvas using ``mask``
       → ``canvas_fg`` (ink pixels only, black elsewhere)
    6. Combine:  ``output = frame_bg | canvas_fg``

Result: the ink appears fully opaque on top of the video, with no alpha
blending, giving vibrant colours that hold up against any background.

The function is kept pure (no side-effects, no stored state) so it is
trivially testable and can be called from any context.
"""

from __future__ import annotations

import cv2
import numpy as np


def composite_layers(
    frame: np.ndarray,
    canvas: np.ndarray,
) -> np.ndarray:
    """
    Composite *canvas* ink on top of *frame*.

    Parameters
    ----------
    frame : np.ndarray
        Live BGR webcam frame, shape (H, W, 3).
    canvas : np.ndarray
        Ink layer BGR array, same shape as *frame*.
        Background must be pure black (0, 0, 0).

    Returns
    -------
    np.ndarray
        Composited BGR frame, same shape as inputs.

    Raises
    ------
    ValueError
        If *frame* and *canvas* do not share the same shape.
    """
    if frame.shape != canvas.shape:
        raise ValueError(
            f"Shape mismatch: frame {frame.shape} vs canvas {canvas.shape}. "
            "Ensure the canvas was initialised with the same dimensions as "
            "the webcam resolution."
        )

    # Step 1 & 2 — binary mask: white where canvas has ink
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Step 3 — invert for background selection
    inv_mask = cv2.bitwise_not(mask)

    # Step 4 — remove ink-region pixels from the frame
    frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)

    # Step 5 — isolate ink pixels
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)

    # Step 6 — merge
    return cv2.bitwise_or(frame_bg, canvas_fg)