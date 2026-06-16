"""
config/settings.py
------------------
Central place for every tunable constant in Air Canvas.
Change values here; no other file needs to be touched.

Sections
--------
1. Camera
2. Canvas & Drawing
3. Hand Detection
4. Drag (2-finger pan)
5. Eraser (open-hand wipe)
6. Snapshot
7. HUD / Overlay
8. Multi-hand Safety
"""

# ======================================================================
# 1. CAMERA
# ======================================================================

# Index of the webcam to open (0 = default built-in camera).
CAMERA_INDEX: int = 0

# Capture resolution.  Higher = more detail but heavier CPU load.
FRAME_WIDTH:  int = 1280
FRAME_HEIGHT: int = 720

# Target frames per second (OpenCV will try to honour this).
TARGET_FPS: int = 30

# Mirror the feed horizontally so movements feel natural (like a mirror).
FLIP_FRAME: bool = True


# ======================================================================
# 2. CANVAS & DRAWING
# ======================================================================

# Default stroke thickness in pixels.
BRUSH_THICKNESS: int = 6

# Minimum thickness selectable via the HUD palette.
MIN_BRUSH_THICKNESS: int = 2

# Maximum thickness selectable via the HUD palette.
MAX_BRUSH_THICKNESS: int = 20

# cv2.line uses BGR.  The actual colours live in colors.py; this is
# the *name* of the colour to load on startup.
DEFAULT_COLOR_NAME: str = "Cyan"

# When True, strokes are drawn with cv2.line between consecutive tip
# positions for smooth continuous lines.  False = dot-per-frame.
SMOOTH_STROKE: bool = True


# ======================================================================
# 3. HAND DETECTION  (MediaPipe Hands)
# ======================================================================

# Maximum number of hands to track simultaneously.
# Air Canvas only draws with ONE hand; the second is a safety guard.
MAX_HANDS: int = 2

# Confidence thresholds (0.0 – 1.0).
DETECTION_CONFIDENCE:  float = 0.80
TRACKING_CONFIDENCE:   float = 0.70

# Which landmark index is used as the drawing cursor.
# 8 = INDEX_FINGER_TIP (standard); change to 12 for middle-finger mode.
CURSOR_LANDMARK_INDEX: int = 8

# Landmark index used as the second point for drag-handle midpoint.
# 12 = MIDDLE_FINGER_TIP.
DRAG_LANDMARK_INDEX: int = 12


# ======================================================================
# 4. DRAG (2-FINGER PAN)
# ======================================================================

# Radius of the drag handle circle drawn between the two finger tips.
DRAG_HANDLE_RADIUS: int = 10

# BGR colour of the drag handle indicator.
DRAG_HANDLE_COLOR: tuple[int, int, int] = (0, 255, 0)   # green


# ======================================================================
# 5. ERASER  (open-hand wipe / 5-finger mode)
# ======================================================================

# Padding in pixels added around the hand bounding box to form the
# eraser rectangle.  Larger = more aggressive wipe.
ERASER_PADDING: int = 20

# BGR colour of the eraser bounding-box preview rectangle.
ERASER_RECT_COLOR: tuple[int, int, int] = (0, 0, 255)   # red

# Thickness of the eraser rectangle border drawn on the HUD.
ERASER_RECT_THICKNESS: int = 2


# ======================================================================
# 6. SNAPSHOT  (4-finger mode)
# ======================================================================

# Directory (relative to the script's working directory) where PNGs
# are saved.  Created automatically if it does not exist.
SNAPSHOT_DIR: str = "snapshots"

# Minimum seconds between two consecutive saves (anti-spam cooldown).
SNAPSHOT_COOLDOWN: float = 3.0


# ======================================================================
# 7. HUD / OVERLAY
# ======================================================================

# Height in pixels of the colour-palette bar shown in 3-finger mode.
PALETTE_BAR_HEIGHT: int = 80

# Each colour swatch is this many pixels wide inside the palette bar.
PALETTE_SWATCH_WIDTH: int = 100

# Font scale for on-screen text (mode labels, warnings, counters).
HUD_FONT_SCALE: float = 0.9

# Thickness of HUD text strokes.
HUD_FONT_THICKNESS: int = 2

# BGR colour for general HUD text.
HUD_TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)   # white

# BGR colour for the multi-hand warning message.
HUD_WARNING_COLOR: tuple[int, int, int] = (0, 0, 255)    # red

# Show a small landmark overlay on the detected hand.
SHOW_HAND_LANDMARKS: bool = True


# ======================================================================
# 8. MULTI-HAND SAFETY
# ======================================================================

# When a second hand enters the frame all drawing is paused and this
# message is displayed on screen.
MULTI_HAND_WARNING: str = "WARNING: Multiple hands detected — drawing paused"


# ---------------------------------------------------------------------
# Snapshot compatibility: tests and legacy code expect frame-based constants
# ---------------------------------------------------------------------
# Convert the cooldown seconds value into a frame count using the target FPS.
SNAPSHOT_COOLDOWN_FRAMES: int = int(SNAPSHOT_COOLDOWN * TARGET_FPS)
# Default flash duration in frames (HUD shutter effect). Kept short.
SNAPSHOT_FLASH_FRAMES: int = 6
