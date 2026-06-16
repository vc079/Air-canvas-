"""
config/colors.py
----------------
Defines every colour used by Air Canvas:

* PALETTE     – ordered list of drawing colours shown in the 3-finger
                colour-picker HUD.
* UI_COLORS   – colours used by the overlay / HUD elements themselves.
* Helpers     – look up a colour by name, find the closest swatch to a
                screen coordinate, cycle through the palette.

All colour tuples are in **BGR** order (OpenCV convention).
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ======================================================================
# Data model
# ======================================================================

@dataclass(frozen=True)
class Color:
    """
    A single named colour entry.

    Attributes
    ----------
    name  : Human-readable label shown in the HUD swatch.
    bgr   : (Blue, Green, Red) tuple in 0-255 range.
    hex   : Optional HTML hex string for reference / export.
    """
    name: str
    bgr:  tuple[int, int, int]
    hex:  str = ""

    # Convenience: flip to RGB for any non-OpenCV usage
    @property
    def rgb(self) -> tuple[int, int, int]:
        b, g, r = self.bgr
        return (r, g, b)


# ======================================================================
# Drawing palette  (shown in the 3-finger colour-picker bar)
# ======================================================================
# Add, remove, or reorder entries freely.
# The first entry is also the startup default (see settings.DEFAULT_COLOR_NAME).

PALETTE: list[Color] = [
    Color("Cyan",    (255, 255,   0), "#00FFFF"),
    Color("White",   (255, 255, 255), "#FFFFFF"),
    Color("Red",     (  0,   0, 255), "#FF0000"),
    Color("Green",   (  0, 255,   0), "#00FF00"),
    Color("Blue",    (255,   0,   0), "#0000FF"),
    Color("Yellow",  (  0, 255, 255), "#FFFF00"),
    Color("Magenta", (255,   0, 255), "#FF00FF"),
    Color("Orange",  (  0, 165, 255), "#FFA500"),
    Color("Purple",  (128,   0, 128), "#800080"),
    Color("Pink",    (203, 192, 255), "#FFC8CB"),
]


# ======================================================================
# UI / HUD colours  (not part of the drawing palette)
# ======================================================================

class UI_COLORS:
    """Static namespace – import and use like: ``UI_COLORS.DRAG_HANDLE``."""

    # Drag-mode handle circle
    DRAG_HANDLE:      tuple[int, int, int] = (  0, 255,   0)   # green

    # Eraser bounding-box rectangle
    ERASER_RECT:      tuple[int, int, int] = (  0,   0, 255)   # red

    # Palette bar background fill
    PALETTE_BG:       tuple[int, int, int] = ( 30,  30,  30)   # near-black

    # Highlight ring drawn around the active swatch
    SWATCH_HIGHLIGHT: tuple[int, int, int] = (255, 255, 255)   # white

    # Snapshot cooldown progress-bar fill
    COOLDOWN_BAR:     tuple[int, int, int] = (  0, 200, 255)   # amber-ish

    # HUD mode label text
    HUD_TEXT:         tuple[int, int, int] = (255, 255, 255)   # white

    # Multi-hand warning text
    HUD_WARNING:      tuple[int, int, int] = (  0,   0, 255)   # red

    # Save-confirmation toast text
    HUD_SAVE_OK:      tuple[int, int, int] = (  0, 255, 128)   # mint green


# ======================================================================
# Helper functions
# ======================================================================

def get_color_by_name(name: str) -> Color:
    """
    Case-insensitive lookup in PALETTE.

    Parameters
    ----------
    name : str  e.g. ``"cyan"`` or ``"Cyan"``

    Returns
    -------
    Color

    Raises
    ------
    KeyError if the name is not found.
    """
    name_lower = name.strip().lower()
    for color in PALETTE:
        if color.name.lower() == name_lower:
            return color
    available = ", ".join(c.name for c in PALETTE)
    raise KeyError(
        f"Color '{name}' not found in PALETTE. "
        f"Available: {available}"
    )


def get_swatch_rects(
    frame_width: int,
    bar_height:  int = 80,
    swatch_width: int | None = None,
) -> list[tuple[Color, tuple[int, int, int, int]]]:
    """
    Compute the screen rectangle for every palette swatch.

    Returns a list of ``(Color, (x1, y1, x2, y2))`` tuples, evenly
    distributed across the top of the frame.

    Parameters
    ----------
    frame_width  : Width of the camera frame in pixels.
    bar_height   : Height of the colour-picker bar in pixels.
    swatch_width : Fixed width per swatch.  If None, swatches are
                   distributed evenly to fill the full frame width.
    """
    n = len(PALETTE)
    if swatch_width is None:
        swatch_width = frame_width // n

    rects = []
    for i, color in enumerate(PALETTE):
        x1 = i * swatch_width
        x2 = x1 + swatch_width
        rects.append((color, (x1, 0, x2, bar_height)))

    return rects


def color_at_x(x: int, frame_width: int, bar_height: int = 80) -> Color | None:
    """
    Return the palette colour whose swatch occupies horizontal position *x*.

    Returns ``None`` if *x* is outside [0, frame_width).
    Useful for hover-detection in the 3-finger colour-picker mode.
    """
    if x < 0 or x >= frame_width:
        return None
    swatch_width = frame_width // len(PALETTE)
    index = min(x // swatch_width, len(PALETTE) - 1)
    return PALETTE[index]


def next_color(current: Color) -> Color:
    """Cycle forward through the palette. Wraps at the end."""
    idx = next(
        (i for i, c in enumerate(PALETTE) if c.name == current.name), 0
    )
    return PALETTE[(idx + 1) % len(PALETTE)]


def prev_color(current: Color) -> Color:
    """Cycle backward through the palette. Wraps at the start."""
    idx = next(
        (i for i, c in enumerate(PALETTE) if c.name == current.name), 0
    )
    return PALETTE[(idx - 1) % len(PALETTE)]


# Convenience mappings and defaults used by the rest of the app.
# These names match the legacy `config.colors` API expected by imports.
COLOR_ORDER: list[str] = [color.name for color in PALETTE]
COLOR_BGR: dict[str, tuple[int, int, int]] = {
    color.name: color.bgr for color in PALETTE
}
DEFAULT_COLOR: str = PALETTE[0].name
